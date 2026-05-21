"""Minimal dataset container (tinygrad style)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..types import Dialogue, Label, Message

# Type alias for loader functions
LoaderFn = Callable[..., "Dataset"]


@dataclass(slots=True)
class Dataset:
    """Minimal dataset: dialogues + labels + optional metadata."""

    dialogues: list[Dialogue]
    labels: list[Label]
    name: str = "dataset"
    metadata: dict[str, list[Any]] | None = None

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        *,
        name: str = "dataset",
        messages_key: str = "messages",
        label_key: str = "label",
        metadata_key: str | None = "metadata",
        include_extra_metadata: bool = True,
    ) -> "Dataset":
        """Build a dataset from row-wise records.

        Each record must contain a dialogue under ``messages_key`` and a binary
        label under ``label_key``. Dialogues may be lists of ``Message`` objects
        or ``{"role": ..., "content": ...}`` dictionaries. Metadata can be
        supplied as a nested dictionary under ``metadata_key``; any extra
        top-level fields are also copied into metadata when
        ``include_extra_metadata`` is true.
        """
        dialogues: list[Dialogue] = []
        labels: list[Label] = []
        row_metadata: list[dict[str, Any]] = []

        for index, record in enumerate(records):
            if messages_key not in record:
                raise KeyError(f"record {index} is missing {messages_key!r}")
            if label_key not in record:
                raise KeyError(f"record {index} is missing {label_key!r}")

            dialogues.append(_coerce_dialogue(record[messages_key]))
            labels.append(_coerce_label(record[label_key]))

            meta: dict[str, Any] = {}
            if metadata_key is not None and metadata_key in record:
                nested = record[metadata_key]
                if nested is not None:
                    if not isinstance(nested, dict):
                        raise TypeError(
                            f"record {index} {metadata_key!r} must be a mapping"
                        )
                    meta.update(nested)
            if include_extra_metadata:
                reserved = {messages_key, label_key}
                if metadata_key is not None:
                    reserved.add(metadata_key)
                meta.update({k: v for k, v in record.items() if k not in reserved})
            row_metadata.append(meta)

        return cls(
            dialogues=dialogues,
            labels=labels,
            name=name,
            metadata=_metadata_columns(row_metadata),
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
        messages_key: str = "messages",
        label_key: str = "label",
        metadata_key: str | None = "metadata",
        include_extra_metadata: bool = True,
    ) -> "Dataset":
        """Load records from JSONL and return a dataset."""
        path = Path(path)
        records = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        return cls.from_records(
            records,
            name=name or path.stem,
            messages_key=messages_key,
            label_key=label_key,
            metadata_key=metadata_key,
            include_extra_metadata=include_extra_metadata,
        )

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int | slice | list) -> "Dataset":
        idx = [idx] if isinstance(idx, int) else list(range(len(self)))[idx] if isinstance(idx, slice) else idx
        meta = {k: [v[i] for i in idx] for k, v in self.metadata.items()} if self.metadata else None
        return Dataset([self.dialogues[i] for i in idx], [self.labels[i] for i in idx], self.name, meta)

    def __add__(self, other: "Dataset") -> "Dataset":
        def _src(ds): return (ds.metadata or {}).get("source") or [ds.name] * len(ds)
        def _get(ds, k): return (ds.metadata or {}).get(k) or [None] * len(ds)
        keys = (set(self.metadata or {}) | set(other.metadata or {})) | {"source"}
        meta = {k: (_src(self) + _src(other) if k == "source" else _get(self, k) + _get(other, k)) for k in keys}
        return Dataset(self.dialogues + other.dialogues, self.labels + other.labels, f"{self.name}+{other.name}", meta)

    def to_records(
        self,
        *,
        messages_key: str = "messages",
        label_key: str = "label",
        metadata_key: str | None = "metadata",
        include_name: bool = False,
    ) -> list[dict[str, Any]]:
        """Return row-wise records with messages, label, and metadata.

        When ``metadata_key`` is a string, metadata columns are nested under
        that key. When it is ``None``, metadata columns are flattened into the
        top-level record.
        """
        records: list[dict[str, Any]] = []
        for index, (dialogue, label) in enumerate(zip(self.dialogues, self.labels, strict=True)):
            record: dict[str, Any] = {
                messages_key: [_message_to_record(message) for message in dialogue],
                label_key: int(label),
            }
            if include_name:
                record["dataset"] = self.name

            meta = {
                key: values[index]
                for key, values in (self.metadata or {}).items()
            }
            if metadata_key is None:
                record.update(meta)
            elif meta:
                record[metadata_key] = meta
            records.append(record)
        return records

    def to_jsonl(
        self,
        path: str | Path,
        *,
        messages_key: str = "messages",
        label_key: str = "label",
        metadata_key: str | None = "metadata",
        include_name: bool = False,
    ) -> None:
        """Write row-wise records to JSONL."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for record in self.to_records(
                messages_key=messages_key,
                label_key=label_key,
                metadata_key=metadata_key,
                include_name=include_name,
            ):
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def shuffle(self, seed: int = 42) -> "Dataset":
        """Return shuffled copy."""
        perm = np.random.default_rng(seed).permutation(len(self)).tolist()
        return self[perm]

    def sample(self, n: int, stratified: bool = False, seed: int = 42) -> "Dataset":
        """Return random sample of n items.

        Args:
            n: Number of samples (capped at dataset size)
            stratified: If True, preserve label proportions
            seed: Random seed for reproducibility
        """
        if n >= len(self):
            return self

        rng = np.random.default_rng(seed)

        if not stratified:
            indices = rng.choice(len(self), size=n, replace=False).tolist()
        else:
            # Stratified: sample proportionally from each class
            pos_idx = [i for i, l in enumerate(self.labels) if l == Label.POSITIVE]
            neg_idx = [i for i, l in enumerate(self.labels) if l == Label.NEGATIVE]

            # Calculate proportional sample sizes
            pos_ratio = len(pos_idx) / len(self) if len(self) > 0 else 0.5
            n_pos = min(int(round(n * pos_ratio)), len(pos_idx))
            n_neg = min(n - n_pos, len(neg_idx))

            # Adjust if we can't get enough from one class
            if n_pos + n_neg < n:
                if n_neg < n - n_pos:
                    n_pos = min(n - n_neg, len(pos_idx))
                else:
                    n_neg = min(n - n_pos, len(neg_idx))

            indices = (rng.choice(pos_idx, size=n_pos, replace=False).tolist() +
                      rng.choice(neg_idx, size=n_neg, replace=False).tolist())
            rng.shuffle(indices)

        return self[indices]

    def split(self, frac: float = 0.8, stratified: bool = False, seed: int = 42) -> tuple["Dataset", "Dataset"]:
        """Split into train/test sets.

        Args:
            frac: Fraction for first split (train)
            stratified: If True, preserve label proportions in both splits
            seed: Random seed for reproducibility
        """
        if not stratified:
            shuffled = self.shuffle(seed)
            n = int(len(self) * frac)
            return shuffled[:n], shuffled[n:]

        # Stratified split
        rng = np.random.default_rng(seed)
        pos_idx = [i for i, l in enumerate(self.labels) if l == Label.POSITIVE]
        neg_idx = [i for i, l in enumerate(self.labels) if l == Label.NEGATIVE]

        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        n_pos_train = int(len(pos_idx) * frac)
        n_neg_train = int(len(neg_idx) * frac)

        train_idx = pos_idx[:n_pos_train] + neg_idx[:n_neg_train]
        test_idx = pos_idx[n_pos_train:] + neg_idx[n_neg_train:]

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        return self[train_idx], self[test_idx]

    def where(self, cond: list[bool]) -> "Dataset":
        """Filter by boolean condition."""
        return self[[i for i, c in enumerate(cond) if c]]

    @property
    def positive(self) -> "Dataset":
        return self.where([l == Label.POSITIVE for l in self.labels])

    @property
    def negative(self) -> "Dataset":
        return self.where([l == Label.NEGATIVE for l in self.labels])

    def __repr__(self) -> str:
        pos = sum(1 for l in self.labels if l == Label.POSITIVE)
        neg = len(self) - pos
        return f"Dataset({self.name!r}, n={len(self)}, pos={pos}, neg={neg})"


def _coerce_dialogue(value: Any) -> Dialogue:
    dialogue: Dialogue = []
    for message in value:
        if isinstance(message, Message):
            dialogue.append(message)
            continue
        if not isinstance(message, dict):
            raise TypeError("messages must contain Message objects or dictionaries")
        dialogue.append(
            Message(
                role=message["role"],
                content=message.get("content", message.get("value", "")),
            )
        )
    return dialogue


def _coerce_label(value: Any) -> Label:
    if isinstance(value, Label):
        return value
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"positive", "pos", "true", "1"}:
            return Label.POSITIVE
        if normalized in {"negative", "neg", "false", "0"}:
            return Label.NEGATIVE
    return Label.POSITIVE if int(value) == 1 else Label.NEGATIVE


def _metadata_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]] | None:
    keys = sorted({key for row in rows for key in row})
    if not keys:
        return None
    return {key: [row.get(key) for row in rows] for key in keys}


def _message_to_record(message: Message) -> dict[str, str]:
    role = message.role.value if hasattr(message.role, "value") else str(message.role)
    return {"role": role, "content": message.content}

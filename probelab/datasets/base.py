"""Minimal dataset container (tinygrad style)."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..types import Dialogue, Label

# Type alias for loader functions
LoaderFn = Callable[..., "Dataset"]


@dataclass(slots=True)
class Dataset:
    """Minimal dataset: dialogues + labels + optional metadata."""

    dialogues: list[Dialogue]
    labels: list[Label]
    name: str = "dataset"
    metadata: dict[str, list[Any]] | None = None

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

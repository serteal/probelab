"""Protocol-locked synthetic datasets for the foundational probing paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..types import Label, Message
from .base import Dataset
from .registry import Topic, _register_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT_V1 = (
    PROJECT_ROOT
    / "dataset-generation"
    / "frozen_probe_sets"
    / "outputs"
    / "frozen_probe_v1"
)
DEFAULT_ROOT_V2 = (
    PROJECT_ROOT
    / "dataset-generation"
    / "frozen_probe_sets"
    / "outputs"
    / "frozen_probe_v2"
)


def _row_label(row: dict[str, Any]) -> Label:
    value = row.get("label_value", row.get("label"))
    if isinstance(value, str):
        value = value.lower()
        if value in {"positive", "pos", "1", "true"}:
            return Label.POSITIVE
        if value in {"negative", "neg", "0", "false"}:
            return Label.NEGATIVE
    if int(value) == 1:
        return Label.POSITIVE
    return Label.NEGATIVE


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_frozen_protocol(
    *,
    version: str,
    domain: str,
    split: str,
    default_root: Path,
    root: str | None = None,
    path: str | None = None,
) -> Dataset:
    data_path = Path(path) if path else Path(root or default_root) / f"{domain}_{split}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{version} split not found: {data_path}. "
            "Run the matching dataset-generation/frozen_probe_sets generator first."
        )

    rows = _read_jsonl(data_path)
    dialogues = []
    labels = []
    metadata: dict[str, list[Any]] = {
        "id": [],
        "domain": [],
        "split": [],
        "site": [],
        "taxonomy_category": [],
        "topic": [],
        "generator_model": [],
        "generation_method": [],
        "content_safety_mode": [],
        "source_protocol_version": [],
        "prohibited_ood_sources": [],
    }
    for row in rows:
        messages = row.get("messages") or []
        if not messages:
            continue
        dialogues.append([
            Message(str(msg["role"]), str(msg["content"]))
            for msg in messages
        ])
        labels.append(_row_label(row))
        for key in metadata:
            metadata[key].append(row.get(key))
    return Dataset(dialogues, labels, f"{version}/{domain}/{split}", metadata).shuffle()


@_register_dataset(
    "frozen_protocol_v1",
    Topic.OOD,
    "Protocol-locked synthetic train/calibration/IID rows",
)
def frozen_protocol_v1(
    domain: str,
    split: str,
    root: str | None = None,
    path: str | None = None,
) -> Dataset:
    """Load a synthetic split generated without OOD benchmark access.

    Args:
        domain: One of ``cbrn``, ``hallucination``, ``cyber_assistant``,
            ``deception``.
        split: ``train``, ``calibration``, or ``iid``.
        root: Optional output root containing ``{domain}_{split}.jsonl``.
        path: Optional exact JSONL path, overriding ``root``.
    """
    return _load_frozen_protocol(
        version="frozen_protocol_v1",
        domain=domain,
        split=split,
        default_root=DEFAULT_ROOT_V1,
        root=root,
        path=path,
    )


@_register_dataset(
    "frozen_protocol_v2",
    Topic.OOD,
    "Protocol-locked v2 synthetic rows with format-diverse hard negatives",
)
def frozen_protocol_v2(
    domain: str,
    split: str,
    root: str | None = None,
    path: str | None = None,
) -> Dataset:
    """Load a v2 synthetic split generated without final OOD benchmark access."""
    return _load_frozen_protocol(
        version="frozen_protocol_v2",
        domain=domain,
        split=split,
        default_root=DEFAULT_ROOT_V2,
        root=root,
        path=path,
    )

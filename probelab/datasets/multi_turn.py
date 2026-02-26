"""Multi-turn and long-context dataset loaders.

Loads generated multi-turn/long-context conversations from HuggingFace Hub,
falling back to local files from the dataset-generation pipeline.
"""

import json
import logging
from pathlib import Path

from ..types import Label, Message
from .base import Dataset
from .registry import Topic, _register_dataset

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MULTI_TURN_DIR = REPO_ROOT.parent / "dataset-generation" / "multi_turn" / "outputs"

# HuggingFace repo mapping
_HF_REPOS = {
    "cyberrisk": "serteal/cyberrisk-multi-turn",
    "virology": "serteal/virology-multi-turn",
    "harmfulness": "serteal/harmfulness-multi-turn",
    "negatives": "serteal/multi-turn-negatives",
}


def _load_multi_turn_json(path: Path) -> Dataset:
    """Generic loader that reads any multi-turn output file and returns a Dataset.

    Expected format: JSON array of objects with:
    - messages: [{"role": "user"|"assistant", "content": "..."}]
    - label: "positive" or "negative"
    - method: str
    - source: str
    - metadata: dict (optional)
    """
    with open(path) as f:
        data = json.load(f)

    return _rows_to_dataset(data, path.stem)


def _rows_to_dataset(data: list[dict], name: str) -> Dataset:
    """Convert a list of conversation dicts to a Dataset."""
    dialogues = []
    labels = []
    metadata_cols: dict[str, list] = {
        "method": [],
        "source": [],
        "base_ids": [],
        "target_turn_idx": [],
    }

    for conv in data:
        messages = [
            Message(role=msg["role"], content=msg["content"])
            for msg in conv["messages"]
        ]
        dialogues.append(messages)

        label = Label.POSITIVE if conv.get("label") == "positive" else Label.NEGATIVE
        labels.append(label)

        metadata_cols["method"].append(conv.get("method", ""))
        metadata_cols["source"].append(conv.get("source", ""))
        metadata_cols["base_ids"].append(conv.get("base_ids", []))
        metadata_cols["target_turn_idx"].append(conv.get("target_turn_idx", -1))

        # Add any extra metadata fields
        for k, v in conv.get("metadata", {}).items():
            if k not in metadata_cols:
                metadata_cols[k] = [None] * (len(dialogues) - 1)
            metadata_cols[k].append(v)

    # Pad metadata columns that were added late
    for k in metadata_cols:
        while len(metadata_cols[k]) < len(dialogues):
            metadata_cols[k].append(None)

    return Dataset(dialogues, labels, name, metadata_cols)


def _hf_dataset_to_rows(hf_ds) -> list[dict]:
    """Convert an HF Dataset (flat Parquet) back to conversation dicts."""
    rows = []
    for row in hf_ds:
        rows.append({
            "id": row.get("id", 0),
            "base_ids": json.loads(row["base_ids"]) if isinstance(row.get("base_ids"), str) else row.get("base_ids", []),
            "method": row.get("method", ""),
            "source": row.get("source", ""),
            "label": row.get("label", ""),
            "target_turn_idx": row.get("target_turn_idx", -1),
            "messages": json.loads(row["messages"]) if isinstance(row.get("messages"), str) else row.get("messages", []),
            "metadata": json.loads(row["metadata"]) if isinstance(row.get("metadata"), str) else row.get("metadata", {}),
        })
    return rows


def _load_from_hf(repo_id: str, config: str, name: str) -> Dataset:
    """Load a multi-turn dataset from HuggingFace Hub."""
    from datasets import load_dataset

    hf_ds = load_dataset(repo_id, config, split="train")
    rows = _hf_dataset_to_rows(hf_ds)
    return _rows_to_dataset(rows, name)


def _make_loader(source: str, method: str, tier: str | None = None):
    """Create a loader function for a specific source/method/tier combination."""
    if tier:
        config = f"long_context_{tier}"
        fname = f"{config}.json"
    else:
        config = method
        fname = f"{method}.json"
    path = MULTI_TURN_DIR / source / fname
    repo_id = _HF_REPOS.get(source, "")

    def loader() -> Dataset:
        # Try HuggingFace first
        if repo_id:
            try:
                return _load_from_hf(repo_id, config, config)
            except Exception as e:
                logger.debug("HF load failed for %s/%s, trying local: %s", repo_id, config, e)

        # Fall back to local file
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found on HuggingFace ({repo_id}/{config}) or locally ({path}).\n"
                f"Run: uv run python dataset-generation/multi_turn/generate.py "
                f"--source {source} --method {method}"
                + (f" --tier {tier}" if tier else "")
            )
        return _load_multi_turn_json(path)

    return loader


# Register all source/method combinations
_SOURCES = ["cyberrisk", "virology", "harmfulness"]
_METHODS = {
    "cyberrisk": ["prefix_stitch", "middle_stitch", "topic_chain", "kill_chain",
                   "backward_context", "escalation"],
    "virology": ["prefix_stitch", "middle_stitch", "topic_chain",
                  "backward_context", "escalation"],
    "harmfulness": ["prefix_stitch", "middle_stitch",
                     "backward_context", "escalation"],
}
_TIERS = ["4k", "8k", "16k", "32k"]

# Topic mapping
_TOPIC_MAP = {
    "cyberrisk": Topic.CYBERSECURITY,
    "virology": Topic.SCIENCE,
    "harmfulness": Topic.HARMFULNESS,
}

# Register multi-turn methods
for _src in _SOURCES:
    for _method in _METHODS.get(_src, []):
        _name = f"{_src}_{_method}"
        _register_dataset(
            _name,
            _TOPIC_MAP[_src],
            f"{_src} {_method.replace('_', ' ')} multi-turn dataset",
        )(_make_loader(_src, _method))

    # Register long-context tiers
    for _tier in _TIERS:
        _name = f"{_src}_long_context_{_tier}"
        _register_dataset(
            _name,
            _TOPIC_MAP[_src],
            f"{_src} long-context {_tier} dataset",
        )(_make_loader(_src, "long_context", _tier))

# Register negative datasets
for _neg_method, _neg_config in [
    ("wildchat_multi_turn", "wildchat_multi_turn"),
] + [
    (f"long_context_{t}", f"long_context_{t}")
    for t in _TIERS
]:
    _neg_name = f"{_neg_method}_negative"
    _neg_path = MULTI_TURN_DIR / "negatives" / f"{_neg_config}.json"
    _neg_repo = _HF_REPOS["negatives"]

    def _make_neg_loader(p=_neg_path, repo=_neg_repo, cfg=_neg_config):
        def loader() -> Dataset:
            # Try HuggingFace first
            if repo:
                try:
                    return _load_from_hf(repo, cfg, cfg)
                except Exception as e:
                    logger.debug("HF load failed for %s/%s, trying local: %s", repo, cfg, e)

            if not p.exists():
                raise FileNotFoundError(
                    f"Negative dataset not found on HuggingFace ({repo}/{cfg}) or locally ({p})"
                )
            return _load_multi_turn_json(p)
        return loader

    _register_dataset(
        _neg_name,
        Topic.HARMFULNESS,
        f"Negative {_neg_name.replace('_', ' ')} dataset",
    )(_make_neg_loader())

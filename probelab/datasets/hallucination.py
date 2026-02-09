"""Hallucination detection datasets.

Imports span-annotated datasets from the hallucination_probes project.
Each sample contains character-level span annotations indicating which parts
of the completion are hallucinated vs supported.
"""

from typing import Any

from datasets import load_dataset

from ..types import Dialogue, Label, Message
from .base import Dataset
from .registry import Topic, _register_dataset

# Label mapping from text labels to scalar values
_LABEL_MAP = {
    "Not Supported": 1.0,
    "NS": 1.0,
    "Insufficient Information": 1.0,
    "Supported": 0.0,
    "S": 0.0,
    "N/A": -100.0,
    None: -100.0,
}


def _has_hallucination(spans: list[dict]) -> bool:
    """Check if any span is hallucinated (label 1.0)."""
    return any(s["label"] == 1.0 for s in spans)


def _convert_longform(hf_dataset, name: str) -> Dataset:
    """Convert longform annotation datasets (longfact, longfact_augmented, healthbench).

    These datasets have:
    - conversation[-2]['content'] = prompt
    - conversation[-1]['content'] = completion
    - annotations: list of {span, label, index}
    """
    dialogues: list[Dialogue] = []
    labels: list[Label] = []
    metadata: dict[str, list[Any]] = {"spans": [], "subset": []}

    for item in hf_dataset:
        prompt = item["conversation"][-2]["content"]
        completion = item["conversation"][-1]["content"]
        annotations = item["annotations"]

        spans: list[dict] = []
        for ann in annotations:
            if ann is None or "index" not in ann or not isinstance(ann["index"], int):
                continue
            text = ann["span"]
            label_str = ann["label"]
            idx = ann["index"]
            if idx is None or not text or text not in completion:
                continue
            spans.append({
                "text": text,
                "label": _LABEL_MAP.get(label_str, -100.0),
                "index": idx,
            })

        dialogue: Dialogue = [
            Message("user", prompt),
            Message("assistant", completion),
        ]
        label = Label.POSITIVE if _has_hallucination(spans) else Label.NEGATIVE

        dialogues.append(dialogue)
        labels.append(label)
        metadata["spans"].append(spans)
        metadata["subset"].append(item.get("model", name))

    return Dataset(dialogues, labels, name, metadata)


def _convert_triviaqa(hf_dataset, name: str) -> Dataset:
    """Convert TriviaQA hallucination dataset.

    These datasets have:
    - question = prompt
    - gt_completion = completion
    - exact_answer = annotated span text
    - llm_judge_label in {S, NS, N/A}
    """
    valid_labels = {"S", "NS", "N/A"}
    label_field = "llm_judge_label" if "llm_judge_label" in hf_dataset.features else "label"

    dialogues: list[Dialogue] = []
    labels: list[Label] = []
    metadata: dict[str, list[Any]] = {"spans": [], "subset": []}

    for item in hf_dataset:
        label_str = item[label_field]
        if label_str not in valid_labels:
            continue

        prompt = item["question"] if "question" in item else item["conversation"][0]["content"]
        completion = item["gt_completion"]
        exact_answer = item.get("exact_answer", "")

        if exact_answer is None or exact_answer not in completion:
            continue

        label_value = _LABEL_MAP[label_str]
        idx = completion.find(exact_answer)

        spans = [{
            "text": exact_answer,
            "label": label_value,
            "index": idx,
        }]

        dialogue: Dialogue = [
            Message("user", prompt),
            Message("assistant", completion),
        ]
        label = Label.POSITIVE if _has_hallucination(spans) else Label.NEGATIVE

        dialogues.append(dialogue)
        labels.append(label)
        metadata["spans"].append(spans)
        metadata["subset"].append(name)

    return Dataset(dialogues, labels, name, metadata)


# ---------------------------------------------------------------------------
# LongFact
# ---------------------------------------------------------------------------

@_register_dataset("longfact", Topic.HALLUCINATION, "LongFact hallucination annotations")
def longfact(variant: str | None = None, split: str = "train") -> Dataset:
    """Load LongFact hallucination annotations.

    Args:
        variant: Model subset (e.g. 'Meta-Llama-3.1-8B-Instruct'). If None, loads all.
        split: HF split name ('train', 'validation', 'test').
    """
    if variant:
        ds = load_dataset("obalcells/longfact-annotations", variant, split=split)
    else:
        ds = load_dataset("obalcells/longfact-annotations", split=split)
    return _convert_longform(ds, "longfact").shuffle()


# ---------------------------------------------------------------------------
# LongFact Augmented
# ---------------------------------------------------------------------------

@_register_dataset("longfact_augmented", Topic.HALLUCINATION, "LongFact augmented hallucination annotations")
def longfact_augmented(variant: str | None = None, split: str = "train") -> Dataset:
    """Load LongFact augmented hallucination annotations.

    Args:
        variant: Model subset (e.g. 'Meta-Llama-3.1-8B-Instruct'). If None, loads all.
        split: HF split name ('train', 'validation', 'test').
    """
    if variant:
        ds = load_dataset("obalcells/longfact-augmented-annotations", variant, split=split)
    else:
        ds = load_dataset("obalcells/longfact-augmented-annotations", split=split)
    return _convert_longform(ds, "longfact_augmented").shuffle()


# ---------------------------------------------------------------------------
# TriviaQA Hallucination
# ---------------------------------------------------------------------------

@_register_dataset("triviaqa_hallucination", Topic.HALLUCINATION, "TriviaQA hallucination detection")
def triviaqa_hallucination(variant: str | None = None, split: str = "train") -> Dataset:
    """Load TriviaQA hallucination dataset.

    Args:
        variant: Model subset (e.g. 'Meta-Llama-3.1-8B-Instruct'). If None, loads all.
        split: HF split name ('train', 'test').
    """
    if variant:
        ds = load_dataset("obalcells/triviaqa-balanced", variant, split=split)
    else:
        ds = load_dataset("obalcells/triviaqa-balanced", split=split)
    return _convert_triviaqa(ds, "triviaqa_hallucination").shuffle()


# ---------------------------------------------------------------------------
# HealthBench
# ---------------------------------------------------------------------------

@_register_dataset("healthbench", Topic.HALLUCINATION, "HealthBench hallucination annotations")
def healthbench(variant: str | None = None, split: str = "test") -> Dataset:
    """Load HealthBench hallucination annotations.

    Args:
        variant: Model subset (e.g. 'Meta-Llama-3.1-8B-Instruct'). If None, loads all.
        split: HF split name ('test' only — HealthBench is eval-only).
    """
    if variant:
        ds = load_dataset("obalcells/healthbench-annotations", variant, split=split)
    else:
        ds = load_dataset("obalcells/healthbench-annotations", split=split)
    return _convert_longform(ds, "healthbench").shuffle()

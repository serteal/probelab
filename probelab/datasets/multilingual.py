"""Multilingual conversation datasets."""

from typing import Any

from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset
from .builders import build_from_messages


def wildchat() -> Dataset:
    """WildChat-1M - 1M+ conversations in 68 languages."""
    data = load_dataset("allenai/WildChat-1M")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"language": [], "model": [], "country": [], "toxic": [], "turn": []}

    for item in data:
        dialogue = build_from_messages(item.get("conversation", []))
        if not dialogue:
            continue

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)
        metadata["language"].append(item.get("language"))
        metadata["model"].append(item.get("model"))
        metadata["country"].append(item.get("country"))
        metadata["toxic"].append(item.get("toxic"))
        metadata["turn"].append(item.get("turn"))

    return Dataset(dialogues, labels, "wildchat", metadata).shuffle()


def multilingual_thinking() -> Dataset:
    """Multilingual Thinking - reasoning traces in 5 languages."""
    data = load_dataset("HuggingFaceH4/Multilingual-Thinking")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"language": []}

    for item in data:
        dialogue = build_from_messages(item.get("messages", []))
        if not dialogue:
            continue

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)
        metadata["language"].append(item.get("language"))

    return Dataset(dialogues, labels, "multilingual_thinking", metadata).shuffle()


def palo_multilingual() -> Dataset:
    """PALO Multilingual - 10 languages vision-language conversations."""
    data = load_dataset("MBZUAI/palo_multilingual_dataset")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"language": []}

    for item in data:
        dialogue = build_from_messages(item.get("conversations", []), role_field="from", content_field="value")
        if not dialogue:
            continue

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)
        metadata["language"].append(item.get("language"))

    return Dataset(dialogues, labels, "palo_multilingual", metadata).shuffle()

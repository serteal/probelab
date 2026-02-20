"""Toxicity detection datasets."""

from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset
from .registry import Topic, _register_dataset


@_register_dataset("toxic_conversations", Topic.HARMFULNESS, "Jigsaw toxic conversations")
def toxic_conversations(split: str = "train") -> Dataset:
    """Load Jigsaw Toxic Conversations dataset.

    Args:
        split: HF split name ('train' or 'test').
    """
    data = load_dataset("SetFit/toxic_conversations_50k")[split]
    dialogues = [[Message("user", str(row["text"]))] for row in data]
    labels = [Label.POSITIVE if row["label"] == 1 else Label.NEGATIVE for row in data]
    return Dataset(dialogues, labels, "toxic_conversations").shuffle()

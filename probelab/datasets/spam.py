"""Spam detection datasets."""

from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset
from .registry import Topic, _register_dataset


@_register_dataset("enron_spam", Topic.HARMFULNESS, "Enron email spam classification")
def enron_spam(split: str = "train") -> Dataset:
    """Load EnronSpam dataset.

    Args:
        split: HF split name ('train' or 'test').
    """
    data = load_dataset("SetFit/enron_spam")[split]
    dialogues = [[Message("user", str(row["text"]))] for row in data]
    labels = [Label.POSITIVE if row["label"] == 1 else Label.NEGATIVE for row in data]
    return Dataset(dialogues, labels, "enron_spam").shuffle()


@_register_dataset("sms_spam", Topic.HARMFULNESS, "SMS spam detection")
def sms_spam() -> Dataset:
    """Load UCI SMS Spam Collection dataset."""
    data = load_dataset("ucirvine/sms_spam", "plain_text")["train"]
    dialogues = [[Message("user", str(row["sms"]))] for row in data]
    labels = [Label.POSITIVE if row["label"] == 1 else Label.NEGATIVE for row in data]
    return Dataset(dialogues, labels, "sms_spam").shuffle()

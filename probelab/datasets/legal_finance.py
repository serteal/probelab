"""Legal and finance domain datasets."""

from typing import Any

from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def caselaw(config: str = "US") -> Dataset:
    """HFforLegal Case Law - legal decisions from various countries."""
    data = load_dataset("HFforLegal/case-law", config)["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"court": []}

    for item in data:
        text = item.get("text", "")
        if not text:
            continue

        dialogues.append([Message("assistant", text)])
        labels.append(Label.NEGATIVE)
        metadata["court"].append(item.get("court") or item.get("tribunal"))

    return Dataset(dialogues, labels, "caselaw", metadata).shuffle()


def finance_tasks() -> Dataset:
    """AdaptLLM Finance Tasks for domain adaptation."""
    data = load_dataset("AdaptLLM/finance-tasks")["train"]

    dialogues, labels = [], []

    for item in data:
        instruction = item.get("instruction") or item.get("input") or ""
        output = item.get("output") or item.get("response") or ""
        if not instruction:
            continue

        dialogue = [Message("user", instruction)]
        if output:
            dialogue.append(Message("assistant", output))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "finance_tasks").shuffle()


def financial_phrasebank(config: str = "sentences_allagree") -> Dataset:
    """Financial Phrasebank 4.8K - financial news sentiment."""
    data = load_dataset("takala/financial_phrasebank", config)["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"sentiment": []}

    for item in data:
        if sentence := item.get("sentence", ""):
            dialogues.append([Message("assistant", sentence)])
            labels.append(Label.NEGATIVE)
            metadata["sentiment"].append(SENTIMENT_MAP.get(item.get("label", 1), "neutral"))

    return Dataset(dialogues, labels, "financial_phrasebank", metadata).shuffle()


def legal_advice_reddit() -> Dataset:
    """Legal Advice from Reddit (subset of Pile of Law)."""
    data = load_dataset("pile-of-law/pile-of-law", "r_legaladvice")["train"]

    dialogues, labels = [], []

    for item in data:
        text = item.get("text", "")
        if not text:
            continue

        dialogues.append([Message("user", text)])
        labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "legal_advice_reddit").shuffle()

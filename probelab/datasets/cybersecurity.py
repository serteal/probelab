"""Cybersecurity and defensive security datasets."""

from typing import Any

from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset


def trendyol_cybersecurity() -> Dataset:
    """Trendyol Cybersecurity 53K covering 200+ domains."""
    data = load_dataset("Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset")["train"]

    dialogues, labels = [], []

    for item in data:
        dialogue = []
        if system := item.get("system", ""):
            dialogue.append(Message("system", system))

        user = item.get("user", "")
        assistant = item.get("assistant", "")

        if user:
            dialogue.append(Message("user", user))
        if assistant:
            dialogue.append(Message("assistant", assistant))

        if dialogue:
            dialogues.append(dialogue)
            labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "trendyol_cybersecurity").shuffle()


def cybersecurity_dpo(include_rejected: bool = True) -> Dataset:
    """Code Vulnerability Security DPO - secure vs insecure code pairs."""
    data = load_dataset("CyberNative/Code_Vulnerability_Security_DPO")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"is_secure": []}

    for item in data:
        prompt = item.get("prompt", item.get("instruction", ""))
        if prompt and (chosen := item.get("chosen", "")):
            dialogues.append([Message("user", prompt), Message("assistant", chosen)])
            labels.append(Label.NEGATIVE)
            metadata["is_secure"].append(True)

        if include_rejected and prompt and (rejected := item.get("rejected", "")):
            dialogues.append([Message("user", prompt), Message("assistant", rejected)])
            labels.append(Label.POSITIVE)
            metadata["is_secure"].append(False)

    return Dataset(dialogues, labels, "cybersecurity_dpo", metadata).shuffle()


def defensive_cybersecurity() -> Dataset:
    """Defensive Cybersecurity V1 - 2.5K high-quality pairs."""
    data = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-v1")["train"]

    dialogues, labels = [], []

    for item in data:
        user = item.get("instruction") or item.get("input") or item.get("question") or ""
        assistant = item.get("response") or item.get("output") or item.get("answer") or ""
        if not user:
            continue

        dialogue = [Message("user", user)]
        if assistant:
            dialogue.append(Message("assistant", assistant))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "defensive_cybersecurity").shuffle()

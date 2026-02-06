"""Mental health and counseling conversation datasets."""

from typing import Any

from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset
from .registry import Topic, _register_dataset


@_register_dataset("mentalchat", Topic.MENTAL_HEALTH, "MentalChat")
def mentalchat() -> Dataset:
    """MentalChat16K - synthetic counselor-client conversations."""
    data = load_dataset("ShenLab/MentalChat16K")["train"]

    dialogues, labels = [], []

    for item in data:
        dialogue = []
        if instruction := item.get("instruction", ""):
            dialogue.append(Message("system", instruction))

        user_input = item.get("input", "")
        output = item.get("output", "")

        if user_input:
            dialogue.append(Message("user", user_input))
        if output:
            dialogue.append(Message("assistant", output))

        if dialogue:
            dialogues.append(dialogue)
            labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "mentalchat").shuffle()


@_register_dataset("mental_health_counseling", Topic.MENTAL_HEALTH, "Mental health counseling")
def mental_health_counseling() -> Dataset:
    """Mental Health Counseling Conversations."""
    data = load_dataset("Amod/mental_health_counseling_conversations")["train"]

    dialogues, labels = [], []

    for item in data:
        user = item.get("Context") or item.get("context") or item.get("input") or ""
        assistant = item.get("Response") or item.get("response") or item.get("output") or ""
        if not user:
            continue

        dialogue = [Message("user", user)]
        if assistant:
            dialogue.append(Message("assistant", assistant))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "mental_health_counseling").shuffle()


@_register_dataset("prosocial_dialog", Topic.MENTAL_HEALTH, "Prosocial dialog")
def prosocial_dialog() -> Dataset:
    """ProsocialDialog 58K - prosocial responses to problematic content."""
    data = load_dataset("allenai/prosocial-dialog")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"safety_label": [], "rots": [], "source": []}

    for item in data:
        context = item.get("context", "")
        response = item.get("response", "")
        if not context:
            continue

        dialogue = [Message("user", context)]
        if response:
            dialogue.append(Message("assistant", response))

        dialogues.append(dialogue)
        # Label based on safety_label: casual is negative, others are positive
        safety_label = item.get("safety_label", "__casual__")
        labels.append(Label.NEGATIVE if safety_label == "__casual__" else Label.POSITIVE)
        metadata["safety_label"].append(safety_label)
        metadata["rots"].append(item.get("rots"))
        metadata["source"].append(item.get("source"))

    return Dataset(dialogues, labels, "prosocial_dialog", metadata).shuffle()


@_register_dataset("emotional_support", Topic.MENTAL_HEALTH, "Emotional support")
def emotional_support() -> Dataset:
    """Mental Health Chatbot Dataset for emotional support."""
    data = load_dataset("heliosbrahma/mental_health_chatbot_dataset")["train"]

    dialogues, labels = [], []

    for item in data:
        user = item.get("question") or item.get("input") or item.get("text") or item.get("context") or ""
        assistant = item.get("answer") or item.get("output") or item.get("response") or item.get("reply") or ""
        if not user:
            continue

        dialogue = [Message("user", user)]
        if assistant:
            dialogue.append(Message("assistant", assistant))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "emotional_support").shuffle()

"""Agent and function calling datasets."""

import json
from typing import Any

from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset
from .builders import build_from_messages


def xlam_function_calling() -> Dataset:
    """xLAM Function Calling 60K from Salesforce."""
    data = load_dataset("Salesforce/xlam-function-calling-60k")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"tools": [], "answers": []}

    for item in data:
        query = item.get("query", "")
        if not query:
            continue

        try:
            tools = json.dumps(json.loads(item.get("tools", "")) if isinstance(item.get("tools"), str) else item.get("tools", ""), indent=2)
        except (json.JSONDecodeError, TypeError):
            tools = str(item.get("tools", ""))

        try:
            answers = json.dumps(json.loads(item.get("answers", "")) if isinstance(item.get("answers"), str) else item.get("answers", ""), indent=2)
        except (json.JSONDecodeError, TypeError):
            answers = str(item.get("answers", ""))

        dialogues.append([
            Message("system", f"You have access to the following tools:\n{tools}"),
            Message("user", query),
            Message("assistant", f"Function calls:\n{answers}"),
        ])
        labels.append(Label.NEGATIVE)
        metadata["tools"].append(item.get("tools"))
        metadata["answers"].append(item.get("answers"))

    return Dataset(dialogues, labels, "xlam_function_calling", metadata).shuffle()


def glaive_function_calling() -> Dataset:
    """Glaive Function Calling 52K."""
    data = load_dataset("glaiveai/glaive-function-calling")["train"]

    dialogues, labels = [], []
    role_map = {"SYSTEM:": "system", "USER:": "user", "ASSISTANT:": "assistant", "FUNCTION RESPONSE:": "user"}

    for item in data:
        text = item.get("sample", "")
        if not text:
            continue

        dialogue, current_role, current_content = [], None, []

        for line in text.split("\n"):
            found_role, line_content = None, ""
            for prefix, role in role_map.items():
                if line.strip().startswith(prefix):
                    found_role, line_content = role, line.strip()[len(prefix):].strip()
                    break

            if found_role:
                if current_role and (content := "\n".join(current_content).strip()):
                    dialogue.append(Message(current_role, content))
                current_role, current_content = found_role, [line_content] if line_content else []
            elif current_role:
                current_content.append(line)

        if current_role and (content := "\n".join(current_content).strip()):
            dialogue.append(Message(current_role, content))

        if dialogue:
            dialogues.append(dialogue)
            labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "glaive_function_calling").shuffle()


def hermes_function_calling() -> Dataset:
    """Hermes Function Calling V1 from NousResearch."""
    data = load_dataset("NousResearch/hermes-function-calling-v1")["train"]

    dialogues, labels = [], []

    for item in data:
        dialogue = build_from_messages(item.get("conversations", []))
        if dialogue:
            dialogues.append(dialogue)
            labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "hermes_function_calling").shuffle()


def function_calling_sharegpt() -> Dataset:
    """Function Calling ShareGPT 86K."""
    data = load_dataset("hypervariance/function-calling-sharegpt")["train"]

    dialogues, labels = [], []

    for item in data:
        dialogue = build_from_messages(item.get("conversations", []), role_field="from", content_field="value")
        if dialogue:
            dialogues.append(dialogue)
            labels.append(Label.NEGATIVE)

    return Dataset(dialogues, labels, "function_calling_sharegpt").shuffle()

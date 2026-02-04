"""Math and reasoning datasets."""

from typing import Any

from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset


def openr1_math(config: str = "default") -> Dataset:
    """OpenR1-Math-220k - large-scale math reasoning."""
    data = load_dataset("open-r1/OpenR1-Math-220k", config)["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"problem_type": [], "source": [], "answer": []}

    for item in data:
        problem = item.get("problem", "")
        solution = item.get("solution", "")

        if not problem:
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    problem = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    solution = msg.get("content", "")

        if not problem:
            continue

        dialogue = [Message("user", problem)]
        if solution:
            dialogue.append(Message("assistant", solution))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)
        metadata["problem_type"].append(item.get("problem_type"))
        metadata["source"].append(item.get("source"))
        metadata["answer"].append(item.get("answer"))

    return Dataset(dialogues, labels, "openr1_math", metadata).shuffle()


def openmath_reasoning() -> Dataset:
    """OpenMathReasoning 306K from NVIDIA."""
    data = load_dataset("nvidia/OpenMathReasoning")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"problem_source": []}

    for item in data:
        user = item.get("problem") or item.get("question") or item.get("input") or ""
        asst = item.get("solution") or item.get("response") or item.get("output") or ""
        if not user:
            continue

        dialogue = [Message("user", user)]
        if asst:
            dialogue.append(Message("assistant", asst))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)
        metadata["problem_source"].append(item.get("source"))

    return Dataset(dialogues, labels, "openmath_reasoning", metadata).shuffle()


def numina_math() -> Dataset:
    """NuminaMath CoT 860K+ competition problems."""
    data = load_dataset("AI-MO/NuminaMath-CoT")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"source": []}

    for item in data:
        problem = item.get("problem", "")
        solution = item.get("solution", "")

        if not problem:
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    problem = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    solution = msg.get("content", "")

        if not problem:
            continue

        dialogue = [Message("user", problem)]
        if solution:
            dialogue.append(Message("assistant", solution))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)
        metadata["source"].append(item.get("source"))

    return Dataset(dialogues, labels, "numina_math", metadata).shuffle()


def math_instruct_enhanced() -> Dataset:
    """Enhanced MathInstruct from TIGER-Lab."""
    data = load_dataset("TIGER-Lab/MathInstruct")["train"]

    dialogues, labels = [], []
    metadata: dict[str, list[Any]] = {"source": []}

    for item in data:
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        if not instruction:
            continue

        dialogue = [Message("user", instruction)]
        if output:
            dialogue.append(Message("assistant", output))

        dialogues.append(dialogue)
        labels.append(Label.NEGATIVE)
        metadata["source"].append(item.get("source"))

    return Dataset(dialogues, labels, "math_instruct_enhanced", metadata).shuffle()

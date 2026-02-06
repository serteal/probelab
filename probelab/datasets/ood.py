"""Out-of-distribution / general conversation datasets."""

from datasets import load_dataset

from ..types import Label, Message
from .base import Dataset
from .builders import build_from_messages
from .registry import Topic, _register_dataset


@_register_dataset("ultrachat", Topic.OOD, "UltraChat dataset")
def ultrachat() -> Dataset:
    data = load_dataset("causal-lm/ultrachat")["train"]
    dialogues = [[Message("user", e["instruction"]), Message("assistant", e["output"])] for e in data]
    return Dataset(dialogues, [Label.NEGATIVE] * len(dialogues), "ultrachat").shuffle()


@_register_dataset("ultrachat_200k", Topic.OOD, "UltraChat 200K")
def ultrachat_200k(split: str = "train_sft") -> Dataset:
    data = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    dialogues = [build_from_messages(item.get("messages", [])) for item in data]
    dialogues = [d for d in dialogues if d]
    return Dataset(dialogues, [Label.NEGATIVE] * len(dialogues), "ultrachat_200k").shuffle()


@_register_dataset("lmsys_chat", Topic.OOD, "LMSYS chat")
def lmsys_chat() -> Dataset:
    data = load_dataset("lmsys/lmsys-chat-1m")["train"]
    dialogues, metadata = [], {"model": [], "language": [], "redacted": [], "conversation_id": []}

    for item in data:
        conv = build_from_messages(item.get("conversation", []))
        if conv:
            dialogues.append(conv)
            metadata["model"].append(item.get("model"))
            metadata["language"].append(item.get("language"))
            metadata["redacted"].append(item.get("redacted", False))
            metadata["conversation_id"].append(item.get("conversation_id"))

    return Dataset(dialogues, [Label.NEGATIVE] * len(dialogues), "lmsys_chat", metadata).shuffle()


@_register_dataset("sharegpt", Topic.OOD, "ShareGPT dataset")
def sharegpt() -> Dataset:
    data = load_dataset("RyokoAI/ShareGPT52K")["train"]
    role_map = {"human": "user", "gpt": "assistant"}
    dialogues, metadata = [], {"conversation_id": []}

    for item in data:
        conv = []
        for msg in item.get("conversations", []):
            role = role_map.get(msg.get("from", "human"), "user")
            content = msg.get("value", "") or msg.get("text", "") or ""
            if content:
                conv.append(Message(role, content))
        if conv:
            dialogues.append(conv)
            metadata["conversation_id"].append(item.get("id"))

    return Dataset(dialogues, [Label.NEGATIVE] * len(dialogues), "sharegpt", metadata).shuffle()


@_register_dataset("chatbot_arena", Topic.OOD, "Chatbot arena")
def chatbot_arena() -> Dataset:
    data = load_dataset("lmsys/chatbot_arena_conversations")["train"]
    dialogues, metadata = [], {"model_a": [], "model_b": [], "winner": [], "language": []}

    for item in data:
        conv = build_from_messages(item.get("conversation_a", []))
        if conv:
            dialogues.append(conv)
            metadata["model_a"].append(item.get("model_a"))
            metadata["model_b"].append(item.get("model_b"))
            metadata["winner"].append(item.get("winner"))
            metadata["language"].append(item.get("language"))

    return Dataset(dialogues, [Label.NEGATIVE] * len(dialogues), "chatbot_arena", metadata).shuffle()


@_register_dataset("alpaca", Topic.OOD, "Alpaca dataset")
def alpaca() -> Dataset:
    data = load_dataset("yahma/alpaca-cleaned")["train"]
    dialogues = [[Message("user", e["instruction"]), Message("assistant", e["output"])] for e in data]
    return Dataset(dialogues, [Label.NEGATIVE] * len(dialogues), "alpaca").shuffle()


@_register_dataset("generic_spanish", Topic.OOD, "Generic Spanish")
def generic_spanish() -> Dataset:
    data = load_dataset("FreedomIntelligence/evol-instruct-spanish")["train"]
    dialogues = [[Message("user", e["conversations"][0]["value"]), Message("assistant", e["conversations"][1]["value"])] for e in data]
    return Dataset(dialogues, [Label.NEGATIVE] * len(dialogues), "generic_spanish").shuffle()


@_register_dataset("math_instruct", Topic.OOD, "Math instruct")
def math_instruct() -> Dataset:
    data = load_dataset("TIGER-Lab/MathInstruct")["train"]
    dialogues = [[Message("user", e["instruction"]), Message("assistant", e["output"])] for e in data]
    return Dataset(dialogues, [Label.NEGATIVE] * len(dialogues), "math_instruct").shuffle()

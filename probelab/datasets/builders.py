"""Reusable dialogue building utilities for HuggingFace datasets."""

from __future__ import annotations

from ..types import Dialogue, Message

# Role normalization mapping
ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "patient": "user",
    "doctor": "assistant",
    "tool": "user",
    "ai": "assistant",
    "bot": "assistant",
}


def normalize_role(role: str) -> str:
    """Normalize role identifiers across datasets."""
    return ROLE_MAP.get(role.lower(), role.lower())


def get_field(item: dict, *field_names: str, default: str = "") -> str:
    """Try multiple field names, return first non-empty value."""
    for field in field_names:
        val = item.get(field)
        if val:
            return str(val)
    return default


def sample_hf_dataset(dataset, max_samples: int | None, seed: int = 42):
    """Apply random sampling to HuggingFace dataset."""
    if max_samples and len(dataset) > max_samples:
        import random

        random.seed(seed)
        indices = random.sample(range(len(dataset)), max_samples)
        return dataset.select(indices)
    return dataset


def build_from_messages(
    messages: list[dict],
    role_field: str = "role",
    content_field: str = "content",
    role_fallback: str = "from",
    content_fallback: str = "value",
) -> Dialogue:
    """Build dialogue from message array [{role, content}, ...]."""
    dialogue: Dialogue = []
    for msg in messages:
        role = normalize_role(msg.get(role_field, msg.get(role_fallback, "user")))
        content = msg.get(content_field, msg.get(content_fallback, ""))
        if role in ("user", "assistant", "system") and content:
            dialogue.append(Message(role=role, content=content))
    return dialogue


def build_from_fields(
    item: dict,
    *,
    system_fields: tuple[str, ...] = (),
    user_fields: tuple[str, ...] = (),
    assistant_fields: tuple[str, ...] = (),
) -> Dialogue:
    """Build dialogue from named fields."""
    dialogue: Dialogue = []

    if system_fields:
        system = get_field(item, *system_fields)
        if system:
            dialogue.append(Message(role="system", content=system))

    if user_fields:
        user = get_field(item, *user_fields)
        if user:
            dialogue.append(Message(role="user", content=user))

    if assistant_fields:
        assistant = get_field(item, *assistant_fields)
        if assistant:
            dialogue.append(Message(role="assistant", content=assistant))

    return dialogue


def build_text_only(text: str, as_assistant: bool = True) -> Dialogue:
    """Build single-message dialogue from text."""
    role = "assistant" if as_assistant else "user"
    return [Message(role=role, content=text)] if text else []

"""Reusable dialogue building utilities."""

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


def build_from_messages(
    messages: list[dict],
    role_field: str = "role",
    content_field: str = "content",
) -> Dialogue:
    """Build dialogue from message array [{role, content}, ...]."""
    dialogue: Dialogue = []
    for msg in messages:
        role = normalize_role(msg.get(role_field, msg.get("from", "user")))
        content = msg.get(content_field, msg.get("value", ""))
        if role in ("user", "assistant", "system") and content:
            dialogue.append(Message(role=role, content=content))
    return dialogue

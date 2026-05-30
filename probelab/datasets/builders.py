"""Reusable dialogue building utilities."""

from collections.abc import Sequence

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


# Common field aliases seen across HuggingFace QA-style datasets.
_USER_KEYS = ("question", "prompt", "instruction", "input", "query", "text")
_ASSISTANT_KEYS = ("answer", "response", "output", "completion", "target")


def build_qa(
    item: dict,
    *,
    user_keys: Sequence[str] = _USER_KEYS,
    assistant_keys: Sequence[str] = _ASSISTANT_KEYS,
    system_keys: Sequence[str] = (),
) -> Dialogue | None:
    """Build a (optionally system-prefixed) single-turn dialogue from a record.

    Tries each key in ``user_keys`` / ``assistant_keys`` / ``system_keys`` in
    order and uses the first non-empty value. This consolidates the
    try-several-field-aliases pattern repeated across topic loaders.

    Returns ``None`` when no user content is found, so callers can skip the row.
    """

    def _first(keys: Sequence[str]) -> str | None:
        for key in keys:
            value = item.get(key)
            if value:
                return str(value)
        return None

    user = _first(user_keys)
    if not user:
        return None

    dialogue: Dialogue = []
    system = _first(system_keys) if system_keys else None
    if system:
        dialogue.append(Message(role="system", content=system))
    dialogue.append(Message(role="user", content=user))
    assistant = _first(assistant_keys)
    if assistant:
        dialogue.append(Message(role="assistant", content=assistant))
    return dialogue

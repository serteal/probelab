"""Mask functions for selective token processing."""

from .base import MaskFunction, TokenMetadata
from .basic import all, first_n_tokens, last_n_tokens, last_token, none
from .composite import AndMask, NotMask, OrMask
from .content import special_tokens
from .position import after, before, between, nth_message, padding
from .role import assistant, role, system, user
from .text import contains, regex

__all__ = [
    # Base classes
    "MaskFunction",
    "TokenMetadata",
    # Basic masks
    "all",
    "none",
    "last_token",
    "last_n_tokens",
    "first_n_tokens",
    # Role masks
    "role",
    "assistant",
    "user",
    "system",
    # Text masks
    "contains",
    "regex",
    # Position masks
    "between",
    "after",
    "before",
    "nth_message",
    "padding",
    # Content masks
    "special_tokens",
    # Composite masks
    "AndMask",
    "OrMask",
    "NotMask",
]

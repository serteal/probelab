"""Functional mask system for selective token processing.

Masks are callable objects that select tokens based on dialogues and metadata.
They compose with boolean operators (&, |, ~) for complex selection logic.

Usage:
    from probelab import masks

    # Simple masks
    mask = masks.assistant()
    mask = masks.nth_message(-1)

    # Composed masks
    mask = masks.assistant() & masks.nth_message(-1)
    mask = masks.contains("yes") | masks.contains("no")
    mask = ~masks.user()  # All non-user tokens
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .types import Dialogue


# =============================================================================
# Core Types
# =============================================================================


@dataclass(slots=True)
class TokenMetadata:
    """Metadata for efficient mask evaluation."""

    token_ids: Tensor  # [batch, seq_len]
    role_ids: Tensor  # [batch, seq_len] - 0=system, 1=user, 2=assistant, -1=none
    message_boundaries: Tensor  # [batch, seq_len] - message index per token
    attention_mask: Tensor  # [batch, seq_len] - 1 for real tokens, 0 for padding

    # Optional fields for text-based masks
    formatted_texts: Sequence[str] | None = None
    char_to_token: Callable[[int, int], int | None] | None = None
    token_to_char: dict | None = None

    # Padding information
    role_ids_no_padding: Tensor | None = None
    architecture: str | None = None
    special_token_ids: set[int] | None = None


@dataclass(frozen=True, slots=True)
class Mask:
    """Composable token selector.

    A Mask wraps a function (dialogues, metadata) -> bool tensor and provides
    boolean composition via &, |, ~. The key tuple enables hashing for caching.
    """

    fn: Callable[[Sequence["Dialogue"], TokenMetadata], Tensor]
    key: tuple = field(default=())

    def __call__(self, dialogues: Sequence["Dialogue"], metadata: TokenMetadata) -> Tensor:
        return self.fn(dialogues, metadata)

    def __and__(self, other: Mask) -> Mask:
        return Mask(lambda d, m: self(d, m) & other(d, m), ("&", self.key, other.key))

    def __or__(self, other: Mask) -> Mask:
        return Mask(lambda d, m: self(d, m) | other(d, m), ("|", self.key, other.key))

    def __invert__(self) -> Mask:
        return Mask(lambda d, m: ~self(d, m) & m.attention_mask.bool(), ("~", self.key))

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mask) and self.key == other.key

    def __repr__(self) -> str:
        return f"Mask({self.key})"


# =============================================================================
# Basic Masks
# =============================================================================


def all() -> Mask:
    """Select all non-padding tokens."""
    return Mask(lambda d, m: m.attention_mask.bool(), ("all",))


def none() -> Mask:
    """Select no tokens."""
    return Mask(lambda d, m: torch.zeros_like(m.attention_mask, dtype=torch.bool), ("none",))


# =============================================================================
# Role Masks
# =============================================================================

_ROLE_IDS = {"system": 0, "user": 1, "assistant": 2}


def role(r: str, include_padding: bool = True) -> Mask:
    """Select tokens from messages with a specific role."""
    role_id = _ROLE_IDS.get(r)
    if role_id is None:
        raise ValueError(f"Invalid role: {r}. Must be one of {list(_ROLE_IDS.keys())}")

    def _impl(dialogues, metadata):
        if include_padding:
            rids = metadata.role_ids
        else:
            rids = metadata.role_ids_no_padding if metadata.role_ids_no_padding is not None else metadata.role_ids
        return (rids == role_id) & metadata.attention_mask.bool()

    return Mask(_impl, ("role", r, include_padding))


def assistant(include_padding: bool = True) -> Mask:
    """Select tokens from assistant messages."""
    return role("assistant", include_padding)


def user(include_padding: bool = True) -> Mask:
    """Select tokens from user messages."""
    return role("user", include_padding)


def system(include_padding: bool = True) -> Mask:
    """Select tokens from system messages."""
    return role("system", include_padding)


# =============================================================================
# Position Masks
# =============================================================================


def nth_message(n: int) -> Mask:
    """Select tokens from the nth message (negative indices count from end)."""

    def _impl(dialogues, metadata):
        batch_size, seq_len = metadata.attention_mask.shape
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        for b in range(batch_size):
            mb = metadata.message_boundaries[b]
            num_msgs = int(mb.max().item()) + 1 if (mb >= 0).any() else 0
            target = n if n >= 0 else num_msgs + n
            if 0 <= target < num_msgs:
                mask[b] = mb == target

        return mask & metadata.attention_mask.bool()

    return Mask(_impl, ("nth_message", n))


def last_n_tokens(n: int) -> Mask:
    """Select the last n tokens of each message."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    def _impl(dialogues, metadata):
        batch_size, seq_len = metadata.attention_mask.shape
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)
        role_ids = metadata.role_ids

        for b in range(batch_size):
            valid_mask = metadata.attention_mask[b].bool()
            if not valid_mask.any():
                continue

            has_role = role_ids[b] != -1

            # Find message boundaries (where role changes)
            role_changes = torch.zeros(seq_len, dtype=torch.bool, device=role_ids.device)
            for i in range(seq_len - 1):
                if valid_mask[i] and has_role[i]:
                    if not valid_mask[i + 1] or not has_role[i + 1] or role_ids[b, i] != role_ids[b, i + 1]:
                        role_changes[i] = True

            # Last valid position with a role is also a boundary
            valid_with_role = valid_mask & has_role
            last_valid = torch.where(valid_with_role)[0]
            if len(last_valid) > 0:
                role_changes[last_valid[-1]] = True

            boundaries = torch.where(role_changes)[0]

            # Mark last n tokens of each segment
            for i, end_idx in enumerate(boundaries):
                if i == 0:
                    start_candidates = torch.where(has_role[: end_idx + 1])[0]
                    if len(start_candidates) > 0:
                        start_idx = start_candidates[0].item()
                    else:
                        continue
                else:
                    start_idx = boundaries[i - 1].item() + 1

                segment_length = end_idx - start_idx + 1
                tokens_to_select = min(n, segment_length)
                for j in range(tokens_to_select):
                    mask[b, end_idx - j] = True

        return mask & metadata.attention_mask.bool()

    return Mask(_impl, ("last_n_tokens", n))


def last_token() -> Mask:
    """Select the last token of each message."""
    return last_n_tokens(1)


def first_n_tokens(n: int) -> Mask:
    """Select the first n tokens of each message."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    def _impl(dialogues, metadata):
        batch_size, seq_len = metadata.attention_mask.shape
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)
        role_ids = metadata.role_ids

        for b in range(batch_size):
            valid_mask = metadata.attention_mask[b].bool()
            if not valid_mask.any():
                continue

            has_role = role_ids[b] != -1
            valid_with_role = valid_mask & has_role

            # Find message starts (where role changes from previous)
            role_changes = torch.zeros(seq_len, dtype=torch.bool, device=role_ids.device)

            first_valid = torch.where(valid_with_role)[0]
            if len(first_valid) > 0:
                role_changes[first_valid[0]] = True

            for i in range(1, seq_len):
                if valid_mask[i] and has_role[i]:
                    if not valid_mask[i - 1] or not has_role[i - 1] or role_ids[b, i] != role_ids[b, i - 1]:
                        role_changes[i] = True

            boundaries = torch.where(role_changes)[0]

            # Mark first n tokens of each segment
            for i, start_idx in enumerate(boundaries):
                if i + 1 < len(boundaries):
                    end_idx = boundaries[i + 1] - 1
                else:
                    last_valid = torch.where(valid_with_role)[0]
                    end_idx = last_valid[-1] if len(last_valid) > 0 else start_idx

                segment_length = end_idx - start_idx + 1
                tokens_to_select = min(n, segment_length)
                for j in range(tokens_to_select):
                    mask[b, start_idx + j] = True

        return mask & metadata.attention_mask.bool()

    return Mask(_impl, ("first_n_tokens", n))


def between(start: str, end: str, inclusive: bool = False) -> Mask:
    """Select tokens between two strings."""
    start_pat = re.compile(re.escape(start))
    end_pat = re.compile(re.escape(end))

    def _impl(dialogues, metadata):
        batch_size, seq_len = metadata.attention_mask.shape
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        if metadata.formatted_texts is None or metadata.char_to_token is None:
            return mask

        for b in range(batch_size):
            if not metadata.attention_mask[b].bool().any():
                continue

            text = metadata.formatted_texts[b]
            start_matches = list(start_pat.finditer(text))
            end_matches = list(end_pat.finditer(text))

            if not start_matches or not end_matches:
                continue

            # Find start token indices
            start_indices = []
            for match in start_matches:
                start_tok = metadata.char_to_token(b, match.start())
                end_tok = metadata.char_to_token(b, match.end() - 1) if match.end() > 0 else None
                if start_tok is not None:
                    if inclusive:
                        start_indices.append(start_tok)
                    elif end_tok is not None and end_tok + 1 < seq_len:
                        start_indices.append(end_tok + 1)

            # Find end token indices
            end_indices = []
            for match in end_matches:
                start_tok = metadata.char_to_token(b, match.start())
                if start_tok is not None:
                    if inclusive:
                        end_tok = metadata.char_to_token(b, match.end() - 1) if match.end() > 0 else None
                        if end_tok is not None:
                            end_indices.append(end_tok)
                    elif start_tok > 0:
                        end_indices.append(start_tok - 1)

            if start_indices and end_indices:
                start_tensor = torch.tensor(start_indices, device=mask.device)
                end_tensor = torch.tensor(end_indices, device=mask.device)

                for start_idx in start_tensor:
                    valid_ends = end_tensor[end_tensor > start_idx]
                    if len(valid_ends) > 0:
                        end_idx = valid_ends[0]
                        mask[b, start_idx : end_idx + 1] = True

        return mask & metadata.attention_mask.bool()

    return Mask(_impl, ("between", start, end, inclusive))


def after(string: str, inclusive: bool = False) -> Mask:
    """Select tokens after a string."""

    def _impl(dialogues, metadata):
        batch_size, seq_len = metadata.attention_mask.shape
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        if metadata.formatted_texts is None or metadata.char_to_token is None:
            return mask

        for b in range(batch_size):
            if not metadata.attention_mask[b].bool().any():
                continue

            text = metadata.formatted_texts[b]
            idx = text.find(string)
            if idx != -1:
                if inclusive:
                    start_tok = metadata.char_to_token(b, idx)
                else:
                    start_tok = metadata.char_to_token(b, idx + len(string))

                if start_tok is not None:
                    mask[b, start_tok:] = True

        return mask & metadata.attention_mask.bool()

    return Mask(_impl, ("after", string, inclusive))


def before(string: str, inclusive: bool = False) -> Mask:
    """Select tokens before a string."""

    def _impl(dialogues, metadata):
        batch_size, seq_len = metadata.attention_mask.shape
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        if metadata.formatted_texts is None or metadata.char_to_token is None:
            return mask

        for b in range(batch_size):
            if not metadata.attention_mask[b].bool().any():
                continue

            text = metadata.formatted_texts[b]
            idx = text.find(string)
            if idx != -1:
                if inclusive:
                    end_tok = metadata.char_to_token(b, idx + len(string) - 1)
                else:
                    end_tok = metadata.char_to_token(b, idx - 1) if idx > 0 else None

                if end_tok is not None:
                    mask[b, : end_tok + 1] = True

        return mask & metadata.attention_mask.bool()

    return Mask(_impl, ("before", string, inclusive))


def padding(base_mask: Mask, before: int = 2, after: int = 2) -> Mask:
    """Expand a mask by adding surrounding context tokens."""

    def _impl(dialogues, metadata):
        base_result = base_mask(dialogues, metadata)
        batch_size, seq_len = base_result.shape
        expanded = base_result.clone()

        for b in range(batch_size):
            base_b = base_result[b]
            if not base_b.any():
                continue

            padded = torch.cat([
                torch.tensor([False], device=base_b.device),
                base_b,
                torch.tensor([False], device=base_b.device),
            ])

            transitions = padded[1:] != padded[:-1]
            starts = torch.where(transitions & padded[1:])[0]
            ends = torch.where(transitions & ~padded[1:])[0]

            for s, e in zip(starts, ends):
                new_start = max(0, s - before)
                new_end = min(seq_len, e + after)
                expanded[b, new_start:new_end] = True

        return expanded & metadata.attention_mask.bool()

    return Mask(_impl, ("padding", base_mask.key, before, after))


# =============================================================================
# Text Masks
# =============================================================================


def contains(text: str, case_sensitive: bool = False) -> Mask:
    """Select tokens containing specific text."""
    search_text = text if case_sensitive else text.lower()

    def _impl(dialogues, metadata):
        batch_size, seq_len = metadata.token_ids.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=metadata.token_ids.device)

        if not search_text or metadata.formatted_texts is None or metadata.char_to_token is None:
            return mask

        for b, formatted_text in enumerate(metadata.formatted_texts):
            haystack = formatted_text if case_sensitive else formatted_text.lower()

            # Find all occurrences
            positions = []
            start = 0
            while True:
                pos = haystack.find(search_text, start)
                if pos == -1:
                    break
                positions.append((pos, pos + len(search_text)))
                start = pos + 1

            if not positions:
                continue

            token_set = set()
            for start_pos, end_pos in positions:
                start_token = metadata.char_to_token(b, start_pos)
                end_token = metadata.char_to_token(b, end_pos - 1)

                if start_token is not None and end_token is not None:
                    for tok_idx in range(start_token, end_token + 1):
                        token_set.add(tok_idx)
                elif start_token is not None:
                    token_set.add(start_token)
                elif end_token is not None:
                    token_set.add(end_token)

            if token_set:
                token_indices = torch.tensor(list(token_set), device=mask.device)
                mask[b, token_indices] = True

        return mask & metadata.attention_mask.bool()

    return Mask(_impl, ("contains", search_text, case_sensitive))


def regex(pattern: str, flags: int = 0) -> Mask:
    """Select tokens matching a regex pattern."""
    compiled = re.compile(pattern, flags)

    def _impl(dialogues, metadata):
        batch_size, seq_len = metadata.token_ids.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=metadata.token_ids.device)

        if metadata.formatted_texts is None or metadata.char_to_token is None:
            return mask

        for b, formatted_text in enumerate(metadata.formatted_texts):
            matches = list(compiled.finditer(formatted_text))
            if not matches:
                continue

            token_set = set()
            for match in matches:
                start_pos, end_pos = match.span()
                start_token = metadata.char_to_token(b, start_pos)
                end_token = metadata.char_to_token(b, end_pos - 1) if end_pos > start_pos else start_token

                if start_token is not None and end_token is not None:
                    for tok_idx in range(start_token, end_token + 1):
                        token_set.add(tok_idx)
                elif start_token is not None:
                    token_set.add(start_token)
                elif end_token is not None:
                    token_set.add(end_token)

            if token_set:
                token_indices = torch.tensor(list(token_set), device=mask.device)
                mask[b, token_indices] = True

        return mask & metadata.attention_mask.bool()

    return Mask(_impl, ("regex", pattern, flags))


# =============================================================================
# Content Masks
# =============================================================================


def special_tokens(special_token_ids: set[int] | None = None) -> Mask:
    """Select special tokens (BOS, EOS, PAD, etc.)."""
    # Common special token IDs as fallback
    _DEFAULT_SPECIAL = {0, 1, 2, 3, 106, 107, 128000, 128001, 128256}

    def _impl(dialogues, metadata):
        batch_size, seq_len = metadata.attention_mask.shape
        token_ids = metadata.token_ids

        # Determine special token IDs
        if special_token_ids is not None:
            ids = special_token_ids
        elif metadata.special_token_ids:
            ids = {int(t) for t in metadata.special_token_ids}
        else:
            ids = _DEFAULT_SPECIAL

        if not ids:
            return torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        special_tensor = torch.tensor(list(ids), device=token_ids.device, dtype=token_ids.dtype)
        mask = torch.isin(token_ids, special_tensor)

        return mask & metadata.attention_mask.bool()

    key = ("special_tokens", tuple(sorted(special_token_ids)) if special_token_ids else None)
    return Mask(_impl, key)


# =============================================================================
# Composite Masks (for explicit construction)
# =============================================================================


def AndMask(mask1: Mask, mask2: Mask) -> Mask:
    """Logical AND of two masks."""
    return mask1 & mask2


def OrMask(mask1: Mask, mask2: Mask) -> Mask:
    """Logical OR of two masks."""
    return mask1 | mask2


def NotMask(mask: Mask, *, keep_padding_masked: bool = True) -> Mask:
    """Logical NOT of a mask.

    Args:
        mask: Mask to negate.
        keep_padding_masked: If True (default), padding tokens remain masked.
    """
    if keep_padding_masked:
        return ~mask
    else:
        return Mask(lambda d, m: ~mask(d, m), ("~raw", mask.key))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core types
    "Mask",
    "TokenMetadata",
    # Basic masks
    "all",
    "none",
    # Role masks
    "role",
    "assistant",
    "user",
    "system",
    # Position masks
    "nth_message",
    "last_token",
    "last_n_tokens",
    "first_n_tokens",
    "between",
    "after",
    "before",
    "padding",
    # Text masks
    "contains",
    "regex",
    # Content masks
    "special_tokens",
    # Composite constructors
    "AndMask",
    "OrMask",
    "NotMask",
]

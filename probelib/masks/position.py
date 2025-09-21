"""Position-based mask functions for token selection."""

from typing import Sequence

import torch
from torch import Tensor

from ..types import Dialogue
from .base import MaskFunction, TokenMetadata


class BetweenMask(MaskFunction):
    """Mask that selects tokens between two strings."""

    def __init__(self, start: str, end: str, inclusive: bool = False):
        """
        Args:
            start: String marking the start of selection
            end: String marking the end of selection
            inclusive: Whether to include the start/end tokens themselves
        """
        self.start = start
        self.end = end
        self.inclusive = inclusive
        # Pre-compile regex patterns for efficiency
        import re

        self._start_pattern = re.compile(re.escape(start))
        self._end_pattern = re.compile(re.escape(end))

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select tokens between start and end strings."""
        batch_size, seq_len = metadata.attention_mask.shape
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        # Early return if no formatted texts available
        if metadata.formatted_texts is None or metadata.char_to_token is None:
            return mask

        # Process each sequence in the batch
        for b in range(batch_size):
            # Get valid tokens for this sequence
            valid_mask = metadata.attention_mask[b].bool()
            if not valid_mask.any():
                continue

            text = metadata.formatted_texts[b]

            # Find all matches using pre-compiled patterns
            start_matches = list(self._start_pattern.finditer(text))
            end_matches = list(self._end_pattern.finditer(text))

            # Early exit if no matches
            if not start_matches or not end_matches:
                continue

            # Pre-calculate all start and end token indices
            start_indices = []
            for start_match in start_matches:
                start_char = start_match.start()
                end_char = start_match.end()

                # Find token containing this character position
                start_tok = metadata.char_to_token(b, start_char)
                end_tok = (
                    metadata.char_to_token(b, end_char - 1) if end_char > 0 else None
                )
                if start_tok is not None:
                    if self.inclusive:
                        start_indices.append(start_tok)
                    else:
                        # Start after the match
                        if end_tok is not None and end_tok + 1 < seq_len:
                            start_indices.append(end_tok + 1)

            end_indices = []
            for end_match in end_matches:
                start_char = end_match.start()
                end_char = end_match.end()

                # Find token containing this character position
                start_tok = metadata.char_to_token(b, start_char)
                if start_tok is not None:
                    if self.inclusive:
                        end_tok = (
                            metadata.char_to_token(b, end_char - 1)
                            if end_char > 0
                            else None
                        )
                        if end_tok is not None:
                            end_indices.append(end_tok)
                    else:
                        # End before the match
                        if start_tok > 0:
                            end_indices.append(start_tok - 1)

            # Efficiently pair up starts and ends using numpy-like operations
            if start_indices and end_indices:
                # Convert to tensors for efficient operations
                start_tensor = torch.tensor(start_indices, device=mask.device)
                end_tensor = torch.tensor(end_indices, device=mask.device)

                # For each start, find the first end after it
                for start_idx in start_tensor:
                    valid_ends = end_tensor[end_tensor > start_idx]
                    if len(valid_ends) > 0:
                        end_idx = valid_ends[0]
                        # Select tokens in range
                        mask[b, start_idx : end_idx + 1] = True

        # Apply attention mask
        mask = mask & metadata.attention_mask.bool()
        return mask

    def __repr__(self) -> str:
        return f"BetweenMask(start={self.start!r}, end={self.end!r}, inclusive={self.inclusive})"


class AfterMask(MaskFunction):
    """Mask that selects tokens after a string."""

    def __init__(self, string: str, inclusive: bool = False):
        """
        Args:
            string: String to search for
            inclusive: Whether to include the matched string
        """
        self.string = string
        self.inclusive = inclusive

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select tokens after the string."""
        batch_size, seq_len = metadata.attention_mask.shape
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        # Process each sequence
        for b in range(batch_size):
            valid_mask = metadata.attention_mask[b].bool()
            if not valid_mask.any():
                continue

            if metadata.formatted_texts is not None:
                text = metadata.formatted_texts[b]

                # Find first occurrence
                idx = text.find(self.string)
                if idx != -1:
                    # Find token position
                    if metadata.char_to_token:
                        if self.inclusive:
                            start_tok = metadata.char_to_token(b, idx)
                        else:
                            # Start after the string
                            end_char = idx + len(self.string)
                            start_tok = metadata.char_to_token(b, end_char)

                        if start_tok is not None:
                            # Select all tokens from this point
                            mask[b, start_tok:] = True

        # Apply attention mask
        mask = mask & metadata.attention_mask.bool()
        return mask

    def __repr__(self) -> str:
        return f"AfterMask(string={self.string!r}, inclusive={self.inclusive})"


class BeforeMask(MaskFunction):
    """Mask that selects tokens before a string."""

    def __init__(self, string: str, inclusive: bool = False):
        """
        Args:
            string: String to search for
            inclusive: Whether to include the matched string
        """
        self.string = string
        self.inclusive = inclusive

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select tokens before the string."""
        batch_size, seq_len = metadata.attention_mask.shape
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        # Process each sequence
        for b in range(batch_size):
            valid_mask = metadata.attention_mask[b].bool()
            if not valid_mask.any():
                continue

            if metadata.formatted_texts is not None:
                text = metadata.formatted_texts[b]

                # Find first occurrence
                idx = text.find(self.string)
                if idx != -1:
                    # Find token position
                    if metadata.char_to_token:
                        if self.inclusive:
                            # Include up to end of string
                            end_char = idx + len(self.string) - 1
                            end_tok = metadata.char_to_token(b, end_char)
                        else:
                            # End before the string
                            if idx > 0:
                                end_tok = metadata.char_to_token(b, idx - 1)
                            else:
                                end_tok = None

                        if end_tok is not None:
                            # Select all tokens up to this point
                            mask[b, : end_tok + 1] = True

        # Apply attention mask
        mask = mask & metadata.attention_mask.bool()
        return mask

    def __repr__(self) -> str:
        return f"BeforeMask(string={self.string!r}, inclusive={self.inclusive})"


class NthMessageMask(MaskFunction):
    """Mask that selects tokens from the nth message."""

    def __init__(self, n: int):
        """
        Args:
            n: Message index (0-based). Negative values count from end.
        """
        self.n = n

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select tokens from the nth message."""
        batch_size, seq_len = metadata.attention_mask.shape
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        # Use message boundaries for efficient selection
        for b in range(batch_size):
            # Determine available message indices from metadata (processed dialogues)
            mb = metadata.message_boundaries[b]
            if (mb >= 0).any():
                max_idx = int(mb.max().item())
                num_messages_processed = max_idx + 1
            else:
                num_messages_processed = 0

            # Compute target index using processed dialogue length (handles Gemma folding)
            target_idx = self.n
            if target_idx < 0:
                target_idx = num_messages_processed + target_idx

            # Check valid range and select tokens
            if 0 <= target_idx < num_messages_processed:
                mask[b] = mb == target_idx

        # Apply attention mask
        mask = mask & metadata.attention_mask.bool()
        return mask

    def __repr__(self) -> str:
        return f"NthMessageMask(n={self.n})"


class PaddingMask(MaskFunction):
    """Mask that expands another mask by adding surrounding context."""

    def __init__(self, base_mask: MaskFunction, before: int = 2, after: int = 2):
        """
        Args:
            base_mask: The mask to expand
            before: Number of tokens to include before selected regions
            after: Number of tokens to include after selected regions
        """
        self.base_mask = base_mask
        self.before = before
        self.after = after

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Expand the base mask with surrounding context."""
        # Get the base mask
        base_result = self.base_mask.evaluate(dialogues, metadata)

        batch_size, seq_len = base_result.shape
        expanded_mask = base_result.clone()

        # Efficiently expand using convolution-like operation
        # For each batch item, find regions and expand them
        for b in range(batch_size):
            # Get the base mask for this batch
            base_b = base_result[b]

            if not base_b.any():
                continue

            # Find transitions (start and end of regions)
            # Pad with False to handle boundaries
            padded = torch.cat(
                [
                    torch.tensor([False], device=base_b.device),
                    base_b,
                    torch.tensor([False], device=base_b.device),
                ]
            )

            # Find where regions start and end
            transitions = padded[1:] != padded[:-1]
            starts = torch.where(transitions & padded[1:])[0]
            ends = torch.where(transitions & ~padded[1:])[0]

            # Expand each region
            for start, end in zip(starts, ends):
                # Expand the region
                new_start = max(0, start - self.before)
                new_end = min(seq_len, end + self.after)
                expanded_mask[b, new_start:new_end] = True

        # Apply attention mask
        expanded_mask = expanded_mask & metadata.attention_mask.bool()
        return expanded_mask

    def __repr__(self) -> str:
        return f"PaddingMask(base_mask={self.base_mask!r}, before={self.before}, after={self.after})"


# Convenience functions
def between(start: str, end: str, inclusive: bool = False) -> BetweenMask:
    """Create a mask for tokens between two strings."""
    return BetweenMask(start, end, inclusive)


def after(string: str, inclusive: bool = False) -> AfterMask:
    """Create a mask for tokens after a string."""
    return AfterMask(string, inclusive)


def before(string: str, inclusive: bool = False) -> BeforeMask:
    """Create a mask for tokens before a string."""
    return BeforeMask(string, inclusive)


def nth_message(n: int) -> NthMessageMask:
    """Create a mask for the nth message (negative indices count from end)."""
    return NthMessageMask(n)


def padding(base_mask: MaskFunction, before: int = 2, after: int = 2) -> PaddingMask:
    """Create a mask that expands another mask with context."""
    return PaddingMask(base_mask, before, after)

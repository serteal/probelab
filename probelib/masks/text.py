"""Text-based mask functions."""

import re
from typing import Sequence

import torch
from torch import Tensor

from ..types import Dialogue
from .base import MaskFunction, TokenMetadata


class ContainsMask(MaskFunction):
    """Mask for selecting tokens that contain specific text."""

    def __init__(self, text: str, case_sensitive: bool = False):
        """
        Args:
            text: Text to search for
            case_sensitive: Whether to do case-sensitive matching
        """
        self.text = text if case_sensitive else text.lower()
        self.case_sensitive = case_sensitive

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select tokens that contain the specified text."""
        batch_size, seq_len = metadata.token_ids.shape
        mask = torch.zeros(
            (batch_size, seq_len), dtype=torch.bool, device=metadata.token_ids.device
        )

        # Handle empty search string - return empty mask
        if not self.text:
            return mask

        if metadata.formatted_texts is None or metadata.char_to_token is None:
            # Can't do text matching without formatted texts and char mapping
            return mask

        # Process each batch
        for batch_idx, formatted_text in enumerate(metadata.formatted_texts):
            search_text = (
                formatted_text if self.case_sensitive else formatted_text.lower()
            )

            # Find all occurrences of the text
            positions = []
            start = 0
            while True:
                pos = search_text.find(self.text, start)
                if pos == -1:
                    break
                positions.append((pos, pos + len(self.text)))
                start = pos + 1

            if not positions:
                continue

            # Batch process token lookups for efficiency
            # Instead of checking every character, check boundaries and a few key points
            token_set = set()
            for start_pos, end_pos in positions:
                # Check start, middle, and end positions to find all spanning tokens
                check_positions = [start_pos]
                if end_pos - start_pos > 1:
                    # Add a middle position for longer texts
                    check_positions.append((start_pos + end_pos) // 2)
                check_positions.append(end_pos - 1)

                # Find token range efficiently
                start_token = metadata.char_to_token(batch_idx, start_pos)
                end_token = metadata.char_to_token(batch_idx, end_pos - 1)

                if start_token is not None and end_token is not None:
                    # All tokens in this range should be selected
                    for tok_idx in range(start_token, end_token + 1):
                        token_set.add(tok_idx)
                elif start_token is not None:
                    token_set.add(start_token)
                elif end_token is not None:
                    token_set.add(end_token)

            # Set mask for all found tokens at once (more efficient than individual sets)
            if token_set:
                token_indices = torch.tensor(list(token_set), device=mask.device)
                mask[batch_idx, token_indices] = True

        # Apply attention mask
        mask = mask & metadata.attention_mask.bool()
        return mask

    def __repr__(self) -> str:
        return f"ContainsMask(text='{self.text}', case_sensitive={self.case_sensitive})"


class RegexMask(MaskFunction):
    """Mask for selecting tokens matching a regex pattern."""

    def __init__(self, pattern: str, flags: int = 0):
        """
        Args:
            pattern: Regex pattern to match
            flags: Regex flags (e.g., re.IGNORECASE)
        """
        self.pattern = pattern
        self.flags = flags
        self.compiled_pattern = re.compile(pattern, flags)

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select tokens matching the regex pattern."""
        batch_size, seq_len = metadata.token_ids.shape
        mask = torch.zeros(
            (batch_size, seq_len), dtype=torch.bool, device=metadata.token_ids.device
        )

        if metadata.formatted_texts is None or metadata.char_to_token is None:
            # Can't do regex matching without formatted texts and char mapping
            return mask

        # Process each batch
        for batch_idx, formatted_text in enumerate(metadata.formatted_texts):
            # Find all matches and collect their positions
            matches = list(self.compiled_pattern.finditer(formatted_text))
            if not matches:
                continue

            # Batch process all matches
            token_set = set()
            for match in matches:
                start_pos, end_pos = match.span()

                # Find token range efficiently
                start_token = metadata.char_to_token(batch_idx, start_pos)
                if end_pos > start_pos:
                    end_token = metadata.char_to_token(batch_idx, end_pos - 1)
                else:
                    end_token = start_token

                if start_token is not None and end_token is not None:
                    # All tokens in this range should be selected
                    for tok_idx in range(start_token, end_token + 1):
                        token_set.add(tok_idx)
                elif start_token is not None:
                    token_set.add(start_token)
                elif end_token is not None:
                    token_set.add(end_token)

            # Set mask for all found tokens at once
            if token_set:
                token_indices = torch.tensor(list(token_set), device=mask.device)
                mask[batch_idx, token_indices] = True

        # Apply attention mask
        mask = mask & metadata.attention_mask.bool()
        return mask

    def __repr__(self) -> str:
        return f"RegexMask(pattern='{self.pattern}')"


# Convenience functions
def contains(text: str, case_sensitive: bool = False) -> ContainsMask:
    """Create a mask for messages containing specific text."""
    return ContainsMask(text, case_sensitive)


def regex(pattern: str, flags: int = 0) -> RegexMask:
    """Create a mask for messages matching a regex pattern."""
    return RegexMask(pattern, flags)

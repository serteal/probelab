"""Basic mask functions."""

from typing import Sequence

import torch
from torch import Tensor

from ..types import Dialogue
from .base import MaskFunction, TokenMetadata


class AllMask(MaskFunction):
    """Mask that selects all tokens."""

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select all non-padding tokens."""
        return metadata.attention_mask.bool()

    def __repr__(self) -> str:
        return "AllMask()"


class NoneMask(MaskFunction):
    """Mask that selects no tokens."""

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select no tokens."""
        return torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

    def __repr__(self) -> str:
        return "NoneMask()"


class LastNTokensMask(MaskFunction):
    """Mask that selects the last n tokens of each message (including special tokens)."""

    def __init__(self, n: int = 1):
        """
        Args:
            n: Number of last tokens to select per message
        """
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        self.n = n

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select the last n tokens of each message."""
        batch_size, seq_len = metadata.attention_mask.shape

        # Initialize result mask
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        # Use the padded role_ids to include special tokens
        role_ids = metadata.role_ids

        # For each batch
        for b in range(batch_size):
            # Find role transitions to identify message boundaries
            # A message ends where the role changes or sequence ends
            valid_mask = metadata.attention_mask[b].bool()

            if not valid_mask.any():
                continue

            # Find where role changes occur (message boundaries)
            role_changes = torch.zeros(
                seq_len, dtype=torch.bool, device=role_ids.device
            )

            # Skip tokens with role_id == -1 (unassigned), they're not part of messages
            has_role = role_ids[b] != -1

            # Compare each position with the next
            for i in range(seq_len - 1):
                if valid_mask[i] and has_role[i]:
                    # Check if next position has different role or is padding or has no role
                    if (
                        not valid_mask[i + 1]
                        or not has_role[i + 1]
                        or role_ids[b, i] != role_ids[b, i + 1]
                    ):
                        role_changes[i] = True

            # Last valid position with a role is also a boundary
            valid_with_role = valid_mask & has_role
            last_valid = torch.where(valid_with_role)[0]
            if len(last_valid) > 0:
                role_changes[last_valid[-1]] = True

            # Find message segments (only those with assigned roles)
            boundaries = torch.where(role_changes)[0]

            # Mark last n tokens of each segment
            for i, end_idx in enumerate(boundaries):
                # Find the start of this segment
                if i == 0:
                    # First segment starts at first token with a role
                    start_candidates = torch.where(has_role[: end_idx + 1])[0]
                    if len(start_candidates) > 0:
                        start_idx = start_candidates[0].item()
                    else:
                        continue
                else:
                    start_idx = boundaries[i - 1].item() + 1

                # This segment is from start_idx to end_idx (inclusive)
                segment_length = end_idx - start_idx + 1

                # Select last n tokens (or all if segment is shorter than n)
                tokens_to_select = min(self.n, segment_length)
                for j in range(tokens_to_select):
                    mask[b, end_idx - j] = True

        # Apply attention mask to exclude padding
        mask = mask & metadata.attention_mask.bool()

        return mask

    def __repr__(self) -> str:
        return f"LastNTokensMask(n={self.n})"


class FirstNTokensMask(MaskFunction):
    """Mask that selects the first n tokens of each message (including special tokens)."""

    def __init__(self, n: int = 1):
        """
        Args:
            n: Number of first tokens to select per message
        """
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        self.n = n

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select the first n tokens of each message."""
        batch_size, seq_len = metadata.attention_mask.shape

        # Initialize result mask
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        # Use the padded role_ids to include special tokens
        role_ids = metadata.role_ids

        # For each batch
        for b in range(batch_size):
            # Find role transitions to identify message boundaries
            valid_mask = metadata.attention_mask[b].bool()

            if not valid_mask.any():
                continue

            # Skip tokens with role_id == -1 (unassigned), they're not part of messages
            has_role = role_ids[b] != -1

            # Find where role changes occur (message boundaries)
            role_changes = torch.zeros(
                seq_len, dtype=torch.bool, device=role_ids.device
            )

            # First valid position with a role is a boundary (start of first message)
            valid_with_role = valid_mask & has_role
            first_valid = torch.where(valid_with_role)[0]
            if len(first_valid) > 0:
                role_changes[first_valid[0]] = True

            # Compare each position with the previous
            for i in range(1, seq_len):
                if valid_mask[i] and has_role[i]:
                    # Check if previous position has different role or is padding or has no role
                    if (
                        not valid_mask[i - 1]
                        or not has_role[i - 1]
                        or role_ids[b, i] != role_ids[b, i - 1]
                    ):
                        role_changes[i] = True

            # Find message segments (only those with assigned roles)
            boundaries = torch.where(role_changes)[0]

            # Mark first n tokens of each segment
            for i, start_idx in enumerate(boundaries):
                # Find the end of this segment
                if i + 1 < len(boundaries):
                    end_idx = boundaries[i + 1] - 1
                else:
                    # Last segment goes to the last valid token with a role
                    last_valid = torch.where(valid_with_role)[0]
                    end_idx = last_valid[-1] if len(last_valid) > 0 else start_idx

                # This segment is from start_idx to end_idx (inclusive)
                segment_length = end_idx - start_idx + 1

                # Select first n tokens (or all if segment is shorter than n)
                tokens_to_select = min(self.n, segment_length)
                for j in range(tokens_to_select):
                    mask[b, start_idx + j] = True

        # Apply attention mask to exclude padding
        mask = mask & metadata.attention_mask.bool()

        return mask

    def __repr__(self) -> str:
        return f"FirstNTokensMask(n={self.n})"


# Convenience functions
def all() -> AllMask:
    """Create a mask that selects all tokens."""
    return AllMask()


def none() -> NoneMask:
    """Create a mask that selects no tokens."""
    return NoneMask()


def last_token() -> LastNTokensMask:
    """Create a mask that selects the last token of each message."""
    return LastNTokensMask(1)


def last_n_tokens(n: int) -> LastNTokensMask:
    """Create a mask that selects the last n tokens of each message."""
    return LastNTokensMask(n)


def first_n_tokens(n: int) -> FirstNTokensMask:
    """Create a mask that selects the first n tokens of each message."""
    return FirstNTokensMask(n)

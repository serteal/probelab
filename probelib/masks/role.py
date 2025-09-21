"""Role-based mask functions."""

from typing import Sequence

from torch import Tensor

from ..types import Dialogue
from .base import MaskFunction, TokenMetadata


class RoleMask(MaskFunction):
    """Mask for selecting tokens from messages with specific roles."""

    ROLE_TO_ID = {"system": 0, "user": 1, "assistant": 2}

    def __init__(self, role: str, include_padding: bool = True):
        """
        Args:
            role: Role to select ("system", "user", or "assistant")
            include_padding: Whether to include special tokens around messages (default: True)
        """
        if role not in self.ROLE_TO_ID:
            raise ValueError(
                f"Invalid role: {role}. Must be one of {list(self.ROLE_TO_ID.keys())}"
            )
        self.role = role
        self.role_id = self.ROLE_TO_ID[role]
        self.include_padding = include_padding

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select tokens from messages with the specified role."""
        # Choose which role_ids to use based on include_padding flag
        if self.include_padding:
            # Use the default role_ids which includes padding
            role_ids = metadata.role_ids
        else:
            # Use the non-padded version if available, otherwise fall back to padded
            if metadata.role_ids_no_padding is not None:
                role_ids = metadata.role_ids_no_padding
            else:
                # Fallback to padded version if no_padding version not available
                role_ids = metadata.role_ids

        # Vectorized comparison - very fast
        mask = role_ids == self.role_id
        # Apply attention mask to exclude padding
        mask = mask & metadata.attention_mask.bool()
        return mask

    def __repr__(self) -> str:
        return f"RoleMask(role='{self.role}', include_padding={self.include_padding})"


# Convenience functions
def role(role_name: str, include_padding: bool = True) -> RoleMask:
    """Create a mask for a specific role."""
    return RoleMask(role_name, include_padding=include_padding)


def assistant(include_padding: bool = True) -> RoleMask:
    """Create a mask for assistant messages."""
    return RoleMask("assistant", include_padding=include_padding)


def user(include_padding: bool = True) -> RoleMask:
    """Create a mask for user messages."""
    return RoleMask("user", include_padding=include_padding)


def system(include_padding: bool = True) -> RoleMask:
    """Create a mask for system messages."""
    return RoleMask("system", include_padding=include_padding)

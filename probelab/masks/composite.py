"""Composite mask functions for combining masks."""

from typing import Sequence

from torch import Tensor

from ..types import Dialogue
from .base import MaskFunction, TokenMetadata


class AndMask(MaskFunction):
    """Logical AND of two masks."""

    def __init__(self, mask1: MaskFunction, mask2: MaskFunction):
        self.mask1 = mask1
        self.mask2 = mask2

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Return intersection of both masks."""
        mask1_result = self.mask1.evaluate(dialogues, metadata)
        mask2_result = self.mask2.evaluate(dialogues, metadata)
        return mask1_result & mask2_result

    def __repr__(self) -> str:
        return f"({self.mask1} & {self.mask2})"


class OrMask(MaskFunction):
    """Logical OR of two masks."""

    def __init__(self, mask1: MaskFunction, mask2: MaskFunction):
        self.mask1 = mask1
        self.mask2 = mask2

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Return union of both masks."""
        mask1_result = self.mask1.evaluate(dialogues, metadata)
        mask2_result = self.mask2.evaluate(dialogues, metadata)
        return mask1_result | mask2_result

    def __repr__(self) -> str:
        return f"({self.mask1} | {self.mask2})"


class NotMask(MaskFunction):
    """Logical NOT of a mask with explicit padding control.

    Args:
        mask: Mask function to negate.
        keep_padding_masked: If True (default), padding tokens remain masked
            after negation. If False, padding tokens may be selected if they
            weren't in the original mask. Default: True.

    Examples:
        Default behavior (padding stays masked):

        >>> not_assistant = ~pl.masks.assistant()  # Padding remains masked
        >>> # Equivalent to:
        >>> not_assistant = pl.masks.NotMask(pl.masks.assistant(), keep_padding_masked=True)

        Include padding in negation:

        >>> not_assistant = pl.masks.NotMask(pl.masks.assistant(), keep_padding_masked=False)
    """

    def __init__(self, mask: MaskFunction, *, keep_padding_masked: bool = True):
        """Initialize NotMask with explicit padding behavior.

        Args:
            mask: Mask function to negate.
            keep_padding_masked: If True, padding tokens stay masked even after
                negation. If False, padding may be selected after negation.
        """
        self.mask = mask
        self.keep_padding_masked = keep_padding_masked

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Return negation of mask with explicit padding handling.

        Args:
            dialogues: Original dialogues.
            metadata: Token metadata including attention mask.

        Returns:
            Boolean tensor with negated mask. If keep_padding_masked=True,
            padding tokens are explicitly excluded from the result.
        """
        mask_result = self.mask.evaluate(dialogues, metadata)
        negated = ~mask_result

        # Explicitly handle padding based on configuration
        if self.keep_padding_masked:
            return negated & metadata.attention_mask.bool()
        else:
            return negated

    def __repr__(self) -> str:
        if self.keep_padding_masked:
            return f"(~{self.mask})"
        else:
            return f"(~{self.mask}, keep_padding=False)"

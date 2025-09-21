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
    """Logical NOT of a mask."""

    def __init__(self, mask: MaskFunction):
        self.mask = mask

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Return negation of mask."""
        mask_result = self.mask.evaluate(dialogues, metadata)
        # Only negate within attention mask (don't select padding)
        return (~mask_result) & metadata.attention_mask.bool()

    def __repr__(self) -> str:
        return f"(~{self.mask})"

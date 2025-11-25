"""Content-type mask functions for token selection."""

from collections.abc import Sequence

import torch
from torch import Tensor

from ..types import Dialogue
from .base import MaskFunction, TokenMetadata


class SpecialTokensMask(MaskFunction):
    """Mask that selects special tokens (e.g., <bos>, <eos>, <pad>, etc.)."""

    def __init__(self, special_token_ids: set[int] | None = None):
        """
        Args:
            special_token_ids: Optional set of special token IDs.
                             If None, will be determined from tokenizer.
        """
        self.special_token_ids = special_token_ids
        self._cached_ids: set[int] | None = (
            set(special_token_ids) if special_token_ids is not None else None
        )
        self._initialized = False

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select special tokens."""
        batch_size, seq_len = metadata.attention_mask.shape
        token_ids = metadata.token_ids

        if not self._initialized:
            if self._cached_ids is None:
                if metadata.special_token_ids:
                    self._cached_ids = {
                        int(token_id) for token_id in metadata.special_token_ids
                    }
                else:
                    self._cached_ids = self._infer_special_ids(token_ids, metadata)
            self._initialized = True

        if not self._cached_ids:
            return torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        special_tensor = torch.tensor(
            list(self._cached_ids),
            device=token_ids.device,
            dtype=token_ids.dtype,
        )

        mask = torch.isin(token_ids, special_tensor)

        mask = mask & metadata.attention_mask.bool()
        return mask

    def __repr__(self) -> str:
        if self.special_token_ids:
            return f"SpecialTokensMask(special_token_ids={self.special_token_ids})"
        return "SpecialTokensMask()"

    def _infer_special_ids(
        self, token_ids: Tensor, metadata: TokenMetadata
    ) -> set[int]:
        """Fallback heuristic for models without reported special IDs."""
        inferred: set[int] = {
            0,
            1,
            2,
            3,
            106,
            107,
            128000,
            128001,
            128256,
        }

        if metadata.formatted_texts and len(metadata.formatted_texts) > 0:
            text_sample = metadata.formatted_texts[0][:200]
            if "<" in text_sample and ">" in text_sample:
                for i in [0, 1, 2]:
                    if i < token_ids.shape[1]:
                        inferred.add(int(token_ids[0, i].item()))

                if metadata.role_ids is not None:
                    role_changes = metadata.role_ids[0, :-1] != metadata.role_ids[0, 1:]
                    change_indices = torch.where(role_changes)[0]
                    for idx in change_indices[:5]:
                        if idx < token_ids.shape[1] - 1:
                            inferred.add(int(token_ids[0, idx].item()))
                            inferred.add(int(token_ids[0, idx + 1].item()))

        return inferred


# Convenience function
def special_tokens(special_token_ids: set[int] | None = None) -> SpecialTokensMask:
    """Create a mask for special tokens."""
    return SpecialTokensMask(special_token_ids)

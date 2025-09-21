"""Content-type mask functions for token selection."""

from typing import Sequence, Set, Optional

import torch
from torch import Tensor

from ..types import Dialogue
from .base import MaskFunction, TokenMetadata


class SpecialTokensMask(MaskFunction):
    """Mask that selects special tokens (e.g., <bos>, <eos>, <pad>, etc.)."""

    def __init__(self, special_token_ids: Optional[Set[int]] = None):
        """
        Args:
            special_token_ids: Optional set of special token IDs.
                             If None, will be determined from tokenizer.
        """
        self.special_token_ids = special_token_ids
        self._cached_ids: Optional[Set[int]] = special_token_ids
        self._initialized = False

    def evaluate(
        self,
        dialogues: Sequence[Dialogue],
        metadata: TokenMetadata,
    ) -> Tensor:
        """Select special tokens."""
        batch_size, seq_len = metadata.attention_mask.shape
        token_ids = metadata.token_ids

        # Initialize cached IDs only once
        if not self._initialized:
            if self._cached_ids is None:
                # Use a more efficient approach with known special token IDs
                # Common special tokens across models
                self._cached_ids = {
                    0,  # Padding token
                    1,  # BOS/CLS token
                    2,  # EOS/SEP/BOS token (Gemma)
                    3,  # UNK token
                    # Add high-value tokens that are often special
                    106,
                    107,  # Common for <start_of_turn>, <end_of_turn>
                    128000,
                    128001,
                    128256,  # Llama3 special tokens
                }

                # Look for special tokens in first sequence only (fast heuristic)
                if metadata.formatted_texts and len(metadata.formatted_texts) > 0:
                    # Quick pattern check on first 100 chars
                    text_sample = metadata.formatted_texts[0][:200]

                    # If we see XML-like tokens, add likely candidates
                    if "<" in text_sample and ">" in text_sample:
                        # Add tokens that are likely special based on position
                        # Check tokens at known special positions
                        for i in [0, 1, 2]:  # First few tokens
                            if i < seq_len:
                                self._cached_ids.add(token_ids[0, i].item())

                        # Check tokens around role transitions (often special)
                        if metadata.role_ids is not None:
                            role_changes = (
                                metadata.role_ids[0, :-1] != metadata.role_ids[0, 1:]
                            )
                            change_indices = torch.where(role_changes)[0]
                            for idx in change_indices[:5]:  # First few transitions
                                if idx < seq_len - 1:
                                    self._cached_ids.add(token_ids[0, idx].item())
                                    self._cached_ids.add(token_ids[0, idx + 1].item())

            self._initialized = True

        # Create mask for special tokens
        mask = torch.zeros_like(metadata.attention_mask, dtype=torch.bool)

        # Vectorized comparison for efficiency
        for special_id in self._cached_ids:
            mask |= token_ids == special_id

        # Apply attention mask
        mask = mask & metadata.attention_mask.bool()
        return mask

    def __repr__(self) -> str:
        if self.special_token_ids:
            return f"SpecialTokensMask(special_token_ids={self.special_token_ids})"
        return "SpecialTokensMask()"


# Convenience function
def special_tokens(special_token_ids: Optional[Set[int]] = None) -> SpecialTokensMask:
    """Create a mask for special tokens."""
    return SpecialTokensMask(special_token_ids)

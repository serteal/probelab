"""Post-probe transformers that operate on score tensors."""

from typing import Literal

import torch

from .base import PostTransformer


class AggregateTokenScores(PostTransformer):
    """Aggregate token-level scores to sequence-level.

    This transformer converts token-level predictions [n_tokens, 2]
    to sequence-level predictions [batch, 2] by aggregating scores
    for each sequence.

    Requires tokens_per_sample information to know how to split
    the token scores back into sequences.

    Args:
        method: Aggregation method:
            - "mean": Average token scores per sequence
            - "max": Maximum token score per sequence
            - "last_token": Use last token score per sequence

    Example:
        >>> # After probe returns [n_tokens, 2]
        >>> transform = AggregateTokenScores("mean")
        >>> transform.tokens_per_sample = tokens_per_sample  # Set during fit
        >>> sequence_scores = transform.transform(token_scores)  # [batch, 2]
    """

    def __init__(self, method: Literal["mean", "max", "last_token"] = "mean"):
        if method not in {"mean", "max", "last_token"}:
            raise ValueError(
                f"Invalid aggregation method: {method}. "
                f"Must be one of: 'mean', 'max', 'last_token'"
            )
        self.method = method
        self.tokens_per_sample = None

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Aggregate token scores to sequence scores.

        Args:
            X: Token-level scores [n_tokens, 2]

        Returns:
            Sequence-level scores [batch, 2]

        Raises:
            ValueError: If tokens_per_sample not set
        """
        if self.tokens_per_sample is None:
            raise ValueError(
                "tokens_per_sample must be set before transform. "
                "This is typically done automatically by the pipeline."
            )

        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(
                f"Expected token scores with shape [n_tokens, 2], got {X.shape}"
            )

        split_sizes = self.tokens_per_sample.tolist()
        aggregated = []
        offset = 0

        for size in split_sizes:
            if size == 0:
                # No tokens for this sequence, return zeros
                zero_scores = torch.zeros(2, device=X.device, dtype=X.dtype)
                aggregated.append(zero_scores)
                continue

            # Extract scores for this sequence
            token_scores = X[offset : offset + size, :]  # [size, 2]

            # Aggregate
            if self.method == "mean":
                seq_score = token_scores.mean(dim=0)  # [2]
            elif self.method == "max":
                seq_score = token_scores.max(dim=0).values  # [2]
            elif self.method == "last_token":
                seq_score = token_scores[-1]  # [2]

            aggregated.append(seq_score)
            offset += size

        # Stack to get [batch, 2]
        return torch.stack(aggregated, dim=0)

    def __repr__(self) -> str:
        return f"AggregateTokenScores(method='{self.method}')"

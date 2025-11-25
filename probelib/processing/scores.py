"""Axis-aware container for prediction scores."""

from enum import IntEnum, auto
from typing import Literal

import torch

from ..types import AggregationMethod


class ScoreAxis(IntEnum):
    """Axes for score tensors."""

    BATCH = auto()  # Batch/sample dimension
    SEQ = auto()  # Sequence/token dimension (for token-level scores)
    CLASS = auto()  # Class probabilities (always size 2 for binary)


class Scores:
    """Axis-aware container for prediction scores.

    Tracks which dimensions are present in the score tensor:
    - Sequence-level scores: axes=(BATCH, CLASS) - shape [batch, 2]
    - Token-level scores: axes=(BATCH, SEQ, CLASS) - shape [batch, seq, 2]

    Provides pool() method for aggregation with automatic axis tracking.

    Args:
        scores: The score tensor
        axes: Tuple of ScoreAxis indicating which dimensions are present
        tokens_per_sample: Number of tokens per sample (for variable-length sequences)
        batch_indices: Original batch indices for tracking

    Example:
        >>> # Sequence-level scores
        >>> scores = Scores.from_sequence_scores(tensor)  # [batch, 2]

        >>> # Token-level scores with pooling
        >>> scores = Scores.from_token_scores(tensor, tokens_per_sample)
        >>> pooled = scores.pool(axis="sequence", method="mean")  # [batch, 2]
    """

    def __init__(
        self,
        scores: torch.Tensor,
        axes: tuple[ScoreAxis, ...],
        tokens_per_sample: torch.Tensor | None = None,
        batch_indices: list[int] | None = None,
    ):
        self.scores = scores
        self.axes = axes
        self.tokens_per_sample = tokens_per_sample
        self.batch_indices = batch_indices

        # Validate axes match tensor shape
        if len(axes) != scores.ndim:
            raise ValueError(
                f"Number of axes ({len(axes)}) must match tensor dimensions ({scores.ndim})"
            )

        # CLASS axis must be last
        if axes[-1] != ScoreAxis.CLASS:
            raise ValueError("CLASS axis must be the last axis")

    @classmethod
    def from_sequence_scores(
        cls,
        scores: torch.Tensor,
        batch_indices: list[int] | None = None,
    ) -> "Scores":
        """Create from sequence-level scores.

        Args:
            scores: Tensor of shape [batch, 2]
            batch_indices: Original batch indices

        Returns:
            Scores with axes (BATCH, CLASS)
        """
        if scores.ndim != 2:
            raise ValueError(f"Expected 2D tensor [batch, 2], got shape {scores.shape}")
        return cls(
            scores=scores,
            axes=(ScoreAxis.BATCH, ScoreAxis.CLASS),
            batch_indices=batch_indices,
        )

    @classmethod
    def from_token_scores(
        cls,
        scores: torch.Tensor,
        tokens_per_sample: torch.Tensor,
        batch_indices: list[int] | None = None,
    ) -> "Scores":
        """Create from token-level scores.

        Handles both padded [batch, seq, 2] and flattened [n_tokens, 2] formats.

        Args:
            scores: Tensor of shape [batch, seq, 2] or [n_tokens, 2]
            tokens_per_sample: Number of valid tokens per sample [batch]
            batch_indices: Original batch indices

        Returns:
            Scores with axes (BATCH, SEQ, CLASS)
        """
        if scores.ndim == 3:
            # Already in [batch, seq, 2] format
            return cls(
                scores=scores,
                axes=(ScoreAxis.BATCH, ScoreAxis.SEQ, ScoreAxis.CLASS),
                tokens_per_sample=tokens_per_sample,
                batch_indices=batch_indices,
            )
        elif scores.ndim == 2:
            # Flattened [n_tokens, 2] format - reshape to [batch, max_seq, 2]
            batch_size = len(tokens_per_sample)
            max_seq = int(tokens_per_sample.max().item())
            n_classes = scores.shape[-1]

            # Create padded tensor
            padded = torch.zeros(
                batch_size, max_seq, n_classes, device=scores.device, dtype=scores.dtype
            )

            # Fill in the scores
            offset = 0
            for i, n_tokens in enumerate(tokens_per_sample.tolist()):
                if n_tokens > 0:
                    padded[i, :n_tokens] = scores[offset : offset + n_tokens]
                    offset += n_tokens

            return cls(
                scores=padded,
                axes=(ScoreAxis.BATCH, ScoreAxis.SEQ, ScoreAxis.CLASS),
                tokens_per_sample=tokens_per_sample,
                batch_indices=batch_indices,
            )
        else:
            raise ValueError(
                f"Expected 2D [n_tokens, 2] or 3D [batch, seq, 2] tensor, got shape {scores.shape}"
            )

    def has_axis(self, axis: ScoreAxis) -> bool:
        """Check if the given axis is present."""
        return axis in self.axes

    def pool(
        self,
        dim: Literal["sequence"] = "sequence",
        method: AggregationMethod | str = "mean",
    ) -> "Scores":
        """Pool over the specified dimension.

        Args:
            dim: Dimension to pool over ("sequence" for SEQ dimension)
            method: Pooling method - "mean", "max", or "last_token"

        Returns:
            New Scores with reduced dimensionality

        Raises:
            ValueError: If dimension not present or invalid method
        """
        if dim != "sequence":
            raise ValueError(f"Only 'sequence' dimension pooling is supported, got {dim!r}")

        if not self.has_axis(ScoreAxis.SEQ):
            raise ValueError("Scores don't have SEQ axis to pool over")

        # Normalize method to enum
        if isinstance(method, str):
            try:
                method = AggregationMethod(method)
            except ValueError:
                raise ValueError(
                    f"Invalid method: {method}. Must be one of: {[m.value for m in AggregationMethod]}"
                )

        # Get axis index for SEQ
        seq_idx = self.axes.index(ScoreAxis.SEQ)

        # Pool based on method
        if self.tokens_per_sample is not None:
            # Variable-length sequences - use tokens_per_sample
            pooled = self._pool_variable_length(method)
        else:
            # Fixed-length sequences - pool entire dimension
            pooled = self._pool_fixed_length(method, seq_idx)

        # Remove SEQ from axes
        new_axes = tuple(a for a in self.axes if a != ScoreAxis.SEQ)

        return Scores(
            scores=pooled,
            axes=new_axes,
            batch_indices=self.batch_indices,
        )

    def _pool_variable_length(self, method: AggregationMethod) -> torch.Tensor:
        """Pool with variable-length sequences using tokens_per_sample.

        Uses vectorized operations for efficiency instead of Python loops.
        """
        with torch.no_grad():
            batch_size = self.scores.shape[0]
            seq_len = self.scores.shape[1]
            n_classes = self.scores.shape[-1]

            # Create mask from tokens_per_sample: [batch, seq]
            seq_indices = torch.arange(seq_len, device=self.scores.device)
            tokens_per_sample = self.tokens_per_sample.to(self.scores.device)
            mask = seq_indices.unsqueeze(0) < tokens_per_sample.unsqueeze(1)

            if method == AggregationMethod.MEAN:
                # Masked mean: sum valid tokens / count
                mask_expanded = mask.unsqueeze(-1).to(self.scores.dtype)  # [batch, seq, 1]
                masked_scores = self.scores * mask_expanded
                counts = tokens_per_sample.clamp(min=1).unsqueeze(-1).to(self.scores.dtype)  # [batch, 1]
                pooled = masked_scores.sum(dim=1) / counts  # [batch, n_classes]

            elif method == AggregationMethod.MAX:
                # Masked max: fill invalid with -inf, take max
                mask_expanded = mask.unsqueeze(-1)  # [batch, seq, 1]
                masked_scores = self.scores.masked_fill(~mask_expanded, float("-inf"))
                pooled = masked_scores.max(dim=1).values  # [batch, n_classes]
                # Handle empty sequences (all -inf -> 0)
                empty_mask = tokens_per_sample == 0
                if empty_mask.any():
                    pooled[empty_mask] = 0.0

            elif method == AggregationMethod.LAST_TOKEN:
                # Gather last valid token for each sample
                last_indices = (tokens_per_sample - 1).clamp(min=0).long()  # [batch]
                # Expand for gather: [batch, 1, n_classes]
                gather_idx = last_indices.view(batch_size, 1, 1).expand(batch_size, 1, n_classes)
                pooled = self.scores.gather(dim=1, index=gather_idx).squeeze(1)  # [batch, n_classes]
                # Handle empty sequences
                empty_mask = tokens_per_sample == 0
                if empty_mask.any():
                    pooled[empty_mask] = 0.0

            else:
                raise ValueError(f"Unknown pooling method: {method}")

            return pooled

    def _pool_fixed_length(self, method: AggregationMethod, seq_idx: int) -> torch.Tensor:
        """Pool fixed-length sequences over the given axis."""
        with torch.no_grad():
            if method == AggregationMethod.MEAN:
                return self.scores.mean(dim=seq_idx)
            elif method == AggregationMethod.MAX:
                return self.scores.max(dim=seq_idx).values
            elif method == AggregationMethod.LAST_TOKEN:
                # Select last position along sequence axis
                return self.scores.select(dim=seq_idx, index=-1)
            else:
                raise ValueError(f"Unknown pooling method: {method}")

    def to(self, device: torch.device | str) -> "Scores":
        """Move scores to device."""
        return Scores(
            scores=self.scores.to(device),
            axes=self.axes,
            tokens_per_sample=(
                self.tokens_per_sample.to(device)
                if self.tokens_per_sample is not None
                else None
            ),
            batch_indices=self.batch_indices,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the score tensor."""
        return tuple(self.scores.shape)

    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""
        if ScoreAxis.BATCH in self.axes:
            return self.scores.shape[self.axes.index(ScoreAxis.BATCH)]
        return 1

    @property
    def device(self) -> torch.device:
        """Device of the score tensor."""
        return self.scores.device

    def __repr__(self) -> str:
        axes_str = ", ".join(a.name for a in self.axes)
        return f"Scores(shape={self.shape}, axes=({axes_str}))"

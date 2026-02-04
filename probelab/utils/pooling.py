"""Shared pooling utilities for activations and scores."""

from __future__ import annotations

import torch

from ..types import AggregationMethod


def masked_pool(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    method: AggregationMethod | str,
    seq_dim: int,
    batch_dim: int = 0,
) -> torch.Tensor:
    """Pool tensor over sequence dimension using only masked positions.

    This is the shared implementation used by Activations, Scores, and
    batch_activations_pooled to avoid code duplication.

    Args:
        tensor: Input tensor of any shape containing a batch and sequence dimension
        mask: Boolean or float mask of shape [batch, seq] where True/1.0 indicates
              valid positions to include in pooling
        method: Pooling method - "mean", "max", or "last_token"
        seq_dim: Index of the sequence dimension in tensor
        batch_dim: Index of the batch dimension in tensor (default: 0)

    Returns:
        Tensor with sequence dimension removed via pooling

    Examples:
        >>> # Pool [batch, seq, hidden] over seq (dim=1)
        >>> pooled = masked_pool(tensor, mask, "mean", seq_dim=1)
        >>> # pooled.shape: [batch, hidden]

        >>> # Pool [layers, batch, seq, hidden] over seq (dim=2)
        >>> pooled = masked_pool(tensor, mask, "mean", seq_dim=2, batch_dim=1)
        >>> # pooled.shape: [layers, batch, hidden]
    """
    if isinstance(method, str):
        try:
            method = AggregationMethod(method)
        except ValueError:
            raise ValueError(
                f"Unknown pooling method: {method}. "
                f"Supported: {[m.value for m in AggregationMethod]}"
            )

    # Ensure mask is on the same device as tensor
    mask = mask.to(tensor.device)
    mask_bool = mask.bool()

    # Build shape for mask expansion to match tensor
    # mask is [batch, seq], we need to expand to tensor's shape
    expand_shape = [1] * tensor.ndim
    expand_shape[batch_dim] = mask.shape[0]
    expand_shape[seq_dim] = mask.shape[1]

    if method == AggregationMethod.MEAN:
        mask_float = mask_bool.to(dtype=tensor.dtype)
        mask_expanded = mask_float.view(expand_shape)

        masked = tensor * mask_expanded
        counts = mask_expanded.sum(dim=seq_dim, keepdim=True).clamp(min=1.0)
        pooled = masked.sum(dim=seq_dim) / counts.squeeze(seq_dim)

    elif method == AggregationMethod.MAX:
        mask_expanded = mask_bool.view(expand_shape)

        masked = tensor.masked_fill(~mask_expanded, float("-inf"))
        pooled = masked.max(dim=seq_dim).values

        # Handle empty sequences (all masked out)
        no_valid = ~mask_bool.any(dim=1)
        if no_valid.any():
            # Build indexer for batch dimension
            indexer = [slice(None)] * pooled.ndim
            indexer[batch_dim] = no_valid
            pooled[tuple(indexer)] = 0.0

    elif method == AggregationMethod.LAST_TOKEN:
        valid_counts = mask_bool.sum(dim=1)
        last_indices = (valid_counts - 1).clamp(min=0).long()

        # Build gather index shape
        # We need to gather along seq_dim, so that dim gets size 1
        # All other dims get their original size
        gather_shape = list(tensor.shape)
        gather_shape[seq_dim] = 1

        # Create index tensor and expand
        # Start with [batch] indices, reshape to put batch in right place
        index_shape = [1] * tensor.ndim
        index_shape[batch_dim] = last_indices.shape[0]
        index_shape[seq_dim] = 1

        gather_idx = last_indices.view(index_shape).expand(gather_shape)
        pooled = tensor.gather(dim=seq_dim, index=gather_idx).squeeze(seq_dim)

        # Handle empty sequences
        no_valid = valid_counts == 0
        if no_valid.any():
            indexer = [slice(None)] * pooled.ndim
            indexer[batch_dim] = no_valid
            pooled[tuple(indexer)] = 0.0

    else:
        raise ValueError(f"Unknown pooling method: {method}")

    return pooled

"""Pooling utilities for tensors and activations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..types import AggregationMethod


def pool(
    x: torch.Tensor,
    mask: torch.Tensor,
    method: str = "mean",
    *,
    dim: int = 1,
    alpha: float = 0.5,
    window_size: int = 10,
) -> torch.Tensor:
    """Pool tensor over a dimension using mask.

    Args:
        x: Input tensor [batch, seq, ...] or any shape with batch and pooling dims
        mask: Boolean mask [batch, seq] where True = valid positions
        method: Pooling method:
            - "mean": Average over valid positions
            - "max": Maximum over valid positions
            - "last_token": Last valid position only
            - "ema": Exponential moving average, then max
            - "rolling": Rolling window mean, then max
        dim: Dimension to pool over (default: 1 for sequence)
        alpha: EMA smoothing factor in (0, 1]. Higher = more weight on recent.
        window_size: Rolling window size (>= 1)

    Returns:
        Tensor with pooled dimension removed

    Examples:
        >>> probs = probe.predict(acts)  # [batch, seq]
        >>> mask = acts.detection_mask.bool()
        >>> pooled = pool(probs, mask, "mean")  # [batch]
        >>> pooled = pool(probs, mask, "ema", alpha=0.3)
        >>> pooled = pool(probs, mask, "rolling", window_size=5)
    """
    return _masked_pool(x, mask, method, seq_dim=dim, batch_dim=0, alpha=alpha, window_size=window_size)


def _masked_pool(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    method: AggregationMethod | str,
    seq_dim: int,
    batch_dim: int = 0,
    alpha: float = 0.5,
    window_size: int = 10,
) -> torch.Tensor:
    """Internal: pool tensor over sequence dimension using only masked positions."""
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
            indexer = [slice(None)] * pooled.ndim
            indexer[batch_dim] = no_valid
            pooled[tuple(indexer)] = 0.0

    elif method == AggregationMethod.LAST_TOKEN:
        valid_counts = mask_bool.sum(dim=1)
        last_indices = (valid_counts - 1).clamp(min=0).long()

        gather_shape = list(tensor.shape)
        gather_shape[seq_dim] = 1

        index_shape = [1] * tensor.ndim
        index_shape[batch_dim] = last_indices.shape[0]
        index_shape[seq_dim] = 1

        gather_idx = last_indices.view(index_shape).expand(gather_shape)
        pooled = tensor.gather(dim=seq_dim, index=gather_idx).squeeze(seq_dim)

        no_valid = valid_counts == 0
        if no_valid.any():
            indexer = [slice(None)] * pooled.ndim
            indexer[batch_dim] = no_valid
            pooled[tuple(indexer)] = 0.0

    elif method == AggregationMethod.EMA:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        mask_float = mask_bool.to(dtype=tensor.dtype)
        mask_expanded = mask_float.view(expand_shape)
        mask_bool_expanded = mask_bool.view(expand_shape)
        seq_len = tensor.shape[seq_dim]

        # Compute EMA along seq_dim
        ema = torch.zeros_like(tensor)

        # Get slice for first position
        idx_0 = [slice(None)] * tensor.ndim
        idx_0[seq_dim] = 0
        mask_0 = [slice(None)] * tensor.ndim
        mask_0[seq_dim] = slice(0, 1)

        ema[tuple(idx_0)] = alpha * tensor[tuple(idx_0)] * mask_expanded[tuple(mask_0)].squeeze(seq_dim)

        for j in range(1, seq_len):
            idx_j = [slice(None)] * tensor.ndim
            idx_j[seq_dim] = j
            idx_prev = [slice(None)] * tensor.ndim
            idx_prev[seq_dim] = j - 1
            mask_j = [slice(None)] * tensor.ndim
            mask_j[seq_dim] = slice(j, j + 1)

            m = mask_expanded[tuple(mask_j)].squeeze(seq_dim)
            ema[tuple(idx_j)] = (alpha * tensor[tuple(idx_j)] + (1 - alpha) * ema[tuple(idx_prev)]) * m
            ema[tuple(idx_j)] += ema[tuple(idx_prev)] * (1 - m)

        # Max over valid positions
        ema_masked = ema.masked_fill(~mask_bool_expanded, float("-inf"))
        pooled = ema_masked.max(dim=seq_dim).values

        no_valid = ~mask_bool.any(dim=1)
        if no_valid.any():
            indexer = [slice(None)] * pooled.ndim
            indexer[batch_dim] = no_valid
            pooled[tuple(indexer)] = 0.0

    elif method == AggregationMethod.ROLLING:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        mask_float = mask_bool.to(dtype=tensor.dtype)
        mask_expanded = mask_float.view(expand_shape)
        seq_len = tensor.shape[seq_dim]
        w = window_size

        # Rolling mean via cumsum along seq_dim
        masked_tensor = tensor * mask_expanded

        # Compute cumsum along seq_dim
        cum_tensor = F.pad(torch.cumsum(masked_tensor, dim=seq_dim), _pad_for_dim(seq_dim, tensor.ndim, w), value=0)
        cum_counts = F.pad(torch.cumsum(mask_expanded, dim=seq_dim), _pad_for_dim(seq_dim, tensor.ndim, w), value=0)

        # Rolling difference
        roll_tensor = _slice_dim(cum_tensor, seq_dim, w, None) - _slice_dim(cum_tensor, seq_dim, None, -w)
        roll_counts = _slice_dim(cum_counts, seq_dim, w, None) - _slice_dim(cum_counts, seq_dim, None, -w)

        rolling_means = roll_tensor / roll_counts.clamp(min=1)

        # Max over valid windows
        valid_windows = roll_counts > 0
        rolling_masked = rolling_means.masked_fill(~valid_windows, float("-inf"))
        pooled = rolling_masked.max(dim=seq_dim).values

        no_valid = ~mask_bool.any(dim=1)
        if no_valid.any():
            indexer = [slice(None)] * pooled.ndim
            indexer[batch_dim] = no_valid
            pooled[tuple(indexer)] = 0.0

    else:
        raise ValueError(f"Unknown pooling method: {method}")

    return pooled


def _pad_for_dim(dim: int, ndim: int, pad_size: int) -> tuple[int, ...]:
    """Create padding tuple for F.pad to pad a specific dimension."""
    pad = [0] * (2 * ndim)
    dim_from_end = ndim - 1 - dim
    pad[2 * dim_from_end] = pad_size
    return tuple(pad)


def _slice_dim(tensor: torch.Tensor, dim: int, start: int | None, end: int | None) -> torch.Tensor:
    """Slice tensor along a specific dimension."""
    idx = [slice(None)] * tensor.ndim
    idx[dim] = slice(start, end)
    return tensor[tuple(idx)]

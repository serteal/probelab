"""Pooling functions for masked tensors.

All functions assume batch dimension is at dim 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mean(x: torch.Tensor, mask: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    """Mean over valid positions.

    Args:
        x: Input tensor with batch at dim 0
        mask: Boolean mask [batch, seq] where True = valid
        dim: Dimension to pool over (sequence dimension)

    Returns:
        Tensor with dim removed
    """
    mask = mask.to(x.device).bool()
    mask_expanded = _expand_mask(mask, x.ndim, dim)
    masked = x * mask_expanded.to(x.dtype)
    counts = mask_expanded.sum(dim=dim, keepdim=True).clamp(min=1)
    return masked.sum(dim=dim) / counts.squeeze(dim).to(x.dtype)


def max(x: torch.Tensor, mask: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    """Max over valid positions.

    Args:
        x: Input tensor with batch at dim 0
        mask: Boolean mask [batch, seq] where True = valid
        dim: Dimension to pool over (sequence dimension)

    Returns:
        Tensor with dim removed
    """
    mask = mask.to(x.device).bool()
    mask_expanded = _expand_mask(mask, x.ndim, dim)
    masked = x.masked_fill(~mask_expanded, float("-inf"))
    result = masked.max(dim=dim).values

    # Handle empty sequences
    no_valid = ~mask.any(dim=1)
    if no_valid.any():
        # Batch dim shifts by 1 if we removed a dim before it
        batch_dim_out = 0 if dim > 0 else 0
        indexer = [slice(None)] * result.ndim
        indexer[batch_dim_out] = no_valid
        result[tuple(indexer)] = 0.0
    return result


def last_token(x: torch.Tensor, mask: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    """Last valid position.

    Args:
        x: Input tensor with batch at dim 0
        mask: Boolean mask [batch, seq] where True = valid
        dim: Dimension to pool over (sequence dimension)

    Returns:
        Tensor with dim removed
    """
    mask = mask.to(x.device).bool()
    valid_counts = mask.sum(dim=1)
    last_idx = (valid_counts - 1).clamp(min=0).long()

    # Build gather index
    gather_shape = list(x.shape)
    gather_shape[dim] = 1
    index_shape = [1] * x.ndim
    index_shape[0] = last_idx.shape[0]  # batch dim
    index_shape[dim] = 1

    gather_idx = last_idx.view(index_shape).expand(gather_shape)
    result = x.gather(dim=dim, index=gather_idx).squeeze(dim)

    # Handle empty sequences
    no_valid = valid_counts == 0
    if no_valid.any():
        indexer = [slice(None)] * result.ndim
        indexer[0] = no_valid  # batch dim
        result[tuple(indexer)] = 0.0
    return result


def ema(x: torch.Tensor, mask: torch.Tensor, *, dim: int = 1, alpha: float = 0.5) -> torch.Tensor:
    """Exponential moving average, then max over valid positions.

    Args:
        x: Input tensor with batch at dim 0
        mask: Boolean mask [batch, seq] where True = valid
        dim: Dimension to pool over (sequence dimension)
        alpha: Smoothing factor in (0, 1]. Higher = more weight on recent.

    Returns:
        Tensor with dim removed
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    mask = mask.to(x.device).bool()
    mask_expanded = _expand_mask(mask, x.ndim, dim)
    mask_float = mask_expanded.to(x.dtype)
    seq_len = x.shape[dim]

    # Compute EMA along dim
    ema_val = torch.zeros_like(x)

    idx_0 = [slice(None)] * x.ndim
    idx_0[dim] = 0
    mask_0 = [slice(None)] * x.ndim
    mask_0[dim] = slice(0, 1)

    ema_val[tuple(idx_0)] = alpha * x[tuple(idx_0)] * mask_float[tuple(mask_0)].squeeze(dim)

    for j in range(1, seq_len):
        idx_j = [slice(None)] * x.ndim
        idx_j[dim] = j
        idx_prev = [slice(None)] * x.ndim
        idx_prev[dim] = j - 1
        mask_j = [slice(None)] * x.ndim
        mask_j[dim] = slice(j, j + 1)

        m = mask_float[tuple(mask_j)].squeeze(dim)
        ema_val[tuple(idx_j)] = (alpha * x[tuple(idx_j)] + (1 - alpha) * ema_val[tuple(idx_prev)]) * m
        ema_val[tuple(idx_j)] += ema_val[tuple(idx_prev)] * (1 - m)

    # Max over valid positions
    ema_masked = ema_val.masked_fill(~mask_expanded, float("-inf"))
    result = ema_masked.max(dim=dim).values

    no_valid = ~mask.any(dim=1)
    if no_valid.any():
        indexer = [slice(None)] * result.ndim
        indexer[0] = no_valid  # batch dim
        result[tuple(indexer)] = 0.0
    return result


def rolling(x: torch.Tensor, mask: torch.Tensor, *, dim: int = 1, window_size: int = 10) -> torch.Tensor:
    """Rolling window mean, then max over valid windows.

    Args:
        x: Input tensor with batch at dim 0
        mask: Boolean mask [batch, seq] where True = valid
        dim: Dimension to pool over (sequence dimension)
        window_size: Size of rolling window (>= 1)

    Returns:
        Tensor with dim removed
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    mask = mask.to(x.device).bool()
    mask_expanded = _expand_mask(mask, x.ndim, dim)
    mask_float = mask_expanded.to(x.dtype)
    w = window_size

    # Rolling mean via cumsum
    masked_x = x * mask_float
    cum_x = F.pad(torch.cumsum(masked_x, dim=dim), _pad_for_dim(dim, x.ndim, w), value=0)
    cum_counts = F.pad(torch.cumsum(mask_float, dim=dim), _pad_for_dim(dim, x.ndim, w), value=0)

    # Rolling difference
    roll_x = _slice_dim(cum_x, dim, w, None) - _slice_dim(cum_x, dim, None, -w)
    roll_counts = _slice_dim(cum_counts, dim, w, None) - _slice_dim(cum_counts, dim, None, -w)

    rolling_means = roll_x / roll_counts.clamp(min=1)

    # Max over valid windows
    valid_windows = roll_counts > 0
    rolling_masked = rolling_means.masked_fill(~valid_windows, float("-inf"))
    result = rolling_masked.max(dim=dim).values

    no_valid = ~mask.any(dim=1)
    if no_valid.any():
        indexer = [slice(None)] * result.ndim
        indexer[0] = no_valid  # batch dim
        result[tuple(indexer)] = 0.0
    return result


# --- Internal helpers ---


def _expand_mask(mask: torch.Tensor, ndim: int, dim: int) -> torch.Tensor:
    """Expand [batch, seq] mask to match tensor shape. Batch is always at dim 0."""
    shape = [1] * ndim
    shape[0] = mask.shape[0]  # batch
    shape[dim] = mask.shape[1]  # seq
    return mask.view(shape)


def _pad_for_dim(dim: int, ndim: int, pad_size: int) -> tuple[int, ...]:
    """Create padding tuple for F.pad."""
    pad = [0] * (2 * ndim)
    dim_from_end = ndim - 1 - dim
    pad[2 * dim_from_end] = pad_size
    return tuple(pad)


def _slice_dim(tensor: torch.Tensor, dim: int, start: int | None, end: int | None) -> torch.Tensor:
    """Slice tensor along dimension."""
    idx = [slice(None)] * tensor.ndim
    idx[dim] = slice(start, end)
    return tensor[tuple(idx)]

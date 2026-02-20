"""Pooling functions for masked tensors.

All functions assume batch dimension is at dim 0.

When *offsets* is given, ``x`` is a flat ``[total_tokens, ...]`` tensor and
``mask`` is a ``[total_tokens]`` bool detection mask.  The flat path avoids
materializing padded rectangular tensors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int = 1,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean over valid positions.

    Args:
        x: Input tensor with batch at dim 0, or flat [T, ...] when offsets given
        mask: Boolean mask [batch, seq] or [T] bool when offsets given
        dim: Dimension to pool over (ignored when offsets given)
        offsets: [batch+1] int64 cumulative token counts for flat path

    Returns:
        Tensor with dim removed
    """
    if offsets is not None:
        return _flat_mean(x, mask, offsets)

    mask = mask.to(x.device).bool()
    mask_expanded = _expand_mask(mask, x.ndim, dim)
    masked = x * mask_expanded.to(x.dtype)
    counts = mask_expanded.sum(dim=dim, keepdim=True).clamp(min=1)
    return masked.sum(dim=dim) / counts.squeeze(dim).to(x.dtype)


def max(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int = 1,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Max over valid positions.

    Args:
        x: Input tensor with batch at dim 0, or flat [T, ...] when offsets given
        mask: Boolean mask [batch, seq] or [T] bool when offsets given
        dim: Dimension to pool over (ignored when offsets given)
        offsets: [batch+1] int64 cumulative token counts for flat path

    Returns:
        Tensor with dim removed
    """
    if offsets is not None:
        return _flat_max(x, mask, offsets)

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


def last_token(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int = 1,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Last valid position.

    Args:
        x: Input tensor with batch at dim 0, or flat [T, ...] when offsets given
        mask: Boolean mask [batch, seq] or [T] bool when offsets given
        dim: Dimension to pool over (ignored when offsets given)
        offsets: [batch+1] int64 cumulative token counts for flat path

    Returns:
        Tensor with dim removed
    """
    if offsets is not None:
        return _flat_last_token(x, mask, offsets)

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


def ema(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int = 1,
    alpha: float = 0.5,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exponential moving average, then max over valid positions.

    Uses a parallel prefix scan (Hillis-Steele) to solve the linear recurrence
    s[j] = (1-alpha*m[j])*s[j-1] + alpha*x[j]*m[j] in O(n log n) vectorized
    steps. Correctly carries forward through masked positions.

    Args:
        x: Input tensor with batch at dim 0
        mask: Boolean mask [batch, seq] where True = valid
        dim: Dimension to pool over (sequence dimension)
        alpha: Smoothing factor in (0, 1]. Higher = more weight on recent.
        offsets: [batch+1] int64 cumulative token counts for flat path

    Returns:
        Tensor with dim removed
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    if offsets is not None:
        padded_x, padded_mask = _pad_flat_for_pool(x, mask, offsets)
        return ema(padded_x, padded_mask, dim=1, alpha=alpha)

    mask = mask.to(x.device).bool()
    mask_expanded = _expand_mask(mask, x.ndim, dim)
    mask_float = mask_expanded.to(x.dtype)

    if alpha >= 1.0:
        # No memory: s[j] = x[j] when valid, carry-forward otherwise.
        # Max over valid positions is equivalent to max pooling.
        return max(x, mask, dim=dim)

    # Linear recurrence: s[j] = a[j]*s[j-1] + b[j]
    # where a[j] = 1 - alpha*m[j] (carry-forward when masked),
    #       b[j] = alpha*x[j]*m[j]
    a = 1 - alpha * mask_float
    b = alpha * x * mask_float
    ema_val = _scan_linear_recurrence(a, b, dim)

    # Max over valid positions
    ema_masked = ema_val.masked_fill(~mask_expanded, float("-inf"))
    result = ema_masked.max(dim=dim).values

    no_valid = ~mask.any(dim=1)
    if no_valid.any():
        indexer = [slice(None)] * result.ndim
        indexer[0] = no_valid  # batch dim
        result[tuple(indexer)] = 0.0
    return result


def rolling(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int = 1,
    window_size: int = 10,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rolling window mean, then max over valid windows.

    Args:
        x: Input tensor with batch at dim 0
        mask: Boolean mask [batch, seq] where True = valid
        dim: Dimension to pool over (sequence dimension)
        window_size: Size of rolling window (>= 1)
        offsets: [batch+1] int64 cumulative token counts for flat path

    Returns:
        Tensor with dim removed
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    if offsets is not None:
        padded_x, padded_mask = _pad_flat_for_pool(x, mask, offsets)
        return rolling(padded_x, padded_mask, dim=1, window_size=window_size)

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


# --- Flat+offsets pool implementations ---


def _segment_ids(offsets: torch.Tensor, total: int) -> torch.Tensor:
    """Build [total] int64 segment IDs from offsets. Token j belongs to segment i."""
    batch = offsets.shape[0] - 1
    lengths = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(batch, device=offsets.device), lengths
    )


def _flat_mean(
    data: torch.Tensor, det: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Mean pool over flat [T, *] data using offsets and det mask.

    Uses scatter_add for vectorized segment reduction (no Python for-loop).
    """
    batch = offsets.shape[0] - 1
    det_bool = det.to(data.device).bool()
    total = data.shape[0]
    extra_dims = data.shape[1:]

    seg_ids = _segment_ids(offsets, total)
    # Zero out non-detected tokens
    masked_data = data.clone()
    masked_data[~det_bool] = 0

    out = data.new_zeros(batch, *extra_dims)
    # scatter_add needs matching shapes: expand seg_ids to [T, *extra_dims]
    idx = seg_ids.view(-1, *([1] * len(extra_dims))).expand_as(masked_data)
    out.scatter_add_(0, idx, masked_data.to(out.dtype))

    # Count valid tokens per segment
    counts = data.new_zeros(batch)
    counts.scatter_add_(0, seg_ids, det_bool.to(counts.dtype))
    counts.clamp_(min=1)
    return out / counts.view(-1, *([1] * len(extra_dims)))


def _flat_max(
    data: torch.Tensor, det: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Max pool over flat [T, *] data using offsets and det mask.

    Uses scatter_reduce for vectorized segment reduction.
    """
    batch = offsets.shape[0] - 1
    det_bool = det.to(data.device).bool()
    total = data.shape[0]
    extra_dims = data.shape[1:]

    seg_ids = _segment_ids(offsets, total)
    # Mask out non-detected tokens with -inf
    masked_data = data.clone()
    masked_data[~det_bool] = float("-inf")

    out = data.new_full((batch, *extra_dims), float("-inf"))
    idx = seg_ids.view(-1, *([1] * len(extra_dims))).expand_as(masked_data)
    out.scatter_reduce_(0, idx, masked_data.to(out.dtype), reduce="amax")

    # Replace -inf with 0 for segments with no valid tokens
    out[out == float("-inf")] = 0
    return out


def _flat_last_token(
    data: torch.Tensor, det: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Last det=True token per segment in flat [T, *] data."""
    batch = offsets.shape[0] - 1
    det_bool = det.to(data.device).bool()
    total = data.shape[0]
    extra_dims = data.shape[1:]

    seg_ids = _segment_ids(offsets, total)
    # For each segment, we want the LAST valid token.
    # Create ascending indices; scatter_reduce with "amax" on (seg_id, index) pairs
    # gives the last valid index per segment.
    arange = torch.arange(total, device=data.device)
    # Set non-detected positions to -1 so they lose the max
    valid_indices = torch.where(det_bool, arange, torch.tensor(-1, device=data.device))
    last_idx = torch.full((batch,), -1, dtype=torch.long, device=data.device)
    last_idx.scatter_reduce_(0, seg_ids, valid_indices, reduce="amax")

    out = data.new_zeros(batch, *extra_dims)
    valid = last_idx >= 0
    if valid.any():
        out[valid] = data[last_idx[valid]]
    return out


def _pad_flat_for_pool(
    data: torch.Tensor, det: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [batch, max_seq, *] padded tensor + mask from flat representation.

    Used by ema/rolling which need rectangular form for their vectorized ops.
    """
    batch = offsets.shape[0] - 1
    lengths = offsets[1:] - offsets[:-1]
    max_seq = int(lengths.max().item()) if lengths.numel() > 0 else 0
    extra_dims = data.shape[1:]
    det_bool = det.to(data.device).bool()

    padded = data.new_zeros(batch, max_seq, *extra_dims)
    padded_mask = torch.zeros(batch, max_seq, dtype=torch.bool, device=data.device)

    # Vectorized: build flat indices into the padded tensor
    seg_ids = _segment_ids(offsets, data.shape[0])
    local_pos = torch.arange(data.shape[0], device=data.device) - offsets[seg_ids]
    padded[seg_ids, local_pos] = data
    padded_mask[seg_ids, local_pos] = det_bool
    return padded, padded_mask


# --- Internal helpers ---


def _scan_linear_recurrence(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    """Solve s[j] = a[j]*s[j-1] + b[j] (s[-1]=0) via Hillis-Steele parallel scan.

    The combining operator (a2,b2) ⊕ (a1,b1) = (a2*a1, a2*b1+b2) is
    associative, enabling a prefix scan in ceil(log2(n)) vectorized steps.
    """
    a = a.movedim(dim, 1).contiguous()
    b = b.movedim(dim, 1).contiguous()
    # Pre-allocate double buffers to avoid clone() per iteration
    a2 = torch.empty_like(a)
    b2 = torch.empty_like(b)
    d = 1
    while d < a.shape[1]:
        a2[:, :d] = a[:, :d]
        b2[:, :d] = b[:, :d]
        a2[:, d:] = a[:, d:] * a[:, :-d]
        b2[:, d:] = a[:, d:] * b[:, :-d] + b[:, d:]
        a, a2 = a2, a
        b, b2 = b2, b
        d *= 2
    return b.movedim(1, dim)


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

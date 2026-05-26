"""Tensor batching helpers for probe training and inference."""

from __future__ import annotations

from collections.abc import Generator

import torch


def iter_feature_batches(
    features: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    indices: list[int] | torch.Tensor | None = None,
    batch_size: int = 1024,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Generator[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor], None, None]:
    """Yield dense feature batches.

    Args:
        features: Feature tensor ``[N, H]``.
        labels: Optional labels ``[N]``.
        indices: Optional subset/order of rows to iterate.
        batch_size: Maximum rows per batch.
        shuffle: Shuffle row order before batching.
        generator: Optional deterministic generator.

    Yields:
        ``(batch_features, batch_labels_or_none, batch_indices)``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if indices is None:
        order = torch.arange(features.shape[0], device="cpu")
    elif isinstance(indices, torch.Tensor):
        order = indices.detach().cpu().to(dtype=torch.long)
    else:
        order = torch.tensor(indices, device="cpu", dtype=torch.long)
    if shuffle and order.numel() > 1:
        perm = torch.randperm(order.numel(), generator=generator)
        order = order[perm]
    for start in range(0, order.numel(), batch_size):
        batch_idx = order[start : start + batch_size]
        feature_idx = batch_idx.to(features.device)
        if labels is not None:
            batch_labels = labels[batch_idx.to(labels.device)]
        else:
            batch_labels = None
        yield features[feature_idx], batch_labels, batch_idx


def iter_sequence_batch_indices(
    offsets: torch.Tensor,
    *,
    indices: list[int] | torch.Tensor | None = None,
    batch_size: int = 32,
    max_padded_tokens: int | None = None,
    sort_by_length: bool = True,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Generator[list[int], None, None]:
    """Yield sample index batches for flat+offsets sequence data.

    ``max_padded_tokens`` caps ``len(batch) * max_sequence_length_in_batch``.
    A single sequence longer than the budget is still yielded alone.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_padded_tokens is not None and max_padded_tokens <= 0:
        raise ValueError("max_padded_tokens must be positive when provided")

    n = offsets.numel() - 1
    if indices is None:
        order = list(range(n))
    elif isinstance(indices, torch.Tensor):
        order = [int(i) for i in indices.detach().cpu().flatten().tolist()]
    else:
        order = [int(i) for i in indices]

    offsets_cpu = offsets.detach().cpu()
    lengths = offsets_cpu[1:] - offsets_cpu[:-1]
    if sort_by_length:
        order.sort(key=lambda i: -int(lengths[i]))

    batches: list[list[int]] = []
    cursor = 0
    while cursor < len(order):
        start = cursor
        current_max = 0
        while cursor < len(order) and (cursor - start) < batch_size:
            idx = order[cursor]
            seq_len = int(lengths[idx])
            next_max = max(current_max, seq_len)
            next_count = (cursor - start) + 1
            if (
                next_count > 1
                and max_padded_tokens is not None
                and next_max * next_count > max_padded_tokens
            ):
                break
            current_max = next_max
            cursor += 1
        if cursor == start:
            cursor += 1
        batches.append(order[start:cursor])

    if shuffle and len(batches) > 1:
        perm = torch.randperm(len(batches), generator=generator).tolist()
        batches = [batches[i] for i in perm]
    yield from batches


def pad_sequence_batch(
    data: torch.Tensor,
    offsets: torch.Tensor,
    detection_mask: torch.Tensor,
    indices: list[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a padded local sequence batch from flat+offsets tensors."""
    if isinstance(indices, torch.Tensor):
        idx_list = [int(i) for i in indices.detach().cpu().flatten().tolist()]
    else:
        idx_list = [int(i) for i in indices]

    hidden = data.shape[-1]
    offsets_cpu = offsets.detach().cpu()
    local_max = 0
    for i in idx_list:
        local_max = max(local_max, int(offsets_cpu[i + 1] - offsets_cpu[i]))

    sub_batch = len(idx_list)
    if data.ndim == 3:
        n_layers = data.shape[1]
        padded = data.new_zeros(sub_batch, n_layers, local_max, hidden)
        padded_mask = torch.zeros(sub_batch, local_max, dtype=torch.bool, device=data.device)
        for j, i in enumerate(idx_list):
            start, end = int(offsets_cpu[i]), int(offsets_cpu[i + 1])
            length = end - start
            if length:
                padded[j, :, :length] = data[start:end].transpose(0, 1)
                padded_mask[j, :length] = detection_mask[start:end]
        return padded, padded_mask

    padded = data.new_zeros(sub_batch, local_max, hidden)
    padded_mask = torch.zeros(sub_batch, local_max, dtype=torch.bool, device=data.device)
    for j, i in enumerate(idx_list):
        start, end = int(offsets_cpu[i]), int(offsets_cpu[i + 1])
        length = end - start
        if length:
            padded[j, :length] = data[start:end]
            padded_mask[j, :length] = detection_mask[start:end]
    return padded, padded_mask


def iter_sequence_batches(
    data: torch.Tensor,
    offsets: torch.Tensor,
    detection_mask: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    indices: list[int] | torch.Tensor | None = None,
    batch_size: int = 32,
    max_padded_tokens: int | None = None,
    sort_by_length: bool = True,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor], None, None]:
    """Yield padded sequence tensor batches from flat+offsets data."""
    for batch_indices in iter_sequence_batch_indices(
        offsets,
        indices=indices,
        batch_size=batch_size,
        max_padded_tokens=max_padded_tokens,
        sort_by_length=sort_by_length,
        shuffle=shuffle,
        generator=generator,
    ):
        seq, mask = pad_sequence_batch(data, offsets, detection_mask, batch_indices)
        idx = torch.tensor(batch_indices, dtype=torch.long, device=labels.device if labels is not None else data.device)
        batch_labels = labels[idx] if labels is not None else None
        yield seq, mask, batch_labels, idx

"""Transformers backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generator, Iterator

import torch

from ..models import HookedModel
from ..types import HookPoint

if TYPE_CHECKING:
    from ..processing.tokenization import Tokens


def _iter_batches(
    tokens: "Tokens",
    batch_size: int,
    *,
    batch_token_budget: int | None = None,
    sort_by_length: bool = True,
) -> Generator[tuple[dict[str, torch.Tensor], list[int]], None, None]:
    """Yield batches for HF extraction.

    If ``batch_token_budget`` is provided, dynamically packs variable-size batches
    such that padded tokens (``max_seq_in_batch * batch_size``) stay under budget,
    while respecting ``batch_size`` as an upper bound on samples per batch.
    """
    lengths = tokens.lengths
    if sort_by_length:
        order = lengths.argsort(descending=True)
    else:
        order = torch.arange(len(tokens), device=lengths.device)

    if batch_token_budget is None:
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            idx_list = idx.tolist()
            batch = tokens.pad_batch(idx_list, padding_side=tokens.padding_side)
            yield batch, idx_list
        return

    if batch_token_budget <= 0:
        raise ValueError("batch_token_budget must be positive")

    n = len(order)
    i = 0
    while i < n:
        start = i
        cur_max = 0

        # Greedily add samples while respecting sample cap and padded-token budget.
        while i < n and (i - start) < batch_size:
            sample_idx = int(order[i].item())
            seq_len = int(lengths[sample_idx].item())
            new_max = max(cur_max, seq_len)
            new_count = (i - start) + 1
            padded_tokens = new_max * new_count

            # Always allow at least one sample even if it exceeds budget.
            if new_count > 1 and padded_tokens > batch_token_budget:
                break

            cur_max = new_max
            i += 1

        if i == start:
            i += 1

        idx = order[start:i]
        idx_list = idx.tolist()
        batch = tokens.pad_batch(idx_list, padding_side=tokens.padding_side)
        yield batch, idx_list


def _extract_batch(
    hooked: HookedModel,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract [layer, batch, seq, hidden] and masks on model device."""
    batch_gpu = {
        k: v.to(hooked.model.device, non_blocking=True)
        for k, v in batch.items()
    }
    result = hooked.get_activations(batch_gpu)
    dev = result.device
    return (
        result,
        batch_gpu["attention_mask"].to(dev).bool(),
        batch_gpu["detection_mask"].to(dev).bool(),
    )


class TransformersBackend:
    """Backend adapter for `transformers.PreTrainedModel` models."""

    name = "transformers"

    def stream_raw(
        self,
        model_obj: object,
        tokens: "Tokens",
        layers: list[int],
        batch_size: int,
        **kwargs: Any,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        hook_point = kwargs.get("hook_point", HookPoint.POST_BLOCK)
        if isinstance(hook_point, str):
            hook_point = HookPoint(hook_point)

        target_device = kwargs.get(
            "transformers_target_device", kwargs.get("target_device")
        )

        batch_token_budget = kwargs.get(
            "transformers_batch_token_budget",
            kwargs.get("batch_token_budget"),
        )
        if batch_token_budget is not None:
            batch_token_budget = int(batch_token_budget)
        sort_by_length = bool(
            kwargs.get("transformers_sort_by_length", kwargs.get("sort_by_length", True))
        )

        with HookedModel(
            model_obj,  # type: ignore[arg-type]
            layers,
            detach_activations=True,
            hook_point=hook_point,
            target_device=target_device,
        ) as hooked:
            for batch, idx in _iter_batches(
                tokens,
                batch_size,
                batch_token_budget=batch_token_budget,
                sort_by_length=sort_by_length,
            ):
                acts, attn_mask, det_mask = _extract_batch(hooked, batch)
                # acts: [n_layers, batch_chunk, seq, hidden]

                # Transpose to [batch_chunk, n_layers, seq, hidden]
                acts_t = acts.transpose(0, 1)
                b_size = acts_t.shape[0]

                # Vectorized flatten: [B, L, S, H] -> [B, S, L, H], then drop padding.
                acts_bslh = acts_t.transpose(1, 2)
                valid = attn_mask
                flat_data = acts_bslh[valid]  # [T, n_layers, hidden]
                flat_det = det_mask[valid]  # [T]

                lengths = valid.sum(dim=1, dtype=torch.int64)  # [batch_chunk]
                offsets = torch.zeros(b_size + 1, dtype=torch.int64, device=acts.device)
                offsets[1:] = lengths.cumsum(0)

                yield flat_data, flat_det, offsets, idx

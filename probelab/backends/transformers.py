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
) -> Generator[tuple[dict[str, torch.Tensor], list[int]], None, None]:
    """Yield length-sorted batches for efficient HF extraction."""
    order = tokens.lengths.argsort(descending=True)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        batch = tokens.pad_batch(idx.tolist(), padding_side=tokens.padding_side)
        yield batch, idx.tolist()


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
    return (
        result,
        batch_gpu["attention_mask"].bool(),
        batch_gpu["detection_mask"].bool(),
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

        with HookedModel(
            model_obj,  # type: ignore[arg-type]
            layers,
            detach_activations=True,
            hook_point=hook_point,
        ) as hooked:
            for batch, idx in _iter_batches(tokens, batch_size):
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

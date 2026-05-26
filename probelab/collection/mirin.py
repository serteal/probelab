"""mirin-backed activation collection adapter."""

from __future__ import annotations

from collections.abc import Generator, Iterable
import math
from typing import Any

import torch

from .. import pool as P
from ..activations import Activations
from ..tokenization import Tokens
from .types import ActivationChunk

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


def _import_mirin():
    try:
        import mirin
    except ImportError as exc:
        raise ImportError(
            "The mirin collection adapter requires mirin. Install with "
            "`probelab[collection-mirin]` or `probelab[collection]`."
        ) from exc
    return mirin


def _ensure_model(model_obj: object) -> object:
    """Require a ``mirin.Model`` for this adapter."""
    mirin = _import_mirin()
    if isinstance(model_obj, mirin.Model):
        return model_obj
    raise TypeError(f"Expected mirin.Model, got {type(model_obj).__name__}")


def _iter_batch_indices(
    tokens: Tokens,
    *,
    batch_size: int,
    sort_by_length: bool,
    batch_token_budget: int | None,
) -> Generator[list[int], None, None]:
    """Yield sample indices for padded collection batches."""
    order = list(range(len(tokens)))
    if sort_by_length:
        lengths = tokens.lengths.tolist()
        order.sort(key=lambda idx: int(lengths[idx]), reverse=True)

    cursor = 0
    while cursor < len(order):
        start = cursor
        current_max = 0
        while cursor < len(order) and (cursor - start) < batch_size:
            idx = order[cursor]
            seq_len = int(tokens.offsets[idx + 1]) - int(tokens.offsets[idx])
            next_max = max(current_max, seq_len)
            next_count = (cursor - start) + 1
            if (
                next_count > 1
                and batch_token_budget is not None
                and next_max * next_count > batch_token_budget
            ):
                break
            current_max = next_max
            cursor += 1
        if cursor == start:
            cursor += 1
        yield order[start:cursor]


def _flatten_batch(
    acts_stacked: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten padded ``[B, S, L, H]`` activations to flat+offsets."""
    device = acts_stacked.device
    attention_mask = batch["attention_mask"].to(device).bool()
    detection_mask = batch["detection_mask"].to(device).bool()

    flat_data = acts_stacked[attention_mask]
    flat_detection_mask = detection_mask[attention_mask]

    batch_size = attention_mask.shape[0]
    valid_lengths = attention_mask.sum(dim=1, dtype=torch.int64)
    offsets = torch.zeros(batch_size + 1, dtype=torch.int64, device=device)
    offsets[1:] = valid_lengths.cumsum(0)
    return flat_data, flat_detection_mask, offsets


def _stack_collect_step_activations(
    step: object,
    proxies: list[object],
) -> torch.Tensor:
    """Build a batched ``[B, S, L, H]`` activation tensor for one collect step."""
    layer_acts: list[torch.Tensor] = []
    for proxy in proxies:
        getter = getattr(step, "__getitem__", None)
        if getter is None:
            raise TypeError("Collect step does not support activation lookup.")
        act = getter(proxy)
        if not isinstance(act, torch.Tensor):
            raise TypeError(f"Expected tensor activations, got {type(act).__name__}.")
        if act.ndim != 3:
            raise ValueError(
                f"Expected batched activations with ndim=3, got shape {tuple(act.shape)}."
            )
        layer_acts.append(act)
    return torch.stack(layer_acts, dim=2)


def _stream_model(
    model: object,
    tokens: Tokens,
    layers: list[int],
    batch_size: int,
    hook_point: str,
    sort_by_length: bool,
    batch_token_budget: int | None,
) -> Generator[ActivationChunk, None, None]:
    """Unified streaming for ``mirin.Model``."""
    mirin = _import_mirin()

    model_obj: mirin.Model = model  # type: ignore[assignment]
    get_proxies = mirin.resolve_layer_sites(model_obj, layers, hook_point=hook_point)
    layer_tuple = tuple(layers)
    for batch_indices in _iter_batch_indices(
        tokens,
        batch_size=batch_size,
        sort_by_length=sort_by_length,
        batch_token_budget=batch_token_budget,
    ):
        batch = tokens.pad_batch(batch_indices, padding_side=tokens.padding_side)

        def _flatten_step(
            step: object,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
            acts_stacked = _stack_collect_step_activations(step, get_proxies)
            batch_value = getattr(step, "batch", None)
            if not isinstance(batch_value, dict):
                raise TypeError("Collect step batch must be a dict.")
            flat_data, detection_mask, offsets = _flatten_batch(acts_stacked, batch_value)
            chunk_indices = getattr(step, "indices", None)
            if not isinstance(chunk_indices, list):
                raise TypeError("Collect step indices must be a list.")
            return flat_data, detection_mask, offsets, chunk_indices

        iterator = model_obj.collect(
            batch,
            get=get_proxies,
            process=_flatten_step,
            max_items=batch_size,
            max_tokens=batch_token_budget,
            sort=False,
            stop_at_last_get=True,
        )
        yielded = False
        for flat_data, detection_mask, offsets, chunk_indices in iterator:
            yielded = True
            yield ActivationChunk(
                data=flat_data,
                detection_mask=detection_mask,
                offsets=offsets,
                indices=[batch_indices[idx] for idx in chunk_indices],
                layers=layer_tuple,
            )
        if yielded:
            continue
        yield ActivationChunk(
            data=torch.zeros(0, len(layers), 0, dtype=torch.float32),
            detection_mask=torch.empty(0, dtype=torch.bool),
            offsets=torch.zeros(len(batch_indices) + 1, dtype=torch.int64),
            indices=batch_indices,
            layers=layer_tuple,
        )


def stream_activations(
    model: object,
    tokens: Tokens,
    layers: list[int] | int,
    batch_size: int = 32,
    *,
    hook_point: str = "block",
    sort_by_length: bool = True,
    batch_token_budget: int | None = None,
) -> Generator[ActivationChunk, None, None]:
    """Yield backend-neutral activation chunks from a ``mirin.Model``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    layer_list = [layers] if isinstance(layers, int) else list(layers)
    model = _ensure_model(model)
    yield from _stream_model(
        model,
        tokens,
        layer_list,
        batch_size,
        hook_point,
        sort_by_length,
        batch_token_budget,
    )


def collect_activations(
    model: object,
    tokens: Tokens,
    layers: list[int] | int,
    batch_size: int = 32,
    pool: str | None = None,
    *,
    hook_point: str = "block",
    sort_by_length: bool = True,
    batch_token_budget: int | None = None,
    progress: bool = False,
    progress_desc: str | None = None,
    progress_leave: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Activations:
    """Collect all mirin activations into a single ``Activations`` object."""
    layer_list = [layers] if isinstance(layers, int) else list(layers)
    single_layer = len(layer_list) == 1
    n = len(tokens)
    metadata = dict(metadata or {})

    def _iter_stream():
        iterator = stream_activations(
            model,
            tokens,
            layer_list,
            batch_size,
            hook_point=hook_point,
            sort_by_length=sort_by_length,
            batch_token_budget=batch_token_budget,
        )
        if progress and tqdm is not None:
            desc = progress_desc or ("collect+pool" if pool else "collect")
            total_batches = math.ceil(n / batch_size) if n > 0 else 0
            return tqdm(iterator, total=total_batches, desc=desc, leave=progress_leave)
        return iterator

    if pool:
        pool_fn = getattr(P, pool, None)
        if pool_fn is None:
            available = [name for name in dir(P) if not name.startswith("_")]
            raise ValueError(f"Unknown pooling method: {pool}. Available: {available}")

        model_obj = _ensure_model(model)
        mirin = _import_mirin()
        get_proxies = mirin.resolve_layer_sites(model_obj, layer_list, hook_point=hook_point)
        out: torch.Tensor | None = None
        iterator: Iterable[list[int]] = _iter_batch_indices(
            tokens,
            batch_size=batch_size,
            sort_by_length=sort_by_length,
            batch_token_budget=batch_token_budget,
        )
        if progress and tqdm is not None:
            desc = progress_desc or "collect+pool"
            total_batches = math.ceil(n / batch_size) if n > 0 else 0
            iterator = tqdm(iterator, total=total_batches, desc=desc, leave=progress_leave)

        for batch_indices in iterator:
            batch = tokens.pad_batch(batch_indices, padding_side=tokens.padding_side)

            def _pool_step(step: object) -> tuple[torch.Tensor, list[int]]:
                acts_stacked = _stack_collect_step_activations(step, get_proxies)
                batch_value = getattr(step, "batch", None)
                if not isinstance(batch_value, dict):
                    raise TypeError("Collect step batch must be a dict.")
                flat_data, detection_mask, offsets = _flatten_batch(acts_stacked, batch_value)
                pooled = pool_fn(flat_data, detection_mask, offsets=offsets)
                chunk_indices = getattr(step, "indices", None)
                if not isinstance(chunk_indices, list):
                    raise TypeError("Collect step indices must be a list.")
                return pooled, chunk_indices

            for pooled, chunk_indices in model_obj.collect(
                batch,
                get=get_proxies,
                process=_pool_step,
                max_items=batch_size,
                max_tokens=batch_token_budget,
                sort=False,
                stop_at_last_get=True,
            ):
                if out is None:
                    out = torch.zeros(
                        n,
                        len(layer_list),
                        pooled.shape[-1],
                        dtype=pooled.dtype,
                        device="cpu",
                    )
                out_idx = torch.tensor(
                    [batch_indices[idx] for idx in chunk_indices],
                    dtype=torch.long,
                )
                out[out_idx] = pooled.detach().cpu()

        if out is None:
            out = torch.zeros(n, len(layer_list), 0, dtype=torch.float32)

        if single_layer:
            return Activations.from_tensor(
                data=out.squeeze(1),
                dims="bh",
                metadata=metadata,
            )
        return Activations.from_tensor(
            data=out,
            dims="blh",
            layers=tuple(layer_list),
            metadata=metadata,
        )

    per_sample: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n
    for chunk in _iter_stream():
        batch_chunk = chunk.offsets.shape[0] - 1
        for j in range(batch_chunk):
            start = int(chunk.offsets[j])
            end = int(chunk.offsets[j + 1])
            per_sample[chunk.indices[j]] = (
                chunk.data[start:end],
                chunk.detection_mask[start:end],
            )

    all_data: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    global_offsets = torch.zeros(n + 1, dtype=torch.int64)
    running = 0
    for i in range(n):
        if per_sample[i] is not None:
            data_i, mask_i = per_sample[i]
            all_data.append(data_i)
            all_masks.append(mask_i)
            running += data_i.shape[0]
        global_offsets[i + 1] = running

    if all_data:
        cat_data = torch.cat(all_data, dim=0)
        cat_mask = torch.cat(all_masks, dim=0)
        if cat_data.is_cuda:
            cat_data = cat_data.cpu()
            cat_mask = cat_mask.cpu()
    else:
        cat_data = torch.zeros(0, len(layer_list), 0, dtype=torch.float32)
        cat_mask = torch.empty(0, dtype=torch.bool)

    if single_layer:
        if cat_data.ndim != 2:
            cat_data = cat_data.squeeze(1)
        return Activations.from_flat(
            data=cat_data,
            offsets=global_offsets,
            detection_mask=cat_mask,
            dims="bsh",
            metadata=metadata,
        )
    return Activations.from_flat(
        data=cat_data,
        offsets=global_offsets,
        detection_mask=cat_mask,
        dims="blsh",
        layers=tuple(layer_list),
        metadata=metadata,
    )

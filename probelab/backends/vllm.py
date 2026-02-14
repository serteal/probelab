"""vLLM activation backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import torch

if TYPE_CHECKING:
    from ..processing.tokenization import Tokens


def _tokens_to_prompt_token_ids(tokens: "Tokens") -> list[list[int]]:
    """Convert flat+offsets token storage into ragged prompt token ID lists."""
    token_ids: list[list[int]] = []
    for i in range(len(tokens)):
        s, e = int(tokens.offsets[i]), int(tokens.offsets[i + 1])
        token_ids.append(tokens.input_ids[s:e].tolist())
    return token_ids


def _ordered_indices(
    token_ids: list[list[int]],
    *,
    batch_size: int | None,
    batch_token_budget: int | None,
    sort_by_length: bool,
    static_shape_bucketing: bool,
) -> list[int]:
    """Mirror ActivationEngine request ordering when preserve_input_order=False."""
    n = len(token_ids)
    if batch_size is None and batch_token_budget is None:
        return list(range(n))

    if static_shape_bucketing:
        buckets: dict[int, list[int]] = {}
        for i, ids in enumerate(token_ids):
            buckets.setdefault(len(ids), []).append(i)
        ordered: list[int] = []
        for seq_len in sorted(buckets.keys(), reverse=True):
            ordered.extend(buckets[seq_len])
        return ordered

    if sort_by_length:
        return sorted(range(n), key=lambda i: len(token_ids[i]), reverse=True)

    return list(range(n))


class VLLMBackend:
    """Backend adapter for activation-capable vLLM engines."""

    name = "vllm"

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
        if not hasattr(model_obj, "collect_flat"):
            raise TypeError(
                "vllm backend requires an activation engine object exposing "
                "`collect_flat(...)`."
            )

        batch_token_budget = kwargs.get(
            "vllm_batch_token_budget",
            kwargs.get("batch_token_budget"),
        )
        sort_by_length = bool(kwargs.get("vllm_sort_by_length", kwargs.get("sort_by_length", True)))
        pool_mode = kwargs.get("vllm_pool_mode")
        if pool_mode not in (None, "mean", "last_token"):
            raise ValueError(
                "vllm_pool_mode must be one of {None, 'mean', 'last_token'}"
            )
        # Staged export is currently unstable with activation-only fast path on
        # Gemma in this repo state (deterministic CUDA illegal access).
        use_staged_export = bool(kwargs.get("vllm_use_staged_export", False))
        static_shape_bucketing = bool(getattr(model_obj, "static_shape_bucketing", False))
        stream_device = kwargs.get(
            "vllm_stream_device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        stream_device = torch.device(stream_device)

        token_ids = _tokens_to_prompt_token_ids(tokens)
        ordered = _ordered_indices(
            token_ids,
            batch_size=batch_size,
            batch_token_budget=batch_token_budget,
            sort_by_length=sort_by_length,
            static_shape_bucketing=static_shape_bucketing,
        )
        prepooled = pool_mode in {"mean", "last_token"}

        def _iter_flat_results() -> Iterator[object]:
            stream_flat = getattr(model_obj, "stream_flat", None)
            if callable(stream_flat):
                yield from stream_flat(  # type: ignore[misc]
                    token_ids=token_ids,
                    batch_size=batch_size,
                    batch_token_budget=batch_token_budget,
                    sort_by_length=sort_by_length,
                    pool_mode=pool_mode,
                    preserve_input_order=False,
                    use_staged_export=use_staged_export,
                )
                return
            yield model_obj.collect_flat(  # type: ignore[attr-defined]
                token_ids=token_ids,
                batch_size=batch_size,
                batch_token_budget=batch_token_budget,
                sort_by_length=sort_by_length,
                pool_mode=pool_mode,
                preserve_input_order=False,
                use_staged_export=use_staged_export,
            )

        def _build_detection_mask(req_lengths: list[int], idx: list[int]) -> torch.Tensor:
            if prepooled:
                total = int(sum(max(int(n), 0) for n in req_lengths))
                if total <= 0:
                    return torch.empty(0, dtype=torch.bool)
                return torch.ones(total, dtype=torch.bool)

            # Some backends/models may return per-request activation lengths
            # that differ from tokenizer lengths (e.g., model-specific prefix
            # handling). Align per-sample detection mask to captured length.
            det_chunks: list[torch.Tensor] = []
            for j, req_len in enumerate(req_lengths):
                if req_len <= 0:
                    continue

                orig_i = idx[j]
                s, e = int(tokens.offsets[orig_i]), int(tokens.offsets[orig_i + 1])
                tok_len = max(e - s, 0)
                base = tokens.detection_mask[s:e].bool()

                if tok_len == req_len:
                    det_chunks.append(base)
                    continue

                if tok_len > req_len:
                    det_chunks.append(base[tok_len - req_len:])
                    continue

                pad = torch.zeros(req_len - tok_len, dtype=torch.bool)
                det_chunks.append(torch.cat([base, pad], dim=0))

            if det_chunks:
                return torch.cat(det_chunks, dim=0)
            return torch.empty(0, dtype=torch.bool)

        def _move_to_stream_device(
            flat_data: torch.Tensor,
            flat_det: torch.Tensor,
            offsets: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if stream_device.type == "cpu":
                if flat_data.is_cuda:
                    flat_data = flat_data.cpu()
                if flat_det.is_cuda:
                    flat_det = flat_det.cpu()
                if offsets.is_cuda:
                    offsets = offsets.cpu()
                return flat_data, flat_det, offsets
            return (
                flat_data if flat_data.device == stream_device else flat_data.to(stream_device, non_blocking=True),
                flat_det if flat_det.device == stream_device else flat_det.to(stream_device, non_blocking=True),
                offsets if offsets.device == stream_device else offsets.to(stream_device, non_blocking=True),
            )

        cursor = 0
        for flat_result in _iter_flat_results():
            layers_payload = getattr(flat_result, "layers", {})
            first_payload = (
                layers_payload.get(layers[0])
                if isinstance(layers_payload, dict)
                else None
            )

            # Fast path: consume chunked flat payload directly.
            if isinstance(first_payload, dict) and first_payload.get("__chunked__", False):
                chunk_lengths = first_payload.get("lengths", [])
                for chunk_idx, lens in enumerate(chunk_lengths):
                    lengths_t = (
                        lens.to(dtype=torch.int64, device="cpu")
                        if isinstance(lens, torch.Tensor)
                        else torch.tensor(lens, dtype=torch.int64)
                    )
                    chunk_size = int(lengths_t.numel())
                    idx = ordered[cursor : cursor + chunk_size]
                    if len(idx) != chunk_size:
                        raise ValueError("vLLM chunk ordering mismatch while streaming activations")
                    cursor += chunk_size

                    offsets = torch.zeros(chunk_size + 1, dtype=torch.int64)
                    if chunk_size > 0:
                        offsets[1:] = torch.cumsum(lengths_t, dim=0)
                    total_tokens = int(offsets[-1].item())

                    if len(layers) == 1:
                        values_chunks = first_payload.get("values", [])
                        if chunk_idx < len(values_chunks):
                            layer_vals = values_chunks[chunk_idx]
                            if not isinstance(layer_vals, torch.Tensor):
                                layer_vals = torch.as_tensor(layer_vals)
                        else:
                            layer_vals = torch.empty((0, 0), dtype=torch.float32)
                        hidden = int(layer_vals.shape[-1]) if layer_vals.ndim == 2 else 0
                        dtype = layer_vals.dtype if layer_vals.ndim == 2 else torch.float32
                        if total_tokens > 0:
                            rows = min(int(layer_vals.shape[0]), total_tokens)
                            if rows == total_tokens:
                                flat_data = layer_vals[:rows].unsqueeze(1)
                            else:
                                flat_data = torch.zeros(
                                    total_tokens,
                                    1,
                                    hidden,
                                    dtype=dtype,
                                    device=layer_vals.device,
                                )
                                if rows > 0:
                                    flat_data[:rows, 0, :] = layer_vals[:rows]
                        else:
                            flat_data = torch.zeros(
                                0,
                                1,
                                hidden,
                                dtype=dtype,
                                device=layer_vals.device,
                            )
                    else:
                        hidden = 0
                        dtype = torch.float32
                        data_device = torch.device("cpu")
                        for layer in layers:
                            payload = layers_payload.get(layer)
                            if payload is None:
                                raise ValueError(f"vLLM output missing requested layer {layer}")
                            vals_list = payload.get("values", [])
                            vals = vals_list[chunk_idx] if chunk_idx < len(vals_list) else torch.empty((0, 0))
                            if not isinstance(vals, torch.Tensor):
                                vals = torch.as_tensor(vals)
                            if vals.ndim == 2:
                                hidden = int(vals.shape[-1])
                                dtype = vals.dtype
                                data_device = vals.device
                                break
                        flat_data = torch.zeros(
                            total_tokens,
                            len(layers),
                            hidden,
                            dtype=dtype,
                            device=data_device,
                        )
                        for li, layer in enumerate(layers):
                            payload = layers_payload.get(layer)
                            if payload is None:
                                raise ValueError(f"vLLM output missing requested layer {layer}")
                            vals_list = payload.get("values", [])
                            vals = vals_list[chunk_idx] if chunk_idx < len(vals_list) else torch.empty((0, hidden))
                            if not isinstance(vals, torch.Tensor):
                                vals = torch.as_tensor(vals, device=data_device)
                            rows = min(int(vals.shape[0]), total_tokens)
                            if rows > 0:
                                flat_data[:rows, li, :] = vals[:rows]

                    req_lengths = lengths_t.tolist()
                    flat_det = _build_detection_mask(req_lengths, idx)
                    flat_data, flat_det, offsets = _move_to_stream_device(
                        flat_data, flat_det, offsets
                    )
                    yield flat_data, flat_det, offsets, idx
                continue

            # Fallback path for non-chunked payloads.
            activation_result = flat_result.to_activation_result()
            n_local = len(getattr(flat_result, "num_tokens", []))
            base = cursor
            for start in range(0, n_local, batch_size):
                end = min(start + batch_size, n_local)
                idx = ordered[base + start : base + end]
                chunk_size = end - start

                req_lengths: list[int] = []
                layer_values = activation_result.activations.get(layers[0])
                if layer_values is None:
                    raise ValueError(f"vLLM output missing requested layer {layers[0]}")
                for i in range(start, end):
                    req_lengths.append(int(layer_values[i].shape[0]))

                total_tokens = sum(req_lengths)
                offsets = torch.zeros(chunk_size + 1, dtype=torch.int64)
                for j, length in enumerate(req_lengths):
                    offsets[j + 1] = offsets[j] + length

                if total_tokens > 0:
                    hidden = int(layer_values[start].shape[-1])
                    dtype = layer_values[start].dtype
                    flat_data = torch.zeros(
                        total_tokens,
                        len(layers),
                        hidden,
                        dtype=dtype,
                        device=layer_values[start].device,
                    )
                    for li, layer in enumerate(layers):
                        req_vals = activation_result.activations.get(layer)
                        if req_vals is None:
                            raise ValueError(f"vLLM output missing requested layer {layer}")
                        for j in range(chunk_size):
                            s, e = int(offsets[j]), int(offsets[j + 1])
                            if s < e:
                                flat_data[s:e, li, :] = req_vals[start + j]
                else:
                    flat_data = torch.zeros(0, len(layers), 0, dtype=torch.float32)

                flat_det = _build_detection_mask(req_lengths, idx)
                flat_data, flat_det, offsets = _move_to_stream_device(
                    flat_data, flat_det, offsets
                )
                yield flat_data, flat_det, offsets, idx
            cursor += n_local

        if cursor != len(ordered):
            raise ValueError("vLLM stream did not return all requested activations")

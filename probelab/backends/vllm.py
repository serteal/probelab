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
        use_staged_export = bool(kwargs.get("vllm_use_staged_export", True))

        token_ids = _tokens_to_prompt_token_ids(tokens)
        flat_result = model_obj.collect_flat(  # type: ignore[attr-defined]
            token_ids=token_ids,
            batch_size=batch_size,
            batch_token_budget=batch_token_budget,
            sort_by_length=sort_by_length,
            preserve_input_order=True,
            use_staged_export=use_staged_export,
        )
        activation_result = flat_result.to_activation_result()

        for start in range(0, len(token_ids), batch_size):
            end = min(start + batch_size, len(token_ids))
            idx = list(range(start, end))
            chunk_size = end - start

            # Build flat [T, n_layers, hidden] directly from ragged per-request tensors
            # First pass: compute per-request lengths and total tokens
            req_lengths: list[int] = []
            for i in range(start, end):
                # Use the first layer to determine sequence length
                layer_values = activation_result.activations.get(layers[0])
                if layer_values is None:
                    raise ValueError(f"vLLM output missing requested layer {layers[0]}")
                req_lengths.append(int(layer_values[i].shape[0]))

            total_tokens = sum(req_lengths)
            offsets = torch.zeros(chunk_size + 1, dtype=torch.int64)
            for j, length in enumerate(req_lengths):
                offsets[j + 1] = offsets[j] + length

            # Build flat data: [T, n_layers, hidden]
            if total_tokens > 0:
                hidden = int(activation_result.activations[layers[0]][start].shape[-1])
                flat_data = torch.zeros(total_tokens, len(layers), hidden, dtype=torch.float32)

                for li, layer in enumerate(layers):
                    layer_values = activation_result.activations.get(layer)
                    if layer_values is None:
                        raise ValueError(f"vLLM output missing requested layer {layer}")
                    for j in range(chunk_size):
                        s, e = int(offsets[j]), int(offsets[j + 1])
                        if s < e:
                            flat_data[s:e, li, :] = layer_values[start + j]
            else:
                flat_data = torch.zeros(0, len(layers), 0, dtype=torch.float32)

            # Build det from detection_mask using flat+offsets
            det_chunks: list[torch.Tensor] = []
            for i in range(start, end):
                s, e = int(tokens.offsets[i]), int(tokens.offsets[i + 1])
                if e > s:
                    det_chunks.append(tokens.detection_mask[s:e].bool())

            if det_chunks:
                flat_det = torch.cat(det_chunks, dim=0)
            else:
                flat_det = torch.empty(0, dtype=torch.bool)

            yield flat_data, flat_det, offsets, idx

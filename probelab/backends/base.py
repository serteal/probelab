"""Backend protocol for activation extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from ..processing.tokenization import Tokens


@runtime_checkable
class ActivationBackend(Protocol):
    """Minimal backend protocol used by activation collection."""

    name: str

    def stream_raw(
        self,
        model_obj: object,
        tokens: "Tokens",
        layers: list[int],
        batch_size: int,
        **kwargs: Any,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]]:
        """Yield (flat_data, det, offsets, indices) per batch.

        flat_data: [total_batch_tokens, n_layers, hidden]
        det: [total_batch_tokens] bool detection mask
        offsets: [batch_chunk+1] int64 cumulative token counts
        indices: original sample indices
        """
        ...

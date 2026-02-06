"""Activation collection and container.

This module provides the Activations container and functions for extracting
activations from language models using hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, Iterator

import torch

from .. import pool as P
from ..models import HookedModel

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .tokenization import Tokens


DIMS = {"bh", "bsh", "blh", "blsh"}


@dataclass(slots=True)
class Activations:
    """Activation tensor with explicit dimension labels.

    Axis order is always [batch, layer?, seq?, hidden].

    Args:
        data: Activation tensor
        dims: Dimension format - one of "bh", "bsh", "blh", "blsh"
        mask: Detection mask [batch, seq], required if "s" in dims
        layers: Layer indices tuple, required if "l" in dims
    """

    data: torch.Tensor
    dims: str
    mask: torch.Tensor | None = None
    layers: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.dims not in DIMS:
            raise ValueError(f"dims must be one of {DIMS}, got {self.dims!r}")
        if len(self.data.shape) != len(self.dims):
            raise ValueError(f"data has {self.data.ndim}D but dims={self.dims!r}")
        if "s" in self.dims and self.mask is None:
            raise ValueError("mask required when dims contains 's'")
        if "s" not in self.dims and self.mask is not None:
            raise ValueError("mask provided but dims has no 's'")
        if "l" in self.dims and self.layers is None:
            raise ValueError("layers required when dims contains 'l'")
        if not self.data.is_floating_point():
            self.data = self.data.float()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def batch_size(self) -> int:
        return self.data.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.data.shape[-1]

    @property
    def seq_len(self) -> int | None:
        return self.data.shape[self.dims.index("s")] if "s" in self.dims else None

    @property
    def n_layers(self) -> int | None:
        return self.data.shape[self.dims.index("l")] if "l" in self.dims else None

    # -------------------------------------------------------------------------
    # Pooling (delegates to pl.pool.*)
    # -------------------------------------------------------------------------

    def mean_pool(self) -> "Activations":
        """Pool sequence dimension by mean over valid tokens."""
        return self._pool(P.mean)

    def max_pool(self) -> "Activations":
        """Pool sequence dimension by max over valid tokens."""
        return self._pool(P.max)

    def last_pool(self) -> "Activations":
        """Pool sequence dimension by taking last valid token."""
        return self._pool(P.last_token)

    def _pool(self, pool_fn) -> "Activations":
        if "s" not in self.dims:
            raise ValueError("No sequence dimension to pool")
        idx = self.dims.index("s")
        pooled = pool_fn(self.data, self.mask, dim=idx)
        new_dims = self.dims.replace("s", "")
        return Activations(pooled, new_dims, mask=None, layers=self.layers)

    # -------------------------------------------------------------------------
    # Layer selection
    # -------------------------------------------------------------------------

    def select_layers(self, layer_or_layers: int | list[int]) -> "Activations":
        """Select layer(s). Single int removes layer axis."""
        if "l" not in self.dims:
            raise ValueError("No layer axis to select from")

        dim = self.dims.index("l")

        if isinstance(layer_or_layers, int):
            # Single layer: remove axis (returns view)
            idx = self.layers.index(layer_or_layers)
            selected = self.data.select(dim, idx)
            new_dims = self.dims.replace("l", "")
            return Activations(selected, new_dims, mask=self.mask, layers=None)
        else:
            # Multiple layers: keep axis
            indices = [self.layers.index(l) for l in layer_or_layers]
            idx_tensor = torch.tensor(indices, device=self.data.device)
            selected = self.data.index_select(dim, idx_tensor)
            return Activations(selected, self.dims, mask=self.mask, layers=tuple(layer_or_layers))

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def iter_layers(self) -> Iterator[tuple[int, "Activations"]]:
        """Yield (layer_index, single-layer Activations)."""
        if "l" not in self.dims:
            layer_idx = self.layers[0] if self.layers else 0
            yield layer_idx, self
            return
        for layer in self.layers:
            yield layer, self.select_layers(layer)

    # -------------------------------------------------------------------------
    # Token extraction (for probe training)
    # -------------------------------------------------------------------------

    def extract_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract masked tokens for token-level training.

        Returns:
            (features, tokens_per_sample) where features is [n_tokens, hidden]
        """
        if "s" not in self.dims:
            raise ValueError("No sequence dimension to extract from")
        if "l" in self.dims and self.n_layers != 1:
            raise ValueError("Must select single layer before extracting tokens")

        tensor = self.data.squeeze(self.dims.index("l")) if "l" in self.dims else self.data
        bool_mask = self.mask.bool()
        tokens_per_sample = bool_mask.sum(dim=1)
        features = tensor[bool_mask]

        return features, tokens_per_sample.to(device=self.data.device)

    # -------------------------------------------------------------------------
    # Device / dtype
    # -------------------------------------------------------------------------

    def to(self, device) -> "Activations":
        return Activations(
            self.data.to(device),
            self.dims,
            self.mask.to(device) if self.mask is not None else None,
            self.layers,
        )

    # -------------------------------------------------------------------------
    # Save / Load (HDF5)
    # -------------------------------------------------------------------------

    def save(self, path: str, compression: str | None = "gzip", compression_opts: int = 4) -> None:
        """Save activations to HDF5 file with compression."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for saving. Install with: pip install probelab[storage]")

        data_cpu = self.data.cpu().numpy()
        chunk_shape = (1,) + data_cpu.shape[1:] if "l" in self.dims else None

        with h5py.File(path, "w") as f:
            f.create_dataset(
                "data", data=data_cpu, compression=compression,
                compression_opts=compression_opts if compression == "gzip" else None,
                chunks=chunk_shape,
            )
            f.attrs["dims"] = self.dims
            f.attrs["dtype"] = str(self.data.dtype)
            f.attrs["probelab_version"] = "0.0.1"

            if self.layers is not None:
                f.create_dataset("layers", data=list(self.layers))
            if self.mask is not None:
                f.create_dataset(
                    "mask", data=self.mask.cpu().numpy(),
                    compression=compression,
                    compression_opts=compression_opts if compression == "gzip" else None,
                )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "Activations":
        """Load activations from HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for loading. Install with: pip install probelab[storage]")

        with h5py.File(path, "r") as f:
            data = torch.tensor(f["data"][:], device=device).float()
            dims = f.attrs["dims"]
            layers = tuple(f["layers"][:]) if "layers" in f else None
            mask = torch.tensor(f["mask"][:], device=device).float() if "mask" in f else None

        return cls(data=data, dims=dims, mask=mask, layers=layers)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _batches(
    tokens: "Tokens", batch_size: int
) -> Generator[tuple[dict[str, torch.Tensor], list[int]], None, None]:
    """Yield (batch_dict, indices) sorted by length for efficiency."""
    seq_lens = tokens.attention_mask.sum(1)
    order = seq_lens.argsort(descending=True)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        max_len = int(seq_lens[idx].max())
        if tokens.padding_side == "right":
            sl = slice(None, max_len)
        else:
            sl = slice(-max_len, None)
        yield {
            "input_ids": tokens.input_ids[idx][..., sl],
            "attention_mask": tokens.attention_mask[idx][..., sl],
            "detection_mask": tokens.detection_mask[idx][..., sl],
        }, idx.tolist()


def _extract(model: "PreTrainedModel", batch: dict, layers: list[int]) -> torch.Tensor:
    """Extract activations [layers, batch, seq, hidden] from one batch."""
    batch_gpu = {k: v.to(model.device) for k, v in batch.items() if k != "detection_mask"}
    with HookedModel(model, layers, detach_activations=True) as h:
        return h.get_activations(batch_gpu).cpu()


def _hidden_dim(model: "PreTrainedModel") -> int:
    """Get hidden dimension from model config."""
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return cfg.text_config.hidden_size
    raise ValueError(f"Cannot determine hidden dimension for {type(model)}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def stream_activations(
    model: "PreTrainedModel",
    tokens: "Tokens",
    layers: list[int] | int,
    batch_size: int = 32,
) -> Generator[tuple[torch.Tensor, list[int], int], None, None]:
    """Yield (activations, indices, seq_len) per batch.

    Args:
        model: Model to extract activations from.
        tokens: Tokenized inputs from tokenize_dialogues().
        layers: Layer index or list of indices.
        batch_size: Batch size for extraction.

    Yields:
        (acts, indices, seq_len) where:
        - acts: [n_layers, batch, seq, hidden] tensor
        - indices: original sample indices for this batch
        - seq_len: sequence length for this batch
    """
    layers = [layers] if isinstance(layers, int) else list(layers)
    for batch, idx in _batches(tokens, batch_size):
        yield _extract(model, batch, layers), idx, batch["input_ids"].shape[1]


def collect_activations(
    model: "PreTrainedModel",
    tokens: "Tokens",
    layers: list[int] | int,
    batch_size: int = 32,
    pool: str | None = None,
) -> Activations:
    """Collect all activations into single Activations object.

    Args:
        model: Model to extract activations from.
        tokens: Tokenized inputs from tokenize_dialogues().
        layers: Layer index or list of indices.
        batch_size: Batch size for extraction.
        pool: Optional pooling over sequence ("mean", "max", "last_token").

    Returns:
        Activations with shape:
        - Single layer: [batch, seq, hidden] or [batch, hidden] if pooled
        - Multiple layers: [batch, layer, seq, hidden] or [batch, layer, hidden] if pooled
    """
    layers = [layers] if isinstance(layers, int) else list(layers)
    single_layer = len(layers) == 1
    n, max_seq, d = len(tokens), tokens.seq_len, _hidden_dim(model)

    if pool:
        pool_fn = getattr(P, pool, None)
        if pool_fn is None:
            available = [name for name in dir(P) if not name.startswith("_")]
            raise ValueError(f"Unknown pooling method: {pool}. Available: {available}")

        # Collect as [batch, layer, hidden]
        out = torch.zeros(n, len(layers), d, dtype=model.dtype)
        for acts, idx, sl in stream_activations(model, tokens, layers, batch_size):
            # acts is [layer, batch_chunk, seq, hidden], transpose to [batch_chunk, layer, seq, hidden]
            acts_t = acts.transpose(0, 1)
            if tokens.padding_side == "right":
                mask = tokens.detection_mask[idx, :sl]
            else:
                mask = tokens.detection_mask[idx, -sl:]
            # Pool over seq (dim=2)
            out[idx] = pool_fn(acts_t, mask, dim=2)

        if single_layer:
            return Activations(
                data=out.squeeze(1),
                dims="bh",
                mask=None,
                layers=None,
            )
        return Activations(
            data=out,
            dims="blh",
            mask=None,
            layers=tuple(layers),
        )

    # Collect as [batch, layer, seq, hidden]
    out = torch.zeros(n, len(layers), max_seq, d, dtype=model.dtype)
    for acts, idx, sl in stream_activations(model, tokens, layers, batch_size):
        # acts is [layer, batch_chunk, seq, hidden], transpose to [batch_chunk, layer, seq, hidden]
        acts_t = acts.transpose(0, 1)
        if tokens.padding_side == "right":
            out[idx, :, :sl] = acts_t
        else:
            out[idx, :, -sl:] = acts_t

    if single_layer:
        return Activations(
            data=out.squeeze(1),
            dims="bsh",
            mask=tokens.detection_mask,
            layers=None,
        )
    return Activations(
        data=out,
        dims="blsh",
        mask=tokens.detection_mask,
        layers=tuple(layers),
    )

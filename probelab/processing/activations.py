"""Activation collection and container.

This module provides the Activations container and functions for extracting
activations from language models using hooks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator, Iterator

import torch

from ..backends import get_context_defaults, resolve_backend
from .. import pool as P

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

if TYPE_CHECKING:
    from .tokenization import Tokens


DIMS = {"bh", "bsh", "blh", "blsh"}


@dataclass(slots=True)
class Activations:
    """Activation tensor with explicit dimension labels.

    When ``"s"`` is in *dims* the sequence dimension is stored in a
    **flat+offsets** layout:

    * ``data`` – ``[total_tokens, hidden]`` (dims ``"bsh"``) or
      ``[total_tokens, n_layers, hidden]`` (dims ``"blsh"``).  The leading
      axis concatenates all samples; the physical ``ndim`` is therefore
      ``len(dims) - 1`` (the batch dimension is implicit).
    * ``offsets`` – ``[batch+1]`` ``int64`` cumulative token counts.
      Sample *i* spans ``data[offsets[i]:offsets[i+1]]``.
    * ``det`` – ``[total_tokens]`` ``bool`` per-real-token detection mask.

    When ``"s"`` is **not** in *dims* the container behaves exactly as
    before (``offsets=None``, ``det=None``).

    Args:
        data: Activation tensor
        dims: Dimension format – one of ``"bh"``, ``"bsh"``, ``"blh"``, ``"blsh"``
        offsets: ``[batch+1]`` int64 cumulative token counts (required when ``"s"`` in dims)
        det: ``[total_tokens]`` bool detection mask (required when ``"s"`` in dims)
        layers: Layer indices tuple (required when ``"l"`` in dims)
    """

    data: torch.Tensor
    dims: str
    offsets: torch.Tensor | None = None
    det: torch.Tensor | None = None
    layers: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.dims not in DIMS:
            raise ValueError(f"dims must be one of {DIMS}, got {self.dims!r}")

        if "s" in self.dims:
            # Flat+offsets: physical ndim is len(dims) - 1
            expected_ndim = len(self.dims) - 1
            if self.data.ndim != expected_ndim:
                raise ValueError(
                    f"data has {self.data.ndim}D but dims={self.dims!r} "
                    f"requires {expected_ndim}D (flat+offsets layout)"
                )
            if self.offsets is None:
                raise ValueError("offsets required when dims contains 's'")
            if self.det is None:
                raise ValueError("det required when dims contains 's'")
        else:
            if self.data.ndim != len(self.dims):
                raise ValueError(
                    f"data has {self.data.ndim}D but dims={self.dims!r}"
                )
            if self.offsets is not None:
                raise ValueError("offsets provided but dims has no 's'")
            if self.det is not None:
                raise ValueError("det provided but dims has no 's'")

        if "l" in self.dims and self.layers is None:
            raise ValueError("layers required when dims contains 'l'")
        if not self.data.is_floating_point():
            self.data = self.data.float()

    # -------------------------------------------------------------------------
    # Construction helpers
    # -------------------------------------------------------------------------

    @classmethod
    def from_padded(
        cls,
        data: torch.Tensor,
        detection_mask: torch.Tensor,
        dims: str,
        layers: tuple[int, ...] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> "Activations":
        """Build flat+offsets Activations from padded rectangular tensors.

        Args:
            data: Padded activation tensor (``[batch, seq, hidden]`` or
                ``[batch, layer, seq, hidden]``).
            detection_mask: ``[batch, seq]`` bool/float – which tokens are
                detection-relevant.
            dims: Dimension string (must contain ``"s"``).
            layers: Layer indices (required when ``"l"`` in dims).
            attention_mask: ``[batch, seq]`` bool/float – which tokens are
                real (attended). If *None*, ``detection_mask`` is used instead
                (i.e. only detection tokens are kept).

        Returns:
            Flat+offsets ``Activations``.
        """
        if "s" not in dims:
            raise ValueError("from_padded requires dims with 's'")

        det_bool = detection_mask.bool()
        keep = attention_mask.bool() if attention_mask is not None else det_bool

        batch = data.shape[0]
        seq_dim = dims.index("s")
        lengths = keep.sum(dim=1)  # [batch]

        # Build flat data and det
        flat_chunks: list[torch.Tensor] = []
        det_chunks: list[torch.Tensor] = []
        for i in range(batch):
            mask_i = keep[i]  # [seq]
            if seq_dim == 1:
                # dims "bsh": data is [batch, seq, hidden]
                flat_chunks.append(data[i][mask_i])
            elif seq_dim == 2:
                # dims "blsh": data is [batch, layer, seq, hidden]
                flat_chunks.append(data[i][:, mask_i].transpose(0, 1))
            det_chunks.append(det_bool[i][mask_i])

        flat_data = torch.cat(flat_chunks, dim=0) if flat_chunks else data.new_empty(0, *data.shape[2:])
        flat_det = torch.cat(det_chunks, dim=0) if det_chunks else torch.empty(0, dtype=torch.bool)

        offsets = torch.zeros(batch + 1, dtype=torch.int64)
        offsets[1:] = lengths.cumsum(0)

        return cls(data=flat_data, dims=dims, offsets=offsets, det=flat_det, layers=layers)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def batch_size(self) -> int:
        if "s" in self.dims:
            return self.offsets.shape[0] - 1
        return self.data.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.data.shape[-1]

    @property
    def seq_len(self) -> int | None:
        if "s" not in self.dims:
            return None
        lengths = self.offsets[1:] - self.offsets[:-1]
        return int(lengths.max().item()) if lengths.numel() > 0 else 0

    @property
    def total_tokens(self) -> int | None:
        if "s" not in self.dims:
            return None
        return self.data.shape[0]

    @property
    def n_layers(self) -> int | None:
        if "l" not in self.dims:
            return None
        if "s" in self.dims:
            # "blsh" → data is [total_tokens, n_layers, hidden], layer dim=1
            return self.data.shape[1]
        return self.data.shape[self.dims.index("l")]

    # -------------------------------------------------------------------------
    # Materialization helpers
    # -------------------------------------------------------------------------

    def to_padded(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize padded ``[batch, (layer,) max_seq, hidden]`` + detection mask.

        Returns:
            ``(padded_data, padded_det)`` where *padded_det* is ``[batch, max_seq]`` bool.
        """
        if "s" not in self.dims:
            raise ValueError("to_padded requires 's' in dims")

        batch = self.batch_size
        max_seq = self.seq_len
        hidden = self.hidden_size

        if "l" in self.dims:
            # data: [T, n_layers, hidden] → padded [batch, n_layers, max_seq, hidden]
            n_layers = self.n_layers
            padded = self.data.new_zeros(batch, n_layers, max_seq, hidden)
            padded_det = torch.zeros(batch, max_seq, dtype=torch.bool, device=self.data.device)
            for i in range(batch):
                s, e = int(self.offsets[i]), int(self.offsets[i + 1])
                length = e - s
                if length > 0:
                    # data[s:e] is [length, n_layers, hidden] → transpose to [n_layers, length, hidden]
                    padded[i, :, :length] = self.data[s:e].transpose(0, 1)
                    padded_det[i, :length] = self.det[s:e]
        else:
            # data: [T, hidden] → padded [batch, max_seq, hidden]
            padded = self.data.new_zeros(batch, max_seq, hidden)
            padded_det = torch.zeros(batch, max_seq, dtype=torch.bool, device=self.data.device)
            for i in range(batch):
                s, e = int(self.offsets[i]), int(self.offsets[i + 1])
                length = e - s
                if length > 0:
                    padded[i, :length] = self.data[s:e]
                    padded_det[i, :length] = self.det[s:e]

        return padded, padded_det

    def pad_batch(self, indices: list[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize padded tensors for a subset of samples.

        Pads to the LOCAL max sequence length of the selected samples, which is
        more memory-efficient than ``to_padded()`` for mini-batching.

        Args:
            indices: Sample indices to extract.

        Returns:
            ``(padded_data, padded_det)`` – same format as ``to_padded()`` but
            only for the requested samples and padded to local max.
        """
        if "s" not in self.dims:
            raise ValueError("pad_batch requires 's' in dims")

        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        hidden = self.hidden_size
        # Compute local max
        local_max = 0
        for i in indices:
            length = int(self.offsets[i + 1]) - int(self.offsets[i])
            local_max = max(local_max, length)

        sub_batch = len(indices)

        if "l" in self.dims:
            n_layers = self.n_layers
            padded = self.data.new_zeros(sub_batch, n_layers, local_max, hidden)
            padded_det = torch.zeros(sub_batch, local_max, dtype=torch.bool, device=self.data.device)
            for j, i in enumerate(indices):
                s, e = int(self.offsets[i]), int(self.offsets[i + 1])
                length = e - s
                if length > 0:
                    padded[j, :, :length] = self.data[s:e].transpose(0, 1)
                    padded_det[j, :length] = self.det[s:e]
        else:
            padded = self.data.new_zeros(sub_batch, local_max, hidden)
            padded_det = torch.zeros(sub_batch, local_max, dtype=torch.bool, device=self.data.device)
            for j, i in enumerate(indices):
                s, e = int(self.offsets[i]), int(self.offsets[i + 1])
                length = e - s
                if length > 0:
                    padded[j, :length] = self.data[s:e]
                    padded_det[j, :length] = self.det[s:e]

        return padded, padded_det

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

        pooled = pool_fn(self.data, self.det, offsets=self.offsets)
        new_dims = self.dims.replace("s", "")
        return Activations(pooled, new_dims, offsets=None, det=None, layers=self.layers)

    # -------------------------------------------------------------------------
    # Layer selection
    # -------------------------------------------------------------------------

    def select_layers(self, layer_or_layers: int | list[int]) -> "Activations":
        """Select layer(s). Single int removes layer axis."""
        if "l" not in self.dims:
            raise ValueError("No layer axis to select from")

        if "s" in self.dims:
            # data is [T, n_layers, hidden], layer dim = 1
            dim = 1
        else:
            dim = self.dims.index("l")

        if isinstance(layer_or_layers, int):
            idx = self.layers.index(layer_or_layers)
            selected = self.data.select(dim, idx)
            new_dims = self.dims.replace("l", "")
            return Activations(selected, new_dims, offsets=self.offsets, det=self.det, layers=None)
        else:
            indices = [self.layers.index(l) for l in layer_or_layers]
            idx_tensor = torch.tensor(indices, device=self.data.device)
            selected = self.data.index_select(dim, idx_tensor)
            return Activations(selected, self.dims, offsets=self.offsets, det=self.det, layers=tuple(layer_or_layers))

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

        data = self.data.squeeze(1) if "l" in self.dims else self.data
        det_bool = self.det.bool()
        features = data[det_bool]

        # Compute tokens_per_sample via offsets
        batch = self.batch_size
        tokens_per_sample = torch.zeros(batch, dtype=torch.long, device=self.data.device)
        for i in range(batch):
            s, e = int(self.offsets[i]), int(self.offsets[i + 1])
            tokens_per_sample[i] = det_bool[s:e].sum()

        return features, tokens_per_sample

    # -------------------------------------------------------------------------
    # Device / dtype
    # -------------------------------------------------------------------------

    def to(self, device) -> "Activations":
        return Activations(
            self.data.to(device),
            self.dims,
            self.offsets.to(device) if self.offsets is not None else None,
            self.det.to(device) if self.det is not None else None,
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
            if self.offsets is not None:
                f.create_dataset("offsets", data=self.offsets.cpu().numpy())
            if self.det is not None:
                f.create_dataset("det", data=self.det.cpu().numpy())

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
            offsets = torch.tensor(f["offsets"][:], dtype=torch.int64, device=device) if "offsets" in f else None
            det = torch.tensor(f["det"][:], dtype=torch.bool, device=device) if "det" in f else None

            # Legacy backward compat: old files have "mask" instead of offsets/det
            if "s" in dims and offsets is None and "mask" in f:
                return cls._load_legacy_padded(f, data, dims, layers, device)

        return cls(data=data, dims=dims, offsets=offsets, det=det, layers=layers)

    @classmethod
    def _load_legacy_padded(cls, f, data, dims, layers, device):
        """Load old-format HDF5 files with padded data and mask."""
        mask = torch.tensor(f["mask"][:], device=device).float()
        return cls.from_padded(data, mask, dims, layers)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def stream_activations(
    model: object,
    tokens: "Tokens",
    layers: list[int] | int,
    batch_size: int = 32,
    *,
    backend: str = "auto",
    **backend_kwargs: Any,
) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]], None, None]:
    """Yield (flat_data, det, offsets, indices) per batch.

    Args:
        model: Loaded model object (HF model or vLLM activation engine).
        tokens: Tokenized inputs from tokenize_dialogues().
        layers: Layer index or list of indices.
        batch_size: Batch size for extraction.
        backend: Backend override ("auto", "transformers", "vllm").
        **backend_kwargs: Backend-specific extraction options.

    Yields:
        (flat_data, det, offsets, indices) where:
        - flat_data: [total_batch_tokens, n_layers, hidden] tensor
        - det: [total_batch_tokens] bool detection mask
        - offsets: [batch_chunk+1] int64 cumulative token counts
        - indices: original sample indices for this batch
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    layers = [layers] if isinstance(layers, int) else list(layers)
    merged_kwargs = get_context_defaults()
    merged_kwargs.update(backend_kwargs)
    selected_backend = backend
    if selected_backend == "auto":
        context_backend = merged_kwargs.pop("backend", "auto")
        if isinstance(context_backend, str):
            selected_backend = context_backend
    else:
        merged_kwargs.pop("backend", None)

    backend_impl = resolve_backend(model, backend=selected_backend)

    for flat_data, det, offsets, idx in backend_impl.stream_raw(
        model_obj=model,
        tokens=tokens,
        layers=layers,
        batch_size=batch_size,
        **merged_kwargs,
    ):
        yield flat_data, det, offsets, idx


def collect_activations(
    model: object,
    tokens: "Tokens",
    layers: list[int] | int,
    batch_size: int = 32,
    pool: str | None = None,
    *,
    backend: str = "auto",
    progress: bool = False,
    progress_desc: str | None = None,
    progress_leave: bool = False,
    **backend_kwargs: Any,
) -> Activations:
    """Collect all activations into single Activations object.

    Args:
        model: Loaded model object (HF model or vLLM activation engine).
        tokens: Tokenized inputs from tokenize_dialogues().
        layers: Layer index or list of indices.
        batch_size: Batch size for extraction.
        pool: Optional pooling over sequence ("mean", "max", "last_token").
        backend: Backend override ("auto", "transformers", "vllm").
        **backend_kwargs: Backend-specific extraction options.

    Returns:
        Activations with shape:
        - Single layer, no pool: flat [total_tokens, hidden] with offsets
        - Multiple layers, no pool: flat [total_tokens, n_layers, hidden] with offsets
        - Pooled: [batch, hidden] or [batch, n_layers, hidden]
    """
    layers = [layers] if isinstance(layers, int) else list(layers)
    single_layer = len(layers) == 1
    n = len(tokens)
    stream_kwargs = dict(backend_kwargs)
    if (
        pool in {"mean", "last_token"}
        and "vllm_pool_mode" not in stream_kwargs
    ):
        try:
            backend_impl = resolve_backend(model, backend=backend)
            if (
                getattr(backend_impl, "name", None) == "vllm"
                and bool(tokens.detection_mask.bool().all().item())
            ):
                # Native vLLM pooled capture avoids exporting full token streams
                # when pooling over all tokens.
                stream_kwargs["vllm_pool_mode"] = pool
        except Exception:
            pass

    def _iter_stream():
        iterator = stream_activations(
            model,
            tokens,
            layers,
            batch_size,
            backend=backend,
            **stream_kwargs,
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

        # Collect as [batch, layer, hidden]
        out: torch.Tensor | None = None
        for flat_data, det, offsets, idx in _iter_stream():
            # flat_data: [T, n_layers, hidden]
            if out is None:
                out = torch.zeros(
                    n,
                    len(layers),
                    flat_data.shape[-1],
                    dtype=flat_data.dtype,
                    device=flat_data.device,
                )

            # Vectorized pooling over all layers at once: [T, L, H] -> [B, L, H]
            pooled = pool_fn(flat_data, det, offsets=offsets)
            out_idx = torch.tensor(idx, dtype=torch.long, device=pooled.device)
            out[out_idx] = pooled

        if out is None:
            out = torch.zeros(n, len(layers), 0, dtype=torch.float32)
        elif out.is_cuda:
            # Keep all reduction work on GPU, move to CPU only once at the end.
            out = out.cpu()
        if out.dtype != torch.float32:
            # Backends may stream lower-precision activations for throughput.
            # Cast once at the boundary instead of per-batch/per-token.
            out = out.float()

        if single_layer:
            return Activations(data=out.squeeze(1), dims="bh", layers=None)
        return Activations(data=out, dims="blh", layers=tuple(layers))

    # No pooling: collect flat data per sample, then concatenate in order
    per_sample: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n
    for flat_data, det, offsets, idx in _iter_stream():
        batch_chunk = offsets.shape[0] - 1
        for j in range(batch_chunk):
            s, e = int(offsets[j]), int(offsets[j + 1])
            per_sample[idx[j]] = (flat_data[s:e], det[s:e])

    # Build global flat tensor and offsets
    all_data: list[torch.Tensor] = []
    all_det: list[torch.Tensor] = []
    global_offsets = torch.zeros(n + 1, dtype=torch.int64)
    running = 0
    for i in range(n):
        if per_sample[i] is not None:
            d, dt = per_sample[i]
            all_data.append(d)
            all_det.append(dt)
            running += d.shape[0]
        global_offsets[i + 1] = running

    if all_data:
        cat_data = torch.cat(all_data, dim=0)
        cat_det = torch.cat(all_det, dim=0)
        if cat_data.is_cuda:
            # Build flat representation on GPU, transfer once after concatenation.
            cat_data = cat_data.cpu()
            cat_det = cat_det.cpu()
    else:
        cat_data = torch.zeros(0, len(layers), 0, dtype=torch.float32)
        cat_det = torch.empty(0, dtype=torch.bool)

    if single_layer:
        # Squeeze layer dim: [T, 1, hidden] → [T, hidden]
        if cat_data.ndim == 2:
            pass  # already [T, hidden]
        else:
            cat_data = cat_data.squeeze(1)
        return Activations(
            data=cat_data,
            dims="bsh",
            offsets=global_offsets,
            det=cat_det,
            layers=None,
        )
    return Activations(
        data=cat_data,
        dims="blsh",
        offsets=global_offsets,
        det=cat_det,
        layers=tuple(layers),
    )

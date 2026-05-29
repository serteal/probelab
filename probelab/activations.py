"""Backend-agnostic activation container."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Iterator

import torch

from . import pool as P


DIMS = {"bh", "bsh", "blh", "blsh"}


def _copy_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _metadata_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    try:
        result = left == right
    except Exception:
        return False
    return result if isinstance(result, bool) else False


def _cat_metadata(items: list["Activations"]) -> dict[str, Any]:
    first = items[0].metadata
    if all(_metadata_equal(item.metadata, first) for item in items[1:]):
        return _copy_metadata(first)
    return {"sources": [_copy_metadata(item.metadata) for item in items]}


def _flatten_padded(
    data: torch.Tensor,
    dims: str,
    *,
    detection_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert natural padded sequence data to flat+offsets storage."""
    if "s" not in dims:
        raise ValueError("_flatten_padded requires dims with 's'")
    if data.ndim != len(dims):
        raise ValueError(
            f"padded data has {data.ndim}D but dims={dims!r} requires {len(dims)}D"
        )

    batch = data.shape[0]
    seq_dim = dims.index("s")
    seq_len = data.shape[seq_dim]
    expected_mask_shape = (batch, seq_len)

    if attention_mask is not None:
        keep = attention_mask.to(data.device).bool()
    elif detection_mask is not None:
        keep = detection_mask.to(data.device).bool()
    else:
        keep = torch.ones(expected_mask_shape, dtype=torch.bool, device=data.device)

    if detection_mask is None:
        det_bool = keep
    else:
        det_bool = detection_mask.to(data.device).bool()

    if tuple(keep.shape) != expected_mask_shape:
        raise ValueError(
            f"attention_mask must have shape {expected_mask_shape}, got {tuple(keep.shape)}"
        )
    if tuple(det_bool.shape) != expected_mask_shape:
        raise ValueError(
            f"detection_mask must have shape {expected_mask_shape}, got {tuple(det_bool.shape)}"
        )

    lengths = keep.sum(dim=1)

    if attention_mask is None and detection_mask is None:
        if seq_dim == 1:
            # [batch, seq, hidden] -> [batch * seq, hidden]
            flat_data = data.reshape(batch * seq_len, *data.shape[2:])
        elif seq_dim == 2:
            # [batch, layer, seq, hidden] -> [batch * seq, layer, hidden]
            flat_data = data.transpose(1, 2).reshape(
                batch * seq_len,
                data.shape[1],
                *data.shape[3:],
            )
        else:  # pragma: no cover - DIMS constrains this today
            raise ValueError(f"unsupported sequence dimension for dims={dims!r}")
        flat_det = det_bool.reshape(-1)
    elif seq_dim == 1:
        # Vectorized gather: [batch, seq, hidden] -> [total_real_tokens, hidden]
        flat_data = data[keep]
        flat_det = det_bool[keep]
    elif seq_dim == 2:
        # Move layer behind tokens before masking: [batch, seq, layer, hidden].
        flat_data = data.transpose(1, 2)[keep]
        flat_det = det_bool[keep]
    else:  # pragma: no cover - DIMS constrains this today
        raise ValueError(f"unsupported sequence dimension for dims={dims!r}")

    offsets = torch.zeros(batch + 1, dtype=torch.int64, device=data.device)
    offsets[1:] = lengths.to(torch.int64).cumsum(0)
    return flat_data, offsets, flat_det


@dataclass(slots=True)
class Activations:
    """Activation tensor with explicit dimension labels.

    The public constructor accepts natural dense layouts:

    * ``dims="bh"``: ``data`` is ``[batch, hidden]``
    * ``dims="blh"``: ``data`` is ``[batch, layer, hidden]``
    * ``dims="bsh"``: ``data`` is padded ``[batch, seq, hidden]``
    * ``dims="blsh"``: ``data`` is padded ``[batch, layer, seq, hidden]``

    When ``"s"`` is in *dims* the sequence dimension is stored in a
    **flat+offsets** layout:

    * ``data`` – ``[total_tokens, hidden]`` (dims ``"bsh"``) or
      ``[total_tokens, n_layers, hidden]`` (dims ``"blsh"``).  The leading
      axis concatenates all samples; the physical ``ndim`` is therefore
      ``len(dims) - 1`` (the batch dimension is implicit).
    * ``offsets`` – ``[batch+1]`` ``int64`` cumulative token counts.
      Sample *i* spans ``data[offsets[i]:offsets[i+1]]``.
    * ``detection_mask`` – ``[total_tokens]`` ``bool`` per-real-token mask.

    When ``"s"`` is **not** in *dims* the container behaves exactly as
    a dense batch tensor (``offsets=None``, ``detection_mask=None``).

    Args:
        data: Activation tensor
        dims: Dimension format – one of ``"bh"``, ``"bsh"``, ``"blh"``, ``"blsh"``
        offsets: ``[batch+1]`` int64 cumulative token counts (required when ``"s"`` in dims)
        detection_mask: ``[total_tokens]`` bool detection mask (required when ``"s"`` in dims)
            for flat inputs, or ``[batch, seq]`` for padded inputs.
        attention_mask: Optional ``[batch, seq]`` bool/float mask marking real
            tokens for padded sequence inputs.
        layers: Layer indices tuple (required when ``"l"`` in dims)
        metadata: Arbitrary runtime metadata dict for provenance.
    """

    data: torch.Tensor
    dims: str
    offsets: torch.Tensor | None = None
    detection_mask: torch.Tensor | None = None
    layers: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    attention_mask: InitVar[torch.Tensor | None] = None

    def __post_init__(self, attention_mask: torch.Tensor | None):
        self.metadata = _copy_metadata(self.metadata)
        if self.dims not in DIMS:
            raise ValueError(f"dims must be one of {DIMS}, got {self.dims!r}")

        if "s" in self.dims:
            padded_ndim = len(self.dims)
            flat_ndim = len(self.dims) - 1
            if self.offsets is None and self.data.ndim == padded_ndim:
                self.data, self.offsets, self.detection_mask = _flatten_padded(
                    self.data,
                    self.dims,
                    detection_mask=self.detection_mask,
                    attention_mask=attention_mask,
                )
            elif attention_mask is not None:
                raise ValueError("attention_mask is only valid for padded sequence inputs")

            if self.data.ndim != flat_ndim:
                raise ValueError(
                    f"data has {self.data.ndim}D but dims={self.dims!r} "
                    f"requires either padded {padded_ndim}D input or flat+offsets {flat_ndim}D input"
                )
            if self.offsets is None:
                raise ValueError(
                    "offsets required for flat sequence inputs. For padded "
                    f"{self.dims!r} tensors, pass data with {padded_ndim} dimensions."
                )
            if self.detection_mask is None:
                raise ValueError("detection_mask required when dims contains 's'")
            if self.offsets.ndim != 1:
                raise ValueError("offsets must be a 1D tensor")
            if self.offsets.numel() == 0:
                raise ValueError("offsets must contain at least one element")
            self.offsets = self.offsets.to(device=self.data.device, dtype=torch.int64)
            self.detection_mask = self.detection_mask.to(
                device=self.data.device,
                dtype=torch.bool,
            )
            if self.detection_mask.shape != (self.data.shape[0],):
                raise ValueError(
                    "detection_mask must have shape "
                    f"({self.data.shape[0]},), got {tuple(self.detection_mask.shape)}"
                )
            if int(self.offsets[0]) != 0:
                raise ValueError("offsets must start at 0")
            if int(self.offsets[-1]) != self.data.shape[0]:
                raise ValueError(
                    "offsets must end at the number of flat tokens: "
                    f"{int(self.offsets[-1])} != {self.data.shape[0]}"
                )
        else:
            if self.data.ndim != len(self.dims):
                raise ValueError(
                    f"data has {self.data.ndim}D but dims={self.dims!r}"
                )
            if self.offsets is not None:
                raise ValueError("offsets provided but dims has no 's'")
            if self.detection_mask is not None:
                raise ValueError("detection_mask provided but dims has no 's'")

        if "l" in self.dims and self.layers is None:
            raise ValueError("layers required when dims contains 'l'")
        if not self.data.is_floating_point():
            self.data = self.data.float()

    # -------------------------------------------------------------------------
    # Construction helpers
    # -------------------------------------------------------------------------

    @classmethod
    def from_tensor(
        cls,
        data: torch.Tensor,
        *,
        dims: str = "bh",
        layers: tuple[int, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Activations":
        """Build dense Activations without a sequence axis."""
        if "s" in dims:
            raise ValueError(
                "from_tensor does not accept dims with 's'; use Activations(...) "
                "for padded tensors or from_flat(...) for flat+offsets tensors"
            )
        return cls(data=data, dims=dims, layers=layers, metadata=_copy_metadata(metadata))

    @classmethod
    def from_flat(
        cls,
        data: torch.Tensor,
        offsets: torch.Tensor,
        detection_mask: torch.Tensor | None = None,
        *,
        dims: str = "bsh",
        layers: tuple[int, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Activations":
        """Build Activations from a flat token tensor and per-sample offsets."""
        if "s" not in dims:
            raise ValueError("from_flat requires dims with 's'")
        if detection_mask is None:
            detection_mask = torch.ones(data.shape[0], dtype=torch.bool, device=data.device)
        return cls(
            data=data,
            dims=dims,
            offsets=offsets,
            detection_mask=detection_mask,
            layers=layers,
            metadata=_copy_metadata(metadata),
        )

    @classmethod
    def from_padded(
        cls,
        data: torch.Tensor,
        detection_mask: torch.Tensor | None = None,
        *,
        dims: str = "bsh",
        layers: tuple[int, ...] | None = None,
        attention_mask: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Activations":
        """Build flat+offsets Activations from padded rectangular tensors.

        Args:
            data: Padded activation tensor (``[batch, seq, hidden]`` or
                ``[batch, layer, seq, hidden]``).
            detection_mask: Optional ``[batch, seq]`` bool/float mask marking
                detection-relevant tokens. Defaults to ``attention_mask`` when
                provided, otherwise all tokens.
            dims: Dimension string (must contain ``"s"``).
            layers: Layer indices (required when ``"l"`` in dims).
            attention_mask: Optional ``[batch, seq]`` bool/float mask marking
                real tokens. Defaults to all positions.

        Returns:
            Flat+offsets ``Activations``.
        """
        if "s" not in dims:
            raise ValueError("from_padded requires dims with 's'")
        return cls(
            data=data,
            dims=dims,
            detection_mask=detection_mask,
            layers=layers,
            attention_mask=attention_mask,
            metadata=_copy_metadata(metadata),
        )

    @classmethod
    def cat(cls, items: list["Activations"]) -> "Activations":
        """Concatenate multiple Activations along the batch dimension.

        Follows the ``torch.cat`` convention.  Works with all four dim
        combinations (``"bh"``, ``"blh"``, ``"bsh"``, ``"blsh"``).

        For flat+offsets layouts (``"s"`` in dims) the token streams and
        offset arrays are merged so that the result is a single contiguous
        Activations object.

        Args:
            items: List of Activations to concatenate.

        Returns:
            A single Activations containing all samples.

        Raises:
            ValueError: If *items* is empty, or if dims / hidden_size /
                layers are inconsistent across items.
        """
        if not items:
            raise ValueError("cat() requires at least one Activations item")
        if len(items) == 1:
            return items[0]

        first = items[0]

        # Validate consistency
        for i, a in enumerate(items[1:], 1):
            if a.dims != first.dims:
                raise ValueError(
                    f"dims mismatch: items[0] has dims={first.dims!r}, "
                    f"items[{i}] has dims={a.dims!r}"
                )
            if a.hidden_size != first.hidden_size:
                raise ValueError(
                    f"hidden_size mismatch: items[0] has {first.hidden_size}, "
                    f"items[{i}] has {a.hidden_size}"
                )
            if "l" in first.dims and a.layers != first.layers:
                raise ValueError(
                    f"layers mismatch: items[0] has layers={first.layers}, "
                    f"items[{i}] has layers={a.layers}"
                )

        if "s" not in first.dims:
            # Simple path: "bh" or "blh" — just cat along dim 0
            cat_data = torch.cat([a.data for a in items], dim=0)
            return cls(
                data=cat_data,
                dims=first.dims,
                layers=first.layers,
                metadata=_cat_metadata(items),
            )

        # Flat+offsets path: "bsh" or "blsh"
        cat_data = torch.cat([a.data for a in items], dim=0)
        cat_det = torch.cat([a.detection_mask for a in items], dim=0)

        # Merge offsets: shift each item's offsets by cumulative token count
        parts = [items[0].offsets]
        running = int(items[0].offsets[-1].item())
        for a in items[1:]:
            parts.append(a.offsets[1:] + running)
            running += int(a.offsets[-1].item())
        cat_offsets = torch.cat(parts, dim=0)

        return cls(
            data=cat_data,
            dims=first.dims,
            offsets=cat_offsets,
            detection_mask=cat_det,
            layers=first.layers,
            metadata=_cat_metadata(items),
        )

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
                    padded_det[i, :length] = self.detection_mask[s:e]
        else:
            # data: [T, hidden] → padded [batch, max_seq, hidden]
            padded = self.data.new_zeros(batch, max_seq, hidden)
            padded_det = torch.zeros(batch, max_seq, dtype=torch.bool, device=self.data.device)
            for i in range(batch):
                s, e = int(self.offsets[i]), int(self.offsets[i + 1])
                length = e - s
                if length > 0:
                    padded[i, :length] = self.data[s:e]
                    padded_det[i, :length] = self.detection_mask[s:e]

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
                    padded_det[j, :length] = self.detection_mask[s:e]
        else:
            padded = self.data.new_zeros(sub_batch, local_max, hidden)
            padded_det = torch.zeros(sub_batch, local_max, dtype=torch.bool, device=self.data.device)
            for j, i in enumerate(indices):
                s, e = int(self.offsets[i]), int(self.offsets[i + 1])
                length = e - s
                if length > 0:
                    padded[j, :length] = self.data[s:e]
                    padded_det[j, :length] = self.detection_mask[s:e]

        return padded, padded_det

    # -------------------------------------------------------------------------
    # Internal reduction helpers
    # -------------------------------------------------------------------------

    def _reduce_seq(self, pool_fn, **kwargs) -> "Activations":
        """Reduce the sequence dimension using *pool_fn*.

        Handles the flat+offsets layout. Extra *kwargs* are forwarded to
        the pool function (e.g. ``alpha`` for EMA, ``window_size`` for rolling).
        """
        if "s" not in self.dims:
            raise ValueError("No sequence dimension to reduce")

        pooled = pool_fn(self.data, self.detection_mask, offsets=self.offsets, **kwargs)
        new_dims = self.dims.replace("s", "")
        return Activations(
            pooled,
            new_dims,
            offsets=None,
            detection_mask=None,
            layers=self.layers,
            metadata=self.metadata,
        )

    def _reduce_layer(self, reduce_fn) -> "Activations":
        """Reduce the layer dimension using *reduce_fn* (``torch.mean`` or ``torch.max``)."""
        if "l" not in self.dims:
            raise ValueError("No layer dimension to reduce")

        if "s" in self.dims:
            # data: [T, n_layers, hidden] → reduce over dim 1
            dim = 1
        else:
            dim = self.dims.index("l")

        if reduce_fn is torch.max:
            reduced = self.data.max(dim=dim).values
        else:
            reduced = reduce_fn(self.data, dim=dim)

        new_dims = self.dims.replace("l", "")
        return Activations(
            reduced,
            new_dims,
            offsets=self.offsets,
            detection_mask=self.detection_mask,
            layers=None,
            metadata=self.metadata,
        )

    # -------------------------------------------------------------------------
    # Reductions — tinygrad-style dim-passing API
    # -------------------------------------------------------------------------

    def mean(self, dim: str) -> "Activations":
        """Mean reduction over *dim* (``"s"`` for sequence, ``"l"`` for layer)."""
        if dim == "s":
            return self._reduce_seq(P.mean)
        if dim == "l":
            return self._reduce_layer(torch.mean)
        raise ValueError(f"dim must be 's' or 'l', got {dim!r}")

    def max(self, dim: str) -> "Activations":
        """Max reduction over *dim* (``"s"`` for sequence, ``"l"`` for layer)."""
        if dim == "s":
            return self._reduce_seq(P.max)
        if dim == "l":
            return self._reduce_layer(torch.max)
        raise ValueError(f"dim must be 's' or 'l', got {dim!r}")

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    def select(self, dim: str, idx: int | list[int]) -> "Activations":
        """Select along *dim*.

        * ``dim="l"``: layer selection (int removes axis, list keeps it).
        * ``dim="s"``: token selection (int only — removes sequence axis).
        """
        if dim == "l":
            return self._select_layer(idx)
        if dim == "s":
            return self._select_seq(idx)
        raise ValueError(f"dim must be 's' or 'l', got {dim!r}")

    def _select_layer(self, layer_or_layers: int | list[int]) -> "Activations":
        """Select layer(s). Single int removes layer axis."""
        if "l" not in self.dims:
            raise ValueError("No layer axis to select from")

        if "s" in self.dims:
            # data is [T, n_layers, hidden], layer dim = 1
            ax = 1
        else:
            ax = self.dims.index("l")

        if isinstance(layer_or_layers, int):
            pos = self.layers.index(layer_or_layers)
            selected = self.data.select(ax, pos)
            new_dims = self.dims.replace("l", "")
            return Activations(
                selected,
                new_dims,
                offsets=self.offsets,
                detection_mask=self.detection_mask,
                layers=None,
                metadata=self.metadata,
            )
        else:
            indices = [self.layers.index(layer) for layer in layer_or_layers]
            idx_tensor = torch.tensor(indices, device=self.data.device)
            selected = self.data.index_select(ax, idx_tensor)
            return Activations(
                selected,
                self.dims,
                offsets=self.offsets,
                detection_mask=self.detection_mask,
                layers=tuple(layer_or_layers),
                metadata=self.metadata,
            )

    def _select_seq(self, idx: int) -> "Activations":
        """Select a single token per sample from the sequence dimension.

        Negative indices are relative to each sample's end (``-1`` = last token).
        """
        if "s" not in self.dims:
            raise ValueError("No sequence dimension to select from")
        if not isinstance(idx, int):
            raise ValueError("select('s', ...) only supports a single int index")

        if idx == -1:
            # Fast path: reuse existing last-token pooling
            return self._reduce_seq(P.last_token)

        batch = self.batch_size
        extra_dims = self.data.shape[1:]  # e.g. (hidden,) or (n_layers, hidden)
        result = self.data.new_zeros(batch, *extra_dims)

        for i in range(batch):
            s, e = int(self.offsets[i]), int(self.offsets[i + 1])
            length = e - s
            if length == 0:
                continue
            actual_idx = idx if idx >= 0 else length + idx
            if 0 <= actual_idx < length:
                result[i] = self.data[s + actual_idx]

        new_dims = self.dims.replace("s", "")
        return Activations(
            result,
            new_dims,
            offsets=None,
            detection_mask=None,
            layers=self.layers,
            metadata=self.metadata,
        )

    # -------------------------------------------------------------------------
    # Sequence-specific sugar
    # -------------------------------------------------------------------------

    def last(self) -> "Activations":
        """Select the last token per sample (sugar for ``select("s", -1)``)."""
        return self.select("s", -1)

    def ema(self, alpha: float = 0.5) -> "Activations":
        """EMA + max over the sequence dimension."""
        return self._reduce_seq(P.ema, alpha=alpha)

    def rolling(self, window: int = 10) -> "Activations":
        """Rolling mean + max over the sequence dimension."""
        return self._reduce_seq(P.rolling, window_size=window)

    # -------------------------------------------------------------------------
    # Layer-specific
    # -------------------------------------------------------------------------

    def flatten(self) -> "Activations":
        """Concatenate layers into the hidden dimension, removing ``"l"``."""
        if "l" not in self.dims:
            raise ValueError("No layer dimension to flatten")
        # data: [T, n_layers, hidden] or [batch, n_layers, hidden] → [*, n_layers*hidden]
        data = self.data.reshape(self.data.shape[0], -1)
        new_dims = self.dims.replace("l", "")
        return Activations(
            data,
            new_dims,
            offsets=self.offsets,
            detection_mask=self.detection_mask,
            metadata=self.metadata,
        )

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
            yield layer, self.select("l", layer)

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
        det_bool = self.detection_mask.bool()
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

    def to(self, device=None, dtype: torch.dtype | None = None) -> "Activations":
        """Move/cast to *device* and/or *dtype*.

        ``offsets`` and ``detection_mask`` only follow the device move (they stay
        int64/bool); *dtype* applies to the activation ``data`` only.
        """
        data = self.data
        if device is not None:
            data = data.to(device)
        if dtype is not None:
            data = data.to(dtype)
        return Activations(
            data=data,
            dims=self.dims,
            offsets=(
                self.offsets.to(device)
                if (self.offsets is not None and device is not None)
                else self.offsets
            ),
            detection_mask=(
                self.detection_mask.to(device)
                if (self.detection_mask is not None and device is not None)
                else self.detection_mask
            ),
            layers=self.layers,
            metadata=self.metadata,
        )

"""
Activation collection with explicit tokenization.

This module provides tools for extracting activations from language models using hooks,
with support for different model architectures and efficient memory management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Generator, Iterable

import torch

from .. import pool as P
from ..models import HookedModel

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .tokenization import Tokens


class Axis(Enum):
    LAYER = auto()
    BATCH = auto()
    SEQ = auto()
    HIDDEN = auto()


@dataclass(slots=True)
class LayerMeta:
    indices: tuple[int, ...]


@dataclass(slots=True)
class SequenceMeta:
    attention_mask: torch.Tensor
    detection_mask: torch.Tensor
    input_ids: torch.Tensor


_DEFAULT_AXES: tuple[Axis, ...] = (Axis.BATCH, Axis.LAYER, Axis.SEQ, Axis.HIDDEN)


def _ensure_canonical_axes(axes: tuple[Axis, ...]) -> tuple[Axis, ...]:
    allowed = tuple(axis for axis in _DEFAULT_AXES if axis in axes)
    if allowed != axes:
        raise ValueError(
            "axes must be an ordered subset of (Axis.BATCH, Axis.LAYER, Axis.SEQ, Axis.HIDDEN)"
        )
    return axes


@dataclass(slots=True)
class Activations:
    """Axis-aware container for activation tensors and metadata."""

    activations: torch.Tensor
    axes: tuple[Axis, ...] = _DEFAULT_AXES
    layer_meta: LayerMeta | None = None
    sequence_meta: SequenceMeta | None = None
    batch_indices: torch.Tensor | None = None

    _axis_positions: dict[Axis, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.axes = _ensure_canonical_axes(self.axes)
        if self.activations.ndim != len(self.axes):
            raise ValueError("Activation tensor rank and axes metadata disagree")
        if not self.activations.is_floating_point():
            self.activations = self.activations.float()
        self._axis_positions = {axis: idx for idx, axis in enumerate(self.axes)}
        self._validate_layer_meta()
        self._validate_sequence_meta()
        self._validate_batch_indices()

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> torch.Size:
        return self.activations.shape

    @property
    def axis_positions(self) -> dict[Axis, int]:
        return dict(self._axis_positions)

    def has_axis(self, axis: Axis) -> bool:
        return axis in self._axis_positions

    def axis_size(self, axis: Axis) -> int:
        try:
            dim = self._axis_positions[axis]
        except KeyError as exc:
            raise AttributeError(f"Axis {axis.name} has been removed") from exc
        return self.activations.shape[dim]

    @property
    def n_layers(self) -> int:
        return self.axis_size(Axis.LAYER)

    @property
    def batch_size(self) -> int:
        return self.axis_size(Axis.BATCH)

    @property
    def seq_len(self) -> int:
        return self.axis_size(Axis.SEQ)

    @property
    def d_model(self) -> int:
        return self.axis_size(Axis.HIDDEN)

    @property
    def attention_mask(self) -> torch.Tensor | None:
        return None if self.sequence_meta is None else self.sequence_meta.attention_mask

    @property
    def detection_mask(self) -> torch.Tensor | None:
        return None if self.sequence_meta is None else self.sequence_meta.detection_mask

    @property
    def input_ids(self) -> torch.Tensor | None:
        return None if self.sequence_meta is None else self.sequence_meta.input_ids

    @property
    def layer_indices(self) -> list[int]:
        return [] if self.layer_meta is None else list(self.layer_meta.indices)

    @property
    def has_sequences(self) -> bool:
        return Axis.SEQ in self._axis_positions

    @property
    def has_layers(self) -> bool:
        return Axis.LAYER in self._axis_positions

    @classmethod
    def from_tensor(
        cls,
        activations: torch.Tensor,
        *,
        layer_indices: list[int] | None = None,
        attention_mask: torch.Tensor | None = None,
        detection_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        batch_indices: torch.Tensor | None = None,
    ) -> "Activations":
        """Create Activations from a tensor with sensible defaults.

        Supports:
        - 2D [batch, hidden]: Already pooled (no SEQ axis)
        - 3D [batch, seq, hidden]: Single layer with sequence
        - 4D [batch, layer, seq, hidden]: Multiple layers with sequence
        """
        if activations.ndim == 2:
            # [batch, hidden] - already pooled, no SEQ or LAYER axis
            return cls(
                activations=activations,
                axes=(Axis.BATCH, Axis.HIDDEN),
                layer_meta=None,
                sequence_meta=None,
                batch_indices=batch_indices,
            )
        elif activations.ndim == 3:
            batch_size, seq_len, hidden_size = activations.shape
            activations = activations.unsqueeze(1)  # [batch, 1, seq, hidden]
            axes = (Axis.BATCH, Axis.LAYER, Axis.SEQ, Axis.HIDDEN)
            if layer_indices is None:
                layer_indices = [0]
        elif activations.ndim == 4:
            batch_size, n_layers, seq_len, hidden_size = activations.shape
            axes = (Axis.BATCH, Axis.LAYER, Axis.SEQ, Axis.HIDDEN)
            if layer_indices is None:
                layer_indices = list(range(n_layers))
        else:
            raise ValueError(
                f"Expected 2D [batch, hidden], 3D [batch, seq, hidden], "
                f"or 4D [batch, layer, seq, hidden] tensor, got shape {activations.shape}"
            )

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
        if detection_mask is None:
            detection_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
        if input_ids is None:
            input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)

        device = activations.device
        attention_mask = attention_mask.to(device)
        detection_mask = detection_mask.to(device)
        input_ids = input_ids.to(device)

        return cls(
            activations=activations,
            axes=axes,
            layer_meta=LayerMeta(indices=tuple(layer_indices)),
            sequence_meta=SequenceMeta(
                attention_mask=attention_mask,
                detection_mask=detection_mask,
                input_ids=input_ids,
            ),
            batch_indices=batch_indices,
        )

    @classmethod
    def from_hidden_states(
        cls,
        hidden_states: tuple[tuple[torch.Tensor, ...], ...] | tuple[torch.Tensor, ...] | torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        detection_mask: torch.Tensor | None = None,
        layer_indices: list[int] | int | None = None,
        batch_indices: torch.Tensor | None = None,
    ) -> "Activations":
        """Create Activations from hidden states (tensor, tuple of tensors, or nested tuple).

        Input format follows transformer convention: [layer, batch, seq, hidden].
        Output is normalized to [batch, layer, seq, hidden].
        """
        # Normalize layer_indices to list
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]

        # Convert to 4D tensor [layer, batch, seq, hidden] (transformer convention)
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.ndim != 4:
                raise ValueError(f"Expected 4D tensor [layer, batch, seq, hidden], got {hidden_states.shape}")
            tensor = hidden_states
            indices = layer_indices or list(range(tensor.shape[0]))
        elif isinstance(hidden_states, tuple) and len(hidden_states) > 0:
            first = hidden_states[0]
            if isinstance(first, torch.Tensor):
                # Tuple of tensors: one tensor per layer
                indices = layer_indices or list(range(len(hidden_states)))
                tensor = torch.stack([hidden_states[i] for i in indices], dim=0)
            elif isinstance(first, (tuple, list)):
                # Nested tuple: (steps, layers) -> concat steps, stack layers
                num_layers = len(first)
                indices = layer_indices or list(range(num_layers))
                tensor = torch.stack([
                    torch.cat([step[i] for step in hidden_states], dim=1)
                    for i in indices
                ], dim=0)
            else:
                raise TypeError(f"Expected tuple of Tensors or tuples, got tuple of {type(first)}")
        else:
            raise TypeError(f"hidden_states must be tensor or non-empty tuple, got {type(hidden_states)}")

        # Transpose from [layer, batch, seq, hidden] to [batch, layer, seq, hidden]
        tensor = tensor.transpose(0, 1)

        return cls.from_tensor(
            activations=tensor,
            layer_indices=indices,
            attention_mask=attention_mask,
            detection_mask=detection_mask,
            input_ids=input_ids,
            batch_indices=batch_indices,
        )

    @classmethod
    def from_components(
        cls,
        *,
        activations: torch.Tensor,
        layer_indices: Iterable[int] | list[int],
        attention_mask: torch.Tensor,
        detection_mask: torch.Tensor,
        input_ids: torch.Tensor,
        batch_indices: Iterable[int] | torch.Tensor | None = None,
        axes: tuple[Axis, ...] | None = None,
    ) -> "Activations":
        """Create Activations from explicit components."""
        if not isinstance(layer_indices, list):
            layer_indices = list(layer_indices)
        return cls.from_tensor(
            activations=activations,
            layer_indices=layer_indices,
            attention_mask=attention_mask,
            detection_mask=detection_mask,
            input_ids=input_ids,
            batch_indices=batch_indices,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_layer_meta(self) -> None:
        has_layer_axis = Axis.LAYER in self._axis_positions
        if has_layer_axis:
            if self.layer_meta is None:
                raise ValueError("layer_meta is required when the layer axis is present")
            indices = tuple(int(i) for i in self.layer_meta.indices)
            if len(indices) != self.axis_size(Axis.LAYER):
                raise ValueError("layer_meta length must match layer dimension")
            self.layer_meta = LayerMeta(indices)
        elif self.layer_meta is not None:
            raise ValueError("layer_meta must be None after reducing the layer axis")

    def _validate_sequence_meta(self) -> None:
        has_seq_axis = Axis.SEQ in self._axis_positions
        if has_seq_axis:
            if self.sequence_meta is None:
                raise ValueError("sequence_meta is required while the sequence axis is present")
            batch = self.axis_size(Axis.BATCH)
            seq = self.axis_size(Axis.SEQ)

            attn = self.sequence_meta.attention_mask.to(self.activations.device)
            detect = self.sequence_meta.detection_mask.to(self.activations.device)
            ids = self.sequence_meta.input_ids.to(self.activations.device)

            expected = (batch, seq)
            for name, tensor in (("attention_mask", attn), ("detection_mask", detect), ("input_ids", ids)):
                if tensor.shape != expected:
                    raise ValueError(f"{name} shape {tensor.shape} expected {expected}")

            allowed_detect_dtypes = {torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool, torch.int32, torch.int64}
            if detect.dtype not in allowed_detect_dtypes:
                raise ValueError("detection_mask must be float, bool, or int tensor")
            if ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must be an integer tensor")

            self.sequence_meta = SequenceMeta(attention_mask=attn, detection_mask=detect, input_ids=ids)
        elif self.sequence_meta is not None:
            raise ValueError("sequence_meta must be None after reducing the sequence axis")

    def _validate_batch_indices(self) -> None:
        if self.batch_indices is None:
            return
        tensor = torch.as_tensor(self.batch_indices, dtype=torch.long)
        if tensor.ndim != 1:
            raise ValueError("batch_indices must be one-dimensional")
        if Axis.BATCH not in self._axis_positions:
            raise ValueError("batch_indices provided but batch axis is absent")
        if tensor.numel() != self.axis_size(Axis.BATCH):
            raise ValueError("batch_indices length must match batch dimension")
        self.batch_indices = tensor

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def expect_axes(self, *axes: Axis) -> None:
        missing = tuple(axis for axis in axes if axis not in self._axis_positions)
        if missing:
            missing_names = ", ".join(axis.name for axis in missing)
            present_names = ", ".join(axis.name for axis in self.axes)
            raise ValueError(
                f"Expected axes [{missing_names}] to be present, but they are missing.\n"
                f"Current axes: [{present_names}]"
            )

    def get_layer_tensor_indices(self, requested_layers: int | list[int]) -> list[int]:
        """Map requested model layer indices to tensor dimension indices."""
        if isinstance(requested_layers, int):
            requested_layers = [requested_layers]
        self.expect_axes(Axis.LAYER)
        layer_indices = self._require_layer_meta().indices

        tensor_indices: list[int] = []
        for layer in requested_layers:
            try:
                tensor_indices.append(layer_indices.index(layer))
            except ValueError as exc:
                available_layers = ", ".join(map(str, layer_indices))
                raise ValueError(f"Layer {layer} is not available. Available: [{available_layers}]") from exc
        return tensor_indices

    def _require_layer_meta(self) -> LayerMeta:
        if self.layer_meta is None:
            raise ValueError("Layer metadata is unavailable because the layer axis was removed.")
        return self.layer_meta

    def _require_sequence_meta(self) -> SequenceMeta:
        if self.sequence_meta is None:
            raise ValueError("Sequence metadata is unavailable because the sequence axis was removed.")
        return self.sequence_meta

    def to(self, *args, **kwargs) -> "Activations":
        converted = self.activations.to(*args, **kwargs)

        target_device: torch.device | None = None
        if "device" in kwargs:
            target_device = torch.device(kwargs["device"])
        else:
            for arg in args:
                if isinstance(arg, torch.device):
                    target_device = arg
                    break
                if isinstance(arg, str):
                    target_device = torch.device(arg)
                    break

        sequence_meta = self.sequence_meta
        if target_device is not None and sequence_meta is not None:
            sequence_meta = SequenceMeta(
                attention_mask=sequence_meta.attention_mask.to(target_device),
                detection_mask=sequence_meta.detection_mask.to(target_device),
                input_ids=sequence_meta.input_ids.to(target_device),
            )

        return Activations(
            activations=converted,
            axes=self.axes,
            layer_meta=self.layer_meta,
            sequence_meta=sequence_meta,
            batch_indices=self.batch_indices,
        )

    # ------------------------------------------------------------------
    # Save / Load (HDF5)
    # ------------------------------------------------------------------
    def save(self, path: str, compression: str | None = "gzip", compression_opts: int = 4) -> None:
        """Save activations to HDF5 file with compression."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for saving activations. Install with: pip install probelab[storage]")

        acts_cpu = self.activations.cpu().numpy()
        chunk_shape = (1,) + acts_cpu.shape[1:] if self.has_axis(Axis.LAYER) else None

        with h5py.File(path, "w") as f:
            f.create_dataset(
                "activations", data=acts_cpu, compression=compression,
                compression_opts=compression_opts if compression == "gzip" else None, chunks=chunk_shape,
            )
            f.create_dataset("axes", data=[ax.value for ax in self.axes])
            if self.layer_meta is not None:
                f.create_dataset("layer_indices", data=list(self.layer_meta.indices))
            if self.sequence_meta is not None:
                f.create_dataset("attention_mask", data=self.sequence_meta.attention_mask.cpu().numpy(),
                                compression=compression, compression_opts=compression_opts if compression == "gzip" else None)
                f.create_dataset("detection_mask", data=self.sequence_meta.detection_mask.cpu().numpy(),
                                compression=compression, compression_opts=compression_opts if compression == "gzip" else None)
                f.create_dataset("input_ids", data=self.sequence_meta.input_ids.cpu().numpy(),
                                compression=compression, compression_opts=compression_opts if compression == "gzip" else None)
            if self.batch_indices is not None:
                f.create_dataset("batch_indices", data=self.batch_indices.cpu().numpy())
            f.attrs["probelab_version"] = "0.1.0"
            f.attrs["dtype"] = str(self.activations.dtype)

    @classmethod
    def load(cls, path: str, layers: list[int] | None = None, batch_slice: slice | None = None, device: str = "cpu") -> "Activations":
        """Load activations from HDF5 file with optional partial loading."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for loading activations. Install with: pip install probelab[storage]")

        import numpy as np

        with h5py.File(path, "r") as f:
            axes_values = f["axes"][:]
            axes = tuple(Axis(v) for v in axes_values)
            has_layer_axis = Axis.LAYER in axes

            stored_layer_indices: list[int] | None = list(f["layer_indices"][:]) if "layer_indices" in f else None

            if layers is not None and has_layer_axis:
                if stored_layer_indices is None:
                    raise ValueError("Cannot select layers: no layer_indices stored in file")
                tensor_indices = []
                for layer in layers:
                    if layer not in stored_layer_indices:
                        raise ValueError(f"Layer {layer} not found. Available: {stored_layer_indices}")
                    tensor_indices.append(stored_layer_indices.index(layer))
                new_layer_indices = layers
            else:
                tensor_indices = None
                new_layer_indices = stored_layer_indices

            acts_dataset = f["activations"]

            if tensor_indices is not None and batch_slice is not None:
                layer_acts = [acts_dataset[tidx, batch_slice] for tidx in tensor_indices]
                activations = torch.from_numpy(np.stack(layer_acts)).to(device=device, dtype=torch.float32)
            elif tensor_indices is not None:
                layer_acts = [acts_dataset[tidx] for tidx in tensor_indices]
                activations = torch.from_numpy(np.stack(layer_acts)).to(device=device, dtype=torch.float32)
            elif batch_slice is not None:
                if has_layer_axis:
                    activations = torch.tensor(acts_dataset[:, batch_slice], device=device).float()
                else:
                    activations = torch.tensor(acts_dataset[batch_slice], device=device).float()
            else:
                activations = torch.tensor(acts_dataset[:], device=device).float()

            layer_meta = LayerMeta(indices=tuple(new_layer_indices)) if new_layer_indices is not None and has_layer_axis else None

            sequence_meta = None
            if "attention_mask" in f:
                if batch_slice is not None:
                    attention_mask = torch.tensor(f["attention_mask"][batch_slice], device=device)
                    detection_mask = torch.tensor(f["detection_mask"][batch_slice], device=device)
                    input_ids = torch.tensor(f["input_ids"][batch_slice], device=device)
                else:
                    attention_mask = torch.tensor(f["attention_mask"][:], device=device)
                    detection_mask = torch.tensor(f["detection_mask"][:], device=device)
                    input_ids = torch.tensor(f["input_ids"][:], device=device)
                sequence_meta = SequenceMeta(attention_mask=attention_mask.float(), detection_mask=detection_mask.float(), input_ids=input_ids.long())

            batch_indices = None
            if "batch_indices" in f:
                batch_indices = torch.tensor(f["batch_indices"][batch_slice] if batch_slice else f["batch_indices"][:], device=device)

        return cls(activations=activations, axes=axes, layer_meta=layer_meta, sequence_meta=sequence_meta, batch_indices=batch_indices)

    # ------------------------------------------------------------------
    # Axis transforms
    # ------------------------------------------------------------------
    def select(self, *, layer: int | None = None, layers: list[int] | range | None = None) -> "Activations":
        """Select layer(s) from multi-layer activations."""
        if layer is not None and layers is not None:
            raise ValueError("Cannot specify both 'layer' and 'layers'.")
        if layer is None and layers is None:
            raise ValueError("Must specify either 'layer' or 'layers'.")

        if layer is not None:
            tensor_idx = self.get_layer_tensor_indices([layer])[0]
            dim = self._axis_positions[Axis.LAYER]
            selected = self.activations.select(dim, tensor_idx)
            new_axes = tuple(ax for ax in self.axes if ax != Axis.LAYER)
            return Activations(activations=selected, axes=new_axes, layer_meta=None,
                             sequence_meta=self.sequence_meta, batch_indices=self.batch_indices)

        if isinstance(layers, range):
            layers = list(layers)
        if not layers:
            raise ValueError(f"layers must be non-empty. Available: {self.layer_indices}")

        tensor_indices = self.get_layer_tensor_indices(layers)
        dim = self._axis_positions[Axis.LAYER]
        index = torch.as_tensor(tensor_indices, device=self.activations.device, dtype=torch.long)
        subset = torch.index_select(self.activations, dim=dim, index=index)

        return Activations(activations=subset, axes=self.axes, layer_meta=LayerMeta(tuple(layers)),
                         sequence_meta=self.sequence_meta, batch_indices=self.batch_indices)

    def mean_pool(self) -> "Activations":
        """Pool sequence dimension by mean over valid tokens.

        Returns:
            Activations with SEQ axis removed
        """
        seq_meta = self._require_sequence_meta()
        seq_dim = self._axis_positions[Axis.SEQ]
        with torch.no_grad():
            reduced = P.mean(self.activations, seq_meta.detection_mask, dim=seq_dim)
        new_axes = tuple(ax for ax in self.axes if ax != Axis.SEQ)
        return Activations(activations=reduced, axes=new_axes, layer_meta=self.layer_meta,
                           sequence_meta=None, batch_indices=self.batch_indices)

    def last_token(self) -> "Activations":
        """Pool sequence dimension by taking last valid token.

        Returns:
            Activations with SEQ axis removed
        """
        seq_meta = self._require_sequence_meta()
        seq_dim = self._axis_positions[Axis.SEQ]
        with torch.no_grad():
            reduced = P.last_token(self.activations, seq_meta.detection_mask, dim=seq_dim)
        new_axes = tuple(ax for ax in self.axes if ax != Axis.SEQ)
        return Activations(activations=reduced, axes=new_axes, layer_meta=self.layer_meta,
                           sequence_meta=None, batch_indices=self.batch_indices)

    def extract_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract detected tokens for token-level training."""
        self.expect_axes(Axis.BATCH, Axis.SEQ, Axis.HIDDEN)

        if self.has_axis(Axis.LAYER):
            if self.n_layers != 1:
                raise ValueError(f"Token extraction requires single layer, but found {self.n_layers} layers.")
            acts = self.activations.squeeze(self._axis_positions[Axis.LAYER])
        else:
            acts = self.activations

        seq_meta = self._require_sequence_meta()
        mask = seq_meta.detection_mask.bool()
        tokens_per_sample = mask.sum(dim=1)

        if tokens_per_sample.sum() == 0:
            features = torch.empty(0, acts.shape[-1], device=acts.device, dtype=acts.dtype)
        else:
            features = acts[mask]

        return features, tokens_per_sample.to(device=acts.device)



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _batches(
    tokens: "Tokens", batch_size: int
) -> Generator[tuple[dict[str, torch.Tensor], list[int]], None, None]:
    """Yield (batch_dict, indices) sorted by length for efficiency."""
    from .tokenization import Tokens

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
    from .acts import collect

    layer_list = [layers] if isinstance(layers, int) else list(layers)
    acts = collect(
        model,
        tokens,
        layers=layer_list,
        batch_size=batch_size,
        dtype=getattr(model, "dtype", torch.float32),
        pool=pool,
        pool_dim="s",
    )
    return acts.to_activations()

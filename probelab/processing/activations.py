"""Activation collection and axis-aware container."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generator, Iterator, Literal

import torch
from tqdm.auto import tqdm

from ..datasets import Dataset
from ..models import HookedModel
from ..models.architectures import ArchitectureRegistry
from ..types import AggregationMethod, HookPoint
from ..utils.pooling import masked_pool
from .tokenization import tokenize_dialogues

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from ..masks import Mask


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


_DEFAULT_AXES: tuple[Axis, ...] = (Axis.LAYER, Axis.BATCH, Axis.SEQ, Axis.HIDDEN)


@dataclass(slots=True)
class Activations:
    """Axis-aware container for activation tensors."""

    activations: torch.Tensor
    axes: tuple[Axis, ...] = _DEFAULT_AXES
    layer_meta: LayerMeta | None = None
    sequence_meta: SequenceMeta | None = None
    batch_indices: torch.Tensor | None = None
    _axis_positions: dict[Axis, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Validate axes order
        allowed = tuple(ax for ax in _DEFAULT_AXES if ax in self.axes)
        if allowed != self.axes:
            raise ValueError("axes must be ordered subset of (LAYER, BATCH, SEQ, HIDDEN)")
        if self.activations.ndim != len(self.axes):
            raise ValueError(f"Tensor rank {self.activations.ndim} != axes count {len(self.axes)}")
        if not self.activations.is_floating_point():
            self.activations = self.activations.float()

        self._axis_positions = {axis: idx for idx, axis in enumerate(self.axes)}

        # Validate layer_meta
        if Axis.LAYER in self._axis_positions:
            if self.layer_meta is None:
                raise ValueError("layer_meta required when LAYER axis present")
            indices = tuple(int(i) for i in self.layer_meta.indices)
            if len(indices) != self.activations.shape[self._axis_positions[Axis.LAYER]]:
                raise ValueError("layer_meta length must match layer dimension")
            self.layer_meta = LayerMeta(indices)
        elif self.layer_meta is not None:
            raise ValueError("layer_meta must be None when LAYER axis absent")

        # Validate sequence_meta
        if Axis.SEQ in self._axis_positions:
            if self.sequence_meta is None:
                raise ValueError("sequence_meta required when SEQ axis present")
            batch = self.activations.shape[self._axis_positions[Axis.BATCH]]
            seq = self.activations.shape[self._axis_positions[Axis.SEQ]]
            device = self.activations.device
            attn = self.sequence_meta.attention_mask.to(device)
            detect = self.sequence_meta.detection_mask.to(device)
            ids = self.sequence_meta.input_ids.to(device)
            if attn.shape != (batch, seq) or detect.shape != (batch, seq) or ids.shape != (batch, seq):
                raise ValueError(f"Mask shapes must be ({batch}, {seq})")
            self.sequence_meta = SequenceMeta(attention_mask=attn, detection_mask=detect, input_ids=ids)
        elif self.sequence_meta is not None:
            raise ValueError("sequence_meta must be None when SEQ axis absent")

        # Validate batch_indices
        if self.batch_indices is not None:
            self.batch_indices = torch.as_tensor(self.batch_indices, dtype=torch.long)
            if self.batch_indices.ndim != 1:
                raise ValueError("batch_indices must be 1D")
            if Axis.BATCH in self._axis_positions:
                if self.batch_indices.numel() != self.activations.shape[self._axis_positions[Axis.BATCH]]:
                    raise ValueError("batch_indices length must match batch dimension")

    # Properties
    @property
    def shape(self) -> torch.Size:
        return self.activations.shape

    @property
    def axis_positions(self) -> dict[Axis, int]:
        return dict(self._axis_positions)

    def has_axis(self, axis: Axis) -> bool:
        return axis in self._axis_positions

    def axis_size(self, axis: Axis) -> int:
        if axis not in self._axis_positions:
            raise AttributeError(f"Axis {axis.name} not present")
        return self.activations.shape[self._axis_positions[axis]]

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

    # Factory methods
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
        """Create from 3D [batch, seq, hidden] or 4D [layer, batch, seq, hidden] tensor."""
        if activations.ndim == 3:
            batch_size, seq_len, _ = activations.shape
            activations = activations.unsqueeze(0)
            layer_indices = layer_indices or [0]
        elif activations.ndim == 4:
            _, batch_size, seq_len, _ = activations.shape
            layer_indices = layer_indices or list(range(activations.shape[0]))
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {activations.shape}")

        device = activations.device
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
        if detection_mask is None:
            detection_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
        if input_ids is None:
            input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

        return cls(
            activations=activations,
            axes=_DEFAULT_AXES,
            layer_meta=LayerMeta(indices=tuple(layer_indices)),
            sequence_meta=SequenceMeta(
                attention_mask=attention_mask.to(device),
                detection_mask=detection_mask.to(device),
                input_ids=input_ids.to(device),
            ),
            batch_indices=batch_indices,
        )

    @classmethod
    def from_hidden_states(
        cls,
        hidden_states: tuple[torch.Tensor, ...] | torch.Tensor,
        *,
        layer_indices: list[int] | None = None,
        attention_mask: torch.Tensor | None = None,
        detection_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        batch_indices: torch.Tensor | None = None,
    ) -> "Activations":
        """Create from HuggingFace hidden_states tuple or stacked tensor."""
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.ndim != 4:
                raise ValueError(f"Expected 4D tensor, got shape {hidden_states.shape}")
            return cls.from_tensor(
                hidden_states, layer_indices=layer_indices, attention_mask=attention_mask,
                detection_mask=detection_mask, input_ids=input_ids, batch_indices=batch_indices,
            )

        if not hidden_states:
            raise ValueError("Empty hidden_states")

        first = hidden_states[0]
        if isinstance(first, torch.Tensor):
            # Tuple of tensors: (layer0, layer1, ...)
            layers = layer_indices or list(range(len(hidden_states)))
            stacked = torch.stack([hidden_states[i] for i in layers])
        elif isinstance(first, (tuple, list)):
            # Nested: ((step0_layer0, step0_layer1), (step1_layer0, step1_layer1), ...)
            n_steps, n_layers = len(hidden_states), len(first)
            layers = layer_indices or list(range(n_layers))
            layer_tensors = []
            for layer_idx in layers:
                step_tensors = [hidden_states[step][layer_idx] for step in range(n_steps)]
                layer_tensors.append(torch.cat(step_tensors, dim=1))
            stacked = torch.stack(layer_tensors)
        else:
            raise TypeError(f"Expected tuple of Tensors, got {type(first)}")

        return cls.from_tensor(
            stacked, layer_indices=layers, attention_mask=attention_mask,
            detection_mask=detection_mask, input_ids=input_ids, batch_indices=batch_indices,
        )

    # Axis helpers
    def expect_axes(self, *axes: Axis) -> None:
        missing = [ax for ax in axes if ax not in self._axis_positions]
        if missing:
            raise ValueError(f"Missing axes: {[ax.name for ax in missing]}. Present: {[ax.name for ax in self.axes]}")

    def get_layer_tensor_indices(self, requested: int | list[int]) -> list[int]:
        """Map model layer indices to tensor dimension indices."""
        if isinstance(requested, int):
            requested = [requested]
        self.expect_axes(Axis.LAYER)
        indices = self.layer_meta.indices  # type: ignore
        result = []
        for layer in requested:
            if layer not in indices:
                raise ValueError(f"Layer {layer} not available. Available: {list(indices)}")
            result.append(indices.index(layer))
        return result

    def to(self, *args, **kwargs) -> "Activations":
        converted = self.activations.to(*args, **kwargs)
        target_device = None
        if "device" in kwargs:
            target_device = torch.device(kwargs["device"])
        else:
            for arg in args:
                if isinstance(arg, (torch.device, str)):
                    target_device = torch.device(arg) if isinstance(arg, str) else arg
                    break

        seq_meta = self.sequence_meta
        if target_device and seq_meta:
            seq_meta = SequenceMeta(
                attention_mask=seq_meta.attention_mask.to(target_device),
                detection_mask=seq_meta.detection_mask.to(target_device),
                input_ids=seq_meta.input_ids.to(target_device),
            )
        return Activations(converted, self.axes, self.layer_meta, seq_meta, self.batch_indices)

    # Save/Load
    def save(self, path: str, compression: str | None = "gzip", compression_opts: int = 4) -> None:
        """Save to HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required: pip install probelab[storage]")

        opts = {"compression": compression, "compression_opts": compression_opts} if compression == "gzip" else {"compression": compression} if compression else {}
        with h5py.File(path, "w") as f:
            f.create_dataset("activations", data=self.activations.cpu().numpy(), chunks=(1,) + self.activations.shape[1:] if self.has_axis(Axis.LAYER) else None, **opts)
            f.create_dataset("axes", data=[ax.value for ax in self.axes])
            if self.layer_meta:
                f.create_dataset("layer_indices", data=list(self.layer_meta.indices))
            if self.sequence_meta:
                f.create_dataset("attention_mask", data=self.sequence_meta.attention_mask.cpu().numpy(), **opts)
                f.create_dataset("detection_mask", data=self.sequence_meta.detection_mask.cpu().numpy(), **opts)
                f.create_dataset("input_ids", data=self.sequence_meta.input_ids.cpu().numpy(), **opts)
            if self.batch_indices is not None:
                f.create_dataset("batch_indices", data=self.batch_indices.cpu().numpy())
            f.attrs["probelab_version"] = "0.1.0"

    @classmethod
    def load(cls, path: str, layers: list[int] | None = None, batch_slice: slice | None = None, device: str = "cpu") -> "Activations":
        """Load from HDF5 file with optional partial loading."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required: pip install probelab[storage]")
        import numpy as np

        with h5py.File(path, "r") as f:
            axes = tuple(Axis(v) for v in f["axes"][:])
            has_layer = Axis.LAYER in axes
            stored_layers = list(f["layer_indices"][:]) if "layer_indices" in f else None

            if layers and has_layer:
                if stored_layers is None:
                    raise ValueError("Cannot select layers: no layer_indices in file")
                tensor_indices = []
                for layer in layers:
                    if layer not in stored_layers:
                        raise ValueError(f"Layer {layer} not found. Available: {stored_layers}")
                    tensor_indices.append(stored_layers.index(layer))
                new_layers = layers
            else:
                tensor_indices = None
                new_layers = stored_layers

            ds = f["activations"]
            if tensor_indices and batch_slice:
                data = np.stack([ds[i, batch_slice] for i in tensor_indices])
            elif tensor_indices:
                data = np.stack([ds[i] for i in tensor_indices])
            elif batch_slice:
                data = ds[:, batch_slice] if has_layer else ds[batch_slice]
            else:
                data = ds[:]
            activations = torch.tensor(data, device=device, dtype=torch.float32)

            layer_meta = LayerMeta(tuple(new_layers)) if new_layers and has_layer else None
            seq_meta = None
            if "attention_mask" in f:
                sl = batch_slice or slice(None)
                seq_meta = SequenceMeta(
                    attention_mask=torch.tensor(f["attention_mask"][sl], device=device).float(),
                    detection_mask=torch.tensor(f["detection_mask"][sl], device=device).float(),
                    input_ids=torch.tensor(f["input_ids"][sl], device=device).long(),
                )
            batch_indices = torch.tensor(f["batch_indices"][batch_slice or slice(None)], device=device) if "batch_indices" in f else None

        return cls(activations, axes, layer_meta, seq_meta, batch_indices)

    # Axis transforms
    def select(self, *, layer: int | None = None, layers: list[int] | range | None = None) -> "Activations":
        """Select layer(s). Single layer removes LAYER axis."""
        if (layer is None) == (layers is None):
            raise ValueError("Specify exactly one of 'layer' or 'layers'")

        if layer is not None:
            idx = self.get_layer_tensor_indices([layer])[0]
            dim = self._axis_positions[Axis.LAYER]
            selected = self.activations.select(dim, idx)
            new_axes = tuple(ax for ax in self.axes if ax != Axis.LAYER)
            return Activations(selected, new_axes, None, self.sequence_meta, self.batch_indices)

        layers_list = list(layers) if isinstance(layers, range) else layers  # type: ignore
        if not layers_list:
            raise ValueError(f"layers must be non-empty. Available: {self.layer_indices}")
        tensor_indices = self.get_layer_tensor_indices(layers_list)
        dim = self._axis_positions[Axis.LAYER]
        index = torch.as_tensor(tensor_indices, device=self.activations.device, dtype=torch.long)
        subset = torch.index_select(self.activations, dim, index)
        return Activations(subset, self.axes, LayerMeta(tuple(layers_list)), self.sequence_meta, self.batch_indices)

    def pool(self, dim: Literal["sequence", "seq", "layer"] = "sequence",
             method: AggregationMethod | str = "mean", use_detection_mask: bool = True) -> "Activations":
        """Pool over dimension, removing that axis."""
        if isinstance(method, str):
            method = AggregationMethod(method)

        if dim in ("sequence", "seq"):
            self.expect_axes(Axis.SEQ)
            if use_detection_mask:
                seq_meta = self.sequence_meta
                assert seq_meta is not None
                reduced = masked_pool(self.activations, seq_meta.detection_mask, method,
                                       seq_dim=self._axis_positions[Axis.SEQ], batch_dim=self._axis_positions[Axis.BATCH])
            else:
                dim_idx = self._axis_positions[Axis.SEQ]
                if method == AggregationMethod.MEAN:
                    reduced = self.activations.mean(dim=dim_idx)
                elif method == AggregationMethod.MAX:
                    reduced = self.activations.max(dim=dim_idx).values
                else:
                    reduced = self.activations.select(dim_idx, -1)
            new_axes = tuple(ax for ax in self.axes if ax != Axis.SEQ)
            return Activations(reduced, new_axes, self.layer_meta, None, self.batch_indices)

        elif dim == "layer":
            self.expect_axes(Axis.LAYER)
            if method == AggregationMethod.LAST_TOKEN:
                raise ValueError("last_token not supported for layer pooling")
            dim_idx = self._axis_positions[Axis.LAYER]
            reduced = self.activations.mean(dim=dim_idx) if method == AggregationMethod.MEAN else self.activations.max(dim=dim_idx).values
            new_axes = tuple(ax for ax in self.axes if ax != Axis.LAYER)
            return Activations(reduced, new_axes, None, self.sequence_meta, self.batch_indices)
        else:
            raise ValueError(f"Unknown dim: {dim}")

    def extract_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract detected tokens. Returns (features, tokens_per_sample)."""
        self.expect_axes(Axis.BATCH, Axis.SEQ, Axis.HIDDEN)
        acts = self.activations
        if self.has_axis(Axis.LAYER):
            if self.n_layers != 1:
                raise ValueError(f"extract_tokens requires single layer, got {self.n_layers}")
            acts = acts.squeeze(self._axis_positions[Axis.LAYER])

        seq_meta = self.sequence_meta
        assert seq_meta is not None
        mask = seq_meta.detection_mask.bool()
        tokens_per_sample = mask.sum(dim=1)
        features = acts[mask] if tokens_per_sample.sum() > 0 else torch.empty(0, acts.shape[-1], device=acts.device, dtype=acts.dtype)
        return features, tokens_per_sample.to(acts.device)


# Batch iteration helpers
def get_batches(inputs: dict[str, torch.Tensor], batch_size: int, tokenizer: "PreTrainedTokenizerBase") -> Iterator[tuple[dict[str, torch.Tensor], list[int]]]:
    """Yield length-sorted batches with original indices."""
    seq_lengths = inputs["attention_mask"].sum(dim=1)
    sorted_indices = torch.sort(seq_lengths, descending=True)[1]
    n = sorted_indices.numel()

    for start in range(0, n, batch_size):
        batch_idx = sorted_indices[start:min(start + batch_size, n)]
        batch_len = int(seq_lengths[batch_idx].max().item())
        if tokenizer.padding_side == "right":
            batch = {k: v[batch_idx][..., :batch_len] for k, v in inputs.items()}
        else:
            batch = {k: v[batch_idx][..., -batch_len:] for k, v in inputs.items()}
        yield batch, batch_idx.tolist()


def _iter_batches(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizerBase", inputs: dict[str, torch.Tensor],
                  layers: list[int], batch_size: int, hook_point: HookPoint, detach: bool) -> Generator[tuple[torch.Tensor, list[int], dict[str, Any]], None, None]:
    """Core generator yielding (activations, indices, meta) per batch."""
    with HookedModel(model, layers, detach_activations=detach, hook_point=hook_point) as hooked:
        for batch, indices in get_batches(inputs, batch_size, tokenizer):
            if batch["input_ids"].device != model.device:
                batch = {k: v.to(model.device) for k, v in batch.items()}
            acts = hooked.get_activations(batch).cpu()
            yield acts, indices, {"attention_mask": batch["attention_mask"].cpu(), "detection_mask": batch["detection_mask"].cpu(),
                                  "input_ids": batch["input_ids"].cpu(), "seq_len": batch["input_ids"].shape[1]}


def _materialize(gen, n_samples: int, max_seq: int, hidden_dim: int, n_layers: int, dtype: torch.dtype,
                 tokenizer: "PreTrainedTokenizerBase", pool: AggregationMethod | None = None) -> torch.Tensor:
    """Materialize generator into single tensor."""
    if pool:
        out = torch.zeros(n_layers, n_samples, hidden_dim, dtype=dtype)
        for acts, indices, meta in gen:
            out[:, indices] = masked_pool(acts, meta["detection_mask"], pool, seq_dim=2, batch_dim=1)
    else:
        out = torch.zeros(n_layers, n_samples, max_seq, hidden_dim, dtype=dtype)
        for acts, indices, meta in gen:
            sl = meta["seq_len"]
            if tokenizer.padding_side == "right":
                out[:, indices, :sl] = acts
            else:
                out[:, indices, -sl:] = acts
    return out


# Streaming iterator
class ActivationIterator:
    """Regenerable iterator for streaming activations."""

    __slots__ = ("_model", "_tokenizer", "_inputs", "_layers", "_batch_size", "_verbose", "_num_batches", "_hook_point", "_detach")

    def __init__(self, model, tokenizer, inputs, layers, batch_size, verbose, num_batches, hook_point=HookPoint.POST_BLOCK, detach=True):
        self._model, self._tokenizer, self._inputs = model, tokenizer, inputs
        self._layers, self._batch_size, self._num_batches = layers, batch_size, num_batches
        self._hook_point, self._detach = hook_point, detach
        self._verbose = verbose if verbose is not None else True

    @property
    def layers(self) -> list[int]:
        return self._layers

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[Activations]:
        gen = _iter_batches(self._model, self._tokenizer, self._inputs, self._layers, self._batch_size, self._hook_point, self._detach)
        if self._verbose:
            gen = tqdm(gen, desc="Collecting activations", total=self._num_batches)
        for acts, indices, meta in gen:
            yield Activations(acts, _DEFAULT_AXES, LayerMeta(tuple(self._layers)),
                              SequenceMeta(meta["attention_mask"], meta["detection_mask"], meta["input_ids"]),
                              torch.as_tensor(indices, dtype=torch.long))


# Public API
def get_n_layers(model: "PreTrainedModel") -> int:
    return ArchitectureRegistry.get_architecture(model).get_num_layers(model)


def get_hidden_dim(model: "PreTrainedModel") -> int:
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return cfg.text_config.hidden_size
    raise ValueError(f"Cannot determine hidden_size for {model.name_or_path}")


def collect_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    dataset: Dataset,
    *,
    layers: int | list[int],
    mask: "Mask",
    batch_size: int = 32,
    streaming: bool = False,
    collection_strategy: Literal["mean", "max", "last_token"] | None = None,
    hook_point: HookPoint = HookPoint.POST_BLOCK,
    add_generation_prompt: bool = False,
    detach_activations: bool = True,
    verbose: bool | None = None,
) -> Activations | ActivationIterator:
    """Collect activations from dataset. Returns ActivationIterator if streaming=True."""
    layers_list = [layers] if isinstance(layers, int) else list(layers)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized = tokenize_dialogues(tokenizer, dataset.dialogues, mask, "cpu", add_generation_prompt, return_tensors="pt", padding=True)
    n_samples, max_seq = tokenized["input_ids"].shape
    num_batches = (n_samples + batch_size - 1) // batch_size

    if streaming:
        return ActivationIterator(model, tokenizer, tokenized, layers_list, batch_size, verbose, num_batches, hook_point, detach_activations)

    show_progress = verbose if verbose is not None else True

    pool = AggregationMethod(collection_strategy) if collection_strategy else None
    gen = _iter_batches(model, tokenizer, tokenized, layers_list, batch_size, hook_point, detach_activations)

    if show_progress:
        gen = tqdm(gen, desc=f"Collecting ({pool.value} pooled)" if pool else "Collecting activations", total=num_batches)

    data = _materialize(gen, n_samples, max_seq, get_hidden_dim(model), len(layers_list), model.dtype, tokenizer, pool)

    if pool:
        return Activations(data, (Axis.LAYER, Axis.BATCH, Axis.HIDDEN), LayerMeta(tuple(layers_list)), None, torch.arange(n_samples))
    return Activations(data, _DEFAULT_AXES, LayerMeta(tuple(layers_list)),
                       SequenceMeta(tokenized["attention_mask"], tokenized["detection_mask"], tokenized["input_ids"]), torch.arange(n_samples))

"""
Simplified activation collection using generators for cleaner API.

This module provides tools for extracting activations from language models using hooks,
with support for different model architectures and efficient memory management.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    overload,
)

import torch
from jaxtyping import Float
from tqdm.auto import tqdm

from ..datasets import DialogueDataset
from ..models import HookedModel
from ..models.architectures import ArchitectureRegistry
from ..types import AggregationMethod, Dialogue, HookPoint
from ..utils.pooling import masked_pool
from .tokenization import tokenize_dialogues

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from ..masks import MaskFunction


def _ensure_on_device(
    batch_inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move batch inputs to device if not already there.

    Args:
        batch_inputs: Dictionary of input tensors
        device: Target device

    Returns:
        Dictionary with tensors on the target device
    """
    if batch_inputs["input_ids"].device == device:
        return batch_inputs
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch_inputs.items()
    }


def get_batches(
    inputs: dict[str, torch.Tensor],
    batch_size: int,
    tokenizer: "PreTrainedTokenizerBase",
) -> Iterator[tuple[dict[str, torch.Tensor], list[int]]]:
    """Yield length-aware batches while preserving original indices.

    Sequences are sorted by non-padding length (descending) so each batch shares
    a similar sequence length. This minimizes padding, keeps GPU transfers tight,
    and processes longest sequences first to reduce memory fragmentation.

    Args:
        inputs: Tokenized inputs keyed by field name.
        batch_size: Maximum number of sequences per batch.
        tokenizer: Provides padding semantics used to trim left/right padding.
    """
    seq_lengths = inputs["attention_mask"].sum(dim=1)
    sorted_indices = torch.sort(seq_lengths, descending=True)[1]

    num_samples = sorted_indices.numel()
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_indices_tensor = sorted_indices[start:end]
        batch_indices = batch_indices_tensor.tolist()

        batch_lengths = seq_lengths.index_select(0, batch_indices_tensor)
        batch_length = int(batch_lengths.max().item())

        if tokenizer.padding_side == "right":
            batch_inputs = {
                key: tensor.index_select(0, batch_indices_tensor)[..., :batch_length]
                for key, tensor in inputs.items()
            }
        elif tokenizer.padding_side == "left":
            batch_inputs = {
                key: tensor.index_select(0, batch_indices_tensor)[..., -batch_length:]
                for key, tensor in inputs.items()
            }
        else:
            raise ValueError(f"Unknown padding side: {tokenizer.padding_side}")

        yield batch_inputs, batch_indices


def get_n_layers(model: "PreTrainedModel") -> int:
    """Get number of layers in the model using the architecture registry."""
    architecture = ArchitectureRegistry.get_architecture(model)
    return architecture.get_num_layers(model)


def get_hidden_dim(model: "PreTrainedModel") -> int:
    """Get hidden dimension of the model."""
    config = model.config

    if hasattr(config, "hidden_size"):
        return config.hidden_size  # type: ignore
    elif hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        # Multimodal models like Gemma3 have text_config.hidden_size
        return config.text_config.hidden_size  # type: ignore
    else:
        raise ValueError(f"Cannot determine hidden dimension for {model.name_or_path}")


def _resolve_verbose(verbose: bool | None) -> bool:
    """Resolve verbose parameter using environment variable defaults.

    Args:
        verbose: Explicit verbose setting, or None to use env defaults.

    Returns:
        Resolved verbose flag (False if progress is disabled).
    """
    if verbose is None:
        verbose = os.environ.get("PROBELAB_VERBOSE", "true").lower() != "false"
    if os.environ.get("PROBELAB_DISABLE_PROGRESS", "").lower() in ("1", "true", "yes"):
        verbose = False
    return verbose


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


_DEFAULT_AXES: tuple[Axis, ...] = (
    Axis.LAYER,
    Axis.BATCH,
    Axis.SEQ,
    Axis.HIDDEN,
)


def _ensure_canonical_axes(axes: tuple[Axis, ...]) -> tuple[Axis, ...]:
    allowed = tuple(axis for axis in _DEFAULT_AXES if axis in axes)
    if allowed != axes:
        raise ValueError(
            "axes must be an ordered subset of (Axis.LAYER, Axis.BATCH, Axis.SEQ, Axis.HIDDEN)"
        )
    return axes


@dataclass(slots=True)
class Activations:
    """Axis-aware container for activation tensors and metadata."""

    activations: Float[torch.Tensor, "..."]
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
        """Check if sequence dimension exists."""
        return Axis.SEQ in self._axis_positions

    @property
    def has_layers(self) -> bool:
        """Check if layer dimension exists."""
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

        This is a convenience method for simple cases. It automatically:
        - Detects axes from tensor shape
        - Creates default masks if not provided
        - Handles layer metadata

        Args:
            activations: Activation tensor with shape:
                - [batch, seq, hidden] for single layer
                - [layer, batch, seq, hidden] for multiple layers
            layer_indices: Which layers these activations come from.
                          Defaults to [0] for single layer or range for multiple.
            attention_mask: Optional attention mask [batch, seq].
                           Defaults to all 1s.
            detection_mask: Optional detection mask [batch, seq].
                           Defaults to all 1s.
            input_ids: Optional input token IDs [batch, seq].
                      Defaults to all 1s.
            batch_indices: Optional batch indices for streaming.

        Returns:
            Activations object ready to use.

        Examples:
            # Simple case - single layer
            acts = torch.randn(4, 10, 768)  # [batch, seq, hidden]
            activations = Activations.from_tensor(acts)

            # With custom masks
            acts = torch.randn(4, 10, 768)
            mask = torch.ones(4, 10)
            mask[:, 5:] = 0  # Mask out later tokens
            activations = Activations.from_tensor(
                acts,
                attention_mask=mask,
                detection_mask=mask
            )

            # Multiple layers
            acts = torch.randn(12, 4, 10, 768)  # [layer, batch, seq, hidden]
            activations = Activations.from_tensor(acts, layer_indices=list(range(12)))
        """
        if activations.ndim == 3:
            # [batch, seq, hidden] - single layer
            batch_size, seq_len, hidden_size = activations.shape
            axes = (Axis.BATCH, Axis.SEQ, Axis.HIDDEN)

            activations = activations.unsqueeze(0)  # [1, batch, seq, hidden]
            axes = (Axis.LAYER, Axis.BATCH, Axis.SEQ, Axis.HIDDEN)

            if layer_indices is None:
                layer_indices = [0]

        elif activations.ndim == 4:
            # [layer, batch, seq, hidden] - multiple layers
            n_layers, batch_size, seq_len, hidden_size = activations.shape
            axes = (Axis.LAYER, Axis.BATCH, Axis.SEQ, Axis.HIDDEN)

            if layer_indices is None:
                layer_indices = list(range(n_layers))
        else:
            raise ValueError(
                f"Expected 3D [batch, seq, hidden] or 4D [layer, batch, seq, hidden] tensor, "
                f"got shape {activations.shape}"
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

        layer_meta = LayerMeta(indices=tuple(layer_indices))
        sequence_meta = SequenceMeta(
            attention_mask=attention_mask,
            detection_mask=detection_mask,
            input_ids=input_ids,
        )

        return cls(
            activations=activations,
            axes=axes,
            layer_meta=layer_meta,
            sequence_meta=sequence_meta,
            batch_indices=batch_indices,
        )

    @classmethod
    def from_hidden_states(
        cls,
        hidden_states: tuple[tuple[torch.Tensor, ...], ...] | torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        detection_mask: torch.Tensor | None = None,
        layer_indices: list[int] | None = None,
        batch_indices: torch.Tensor | None = None,
    ) -> "Activations":
        """Create Activations from pre-extracted hidden states.

        This is designed for integration with external libraries (like inspect_ai or
        control-arena) that return hidden states from model calls. It handles three formats:
        standard HuggingFace, generation (nested tuples), and pre-stacked tensors.

        Args:
            hidden_states: Hidden states in one of three formats:
                - **Standard HF format** (most common): tuple[Tensor, ...] where each tensor
                  is a layer with shape [batch, seq, hidden].
                  Example: `outputs.hidden_states` from `model(..., output_hidden_states=True)`
                - **Generation format**: tuple[tuple[Tensor, ...], ...] where outer tuple
                  is generation steps, inner tuple is layers. Each tensor is [batch, 1, hidden].
                  Example: ((layer0_step0, layer1_step0), (layer0_step1, layer1_step1))
                - **Stacked format**: Tensor of shape [layer, batch, seq, hidden]
            input_ids: Token IDs [batch, seq] (optional, inferred from shapes)
            attention_mask: Attention mask [batch, seq] (optional, defaults to all ones)
            detection_mask: Detection mask [batch, seq] (optional, defaults to all ones)
            layer_indices: Which layers to use (optional). For standard/generation format,
                selects specific layers. For stacked format, indicates layer IDs.
            batch_indices: Batch indices for streaming (optional)

        Returns:
            Activations object ready for probe inference

        Examples:
            # Standard HuggingFace format (simplest)
            >>> outputs = model(input_ids, output_hidden_states=True)
            >>> acts = Activations.from_hidden_states(
            ...     outputs.hidden_states,
            ...     layer_indices=[16],  # Select layer 16
            ... )

            # With detection mask
            >>> acts = Activations.from_hidden_states(
            ...     outputs.hidden_states,
            ...     layer_indices=[16],
            ...     detection_mask=my_mask  # [batch, seq]
            ... )

            # From pre-stacked tensor
            >>> stacked = torch.randn(32, 1, 128, 4096)  # [layers, batch, seq, hidden]
            >>> acts = Activations.from_hidden_states(stacked)

        Raises:
            ValueError: If hidden_states format is invalid or empty
            TypeError: If hidden_states is not tuple or Tensor
        """
        if isinstance(hidden_states, tuple):
            if len(hidden_states) == 0:
                raise ValueError("Empty hidden_states tuple")

            first_elem = hidden_states[0]

            if isinstance(first_elem, torch.Tensor):
                # Standard HF format: (layer_0, layer_1, ..., layer_N), each [batch, seq, hidden]
                if layer_indices is not None:
                    if not isinstance(layer_indices, (list, tuple)):
                        layer_indices_to_select = [layer_indices]
                    else:
                        layer_indices_to_select = list(layer_indices)
                    selected_layers = [
                        hidden_states[idx] for idx in layer_indices_to_select
                    ]
                    actual_layer_indices = layer_indices_to_select
                else:
                    selected_layers = list(hidden_states)
                    actual_layer_indices = list(range(len(hidden_states)))

                activations_tensor = torch.stack(
                    selected_layers, dim=0
                )  # [layer, batch, seq, hidden]

            elif isinstance(first_elem, (tuple, list)):
                # Generation format: ((step0_layer0, ...), (step1_layer0, ...)), each tensor [batch, 1, hidden]
                num_steps = len(hidden_states)
                num_layers = len(hidden_states[0])

                if num_layers == 0:
                    raise ValueError("Empty layer tuple in hidden_states")

                # Verify all steps have same number of layers
                for step_idx, step_tuple in enumerate(hidden_states):
                    if len(step_tuple) != num_layers:
                        raise ValueError(
                            f"Inconsistent layer count: step 0 has {num_layers} layers, "
                            f"step {step_idx} has {len(step_tuple)} layers"
                        )

                # Determine which layers to use
                if layer_indices is not None:
                    if not isinstance(layer_indices, (list, tuple)):
                        selected_layer_indices = [layer_indices]
                    else:
                        selected_layer_indices = list(layer_indices)
                    actual_layer_indices = selected_layer_indices
                else:
                    selected_layer_indices = list(range(num_layers))
                    actual_layer_indices = selected_layer_indices

                layer_tensors = []
                for layer_idx in selected_layer_indices:
                    step_tensors = [
                        hidden_states[step_idx][layer_idx]
                        for step_idx in range(num_steps)
                    ]
                    layer_tensor = torch.cat(
                        step_tensors, dim=1
                    )  # [batch, total_seq, hidden]
                    layer_tensors.append(layer_tensor)

                activations_tensor = torch.stack(
                    layer_tensors, dim=0
                )  # [layer, batch, seq, hidden]
            else:
                raise TypeError(
                    f"Expected tuple of Tensors or tuple of tuples, "
                    f"got tuple of {type(first_elem)}"
                )

        elif isinstance(hidden_states, torch.Tensor):
            activations_tensor = hidden_states
            if activations_tensor.ndim != 4:
                raise ValueError(
                    f"Expected 4D tensor [layer, batch, seq, hidden], "
                    f"got shape {activations_tensor.shape}"
                )
            actual_layer_indices = layer_indices
        else:
            raise TypeError(
                f"hidden_states must be tuple or Tensor, got {type(hidden_states)}"
            )

        # Use existing from_tensor to handle the rest
        return cls.from_tensor(
            activations=activations_tensor,
            layer_indices=actual_layer_indices,
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
        """Create Activations from explicit components.

        This is an alias for from_tensor() with all parameters explicitly provided.
        It exists for backward compatibility and explicitness when you have all
        components ready.

        Args:
            activations: Activation tensor (should be 4D: [layer, batch, seq, hidden])
            layer_indices: Layer indices for the activations
            attention_mask: Attention mask [batch, seq]
            detection_mask: Detection mask [batch, seq]
            input_ids: Input IDs [batch, seq]
            batch_indices: Optional batch indices for streaming
            axes: Optional custom axes (defaults to standard ordering)

        Returns:
            Activations object
        """
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
                raise ValueError(
                    "layer_meta is required when the layer axis is present"
                )
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
                raise ValueError(
                    "sequence_meta is required while the sequence axis is present"
                )
            batch = self.axis_size(Axis.BATCH)
            seq = self.axis_size(Axis.SEQ)

            attn = self.sequence_meta.attention_mask.to(self.activations.device)
            detect = self.sequence_meta.detection_mask.to(self.activations.device)
            ids = self.sequence_meta.input_ids.to(self.activations.device)

            expected = (batch, seq)
            for name, tensor in (
                ("attention_mask", attn),
                ("detection_mask", detect),
                ("input_ids", ids),
            ):
                if tensor.shape != expected:
                    raise ValueError(f"{name} shape {tensor.shape} expected {expected}")

            allowed_detect_dtypes = {
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
                torch.bool,
                torch.int32,
                torch.int64,
            }
            if detect.dtype not in allowed_detect_dtypes:
                raise ValueError("detection_mask must be float, bool, or int tensor")

            if ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must be an integer tensor")

            self.sequence_meta = SequenceMeta(
                attention_mask=attn,
                detection_mask=detect,
                input_ids=ids,
            )
        elif self.sequence_meta is not None:
            raise ValueError(
                "sequence_meta must be None after reducing the sequence axis"
            )

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
                f"Current axes: [{present_names}]\n"
                f"Hint: Axes may have been removed by pooling (e.g., pool(dim='sequence')) "
                f"or layer selection (e.g., select(layers=10))."
            )

    def get_layer_tensor_indices(self, requested_layers: int | list[int]) -> list[int]:
        """
        Map requested model layer indices to tensor dimension indices.

        Args:
            requested_layers: List of original model layer indices

        Returns:
            List of tensor indices that correspond to the requested layers

        Raises:
            ValueError: If any requested layer is not available in this Activations object
        """
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
                raise ValueError(
                    f"Layer {layer} is not available in this Activations object.\n"
                    f"Available layers: [{available_layers}]\n"
                    f"Hint: Use acts.select(layer=...) for single layer or "
                    f"acts.select(layers=[...]) for multiple layers."
                ) from exc
        return tensor_indices

    def _require_layer_meta(self) -> LayerMeta:
        if self.layer_meta is None:
            raise ValueError(
                "Layer metadata is unavailable because the layer axis was removed.\n"
                "Hint: This happens after pooling over layers (e.g., acts.pool(dim='layer')) "
                "or selecting a single layer (e.g., acts.select(layer=10))."
            )
        return self.layer_meta

    def _require_sequence_meta(self) -> SequenceMeta:
        if self.sequence_meta is None:
            raise ValueError(
                "Sequence metadata is unavailable because the sequence axis was removed.\n"
                "Hint: This happens after pooling over sequences (e.g., acts.pool(dim='sequence')). "
                "If you need token-level data, use acts.extract_tokens() before pooling."
            )
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
    def save(
        self,
        path: str,
        compression: str | None = "gzip",
        compression_opts: int = 4,
    ) -> None:
        """Save activations to HDF5 file with compression.

        The file is chunked by layer to enable efficient partial loading
        of specific layers without reading the entire file.

        Args:
            path: Output path (should end with .h5 or .hdf5)
            compression: Compression algorithm ("gzip", "lzf", or None)
            compression_opts: Compression level for gzip (1-9, default 4)

        Storage layout:
            /activations     - Main tensor, chunked by layer
            /axes            - Axis enum values as integers
            /layer_indices   - Which model layers are stored (if LAYER axis present)
            /attention_mask  - [batch, seq] (if SEQ axis present)
            /detection_mask  - [batch, seq] (if SEQ axis present)
            /input_ids       - [batch, seq] (if SEQ axis present)
            /batch_indices   - [batch] (if present)

        Example:
            >>> acts = pl.collect_activations(...)
            >>> acts.save("activations.h5")
            >>> acts.save("activations.h5", compression="gzip", compression_opts=9)
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for saving activations. "
                "Install with: pip install probelab[storage]"
            )

        # Move to CPU for saving
        acts_cpu = self.activations.cpu().numpy()

        # Determine chunking - chunk by layer if LAYER axis present
        if self.has_axis(Axis.LAYER):
            # Chunk shape: (1, batch, seq, hidden) or (1, batch, hidden)
            chunk_shape = (1,) + acts_cpu.shape[1:]
        else:
            # No layer axis - use auto chunking
            chunk_shape = None

        with h5py.File(path, "w") as f:
            # Store main activations with compression and chunking
            f.create_dataset(
                "activations",
                data=acts_cpu,
                compression=compression,
                compression_opts=compression_opts if compression == "gzip" else None,
                chunks=chunk_shape,
            )

            # Store axes as integer values
            f.create_dataset("axes", data=[ax.value for ax in self.axes])

            # Store layer metadata
            if self.layer_meta is not None:
                f.create_dataset("layer_indices", data=list(self.layer_meta.indices))

            # Store sequence metadata
            if self.sequence_meta is not None:
                f.create_dataset(
                    "attention_mask",
                    data=self.sequence_meta.attention_mask.cpu().numpy(),
                    compression=compression,
                    compression_opts=compression_opts if compression == "gzip" else None,
                )
                f.create_dataset(
                    "detection_mask",
                    data=self.sequence_meta.detection_mask.cpu().numpy(),
                    compression=compression,
                    compression_opts=compression_opts if compression == "gzip" else None,
                )
                f.create_dataset(
                    "input_ids",
                    data=self.sequence_meta.input_ids.cpu().numpy(),
                    compression=compression,
                    compression_opts=compression_opts if compression == "gzip" else None,
                )

            # Store batch indices
            if self.batch_indices is not None:
                f.create_dataset("batch_indices", data=self.batch_indices.cpu().numpy())

            # Store metadata attributes
            f.attrs["probelab_version"] = "0.1.0"
            f.attrs["dtype"] = str(self.activations.dtype)

    @classmethod
    def load(
        cls,
        path: str,
        layers: list[int] | None = None,
        batch_slice: slice | None = None,
        device: str = "cpu",
    ) -> "Activations":
        """Load activations from HDF5 file with optional partial loading.

        Supports loading specific layers without reading the entire file,
        which is efficient for large activation files.

        Args:
            path: Path to HDF5 file
            layers: Specific layer indices to load (None = all layers).
                   Only loads requested layers from disk.
            batch_slice: Slice of batch dimension to load (None = all).
                        Example: slice(0, 100) for first 100 samples.
            device: Device to load tensors onto ("cpu" or "cuda")

        Returns:
            Activations object with requested data

        Example:
            >>> # Load all activations
            >>> acts = Activations.load("activations.h5")
            >>>
            >>> # Load only layers 12 and 16
            >>> acts = Activations.load("activations.h5", layers=[12, 16])
            >>>
            >>> # Load first 100 samples from layer 16
            >>> acts = Activations.load(
            ...     "activations.h5",
            ...     layers=[16],
            ...     batch_slice=slice(0, 100)
            ... )
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for loading activations. "
                "Install with: pip install probelab[storage]"
            )

        with h5py.File(path, "r") as f:
            # Load axes
            axes_values = f["axes"][:]
            axes = tuple(Axis(v) for v in axes_values)

            # Check if LAYER axis exists
            has_layer_axis = Axis.LAYER in axes

            # Load layer indices if present
            stored_layer_indices: list[int] | None = None
            if "layer_indices" in f:
                stored_layer_indices = list(f["layer_indices"][:])

            # Determine which layer tensor indices to load
            if layers is not None and has_layer_axis:
                if stored_layer_indices is None:
                    raise ValueError(
                        "Cannot select layers: no layer_indices stored in file"
                    )
                # Map requested layer indices to tensor indices
                tensor_indices = []
                for layer in layers:
                    if layer not in stored_layer_indices:
                        raise ValueError(
                            f"Layer {layer} not found in file. "
                            f"Available: {stored_layer_indices}"
                        )
                    tensor_indices.append(stored_layer_indices.index(layer))
                new_layer_indices = layers
            else:
                tensor_indices = None
                new_layer_indices = stored_layer_indices

            # Build slicing for activations
            # Shape is typically [layer, batch, seq, hidden] or [batch, seq, hidden]
            if has_layer_axis:
                layer_dim = axes.index(Axis.LAYER)
                batch_dim = axes.index(Axis.BATCH)
            else:
                layer_dim = None
                batch_dim = axes.index(Axis.BATCH)

            # Load activations with partial reads
            acts_dataset = f["activations"]

            import numpy as np

            if tensor_indices is not None and batch_slice is not None:
                # Both layer and batch selection
                # Need to load layer by layer due to HDF5 limitations
                layer_acts = []
                for tidx in tensor_indices:
                    if layer_dim == 0:
                        layer_data = acts_dataset[tidx, batch_slice]
                    else:
                        # Handle other axis orders if needed
                        layer_data = acts_dataset[tidx, batch_slice]
                    layer_acts.append(layer_data)
                # Stack numpy arrays first to avoid slow tensor creation
                activations = torch.from_numpy(np.stack(layer_acts)).to(
                    device=device, dtype=torch.float32
                )
            elif tensor_indices is not None:
                # Only layer selection
                layer_acts = []
                for tidx in tensor_indices:
                    layer_acts.append(acts_dataset[tidx])
                # Stack numpy arrays first to avoid slow tensor creation
                activations = torch.from_numpy(np.stack(layer_acts)).to(
                    device=device, dtype=torch.float32
                )
            elif batch_slice is not None:
                # Only batch selection
                if has_layer_axis and layer_dim == 0:
                    activations = torch.tensor(
                        acts_dataset[:, batch_slice], device=device
                    ).float()
                else:
                    activations = torch.tensor(
                        acts_dataset[batch_slice], device=device
                    ).float()
            else:
                # Load everything
                activations = torch.tensor(acts_dataset[:], device=device).float()

            # Adjust axes if layers were selected (keeping LAYER axis)
            new_axes = axes

            # Load layer meta
            layer_meta = None
            if new_layer_indices is not None and has_layer_axis:
                layer_meta = LayerMeta(indices=tuple(new_layer_indices))

            # Load sequence meta if present
            sequence_meta = None
            if "attention_mask" in f:
                if batch_slice is not None:
                    attention_mask = torch.tensor(
                        f["attention_mask"][batch_slice], device=device
                    )
                    detection_mask = torch.tensor(
                        f["detection_mask"][batch_slice], device=device
                    )
                    input_ids = torch.tensor(
                        f["input_ids"][batch_slice], device=device
                    )
                else:
                    attention_mask = torch.tensor(
                        f["attention_mask"][:], device=device
                    )
                    detection_mask = torch.tensor(
                        f["detection_mask"][:], device=device
                    )
                    input_ids = torch.tensor(f["input_ids"][:], device=device)

                sequence_meta = SequenceMeta(
                    attention_mask=attention_mask.float(),
                    detection_mask=detection_mask.float(),
                    input_ids=input_ids.long(),
                )

            # Load batch indices if present
            batch_indices = None
            if "batch_indices" in f:
                if batch_slice is not None:
                    batch_indices = torch.tensor(
                        f["batch_indices"][batch_slice], device=device
                    )
                else:
                    batch_indices = torch.tensor(
                        f["batch_indices"][:], device=device
                    )

        return cls(
            activations=activations,
            axes=new_axes,
            layer_meta=layer_meta,
            sequence_meta=sequence_meta,
            batch_indices=batch_indices,
        )

    # ------------------------------------------------------------------
    # Axis transforms
    # ------------------------------------------------------------------

    def select(
        self,
        *,
        layer: int | None = None,
        layers: list[int] | range | None = None,
    ) -> "Activations":
        """
        Select layer(s) from multi-layer activations.

        Use ``layer`` (singular) to select a single layer and remove the LAYER axis.
        Use ``layers`` (plural) to select multiple layers while keeping the LAYER axis.

        Args:
            layer: Single layer index to select. Removes LAYER axis from result.
            layers: List or range of layer indices to select. Keeps LAYER axis.

        Returns:
            New Activations with selected layer(s).

        Raises:
            ValueError: If both ``layer`` and ``layers`` are provided.
            ValueError: If neither ``layer`` nor ``layers`` are provided.
            ValueError: If requested layer(s) are not available.

        Examples:
            # Select single layer (removes LAYER axis)
            >>> acts.select(layer=10)  # Returns [batch, seq, hidden]

            # Select multiple layers (keeps LAYER axis)
            >>> acts.select(layers=[10, 15, 20])  # Returns [3, batch, seq, hidden]

            # Select range of layers
            >>> acts.select(layers=range(10, 20))
        """
        if layer is not None and layers is not None:
            raise ValueError(
                "Cannot specify both 'layer' and 'layers'. "
                "Use 'layer' for single layer selection (removes axis), "
                "'layers' for multiple layer selection (keeps axis)."
            )

        if layer is None and layers is None:
            raise ValueError(
                "Must specify either 'layer' (single, removes axis) or "
                "'layers' (multiple, keeps axis)."
            )

        # Note: Axis validation is caller's responsibility (use check_activations())

        if layer is not None:
            tensor_idx = self.get_layer_tensor_indices([layer])[0]
            dim = self._axis_positions[Axis.LAYER]
            selected = self.activations.select(dim, tensor_idx)

            new_axes = tuple(ax for ax in self.axes if ax != Axis.LAYER)
            return Activations(
                activations=selected,
                axes=new_axes,
                layer_meta=None,
                sequence_meta=self.sequence_meta,
                batch_indices=self.batch_indices,
            )

        if isinstance(layers, range):
            layers = list(layers)

        if not layers:
            raise ValueError(
                f"layers must be non-empty. Available layers: {self.layer_indices}"
            )

        tensor_indices = self.get_layer_tensor_indices(layers)
        dim = self._axis_positions[Axis.LAYER]
        index = torch.as_tensor(
            tensor_indices, device=self.activations.device, dtype=torch.long
        )
        subset = torch.index_select(self.activations, dim=dim, index=index)

        new_meta = LayerMeta(tuple(layers))
        return Activations(
            activations=subset,
            axes=self.axes,
            layer_meta=new_meta,
            sequence_meta=self.sequence_meta,
            batch_indices=self.batch_indices,
        )

    def pool(
        self,
        dim: Literal["sequence", "seq", "layer"] = "sequence",
        method: AggregationMethod | str = AggregationMethod.MEAN,
        use_detection_mask: bool = True,
    ) -> "Activations":
        """
        Pool over a specified dimension, removing that axis.

        This is a unified API for dimension reduction that works across different axes.
        Replaces the older `sequence_pool()` and `aggregate()` methods with a more
        consistent interface.

        Args:
            dim: Dimension to pool over. Options:
                - "sequence" or "seq": Pool over sequence dimension
                - "layer": Pool over layer dimension
            method: Pooling method to use:
                - "mean": Average pooling
                - "max": Max pooling
                - "last_token": Use last valid token (sequence dim only)
            use_detection_mask: If True, only pool over detected tokens (default).
                               Only applies to sequence dimension.

        Returns:
            New Activations without the pooled dimension

        Examples:
            # Pool over sequence dimension (most common)
            >>> pooled = acts.pool(dim="sequence", method="mean")

            # Pool over layer dimension to get layer-averaged features
            >>> pooled = acts.pool(dim="layer", method="mean")

            # Pool without detection mask (use all tokens)
            >>> pooled = acts.pool(dim="sequence", use_detection_mask=False)

        Note:
            For backwards compatibility, `sequence_pool()` and `aggregate()` are
            still available but may be deprecated in future versions.
        """
        if isinstance(method, str):
            try:
                method = AggregationMethod(method)
            except ValueError:
                raise ValueError(
                    f"Unknown pooling method: {method}. "
                    f"Supported: {[m.value for m in AggregationMethod]}"
                )

        # Note: Axis validation is caller's responsibility (use check_activations())
        if dim in ("sequence", "seq"):
            axis = Axis.SEQ
        elif dim == "layer":
            axis = Axis.LAYER
        else:
            raise ValueError(
                f"Unknown dimension: {dim}. Supported: 'sequence', 'seq', 'layer'"
            )

        if axis == Axis.SEQ:
            if use_detection_mask:
                reduced = self._reduce_sequence(method)
            else:
                dim_idx = self._axis_positions[Axis.SEQ]
                if method == AggregationMethod.MEAN:
                    reduced = self.activations.mean(dim=dim_idx)
                elif method == AggregationMethod.MAX:
                    reduced = self.activations.max(dim=dim_idx).values
                elif method == AggregationMethod.LAST_TOKEN:
                    reduced = self.activations.select(dim_idx, -1)
                else:
                    raise ValueError(
                        f"Unknown pooling method: {method}. "
                        f"Supported: {[m.value for m in AggregationMethod]}"
                    )

            new_axes = tuple(ax for ax in self.axes if ax != Axis.SEQ)
            return Activations(
                activations=reduced,
                axes=new_axes,
                layer_meta=self.layer_meta,
                sequence_meta=None,
                batch_indices=self.batch_indices,
            )

        elif axis == Axis.LAYER:
            if method == AggregationMethod.LAST_TOKEN:
                raise ValueError(
                    "'last_token' pooling is only supported for sequence dimension"
                )

            dim_idx = self._axis_positions[Axis.LAYER]
            if method == AggregationMethod.MEAN:
                reduced = self.activations.mean(dim=dim_idx)
            elif method == AggregationMethod.MAX:
                reduced = self.activations.max(dim=dim_idx).values
            else:
                raise ValueError(
                    f"Unknown pooling method: {method}. "
                    f"Supported for layer dimension: 'mean', 'max'"
                )

            new_axes = tuple(ax for ax in self.axes if ax != Axis.LAYER)
            return Activations(
                activations=reduced,
                axes=new_axes,
                layer_meta=None,
                sequence_meta=self.sequence_meta,
                batch_indices=self.batch_indices,
            )

    def extract_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract detected tokens for token-level training.

        Returns:
            Tuple of (features, tokens_per_sample) where:
            - features: [n_tokens, hidden] tensor of detected tokens
            - tokens_per_sample: [batch] tensor of token counts per sample
        """
        self.expect_axes(Axis.BATCH, Axis.SEQ, Axis.HIDDEN)

        if self.has_axis(Axis.LAYER):
            if self.n_layers != 1:
                raise ValueError(
                    f"Token extraction requires single layer, but found {self.n_layers} layers.\n"
                    f"Available layers: {self.layer_indices}\n"
                    f"Hint: Use acts.select(layer=i) to select a single layer before extracting tokens."
                )
            # Remove layer dimension
            acts = self.activations.squeeze(self._axis_positions[Axis.LAYER])
        else:
            acts = self.activations

        seq_meta = self._require_sequence_meta()
        mask = seq_meta.detection_mask.bool()
        tokens_per_sample = mask.sum(dim=1)

        if tokens_per_sample.sum() == 0:
            features = torch.empty(
                0,
                acts.shape[-1],
                device=acts.device,
                dtype=acts.dtype,
            )
        else:
            features = acts[mask]

        return features, tokens_per_sample.to(device=acts.device)

    # ------------------------------------------------------------------
    # Internal sequence reduction
    # ------------------------------------------------------------------
    def _reduce_sequence(self, method: AggregationMethod | str) -> torch.Tensor:
        """Reduce sequence dimension using specified method.

        Uses torch.no_grad() since this is a pure aggregation operation
        that doesn't need gradient tracking.
        """
        with torch.no_grad():
            seq_meta = self._require_sequence_meta()
            detection_mask = seq_meta.detection_mask

            batch_dim = self._axis_positions[Axis.BATCH]
            seq_dim = self._axis_positions[Axis.SEQ]

            return masked_pool(
                tensor=self.activations,
                mask=detection_mask,
                method=method,
                seq_dim=seq_dim,
                batch_dim=batch_dim,
            )


def streaming_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    tokenized_inputs: dict[str, torch.Tensor],
    layers: list[int],
    batch_size: int = 8,
    verbose: bool | None = None,
    hook_point: HookPoint = HookPoint.POST_BLOCK,
    detach_activations: bool = True,
) -> Generator[Activations, None, None]:
    """
    Generator that yields Activations batches with all optimizations preserved.

    Key optimizations:
    - Single HookedModel context for entire iteration
    - Sorted batching for minimal padding
    - Tensor views via narrow() where possible
    - Non-blocking GPU transfers
    - Sequence length optimization
    - Buffer reuse in HookedModel

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer for padding info
        tokenized_inputs: Pre-tokenized inputs dictionary
        layers: Layer indices to extract
        batch_size: Batch size for processing
        verbose: Whether to show progress bar. If None, uses VERBOSE from config.
        hook_point: Where to extract activations:
            - "post_block": After transformer block (aligns with HF hidden_states)
            - "pre_layernorm": Before layer normalization (legacy behavior)

    Yields:
        Activations objects for each batch
    """
    verbose = _resolve_verbose(verbose)
    n_samples = tokenized_inputs["input_ids"].shape[0]

    with HookedModel(
        model, layers, detach_activations=detach_activations, hook_point=hook_point
    ) as hooked_model:
        batch_iter = get_batches(tokenized_inputs, batch_size, tokenizer)

        if verbose:
            batch_iter = tqdm(
                batch_iter,
                desc="Collecting activations",
                total=(n_samples + batch_size - 1) // batch_size,
            )

        for batch_inputs, batch_indices in batch_iter:
            batch_inputs = _ensure_on_device(batch_inputs, model.device)
            batch_acts = hooked_model.get_activations(batch_inputs)

            yield Activations(
                activations=batch_acts,
                axes=_DEFAULT_AXES,
                layer_meta=LayerMeta(tuple(layers)),
                sequence_meta=SequenceMeta(
                    attention_mask=batch_inputs["attention_mask"],
                    detection_mask=batch_inputs["detection_mask"],
                    input_ids=batch_inputs["input_ids"],
                ),
                batch_indices=torch.as_tensor(batch_indices, dtype=torch.long),
            )


def batch_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    tokenized_inputs: dict[str, torch.Tensor],
    layers: list[int],
    batch_size: int = 8,
    verbose: bool | None = None,
    hook_point: HookPoint = HookPoint.POST_BLOCK,
    detach_activations: bool = True,
) -> Activations:
    """
    Collect all activations at once into a single Activations object.

    Uses different memory strategies based on dataset size:
    - Small datasets: Pre-allocate full tensor
    - Large datasets: Accumulate batches then concatenate

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer for padding info
        tokenized_inputs: Pre-tokenized inputs
        layers: Layer indices to extract
        batch_size: Batch size for processing
        verbose: Whether to show progress. If None, uses VERBOSE from config.
        hook_point: Where to extract activations:
            - "post_block": After transformer block (aligns with HF hidden_states)
            - "pre_layernorm": Before layer normalization (legacy behavior)

    Returns:
        Single Activations object with all data
    """
    verbose = _resolve_verbose(verbose)
    n_samples, max_seq_len = tokenized_inputs["input_ids"].shape
    hidden_dim = get_hidden_dim(model)

    # Use different strategies based on dataset size
    if n_samples * max_seq_len < 100000:  # Small dataset
        # Pre-allocate full tensor for efficiency
        all_activations = torch.zeros(
            (len(layers), n_samples, max_seq_len, hidden_dim),
            device="cpu",
            dtype=model.dtype,
        )

        with HookedModel(
            model, layers, detach_activations=detach_activations, hook_point=hook_point
        ) as hooked_model:
            batches = get_batches(tokenized_inputs, batch_size, tokenizer)

            if verbose:
                batches = tqdm(
                    batches,
                    desc="Collecting activations",
                    total=(n_samples + batch_size - 1) // batch_size,
                )

            for batch_inputs, batch_indices in batches:
                batch_inputs = _ensure_on_device(batch_inputs, model.device)

                seq_len = batch_inputs["input_ids"].shape[1]
                batch_acts = hooked_model.get_activations(batch_inputs)

                # Use blocking transfer for small datasets to avoid race conditions
                batch_acts = batch_acts.to("cpu")

                # Store in correct positions
                if tokenizer.padding_side == "right":
                    all_activations[:, batch_indices, :seq_len] = batch_acts
                else:
                    all_activations[:, batch_indices, -seq_len:] = batch_acts
    else:
        # Large dataset - accumulate batches to avoid huge upfront allocation
        batch_list = []
        indices_list = []

        with HookedModel(
            model, layers, detach_activations=detach_activations, hook_point=hook_point
        ) as hooked_model:
            batches = get_batches(tokenized_inputs, batch_size, tokenizer)

            if verbose:
                batches = tqdm(
                    batches,
                    desc="Collecting activations",
                    total=(n_samples + batch_size - 1) // batch_size,
                )

            for batch_idx, (batch_inputs, batch_indices) in enumerate(batches):
                batch_inputs = _ensure_on_device(batch_inputs, model.device)
                batch_acts = hooked_model.get_activations(batch_inputs)

                batch_acts = batch_acts.to("cpu", non_blocking=True)
                batch_list.append(batch_acts)
                indices_list.append(batch_indices)

                # Periodic sync to prevent memory buffering
                if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.synchronize()

        # Final synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        all_activations = torch.zeros(
            (len(layers), n_samples, max_seq_len, hidden_dim),
            device="cpu",
            dtype=model.dtype,
        )

        for batch_acts, batch_indices in zip(batch_list, indices_list):
            seq_len = batch_acts.shape[2]
            if tokenizer.padding_side == "right":
                all_activations[:, batch_indices, :seq_len] = batch_acts
            else:
                all_activations[:, batch_indices, -seq_len:] = batch_acts

    return Activations(
        activations=all_activations,
        axes=_DEFAULT_AXES,
        layer_meta=LayerMeta(tuple(layers)),
        sequence_meta=SequenceMeta(
            attention_mask=tokenized_inputs["attention_mask"],
            detection_mask=tokenized_inputs["detection_mask"],
            input_ids=tokenized_inputs["input_ids"],
        ),
        batch_indices=torch.arange(n_samples, dtype=torch.long),
    )


def batch_activations_pooled(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    tokenized_inputs: dict[str, torch.Tensor],
    layers: list[int],
    batch_size: int,
    pooling_method: AggregationMethod | str,
    verbose: bool | None = None,
    hook_point: HookPoint = HookPoint.POST_BLOCK,
    detach_activations: bool = True,
) -> Activations:
    """Collect and pool in one pass - optimized for minimal memory.

    Peak memory: One batch forward pass only
    Resident memory: Pooled storage [L, N, H] on CPU

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer for padding info
        tokenized_inputs: Pre-tokenized inputs
        layers: Layer indices to extract
        batch_size: Batch size for processing
        pooling_method: How to pool sequences ("mean", "max", "last_token")
        verbose: Whether to show progress. If None, uses VERBOSE from config.
        hook_point: Where to extract activations
        detach_activations: Whether to detach from computation graph

    Returns:
        Activations without sequence axis (pooled)
    """
    verbose = _resolve_verbose(verbose)

    # Normalize pooling method
    if isinstance(pooling_method, str):
        pooling_method = AggregationMethod(pooling_method)

    n_samples = tokenized_inputs["input_ids"].shape[0]
    hidden_dim = get_hidden_dim(model)

    # Allocate ONLY pooled storage [L, N, H] - NO sequence dimension
    pooled_storage = torch.zeros(
        (len(layers), n_samples, hidden_dim),
        device="cpu",
        dtype=model.dtype,
    )

    num_batches = (n_samples + batch_size - 1) // batch_size

    with HookedModel(
        model, layers, detach_activations=detach_activations, hook_point=hook_point
    ) as hooked_model:
        batches = get_batches(tokenized_inputs, batch_size, tokenizer)

        if verbose:
            batches = tqdm(
                batches,
                desc=f"Collecting ({pooling_method.value} pooled)",
                total=num_batches,
            )

        for batch_idx, (batch_inputs, batch_indices) in enumerate(batches):
            batch_inputs_gpu = _ensure_on_device(batch_inputs, model.device)

            # Get batch activations [n_layers, batch_size, seq_len, hidden]
            batch_acts = hooked_model.get_activations(batch_inputs_gpu)

            # Pool using shared utility
            # batch_acts shape: [n_layers, batch_size, seq_len, hidden]
            # batch_dim=1, seq_dim=2
            detection_mask = batch_inputs_gpu["detection_mask"]  # [batch, seq]
            pooled = masked_pool(
                tensor=batch_acts,
                mask=detection_mask,
                method=pooling_method,
                seq_dim=2,
                batch_dim=1,
            )  # [n_layers, batch, hidden]

            # Store pooled results (blocking transfer to avoid race conditions)
            pooled_storage[:, batch_indices] = pooled.to("cpu")

            # Explicit cleanup to free GPU memory immediately
            del batch_acts, pooled, batch_inputs_gpu
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Return WITHOUT sequence axis
    return Activations(
        activations=pooled_storage,
        axes=(Axis.LAYER, Axis.BATCH, Axis.HIDDEN),
        layer_meta=LayerMeta(tuple(layers)),
        sequence_meta=None,
        batch_indices=torch.arange(n_samples, dtype=torch.long),
    )


class ActivationIterator:
    """
    Regenerable iterator for streaming activations.

    This iterator can be iterated multiple times, creating a fresh generator
    each time. This allows multiple passes over the data without needing to
    recreate the iterator.
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizerBase",
        tokenized_inputs: dict[str, torch.Tensor],
        layers: list[int],
        batch_size: int,
        verbose: bool | None,
        num_batches: int,
        hook_point: HookPoint = HookPoint.POST_BLOCK,
        detach_activations: bool = True,
    ):
        """
        Initialize the regenerable iterator.

        Args:
            model: Model to extract activations from
            tokenizer: Tokenizer for padding info
            tokenized_inputs: Pre-tokenized inputs
            layers: Layer indices to extract
            batch_size: Batch size for processing
            verbose: Whether to show progress. If None, uses VERBOSE from config.
            num_batches: Total number of batches
            detach_activations: Whether to detach activations
            hook_point: Where to extract activations
        """
        self._model = model
        self._tokenizer = tokenizer
        self._tokenized_inputs = tokenized_inputs
        self._layers = layers
        self._batch_size = batch_size
        self._verbose = _resolve_verbose(verbose)
        self._num_batches = num_batches
        self._hook_point = hook_point
        self._detach_activations = detach_activations

    def __iter__(self) -> Iterator[Activations]:
        """Create and return a fresh generator for activation batches."""
        return streaming_activations(
            model=self._model,
            tokenizer=self._tokenizer,
            tokenized_inputs=self._tokenized_inputs,
            layers=self._layers,
            batch_size=self._batch_size,
            detach_activations=self._detach_activations,
            verbose=self._verbose,
            hook_point=self._hook_point,
        )

    def __len__(self) -> int:
        """Return number of batches."""
        return self._num_batches

    @property
    def layers(self) -> list[int]:
        """Return layer indices this iterator provides."""
        return self._layers


# Type alias for collection strategy
CollectionStrategy = Literal["mean", "max", "last_token"]


# Adds overlading for correct return type inference from collect_activations
@overload
def collect_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    dataset: DialogueDataset,
    *,
    layers: int | list[int],
    mask: "MaskFunction",
    batch_size: int = 32,
    streaming: Literal[False] = False,
    collection_strategy: CollectionStrategy | None = None,
    hook_point: HookPoint = HookPoint.POST_BLOCK,
    add_generation_prompt: bool = False,
    detach_activations: bool = True,
    verbose: bool | None = None,
) -> Activations: ...


# Adds overlading for correct return type inference from collect_activations
@overload
def collect_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    dataset: DialogueDataset,
    *,
    layers: int | list[int],
    mask: "MaskFunction",
    batch_size: int = 32,
    streaming: Literal[True],
    collection_strategy: CollectionStrategy | None = None,
    hook_point: HookPoint = HookPoint.POST_BLOCK,
    add_generation_prompt: bool = False,
    detach_activations: bool = True,
    verbose: bool | None = None,
) -> ActivationIterator: ...


def collect_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    dataset: DialogueDataset,
    *,
    layers: int | list[int],
    mask: "MaskFunction",
    batch_size: int = 32,
    streaming: bool = False,
    collection_strategy: CollectionStrategy | None = None,
    hook_point: HookPoint = HookPoint.POST_BLOCK,
    add_generation_prompt: bool = False,
    detach_activations: bool = True,
    verbose: bool | None = None,
) -> Activations | ActivationIterator:
    """Entry point for activation collection from datasets.

    Collects activations from specified layers using an explicit mask function.
    All parameters are explicit - no auto-detection or silent defaults.

    Args:
        model: Model providing hidden states.
        tokenizer: Tokenizer aligned with ``model``.
        dataset: DialogueDataset containing dialogues and labels.
        layers: Layer index or indices to record. Required - no auto-detection.
        mask: Mask function determining which tokens to detect. Required - no defaults.
              Use ``masks.all()`` to detect all non-padding tokens explicitly.
        batch_size: Number of sequences per activation batch. Default: 32.
        streaming: When ``True`` yield batches lazily, otherwise load all
            activations into memory. Default: False.
        collection_strategy: Optional pooling strategy to apply during collection.
            - None (default): Dense collection [layers, batch, seq, hidden]
            - "mean": Pool sequences via mean during collection [layers, batch, hidden]
            - "max": Pool sequences via max during collection [layers, batch, hidden]
            - "last_token": Use last detected token [layers, batch, hidden]
            Using a collection strategy provides ~440x memory reduction and ~2x throughput.
        hook_point: Where to extract activations from the model:
            - "post_block" (default): After transformer block + final layernorm.
            - "pre_layernorm": Before layer normalization (earlier in computation).
        add_generation_prompt: Whether to append generation tokens before
            tokenization. Default: False.
        detach_activations: Whether to detach activations from the computation graph.
            - True (default): Detach for memory efficiency (probe training/evaluation)
            - False: Keep gradients (enables differentiable probe predictions)
        verbose: Toggle progress reporting. If None, uses VERBOSE from config.

    Returns:
        - Activations: Dense or pooled collection depending on collection_strategy
        - ActivationIterator: For streaming mode (collection_strategy ignored)

    Examples:
        >>> import probelab as pl
        >>> dataset = pl.datasets.CircuitBreakersDataset()
        >>>
        >>> # Dense collection (default)
        >>> acts = collect_activations(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     dataset=dataset,
        ...     layers=[16, 20, 24],
        ...     mask=pl.masks.assistant(),
        ...     batch_size=32,
        ... )
        >>> acts.shape  # [3, batch_size, seq_len, hidden_dim]

        >>> # Pooled collection (memory efficient)
        >>> acts = collect_activations(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     dataset=dataset,
        ...     layers=[16],
        ...     mask=pl.masks.assistant(),
        ...     collection_strategy="mean",  # Pool during collection
        ... )
        >>> acts.shape  # [1, batch_size, hidden_dim] - no seq dimension
    """
    if isinstance(layers, int):
        layers = [layers]

    dialogues = dataset.dialogues

    # Set up tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize all dialogues once (key optimization)
    tokenized_inputs = tokenize_dialogues(
        tokenizer=tokenizer,
        dialogues=dialogues,
        mask=mask,  # Mask is now required
        device="cpu",
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
        padding=True,
    )

    n_samples = tokenized_inputs["input_ids"].shape[0]
    num_batches = (n_samples + batch_size - 1) // batch_size

    if streaming:
        # Return regenerable iterator that creates fresh generators
        # Note: collection_strategy is ignored for streaming mode
        return ActivationIterator(
            model=model,
            tokenizer=tokenizer,
            tokenized_inputs=tokenized_inputs,
            layers=layers,
            batch_size=batch_size,
            verbose=verbose,
            num_batches=num_batches,
            hook_point=hook_point,
            detach_activations=detach_activations,
        )
    elif collection_strategy in ("mean", "max", "last_token"):
        # Pooled collection - pools each batch during collection for memory efficiency
        return batch_activations_pooled(
            model,
            tokenizer,
            tokenized_inputs,
            layers,
            batch_size,
            collection_strategy,
            verbose,
            hook_point,
            detach_activations,
        )
    else:
        # Dense collection - full sequences [layers, batch, seq, hidden]
        return batch_activations(
            model,
            tokenizer,
            tokenized_inputs,
            layers,
            batch_size,
            verbose,
            hook_point,
            detach_activations,
        )

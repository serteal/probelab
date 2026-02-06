"""Lazy activation container for probe workflows.

The design is intentionally narrow and tinygrad-inspired:
- chainable operations that return new objects,
- lazy execution until ``realize`` / ``save`` / ``cache``,
- explicit axis labels through ``dims``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Literal

import torch

from .. import pool as P
from ..models import HookedModel

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .tokenization import Tokens


AxisName = Literal["b", "l", "s", "h"]
PoolMethod = Literal["mean", "max", "last", "sum"]
_DTYPE_STR_TO_TORCH: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.bfloat16": torch.bfloat16,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def _normalize_dims(dims: str) -> str:
    if not dims:
        raise ValueError("dims must be non-empty")
    allowed = "blsh"
    if any(ch not in allowed for ch in dims):
        raise ValueError(f"dims must be a subset of {allowed!r}, got {dims!r}")
    if len(set(dims)) != len(dims):
        raise ValueError(f"dims cannot contain duplicates, got {dims!r}")
    normalized = "".join(ch for ch in allowed if ch in dims)
    if normalized != dims:
        raise ValueError(
            "dims must preserve canonical order 'b' -> 'l' -> 's' -> 'h'. "
            f"Received {dims!r}."
        )
    if "b" not in dims:
        raise ValueError("dims must include batch axis 'b'")
    if "h" not in dims:
        raise ValueError("dims must include hidden axis 'h'")
    return dims


def _axis_char(dim: int | str, dims: str) -> str:
    if isinstance(dim, str):
        if dim not in {"b", "l", "s", "h"}:
            raise ValueError(f"Unknown dim name {dim!r}. Expected one of 'b', 'l', 's', 'h'.")
        if dim not in dims:
            raise ValueError(f"Axis {dim!r} is not present in dims {dims!r}")
        return dim

    n = len(dims)
    idx = dim if dim >= 0 else n + dim
    if idx < 0 or idx >= n:
        raise IndexError(f"dim index out of range for dims {dims!r}: {dim}")
    return dims[idx]


def _axis_index(dim: int | str, dims: str) -> int:
    return dims.index(_axis_char(dim, dims))


def _slice_indices(spec: slice | range | list[int] | torch.Tensor | tuple[int, ...], size: int) -> tuple[int, ...]:
    if isinstance(spec, slice):
        return tuple(range(size)[spec])
    if isinstance(spec, range):
        return tuple(spec)
    if isinstance(spec, torch.Tensor):
        if spec.ndim != 1:
            raise ValueError("Batch index tensor must be 1D")
        return tuple(int(x) for x in spec.tolist())
    if isinstance(spec, tuple):
        out = spec
    else:
        out = tuple(int(x) for x in spec)

    normalized: list[int] = []
    for i in out:
        j = i + size if i < 0 else i
        if j < 0 or j >= size:
            raise IndexError(f"Batch index out of bounds: {i} for size {size}")
        normalized.append(j)
    return tuple(normalized)


def _resolve_layer_ids(layer_ids: tuple[int, ...] | None, n_layers: int) -> tuple[int, ...]:
    if layer_ids is None:
        return tuple(range(n_layers))
    if len(layer_ids) != n_layers:
        raise ValueError(f"layer_ids length ({len(layer_ids)}) must match layer dim ({n_layers})")
    return layer_ids


def _hidden_dim(model: "PreTrainedModel") -> int:
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    raise ValueError(f"Cannot determine hidden dimension for {type(model)}")


def _pool_masked(x: torch.Tensor, mask: torch.Tensor, *, method: PoolMethod, dim: int) -> torch.Tensor:
    if method == "mean":
        return P.mean(x, mask, dim=dim)
    if method == "max":
        return P.max(x, mask, dim=dim)
    if method == "last":
        return P.last_token(x, mask, dim=dim)
    if method == "sum":
        mask_bool = mask.to(x.device).bool()
        shape = [1] * x.ndim
        shape[0] = mask_bool.shape[0]
        shape[dim] = mask_bool.shape[1]
        return (x * mask_bool.view(shape).to(x.dtype)).sum(dim=dim)
    raise ValueError(f"Unknown pool method: {method}")


@dataclass(frozen=True, slots=True)
class _TensorSource:
    tensor: torch.Tensor
    dims: str
    seq_mask: torch.Tensor | None
    layer_ids: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class _ModelSource:
    model: "PreTrainedModel"
    tokens: "Tokens"
    layers: tuple[int, ...]
    batch_size: int
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class _DiskSource:
    path: str


@dataclass(frozen=True, slots=True)
class _SelectLayersOp:
    layer_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SliceBatchOp:
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PoolOp:
    method: PoolMethod
    axis: str


@dataclass(frozen=True, slots=True)
class _CastOp:
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class _ToDeviceOp:
    device: torch.device


_Op = _SelectLayersOp | _SliceBatchOp | _PoolOp | _CastOp | _ToDeviceOp


class Acts:
    """Lazy activations plan.

    Public constructor is tensor-backed:

    ```python
    acts = Acts(tensor, dims="blsh", seq_mask=mask)
    ```
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        dims: Literal["bh", "bsh", "blh", "blsh"],
        seq_mask: torch.Tensor | None = None,
        layer_ids: Iterable[int] | None = None,
    ):
        dims_n = _normalize_dims(dims)
        if tensor.ndim != len(dims_n):
            raise ValueError(
                f"Tensor rank {tensor.ndim} does not match dims {dims_n!r} ({len(dims_n)})."
            )

        layer_ids_tuple: tuple[int, ...] | None = None
        if "l" in dims_n:
            n_layers = tensor.shape[dims_n.index("l")]
            layer_ids_tuple = _resolve_layer_ids(
                tuple(int(x) for x in layer_ids) if layer_ids is not None else None,
                n_layers,
            )
        elif layer_ids is not None:
            raise ValueError("layer_ids provided but dims has no 'l' axis")

        seq_mask_t: torch.Tensor | None = None
        if "s" in dims_n:
            if seq_mask is None:
                raise ValueError("seq_mask is required when dims contains 's'")
            expected = (tensor.shape[dims_n.index("b")], tensor.shape[dims_n.index("s")])
            seq_mask_t = seq_mask.to(tensor.device)
            if tuple(seq_mask_t.shape) != expected:
                raise ValueError(f"seq_mask shape {tuple(seq_mask_t.shape)} must match {expected}")
        elif seq_mask is not None:
            raise ValueError("seq_mask provided but dims has no 's' axis")

        self._source: _TensorSource | _ModelSource | _DiskSource = _TensorSource(
            tensor=tensor,
            dims=dims_n,
            seq_mask=seq_mask_t,
            layer_ids=layer_ids_tuple,
        )
        self._ops: tuple[_Op, ...] = ()
        self._dims = dims_n
        self._shape = tuple(int(x) for x in tensor.shape)
        self._dtype = tensor.dtype
        self._device = tensor.device
        self._seq_mask = seq_mask_t
        self._layer_ids = layer_ids_tuple

    @classmethod
    def from_activations(cls, acts: "Activations") -> "Acts":
        from .activations import Axis

        if acts.has_axis(Axis.LAYER) and acts.has_axis(Axis.SEQ):
            dims = "blsh"
        elif acts.has_axis(Axis.LAYER):
            dims = "blh"
        elif acts.has_axis(Axis.SEQ):
            dims = "bsh"
        else:
            dims = "bh"

        seq_mask = acts.detection_mask if acts.has_axis(Axis.SEQ) else None
        layer_ids = acts.layer_indices if acts.has_axis(Axis.LAYER) else None
        return cls(acts.activations, dims=dims, seq_mask=seq_mask, layer_ids=layer_ids)

    @classmethod
    def _from_model(
        cls,
        model: "PreTrainedModel",
        tokens: "Tokens",
        layers: Iterable[int],
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> "Acts":
        layer_ids = tuple(int(x) for x in layers)
        if not layer_ids:
            raise ValueError("layers must be non-empty")
        dims = "blsh"
        shape = (len(tokens), len(layer_ids), tokens.seq_len, _hidden_dim(model))

        obj = cls.__new__(cls)
        obj._source = _ModelSource(
            model=model,
            tokens=tokens,
            layers=layer_ids,
            batch_size=batch_size,
            dtype=dtype,
        )
        obj._ops = ()
        obj._dims = dims
        obj._shape = shape
        obj._dtype = dtype
        obj._device = torch.device("cpu")
        obj._seq_mask = tokens.detection_mask
        obj._layer_ids = layer_ids
        return obj

    @classmethod
    def load(cls, path: str | Path) -> "Acts":
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required for Acts.load. Install with: pip install probelab[storage]") from exc

        path = str(path)
        with h5py.File(path, "r") as f:
            if "acts" not in f:
                raise ValueError(f"{path!r} does not contain an 'acts' dataset")

            dims_attr = f.attrs.get("dims")
            if dims_attr is None:
                raise ValueError(f"{path!r} is missing required attribute 'dims'")

            dims = _normalize_dims(str(dims_attr))
            shape = tuple(int(x) for x in f["acts"].shape)

            dtype_attr = f.attrs.get("dtype", "torch.float32")
            dtype = _DTYPE_STR_TO_TORCH.get(str(dtype_attr), torch.float32)

            layer_ids = None
            if "layer_ids" in f:
                layer_ids = tuple(int(x) for x in f["layer_ids"][:])

            seq_mask = None
            if "seq_mask" in f:
                seq_mask = torch.from_numpy(f["seq_mask"][:]).float()

        obj = cls.__new__(cls)
        obj._source = _DiskSource(path=path)
        obj._ops = ()
        obj._dims = dims
        obj._shape = shape
        obj._dtype = dtype
        obj._device = torch.device("cpu")
        obj._seq_mask = seq_mask
        obj._layer_ids = layer_ids
        return obj

    @classmethod
    def _from_state(
        cls,
        source: _TensorSource | _ModelSource | _DiskSource,
        ops: tuple[_Op, ...],
        *,
        dims: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        seq_mask: torch.Tensor | None,
        layer_ids: tuple[int, ...] | None,
    ) -> "Acts":
        obj = cls.__new__(cls)
        obj._source = source
        obj._ops = ops
        obj._dims = dims
        obj._shape = shape
        obj._dtype = dtype
        obj._device = device
        obj._seq_mask = seq_mask
        obj._layer_ids = layer_ids
        return obj

    @property
    def dims(self) -> str:
        return self._dims

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def seq_mask(self) -> torch.Tensor | None:
        return self._seq_mask

    @property
    def layer_ids(self) -> tuple[int, ...] | None:
        return self._layer_ids

    def has_dim(self, dim: AxisName) -> bool:
        return dim in self._dims

    def _clone(
        self,
        op: _Op,
        *,
        dims: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        seq_mask: torch.Tensor | None,
        layer_ids: tuple[int, ...] | None,
    ) -> "Acts":
        return Acts._from_state(
            self._source,
            self._ops + (op,),
            dims=dims,
            shape=shape,
            dtype=dtype,
            device=device,
            seq_mask=seq_mask,
            layer_ids=layer_ids,
        )

    def select_layers(self, layer_or_layers: int | Iterable[int]) -> "Acts":
        if "l" not in self._dims:
            raise ValueError(f"select_layers requires an 'l' axis, got dims {self._dims!r}")

        if isinstance(layer_or_layers, int):
            requested = (int(layer_or_layers),)
        else:
            requested = tuple(int(x) for x in layer_or_layers)
        if not requested:
            raise ValueError("select_layers requires at least one layer")

        ldim = self._dims.index("l")
        current_layer_ids = _resolve_layer_ids(self._layer_ids, self._shape[ldim])
        current_set = set(current_layer_ids)
        for layer in requested:
            if layer not in current_set:
                raise ValueError(
                    f"Layer {layer} not available. Available layers: {list(current_layer_ids)}"
                )

        new_shape = list(self._shape)
        new_shape[ldim] = len(requested)

        return self._clone(
            _SelectLayersOp(layer_ids=requested),
            dims=self._dims,
            shape=tuple(new_shape),
            dtype=self._dtype,
            device=self._device,
            seq_mask=self._seq_mask,
            layer_ids=requested,
        )

    def slice_batch(self, slice_or_indices: slice | range | list[int] | torch.Tensor | tuple[int, ...]) -> "Acts":
        bdim = _axis_index("b", self._dims)
        idx = _slice_indices(slice_or_indices, self._shape[bdim])

        new_shape = list(self._shape)
        new_shape[bdim] = len(idx)

        new_mask = self._seq_mask
        if new_mask is not None:
            new_mask = new_mask[list(idx)]

        return self._clone(
            _SliceBatchOp(indices=idx),
            dims=self._dims,
            shape=tuple(new_shape),
            dtype=self._dtype,
            device=self._device,
            seq_mask=new_mask,
            layer_ids=self._layer_ids,
        )

    def _pool(self, method: PoolMethod, dim: int | str = "s") -> "Acts":
        axis = _axis_char(dim, self._dims)
        if axis == "b":
            raise ValueError("Pooling over batch axis 'b' is not supported in Acts v1")

        axis_idx = self._dims.index(axis)
        new_dims = self._dims.replace(axis, "")
        new_shape = tuple(s for i, s in enumerate(self._shape) if i != axis_idx)

        new_seq_mask = self._seq_mask
        if axis == "s":
            if self._seq_mask is None:
                raise ValueError("Cannot pool over 's' without seq_mask")
            new_seq_mask = None

        new_layer_ids = self._layer_ids
        if axis == "l":
            new_layer_ids = None

        return self._clone(
            _PoolOp(method=method, axis=axis),
            dims=new_dims,
            shape=new_shape,
            dtype=self._dtype,
            device=self._device,
            seq_mask=new_seq_mask,
            layer_ids=new_layer_ids,
        )

    def mean_pool(self, dim: int | str = "s") -> "Acts":
        return self._pool("mean", dim)

    def max_pool(self, dim: int | str = "s") -> "Acts":
        return self._pool("max", dim)

    def last_pool(self, dim: int | str = "s") -> "Acts":
        return self._pool("last", dim)

    def sum_pool(self, dim: int | str = "s") -> "Acts":
        return self._pool("sum", dim)

    def cast(self, dtype: torch.dtype) -> "Acts":
        return self._clone(
            _CastOp(dtype=dtype),
            dims=self._dims,
            shape=self._shape,
            dtype=dtype,
            device=self._device,
            seq_mask=self._seq_mask,
            layer_ids=self._layer_ids,
        )

    def to(self, device: str | torch.device) -> "Acts":
        tgt = torch.device(device)
        return self._clone(
            _ToDeviceOp(device=tgt),
            dims=self._dims,
            shape=self._shape,
            dtype=self._dtype,
            device=tgt,
            seq_mask=self._seq_mask.to(tgt) if self._seq_mask is not None else None,
            layer_ids=self._layer_ids,
        )

    def _split_prefix_pushdown(self) -> tuple[tuple[_Op, ...], tuple[_Op, ...]]:
        pushable: list[_Op] = []
        i = 0
        while i < len(self._ops):
            op = self._ops[i]
            if isinstance(op, (_SelectLayersOp, _SliceBatchOp)):
                pushable.append(op)
                i += 1
                continue
            if isinstance(op, _PoolOp) and op.axis == "s":
                pushable.append(op)
                i += 1
                continue
            break
        return tuple(pushable), self._ops[i:]

    def _apply_ops_to_tensor(
        self,
        tensor: torch.Tensor,
        dims: str,
        seq_mask: torch.Tensor | None,
        layer_ids: tuple[int, ...] | None,
        ops: Iterable[_Op],
    ) -> tuple[torch.Tensor, str, torch.Tensor | None, tuple[int, ...] | None, torch.dtype, torch.device]:
        cur = tensor
        cur_dims = dims
        cur_mask = seq_mask
        cur_layers = layer_ids

        for op in ops:
            if isinstance(op, _SelectLayersOp):
                if "l" not in cur_dims:
                    raise ValueError("select_layers encountered after layer axis removal")
                ldim = cur_dims.index("l")
                layer_values = _resolve_layer_ids(cur_layers, cur.shape[ldim])
                pos = [layer_values.index(layer_id) for layer_id in op.layer_ids]
                idx = torch.as_tensor(pos, device=cur.device, dtype=torch.long)
                cur = torch.index_select(cur, dim=ldim, index=idx)
                cur_layers = op.layer_ids
                continue

            if isinstance(op, _SliceBatchOp):
                bdim = cur_dims.index("b")
                idx = torch.as_tensor(op.indices, device=cur.device, dtype=torch.long)
                cur = torch.index_select(cur, dim=bdim, index=idx)
                if cur_mask is not None:
                    cur_mask = cur_mask.index_select(0, idx.cpu())
                continue

            if isinstance(op, _PoolOp):
                axis_idx = cur_dims.index(op.axis)
                if op.axis == "s":
                    if cur_mask is None:
                        raise ValueError("Cannot pool over 's' without seq_mask")
                    cur = _pool_masked(cur, cur_mask, method=op.method, dim=axis_idx)
                    cur_mask = None
                else:
                    if op.method == "mean":
                        cur = cur.mean(dim=axis_idx)
                    elif op.method == "max":
                        cur = cur.max(dim=axis_idx).values
                    elif op.method == "sum":
                        cur = cur.sum(dim=axis_idx)
                    elif op.method == "last":
                        cur = cur.select(axis_idx, cur.shape[axis_idx] - 1)
                    else:
                        raise ValueError(f"Unknown pool method: {op.method}")

                if op.axis == "l":
                    cur_layers = None
                cur_dims = cur_dims.replace(op.axis, "")
                continue

            if isinstance(op, _CastOp):
                cur = cur.to(op.dtype)
                continue

            if isinstance(op, _ToDeviceOp):
                cur = cur.to(op.device)
                if cur_mask is not None:
                    cur_mask = cur_mask.to(op.device)
                continue

            raise TypeError(f"Unknown op type: {type(op)}")

        return cur, cur_dims, cur_mask, cur_layers, cur.dtype, cur.device

    def _materialize_tensor_source(self) -> tuple[torch.Tensor, str, torch.Tensor | None, tuple[int, ...] | None]:
        assert isinstance(self._source, _TensorSource)
        return self._source.tensor, self._source.dims, self._source.seq_mask, self._source.layer_ids

    def _extract_batch_model(
        self,
        model: "PreTrainedModel",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layers: list[int],
    ) -> torch.Tensor:
        batch_gpu = {
            "input_ids": input_ids.to(model.device),
            "attention_mask": attention_mask.to(model.device),
        }
        with HookedModel(model, layers, detach_activations=True) as hooked:
            return hooked.get_activations(batch_gpu).cpu()

    def _materialize_model_source(
        self,
        pushdown: tuple[_Op, ...],
    ) -> tuple[torch.Tensor, str, torch.Tensor | None, tuple[int, ...] | None, tuple[_Op, ...]]:
        assert isinstance(self._source, _ModelSource)

        tokens = self._source.tokens
        layer_ids = list(self._source.layers)
        push_pool: _PoolOp | None = None

        for op in pushdown:
            if isinstance(op, _SelectLayersOp):
                layer_ids = list(op.layer_ids)
            elif isinstance(op, _SliceBatchOp):
                tokens = tokens[list(op.indices)]
            elif isinstance(op, _PoolOp):
                push_pool = op

        n = len(tokens)
        n_layers = len(layer_ids)
        hidden = _hidden_dim(self._source.model)

        if push_pool is None:
            out = torch.zeros(n, n_layers, tokens.seq_len, hidden, dtype=self._source.dtype)
            out_dims = "blsh"
            out_mask: torch.Tensor | None = tokens.detection_mask
        else:
            out = torch.zeros(n, n_layers, hidden, dtype=self._source.dtype)
            out_dims = "blh"
            out_mask = None

        bs = self._source.batch_size
        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch_ids = list(range(start, end))
            input_ids = tokens.input_ids[batch_ids]
            attn = tokens.attention_mask[batch_ids]

            seq_lens = attn.sum(1)
            slen = int(seq_lens.max().item())
            sl = slice(None, slen) if tokens.padding_side == "right" else slice(-slen, None)

            input_ids_b = input_ids[..., sl]
            attn_b = attn[..., sl]
            det_b = tokens.detection_mask[batch_ids][..., sl]

            acts_lbsd = self._extract_batch_model(self._source.model, input_ids_b, attn_b, layer_ids)
            acts_blsd = acts_lbsd.transpose(0, 1)

            if push_pool is None:
                if tokens.padding_side == "right":
                    out[start:end, :, :slen] = acts_blsd
                else:
                    out[start:end, :, -slen:] = acts_blsd
            else:
                seq_dim = 2
                out[start:end] = _pool_masked(acts_blsd, det_b, method=push_pool.method, dim=seq_dim)

        return out, out_dims, out_mask, tuple(layer_ids), self._ops[len(pushdown):]

    def _materialize_disk_source(
        self,
        pushdown: tuple[_Op, ...],
    ) -> tuple[torch.Tensor, str, torch.Tensor | None, tuple[int, ...] | None, tuple[_Op, ...]]:
        assert isinstance(self._source, _DiskSource)

        try:
            import h5py
            import numpy as np
        except ImportError as exc:
            raise ImportError("h5py is required for disk-backed Acts. Install with: pip install probelab[storage]") from exc

        batch_indices: tuple[int, ...] | None = None
        layer_ids: tuple[int, ...] | None = self._layer_ids
        pool_op: _PoolOp | None = None
        for op in pushdown:
            if isinstance(op, _SliceBatchOp):
                batch_indices = op.indices
            elif isinstance(op, _SelectLayersOp):
                layer_ids = op.layer_ids
            elif isinstance(op, _PoolOp):
                if op.axis != "s":
                    raise ValueError(f"Disk pushdown only supports pooling over 's', got axis={op.axis!r}")
                pool_op = op

        with h5py.File(self._source.path, "r") as f:
            ds = f["acts"]
            base_dims = _normalize_dims(str(f.attrs["dims"]))

            if "l" in base_dims:
                all_layer_ids = tuple(int(x) for x in f["layer_ids"][:]) if "layer_ids" in f else tuple(range(ds.shape[base_dims.index("l")]))
            else:
                all_layer_ids = None

            if batch_indices is None:
                batch_key: slice | list[int] = slice(None)
            else:
                batch_key = list(batch_indices)

            if "l" in base_dims and layer_ids is not None:
                selected_arrays = []
                for layer_id in layer_ids:
                    if all_layer_ids is None or layer_id not in all_layer_ids:
                        raise ValueError(f"Layer {layer_id} not found in disk cache")
                    pos = all_layer_ids.index(layer_id)
                    selected_arrays.append(ds[batch_key, pos])
                arr = np.stack(selected_arrays, axis=1)
            else:
                arr = ds[batch_key]
                if "l" in base_dims:
                    layer_ids = all_layer_ids

            tensor = torch.from_numpy(arr)

            seq_mask = None
            if "seq_mask" in f:
                seq_np = f["seq_mask"][batch_key]
                seq_mask = torch.from_numpy(seq_np).float()

        out_dims = base_dims
        if pool_op is not None:
            if "s" not in out_dims:
                raise ValueError(f"Cannot pool over 's' for dims {out_dims!r}")
            if seq_mask is None:
                raise ValueError("Cannot pool over 's' without seq_mask in disk source")
            seq_dim = out_dims.index("s")
            tensor = _pool_masked(tensor, seq_mask, method=pool_op.method, dim=seq_dim)
            out_dims = out_dims.replace("s", "")
            seq_mask = None

        return tensor, out_dims, seq_mask, layer_ids, self._ops[len(pushdown):]

    def realize(self) -> torch.Tensor:
        pushdown, rest = self._split_prefix_pushdown()

        if isinstance(self._source, _TensorSource):
            tensor, dims, seq_mask, layer_ids = self._materialize_tensor_source()
            ops_to_apply = self._ops
        elif isinstance(self._source, _ModelSource):
            tensor, dims, seq_mask, layer_ids, rest = self._materialize_model_source(pushdown)
            ops_to_apply = rest
        elif isinstance(self._source, _DiskSource):
            tensor, dims, seq_mask, layer_ids, rest = self._materialize_disk_source(pushdown)
            ops_to_apply = rest
        else:
            raise TypeError(f"Unknown source type: {type(self._source)}")

        out, _, _, _, _, _ = self._apply_ops_to_tensor(tensor, dims, seq_mask, layer_ids, ops_to_apply)
        return out

    def _batch_local_supported(self, ops: tuple[_Op, ...]) -> bool:
        for op in ops:
            if isinstance(op, (_SelectLayersOp, _SliceBatchOp)):
                return False
            if isinstance(op, _PoolOp) and op.axis == "b":
                return False
        return True

    def iter_batches(self, batch_size: int) -> Iterator["Acts"]:
        if "b" not in self._dims:
            raise ValueError(f"iter_batches requires 'b' axis, got dims {self._dims!r}")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        pushdown, rest = self._split_prefix_pushdown()
        if not self._batch_local_supported(rest):
            realized = self.realize()
            bdim = self._dims.index("b")
            if bdim != 0:
                raise ValueError("Unsupported non-leading batch axis in fallback iter_batches")
            for start in range(0, realized.shape[0], batch_size):
                end = min(start + batch_size, realized.shape[0])
                mask = self._seq_mask[start:end] if self._seq_mask is not None else None
                yield Acts(realized[start:end], dims=self._dims, seq_mask=mask, layer_ids=self._layer_ids)
            return

        if isinstance(self._source, _TensorSource):
            tensor, dims, seq_mask, layer_ids = self._materialize_tensor_source()
            tensor, dims, seq_mask, layer_ids, _, _ = self._apply_ops_to_tensor(
                tensor, dims, seq_mask, layer_ids, pushdown
            )
            if dims.index("b") != 0:
                raise ValueError("iter_batches currently requires batch axis at position 0")
            for start in range(0, tensor.shape[0], batch_size):
                end = min(start + batch_size, tensor.shape[0])
                chunk = tensor[start:end]
                chunk_mask = seq_mask[start:end] if seq_mask is not None else None
                out, out_dims, out_mask, out_layers, _, _ = self._apply_ops_to_tensor(
                    chunk, dims, chunk_mask, layer_ids, rest
                )
                yield Acts(out, dims=out_dims, seq_mask=out_mask, layer_ids=out_layers)
            return

        if isinstance(self._source, _DiskSource):
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("h5py is required for disk-backed Acts") from exc

            batch_indices: tuple[int, ...] | None = None
            layer_ids = self._layer_ids
            for op in pushdown:
                if isinstance(op, _SliceBatchOp):
                    batch_indices = op.indices
                elif isinstance(op, _SelectLayersOp):
                    layer_ids = op.layer_ids

            with h5py.File(self._source.path, "r") as f:
                ds = f["acts"]
                base_dims = _normalize_dims(str(f.attrs["dims"]))
                all_layer_ids = None
                if "l" in base_dims:
                    all_layer_ids = tuple(int(x) for x in f["layer_ids"][:]) if "layer_ids" in f else tuple(range(ds.shape[base_dims.index("l")]))

                if batch_indices is None:
                    selected = tuple(range(ds.shape[base_dims.index("b")]))
                else:
                    selected = batch_indices

                mask_all = None
                if "seq_mask" in f:
                    mask_all = torch.from_numpy(f["seq_mask"][:]).float()

                for start in range(0, len(selected), batch_size):
                    end = min(start + batch_size, len(selected))
                    chunk_ids = list(selected[start:end])

                    if "l" in base_dims and layer_ids is not None:
                        per_layer = []
                        for layer_id in layer_ids:
                            if all_layer_ids is None or layer_id not in all_layer_ids:
                                raise ValueError(f"Layer {layer_id} not found in disk cache")
                            pos = all_layer_ids.index(layer_id)
                            per_layer.append(ds[chunk_ids, pos])
                        import numpy as np

                        arr = np.stack(per_layer, axis=1)
                        chunk = torch.from_numpy(arr)
                        chunk_layers = layer_ids
                    else:
                        chunk = torch.from_numpy(ds[chunk_ids])
                        chunk_layers = all_layer_ids if "l" in base_dims else None

                    chunk_mask = mask_all[chunk_ids] if mask_all is not None else None
                    out, out_dims, out_mask, out_layers, _, _ = self._apply_ops_to_tensor(
                        chunk, base_dims, chunk_mask, chunk_layers, rest
                    )
                    yield Acts(out, dims=out_dims, seq_mask=out_mask, layer_ids=out_layers)
            return

        if isinstance(self._source, _ModelSource):
            assert isinstance(self._source, _ModelSource)

            tokens = self._source.tokens
            layer_ids = list(self._source.layers)
            pool_op: _PoolOp | None = None
            for op in pushdown:
                if isinstance(op, _SelectLayersOp):
                    layer_ids = list(op.layer_ids)
                elif isinstance(op, _SliceBatchOp):
                    tokens = tokens[list(op.indices)]
                elif isinstance(op, _PoolOp):
                    pool_op = op

            n = len(tokens)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_ids = list(range(start, end))
                input_ids = tokens.input_ids[batch_ids]
                attn = tokens.attention_mask[batch_ids]
                seq_lens = attn.sum(1)
                slen = int(seq_lens.max().item())
                sl = slice(None, slen) if tokens.padding_side == "right" else slice(-slen, None)

                input_ids_b = input_ids[..., sl]
                attn_b = attn[..., sl]
                det_b = tokens.detection_mask[batch_ids][..., sl]
                acts_lbsd = self._extract_batch_model(self._source.model, input_ids_b, attn_b, layer_ids)
                acts_blsd = acts_lbsd.transpose(0, 1)

                if pool_op is None:
                    chunk = acts_blsd
                    chunk_dims = "blsh"
                    chunk_mask = det_b
                else:
                    chunk = _pool_masked(acts_blsd, det_b, method=pool_op.method, dim=2)
                    chunk_dims = "blh"
                    chunk_mask = None

                out, out_dims, out_mask, out_layers, _, _ = self._apply_ops_to_tensor(
                    chunk, chunk_dims, chunk_mask, tuple(layer_ids), rest
                )
                yield Acts(out, dims=out_dims, seq_mask=out_mask, layer_ids=out_layers)
            return

        raise TypeError(f"Unknown source type: {type(self._source)}")

    def iter_layers(self) -> Iterator[tuple[int, torch.Tensor]]:
        if "l" not in self._dims:
            raise ValueError(f"iter_layers requires 'l' axis, got dims {self._dims!r}")

        layer_ids = _resolve_layer_ids(self._layer_ids, self._shape[self._dims.index("l")])
        for layer_id in layer_ids:
            layer_plan = self.select_layers(layer_id)
            tensor = layer_plan.realize()
            ldim = layer_plan.dims.index("l")
            yield layer_id, tensor.select(ldim, 0)

    def save(self, path: str | Path, compression: str | None = "gzip", compression_opts: int = 4) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required for Acts.save. Install with: pip install probelab[storage]") from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tensor = self.realize().cpu()
        mask = self._seq_mask.cpu() if self._seq_mask is not None else None

        with h5py.File(path, "w") as f:
            f.create_dataset(
                "acts",
                data=tensor.numpy(),
                compression=compression,
                compression_opts=compression_opts if compression == "gzip" else None,
            )
            if mask is not None:
                f.create_dataset(
                    "seq_mask",
                    data=mask.numpy(),
                    compression=compression,
                    compression_opts=compression_opts if compression == "gzip" else None,
                )
            if self._layer_ids is not None:
                f.create_dataset("layer_ids", data=list(self._layer_ids))

            f.attrs["dims"] = self._dims
            f.attrs["dtype"] = str(tensor.dtype)
            f.attrs["version"] = "acts.v1"

    def cache(self, path: str | Path | None = None) -> "Acts":
        if path is None:
            tensor = self.realize()
            return Acts(tensor, dims=self._dims, seq_mask=self._seq_mask, layer_ids=self._layer_ids)
        self.save(path)
        return Acts.load(path)

    def to_activations(self) -> "Activations":
        from .activations import Activations, Axis, LayerMeta, SequenceMeta

        tensor = self.realize()
        b = tensor.shape[self._dims.index("b")]

        if self._dims == "bh":
            return Activations.from_tensor(tensor)

        if self._dims == "bsh":
            if self._seq_mask is None:
                raise ValueError("Cannot convert bsh Acts to Activations without seq_mask")
            seq = tensor.shape[self._dims.index("s")]
            seq_meta = SequenceMeta(
                attention_mask=self._seq_mask.to(tensor.device).float(),
                detection_mask=self._seq_mask.to(tensor.device).float(),
                input_ids=torch.ones(b, seq, dtype=torch.long, device=tensor.device),
            )
            return Activations(
                activations=tensor,
                axes=(Axis.BATCH, Axis.SEQ, Axis.HIDDEN),
                layer_meta=None,
                sequence_meta=seq_meta,
                batch_indices=torch.arange(b, device=tensor.device),
            )

        if self._dims == "blh":
            ldim = self._dims.index("l")
            layer_ids = _resolve_layer_ids(self._layer_ids, tensor.shape[ldim])
            return Activations(
                activations=tensor,
                axes=(Axis.BATCH, Axis.LAYER, Axis.HIDDEN),
                layer_meta=LayerMeta(indices=layer_ids),
                sequence_meta=None,
                batch_indices=torch.arange(b, device=tensor.device),
            )

        if self._dims == "blsh":
            if self._seq_mask is None:
                raise ValueError("Cannot convert blsh Acts to Activations without seq_mask")
            sdim = self._dims.index("s")
            seq = tensor.shape[sdim]
            layer_ids = _resolve_layer_ids(self._layer_ids, tensor.shape[self._dims.index("l")])
            seq_meta = SequenceMeta(
                attention_mask=self._seq_mask.to(tensor.device).float(),
                detection_mask=self._seq_mask.to(tensor.device).float(),
                input_ids=torch.ones(b, seq, dtype=torch.long, device=tensor.device),
            )
            return Activations(
                activations=tensor,
                axes=(Axis.BATCH, Axis.LAYER, Axis.SEQ, Axis.HIDDEN),
                layer_meta=LayerMeta(indices=layer_ids),
                sequence_meta=seq_meta,
                batch_indices=torch.arange(b, device=tensor.device),
            )

        raise ValueError(
            f"Cannot convert dims {self._dims!r} to Activations. "
            "Supported conversion dims are: bh, bsh, blh, blsh."
        )

    def __repr__(self) -> str:
        src = type(self._source).__name__.replace("_", "")
        return (
            f"Acts(source={src}, dims={self._dims!r}, shape={self._shape}, "
            f"dtype={self._dtype}, device={self._device}, ops={len(self._ops)})"
        )


def collect(
    model: "PreTrainedModel",
    tokens: "Tokens",
    layers: list[int] | range | tuple[int, ...] | int,
    *,
    batch_size: int = 32,
    dtype: torch.dtype | None = None,
    pool: str | None = None,
    pool_dim: int | str = "s",
) -> Acts:
    """Create a lazy model-backed Acts plan."""
    if isinstance(layers, int):
        layer_ids = [layers]
    else:
        layer_ids = list(layers)

    if not layer_ids:
        raise ValueError("layers must be non-empty")

    if dtype is None:
        dtype = getattr(model, "dtype", torch.float32)

    acts = Acts._from_model(
        model=model,
        tokens=tokens,
        layers=layer_ids,
        batch_size=batch_size,
        dtype=dtype,
    )

    if pool is None:
        return acts

    pool_l = pool.lower()
    if pool_l == "mean":
        return acts.mean_pool(dim=pool_dim)
    if pool_l == "max":
        return acts.max_pool(dim=pool_dim)
    if pool_l in {"last", "last_token"}:
        return acts.last_pool(dim=pool_dim)
    if pool_l == "sum":
        return acts.sum_pool(dim=pool_dim)

    raise ValueError(f"Unknown pool method {pool!r}. Expected one of: mean, max, last, sum")


def load(path: str | Path) -> Acts:
    """Load a disk-backed Acts plan."""
    return Acts.load(path)

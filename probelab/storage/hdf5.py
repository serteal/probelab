"""HDF5 persistence for Activations."""

from __future__ import annotations

from collections.abc import Generator
import json

import numpy as np
import torch

from ..activations import Activations

_CAST_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "HDF5 activation storage requires h5py. Install with `probelab[storage]`."
        ) from exc
    return h5py


def _metadata_json(metadata: dict) -> str:
    try:
        return json.dumps(metadata, sort_keys=True)
    except TypeError as exc:
        raise TypeError(
            "Activations metadata must be JSON-serializable for HDF5 storage."
        ) from exc


def _load_metadata_json(value: object) -> dict:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def save(
    activations: Activations,
    path: str,
    *,
    dtype: str = "bfloat16",
    compression: str | None = None,
    compression_opts: int | None = None,
) -> None:
    """Save activations to an HDF5 file."""
    h5py = _import_h5py()

    data_cpu = activations.data.cpu()
    if dtype == "bfloat16":
        data_np = data_cpu.bfloat16().view(torch.uint16).numpy()
        dtype_str = "torch.bfloat16"
    elif dtype == "float32":
        data_np = data_cpu.float().numpy()
        dtype_str = "torch.float32"
    else:
        raise ValueError(f"dtype must be 'bfloat16' or 'float32', got {dtype!r}")

    chunk_shape = None
    if "s" in activations.dims:
        chunk_shape = (min(4096, data_np.shape[0]),) + data_np.shape[1:]

    comp_kwargs = {}
    if compression:
        comp_kwargs["compression"] = compression
        if compression_opts is not None:
            comp_kwargs["compression_opts"] = compression_opts

    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data_np, chunks=chunk_shape, **comp_kwargs)
        handle.attrs["dims"] = activations.dims
        handle.attrs["dtype"] = dtype_str
        handle.attrs["metadata"] = _metadata_json(activations.metadata)
        handle.attrs["probelab_version"] = "0.1.0"

        if activations.layers is not None:
            handle.create_dataset("layers", data=list(activations.layers))
        if activations.offsets is not None:
            handle.create_dataset("offsets", data=activations.offsets.cpu().numpy())
        if activations.detection_mask is not None:
            handle.create_dataset(
                "detection_mask",
                data=activations.detection_mask.cpu().numpy(),
            )


def load(
    path: str,
    *,
    device: str = "cpu",
    layers: int | list[int] | None = None,
    cast: str | None = None,
) -> Activations:
    """Load activations from an HDF5 file."""
    h5py = _import_h5py()
    target_dtype = _CAST_MAP[cast] if cast else None

    with h5py.File(path, "r") as handle:
        stored_dtype = handle.attrs.get("dtype", "")
        if "bfloat16" in stored_dtype:
            raw_np = np.asarray(handle["data"][:], dtype=np.int16)
            data = torch.from_numpy(raw_np).view(torch.bfloat16)
        else:
            data = torch.tensor(handle["data"][:])
        if target_dtype is not None:
            data = data.to(target_dtype)
        data = data.to(device)

        dims = str(handle.attrs["dims"])
        stored_layers = (
            tuple(int(x) for x in handle["layers"][:])
            if "layers" in handle
            else None
        )
        offsets = (
            torch.tensor(handle["offsets"][:], dtype=torch.int64, device=device)
            if "offsets" in handle
            else None
        )
        detection_mask = (
            torch.tensor(
                handle["detection_mask"][:],
                dtype=torch.bool,
                device=device,
            )
            if "detection_mask" in handle
            else None
        )
        metadata = (
            _load_metadata_json(handle.attrs["metadata"])
            if "metadata" in handle.attrs
            else {}
        )

    activations = Activations(
        data=data,
        dims=dims,
        offsets=offsets,
        detection_mask=detection_mask,
        layers=stored_layers,
        metadata=metadata,
    )
    if layers is not None and "l" in dims:
        activations = activations.select("l", layers)
    return activations


def stream(
    path: str,
    *,
    chunk_tokens: int = 100_000,
    cast: str | None = None,
) -> Generator[tuple[Activations, list[int]], None, None]:
    """Yield activation chunks from an HDF5 file without loading it all."""
    h5py = _import_h5py()
    target_dtype = _CAST_MAP[cast] if cast else None

    with h5py.File(path, "r") as handle:
        offsets_np = handle["offsets"][:]
        stored_dtype = handle.attrs.get("dtype", "")
        dims = str(handle.attrs["dims"])
        stored_layers = (
            tuple(int(x) for x in handle["layers"][:])
            if "layers" in handle
            else None
        )
        data_ds = handle["data"]
        detection_mask_ds = handle["detection_mask"]
        metadata = (
            _load_metadata_json(handle.attrs["metadata"])
            if "metadata" in handle.attrs
            else {}
        )
        n_samples = len(offsets_np) - 1

        sample_idx = 0
        while sample_idx < n_samples:
            chunk_start = sample_idx
            tok_start = int(offsets_np[chunk_start])
            chunk_end = chunk_start
            while chunk_end < n_samples:
                next_tok = int(offsets_np[chunk_end + 1])
                if next_tok - tok_start > chunk_tokens and chunk_end > chunk_start:
                    break
                chunk_end += 1

            tok_end = int(offsets_np[chunk_end])
            n_tokens = tok_end - tok_start
            if n_tokens == 0:
                sample_idx = chunk_end
                continue

            raw = data_ds[tok_start:tok_end]
            if "bfloat16" in stored_dtype:
                raw_tensor = torch.from_numpy(np.asarray(raw, dtype=np.int16))
                chunk_data = raw_tensor.view(torch.bfloat16)
            else:
                chunk_data = torch.from_numpy(np.asarray(raw))
            if target_dtype is not None:
                chunk_data = chunk_data.to(target_dtype)

            chunk_mask = torch.from_numpy(
                np.asarray(detection_mask_ds[tok_start:tok_end], dtype=bool)
            )
            local_offsets = torch.zeros(chunk_end - chunk_start + 1, dtype=torch.int64)
            for i in range(chunk_start, chunk_end):
                local_offsets[i - chunk_start + 1] = int(offsets_np[i + 1]) - tok_start

            indices = list(range(chunk_start, chunk_end))
            yield Activations(
                data=chunk_data,
                dims=dims,
                offsets=local_offsets,
                detection_mask=chunk_mask,
                layers=stored_layers,
                metadata=metadata,
            ), indices
            sample_idx = chunk_end

"""Memmap persistence for flat multilayer Activations."""

from __future__ import annotations

from collections.abc import Generator
import json
import os
from pathlib import Path
import shutil
import warnings

import numpy as np
import torch

from ..activations import Activations


def _json_metadata(metadata: dict) -> dict:
    try:
        return json.loads(json.dumps(metadata, sort_keys=True))
    except TypeError as exc:
        raise TypeError(
            "Activations metadata must be JSON-serializable for memmap storage."
        ) from exc


def save(activations: Activations, dir_path: str) -> None:
    """Save flat multilayer activations as per-layer memmap files."""
    path = Path(dir_path)
    if "s" not in activations.dims or "l" not in activations.dims:
        raise ValueError(
            f"memmap save requires dims with 's' and 'l', got {activations.dims!r}"
        )
    if activations.layers is None:
        raise ValueError("memmap save requires layer ids")
    if activations.offsets is None or activations.detection_mask is None:
        raise ValueError("memmap save requires offsets and detection_mask")

    data_cpu = activations.data.cpu()
    total_tokens, n_layers, hidden = data_cpu.shape

    tmp_dir = path.parent / f"{path.name}._tmp_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    offsets_mm = np.memmap(
        tmp_dir / "offsets.bin",
        dtype=np.int64,
        mode="w+",
        shape=activations.offsets.shape,
    )
    offsets_mm[:] = activations.offsets.cpu().numpy()
    offsets_mm.flush()
    del offsets_mm

    mask_mm = np.memmap(
        tmp_dir / "detection_mask.bin",
        dtype=np.uint8,
        mode="w+",
        shape=(total_tokens,),
    )
    mask_mm[:] = activations.detection_mask.cpu().numpy().astype(np.uint8)
    mask_mm.flush()
    del mask_mm

    data_int16 = data_cpu.bfloat16().view(torch.int16).numpy()
    for layer_pos, layer_id in enumerate(activations.layers):
        layer_data = data_int16[:, layer_pos, :]
        layer_mm = np.memmap(
            tmp_dir / f"layer_{layer_id}.bin",
            dtype=np.int16,
            mode="w+",
            shape=(total_tokens, hidden),
        )
        layer_mm[:] = layer_data
        layer_mm.flush()
        del layer_mm

    meta = {
        "format": "memmap_v1",
        "dims": activations.dims,
        "total_tokens": total_tokens,
        "n_layers": n_layers,
        "hidden_dim": hidden,
        "layers": list(activations.layers),
        "n_samples": len(activations.offsets) - 1,
        "metadata": _json_metadata(activations.metadata),
    }
    with open(tmp_dir / "meta.json", "w") as handle:
        json.dump(meta, handle)

    if path.exists():
        shutil.rmtree(path)
    tmp_dir.rename(path)


def load(
    dir_path: str,
    *,
    layer: int | list[int] | None = None,
    device: str = "cpu",
) -> Activations:
    """Load activations from a memmap directory."""
    path = Path(dir_path)
    with open(path / "meta.json") as handle:
        meta = json.load(handle)

    total_tokens = meta["total_tokens"]
    hidden = meta["hidden_dim"]
    all_layers = meta["layers"]
    n_samples = meta["n_samples"]
    metadata = meta.get("metadata", {})

    offsets_mm = np.memmap(
        path / "offsets.bin",
        dtype=np.int64,
        mode="r",
        shape=(n_samples + 1,),
    )
    offsets = torch.from_numpy(offsets_mm.copy()).to(device)

    mask_mm = np.memmap(
        path / "detection_mask.bin",
        dtype=np.uint8,
        mode="r",
        shape=(total_tokens,),
    )
    detection_mask = torch.from_numpy(mask_mm.copy()).to(torch.bool).to(device)

    if layer is None:
        load_layers = all_layers
    elif isinstance(layer, int):
        load_layers = [layer]
    else:
        load_layers = list(layer)

    layer_tensors = []
    for layer_id in load_layers:
        if layer_id not in all_layers:
            raise ValueError(f"Layer {layer_id} not in stored layers {all_layers}")
        layer_mm = np.memmap(
            path / f"layer_{layer_id}.bin",
            dtype=np.int16,
            mode="r",
            shape=(total_tokens, hidden),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "The given NumPy array is not writable")
            tensor = torch.from_numpy(layer_mm).view(torch.bfloat16).to(device)
        layer_tensors.append(tensor)

    if isinstance(layer, int):
        return Activations(
            data=layer_tensors[0],
            dims="bsh",
            offsets=offsets,
            detection_mask=detection_mask,
            metadata=metadata,
        )

    return Activations(
        data=torch.stack(layer_tensors, dim=1),
        dims="blsh",
        offsets=offsets,
        detection_mask=detection_mask,
        layers=tuple(load_layers),
        metadata=metadata,
    )


def stream(
    dir_path: str,
    *,
    layer: int | None = None,
    chunk_tokens: int = 500_000,
) -> Generator[tuple[Activations, list[int]], None, None]:
    """Yield activation chunks from memmap files."""
    path = Path(dir_path)
    with open(path / "meta.json") as handle:
        meta = json.load(handle)

    total_tokens = meta["total_tokens"]
    hidden = meta["hidden_dim"]
    all_layers = meta["layers"]
    n_samples = meta["n_samples"]
    metadata = meta.get("metadata", {})

    offsets_mm = np.memmap(
        path / "offsets.bin",
        dtype=np.int64,
        mode="r",
        shape=(n_samples + 1,),
    )
    offsets_np = np.array(offsets_mm)
    mask_mm = np.memmap(
        path / "detection_mask.bin",
        dtype=np.uint8,
        mode="r",
        shape=(total_tokens,),
    )

    load_layers = [layer] if layer is not None else all_layers
    layer_mms = {
        layer_id: np.memmap(
            path / f"layer_{layer_id}.bin",
            dtype=np.int16,
            mode="r",
            shape=(total_tokens, hidden),
        )
        for layer_id in load_layers
    }

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
        if tok_end == tok_start:
            sample_idx = chunk_end
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "The given NumPy array is not writable")
            if layer is not None:
                chunk_data = torch.from_numpy(
                    layer_mms[layer][tok_start:tok_end]
                ).view(torch.bfloat16)
                dims = "bsh"
                stored_layers = None
            else:
                layer_chunks = [
                    torch.from_numpy(layer_mms[layer_id][tok_start:tok_end]).view(
                        torch.bfloat16
                    )
                    for layer_id in all_layers
                ]
                chunk_data = torch.stack(layer_chunks, dim=1)
                dims = "blsh"
                stored_layers = tuple(all_layers)

        chunk_mask = torch.from_numpy(mask_mm[tok_start:tok_end].astype(bool).copy())
        local_offsets = torch.zeros(chunk_end - chunk_start + 1, dtype=torch.int64)
        for i in range(chunk_start, chunk_end):
            local_offsets[i - chunk_start + 1] = int(offsets_np[i + 1]) - tok_start

        yield Activations(
            data=chunk_data,
            dims=dims,
            offsets=local_offsets,
            detection_mask=chunk_mask,
            layers=stored_layers,
            metadata=metadata,
        ), list(range(chunk_start, chunk_end))
        sample_idx = chunk_end


def has_memmap(dir_path: str) -> bool:
    """Return whether a directory contains probelab memmap activations."""
    meta_path = Path(dir_path) / "meta.json"
    if not meta_path.exists():
        return False
    with open(meta_path) as handle:
        meta = json.load(handle)
    return meta.get("format") == "memmap_v1"

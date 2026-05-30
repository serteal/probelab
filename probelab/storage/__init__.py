"""Explicit persistence helpers for Activations.

Two on-disk backends are available:

* **hdf5** — a single ``.h5`` file; general (works for any ``dims``).
  Requires the optional ``h5py`` dependency (``probelab[storage]``).
* **memmap** — a directory of per-layer memmap files; specialised for flat
  multilayer (``"blsh"``) activations and zero-copy partial loads.

Use the backend functions directly (``save_hdf5``/``load_memmap``/...) or the
format-dispatching :func:`save` / :func:`load` / :func:`stream` helpers, which
pick a backend from ``format`` or the path.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

from .hdf5 import load as load_hdf5
from .hdf5 import save as save_hdf5
from .hdf5 import stream as stream_hdf5
from .memmap import has_memmap, load as load_memmap
from .memmap import save as save_memmap
from .memmap import stream as stream_memmap

if TYPE_CHECKING:
    from ..activations import Activations

__all__ = [
    "save",
    "load",
    "stream",
    "save_hdf5",
    "load_hdf5",
    "stream_hdf5",
    "save_memmap",
    "load_memmap",
    "stream_memmap",
    "has_memmap",
]

_HDF5_SUFFIXES = (".h5", ".hdf5", ".he5")


def _resolve_format(path: str, format: str, *, for_load: bool) -> str:
    """Map ``format="auto"`` to a concrete backend from the path."""
    if format not in {"auto", "hdf5", "memmap"}:
        raise ValueError(f"format must be 'auto', 'hdf5', or 'memmap', got {format!r}")
    if format != "auto":
        return format
    if for_load and has_memmap(str(path)):
        return "memmap"
    if str(path).lower().endswith(_HDF5_SUFFIXES):
        return "hdf5"
    if for_load and Path(path).is_dir():
        return "memmap"
    if not for_load and Path(path).suffix == "":
        return "memmap"
    return "hdf5"


def save(activations: "Activations", path: str, *, format: str = "auto", **kwargs) -> None:
    """Save activations, choosing a backend from ``format`` or the path suffix.

    ``format="auto"`` writes HDF5 for ``.h5``/``.hdf5`` paths and memmap for
    suffixless (directory) paths. Backend-specific keyword arguments (e.g.
    ``dtype`` / ``compression`` for HDF5) are forwarded.
    """
    if _resolve_format(path, format, for_load=False) == "memmap":
        save_memmap(activations, path, **kwargs)
    else:
        save_hdf5(activations, path, **kwargs)


def load(path: str, *, format: str = "auto", **kwargs) -> "Activations":
    """Load activations, choosing a backend from ``format`` or the path.

    ``format="auto"`` detects an existing memmap directory, then falls back to
    HDF5. Backend keyword arguments (``layers``/``device``/``cast``) are
    forwarded.
    """
    if _resolve_format(path, format, for_load=True) == "memmap":
        return load_memmap(path, **kwargs)
    return load_hdf5(path, **kwargs)


def stream(
    path: str, *, format: str = "auto", **kwargs
) -> Generator[tuple["Activations", list[int]], None, None]:
    """Stream activation chunks, choosing a backend from ``format`` or the path."""
    if _resolve_format(path, format, for_load=True) == "memmap":
        yield from stream_memmap(path, **kwargs)
    else:
        yield from stream_hdf5(path, **kwargs)

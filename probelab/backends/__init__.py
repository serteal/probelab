"""Backend entry points."""

from .base import ActivationBackend
from .context import Context, get_context_defaults
from .registry import SUPPORTED_BACKENDS, resolve_backend, resolve_backend_name

__all__ = [
    "ActivationBackend",
    "Context",
    "SUPPORTED_BACKENDS",
    "get_context_defaults",
    "resolve_backend",
    "resolve_backend_name",
]


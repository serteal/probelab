"""Context-scoped backend defaults."""

from __future__ import annotations

from contextlib import ContextDecorator
from contextvars import ContextVar, Token
from typing import Any

_BACKEND_DEFAULTS: ContextVar[dict[str, Any]] = ContextVar(
    "probelab_backend_defaults",
    default={},
)


class Context(ContextDecorator):
    """Temporarily set backend defaults within a `with` block."""

    def __init__(self, **kwargs: Any):
        self._kwargs = kwargs
        self._token: Token[dict[str, Any]] | None = None

    def __enter__(self) -> "Context":
        merged = dict(_BACKEND_DEFAULTS.get())
        merged.update(self._kwargs)
        self._token = _BACKEND_DEFAULTS.set(merged)
        return self

    def __exit__(self, *args: object) -> None:
        if self._token is not None:
            _BACKEND_DEFAULTS.reset(self._token)
            self._token = None


def get_context_defaults() -> dict[str, Any]:
    """Return current context defaults."""
    return dict(_BACKEND_DEFAULTS.get())


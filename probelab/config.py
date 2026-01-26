"""Context-based configuration system for probelab.

This module provides thread-safe, context-scoped configuration variables
that can be overridden locally without affecting global state.

Usage:
    import probelab as pl

    # Scoped overrides
    with pl.Context(DEBUG=4, VERBOSE=False, PROFILE=True):
        # ... code runs with these settings ...

    # Check config
    pl.get_config()  # Returns dict of all settings

    # Set global defaults
    pl.set_defaults(DEFAULT_DEVICE="cuda:1")

    # Environment variables also work
    # PROBELAB_DEBUG=4 PROBELAB_VERBOSE=false python script.py
"""

from __future__ import annotations

import os
from contextvars import ContextVar as _ContextVar, Token
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


def _bool_converter(value: str) -> bool:
    """Convert string to boolean."""
    return value.lower() in ("true", "1", "yes", "on")


def _int_converter(value: str) -> int:
    """Convert string to integer."""
    return int(value)


class ConfigVar(Generic[T]):
    """Thread-safe configuration variable with environment variable fallback.

    ConfigVar uses Python's contextvars to provide thread-safe, context-scoped
    configuration. Values can be set via:
    1. Environment variables (PROBELAB_<NAME>)
    2. Global defaults via set_defaults()
    3. Local overrides via Context()

    Args:
        name: Variable name (used for PROBELAB_<name> env var lookup)
        default: Default value if not set
        converter: Optional function to convert env var string to target type
    """

    def __init__(
        self,
        name: str,
        default: T,
        converter: Callable[[str], T] | None = None,
    ):
        self.name = name
        self.default = default
        self.converter = converter
        self._env_key = f"PROBELAB_{name}"
        initial = self._get_initial_value()
        self._var: _ContextVar[T] = _ContextVar(
            f"probelab_{name.lower()}", default=initial
        )

    def _get_initial_value(self) -> T:
        """Get initial value from environment or default."""
        env_value = os.environ.get(self._env_key)
        if env_value is None:
            return self.default
        if self.converter is not None:
            return self.converter(env_value)
        if isinstance(self.default, str):
            return env_value  # type: ignore
        return self.default

    def get(self) -> T:
        """Get current value (respects context overrides)."""
        return self._var.get()

    def set(self, value: T) -> Token[T]:
        """Set value and return token for later reset."""
        return self._var.set(value)

    def reset(self, token: Token[T]) -> None:
        """Reset to value before the given token was created."""
        self._var.reset(token)

    def __repr__(self) -> str:
        return f"ConfigVar({self.name}={self.get()!r})"


# ==============================================================================
# Configuration Variables
# ==============================================================================

LOG_LEVEL = ConfigVar("LOG_LEVEL", "INFO")
"""Logging level for probelab logger. Env: PROBELAB_LOG_LEVEL"""

VERBOSE = ConfigVar("VERBOSE", True, _bool_converter)
"""Whether to show progress bars and verbose output. Env: PROBELAB_VERBOSE"""

DEFAULT_DEVICE = ConfigVar("DEFAULT_DEVICE", "cuda")
"""Default device for probes when not specified. Env: PROBELAB_DEFAULT_DEVICE"""

DISABLE_PROGRESS = ConfigVar("DISABLE_PROGRESS", False, _bool_converter)
"""Disable progress bars even when verbose=True. Env: PROBELAB_DISABLE_PROGRESS"""

DEBUG = ConfigVar("DEBUG", 0, _int_converter)
"""Debug level (0=off, higher=more verbose). Env: PROBELAB_DEBUG"""

PROFILE = ConfigVar("PROFILE", False, _bool_converter)
"""Enable profiling counters and timing. Env: PROBELAB_PROFILE"""

# Registry of all config variables for Context and get_config
_CONFIG_VARS: dict[str, ConfigVar[Any]] = {
    "LOG_LEVEL": LOG_LEVEL,
    "VERBOSE": VERBOSE,
    "DEFAULT_DEVICE": DEFAULT_DEVICE,
    "DISABLE_PROGRESS": DISABLE_PROGRESS,
    "DEBUG": DEBUG,
    "PROFILE": PROFILE,
}


# ==============================================================================
# Context Manager
# ==============================================================================


class Context:
    """Scoped configuration overrides.

    Use as a context manager to temporarily override configuration values.
    Values are restored when exiting the context.

    Example:
        >>> with Context(DEBUG=4, VERBOSE=False):
        ...     print(DEBUG.get())  # 4
        >>> print(DEBUG.get())  # 0 (restored)

        >>> # Nested contexts work correctly
        >>> with Context(DEBUG=1):
        ...     with Context(DEBUG=2):
        ...         print(DEBUG.get())  # 2
        ...     print(DEBUG.get())  # 1
    """

    def __init__(self, **kwargs: Any):
        """Initialize context with overrides.

        Args:
            **kwargs: Config variable names (case-insensitive) and their values.
                     Valid names: LOG_LEVEL, VERBOSE, DEFAULT_DEVICE,
                                 DISABLE_PROGRESS, DEBUG, PROFILE

        Raises:
            ValueError: If an unknown config key is provided
        """
        self._overrides = kwargs
        self._tokens: list[tuple[ConfigVar[Any], Token[Any]]] = []

    def __enter__(self) -> "Context":
        """Enter context and apply overrides."""
        for key, value in self._overrides.items():
            key_upper = key.upper()
            if key_upper not in _CONFIG_VARS:
                available = ", ".join(_CONFIG_VARS.keys())
                raise ValueError(
                    f"Unknown config key: {key}. Available: {available}"
                )
            config_var = _CONFIG_VARS[key_upper]
            token = config_var.set(value)
            self._tokens.append((config_var, token))
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore previous values."""
        # Restore in reverse order to handle nested overrides correctly
        for config_var, token in reversed(self._tokens):
            config_var.reset(token)
        self._tokens.clear()


# ==============================================================================
# Convenience Functions
# ==============================================================================


def get_config() -> dict[str, Any]:
    """Get current configuration as a dictionary.

    Returns:
        Dictionary mapping config variable names to their current values.

    Example:
        >>> get_config()
        {'LOG_LEVEL': 'INFO', 'VERBOSE': True, 'DEFAULT_DEVICE': 'cuda', ...}
    """
    return {name: var.get() for name, var in _CONFIG_VARS.items()}


def set_defaults(**kwargs: Any) -> None:
    """Set global default values for configuration variables.

    These values persist for the rest of the session. Use Context() for
    temporary overrides instead.

    Args:
        **kwargs: Config variable names (case-insensitive) and their values.

    Raises:
        ValueError: If an unknown config key is provided

    Example:
        >>> set_defaults(DEFAULT_DEVICE="cuda:1", VERBOSE=False)
    """
    for key, value in kwargs.items():
        key_upper = key.upper()
        if key_upper not in _CONFIG_VARS:
            available = ", ".join(_CONFIG_VARS.keys())
            raise ValueError(
                f"Unknown config key: {key}. Available: {available}"
            )
        _CONFIG_VARS[key_upper].set(value)

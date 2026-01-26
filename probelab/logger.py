"""Logging utilities for probelab.

Provides a library-local logger named "probelab" without modifying the root logger.
Defaults to console-only logging with no files written on import.

Usage:
    from probelab.logger import logger  # console-only by default
    logger.info("Hello")

Opt-in file logging:
    from probelab.logger import setup_logger
    setup_logger(logfile="probelab.log")  # add file handler

Configure via config:
    import probelab as pl
    with pl.Context(LOG_LEVEL="DEBUG"):
        # Logger will use DEBUG level
"""

import logging
from pathlib import Path

from .config import LOG_LEVEL

_DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _ensure_console_handler(logger: logging.Logger, level: str) -> None:
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(
            h, "_probelab_console", False
        ):
            h.setLevel(level)
            return
    console = logging.StreamHandler()
    console.setLevel(level)
    fmt = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
    console.setFormatter(fmt)
    # Mark to avoid duplicate additions
    console._probelab_console = True  # type: ignore[attr-defined]
    logger.addHandler(console)


def _has_file_handler(logger: logging.Logger, path: Path) -> bool:
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path:  # type: ignore[attr-defined]
            return True
    return False


def setup_logger(
    *,
    logfile: str | Path | None = None,
    level: str | None = None,
    file_level: str = "WARNING",
) -> logging.Logger:
    """Configure the library-local "probelab" logger.

    - By default (logfile=None) configures console-only logging and does NOT write files.
    - If `logfile` is provided, adds a file handler at `file_level`.
    - If `level` is None, uses LOG_LEVEL from config (default: "INFO").

    Returns the configured "probelab" logger.
    """
    if level is None:
        level = LOG_LEVEL.get()

    logger = logging.getLogger("probelab")
    logger.setLevel(level)
    logger.propagate = False

    _ensure_console_handler(logger, level)

    if logfile is not None:
        log_path = Path(logfile)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not _has_file_handler(logger, log_path):
            file_h = logging.FileHandler(log_path)
            file_h.setLevel(file_level)
            fmt = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
            file_h.setFormatter(fmt)
            logger.addHandler(file_h)

    return logger


# Configure library-local console-only logging on import (no file writes by default)
logger = setup_logger()

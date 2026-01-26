"""First-class profiling system for probelab.

This module provides built-in counters and timing with zero overhead when disabled.
Enable profiling via Context(PROFILE=True) or PROBELAB_PROFILE=1 environment variable.

Usage:
    import probelab as pl
    from probelab.profiling import ProbelabCounters, profile_section

    # Enable via context
    with pl.Context(PROFILE=True):
        ProbelabCounters.reset()

        # Run your code...
        acts = pl.collect_activations(...)
        pipeline.fit(acts, labels)

        # Print summary
        ProbelabCounters.print_summary()

    # Custom profiling sections
    with profile_section("my_operation", batch_size=32):
        # ... code ...
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generator

import torch


def _get_profile_flag() -> bool:
    """Check if profiling is enabled via config or env.

    Checks config first, falls back to environment variable.
    """
    try:
        from .config import PROFILE

        return PROFILE.get()
    except ImportError:
        pass
    return os.environ.get("PROBELAB_PROFILE", "").lower() in ("1", "true", "yes", "on")


def is_profiling() -> bool:
    """Check if profiling is currently enabled.

    Returns:
        True if PROBELAB_PROFILE is enabled via config or environment variable.
    """
    return _get_profile_flag()


@dataclass
class ProfileEvent:
    """A single profiled operation."""

    name: str
    start_time: float
    end_time: float
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ProbelabCounters:
    """Global profiling counters with ClassVar storage.

    All counters are class variables, providing a single global namespace
    for profiling data. Use reset() to clear all counters before a profiling
    session.

    Counters:
        - Integer counters: tokens_processed, forward_passes, activations_collected, batches_processed
        - Timing counters (seconds): model_forward_time_s, probe_train_time_s, probe_predict_time_s,
                                     pooling_time_s, total_collection_time_s
        - Memory: gpu_memory_peak_mb

    Example:
        >>> ProbelabCounters.reset()
        >>> # ... run code with PROFILE=True ...
        >>> ProbelabCounters.print_summary()
    """

    # Integer counters
    tokens_processed: ClassVar[int] = 0
    forward_passes: ClassVar[int] = 0
    activations_collected: ClassVar[int] = 0
    batches_processed: ClassVar[int] = 0

    # Timing counters (seconds)
    model_forward_time_s: ClassVar[float] = 0.0
    probe_train_time_s: ClassVar[float] = 0.0
    probe_predict_time_s: ClassVar[float] = 0.0
    pooling_time_s: ClassVar[float] = 0.0
    total_collection_time_s: ClassVar[float] = 0.0

    # Memory tracking
    gpu_memory_peak_mb: ClassVar[float] = 0.0

    # Event log
    _events: ClassVar[list[ProfileEvent]] = []
    _start_wall_time: ClassVar[float | None] = None

    @classmethod
    def reset(cls) -> None:
        """Reset all counters to zero.

        Also resets CUDA peak memory stats if CUDA is available.
        Call this at the start of each profiling session.
        """
        cls.tokens_processed = 0
        cls.forward_passes = 0
        cls.activations_collected = 0
        cls.batches_processed = 0
        cls.model_forward_time_s = 0.0
        cls.probe_train_time_s = 0.0
        cls.probe_predict_time_s = 0.0
        cls.pooling_time_s = 0.0
        cls.total_collection_time_s = 0.0
        cls.gpu_memory_peak_mb = 0.0
        cls._events.clear()
        cls._start_wall_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @classmethod
    def update_gpu_memory(cls) -> None:
        """Update peak GPU memory from CUDA stats.

        Updates gpu_memory_peak_mb with the maximum of current value
        and CUDA's max_memory_allocated.
        """
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            cls.gpu_memory_peak_mb = max(cls.gpu_memory_peak_mb, peak_mb)

    @classmethod
    def get_events(cls) -> list[ProfileEvent]:
        """Get list of all recorded profile events.

        Returns:
            List of ProfileEvent objects from the current session.
        """
        return list(cls._events)

    @classmethod
    def get_summary(cls) -> dict[str, Any]:
        """Get profiling summary as a dictionary.

        Returns:
            Dictionary with all counter values and derived metrics.
        """
        wall_time = time.perf_counter() - (
            cls._start_wall_time or time.perf_counter()
        )
        cls.update_gpu_memory()

        summary = {
            "counts": {
                "tokens_processed": cls.tokens_processed,
                "forward_passes": cls.forward_passes,
                "batches_processed": cls.batches_processed,
                "activations_collected": cls.activations_collected,
            },
            "timing_s": {
                "model_forward": cls.model_forward_time_s,
                "probe_train": cls.probe_train_time_s,
                "probe_predict": cls.probe_predict_time_s,
                "pooling": cls.pooling_time_s,
                "total_collection": cls.total_collection_time_s,
                "wall_clock": wall_time,
            },
            "memory_mb": {
                "gpu_peak": cls.gpu_memory_peak_mb,
            },
        }

        # Add throughput if we have data
        if cls.model_forward_time_s > 0 and cls.tokens_processed > 0:
            summary["throughput"] = {
                "tokens_per_sec": cls.tokens_processed / cls.model_forward_time_s,
            }

        return summary

    @classmethod
    def print_summary(cls) -> None:
        """Print a formatted summary of profiling data."""
        wall_time = time.perf_counter() - (
            cls._start_wall_time or time.perf_counter()
        )
        cls.update_gpu_memory()

        print("\n" + "=" * 60)
        print("PROBELAB PROFILE SUMMARY")
        print("=" * 60)
        print("\n--- Counts ---")
        print(f"  Tokens processed:      {cls.tokens_processed:,}")
        print(f"  Forward passes:        {cls.forward_passes:,}")
        print(f"  Batches processed:     {cls.batches_processed:,}")
        print(f"  Activations collected: {cls.activations_collected:,}")
        print("\n--- Timing ---")
        print(f"  Model forward:         {cls.model_forward_time_s:.3f}s")
        print(f"  Probe training:        {cls.probe_train_time_s:.3f}s")
        print(f"  Probe prediction:      {cls.probe_predict_time_s:.3f}s")
        print(f"  Pooling:               {cls.pooling_time_s:.3f}s")
        print(f"  Total collection:      {cls.total_collection_time_s:.3f}s")
        print(f"  Wall clock:            {wall_time:.3f}s")
        print("\n--- Memory ---")
        print(f"  GPU peak:              {cls.gpu_memory_peak_mb:.1f} MB")
        if cls.model_forward_time_s > 0 and cls.tokens_processed > 0:
            print("\n--- Throughput ---")
            print(
                f"  Tokens/sec:            {cls.tokens_processed / cls.model_forward_time_s:,.0f}"
            )
        print("=" * 60 + "\n")


@contextmanager
def profile_section(
    name: str, **metadata: Any
) -> Generator[dict[str, Any], None, None]:
    """Profile a code section. No-op when PROFILE=False.

    This context manager measures execution time of a code block and records
    it as a ProfileEvent. When profiling is disabled, this is a no-op with
    minimal overhead.

    Args:
        name: Name for this profile section (e.g., "forward_pass", "probe_fit")
        **metadata: Additional metadata to attach to the event

    Yields:
        Context dictionary that can be used to store additional data.
        The dictionary will include 'duration_s' after the context exits.

    Example:
        >>> with profile_section("my_operation", batch_size=32) as ctx:
        ...     result = expensive_operation()
        ...     ctx["result_size"] = len(result)
        >>> # ctx now has duration_s set
    """
    if not _get_profile_flag():
        yield {}
        return

    # Synchronize CUDA before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    ctx: dict[str, Any] = dict(metadata)
    start = time.perf_counter()

    try:
        yield ctx
    finally:
        # Synchronize CUDA after timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        duration = end - start
        ctx["duration_s"] = duration

        event = ProfileEvent(
            name=name,
            start_time=start,
            end_time=end,
            duration_s=duration,
            metadata=ctx,
        )
        ProbelabCounters._events.append(event)

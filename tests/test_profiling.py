"""Tests for probelab profiling module."""

import time

import pytest

from probelab.config import Context
from probelab.profiling import (
    ProbelabCounters,
    ProfileEvent,
    is_profiling,
    profile_section,
)


class TestIsProfiing:
    """Tests for is_profiling function."""

    def test_default_is_false(self):
        # By default, profiling should be disabled
        # This assumes no env var is set
        assert is_profiling() is False

    def test_with_context_enabled(self):
        with Context(PROFILE=True):
            assert is_profiling() is True

    def test_with_context_disabled(self):
        with Context(PROFILE=False):
            assert is_profiling() is False

    def test_nested_contexts(self):
        with Context(PROFILE=True):
            assert is_profiling() is True
            with Context(PROFILE=False):
                assert is_profiling() is False
            assert is_profiling() is True


class TestProfileEvent:
    """Tests for ProfileEvent dataclass."""

    def test_creation(self):
        event = ProfileEvent(
            name="test_event",
            start_time=1.0,
            end_time=2.0,
            duration_s=1.0,
        )
        assert event.name == "test_event"
        assert event.start_time == 1.0
        assert event.end_time == 2.0
        assert event.duration_s == 1.0
        assert event.metadata == {}

    def test_creation_with_metadata(self):
        event = ProfileEvent(
            name="test_event",
            start_time=0.0,
            end_time=1.0,
            duration_s=1.0,
            metadata={"batch_size": 32},
        )
        assert event.metadata == {"batch_size": 32}


class TestProbelabCounters:
    """Tests for ProbelabCounters class."""

    def test_reset(self):
        # Set some values
        ProbelabCounters.tokens_processed = 100
        ProbelabCounters.forward_passes = 10

        # Reset
        ProbelabCounters.reset()

        # Check values are zeroed
        assert ProbelabCounters.tokens_processed == 0
        assert ProbelabCounters.forward_passes == 0
        assert ProbelabCounters.activations_collected == 0
        assert ProbelabCounters.batches_processed == 0
        assert ProbelabCounters.model_forward_time_s == 0.0
        assert ProbelabCounters.probe_train_time_s == 0.0
        assert ProbelabCounters.probe_predict_time_s == 0.0
        assert ProbelabCounters.pooling_time_s == 0.0
        assert ProbelabCounters.total_collection_time_s == 0.0
        assert ProbelabCounters.gpu_memory_peak_mb == 0.0
        assert len(ProbelabCounters._events) == 0

    def test_get_events(self):
        ProbelabCounters.reset()

        # Add some events manually
        event1 = ProfileEvent("event1", 0.0, 1.0, 1.0)
        event2 = ProfileEvent("event2", 1.0, 2.0, 1.0)
        ProbelabCounters._events.append(event1)
        ProbelabCounters._events.append(event2)

        events = ProbelabCounters.get_events()
        assert len(events) == 2
        assert events[0].name == "event1"
        assert events[1].name == "event2"

    def test_get_summary(self):
        ProbelabCounters.reset()
        ProbelabCounters.tokens_processed = 1000
        ProbelabCounters.forward_passes = 10
        ProbelabCounters.batches_processed = 10
        ProbelabCounters.model_forward_time_s = 1.0

        summary = ProbelabCounters.get_summary()

        assert "counts" in summary
        assert "timing_s" in summary
        assert "memory_mb" in summary
        assert summary["counts"]["tokens_processed"] == 1000
        assert summary["counts"]["forward_passes"] == 10

        # Should have throughput since we have forward time
        assert "throughput" in summary
        assert summary["throughput"]["tokens_per_sec"] == 1000.0

    def test_get_summary_no_throughput_when_no_data(self):
        ProbelabCounters.reset()
        summary = ProbelabCounters.get_summary()

        # No throughput without data
        assert "throughput" not in summary

    def test_print_summary(self, capsys):
        ProbelabCounters.reset()
        ProbelabCounters.tokens_processed = 500
        ProbelabCounters.forward_passes = 5

        ProbelabCounters.print_summary()

        captured = capsys.readouterr()
        assert "PROBELAB PROFILE SUMMARY" in captured.out
        assert "500" in captured.out
        assert "5" in captured.out


class TestProfileSection:
    """Tests for profile_section context manager."""

    def test_noop_when_disabled(self):
        ProbelabCounters.reset()

        with Context(PROFILE=False):
            with profile_section("test_section") as ctx:
                time.sleep(0.01)

        # Should be empty dict
        assert ctx == {}
        # No events recorded
        assert len(ProbelabCounters._events) == 0

    def test_records_when_enabled(self):
        ProbelabCounters.reset()

        with Context(PROFILE=True):
            with profile_section("test_section") as ctx:
                time.sleep(0.01)

        # Should have duration
        assert "duration_s" in ctx
        assert ctx["duration_s"] > 0

        # Should have event recorded
        assert len(ProbelabCounters._events) == 1
        assert ProbelabCounters._events[0].name == "test_section"

    def test_metadata_passed_through(self):
        ProbelabCounters.reset()

        with Context(PROFILE=True):
            with profile_section("test_section", batch_size=32, layers=[16]) as ctx:
                pass

        assert ctx.get("batch_size") == 32
        assert ctx.get("layers") == [16]

        event = ProbelabCounters._events[0]
        assert event.metadata.get("batch_size") == 32
        assert event.metadata.get("layers") == [16]

    def test_ctx_can_be_modified(self):
        ProbelabCounters.reset()

        with Context(PROFILE=True):
            with profile_section("test_section") as ctx:
                ctx["custom_key"] = "custom_value"

        assert ctx.get("custom_key") == "custom_value"

    def test_duration_is_accurate(self):
        ProbelabCounters.reset()

        with Context(PROFILE=True):
            with profile_section("test_section") as ctx:
                time.sleep(0.05)

        # Should be at least 50ms, allowing for some timing variance
        assert ctx["duration_s"] >= 0.04

    def test_exception_still_records(self):
        ProbelabCounters.reset()

        with Context(PROFILE=True):
            try:
                with profile_section("test_section") as ctx:
                    raise ValueError("test error")
            except ValueError:
                pass

        # Event should still be recorded
        assert len(ProbelabCounters._events) == 1
        assert "duration_s" in ctx


class TestIntegrationWithProbelab:
    """Integration tests with probelab imports."""

    def test_profiling_via_context(self):
        import probelab as pl

        ProbelabCounters.reset()

        with pl.Context(PROFILE=True):
            assert is_profiling() is True

            with profile_section("test_section"):
                pass

        assert len(ProbelabCounters._events) == 1


class TestCounterAccumulation:
    """Tests for counter accumulation during profiling."""

    def test_multiple_sections_accumulate(self):
        ProbelabCounters.reset()

        with Context(PROFILE=True):
            for i in range(3):
                with profile_section(f"section_{i}"):
                    time.sleep(0.01)

        assert len(ProbelabCounters._events) == 3

    def test_nested_sections_work(self):
        ProbelabCounters.reset()

        with Context(PROFILE=True):
            with profile_section("outer"):
                with profile_section("inner"):
                    time.sleep(0.01)

        assert len(ProbelabCounters._events) == 2

        # Find events by name
        outer = next(e for e in ProbelabCounters._events if e.name == "outer")
        inner = next(e for e in ProbelabCounters._events if e.name == "inner")

        # Outer should have longer duration than inner
        assert outer.duration_s >= inner.duration_s

"""Tests for standard benchmark registry."""

import pytest

import probelib as pl
from probelib.benchmarks import Benchmark, get_benchmark, list_benchmarks


class TestStandardBenchmarks:
    """Tests for standard benchmark factory functions."""

    def test_deception_suite(self):
        """Test deception_suite benchmark."""
        benchmark = pl.benchmarks.standard.deception_suite()

        assert isinstance(benchmark, Benchmark)
        assert benchmark.name == "Deception Detection Suite v1"
        assert benchmark.default_model == "meta-llama/Llama-2-7b-chat-hf"
        assert benchmark.training is not None
        assert len(benchmark.training.datasets) == 3
        assert len(benchmark.evaluations) == 4

    def test_deception_suite_custom_model(self):
        """Test deception_suite with custom model."""
        benchmark = pl.benchmarks.standard.deception_suite(
            model="meta-llama/Llama-3-8b-Instruct"
        )

        assert benchmark.default_model == "meta-llama/Llama-3-8b-Instruct"

    def test_harmfulness_suite(self):
        """Test harmfulness_suite benchmark."""
        benchmark = pl.benchmarks.standard.harmfulness_suite()

        assert isinstance(benchmark, Benchmark)
        assert benchmark.name == "Harmfulness Detection Suite v1"
        assert benchmark.training is not None
        assert len(benchmark.training.datasets) == 2
        assert len(benchmark.evaluations) == 4

    def test_cross_model_deception(self):
        """Test cross_model_deception benchmark."""
        benchmark = pl.benchmarks.standard.cross_model_deception()

        assert isinstance(benchmark, Benchmark)
        assert benchmark.name == "Cross-Model Deception Generalization"
        assert benchmark.training is not None
        # Should have evaluations with different models
        assert len(benchmark.evaluations) == 3
        # Check that different models are specified
        models = {eval_spec.model for eval_spec in benchmark.evaluations}
        assert len(models) > 1  # Multiple different models


class TestBenchmarkRegistry:
    """Tests for benchmark registry functions."""

    def test_list_benchmarks(self):
        """Test list_benchmarks() returns available benchmarks."""
        benchmarks = list_benchmarks()

        assert isinstance(benchmarks, dict)
        assert "deception_suite" in benchmarks
        assert "harmfulness_suite" in benchmarks
        assert "cross_model_deception" in benchmarks

        # Check descriptions exist
        for name, description in benchmarks.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_get_benchmark(self):
        """Test get_benchmark() returns correct benchmark."""
        benchmark = get_benchmark("deception_suite")

        assert isinstance(benchmark, Benchmark)
        assert benchmark.name == "Deception Detection Suite v1"

    def test_get_benchmark_with_kwargs(self):
        """Test get_benchmark() passes kwargs to factory."""
        benchmark = get_benchmark(
            "deception_suite",
            model="meta-llama/Llama-3-8b-Instruct",
        )

        assert benchmark.default_model == "meta-llama/Llama-3-8b-Instruct"

    def test_get_benchmark_unknown_raises(self):
        """Test get_benchmark() raises on unknown benchmark."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark("nonexistent_benchmark")

    def test_get_benchmark_error_message_shows_available(self):
        """Test that error message shows available benchmarks."""
        with pytest.raises(ValueError, match="Available benchmarks"):
            get_benchmark("fake_benchmark")
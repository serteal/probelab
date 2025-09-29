"""Tests for benchmark specs and results."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

import probelib as pl
from probelib.benchmarks import BenchmarkResults, EvalSpec, TrainSpec


class TestEvalSpec:
    """Tests for EvalSpec dataclass."""

    def test_eval_spec_creation(self):
        """Test creating an EvalSpec."""
        dataset = pl.datasets.AIAuditDataset(split="test")
        spec = EvalSpec(
            dataset=dataset,
            model="meta-llama/Llama-2-7b-chat-hf",
            mask=pl.masks.assistant(),
            metrics=["auroc", "accuracy"],
        )

        assert spec.dataset == dataset
        assert spec.model == "meta-llama/Llama-2-7b-chat-hf"
        assert spec.split == "test"
        assert spec.metrics == ["auroc", "accuracy"]

    def test_eval_spec_default_name(self):
        """Test that default name is generated from dataset."""
        dataset = pl.datasets.AIAuditDataset(split="test")
        spec = EvalSpec(dataset=dataset)

        assert spec.name == "AIAudit (test)"

    def test_eval_spec_custom_name(self):
        """Test custom name override."""
        dataset = pl.datasets.AIAuditDataset(split="test")
        spec = EvalSpec(dataset=dataset, name="Custom Name")

        assert spec.name == "Custom Name"


class TestTrainSpec:
    """Tests for TrainSpec dataclass."""

    def test_train_spec_creation(self):
        """Test creating a TrainSpec."""
        datasets = [
            pl.datasets.AIAuditDataset(split="train"),
            pl.datasets.AILiarDataset(split="train"),
        ]
        spec = TrainSpec(
            datasets=datasets,
            model="meta-llama/Llama-2-7b-chat-hf",
            mask=pl.masks.assistant(),
            probe_config={"layer": 16, "sequence_aggregation": "mean"},
            train_kwargs={"batch_size": 8},
        )

        assert spec.datasets == datasets
        assert spec.model == "meta-llama/Llama-2-7b-chat-hf"
        assert spec.probe_config["layer"] == 16
        assert spec.train_kwargs["batch_size"] == 8

    def test_train_spec_defaults(self):
        """Test TrainSpec with default values."""
        datasets = [pl.datasets.AIAuditDataset(split="train")]
        spec = TrainSpec(datasets=datasets)

        assert spec.model is None
        assert spec.mask is None
        assert spec.probe_config == {}
        assert spec.train_kwargs == {}


class TestBenchmarkResults:
    """Tests for BenchmarkResults class."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        return BenchmarkResults(
            benchmark_name="Test Benchmark",
            results={
                "Eval 1": {"auroc": 0.85, "accuracy": 0.78},
                "Eval 2": {"auroc": 0.82, "accuracy": 0.75},
            },
            probe=None,
            config={"test": "config"},
            timestamp="2024-01-15T10:30:00",
        )

    def test_benchmark_results_creation(self, sample_results):
        """Test creating BenchmarkResults."""
        assert sample_results.benchmark_name == "Test Benchmark"
        assert len(sample_results.results) == 2
        assert sample_results.results["Eval 1"]["auroc"] == 0.85

    def test_summary_dataframe(self, sample_results):
        """Test summary() returns DataFrame."""
        df = sample_results.summary()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Two evaluations
        assert "auroc" in df.columns
        assert "accuracy" in df.columns
        assert df.loc["Eval 1", "auroc"] == 0.85

    def test_summary_rounding(self, sample_results):
        """Test summary() rounding."""
        # Add result with many decimal places
        sample_results.results["Eval 3"] = {"auroc": 0.123456789}
        df = sample_results.summary(round_digits=3)

        assert df.loc["Eval 3", "auroc"] == 0.123

    def test_save_and_load(self, sample_results):
        """Test saving and loading results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_results"

            # Save
            sample_results.save(path)

            # Check files exist
            assert (path / "results.json").exists()
            assert (path / "summary.csv").exists()

            # Load
            loaded = BenchmarkResults.load(path)

            assert loaded.benchmark_name == sample_results.benchmark_name
            assert loaded.results == sample_results.results
            assert loaded.timestamp == sample_results.timestamp
            assert loaded.config == sample_results.config

    def test_compare_single_metric(self):
        """Test comparing two results on a single metric."""
        results1 = BenchmarkResults(
            benchmark_name="Run 1",
            results={
                "Eval A": {"auroc": 0.80, "accuracy": 0.75},
                "Eval B": {"auroc": 0.85, "accuracy": 0.80},
            },
            probe=None,
            config={},
            timestamp="2024-01-15T10:00:00",
        )

        results2 = BenchmarkResults(
            benchmark_name="Run 2",
            results={
                "Eval A": {"auroc": 0.82, "accuracy": 0.77},
                "Eval B": {"auroc": 0.87, "accuracy": 0.82},
            },
            probe=None,
            config={},
            timestamp="2024-01-15T11:00:00",
        )

        comparison = results1.compare(results2, metric="auroc")

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2  # Two common evaluations
        assert "Difference" in comparison.columns

        # Check differences
        assert comparison.loc["Eval A", "Difference"] == 0.02
        assert comparison.loc["Eval B", "Difference"] == 0.02

    def test_compare_all_metrics(self):
        """Test comparing two results on all metrics."""
        results1 = BenchmarkResults(
            benchmark_name="Run 1",
            results={"Eval A": {"auroc": 0.80, "accuracy": 0.75}},
            probe=None,
            config={},
            timestamp="2024-01-15T10:00:00",
        )

        results2 = BenchmarkResults(
            benchmark_name="Run 2",
            results={"Eval A": {"auroc": 0.82, "accuracy": 0.77}},
            probe=None,
            config={},
            timestamp="2024-01-15T11:00:00",
        )

        comparison = results1.compare(results2)

        assert isinstance(comparison, pd.DataFrame)
        # Should have multi-level columns
        assert isinstance(comparison.columns, pd.MultiIndex)

    def test_compare_missing_metric_raises(self):
        """Test that comparing on missing metric raises error."""
        results1 = BenchmarkResults(
            benchmark_name="Run 1",
            results={"Eval A": {"auroc": 0.80}},
            probe=None,
            config={},
            timestamp="2024-01-15T10:00:00",
        )

        results2 = BenchmarkResults(
            benchmark_name="Run 2",
            results={"Eval A": {"accuracy": 0.75}},
            probe=None,
            config={},
            timestamp="2024-01-15T11:00:00",
        )

        with pytest.raises(ValueError, match="Metric.*not found"):
            results1.compare(results2, metric="auroc")

    def test_repr(self, sample_results):
        """Test __repr__ method."""
        repr_str = repr(sample_results)

        assert "BenchmarkResults" in repr_str
        assert "Test Benchmark" in repr_str
        assert "n_evaluations=2" in repr_str
        assert "2024-01-15T10:30:00" in repr_str
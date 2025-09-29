"""Tests for Benchmark class."""

import pytest

import probelib as pl
from probelib.benchmarks import Benchmark, EvalSpec, TrainSpec


class TestBenchmarkCreation:
    """Tests for Benchmark initialization."""

    def test_benchmark_creation(self):
        """Test creating a Benchmark."""
        benchmark = Benchmark(
            name="Test Benchmark",
            default_model="meta-llama/Llama-2-7b-chat-hf",
            evaluations=[
                EvalSpec(
                    dataset=pl.datasets.AIAuditDataset(split="test"),
                    mask=pl.masks.assistant(),
                )
            ],
        )

        assert benchmark.name == "Test Benchmark"
        assert benchmark.default_model == "meta-llama/Llama-2-7b-chat-hf"
        assert len(benchmark.evaluations) == 1
        assert benchmark.training is None

    def test_benchmark_with_training(self):
        """Test creating a Benchmark with training config."""
        benchmark = Benchmark(
            name="Train-Eval Benchmark",
            default_model="meta-llama/Llama-2-7b-chat-hf",
            training=TrainSpec(
                datasets=[pl.datasets.AIAuditDataset(split="train")],
                mask=pl.masks.assistant(),
                probe_config={"layer": 16},
            ),
            evaluations=[
                EvalSpec(
                    dataset=pl.datasets.AIAuditDataset(split="test"),
                    mask=pl.masks.assistant(),
                )
            ],
        )

        assert benchmark.training is not None
        assert len(benchmark.training.datasets) == 1
        assert benchmark.training.probe_config["layer"] == 16

    def test_benchmark_requires_evaluations(self):
        """Test that benchmark requires at least one evaluation."""
        with pytest.raises(ValueError, match="at least one evaluation"):
            Benchmark(
                name="Empty Benchmark",
                evaluations=[],
            )

    def test_benchmark_repr(self):
        """Test __repr__ method."""
        benchmark = Benchmark(
            name="Test",
            evaluations=[
                EvalSpec(dataset=pl.datasets.AIAuditDataset(split="test"))
            ],
        )

        repr_str = repr(benchmark)
        assert "Benchmark" in repr_str
        assert "Test" in repr_str
        assert "n_evaluations=1" in repr_str


class TestBenchmarkValidation:
    """Tests for Benchmark.run() validation."""

    @pytest.fixture
    def simple_benchmark(self):
        """Create a simple benchmark for testing."""
        return Benchmark(
            name="Simple",
            default_model="meta-llama/Llama-2-7b-chat-hf",
            evaluations=[
                EvalSpec(
                    dataset=pl.datasets.AIAuditDataset(split="test"),
                    mask=pl.masks.assistant(),
                )
            ],
        )

    @pytest.fixture
    def train_benchmark(self):
        """Create a benchmark with training."""
        return Benchmark(
            name="Train",
            default_model="meta-llama/Llama-2-7b-chat-hf",
            training=TrainSpec(
                datasets=[pl.datasets.AIAuditDataset(split="train")],
                mask=pl.masks.assistant(),
                probe_config={"layer": 16, "sequence_aggregation": "mean"},
            ),
            evaluations=[
                EvalSpec(
                    dataset=pl.datasets.AIAuditDataset(split="test"),
                    mask=pl.masks.assistant(),
                )
            ],
        )

    def test_run_requires_probe_or_probe_class(self, simple_benchmark):
        """Test that run() requires either probe or probe_class."""
        with pytest.raises(
            ValueError, match="Must provide either 'probe'.*or 'probe_class'"
        ):
            simple_benchmark.run()

    def test_run_cannot_have_both_probe_and_probe_class(self, simple_benchmark):
        """Test that run() cannot have both probe and probe_class."""
        probe = pl.probes.Logistic(layer=16, sequence_aggregation="mean")

        with pytest.raises(ValueError, match="Cannot specify both"):
            simple_benchmark.run(probe=probe, probe_class=pl.probes.Logistic)

    def test_run_probe_class_requires_training(self, simple_benchmark):
        """Test that probe_class requires training config."""
        with pytest.raises(ValueError, match="without training config"):
            simple_benchmark.run(probe_class=pl.probes.Logistic)

    def test_get_config(self, train_benchmark):
        """Test _get_config() method."""
        config = train_benchmark._get_config()

        assert config["name"] == "Train"
        assert config["default_model"] == "meta-llama/Llama-2-7b-chat-hf"
        assert config["n_evaluations"] == 1
        assert "training" in config
        assert config["training"]["n_datasets"] == 1


class TestBenchmarkMultiEval:
    """Tests for benchmarks with multiple evaluations."""

    def test_multiple_evaluations(self):
        """Test benchmark with multiple evaluation specs."""
        benchmark = Benchmark(
            name="Multi-Eval",
            default_model="meta-llama/Llama-2-7b-chat-hf",
            evaluations=[
                EvalSpec(
                    dataset=pl.datasets.AIAuditDataset(split="test"),
                    mask=pl.masks.assistant(),
                    name="AI Audit",
                ),
                EvalSpec(
                    dataset=pl.datasets.AILiarDataset(split="test"),
                    mask=pl.masks.assistant(),
                    name="AI Liar",
                ),
            ],
        )

        assert len(benchmark.evaluations) == 2
        assert benchmark.evaluations[0].name == "AI Audit"
        assert benchmark.evaluations[1].name == "AI Liar"

    def test_per_eval_model_override(self):
        """Test that evaluations can override the default model."""
        benchmark = Benchmark(
            name="Multi-Model",
            default_model="meta-llama/Llama-2-7b-chat-hf",
            evaluations=[
                EvalSpec(
                    dataset=pl.datasets.AIAuditDataset(split="test"),
                    mask=pl.masks.assistant(),
                    # Uses default model
                ),
                EvalSpec(
                    dataset=pl.datasets.AIAuditDataset(split="test"),
                    model="google/gemma-2-9b-it",  # Override
                    mask=pl.masks.assistant(),
                    name="Gemma eval",
                ),
            ],
        )

        assert benchmark.evaluations[0].model is None
        assert benchmark.evaluations[1].model == "google/gemma-2-9b-it"
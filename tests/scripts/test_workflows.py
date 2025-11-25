"""Tests for workflow functions."""

import functools

import pytest
import torch

from probelab import Pipeline
from probelab.metrics import auroc, recall_at_fpr
from probelab.preprocessing import Pool, SelectLayer
from probelab.probes import Logistic, MLP
from probelab.processing.activations import Activations, Axis
from probelab.scripts.workflows import (
    _create_pipeline_without_pooling,
    _detect_collection_strategy_from_pipelines,
    _metric_display_name,
    evaluate_pipelines,
    train_pipelines,
)
from probelab.types import Label


def create_activations(
    n_layers: int = 2,
    batch_size: int = 10,
    seq_len: int = 8,
    d_model: int = 16,
    layer_indices: list[int] | None = None,
) -> Activations:
    """Create test activations with 4D shape."""
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    acts = torch.randn(n_layers, batch_size, seq_len, d_model)
    attention_mask = torch.ones(batch_size, seq_len)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    detection_mask = torch.ones(batch_size, seq_len).bool()

    return Activations.from_components(
        activations=acts,
        attention_mask=attention_mask,
        input_ids=input_ids,
        detection_mask=detection_mask,
        layer_indices=layer_indices,
    )


def create_labels(batch_size: int = 10) -> torch.Tensor:
    """Create random test labels."""
    return torch.randint(0, 2, (batch_size,))


class TestMetricDisplayName:
    """Test _metric_display_name helper."""

    def test_regular_function(self):
        """Test with regular function."""
        name = _metric_display_name(auroc)
        assert name == "auroc"

    def test_partial_function(self):
        """Test with functools.partial."""
        partial_fn = functools.partial(recall_at_fpr, fpr=0.01)
        name = _metric_display_name(partial_fn)
        assert "recall_at_fpr" in name
        assert "0.01" in name

    def test_partial_with_multiple_kwargs(self):
        """Test partial with multiple keyword arguments."""
        partial_fn = functools.partial(recall_at_fpr, fpr=0.05)
        name = _metric_display_name(partial_fn)
        assert "fpr" in name

    def test_lambda_function(self):
        """Test with lambda function."""
        lambda_fn = lambda y, p: 0.5  # noqa: E731
        name = _metric_display_name(lambda_fn)
        # Lambda should return some representation
        assert name is not None


class TestDetectCollectionStrategy:
    """Test _detect_collection_strategy_from_pipelines."""

    def test_single_pipeline_with_mean_pooling(self):
        """Test detection with single pipeline using mean pooling."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        strategy = _detect_collection_strategy_from_pipelines(pipeline)
        assert strategy == "mean"

    def test_single_pipeline_with_max_pooling(self):
        """Test detection with single pipeline using max pooling."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="max")),
            ("probe", Logistic()),
        ])

        strategy = _detect_collection_strategy_from_pipelines(pipeline)
        assert strategy == "max"

    def test_single_pipeline_with_last_token_pooling(self):
        """Test detection with single pipeline using last_token pooling."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="last_token")),
            ("probe", Logistic()),
        ])

        strategy = _detect_collection_strategy_from_pipelines(pipeline)
        assert strategy == "last_token"

    def test_pipeline_without_pooling(self):
        """Test that pipeline without pooling returns None."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("probe", Logistic()),
        ])

        strategy = _detect_collection_strategy_from_pipelines(pipeline)
        assert strategy is None

    def test_pipeline_with_post_transforms(self):
        """Test that pipeline with post-transforms returns None."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("probe", Logistic()),
            ("pool", Pool(dim="sequence", method="mean")),
        ])

        strategy = _detect_collection_strategy_from_pipelines(pipeline)
        assert strategy is None

    def test_multiple_pipelines_same_method(self):
        """Test with multiple pipelines using same method."""
        pipelines = {
            "p1": Pipeline([
                ("select", SelectLayer(0)),
                ("pool", Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "p2": Pipeline([
                ("select", SelectLayer(1)),
                ("pool", Pool(dim="sequence", method="mean")),
                ("probe", MLP()),
            ]),
        }

        strategy = _detect_collection_strategy_from_pipelines(pipelines)
        assert strategy == "mean"

    def test_multiple_pipelines_different_methods(self):
        """Test with multiple pipelines using different methods."""
        pipelines = {
            "p1": Pipeline([
                ("select", SelectLayer(0)),
                ("pool", Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "p2": Pipeline([
                ("select", SelectLayer(1)),
                ("pool", Pool(dim="sequence", method="max")),
                ("probe", Logistic()),
            ]),
        }

        strategy = _detect_collection_strategy_from_pipelines(pipelines)
        assert strategy is None

    def test_pipeline_with_layer_pooling_only(self):
        """Test pipeline with only layer pooling (no sequence pooling)."""
        pipeline = Pipeline([
            ("pool", Pool(dim="layer", method="mean")),
            ("probe", Logistic()),
        ])

        strategy = _detect_collection_strategy_from_pipelines(pipeline)
        assert strategy is None


class TestCreatePipelineWithoutPooling:
    """Test _create_pipeline_without_pooling."""

    def test_removes_sequence_pool(self):
        """Test that sequence Pool step is removed."""
        original = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        modified = _create_pipeline_without_pooling(original)

        # Check Pool step is removed
        step_names = [name for name, _ in modified.steps]
        assert "pool" not in step_names
        assert "select" in step_names
        assert "probe" in step_names

    def test_preserves_other_steps(self):
        """Test that other steps are preserved."""
        original = Pipeline([
            ("select", SelectLayer(5)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        modified = _create_pipeline_without_pooling(original)

        # SelectLayer should still work
        assert modified["select"].layer == 5

    def test_preserves_layer_pool(self):
        """Test that layer Pool is not removed."""
        original = Pipeline([
            ("pool_layer", Pool(dim="layer", method="mean")),
            ("pool_seq", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        modified = _create_pipeline_without_pooling(original)

        step_names = [name for name, _ in modified.steps]
        assert "pool_layer" in step_names
        assert "pool_seq" not in step_names


class TestTrainPipelines:
    """Test train_pipelines function."""

    def test_train_single_pipeline(self):
        """Test training a single pipeline."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        acts = create_activations(n_layers=2, batch_size=20, layer_indices=[0, 1])
        labels = create_labels(batch_size=20)

        train_pipelines(pipeline, acts, labels, verbose=False)

        assert pipeline.get_probe()._fitted

    def test_train_multiple_pipelines(self):
        """Test training multiple pipelines."""
        pipelines = {
            "layer_0": Pipeline([
                ("select", SelectLayer(0)),
                ("pool", Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "layer_1": Pipeline([
                ("select", SelectLayer(1)),
                ("pool", Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
        }

        acts = create_activations(n_layers=2, batch_size=20, layer_indices=[0, 1])
        labels = create_labels(batch_size=20)

        train_pipelines(pipelines, acts, labels, verbose=False)

        for name, pipeline in pipelines.items():
            assert pipeline.get_probe()._fitted, f"Pipeline {name} not fitted"

    def test_train_with_label_list(self):
        """Test training with list of Label enums."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        acts = create_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        labels = [Label.NEGATIVE if i % 2 == 0 else Label.POSITIVE for i in range(10)]

        train_pipelines(pipeline, acts, labels, verbose=False)

        assert pipeline.get_probe()._fitted

    def test_train_with_int_list(self):
        """Test training with list of ints."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        acts = create_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

        train_pipelines(pipeline, acts, labels, verbose=False)

        assert pipeline.get_probe()._fitted


class TestEvaluatePipelines:
    """Test evaluate_pipelines function."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted pipeline."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        acts = create_activations(n_layers=2, batch_size=30, layer_indices=[0, 1])
        labels = create_labels(batch_size=30)

        train_pipelines(pipeline, acts, labels, verbose=False)
        return pipeline

    def test_evaluate_returns_predictions_and_metrics(self, fitted_pipeline):
        """Test that evaluation returns predictions and metrics."""
        acts = create_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        labels = create_labels(batch_size=10)

        predictions, metrics = evaluate_pipelines(
            fitted_pipeline, acts, labels, metrics=["auroc"]
        )

        assert predictions.shape == (10,)
        assert "auroc" in metrics
        assert isinstance(metrics["auroc"], float)

    def test_evaluate_multiple_pipelines(self):
        """Test evaluating multiple pipelines."""
        # Train multiple pipelines
        pipelines = {
            "layer_0": Pipeline([
                ("select", SelectLayer(0)),
                ("pool", Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "layer_1": Pipeline([
                ("select", SelectLayer(1)),
                ("pool", Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
        }

        train_acts = create_activations(n_layers=2, batch_size=30, layer_indices=[0, 1])
        train_labels = create_labels(batch_size=30)
        train_pipelines(pipelines, train_acts, train_labels, verbose=False)

        # Evaluate
        test_acts = create_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        test_labels = create_labels(batch_size=10)

        predictions, metrics = evaluate_pipelines(
            pipelines, test_acts, test_labels, metrics=["auroc"]
        )

        assert "layer_0" in predictions
        assert "layer_1" in predictions
        assert "layer_0" in metrics
        assert "layer_1" in metrics

    def test_evaluate_with_default_metrics(self, fitted_pipeline):
        """Test evaluation with default metrics."""
        acts = create_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        labels = create_labels(batch_size=10)

        predictions, metrics = evaluate_pipelines(fitted_pipeline, acts, labels)

        # Default metrics include auroc and recall@fpr
        assert "auroc" in metrics

    def test_evaluate_with_custom_metrics(self, fitted_pipeline):
        """Test evaluation with custom metric functions."""
        acts = create_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        labels = create_labels(batch_size=10)

        custom_metrics = [
            auroc,
            functools.partial(recall_at_fpr, fpr=0.05),
        ]

        predictions, metrics = evaluate_pipelines(
            fitted_pipeline, acts, labels, metrics=custom_metrics
        )

        assert "auroc" in metrics
        assert any("recall_at_fpr" in key for key in metrics.keys())

    def test_evaluate_with_string_metrics(self, fitted_pipeline):
        """Test evaluation with string metric names."""
        acts = create_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        labels = create_labels(batch_size=10)

        predictions, metrics = evaluate_pipelines(
            fitted_pipeline, acts, labels, metrics=["auroc", "accuracy"]
        )

        assert "auroc" in metrics
        assert "accuracy" in metrics

    def test_evaluate_with_label_enum_list(self, fitted_pipeline):
        """Test evaluation with Label enum list."""
        acts = create_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        labels = [Label.NEGATIVE if i % 2 == 0 else Label.POSITIVE for i in range(10)]

        predictions, metrics = evaluate_pipelines(
            fitted_pipeline, acts, labels, metrics=["auroc"]
        )

        assert predictions.shape == (10,)
        assert "auroc" in metrics

    def test_evaluate_predictions_range(self, fitted_pipeline):
        """Test that predictions are valid probabilities."""
        acts = create_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        labels = create_labels(batch_size=10)

        predictions, metrics = evaluate_pipelines(
            fitted_pipeline, acts, labels, metrics=["auroc"]
        )

        # Predictions should be between 0 and 1
        assert torch.all(predictions >= 0)
        assert torch.all(predictions <= 1)


class TestTrainAndEvaluateIntegration:
    """Integration tests for train + evaluate workflows."""

    def test_full_workflow(self):
        """Test complete train -> evaluate workflow."""
        torch.manual_seed(42)

        # Create pipeline
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        # Create train data
        train_acts = create_activations(n_layers=2, batch_size=50, layer_indices=[0, 1])
        train_labels = create_labels(batch_size=50)

        # Train
        train_pipelines(pipeline, train_acts, train_labels, verbose=False)

        # Create test data
        test_acts = create_activations(n_layers=2, batch_size=20, layer_indices=[0, 1])
        test_labels = create_labels(batch_size=20)

        # Evaluate
        predictions, metrics = evaluate_pipelines(
            pipeline, test_acts, test_labels, metrics=["auroc", "accuracy"]
        )

        assert predictions.shape == (20,)
        assert 0 <= metrics["auroc"] <= 1
        assert 0 <= metrics["accuracy"] <= 1

    def test_workflow_with_mlp_probe(self):
        """Test workflow with MLP probe."""
        torch.manual_seed(42)

        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", MLP(hidden_dim=8, n_epochs=10)),
        ])

        train_acts = create_activations(
            n_layers=2, batch_size=30, d_model=16, layer_indices=[0, 1]
        )
        train_labels = create_labels(batch_size=30)

        train_pipelines(pipeline, train_acts, train_labels, verbose=False)

        test_acts = create_activations(
            n_layers=2, batch_size=10, d_model=16, layer_indices=[0, 1]
        )
        test_labels = create_labels(batch_size=10)

        predictions, metrics = evaluate_pipelines(
            pipeline, test_acts, test_labels, metrics=["auroc"]
        )

        assert predictions.shape == (10,)
        assert "auroc" in metrics

    def test_multi_pipeline_comparison(self):
        """Test comparing multiple pipelines."""
        torch.manual_seed(42)

        pipelines = {
            "mean": Pipeline([
                ("select", SelectLayer(0)),
                ("pool", Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "max": Pipeline([
                ("select", SelectLayer(0)),
                ("pool", Pool(dim="sequence", method="max")),
                ("probe", Logistic()),
            ]),
            "last": Pipeline([
                ("select", SelectLayer(0)),
                ("pool", Pool(dim="sequence", method="last_token")),
                ("probe", Logistic()),
            ]),
        }

        train_acts = create_activations(n_layers=2, batch_size=30, layer_indices=[0, 1])
        train_labels = create_labels(batch_size=30)

        train_pipelines(pipelines, train_acts, train_labels, verbose=False)

        test_acts = create_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        test_labels = create_labels(batch_size=10)

        predictions, metrics = evaluate_pipelines(
            pipelines, test_acts, test_labels, metrics=["auroc"]
        )

        for name in ["mean", "max", "last"]:
            assert name in predictions
            assert name in metrics
            assert "auroc" in metrics[name]

"""Tests for Pipeline class."""

import pytest
import torch

from probelab import Pipeline
from probelab.preprocessing import Pool, SelectLayer
from probelab.preprocessing.base import PreTransformer
from probelab.probes import Logistic, MLP
from probelab.probes.base import BaseProbe
from probelab.processing.activations import Activations, Axis


def create_activations(
    n_layers: int = 2,
    batch_size: int = 10,
    seq_len: int = 8,
    d_model: int = 16,
    layer_indices: list[int] | None = None,
) -> Activations:
    """Create test activations."""
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
    """Create test labels."""
    return torch.randint(0, 2, (batch_size,))


class TestPipelineInit:
    """Test Pipeline initialization and validation."""

    def test_init_with_named_steps(self):
        """Test creating pipeline with named steps."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        assert len(pipeline.steps) == 3
        assert pipeline.steps[0][0] == "select"
        assert pipeline.steps[1][0] == "pool"
        assert pipeline.steps[2][0] == "probe"

    def test_init_validates_probe_exists(self):
        """Test that pipeline requires exactly one probe."""
        with pytest.raises(ValueError, match="must contain exactly one BaseProbe"):
            Pipeline([
                ("select", SelectLayer(0)),
                ("pool", Pool(dim="sequence", method="mean")),
            ])

    def test_init_rejects_multiple_probes(self):
        """Test that pipeline rejects multiple probes."""
        with pytest.raises(ValueError, match="must contain exactly one BaseProbe"):
            Pipeline([
                ("select", SelectLayer(0)),
                ("probe1", Logistic()),
                ("probe2", MLP()),
            ])

    def test_init_rejects_empty_pipeline(self):
        """Test that empty pipeline raises error."""
        with pytest.raises(ValueError, match="Pipeline cannot be empty"):
            Pipeline([])

    def test_init_validates_pre_probe_steps(self):
        """Test that pre-probe steps must be PreTransformers."""

        class NotATransformer:
            pass

        with pytest.raises(ValueError, match="must be a PreTransformer"):
            Pipeline([
                ("bad", NotATransformer()),
                ("probe", Logistic()),
            ])

    def test_init_validates_post_probe_steps(self):
        """Test that post-probe steps must be transforms."""

        class NotATransformer:
            pass

        with pytest.raises(ValueError, match="must be a transform"):
            Pipeline([
                ("probe", Logistic()),
                ("bad", NotATransformer()),
            ])


class TestPipelineFit:
    """Test Pipeline.fit() method."""

    def test_fit_basic(self):
        """Test basic pipeline fitting."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        acts = create_activations(n_layers=2, batch_size=20, layer_indices=[0, 1])
        labels = create_labels(batch_size=20)

        result = pipeline.fit(acts, labels)

        assert result is pipeline  # Returns self
        assert pipeline._probe._fitted

    def test_fit_transforms_through_steps(self):
        """Test that fit transforms activations through all steps."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        # Start with 2 layers, 20 batch, 8 seq, 16 hidden
        acts = create_activations(
            n_layers=2, batch_size=20, seq_len=8, d_model=16, layer_indices=[0, 1]
        )
        labels = create_labels(batch_size=20)

        pipeline.fit(acts, labels)

        # Probe should have been fitted on [batch, hidden] shaped data
        assert pipeline._probe._fitted

    def test_fit_with_label_list(self):
        """Test fitting with list of labels."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        acts = create_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

        pipeline.fit(acts, labels)
        assert pipeline._probe._fitted


class TestPipelinePredict:
    """Test Pipeline.predict() and predict_proba()."""

    @pytest.fixture
    def fitted_pipeline(self):
        """Create a fitted pipeline."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        acts = create_activations(n_layers=2, batch_size=20, layer_indices=[0, 1])
        labels = create_labels(batch_size=20)
        pipeline.fit(acts, labels)

        return pipeline

    def test_predict_proba_returns_probabilities(self, fitted_pipeline):
        """Test that predict_proba returns valid probabilities."""
        acts = create_activations(n_layers=2, batch_size=5, layer_indices=[0, 1])

        probs = fitted_pipeline.predict_proba(acts)

        assert probs.shape == (5, 2)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        # Move expected tensor to same device as probs
        assert torch.allclose(probs.sum(dim=1), torch.ones(5, device=probs.device), atol=1e-6)

    def test_predict_returns_classes(self, fitted_pipeline):
        """Test that predict returns class labels."""
        acts = create_activations(n_layers=2, batch_size=5, layer_indices=[0, 1])

        preds = fitted_pipeline.predict(acts)

        assert preds.shape == (5,)
        assert preds.dtype == torch.long
        assert torch.all((preds == 0) | (preds == 1))

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        acts = create_activations(n_layers=2, batch_size=5, layer_indices=[0, 1])

        with pytest.raises(ValueError, match="must be fitted"):
            pipeline.predict_proba(acts)


class TestPipelineScore:
    """Test Pipeline.score() method."""

    def test_score_returns_accuracy(self):
        """Test that score returns accuracy."""
        torch.manual_seed(42)

        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        # Create separable data
        acts = create_activations(n_layers=2, batch_size=20, layer_indices=[0, 1])
        labels = torch.randint(0, 2, (20,))

        pipeline.fit(acts, labels)
        score = pipeline.score(acts, labels)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestPipelineItemAccess:
    """Test Pipeline step access."""

    def test_getitem_by_name(self):
        """Test accessing steps by name."""
        select_layer = SelectLayer(0)
        pool = Pool(dim="sequence", method="mean")
        probe = Logistic()

        pipeline = Pipeline([
            ("select", select_layer),
            ("pool", pool),
            ("probe", probe),
        ])

        assert pipeline["select"] is select_layer
        assert pipeline["pool"] is pool
        assert pipeline["probe"] is probe

    def test_getitem_by_index(self):
        """Test accessing steps by index."""
        select_layer = SelectLayer(0)
        pool = Pool(dim="sequence", method="mean")
        probe = Logistic()

        pipeline = Pipeline([
            ("select", select_layer),
            ("pool", pool),
            ("probe", probe),
        ])

        assert pipeline[0] is select_layer
        assert pipeline[1] is pool
        assert pipeline[2] is probe

    def test_getitem_invalid_name_raises(self):
        """Test that invalid name raises KeyError."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("probe", Logistic()),
        ])

        with pytest.raises(KeyError, match="not found"):
            pipeline["invalid"]

    def test_get_probe_returns_probe(self):
        """Test get_probe returns the probe step."""
        probe = Logistic()
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", probe),
        ])

        assert pipeline.get_probe() is probe


class TestPipelineWithPostTransforms:
    """Test Pipeline with post-probe transforms."""

    def test_post_probe_pool(self):
        """Test pipeline with pool after probe (token-level training)."""
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("probe", Logistic()),
            ("pool", Pool(dim="sequence", method="mean")),
        ])

        acts = create_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        labels = create_labels(batch_size=10)

        pipeline.fit(acts, labels)

        # Predict should return sequence-level predictions
        test_acts = create_activations(n_layers=2, batch_size=5, layer_indices=[0, 1])
        probs = pipeline.predict_proba(test_acts)

        assert probs.shape == (5, 2)


class TestPipelineRepr:
    """Test Pipeline string representation."""

    def test_repr(self):
        """Test repr shows all steps."""
        pipeline = Pipeline([
            ("select", SelectLayer(16)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        repr_str = repr(pipeline)

        assert "Pipeline" in repr_str
        assert "select" in repr_str
        assert "pool" in repr_str
        assert "probe" in repr_str
        assert "SelectLayer" in repr_str


class TestPipelineWithMLP:
    """Test Pipeline with MLP probe."""

    def test_pipeline_with_mlp(self):
        """Test pipeline works with MLP probe."""
        torch.manual_seed(42)

        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", MLP(hidden_dim=8)),
        ])

        acts = create_activations(
            n_layers=2, batch_size=20, d_model=16, layer_indices=[0, 1]
        )
        labels = create_labels(batch_size=20)

        pipeline.fit(acts, labels)

        test_acts = create_activations(
            n_layers=2, batch_size=5, d_model=16, layer_indices=[0, 1]
        )
        probs = pipeline.predict_proba(test_acts)

        assert probs.shape == (5, 2)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)


class TestPipelineAutoSelectLayer:
    """Test Pipeline auto-selection of single layer."""

    def test_single_layer_auto_select(self):
        """Test that single-layer activations work without SelectLayer."""
        # Pipeline WITHOUT SelectLayer
        pipeline = Pipeline([
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        # Single-layer activations
        acts = create_activations(n_layers=1, batch_size=20, layer_indices=[16])
        labels = create_labels(batch_size=20)

        # Should work without SelectLayer
        pipeline.fit(acts, labels)
        assert pipeline._probe._fitted

        # Predict should also work
        test_acts = create_activations(n_layers=1, batch_size=5, layer_indices=[16])
        probs = pipeline.predict_proba(test_acts)
        assert probs.shape == (5, 2)

    def test_single_layer_auto_select_with_token_level(self):
        """Test auto-selection works with token-level training."""
        # Pipeline WITHOUT SelectLayer, with post-probe pool
        pipeline = Pipeline([
            ("probe", Logistic()),
            ("pool", Pool(dim="sequence", method="mean")),
        ])

        # Single-layer activations
        acts = create_activations(n_layers=1, batch_size=20, layer_indices=[8])
        labels = create_labels(batch_size=20)

        pipeline.fit(acts, labels)
        assert pipeline._probe._fitted

        test_acts = create_activations(n_layers=1, batch_size=5, layer_indices=[8])
        probs = pipeline.predict_proba(test_acts)
        assert probs.shape == (5, 2)

    def test_multi_layer_without_handling_raises(self):
        """Test that multi-layer activations without layer handling raises error."""
        # Pipeline WITHOUT SelectLayer or Pool(dim="layer")
        pipeline = Pipeline([
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        # Multi-layer activations
        acts = create_activations(n_layers=3, batch_size=20, layer_indices=[12, 16, 20])
        labels = create_labels(batch_size=20)

        with pytest.raises(ValueError, match="Activations contain 3 layers"):
            pipeline.fit(acts, labels)

    def test_multi_layer_with_select_layer_works(self):
        """Test that multi-layer activations work with explicit SelectLayer."""
        # Pipeline WITH SelectLayer
        pipeline = Pipeline([
            ("select", SelectLayer(16)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        # Multi-layer activations
        acts = create_activations(n_layers=3, batch_size=20, layer_indices=[12, 16, 20])
        labels = create_labels(batch_size=20)

        pipeline.fit(acts, labels)
        assert pipeline._probe._fitted

    def test_multi_layer_with_pool_layer_works(self):
        """Test that multi-layer activations work with Pool(dim='layer')."""
        # Pipeline WITH Pool(dim="layer")
        pipeline = Pipeline([
            ("pool_layer", Pool(dim="layer", method="mean")),
            ("pool_seq", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        # Multi-layer activations
        acts = create_activations(n_layers=3, batch_size=20, layer_indices=[12, 16, 20])
        labels = create_labels(batch_size=20)

        pipeline.fit(acts, labels)
        assert pipeline._probe._fitted

    def test_explicit_select_layer_still_works_for_single_layer(self):
        """Test that explicit SelectLayer works even with single-layer activations."""
        # Pipeline WITH explicit SelectLayer
        pipeline = Pipeline([
            ("select", SelectLayer(16)),
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        # Single-layer activations
        acts = create_activations(n_layers=1, batch_size=20, layer_indices=[16])
        labels = create_labels(batch_size=20)

        pipeline.fit(acts, labels)
        assert pipeline._probe._fitted

    def test_no_layer_axis_works(self):
        """Test that activations without LAYER axis work (already selected)."""
        # Create single-layer activations and pre-select
        acts = create_activations(n_layers=1, batch_size=20, layer_indices=[16])
        acts_no_layer = acts.select(layer=16)  # Remove LAYER axis

        # Pipeline WITHOUT SelectLayer
        pipeline = Pipeline([
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        labels = create_labels(batch_size=20)
        pipeline.fit(acts_no_layer, labels)
        assert pipeline._probe._fitted

    def test_error_message_includes_options(self):
        """Test that error message for multi-layer includes helpful options."""
        pipeline = Pipeline([
            ("pool", Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        acts = create_activations(n_layers=2, batch_size=10, layer_indices=[12, 20])
        labels = create_labels(batch_size=10)

        with pytest.raises(ValueError) as exc_info:
            pipeline.fit(acts, labels)

        error_msg = str(exc_info.value)
        assert "12" in error_msg  # First layer index mentioned
        assert "SelectLayer" in error_msg  # Option 2
        assert "Pool(dim='layer'" in error_msg  # Option 3

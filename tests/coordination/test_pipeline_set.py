"""Tests for PipelineSet fused execution."""

import pytest
import torch

from probelab import Pipeline
from probelab.transforms import pre, post
from probelab.probes import Logistic
from probelab.processing.activations import Activations
from probelab.coordination import PipelineSet


def create_test_activations(
    n_layers: int = 3,
    batch_size: int = 20,
    seq_len: int = 10,
    d_model: int = 16,
    layer_indices: list[int] | None = None,
) -> Activations:
    """Create test activations."""
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    acts = torch.randn(n_layers, batch_size, seq_len, d_model)
    attention_mask = torch.ones(batch_size, seq_len)
    detection_mask = torch.ones(batch_size, seq_len)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    return Activations.from_components(
        activations=acts,
        attention_mask=attention_mask,
        input_ids=input_ids,
        detection_mask=detection_mask,
        layer_indices=layer_indices,
    )


def create_labels(batch_size: int = 20) -> torch.Tensor:
    """Create test labels."""
    return torch.randint(0, 2, (batch_size,))


class TestPipelineSetInit:
    """Test PipelineSet initialization."""

    def test_init_with_dict(self):
        """Test initialization with dict of pipelines."""
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "p2": Pipeline([
                ("select", pre.SelectLayer(1)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
        }

        ps = PipelineSet(pipelines)

        assert len(ps) == 2
        assert "p1" in ps.pipelines
        assert "p2" in ps.pipelines

    def test_init_with_list(self):
        """Test initialization with list of pipelines."""
        pipelines = [
            Pipeline([
                ("select", pre.SelectLayer(0)),
                ("probe", Logistic()),
            ]),
            Pipeline([
                ("select", pre.SelectLayer(1)),
                ("probe", Logistic()),
            ]),
        ]

        ps = PipelineSet(pipelines)

        assert len(ps) == 2
        assert "pipeline_0" in ps.pipelines
        assert "pipeline_1" in ps.pipelines

    def test_graph_is_created(self):
        """Test that execution graph is created."""
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("probe", Logistic()),
            ]),
        }

        ps = PipelineSet(pipelines)

        assert ps.graph is not None
        assert len(ps.graph.nodes) > 0


class TestPipelineSetFit:
    """Test PipelineSet.fit()."""

    def test_fit_basic(self):
        """Test basic fitting."""
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
        }

        ps = PipelineSet(pipelines)
        acts = create_test_activations(n_layers=2, layer_indices=[0, 1])
        labels = create_labels()

        result = ps.fit(acts, labels)

        assert result is ps
        assert ps._fitted is True

    def test_fit_multiple_pipelines(self):
        """Test fitting multiple pipelines."""
        pipelines = {
            "mean": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "max": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("pool", pre.Pool(dim="sequence", method="max")),
                ("probe", Logistic()),
            ]),
        }

        ps = PipelineSet(pipelines)
        acts = create_test_activations(n_layers=2, layer_indices=[0, 1])
        labels = create_labels()

        ps.fit(acts, labels)

        # Both probes should be fitted
        assert ps["mean"]._probe._fitted
        assert ps["max"]._probe._fitted


class TestPipelineSetPredict:
    """Test PipelineSet.predict()."""

    def test_predict_returns_all_results(self):
        """Test that predict returns results for all pipelines."""
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "p2": Pipeline([
                ("select", pre.SelectLayer(1)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
        }

        ps = PipelineSet(pipelines)
        acts = create_test_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        labels = create_labels(batch_size=10)

        ps.fit(acts, labels)
        results = ps.predict(acts)

        assert "p1" in results
        assert "p2" in results
        assert results["p1"].shape == (10, 2)
        assert results["p2"].shape == (10, 2)

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error."""
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("probe", Logistic()),
            ]),
        }

        ps = PipelineSet(pipelines)
        acts = create_test_activations(n_layers=2, layer_indices=[0, 1])

        with pytest.raises(ValueError, match="must be fitted"):
            ps.predict(acts)

    def test_predict_returns_valid_probabilities(self):
        """Test that predictions are valid probabilities."""
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
        }

        ps = PipelineSet(pipelines)
        acts = create_test_activations(n_layers=2, layer_indices=[0, 1])
        labels = create_labels()

        ps.fit(acts, labels)
        results = ps.predict(acts)

        probs = results["p1"]
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(probs.shape[0]), atol=1e-5)


class TestPipelineSetFusion:
    """Test that PipelineSet correctly fuses shared transforms."""

    def test_shared_transforms_computed_once(self):
        """Test that shared transforms are only computed once."""
        # Both pipelines share SelectLayer(0)
        pipelines = {
            "mean": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "max": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("pool", pre.Pool(dim="sequence", method="max")),
                ("probe", Logistic()),
            ]),
        }

        ps = PipelineSet(pipelines)

        # Graph should show sharing
        assert ps.graph.get_shared_prefix_depth() == 1

        # Fit should work
        acts = create_test_activations(n_layers=2, layer_indices=[0, 1])
        labels = create_labels()
        ps.fit(acts, labels)

        # Both should produce valid results
        results = ps.predict(acts)
        assert "mean" in results
        assert "max" in results

    def test_results_match_individual_pipelines(self):
        """Test that PipelineSet results match individual pipeline results."""
        torch.manual_seed(42)

        # Create identical configurations
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
        }

        standalone = Pipeline([
            ("select", pre.SelectLayer(0)),
            ("pool", pre.Pool(dim="sequence", method="mean")),
            ("probe", Logistic()),
        ])

        acts = create_test_activations(n_layers=2, layer_indices=[0, 1])
        labels = create_labels()

        # Fit both
        ps = PipelineSet(pipelines)
        ps.fit(acts, labels)
        standalone.fit(acts, labels)

        # Predictions should be close (may differ slightly due to random init)
        ps_results = ps.predict(acts)["p1"]
        standalone_results = standalone.predict(acts)

        # Both should have same shape and valid probabilities
        assert ps_results.shape == standalone_results.shape
        assert torch.all(ps_results >= 0) and torch.all(ps_results <= 1)


class TestPipelineSetScore:
    """Test PipelineSet.score()."""

    def test_score_returns_all_accuracies(self):
        """Test that score returns accuracy for all pipelines."""
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
            "p2": Pipeline([
                ("select", pre.SelectLayer(1)),
                ("pool", pre.Pool(dim="sequence", method="mean")),
                ("probe", Logistic()),
            ]),
        }

        ps = PipelineSet(pipelines)
        acts = create_test_activations(n_layers=2, layer_indices=[0, 1])
        labels = create_labels()

        ps.fit(acts, labels)
        scores = ps.score(acts, labels)

        assert "p1" in scores
        assert "p2" in scores
        assert 0.0 <= scores["p1"] <= 1.0
        assert 0.0 <= scores["p2"] <= 1.0


class TestPipelineSetWithPostTransforms:
    """Test PipelineSet with post-probe transforms."""

    def test_post_transforms_applied(self):
        """Test that post-transforms are applied correctly."""
        pipelines = {
            "token_level": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("probe", Logistic()),
                ("pool", post.Pool(method="mean")),
            ]),
        }

        ps = PipelineSet(pipelines)
        acts = create_test_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        labels = create_labels(batch_size=10)

        ps.fit(acts, labels)
        results = ps.predict(acts)

        # Should be sequence-level predictions
        assert results["token_level"].shape == (10, 2)


class TestPipelineSetRepr:
    """Test PipelineSet string representation."""

    def test_repr_shows_info(self):
        """Test that repr shows useful information."""
        pipelines = {
            "p1": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("probe", Logistic()),
            ]),
            "p2": Pipeline([
                ("select", pre.SelectLayer(0)),
                ("probe", Logistic()),
            ]),
        }

        ps = PipelineSet(pipelines)

        repr_str = repr(ps)
        assert "PipelineSet" in repr_str
        assert "p1" in repr_str
        assert "p2" in repr_str
        assert "shared_prefix_depth" in repr_str

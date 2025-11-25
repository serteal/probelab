"""Tests for preprocessing transformers."""

import pytest
import torch

from probelib.preprocessing import Normalize, Pool, SelectLayer, SelectLayers
from probelib.processing.activations import Activations, Axis
from probelib.processing.scores import Scores


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


class TestSelectLayer:
    """Test SelectLayer transformer."""

    def test_transform_selects_correct_layer(self):
        """Test that transform selects the correct layer."""
        acts = create_activations(n_layers=3, layer_indices=[0, 5, 10])
        transform = SelectLayer(layer=5)

        result = transform.transform(acts)

        # Should have selected layer 5 (tensor index 1)
        assert torch.equal(result.activations, acts.activations[1])

    def test_transform_removes_layer_axis(self):
        """Test that LAYER axis is removed after selection."""
        acts = create_activations(n_layers=3, layer_indices=[0, 5, 10])
        assert acts.has_axis(Axis.LAYER)

        transform = SelectLayer(layer=5)
        result = transform.transform(acts)

        assert not result.has_axis(Axis.LAYER)
        assert result.has_axis(Axis.BATCH)
        assert result.has_axis(Axis.SEQ)
        assert result.has_axis(Axis.HIDDEN)

    def test_transform_preserves_other_axes(self):
        """Test that other axes are preserved."""
        acts = create_activations(
            n_layers=2, batch_size=5, seq_len=10, d_model=32, layer_indices=[0, 1]
        )

        transform = SelectLayer(layer=0)
        result = transform.transform(acts)

        assert result.batch_size == 5
        assert result.seq_len == 10
        assert result.d_model == 32

    def test_raises_on_missing_layer_axis(self):
        """Test error when LAYER axis is missing."""
        acts = create_activations(n_layers=2, layer_indices=[0, 1])
        # Remove layer axis by selecting
        single_layer = acts.select(layer=0)

        transform = SelectLayer(layer=0)
        with pytest.raises(ValueError, match="don't have LAYER axis"):
            transform.transform(single_layer)

    def test_raises_on_invalid_layer_index(self):
        """Test error for unavailable layer."""
        acts = create_activations(n_layers=2, layer_indices=[0, 5])

        transform = SelectLayer(layer=10)  # Not in layer_indices
        with pytest.raises(ValueError, match="not available"):
            transform.transform(acts)

    def test_repr(self):
        """Test string representation."""
        transform = SelectLayer(layer=16)
        assert "SelectLayer" in repr(transform)
        assert "16" in repr(transform)


class TestSelectLayers:
    """Test SelectLayers transformer."""

    def test_transform_selects_multiple_layers(self):
        """Test selecting multiple layers."""
        acts = create_activations(n_layers=4, layer_indices=[0, 5, 10, 15])
        transform = SelectLayers(layers=[5, 15])

        result = transform.transform(acts)

        assert result.n_layers == 2
        assert result.layer_indices == [5, 15]

    def test_transform_keeps_layer_axis(self):
        """Test that LAYER axis is preserved."""
        acts = create_activations(n_layers=4, layer_indices=[0, 5, 10, 15])

        transform = SelectLayers(layers=[5, 10])
        result = transform.transform(acts)

        assert result.has_axis(Axis.LAYER)
        assert result.n_layers == 2

    def test_layer_order_preserved(self):
        """Test that layer order matches request."""
        acts = create_activations(n_layers=4, layer_indices=[0, 5, 10, 15])
        transform = SelectLayers(layers=[15, 5])  # Reverse order

        result = transform.transform(acts)

        # Order should match the requested layers
        assert result.layer_indices == [15, 5]

    def test_repr(self):
        """Test string representation."""
        transform = SelectLayers(layers=[8, 16, 24])
        assert "SelectLayers" in repr(transform)
        assert "[8, 16, 24]" in repr(transform)


class TestPool:
    """Test Pool transformer."""

    def test_pool_sequence_mean(self):
        """Test mean pooling over sequence dimension."""
        torch.manual_seed(42)
        acts = create_activations(n_layers=1, batch_size=5, seq_len=10)
        transform = Pool(dim="sequence", method="mean")

        result = transform.transform(acts)

        assert not result.has_axis(Axis.SEQ)
        assert result.batch_size == 5

    def test_pool_sequence_max(self):
        """Test max pooling over sequence dimension."""
        acts = create_activations(n_layers=1, batch_size=5, seq_len=10)
        transform = Pool(dim="sequence", method="max")

        result = transform.transform(acts)

        assert not result.has_axis(Axis.SEQ)

    def test_pool_sequence_last_token(self):
        """Test last_token pooling."""
        acts = create_activations(n_layers=1, batch_size=5, seq_len=10)
        transform = Pool(dim="sequence", method="last_token")

        result = transform.transform(acts)

        assert not result.has_axis(Axis.SEQ)

    def test_pool_layer_mean(self):
        """Test mean pooling over layer dimension."""
        acts = create_activations(n_layers=4, layer_indices=[0, 5, 10, 15])
        transform = Pool(dim="layer", method="mean")

        result = transform.transform(acts)

        assert not result.has_axis(Axis.LAYER)

    def test_pool_layer_max(self):
        """Test max pooling over layer dimension."""
        acts = create_activations(n_layers=4, layer_indices=[0, 5, 10, 15])
        transform = Pool(dim="layer", method="max")

        result = transform.transform(acts)

        assert not result.has_axis(Axis.LAYER)

    def test_pool_removes_pooled_axis(self):
        """Test that pooled axis is removed."""
        acts = create_activations(n_layers=2, batch_size=5, seq_len=10)

        # Pool sequence
        transform = Pool(dim="sequence", method="mean")
        result = transform.transform(acts)
        assert not result.has_axis(Axis.SEQ)
        assert result.has_axis(Axis.LAYER)

        # Pool layer
        transform2 = Pool(dim="layer", method="mean")
        result2 = transform2.transform(result)
        assert not result2.has_axis(Axis.LAYER)

    def test_pool_on_scores(self):
        """Test Pool works on Scores objects."""
        scores = Scores.from_token_scores(
            scores=torch.randn(5, 10, 2),
            tokens_per_sample=torch.tensor([10, 8, 6, 4, 2]),
        )
        transform = Pool(dim="sequence", method="mean")

        result = transform.transform(scores)

        assert result.scores.shape == (5, 2)

    def test_pool_invalid_dim_raises(self):
        """Test error for invalid dimension."""
        with pytest.raises(ValueError, match="dim must be one of"):
            Pool(dim="invalid", method="mean")

    def test_pool_invalid_method_raises(self):
        """Test error for invalid method."""
        with pytest.raises(ValueError, match="method must be one of"):
            Pool(dim="sequence", method="invalid")

    def test_pool_last_token_on_layer_raises(self):
        """Test error for last_token on layer dimension."""
        with pytest.raises(ValueError, match="last_token method not supported"):
            Pool(dim="layer", method="last_token")

    def test_pool_idempotent_when_axis_missing(self):
        """Test that pool is idempotent when axis already missing."""
        acts = create_activations(n_layers=1, batch_size=5, seq_len=10)
        # First pool removes SEQ axis
        transform = Pool(dim="sequence", method="mean")
        pooled = transform.transform(acts)

        # Second pool should return same thing (no SEQ axis to pool)
        result = transform.transform(pooled)
        assert torch.equal(result.activations, pooled.activations)

    def test_repr(self):
        """Test string representation."""
        transform = Pool(dim="sequence", method="mean")
        assert "Pool" in repr(transform)
        assert "sequence" in repr(transform)
        assert "mean" in repr(transform)


class TestNormalize:
    """Test Normalize transformer."""

    def test_fit_computes_statistics(self):
        """Test that fit computes mean and std."""
        acts = create_activations(n_layers=1, batch_size=20, seq_len=10)
        transform = Normalize()

        transform.fit(acts)

        assert transform._fitted
        assert transform.mean_ is not None
        assert transform.std_ is not None

    def test_transform_normalizes_data(self):
        """Test that transform normalizes data."""
        torch.manual_seed(42)
        # Create data with known mean and std
        acts = create_activations(n_layers=1, batch_size=100, seq_len=10, d_model=16)

        transform = Normalize()
        transform.fit(acts)
        result = transform.transform(acts)

        # Normalized data should have approximately zero mean and unit std
        # (approximately because of finite sample)
        result_tensor = result.activations
        axes = tuple(range(result_tensor.ndim - 1))
        mean = result_tensor.mean(dim=axes)
        std = result_tensor.std(dim=axes)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.1)
        assert torch.allclose(std, torch.ones_like(std), atol=0.2)

    def test_transform_before_fit_raises(self):
        """Test error when transform called before fit."""
        acts = create_activations(n_layers=1)
        transform = Normalize()

        with pytest.raises(ValueError, match="must be fitted"):
            transform.transform(acts)

    def test_partial_fit_online_learning(self):
        """Test partial_fit for online learning."""
        torch.manual_seed(42)
        transform = Normalize()

        # Fit with multiple batches
        for _ in range(5):
            acts = create_activations(n_layers=1, batch_size=10)
            transform.partial_fit(acts)

        assert transform._fitted
        assert transform._n_samples_seen > 0

    def test_freeze_locks_statistics(self):
        """Test that freeze prevents further updates."""
        transform = Normalize()

        # Initial fit
        acts1 = create_activations(n_layers=1, batch_size=10)
        transform.partial_fit(acts1)
        mean_before = transform.mean_.clone()

        # Freeze
        transform.freeze()

        # Further partial_fit should not change statistics
        acts2 = create_activations(n_layers=1, batch_size=10)
        transform.partial_fit(acts2)

        assert torch.equal(transform.mean_, mean_before)

    def test_unfreeze_allows_updates(self):
        """Test that unfreeze allows further updates."""
        transform = Normalize()

        acts1 = create_activations(n_layers=1, batch_size=10)
        transform.partial_fit(acts1)
        transform.freeze()
        transform.unfreeze()

        # Should be able to update after unfreeze
        n_before = transform._n_samples_seen
        acts2 = create_activations(n_layers=1, batch_size=10)
        transform.partial_fit(acts2)

        assert transform._n_samples_seen > n_before

    def test_normalization_preserves_shape(self):
        """Test that normalization preserves activation shape."""
        acts = create_activations(n_layers=2, batch_size=5, seq_len=10, d_model=16)
        transform = Normalize()

        transform.fit(acts)
        result = transform.transform(acts)

        assert result.shape == acts.shape
        assert result.axes == acts.axes
        assert result.layer_indices == acts.layer_indices

    def test_repr(self):
        """Test string representation."""
        transform = Normalize()
        assert "Normalize" in repr(transform)
        assert "not fitted" in repr(transform)

        acts = create_activations(n_layers=1)
        transform.fit(acts)
        assert "fitted" in repr(transform)

        transform.freeze()
        assert "frozen" in repr(transform)


class TestFitTransform:
    """Test fit_transform convenience method."""

    def test_normalize_fit_transform(self):
        """Test fit_transform returns same as fit then transform."""
        torch.manual_seed(42)
        acts = create_activations(n_layers=1, batch_size=20)

        transform1 = Normalize()
        result1 = transform1.fit_transform(acts)

        transform2 = Normalize()
        transform2.fit(acts)
        result2 = transform2.transform(acts)

        assert torch.allclose(result1.activations, result2.activations)

    def test_select_layer_fit_transform(self):
        """Test fit_transform for stateless transforms."""
        acts = create_activations(n_layers=2, layer_indices=[0, 5])
        transform = SelectLayer(layer=5)

        # fit_transform should work even though SelectLayer is stateless
        result = transform.fit_transform(acts)

        assert not result.has_axis(Axis.LAYER)

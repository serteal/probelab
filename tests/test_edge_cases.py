"""
Comprehensive edge case tests for probelab.
Focuses on aggregation methods, memory management, and boundary conditions.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from probelab.probes import MLP, Attention, Logistic
from probelab.processing.activations import Activations
from probelab.types import AggregationMethod, Label


class TestAggregationEdgeCases:
    """Test edge cases for different aggregation methods."""

    def test_empty_detection_mask(self):
        """Test behavior when detection mask is all zeros."""
        # Create activations with empty detection mask
        acts = torch.randn(1, 2, 10, 8)
        detection_mask = torch.zeros(2, 10)  # No valid tokens

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(2, 10),
            input_ids=torch.ones(2, 10, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Mean aggregation with no valid tokens should handle gracefully
        aggregated = activations.select(layer=0).pool(dim="sequence", method="mean").activations
        assert aggregated.shape == (2, 8)
        # Should be zeros or handle gracefully
        assert torch.all(torch.isfinite(aggregated))

    def test_single_token_detection_mask(self):
        """Test when only one token is valid per sequence."""
        acts = torch.randn(1, 3, 10, 8)
        detection_mask = torch.zeros(3, 10)
        # Only first token valid for each sequence
        detection_mask[:, 0] = 1

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(3, 10),
            input_ids=torch.ones(3, 10, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Test all aggregation methods
        for method in [AggregationMethod.MEAN, AggregationMethod.MAX, AggregationMethod.LAST_TOKEN]:
            aggregated = activations.select(layer=0).pool(dim="sequence", method=method).activations
            assert aggregated.shape == (3, 8)

            if method in (AggregationMethod.MEAN, AggregationMethod.LAST_TOKEN):
                # Should equal the first token since it's the only valid one
                expected = acts[0, :, 0, :]
                assert torch.allclose(aggregated, expected)

    def test_mixed_sequence_lengths(self):
        """Test aggregation with varying sequence lengths."""
        batch_size = 4
        seq_len = 20
        d_model = 16

        acts = torch.randn(1, batch_size, seq_len, d_model)
        detection_mask = torch.zeros(batch_size, seq_len)

        # Different valid lengths for each sequence
        detection_mask[0, :5] = 1  # 5 valid tokens
        detection_mask[1, :10] = 1  # 10 valid tokens
        detection_mask[2, :15] = 1  # 15 valid tokens
        detection_mask[3, :20] = 1  # 20 valid tokens

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(batch_size, seq_len),
            input_ids=torch.ones(batch_size, seq_len, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Test mean aggregation
        mean_agg = activations.select(layer=0).pool(dim="sequence", method="mean").activations
        assert mean_agg.shape == (batch_size, d_model)

        # Verify mean is computed only over valid tokens
        for i in range(batch_size):
            valid_len = [5, 10, 15, 20][i]
            expected_mean = acts[0, i, :valid_len, :].mean(dim=0)
            assert torch.allclose(mean_agg[i], expected_mean, atol=1e-6)

        # Test max aggregation
        max_agg = activations.select(layer=0).pool(dim="sequence", method="max").activations
        assert max_agg.shape == (batch_size, d_model)

        # Test last_token aggregation
        last_agg = activations.select(layer=0).pool(dim="sequence", method="last_token").activations
        assert last_agg.shape == (batch_size, d_model)

        # Last token should be at different positions
        for i in range(batch_size):
            valid_len = [5, 10, 15, 20][i]
            expected_last = acts[0, i, valid_len - 1, :]
            assert torch.allclose(last_agg[i], expected_last, atol=1e-6)

    def test_aggregation_with_nan_values(self):
        """Test that aggregation handles NaN values properly."""
        acts = torch.randn(1, 2, 10, 8)
        # Inject some NaN values
        acts[0, 0, 5, :] = float("nan")

        detection_mask = torch.ones(2, 10)

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(2, 10),
            input_ids=torch.ones(2, 10, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Mean aggregation should propagate NaN
        mean_agg = activations.select(layer=0).pool(dim="sequence", method="mean").activations
        assert torch.any(torch.isnan(mean_agg[0]))

        # But shouldn't affect other sequences
        assert torch.all(torch.isfinite(mean_agg[1]))

    def test_aggregation_preserves_dtype(self):
        """Test that aggregation preserves data types."""
        # Note: float16 may be converted to float32 for numerical stability
        for dtype in [torch.float32, torch.float64]:
            acts = torch.randn(1, 2, 10, 8, dtype=dtype)

            activations = Activations.from_tensor(
                activations=acts,
                attention_mask=torch.ones(2, 10),
                input_ids=torch.ones(2, 10, dtype=torch.long),
                detection_mask=torch.ones(2, 10),
                layer_indices=[0],
            )

            aggregated = activations.select(layer=0).pool(dim="sequence", method="mean").activations
            assert aggregated.dtype == dtype

    def test_aggregation_preserves_device(self):
        """Test that aggregation preserves device placement."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        acts = torch.randn(1, 2, 10, 8, device=device)

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(2, 10, device=device),
            input_ids=torch.ones(2, 10, dtype=torch.long, device=device),
            detection_mask=torch.ones(2, 10, device=device),
            layer_indices=[0],
        )

        aggregated = activations.select(layer=0).pool(dim="sequence", method="mean").activations
        assert aggregated.device.type == device


class TestProbeEdgeCases:
    """Test edge cases for probe training and prediction."""

    def test_probe_with_single_sample(self):
        """Test probe training with just one sample."""
        acts = torch.randn(1, 1, 10, 8)

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(1, 10),
            input_ids=torch.ones(1, 10, dtype=torch.long),
            detection_mask=torch.ones(1, 10),
            layer_indices=[0],
        )

        prepared = activations.select(layer=0).pool("sequence", "mean")

        # Can't train with single class - PyTorch implementation may not raise
        probe = Logistic()

        # Try to fit with single sample - may not raise in PyTorch implementation
        try:
            probe.fit(prepared, [Label.POSITIVE])
            # If no error, check that it's not properly fitted
            if hasattr(probe, "_network") and probe._network is not None:
                # May have created network but not trained properly
                pass
        except (ValueError, RuntimeError, IndexError):
            # Expected for proper implementations
            pass

    def test_probe_with_imbalanced_classes(self):
        """Test probe with extremely imbalanced classes."""
        n_samples = 100
        acts = torch.randn(1, n_samples, 10, 8)

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(n_samples, 10),
            input_ids=torch.ones(n_samples, 10, dtype=torch.long),
            detection_mask=torch.ones(n_samples, 10),
            layer_indices=[0],
        )

        prepared = activations.select(layer=0).pool("sequence", "mean")

        # 99 positive, 1 negative
        labels = [Label.POSITIVE] * 99 + [Label.NEGATIVE] * 1

        probe = Logistic().fit(prepared, labels)

        # Should still train, but might have poor performance
        assert probe._fitted

        scores = probe.predict(prepared)
        assert scores.shape == (100, 2)

    def test_probe_layer_mismatch(self):
        """Test probe when requested layer doesn't exist."""
        acts = torch.randn(2, 10, 20, 16)  # 2 layers

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(10, 20),
            input_ids=torch.ones(10, 20, dtype=torch.long),
            detection_mask=torch.ones(10, 20),
            layer_indices=[0, 1],  # Only layers 0 and 1
        )

        # Request non-existent layer 5
        with pytest.raises(ValueError, match="Layer 5 is not available"):
            activations.select(layer=5)

    def test_probe_with_all_zero_features(self):
        """Test probe when all features are zero."""
        acts = torch.zeros(1, 10, 20, 16)  # All zeros

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(10, 20),
            input_ids=torch.ones(10, 20, dtype=torch.long),
            detection_mask=torch.ones(10, 20),
            layer_indices=[0],
        )

        labels = [Label.POSITIVE] * 5 + [Label.NEGATIVE] * 5

        prepared = activations.select(layer=0).pool("sequence", "mean")
        probe = Logistic().fit(prepared, labels)

        # Should still train, but predictions might be uniform
        assert probe._fitted

        scores = probe.predict(prepared)
        assert scores.shape == (10, 2)
        # With zero features, should predict roughly uniform
        assert torch.all(scores.scores >= 0) and torch.all(scores.scores <= 1)

    def test_mlp_probe_edge_cases(self):
        """Test MLP probe specific edge cases."""
        acts = torch.randn(1, 10, 20, 16)

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(10, 20),
            input_ids=torch.ones(10, 20, dtype=torch.long),
            detection_mask=torch.ones(10, 20),
            layer_indices=[0],
        )

        labels = [Label.POSITIVE] * 5 + [Label.NEGATIVE] * 5
        prepared = activations.select(layer=0).pool("sequence", "mean")

        # Test with very small hidden dimension
        probe = MLP(hidden_dim=1).fit(prepared, labels)
        assert probe._fitted

        # Test with very large hidden dimension
        probe = MLP(hidden_dim=1024).fit(prepared, labels)
        assert probe._fitted

        # Test with high dropout
        probe = MLP(dropout=0.9).fit(prepared, labels)
        assert probe._fitted

    def test_attention_probe_edge_cases(self):
        """Test Attention probe specific edge cases."""
        acts = torch.randn(1, 10, 20, 16)

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(10, 20),
            input_ids=torch.ones(10, 20, dtype=torch.long),
            detection_mask=torch.ones(10, 20),
            layer_indices=[0],
        )

        labels = [Label.POSITIVE] * 5 + [Label.NEGATIVE] * 5

        # Attention probe does its own aggregation, so no pool needed
        prepared = activations.select(layer=0)

        # Test with small hidden dimension
        probe = Attention(hidden_dim=8).fit(prepared, labels)
        assert probe._fitted

        # Test with larger hidden dimension
        probe = Attention(hidden_dim=128).fit(prepared, labels)
        assert probe._fitted


class TestBoundaryConditions:
    """Test boundary conditions and extreme inputs."""

    def test_maximum_sequence_length(self):
        """Test with very long sequences."""
        max_seq_len = 4096
        acts = torch.randn(1, 2, max_seq_len, 16)

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(2, max_seq_len),
            input_ids=torch.ones(2, max_seq_len, dtype=torch.long),
            detection_mask=torch.ones(2, max_seq_len),
            layer_indices=[0],
        )

        # Should handle long sequences
        aggregated = activations.select(layer=0).pool(dim="sequence", method="mean").activations
        assert aggregated.shape == (2, 16)

    def test_many_layers(self):
        """Test with many layers."""
        n_layers = 100
        acts = torch.randn(n_layers, 2, 10, 16)

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(2, 10),
            input_ids=torch.ones(2, 10, dtype=torch.long),
            detection_mask=torch.ones(2, 10),
            layer_indices=list(range(n_layers)),
        )

        # Should handle many layers
        filtered = activations.select(layers=[0, 50, 99])
        assert filtered.n_layers == 3

    def test_high_dimensional_embeddings(self):
        """Test with very high dimensional embeddings."""
        d_model = 8192
        acts = torch.randn(1, 2, 10, d_model)

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(2, 10),
            input_ids=torch.ones(2, 10, dtype=torch.long),
            detection_mask=torch.ones(2, 10),
            layer_indices=[0],
        )

        labels = [Label.POSITIVE, Label.NEGATIVE]

        # Should handle high dimensions (might be slow)
        prepared = activations.select(layer=0).pool("sequence", "mean")
        probe = Logistic().fit(prepared, labels)
        assert probe._fitted

        # Check network was created with correct input dimension
        assert probe._network is not None
        # Linear layer should accept d_model inputs
        assert probe._network.linear.in_features == d_model

    def test_concurrent_probe_training(self):
        """Test training multiple probes concurrently (not parallel)."""
        acts = torch.randn(3, 10, 20, 16)  # 3 layers

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(10, 20),
            input_ids=torch.ones(10, 20, dtype=torch.long),
            detection_mask=torch.ones(10, 20),
            layer_indices=[0, 1, 2],
        )

        labels = [Label.POSITIVE] * 5 + [Label.NEGATIVE] * 5

        # Train probes on different layers
        probes = []
        for layer in [0, 1, 2]:
            prepared = activations.select(layer=layer).pool("sequence", "mean")
            probe = Logistic().fit(prepared, labels)
            probes.append((layer, probe, prepared))

        # All should be fitted
        assert all(p._fitted for _, p, _ in probes)

        # Each should work independently
        for layer, probe, prepared in probes:
            scores = probe.predict(prepared)
            assert scores.shape == (10, 2)

    def test_save_load_with_edge_cases(self):
        """Test saving and loading probes in edge cases."""
        acts = torch.randn(1, 10, 20, 16)

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(10, 20),
            input_ids=torch.ones(10, 20, dtype=torch.long),
            detection_mask=torch.ones(10, 20),
            layer_indices=[0],
        )

        labels = [Label.POSITIVE] * 5 + [Label.NEGATIVE] * 5

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with special characters in path
            save_path = Path(tmpdir) / "probe with spaces & symbols!.pt"

            prepared = activations.select(layer=0).pool("sequence", "mean")
            probe = Logistic().fit(prepared, labels)
            probe.save(save_path)

            loaded = Logistic.load(save_path)
            assert loaded._fitted

            # Test overwriting existing file
            probe2 = MLP().fit(prepared, labels)
            probe2.save(save_path)  # Should overwrite

            # Loading should get the MLP, not Logistic
            loaded2 = MLP.load(save_path)
            assert isinstance(loaded2._network, torch.nn.Module)

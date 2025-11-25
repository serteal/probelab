"""
Test that probes correctly handle empty batches when no tokens are detected.

Note: Many tests have been removed as they tested internal implementation details
that no longer exist in the new Pipeline API. The core empty batch handling is now
tested implicitly through the main probe tests.
"""

import pytest
import torch

from probelab.processing.activations import Activations


class TestEmptyBatchHandling:
    """Test that probes handle empty batches gracefully."""

    def test_token_level_extraction_with_no_detected_tokens(self):
        """Test that to_token_level handles samples with no detected tokens."""
        batch_size = 3
        seq_len = 50
        d_model = 768

        # Create mixed detection mask
        detection_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        detection_mask[0, 10:20] = True  # First sample has tokens
        # Second sample has NO tokens (all False)
        detection_mask[2, 5:10] = True  # Third sample has tokens

        activations = Activations.from_tensor(
            activations=torch.randn(1, batch_size, seq_len, d_model),
            attention_mask=torch.ones(batch_size, seq_len),
            input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
            detection_mask=detection_mask,
            layer_indices=[12],
        )

        # Extract token-level features
        layer_selected = activations.select(layer=12)
        features, tokens_per_sample = layer_selected.extract_tokens()

        # Check results
        assert features.shape[0] == 15, "Should have 10 + 0 + 5 = 15 tokens"
        assert tokens_per_sample.tolist() == [10, 0, 5], (
            "Tokens per sample should be [10, 0, 5]"
        )

    def test_sequence_pooling_with_empty_samples(self):
        """Test sequence aggregation with some samples having no detected tokens."""
        batch_size = 3
        seq_len = 50
        d_model = 768

        # Create detection mask where middle sample has no detected tokens
        detection_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        detection_mask[0, 10:20] = True  # First sample has tokens
        # Second sample has NO tokens
        detection_mask[2, 5:10] = True  # Third sample has tokens

        activations = Activations.from_tensor(
            activations=torch.randn(1, batch_size, seq_len, d_model),
            attention_mask=torch.ones(batch_size, seq_len),
            input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
            detection_mask=detection_mask,
            layer_indices=[12],
        )

        # Select layer first, then aggregate
        layer_selected = activations.select(layer=12)
        aggregated = layer_selected.pool(dim="sequence", method="mean")
        assert aggregated.activations.shape == (batch_size, d_model), "Should have 3 samples"

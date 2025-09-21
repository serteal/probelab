"""
Unit tests for get_batches focusing on key preservation and view behavior.
These tests avoid network and model dependencies.
"""

import torch

from probelib.processing.activations import get_batches


class DummyTokenizer:
    def __init__(self, pad_token_id: int = 0, padding_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side


class TestGetBatches:
    def test_get_batches_preserves_all_keys(self):
        # Create mock tokenized inputs with detection mask
        batch_size = 4
        seq_len = 100
        tokenized_inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "detection_mask": torch.randint(
                0, 2, (batch_size, seq_len), dtype=torch.bool
            ),
            "custom_key": torch.randn(batch_size, seq_len),
        }

        tokenizer = DummyTokenizer(pad_token_id=0, padding_side="right")
        tokenized_inputs["input_ids"][:, 80:] = tokenizer.pad_token_id
        tokenized_inputs["attention_mask"][:, 80:] = 0

        # Get batches
        batches = list(get_batches(tokenized_inputs, batch_size=2, tokenizer=tokenizer))
        assert len(batches) == 2

        for batch_inputs, _ in batches:
            assert set(batch_inputs.keys()) == set(tokenized_inputs.keys())
            assert "detection_mask" in batch_inputs

    def test_get_batches_with_views(self):
        # Create tokenized inputs
        n_samples = 4
        seq_len = 50
        tokenized_inputs = {
            "input_ids": torch.arange(n_samples * seq_len).reshape(n_samples, seq_len),
            "attention_mask": torch.ones(n_samples, seq_len),
            "detection_mask": torch.ones(n_samples, seq_len, dtype=torch.bool),
        }

        tokenizer = DummyTokenizer(pad_token_id=0, padding_side="right")
        for i in range(n_samples):
            pad_start = 30 + i * 5
            tokenized_inputs["input_ids"][i, pad_start:] = tokenizer.pad_token_id
            tokenized_inputs["attention_mask"][i, pad_start:] = 0
            tokenized_inputs["detection_mask"][i, pad_start:] = False

        batches = list(get_batches(tokenized_inputs, batch_size=2, tokenizer=tokenizer))
        assert len(batches) == 2
        for batch_inputs, batch_indices in batches:
            assert batch_inputs["detection_mask"].shape[0] == len(batch_indices)
            assert batch_inputs["detection_mask"].dtype == torch.bool

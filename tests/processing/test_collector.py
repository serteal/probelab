"""Tests for ActivationCollector class."""

from unittest.mock import Mock, patch

import pytest
import torch

from probelib.datasets.base import DialogueDataset
from probelib.masks import MaskFunction, assistant
from probelib.processing import ActivationCollector, ActivationIterator, Activations
from probelib.types import Dialogue, DialogueDataType, Label, Message


class MockDialogueDataset(DialogueDataset):
    """Mock dataset for testing."""

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        """Not used since we pass dialogues directly."""
        raise NotImplementedError("This method should not be called in tests")


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = Mock()
    model.config = Mock()
    model.config.num_hidden_layers = 3
    model.config.hidden_size = 64
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.padding_side = "right"
    tokenizer.name_or_path = "meta-llama/Llama-2-7b-hf"

    def mock_apply_chat_template(dialogues, **kwargs):
        formatted = []
        for dialogue_list in dialogues:
            formatted_text = ""
            for msg in dialogue_list:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    formatted_text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "assistant":
                    formatted_text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
            formatted.append(formatted_text)
        return formatted

    tokenizer.apply_chat_template = mock_apply_chat_template
    return tokenizer


@pytest.fixture
def small_dataset():
    """Create a small mock dataset (< 10k samples)."""
    dialogues = [
        [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        for _ in range(100)
    ]
    labels = [Label.POSITIVE if i % 2 == 0 else Label.NEGATIVE for i in range(100)]

    return MockDialogueDataset(
        dialogues=dialogues, labels=labels, shuffle_upon_init=False
    )


@pytest.fixture
def large_dataset():
    """Create a large mock dataset (> 10k samples) for streaming tests."""
    dialogues = [
        [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        for _ in range(15000)
    ]
    labels = [
        Label.POSITIVE if i % 2 == 0 else Label.NEGATIVE for i in range(15000)
    ]

    return MockDialogueDataset(
        dialogues=dialogues, labels=labels, shuffle_upon_init=False
    )


class TestActivationCollectorInit:
    """Tests for ActivationCollector initialization."""

    def test_basic_instantiation(self, mock_model, mock_tokenizer):
        """Test basic collector instantiation with minimal parameters."""
        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[0, 1]
        )

        assert collector.model is mock_model
        assert collector.tokenizer is mock_tokenizer
        assert collector.layers == [0, 1]
        assert collector.batch_size == 32  # Default
        assert collector.hook_point == "post_block"  # Default
        assert collector.detach_activations is True  # Default
        assert collector.verbose is True  # Default

    def test_single_layer_normalization(self, mock_model, mock_tokenizer):
        """Test that single layer is normalized to list."""
        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=16
        )

        assert collector.layers == [16]
        assert isinstance(collector.layers, list)

    def test_custom_parameters(self, mock_model, mock_tokenizer):
        """Test collector with all custom parameters."""
        collector = ActivationCollector(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layers=[8, 16, 24],
            batch_size=64,
            hook_point="pre_layernorm",
            detach_activations=False,
            device="cuda",
            verbose=False,
        )

        assert collector.layers == [8, 16, 24]
        assert collector.batch_size == 64
        assert collector.hook_point == "pre_layernorm"
        assert collector.detach_activations is False
        assert collector.device == "cuda"
        assert collector.verbose is False

    def test_device_auto_detection(self, mock_model, mock_tokenizer):
        """Test that device is auto-detected when not specified."""
        # Mock cuda availability
        with patch("torch.cuda.is_available", return_value=True):
            collector = ActivationCollector(
                model=mock_model, tokenizer=mock_tokenizer, layers=[0]
            )
            assert collector.device == "cuda"

        with patch("torch.cuda.is_available", return_value=False):
            collector = ActivationCollector(
                model=mock_model, tokenizer=mock_tokenizer, layers=[0]
            )
            assert collector.device == "cpu"

    def test_repr(self, mock_model, mock_tokenizer):
        """Test string representation."""
        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16, 20, 24]
        )

        repr_str = repr(collector)
        assert "ActivationCollector" in repr_str
        assert "[16, 20, 24]" in repr_str
        assert "batch_size=32" in repr_str

    def test_info(self, mock_model, mock_tokenizer):
        """Test info() method returns correct configuration."""
        collector = ActivationCollector(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layers=[8, 16],
            batch_size=16,
            hook_point="pre_layernorm",
        )

        info = collector.info()
        assert info["layers"] == [8, 16]
        assert info["batch_size"] == 16
        assert info["hook_point"] == "pre_layernorm"
        assert info["detach_activations"] is True
        assert "device" in info
        assert "verbose" in info


class TestActivationCollectorCollect:
    """Tests for main collect() method."""

    @patch("probelib.processing.collector.collect_activations")
    def test_collect_delegates_to_function(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test that collect() delegates to collect_activations()."""
        # Setup mock return value
        mock_collect.return_value = Mock(spec=Activations)

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        result = collector.collect(small_dataset, streaming=False)

        # Verify collect_activations was called with correct parameters
        mock_collect.assert_called_once()
        call_kwargs = mock_collect.call_args.kwargs

        assert call_kwargs["model"] is mock_model
        assert call_kwargs["tokenizer"] is mock_tokenizer
        assert call_kwargs["dataset"] is small_dataset
        assert call_kwargs["layers"] == [16]
        assert call_kwargs["streaming"] is False

    @patch("probelib.processing.collector.collect_activations")
    def test_auto_streaming_small_dataset(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test that streaming='auto' uses batch mode for small datasets."""
        mock_collect.return_value = Mock(spec=Activations)

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        collector.collect(small_dataset, streaming="auto")

        # Should use batch mode (streaming=False) for small datasets
        call_kwargs = mock_collect.call_args.kwargs
        assert call_kwargs["streaming"] is False

    @patch("probelib.processing.collector.collect_activations")
    def test_auto_streaming_large_dataset(
        self, mock_collect, mock_model, mock_tokenizer, large_dataset
    ):
        """Test that streaming='auto' uses streaming mode for large datasets."""
        mock_collect.return_value = Mock(spec=ActivationIterator)

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        collector.collect(large_dataset, streaming="auto")

        # Should use streaming mode (streaming=True) for large datasets
        call_kwargs = mock_collect.call_args.kwargs
        assert call_kwargs["streaming"] is True

    @patch("probelib.processing.collector.collect_activations")
    def test_uses_dataset_default_mask(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test that dataset's default_mask is used when no mask provided."""
        mock_collect.return_value = Mock(spec=Activations)

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        # Collect without specifying mask
        collector.collect(small_dataset)

        # Should use dataset's default mask (check type since objects aren't identical)
        call_kwargs = mock_collect.call_args.kwargs
        # Verify mask was passed and is of the same type as default mask
        assert call_kwargs["mask"] is not None
        assert type(call_kwargs["mask"]) == type(small_dataset.default_mask)

    @patch("probelib.processing.collector.collect_activations")
    def test_custom_mask_overrides_default(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test that custom mask overrides dataset's default mask."""
        mock_collect.return_value = Mock(spec=Activations)

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        # Create custom mask
        custom_mask = assistant()

        # Collect with custom mask
        collector.collect(small_dataset, mask=custom_mask)

        # Should use custom mask, not dataset's default
        call_kwargs = mock_collect.call_args.kwargs
        assert call_kwargs["mask"] is custom_mask

    @patch("probelib.processing.collector.collect_activations")
    def test_collection_strategy_passed(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test that collection_strategy is passed correctly."""
        mock_collect.return_value = Mock(spec=Activations)

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        collector.collect(small_dataset, collection_strategy="mean")

        call_kwargs = mock_collect.call_args.kwargs
        assert call_kwargs["collection_strategy"] == "mean"

    @patch("probelib.processing.collector.collect_activations")
    def test_additional_kwargs_forwarded(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test that additional kwargs are forwarded to collect_activations."""
        mock_collect.return_value = Mock(spec=Activations)

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        collector.collect(
            small_dataset, add_generation_prompt=True, custom_param="value"
        )

        call_kwargs = mock_collect.call_args.kwargs
        assert call_kwargs["add_generation_prompt"] is True
        assert call_kwargs["custom_param"] == "value"


class TestActivationCollectorConvenienceMethods:
    """Tests for convenience methods (collect_dense, collect_pooled, collect_streaming)."""

    @patch("probelib.processing.collector.collect_activations")
    def test_collect_dense(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test collect_dense() returns Activations with sequences."""
        # Mock return value
        mock_activations = Mock(spec=Activations)
        mock_collect.return_value = mock_activations

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        result = collector.collect_dense(small_dataset)

        # Verify correct parameters
        call_kwargs = mock_collect.call_args.kwargs
        assert call_kwargs["streaming"] is False
        assert call_kwargs["collection_strategy"] is None

        # Verify return type
        assert result is mock_activations

    @patch("probelib.processing.collector.collect_activations")
    def test_collect_pooled_mean(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test collect_pooled() with mean pooling."""
        mock_activations = Mock(spec=Activations)
        mock_collect.return_value = mock_activations

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        result = collector.collect_pooled(small_dataset, method="mean")

        # Verify correct parameters
        call_kwargs = mock_collect.call_args.kwargs
        assert call_kwargs["streaming"] is False
        assert call_kwargs["collection_strategy"] == "mean"

        assert result is mock_activations

    @patch("probelib.processing.collector.collect_activations")
    def test_collect_pooled_max(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test collect_pooled() with max pooling."""
        mock_activations = Mock(spec=Activations)
        mock_collect.return_value = mock_activations

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        result = collector.collect_pooled(small_dataset, method="max")

        call_kwargs = mock_collect.call_args.kwargs
        assert call_kwargs["collection_strategy"] == "max"

    @patch("probelib.processing.collector.collect_activations")
    def test_collect_pooled_last_token(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test collect_pooled() with last_token pooling."""
        mock_activations = Mock(spec=Activations)
        mock_collect.return_value = mock_activations

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        result = collector.collect_pooled(small_dataset, method="last_token")

        call_kwargs = mock_collect.call_args.kwargs
        assert call_kwargs["collection_strategy"] == "last_token"

    @patch("probelib.processing.collector.collect_activations")
    def test_collect_streaming(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test collect_streaming() returns ActivationIterator."""
        mock_iterator = Mock(spec=ActivationIterator)
        mock_collect.return_value = mock_iterator

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        result = collector.collect_streaming(small_dataset)

        # Verify correct parameters
        call_kwargs = mock_collect.call_args.kwargs
        assert call_kwargs["streaming"] is True

        # Verify return type
        assert result is mock_iterator


class TestActivationCollectorReusability:
    """Tests for collector reusability across multiple datasets."""

    @patch("probelib.processing.collector.collect_activations")
    def test_reuse_across_datasets(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test that collector can be reused across multiple datasets."""
        mock_collect.return_value = Mock(spec=Activations)

        # Create collector once
        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16], batch_size=16
        )

        # Create multiple datasets
        dataset1 = small_dataset[:50]
        dataset2 = small_dataset[50:]

        # Collect from both
        collector.collect(dataset1)
        collector.collect(dataset2)

        # Should have been called twice with same configuration
        assert mock_collect.call_count == 2

        # Verify both calls used same collector settings
        call1_kwargs = mock_collect.call_args_list[0].kwargs
        call2_kwargs = mock_collect.call_args_list[1].kwargs

        assert call1_kwargs["model"] is call2_kwargs["model"]
        assert call1_kwargs["layers"] == call2_kwargs["layers"]
        assert call1_kwargs["batch_size"] == call2_kwargs["batch_size"]

    @patch("probelib.processing.collector.collect_activations")
    def test_different_masks_per_dataset(
        self, mock_collect, mock_model, mock_tokenizer, small_dataset
    ):
        """Test using different masks with same collector."""
        mock_collect.return_value = Mock(spec=Activations)

        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[16]
        )

        mask1 = assistant()
        mask2 = Mock(spec=MaskFunction)

        # Collect with different masks
        collector.collect(small_dataset, mask=mask1)
        collector.collect(small_dataset, mask=mask2)

        # Verify different masks were used
        call1_mask = mock_collect.call_args_list[0].kwargs["mask"]
        call2_mask = mock_collect.call_args_list[1].kwargs["mask"]

        assert call1_mask is mask1
        assert call2_mask is mask2


class TestActivationCollectorEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_layers_list(self, mock_model, mock_tokenizer):
        """Test collector with empty layers list."""
        # This should work but may cause issues during collection
        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[]
        )

        assert collector.layers == []

    def test_negative_batch_size(self, mock_model, mock_tokenizer):
        """Test collector allows negative batch size (error handled by collect_activations)."""
        # Constructor doesn't validate, delegate to collect_activations
        collector = ActivationCollector(
            model=mock_model, tokenizer=mock_tokenizer, layers=[0], batch_size=-1
        )

        assert collector.batch_size == -1

    def test_invalid_hook_point(self, mock_model, mock_tokenizer):
        """Test collector allows invalid hook point (error handled by collect_activations)."""
        # Constructor doesn't validate, delegate to collect_activations
        collector = ActivationCollector(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layers=[0],
            hook_point="invalid",  # type: ignore
        )

        assert collector.hook_point == "invalid"

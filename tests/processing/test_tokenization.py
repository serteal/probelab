"""Tests for tokenization utilities."""

from unittest.mock import Mock, patch

import pytest
import torch

from probelib.datasets.base import DialogueDataset
from probelib.masks import all as all_tokens
from probelib.processing.tokenization import (
    _get_prefix_pattern,
    get_model_family,
    preprocess_dialogue,
    tokenize_dataset,
    tokenize_dialogues,
)
from probelib.types import Dialogue, DialogueDataType, Message


class MockDialogueDataset(DialogueDataset):
    """Concrete implementation of DialogueDataset for testing."""

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        """Not used since we pass dialogues directly."""
        raise NotImplementedError("This method should not be called in tests")


class TestPreprocessDialogue:
    """Test preprocess_dialogue function."""

    def test_basic_preprocessing(self):
        """Test basic dialogue preprocessing."""
        dialogue = Dialogue(
            [
                Message(role="user", content="Hello, how are you?"),
                Message(role="assistant", content="I'm doing well, thank you!"),
            ]
        )

        processed = preprocess_dialogue(dialogue)

        assert len(processed) == 2
        assert processed[0] == {"role": "user", "content": "Hello, how are you?"}
        assert processed[1] == {
            "role": "assistant",
            "content": "I'm doing well, thank you!",
        }

    def test_concatenate_same_role_messages(self):
        """Test concatenation of adjacent same-role messages."""
        dialogue = Dialogue(
            [
                Message(role="user", content="First part. "),
                Message(role="user", content="Second part. "),
                Message(role="assistant", content="Response."),
            ]
        )

        processed = preprocess_dialogue(dialogue)

        assert len(processed) == 2
        assert processed[0] == {"role": "user", "content": "First part.Second part."}
        assert processed[1] == {"role": "assistant", "content": "Response."}

    def test_fold_system_message(self):
        """Test folding system message into first user message."""
        dialogue = Dialogue(
            [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="What's the weather?"),
                Message(
                    role="assistant",
                    content="I don't have access to weather data.",
                ),
            ]
        )

        processed = preprocess_dialogue(dialogue, fold_system=True)

        assert len(processed) == 2
        assert processed[0]["role"] == "user"
        assert (
            processed[0]["content"]
            == "You are a helpful assistant.\n\nWhat's the weather?"
        )
        assert processed[1] == {
            "role": "assistant",
            "content": "I don't have access to weather data.",
        }

    def test_no_fold_system_message(self):
        """Test keeping system message separate when fold_system=False."""
        dialogue = Dialogue(
            [
                Message(role="system", content="System prompt"),
                Message(role="user", content="User message"),
            ]
        )

        processed = preprocess_dialogue(dialogue, fold_system=False)

        assert len(processed) == 2
        assert processed[0] == {"role": "system", "content": "System prompt"}
        assert processed[1] == {"role": "user", "content": "User message"}

    def test_empty_dialogue(self):
        """Test preprocessing empty dialogue."""
        dialogue = Dialogue([])
        processed = preprocess_dialogue(dialogue)
        assert processed == []

    def test_whitespace_handling(self):
        """Test that whitespace is stripped correctly."""
        dialogue = Dialogue(
            [
                Message(role="user", content="  Hello  \n"),
                Message(role="assistant", content="\tHi there\t  "),
            ]
        )

        processed = preprocess_dialogue(dialogue)

        assert processed[0]["content"] == "Hello"
        assert processed[1]["content"] == "Hi there"


class TestGetModelFamily:
    """Test get_model_family function."""

    def test_llama_family(self):
        """Test detecting Llama model family."""
        tokenizer = Mock()

        # Various Llama model names
        test_cases = [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-chat",
            "NousResearch/Llama-2-7b-hf",
            "codellama/CodeLlama-7b-Python-hf",
        ]

        for name in test_cases:
            tokenizer.name_or_path = name
            assert get_model_family(tokenizer) == "llama"

    def test_gemma_family(self):
        """Test detecting Gemma model family."""
        tokenizer = Mock()

        test_cases = [
            "google/gemma-7b",
            "google/gemma-2b-it",
            "gemma-7b-it",
        ]

        for name in test_cases:
            tokenizer.name_or_path = name
            assert get_model_family(tokenizer) == "gemma"

    def test_unknown_family_raises_error(self):
        """Test that unknown model families raise an error."""
        tokenizer = Mock()
        tokenizer.name_or_path = "unknown/model-name"

        # Should raise ValueError for unsupported architectures
        with pytest.raises(ValueError, match="Unable to detect architecture"):
            get_model_family(tokenizer)

    def test_case_insensitive(self):
        """Test that model family detection is case insensitive."""
        tokenizer = Mock()

        tokenizer.name_or_path = "META-LLAMA/LLAMA-2-7B"
        assert get_model_family(tokenizer) == "llama"

        tokenizer.name_or_path = "Google/GEMMA-7B"
        assert get_model_family(tokenizer) == "gemma"


class TestGetPrefixPattern:
    """Test _get_prefix_pattern function."""

    def test_gemma_pattern(self):
        """Test Gemma prefix pattern."""
        pattern = _get_prefix_pattern("gemma")

        # Test various Gemma prefixes
        test_cases = [
            "<bos><start_of_turn>user\n",
            "<pad><pad><bos><start_of_turn>model\n",
            "<end_of_turn>\n<start_of_turn>user\n",
            "\n\n",
        ]

        for case in test_cases:
            match = pattern.match(case)
            assert match is not None, f"Failed to match: {case}"

    def test_llama_pattern(self):
        """Test Llama prefix pattern."""
        pattern = _get_prefix_pattern("llama")

        # Test various Llama prefixes
        test_cases = [
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            "<|pad|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "\n\n",
        ]

        for case in test_cases:
            match = pattern.match(case)
            assert match is not None, f"Failed to match: {case}"

        # Test with date info
        with_date = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 15 Jan 2024\n\n"
        match = pattern.match(with_date)
        assert match is not None

    def test_default_pattern(self):
        """Test default fallback pattern."""
        pattern = _get_prefix_pattern("unknown")

        # Should match simple separators
        test_cases = ["", "\n\n"]

        for case in test_cases:
            match = pattern.match(case)
            assert match is not None, f"Failed to match: {case}"


class MockBatchEncoding(dict):
    """Mock BatchEncoding that supports char_to_token method."""

    def __init__(self, data: dict, seq_len: int = 3):
        super().__init__(data)
        self._seq_len = seq_len

    def char_to_token(self, batch_idx: int, char_idx: int) -> int | None:
        """Map character index to token index."""
        # Simple mapping: ~10 chars per token
        token_idx = char_idx // 10
        if token_idx >= self._seq_len:
            return None
        return token_idx


class TestTokenizeDialogues:
    """Test tokenize_dialogues function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.name_or_path = "meta-llama/Llama-2-7b"
        tokenizer.padding_side = "right"
        tokenizer.chat_template = "mock_template"
        tokenizer.all_special_ids = [0, 1]

        # Mock apply_chat_template
        tokenizer.apply_chat_template = Mock(
            return_value=[
                "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHi",
                "<|start_header_id|>user<|end_header_id|>\n\nTest<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nResponse",
            ]
        )

        # Mock tokenizer call to return BatchEncoding-like object
        def tokenizer_call(*args, **kwargs):
            n_samples = len(args[0]) if args else 2
            seq_len = 20
            data = {
                "input_ids": torch.randint(2, 1000, (n_samples, seq_len)),
                "attention_mask": torch.ones(n_samples, seq_len),
            }
            return MockBatchEncoding(data, seq_len=seq_len)

        tokenizer.side_effect = tokenizer_call

        return tokenizer

    def test_basic_tokenization(self, mock_tokenizer):
        """Test basic dialogue tokenization."""
        dialogues = [
            Dialogue(
                [
                    Message(role="user", content="Hello"),
                    Message(role="assistant", content="Hi"),
                ]
            ),
            Dialogue(
                [
                    Message(role="user", content="Test"),
                    Message(role="assistant", content="Response"),
                ]
            ),
        ]

        result = tokenize_dialogues(
            tokenizer=mock_tokenizer,
            dialogues=dialogues,
            mask=all_tokens(),
            device="cpu",
        )

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "detection_mask" in result
        # Check shapes match (seq_len determined by mock)
        assert result["input_ids"].shape[0] == 2
        assert result["attention_mask"].shape == result["input_ids"].shape
        assert result["detection_mask"].shape == result["input_ids"].shape

    def test_all_mask_selects_all_tokens(self, mock_tokenizer):
        """Test that all() mask selects all real tokens."""
        dialogues = [
            Dialogue(
                [
                    Message(role="user", content="Hello"),
                    Message(role="assistant", content="Hi"),
                ]
            ),
        ]

        result = tokenize_dialogues(
            tokenizer=mock_tokenizer,
            dialogues=dialogues,
            mask=all_tokens(),
            device="cpu",
        )

        expected = result["attention_mask"].bool()
        assert torch.equal(result["detection_mask"], expected)

    def test_gemma_system_folding(self, mock_tokenizer):
        """Test that Gemma models fold system messages."""
        mock_tokenizer.name_or_path = "google/gemma-7b"

        dialogues = [
            Dialogue(
                [
                    Message(role="system", content="System prompt"),
                    Message(role="user", content="User message"),
                ]
            ),
        ]

        tokenize_dialogues(
            tokenizer=mock_tokenizer,
            dialogues=dialogues,
            mask=all_tokens(),
            device="cpu",
        )

        # Check that apply_chat_template was called with folded dialogue
        call_args = mock_tokenizer.apply_chat_template.call_args[0][0]
        assert len(call_args) == 1  # One dialogue
        assert len(call_args[0]) == 1  # System folded into user
        assert call_args[0][0]["role"] == "user"
        assert "System prompt" in call_args[0][0]["content"]

    def test_device_placement(self, mock_tokenizer):
        """Test tensor device placement."""
        dialogues = [Dialogue([Message(role="user", content="Test")])]

        # Test CPU placement
        result = tokenize_dialogues(
            tokenizer=mock_tokenizer,
            dialogues=dialogues,
            mask=all_tokens(),
            device="cpu",
        )

        assert result["input_ids"].device.type == "cpu"
        assert result["attention_mask"].device.type == "cpu"
        assert result["detection_mask"].device.type == "cpu"

    def test_tokenizer_kwargs(self, mock_tokenizer):
        """Test passing additional tokenizer kwargs."""
        dialogues = [Dialogue([Message(role="user", content="Test")])]

        tokenize_dialogues(
            tokenizer=mock_tokenizer,
            dialogues=dialogues,
            mask=all_tokens(),
            max_length=512,
            truncation=True,
        )

        # Check that kwargs were passed to tokenizer
        _, kwargs = mock_tokenizer.call_args
        assert kwargs["max_length"] == 512
        assert kwargs["truncation"]
        assert kwargs["return_tensors"] == "pt"  # Default
        assert kwargs["padding"]  # Default

    def test_missing_attention_mask_error(self, mock_tokenizer):
        """Test error when tokenizer doesn't return attention mask."""
        # Override side_effect to return dict without attention_mask
        mock_tokenizer.side_effect = None
        mock_tokenizer.return_value = MockBatchEncoding(
            {"input_ids": torch.tensor([[1, 2, 3]])}, seq_len=3
        )

        dialogues = [Dialogue([Message(role="user", content="Test")])]

        with pytest.raises(
            ValueError, match="Tokenizer output must include attention mask"
        ):
            tokenize_dialogues(
                tokenizer=mock_tokenizer,
                dialogues=dialogues,
                mask=all_tokens(),
            )

    def test_generation_prompt_options(self, mock_tokenizer):
        """Test add_generation_prompt option."""
        dialogues = [Dialogue([Message(role="user", content="Test")])]

        # Test with generation prompt
        tokenize_dialogues(
            tokenizer=mock_tokenizer,
            dialogues=dialogues,
            mask=all_tokens(),
            add_generation_prompt=True,
        )

        # Check apply_chat_template was called with add_generation_prompt=True
        _, kwargs = mock_tokenizer.apply_chat_template.call_args
        assert kwargs["add_generation_prompt"]


class TestTokenizeDataset:
    """Test tokenize_dataset function."""

    def test_tokenize_dataset_wrapper(self):
        """Test that tokenize_dataset properly wraps tokenize_dialogues."""
        dialogues = [
            Dialogue(
                [
                    Message(role="user", content="Hello"),
                    Message(role="assistant", content="Hi"),
                ]
            ),
        ]
        dataset = MockDialogueDataset(dialogues=dialogues, labels=[1])

        mock_tokenizer = Mock()
        mock_tokenizer.name_or_path = "llama"
        mask = all_tokens()

        with patch(
            "probelib.processing.tokenization.tokenize_dialogues"
        ) as mock_tokenize:
            mock_return = {"input_ids": torch.tensor([[1, 2, 3]])}
            mock_tokenize.return_value = mock_return

            result = tokenize_dataset(
                dataset=dataset,
                tokenizer=mock_tokenizer,
                mask=mask,
                device="cuda",
                max_length=512,
            )

        assert result == mock_return

        # Check that tokenize_dialogues was called with correct args
        mock_tokenize.assert_called_once()
        call_kwargs = mock_tokenize.call_args[1]
        assert call_kwargs["tokenizer"] == mock_tokenizer
        assert call_kwargs["dialogues"] == dataset.dialogues
        assert call_kwargs["mask"] == mask
        assert call_kwargs["device"] == "cuda"
        assert call_kwargs["max_length"] == 512


class TestIntegration:
    """Integration tests for tokenization pipeline."""

    def test_full_tokenization_pipeline(self):
        """Test complete tokenization pipeline with realistic data."""
        # Create realistic dialogues
        dialogues = [
            Dialogue(
                [
                    Message(
                        role="system",
                        content="You are a helpful AI assistant.",
                    ),
                    Message(role="user", content="What is machine learning?"),
                    Message(
                        role="assistant",
                        content="Machine learning is a subset of AI...",
                    ),
                ]
            ),
            Dialogue(
                [
                    Message(role="user", content="Explain"),
                    Message(
                        role="user", content="neural networks"
                    ),  # Adjacent messages
                    Message(role="assistant", content="Neural networks are..."),
                ]
            ),
        ]

        # Create dataset with shuffling disabled for predictable test behavior
        dataset = MockDialogueDataset(
            dialogues=dialogues,
            labels=[1, 0],
            metadata={"source": "test"},
            padding={"llama": {"left": 2, "right": 1}},
            shuffle_upon_init=False,  # Disable shuffling for test
        )

        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.name_or_path = "meta-llama/Llama-2-7b"
        tokenizer.padding_side = "right"

        # Mock chat template formatting that dynamically formats based on input
        def apply_chat_template_mock(conversations, **kwargs):
            # Format each dialogue properly based on what's actually passed
            formatted = []
            for conv in conversations:
                parts = []
                for msg in conv:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        parts.append(
                            f"<|start_header_id|>system<|end_header_id|>\n\n{content}"
                        )
                    elif role == "user":
                        if parts:  # Add end token if not first message
                            parts.append("<|eot_id|>")
                        parts.append(
                            f"<|start_header_id|>user<|end_header_id|>\n\n{content}"
                        )
                    elif role == "assistant":
                        parts.append(
                            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{content}"
                        )
                formatted.append("".join(parts))
            return formatted

        tokenizer.apply_chat_template = Mock(side_effect=apply_chat_template_mock)

        # Mock tokenization that creates proper output based on formatted text
        def tokenizer_mock(texts, **kwargs):
            # Create a mock tokenizer output with char_to_token method
            batch_size = len(texts)
            seq_len = 50

            # Create char_to_token mock that handles the actual formatted text
            def char_to_token_mock(batch_idx, char_idx):
                # Simple mapping that returns token indices for valid character positions
                # This ensures content characters map to tokens
                if 0 <= char_idx < 500:  # Reasonable bounds for our test
                    return min(char_idx // 10, seq_len - 1)
                return None

            # Create mock output object with both dict-like access and char_to_token method
            output = {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
            }

            # Create a custom class that behaves like a dict but has char_to_token method
            class TokenizerOutput(dict):
                def __init__(self, data):
                    super().__init__(data)
                    self.char_to_token = Mock(side_effect=char_to_token_mock)

            return TokenizerOutput(output)

        tokenizer.side_effect = tokenizer_mock

        # Run tokenization
        result = tokenize_dataset(dataset, tokenizer, mask=all_tokens())

        # Verify structure
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "detection_mask" in result

        # Verify shapes match
        assert result["input_ids"].shape == (2, 50)
        assert result["attention_mask"].shape == (2, 50)
        assert result["detection_mask"].shape == (2, 50)

        # Verify detection mask is boolean
        assert result["detection_mask"].dtype == torch.bool

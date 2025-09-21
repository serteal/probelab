"""Tests for probelib.visualization module."""

import torch
from probelib.visualization import show_detection_mask_in_html


class TestVisualization:
    """Tests for visualization utilities."""

    def test_show_detection_mask_basic(self):
        """Test basic HTML generation for detection mask."""
        input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]])
        detection_mask = torch.tensor([[0, 1, 1, 0, 0, 0]], dtype=torch.bool)

        # Mock tokenizer
        class MockTokenizer:
            def convert_ids_to_tokens(self, id_):
                # Map individual IDs to tokens
                id_to_token = {
                    101: "[CLS]",
                    2023: "this",
                    2003: "is",
                    1037: "a",
                    3231: "test",
                    102: "[SEP]",
                }
                return id_to_token.get(id_, "[UNK]")

        tokenizer = MockTokenizer()

        # This should not raise an error
        html_output = show_detection_mask_in_html(
            input_ids[0],  # Pass single example
            detection_mask[0],  # Pass single example
            tokenizer,
        )

        # Check that HTML contains expected elements
        # The function returns an IPython HTML object, not a string
        html_str = html_output.data
        assert isinstance(html_str, str)
        assert "<span" in html_str
        assert "color: red" in html_str  # Detected tokens are in red

    def test_show_detection_mask_with_newlines(self):
        """Test HTML generation with newline tokens."""
        input_ids = torch.tensor([101, 2023, 2003, 1037, 3231, 102])
        detection_mask = torch.tensor([0, 1, 1, 0, 0, 0], dtype=torch.bool)

        class MockTokenizer:
            def convert_ids_to_tokens(self, id_):
                # Map individual IDs to tokens
                id_to_token = {
                    101: "[CLS]",
                    2023: "this",
                    2003: "Ċ",  # GPT-2 style newline token
                    1037: "a",
                    3231: "test",
                    102: "[SEP]",
                }
                return id_to_token.get(id_, "[UNK]")

        tokenizer = MockTokenizer()

        html_output = show_detection_mask_in_html(input_ids, detection_mask, tokenizer)

        html_str = html_output.data
        assert isinstance(html_str, str)
        assert "<br>" in html_str  # Newline tokens should be followed by <br>

    def test_show_detection_mask_multiple_examples(self):
        """Test with multiple examples."""
        input_ids = torch.tensor(
            [[101, 2023, 2003, 102, 0, 0], [101, 1037, 3231, 102, 0, 0]]
        )
        detection_mask = torch.tensor(
            [[0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=torch.bool
        )

        class MockTokenizer:
            def convert_ids_to_tokens(self, id_):
                # Map individual IDs to tokens
                id_to_token = {
                    101: "[CLS]",
                    2023: "this",
                    2003: "is",
                    1037: "a",
                    3231: "test",
                    102: "[SEP]",
                    0: "[PAD]",
                }
                return id_to_token.get(id_, "[UNK]")

        tokenizer = MockTokenizer()

        # Test first example
        html1 = show_detection_mask_in_html(input_ids[0], detection_mask[0], tokenizer)
        assert "this" in html1.data

        # Test second example
        html2 = show_detection_mask_in_html(input_ids[1], detection_mask[1], tokenizer)
        assert "test" in html2.data

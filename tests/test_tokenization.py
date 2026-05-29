"""Tests for build_token_metadata and padding expansion in tokenization.py."""

import re
import unittest
from unittest.mock import patch

import pytest
import torch

try:
    import transformers
except ImportError:
    transformers = None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        transformers is None,
        reason="tokenization tests require the optional tokenization dependency",
    ),
]
AutoTokenizer = transformers.AutoTokenizer if transformers else None

from probelab import masks
from probelab.tokenization import Tokens, build_token_metadata, tokenize_dialogues
from probelab.types import Message


def _load_tokenizer():
    """Load a small fast tokenizer for testing, skipping if unavailable."""
    from helpers import load_tokenizer_or_skip

    return load_tokenizer_or_skip("gpt2")


class TestBuildTokenMetadataRoles(unittest.TestCase):
    """Test role_ids assignment in build_token_metadata."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _load_tokenizer()

    def _build(self, dialogues, formatted_dialogues):
        """Helper: tokenize formatted text and build metadata."""
        encoding = self.tokenizer(
            formatted_dialogues,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        return build_token_metadata(
            dialogues, formatted_dialogues, self.tokenizer, encoding
        )

    def test_single_user_message(self):
        """All tokens in a single user message get role_id=1."""
        dialogues = [[Message("user", "Hello world")]]
        formatted = ["Hello world"]
        meta = self._build(dialogues, formatted)

        active = meta.attention_mask.bool()[0]
        self.assertTrue(active.any(), "Should have active tokens")
        self.assertTrue(
            (meta.role_ids_no_padding[0, active] == 1).all(),
            "All active tokens should be user (1)",
        )

    def test_single_assistant_message(self):
        """All tokens in a single assistant message get role_id=2."""
        dialogues = [[Message("assistant", "I can help")]]
        formatted = ["I can help"]
        meta = self._build(dialogues, formatted)

        active = meta.attention_mask.bool()[0]
        self.assertTrue(
            (meta.role_ids_no_padding[0, active] == 2).all(),
            "All active tokens should be assistant (2)",
        )

    def test_user_and_assistant_distinct(self):
        """User and assistant tokens get distinct role_ids."""
        dialogues = [[Message("user", "Hi"), Message("assistant", "Hello there")]]
        formatted = ["Hi\n\nHello there"]
        meta = self._build(dialogues, formatted)

        has_user = (meta.role_ids_no_padding == 1).any()
        has_asst = (meta.role_ids_no_padding == 2).any()
        self.assertTrue(has_user, "Should have user tokens")
        self.assertTrue(has_asst, "Should have assistant tokens")

    def test_user_assistant_no_overlap(self):
        """User and assistant role regions don't overlap."""
        dialogues = [[Message("user", "Hi friend"), Message("assistant", "Hello there friend")]]
        formatted = ["Hi friend\n\nHello there friend"]
        meta = self._build(dialogues, formatted)

        user_mask = meta.role_ids_no_padding == 1
        asst_mask = meta.role_ids_no_padding == 2
        overlap = user_mask & asst_mask
        self.assertFalse(overlap.any(), "User and assistant should not overlap")

    def test_three_roles(self):
        """System, user, assistant all assigned correctly."""
        dialogues = [[
            Message("system", "Be helpful"),
            Message("user", "Hi"),
            Message("assistant", "Hello"),
        ]]
        formatted = ["Be helpful\n\nHi\n\nHello"]
        meta = self._build(dialogues, formatted)

        self.assertTrue((meta.role_ids_no_padding == 0).any(), "Should have system tokens")
        self.assertTrue((meta.role_ids_no_padding == 1).any(), "Should have user tokens")
        self.assertTrue((meta.role_ids_no_padding == 2).any(), "Should have assistant tokens")

    def test_message_boundaries(self):
        """Different messages get different boundary indices."""
        dialogues = [[Message("user", "Hi"), Message("assistant", "Hello there")]]
        formatted = ["Hi\n\nHello there"]
        meta = self._build(dialogues, formatted)

        has_msg0 = (meta.message_boundaries == 0).any()
        has_msg1 = (meta.message_boundaries == 1).any()
        self.assertTrue(has_msg0, "Should have message boundary 0")
        self.assertTrue(has_msg1, "Should have message boundary 1")

    def test_batch_independence(self):
        """Each sample in a batch gets independent metadata."""
        dialogues = [
            [Message("user", "Short")],
            [Message("user", "A longer message with more tokens")],
        ]
        formatted = ["Short", "A longer message with more tokens"]
        meta = self._build(dialogues, formatted)

        self.assertEqual(meta.role_ids_no_padding.shape[0], 2)
        for i in range(2):
            active = meta.attention_mask.bool()[i]
            self.assertTrue(
                (meta.role_ids_no_padding[i, active] == 1).any(),
                f"Sample {i} should have user tokens",
            )

    def test_empty_content_message(self):
        """Empty message content doesn't crash."""
        dialogues = [[Message("user", ""), Message("assistant", "Hello")]]
        formatted = ["\n\nHello"]
        meta = self._build(dialogues, formatted)
        # Should not raise, and assistant tokens should exist
        self.assertTrue((meta.role_ids_no_padding == 2).any())

    def test_metadata_output_shapes(self):
        """Metadata tensors have correct shapes."""
        dialogues = [
            [Message("user", "Hello")],
            [Message("user", "World")],
        ]
        formatted = ["Hello", "World"]
        meta = self._build(dialogues, formatted)

        batch, seq = meta.token_ids.shape
        self.assertEqual(meta.role_ids.shape, (batch, seq))
        self.assertEqual(meta.role_ids_no_padding.shape, (batch, seq))
        self.assertEqual(meta.message_boundaries.shape, (batch, seq))
        self.assertEqual(meta.attention_mask.shape, (batch, seq))


class TestPaddingExpansion(unittest.TestCase):
    """Test the padding expansion logic in build_token_metadata."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _load_tokenizer()

    def _build_with_padding(self, dialogues, formatted, pad_left, pad_right):
        """Build metadata with custom padding values."""
        encoding = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        template_config = {
            "prefix_pattern": re.compile(r"(\n\n)?"),
            "fold_system": False,
            "token_padding": (pad_left, pad_right),
        }
        with (
            patch("probelab.tokenization.get_template", return_value=template_config),
            patch("probelab.tokenization.detect_template", return_value="test"),
        ):
            return build_token_metadata(
                dialogues, formatted, self.tokenizer, encoding
            )

    def test_zero_padding_matches_no_padding(self):
        """With (0,0) padding, role_ids equals role_ids_no_padding."""
        dialogues = [[Message("user", "Hello world")]]
        formatted = ["Hello world"]
        meta = self._build_with_padding(dialogues, formatted, 0, 0)

        # With no BOS token added (add_special_tokens=False), they should match
        torch.testing.assert_close(
            meta.role_ids.int(), meta.role_ids_no_padding.int()
        )

    def test_padding_expands_region(self):
        """Padding expands role regions beyond original content tokens."""
        dialogues = [[Message("user", "Hi"), Message("assistant", "Hello there friend")]]
        formatted = ["Hi\n\nHello there friend"]
        meta = self._build_with_padding(dialogues, formatted, 2, 1)

        padded_assigned = (meta.role_ids >= 0).sum()
        no_pad_assigned = (meta.role_ids_no_padding >= 0).sum()
        self.assertGreaterEqual(
            padded_assigned.item(), no_pad_assigned.item(),
            "Padded version should have at least as many assigned tokens",
        )

    def test_padding_preserves_content_roles(self):
        """Original content tokens keep their roles after padding expansion."""
        dialogues = [[Message("user", "Hello world")]]
        formatted = ["Hello world"]
        meta = self._build_with_padding(dialogues, formatted, 2, 1)

        user_no_pad = meta.role_ids_no_padding == 1
        if user_no_pad.any():
            self.assertTrue(
                (meta.role_ids[user_no_pad] == 1).all(),
                "Original user tokens should still be user after padding",
            )

    def test_padding_does_not_exceed_sequence(self):
        """Padding expansion stays within sequence bounds."""
        dialogues = [[Message("user", "Hello world")]]
        formatted = ["Hello world"]
        meta = self._build_with_padding(dialogues, formatted, 100, 100)

        self.assertEqual(
            meta.role_ids.shape, meta.role_ids_no_padding.shape,
            "Shapes should be unchanged regardless of padding size",
        )

    def test_large_padding_covers_entire_sequence(self):
        """Very large padding should expand to cover the full sequence."""
        dialogues = [[Message("user", "Hello")]]
        formatted = ["Hello"]
        meta = self._build_with_padding(dialogues, formatted, 100, 100)

        active = meta.attention_mask.bool()[0]
        # With huge padding, all active tokens should get the user role
        assigned = meta.role_ids[0, active] >= 0
        self.assertTrue(assigned.all(), "Large padding should cover all active tokens")

    def test_padding_symmetry(self):
        """Padding left and right expand in correct directions."""
        # Use a message in the middle so we can observe directional expansion
        dialogues = [[
            Message("user", "question"),
            Message("assistant", "answer"),
        ]]
        formatted = ["question\n\nanswer"]

        # Only pad left
        meta_left = self._build_with_padding(dialogues, formatted, 3, 0)
        # Only pad right
        meta_right = self._build_with_padding(dialogues, formatted, 0, 3)

        # Both should expand but in different directions
        left_assigned = (meta_left.role_ids >= 0).sum()
        right_assigned = (meta_right.role_ids >= 0).sum()
        no_pad = self._build_with_padding(dialogues, formatted, 0, 0)
        no_pad_assigned = (no_pad.role_ids >= 0).sum()

        self.assertGreaterEqual(left_assigned.item(), no_pad_assigned.item())
        self.assertGreaterEqual(right_assigned.item(), no_pad_assigned.item())

    def test_multiple_roles_padding(self):
        """Padding expansion works correctly with multiple roles."""
        dialogues = [[
            Message("user", "question here"),
            Message("assistant", "answer here"),
        ]]
        formatted = ["question here\n\nanswer here"]
        meta = self._build_with_padding(dialogues, formatted, 1, 1)

        # Both roles should still exist
        has_user = (meta.role_ids == 1).any()
        has_asst = (meta.role_ids == 2).any()
        self.assertTrue(has_user, "User role should exist after padding")
        self.assertTrue(has_asst, "Assistant role should exist after padding")


class TestTokenizeDialoguesIntegration(unittest.TestCase):
    """Integration tests through tokenize_dialogues."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _load_tokenizer()

    def _tokenize(self, dialogues, formatted, mask):
        """Tokenize with patched formatting to bypass apply_chat_template issues."""
        template_config = {
            "prefix_pattern": re.compile(r"(\n\n)?"),
            "fold_system": False,
            "token_padding": (0, 0),
        }
        with (
            patch("probelab.tokenization.get_template", return_value=template_config),
            patch("probelab.tokenization.detect_template", return_value="test"),
        ):
            return tokenize_dialogues(
                self.tokenizer, dialogues, mask=mask,
            )

    def test_user_mask_selects_user_tokens(self):
        """User mask selects tokens from user messages."""
        dialogues = [[Message("user", "Hello friend"), Message("assistant", "Greetings")]]
        tokens = self._tokenize(dialogues, None, masks.user())
        self.assertTrue(tokens.detection_mask.any(), "User mask should select some tokens")

    def test_assistant_mask_selects_assistant_tokens(self):
        """Assistant mask selects tokens from assistant messages."""
        dialogues = [[Message("user", "Hello"), Message("assistant", "Greetings friend")]]
        tokens = self._tokenize(dialogues, None, masks.assistant())
        self.assertTrue(tokens.detection_mask.any(), "Assistant mask should select some tokens")

    def test_user_and_assistant_masks_disjoint(self):
        """User and assistant detection masks don't overlap."""
        dialogues = [[Message("user", "Hello friend"), Message("assistant", "Greetings friend")]]
        user_tokens = self._tokenize(dialogues, None, masks.user())
        asst_tokens = self._tokenize(dialogues, None, masks.assistant())

        overlap = user_tokens.detection_mask & asst_tokens.detection_mask
        self.assertFalse(overlap.any(), "User and assistant masks should not overlap")

    def test_all_mask_covers_both_roles(self):
        """All mask covers at least what user + assistant cover."""
        dialogues = [[Message("user", "Hi"), Message("assistant", "Hello")]]
        all_tok = self._tokenize(dialogues, None, masks.all())
        user_tok = self._tokenize(dialogues, None, masks.user())
        asst_tok = self._tokenize(dialogues, None, masks.assistant())

        combined = user_tok.detection_mask | asst_tok.detection_mask
        # All mask should cover everything that user + assistant cover
        covered = all_tok.detection_mask | ~combined  # True where all covers or combined is False
        self.assertTrue(covered.all())

    def test_tokens_output_structure(self):
        """Tokens object has flat+offsets layout."""
        dialogues = [[Message("user", "Hello world")]]
        tokens = self._tokenize(dialogues, None, masks.user())

        self.assertEqual(tokens.input_ids.ndim, 1)
        self.assertEqual(tokens.detection_mask.ndim, 1)
        self.assertEqual(tokens.input_ids.shape[0], tokens.detection_mask.shape[0])
        self.assertEqual(tokens.offsets.ndim, 1)
        self.assertEqual(tokens.offsets.shape[0], len(tokens) + 1)

    def test_batch_tokenization(self):
        """Multiple dialogues tokenized correctly as a batch."""
        dialogues = [
            [Message("user", "First question"), Message("assistant", "First answer")],
            [Message("user", "Second question"), Message("assistant", "Second answer")],
            [Message("user", "Third question"), Message("assistant", "Third answer")],
        ]
        tokens = self._tokenize(dialogues, None, masks.user())

        self.assertEqual(len(tokens), 3)
        for i in range(3):
            s, e = int(tokens.offsets[i]), int(tokens.offsets[i + 1])
            self.assertTrue(
                tokens.detection_mask[s:e].any(),
                f"Sample {i} should have detected tokens",
            )

    def test_custom_mask_receives_processed_dialogues(self):
        """Custom masks should see the same processed messages used for metadata."""
        seen: list[list[Message]] = []

        def mask(dialogues, metadata):
            seen.extend(dialogues)
            return metadata.attention_mask.bool()

        dialogues = [[
            Message("system", "Follow rules"),
            Message("user", "Hello"),
            Message("user", "again"),
        ]]
        template_config = {
            "prefix_pattern": re.compile(r"(\n\n)?"),
            "fold_system": True,
            "token_padding": (0, 0),
        }

        with (
            patch("probelab.tokenization.get_template", return_value=template_config),
            patch("probelab.tokenization.detect_template", return_value="test"),
        ):
            tokenize_dialogues(self.tokenizer, dialogues, mask=mask)

        self.assertEqual([[m.role for m in d] for d in seen], [["user"]])
        self.assertIn("Follow rules", seen[0][0].content)
        self.assertIn("Hello again", seen[0][0].content)


class TestTokensTruncate(unittest.TestCase):
    """Tests for flat+offsets token truncation."""

    def _tokens(self):
        return Tokens(
            input_ids=torch.arange(10),
            offsets=torch.tensor([0, 6, 10]),
            detection_mask=torch.tensor(
                [False, False, False, False, True, True, False, True, False, False]
            ),
            pad_token_id=0,
            padding_side="right",
            formatted_texts=("sample0", "sample1"),
        )

    def test_left_truncate_preserves_detection_by_default(self):
        tokens = self._tokens()

        truncated = tokens.truncate(3, side="left")

        self.assertEqual(truncated.input_ids.tolist(), [3, 4, 5, 7, 8, 9])
        self.assertEqual(truncated.offsets.tolist(), [0, 3, 6])
        self.assertEqual(
            truncated.detection_mask.tolist(),
            [False, True, True, True, False, False],
        )
        self.assertIsNone(truncated.formatted_texts)

    def test_right_truncate(self):
        tokens = Tokens(
            input_ids=torch.arange(7),
            offsets=torch.tensor([0, 4, 7]),
            detection_mask=torch.tensor([True, False, False, False, True, False, False]),
            pad_token_id=0,
            padding_side="right",
            formatted_texts=("a", "b"),
        )

        truncated = tokens.truncate(2, side="right")

        self.assertEqual(truncated.input_ids.tolist(), [0, 1, 4, 5])
        self.assertEqual(truncated.offsets.tolist(), [0, 2, 4])
        self.assertEqual(truncated.detection_mask.tolist(), [True, False, True, False])

    def test_raises_when_preserved_tokens_would_be_removed(self):
        tokens = self._tokens()
        preserve_mask = torch.zeros_like(tokens.detection_mask)
        preserve_mask[0] = True

        with self.assertRaises(ValueError):
            tokens.truncate(3, side="left", preserve_mask=preserve_mask)

    def test_retains_formatted_texts_when_unchanged(self):
        tokens = self._tokens()

        truncated = tokens.truncate(10)

        self.assertEqual(truncated.input_ids.tolist(), tokens.input_ids.tolist())
        self.assertEqual(truncated.formatted_texts, tokens.formatted_texts)

    def test_rejects_bad_preserve_shape(self):
        tokens = self._tokens()

        with self.assertRaises(ValueError):
            tokens.truncate(3, preserve_mask=torch.ones(3, dtype=torch.bool))


class TestTokensIndexing(unittest.TestCase):
    """Tests for flat+offsets sample slicing."""

    def _tokens(self):
        return Tokens(
            input_ids=torch.tensor([10, 11, 20, 21, 22]),
            offsets=torch.tensor([0, 2, 5]),
            detection_mask=torch.tensor([True, False, False, True, True]),
            pad_token_id=0,
            padding_side="right",
            formatted_texts=("first", "second"),
        )

    def test_negative_integer_index_selects_last_sample(self):
        tokens = self._tokens()

        selected = tokens[-1]

        self.assertEqual(selected.input_ids.tolist(), [20, 21, 22])
        self.assertEqual(selected.offsets.tolist(), [0, 3])
        self.assertEqual(selected.detection_mask.tolist(), [False, True, True])
        self.assertEqual(selected.formatted_texts, ("second",))

    def test_negative_list_index_selects_last_sample(self):
        tokens = self._tokens()

        selected = tokens[[-1]]

        self.assertEqual(selected.input_ids.tolist(), [20, 21, 22])
        self.assertEqual(selected.offsets.tolist(), [0, 3])
        self.assertEqual(selected.detection_mask.tolist(), [False, True, True])


if __name__ == "__main__":
    unittest.main()

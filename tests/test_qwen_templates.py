"""Tests for Qwen2.5 and Qwen3 chat template support."""

import inspect
import re
import unittest
from types import SimpleNamespace

from probelab.processing.chat_templates import (
    TEMPLATES,
    _QWEN_PREFIX,
    detect_template,
)
from probelab.processing.tokenization import tokenize_dataset, tokenize_dialogues


# =============================================================================
# Qwen prefix regex tests
# =============================================================================


class TestQwenPrefixPattern(unittest.TestCase):
    """Test _QWEN_PREFIX regex matches ChatML-formatted text."""

    def test_matches_first_message_prefix(self):
        text = "<|im_start|>user\nHello"
        m = re.match(_QWEN_PREFIX, text)
        self.assertIsNotNone(m)
        self.assertEqual(m.end(), len("<|im_start|>user\n"))

    def test_matches_system_role(self):
        text = "<|im_start|>system\nYou are helpful."
        m = re.match(_QWEN_PREFIX, text)
        self.assertIsNotNone(m)
        self.assertEqual(m.end(), len("<|im_start|>system\n"))

    def test_matches_assistant_role(self):
        text = "<|im_start|>assistant\nSure!"
        m = re.match(_QWEN_PREFIX, text)
        self.assertIsNotNone(m)
        self.assertEqual(m.end(), len("<|im_start|>assistant\n"))

    def test_matches_subsequent_message(self):
        text = "<|im_end|>\n<|im_start|>assistant\nHi there"
        m = re.match(_QWEN_PREFIX, text)
        self.assertIsNotNone(m)
        self.assertEqual(m.end(), len("<|im_end|>\n<|im_start|>assistant\n"))

    def test_matches_thinking_block(self):
        text = "<|im_start|>assistant\n<think>\n\n</think>\n\nActual answer"
        m = re.match(_QWEN_PREFIX, text)
        self.assertIsNotNone(m)
        self.assertEqual(
            m.end(),
            len("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        )

    def test_matches_subsequent_with_thinking(self):
        text = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer"
        m = re.match(_QWEN_PREFIX, text)
        self.assertIsNotNone(m)
        self.assertEqual(
            m.end(),
            len("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        )

    def test_no_thinking_for_user(self):
        """User messages should not have thinking blocks."""
        text = "<|im_start|>user\nQuestion?"
        m = re.match(_QWEN_PREFIX, text)
        self.assertIsNotNone(m)
        # Group 3 is the thinking capture in the first alternative
        self.assertIsNone(m.group(3))

    def test_matches_double_newline(self):
        """Regex also matches a bare double-newline."""
        text = "\n\nsome content"
        m = re.match(_QWEN_PREFIX, text)
        self.assertIsNotNone(m)
        self.assertEqual(m.end(), 2)

    def test_full_dialogue_parse(self):
        """Walk through a full Qwen2.5-style dialogue consuming prefixes."""
        dialogue = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi!"
        )
        pos = 0
        messages_found = []
        while pos < len(dialogue):
            m = re.match(_QWEN_PREFIX, dialogue[pos:])
            if m and m.end() > 0:
                pos += m.end()
            else:
                # Read content until next prefix marker or end
                end = dialogue.find("<|im_end|>", pos)
                if end == -1:
                    end = len(dialogue)
                messages_found.append(dialogue[pos:end])
                pos = end
        self.assertEqual(len(messages_found), 3)

    def test_qwen3_dialogue_with_thinking(self):
        """Walk through a Qwen3-style dialogue with thinking blocks."""
        dialogue = (
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n4"
        )
        pos = 0
        messages_found = []
        while pos < len(dialogue):
            m = re.match(_QWEN_PREFIX, dialogue[pos:])
            if m and m.end() > 0:
                pos += m.end()
            else:
                end = dialogue.find("<|im_end|>", pos)
                if end == -1:
                    end = len(dialogue)
                messages_found.append(dialogue[pos:end])
                pos = end
        self.assertEqual(len(messages_found), 2)
        self.assertEqual(messages_found[0], "What is 2+2?")
        self.assertEqual(messages_found[1], "4")


# =============================================================================
# detect_template tests
# =============================================================================


class TestDetectTemplate(unittest.TestCase):
    """Test that detect_template recognizes Qwen model names."""

    def _tokenizer(self, name: str):
        return SimpleNamespace(name_or_path=name)

    def test_detects_qwen2_5(self):
        tok = self._tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(detect_template(tok), "qwen")

    def test_detects_qwen2(self):
        tok = self._tokenizer("Qwen/Qwen2-7B-Instruct")
        self.assertEqual(detect_template(tok), "qwen")

    def test_detects_qwen3(self):
        tok = self._tokenizer("Qwen/Qwen3-0.6B")
        self.assertEqual(detect_template(tok), "qwen3")

    def test_qwen3_not_misdetected_as_qwen(self):
        tok = self._tokenizer("Qwen/Qwen3-0.6B")
        result = detect_template(tok)
        self.assertNotEqual(result, "qwen")
        self.assertEqual(result, "qwen3")

    def test_llama_still_detected(self):
        tok = self._tokenizer("meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(detect_template(tok), "llama")

    def test_gemma_still_detected(self):
        tok = self._tokenizer("google/gemma-2-9b-it")
        self.assertEqual(detect_template(tok), "gemma")

    def test_unknown_model(self):
        tok = self._tokenizer("some-other-model")
        self.assertEqual(detect_template(tok), "unknown")


# =============================================================================
# TEMPLATES entries
# =============================================================================


class TestQwenTemplateEntries(unittest.TestCase):
    """Test that TEMPLATES dict has correct Qwen entries."""

    def test_qwen_in_templates(self):
        self.assertIn("qwen", TEMPLATES)

    def test_qwen3_in_templates(self):
        self.assertIn("qwen3", TEMPLATES)

    def test_qwen_token_padding(self):
        self.assertEqual(TEMPLATES["qwen"]["token_padding"], (3, 2))

    def test_qwen3_token_padding(self):
        self.assertEqual(TEMPLATES["qwen3"]["token_padding"], (7, 2))

    def test_qwen_fold_system_false(self):
        self.assertFalse(TEMPLATES["qwen"]["fold_system"])

    def test_qwen3_fold_system_false(self):
        self.assertFalse(TEMPLATES["qwen3"]["fold_system"])

    def test_qwen_has_prefix_pattern(self):
        self.assertIs(TEMPLATES["qwen"]["prefix_pattern"], _QWEN_PREFIX)

    def test_qwen3_shares_prefix_pattern(self):
        self.assertIs(
            TEMPLATES["qwen3"]["prefix_pattern"],
            TEMPLATES["qwen"]["prefix_pattern"],
        )


# =============================================================================
# template_kwargs parameter
# =============================================================================


class TestTemplateKwargs(unittest.TestCase):
    """Test that template_kwargs parameter exists in tokenization functions."""

    def test_tokenize_dialogues_has_template_kwargs(self):
        sig = inspect.signature(tokenize_dialogues)
        self.assertIn("template_kwargs", sig.parameters)
        param = sig.parameters["template_kwargs"]
        self.assertEqual(param.default, None)

    def test_tokenize_dataset_has_template_kwargs(self):
        sig = inspect.signature(tokenize_dataset)
        self.assertIn("template_kwargs", sig.parameters)
        param = sig.parameters["template_kwargs"]
        self.assertEqual(param.default, None)


if __name__ == "__main__":
    unittest.main()

"""Offline tests for chat-template detection (no tokenizer download)."""

import unittest

from probelab.chat_templates import TEMPLATES, detect_template, get_template


class _FakeTokenizer:
    """Minimal stand-in exposing the attributes detect_template inspects."""

    def __init__(self, name_or_path="", chat_template=None):
        self.name_or_path = name_or_path
        self.chat_template = chat_template


class TestDetectTemplate(unittest.TestCase):
    def test_detects_from_name(self):
        self.assertEqual(detect_template(_FakeTokenizer("meta-llama/Llama-3")), "llama")
        self.assertEqual(detect_template(_FakeTokenizer("google/gemma-2b")), "gemma")
        self.assertEqual(detect_template(_FakeTokenizer("Qwen/Qwen3-8B")), "qwen3")
        self.assertEqual(detect_template(_FakeTokenizer("Qwen/Qwen2.5-7B")), "qwen")

    def test_falls_back_to_chat_template_markup(self):
        # A renamed checkpoint without a recognisable name still resolves via
        # its chat_template markup.
        tok = _FakeTokenizer("my-org/custom", chat_template="...<|start_header_id|>...")
        self.assertEqual(detect_template(tok), "llama")
        tok = _FakeTokenizer("my-org/custom", chat_template="...<start_of_turn>...")
        self.assertEqual(detect_template(tok), "gemma")
        tok = _FakeTokenizer("my-org/custom", chat_template="...<|im_start|>...")
        self.assertEqual(detect_template(tok), "qwen")

    def test_unknown_when_no_signal(self):
        self.assertEqual(detect_template(_FakeTokenizer("mystery")), "unknown")


class TestGetTemplate(unittest.TestCase):
    def test_explicit_override(self):
        cfg = get_template(_FakeTokenizer("mystery"), template="llama")
        self.assertIs(cfg, TEMPLATES["llama"])

    def test_unknown_override_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown template"):
            get_template(_FakeTokenizer("x"), template="not_a_template")

    def test_unknown_tokenizer_uses_default(self):
        cfg = get_template(_FakeTokenizer("mystery"))
        self.assertEqual(cfg["fold_system"], False)
        self.assertEqual(cfg["token_padding"], (0, 0))


if __name__ == "__main__":
    unittest.main()

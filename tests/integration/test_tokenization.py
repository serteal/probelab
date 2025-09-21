"""
Tokenization integration tests using helper assertions (no pre-saved goldens).

Checks that:
- input_ids and attention_mask match an HF-baseline (apply_chat_template + tokenize)
- detection_mask matches messages selected by role-based selection
"""

import pytest

import probelib as pl
from tests.utils.assertions import (
    assert_detection_mask_text_equal,
    assert_tokenization_equal,
)


@pytest.mark.integration
@pytest.mark.llama3
def test_llama3_tokenization_and_assistant_mask():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dialogues = [
        [
            pl.Message("system", "You are a helpful assistant."),
            pl.Message("user", "What is 2+2?"),
            pl.Message("assistant", "2+2 equals 4."),
        ],
        [
            pl.Message("user", "Say hello"),
            pl.Message("assistant", "Hello!"),
        ],
    ]

    # input_ids/attention_mask equality against direct HF baseline
    assert_tokenization_equal(
        model_name,
        dialogues,
    )

    # And the selected text matches exactly (assistant content only)
    assert_detection_mask_text_equal(
        model_name,
        dialogues,
        mask=pl.masks.assistant(include_padding=False),
        expected_texts=[
            "2+2 equals 4.",
            "Hello!",
        ],
    )
    assert_detection_mask_text_equal(
        model_name,
        dialogues,
        mask=pl.masks.assistant(),
        expected_texts=[
            "<|start_header_id|>assistant<|end_header_id|>\n\n2+2 equals 4.<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>",
        ],
    )


@pytest.mark.integration
@pytest.mark.gemma2
def test_gemma2_tokenization_and_assistant_mask():
    model_name = "google/gemma-2-2b-it"
    dialogues = [
        [
            pl.Message("system", "Follow the user instructions."),
            pl.Message("user", "Translate 'hello' to French."),
            pl.Message("assistant", "Bonjour"),
        ],
        [
            pl.Message("user", "What is the capital of France?"),
            pl.Message("assistant", "Paris."),
        ],
    ]

    # input_ids/attention_mask equality against direct HF baseline
    assert_tokenization_equal(
        model_name,
        dialogues,
    )

    # And the selected text matches exactly (assistant content only)
    assert_detection_mask_text_equal(
        model_name,
        dialogues,
        mask=pl.masks.assistant(include_padding=False),
        expected_texts=[
            "Bonjour",
            "Paris.",
        ],
    )
    assert_detection_mask_text_equal(
        model_name,
        dialogues,
        mask=pl.masks.assistant(),
        expected_texts=[
            "<start_of_turn>model\nBonjour<end_of_turn>",
            "<start_of_turn>model\nParis.<end_of_turn>",
        ],
    )

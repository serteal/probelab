"""Minimal integration tests for masks using message text equality."""

import pytest

import probelib as pl
from tests.utils.assertions import (
    assert_detection_mask_text_equal,
    get_formatted_dialogue_texts,
)

# Mark these as integration/network/model-specific since they use real tokenizers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.llama3,
    pytest.mark.gemma2,
]


def test_assistant_mask_content_llama3():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dialogues = [
        [
            pl.Message("system", "You are helpful."),
            pl.Message("user", "Hello"),
            pl.Message("assistant", "Hi!"),
        ]
    ]

    assert_detection_mask_text_equal(
        model_name,
        dialogues,
        mask=pl.masks.assistant(include_padding=False),
        expected_texts=["Hi!"],
        add_generation_prompt=False,
        padding_side="left",
    )


def test_assistant_mask_content_gemma2():
    model_name = "google/gemma-2-2b-it"
    dialogues = [
        [
            pl.Message("system", "Follow the user instructions."),
            pl.Message("user", "Translate 'hello' to French."),
            pl.Message("assistant", "Bonjour"),
        ]
    ]

    assert_detection_mask_text_equal(
        model_name,
        dialogues,
        mask=pl.masks.assistant(include_padding=False),
        expected_texts=["Bonjour"],
        add_generation_prompt=False,
        padding_side="left",
    )


def test_all_and_none_masks_llama3_multiple_dialogues():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dialogues = [
        [
            pl.Message("system", "You are helpful."),
            pl.Message("user", "Say hi"),
            pl.Message("assistant", "Hi!"),
        ],
        [
            pl.Message("user", "What is 2+2?"),
            pl.Message("assistant", "4"),
        ],
    ]

    # all() should decode to full sequence text
    expected_all = get_formatted_dialogue_texts(
        model_name, dialogues, add_generation_prompt=False, padding_side="left"
    )
    assert_detection_mask_text_equal(
        model_name,
        dialogues,
        mask=pl.masks.all(),
        expected_texts=expected_all,
        add_generation_prompt=False,
        padding_side="left",
    )

    # none() should decode to empty strings
    assert_detection_mask_text_equal(
        model_name,
        dialogues,
        mask=pl.masks.none(),
        expected_texts=["", ""],
        add_generation_prompt=False,
        padding_side="left",
    )


def test_role_masks_multiple_dialogues():
    # Test across both models to ensure consistency
    for model_name in [
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-2b-it",
    ]:
        if "gemma" in model_name:
            dialogues = [
                [
                    pl.Message("system", "Follow the user instructions."),
                    pl.Message("user", "Question A"),
                    pl.Message("assistant", "Answer A"),
                ],
                [
                    pl.Message("user", "Question B"),
                    pl.Message("assistant", "Answer B"),
                ],
            ]
            # With folding, the first user message includes the system content
            expected_user = [
                "Follow the user instructions.\n\nQuestion A",
                "Question B",
            ]
        else:
            dialogues = [
                [
                    pl.Message("system", "System Preamble."),
                    pl.Message("user", "Question A"),
                    pl.Message("assistant", "Answer A"),
                ],
                [
                    pl.Message("user", "Question B"),
                    pl.Message("assistant", "Answer B"),
                ],
            ]
            expected_user = ["Question A", "Question B"]

        # assistant content
        expected_assistant = ["Answer A", "Answer B"]
        assert_detection_mask_text_equal(
            model_name,
            dialogues,
            mask=pl.masks.assistant(include_padding=False),
            expected_texts=expected_assistant,
            add_generation_prompt=False,
            padding_side="left",
        )

        # user content
        assert_detection_mask_text_equal(
            model_name,
            dialogues,
            mask=pl.masks.user(include_padding=False),
            expected_texts=expected_user,
            add_generation_prompt=False,
            padding_side="left",
        )

        # system content: LLaMA has content, Gemma's system is folded (no separate content)
        if "gemma" in model_name:
            expected_system = ["", ""]
        else:
            expected_system = ["System Preamble.", ""]
        assert_detection_mask_text_equal(
            model_name,
            dialogues,
            mask=pl.masks.system(include_padding=False),
            expected_texts=expected_system,
            add_generation_prompt=False,
            padding_side="left",
        )


def test_text_masks_contains_and_regex_llama3():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dialogues = [
        [
            pl.Message("user", "Find the keyword ZEBRA here."),
            pl.Message("assistant", "I found ZEBRA and noted it."),
        ],
        [
            pl.Message("user", "Nothing to see here."),
            pl.Message("assistant", "No animals mentioned."),
        ],
    ]

    # contains should select the literal substring occurrences
    assert_detection_mask_text_equal(
        model_name,
        dialogues,
        mask=pl.masks.contains("ZEBRA"),
        expected_texts=["ZEBRA ZEBRA", ""],
        add_generation_prompt=False,
        padding_side="left",
        collapse_ws=True,
        case_insensitive=False,
    )

    # regex selects according to pattern; expect only first dialogue assistant occurrence 'ZEBRA'
    assert_detection_mask_text_equal(
        model_name,
        dialogues,
        mask=pl.masks.regex(r"ZEBRA"),
        expected_texts=["ZEBRA ZEBRA", ""],
        add_generation_prompt=False,
        padding_side="left",
        collapse_ws=True,
    )


def test_nth_message_mask_gemma2():
    model_name = "google/gemma-2-2b-it"
    dialogues = [
        [
            pl.Message("system", "Be precise."),
            pl.Message("user", "State the color."),
            pl.Message("assistant", "Blue."),
        ]
    ]

    # After folding: processed messages: [user("Be precise.\n\nState the color."), assistant("Blue.")]
    # nth_message(0) => folded user content
    assert_detection_mask_text_equal(
        model_name,
        dialogues,
        mask=pl.masks.nth_message(0),
        expected_texts=["Be precise.\n\nState the color."],
        add_generation_prompt=False,
        padding_side="left",
    )

    # nth_message(-1) => assistant content
    assert_detection_mask_text_equal(
        model_name,
        dialogues,
        mask=pl.masks.nth_message(-1),
        expected_texts=["Blue."],
        add_generation_prompt=False,
        padding_side="left",
    )

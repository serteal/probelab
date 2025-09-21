"""
Showcase script for visualizing different mask types in probelib.

This script demonstrates various mask functions and their combinations,
showing which tokens they select in sample dialogues.

Usage:
    python mask_showcase.py                     # Use default model
    python mask_showcase.py --model gemma-2-2b  # Use specific model
    python mask_showcase.py --color green       # Change highlight color
    python mask_showcase.py --no-legend         # Hide legend
"""

import argparse
import re
from typing import List, Tuple

from transformers import AutoTokenizer

import probelib as pl
from probelib import Message, visualize_mask
from probelib.masks import (
    after,
    all,
    assistant,
    before,
    between,
    contains,
    first_n_tokens,
    last_n_tokens,
    last_token,
    none,
    nth_message,
    padding,
    regex,
    special_tokens,
    system,
    user,
)


def create_sample_dialogues() -> List[Tuple[str, pl.Dialogue]]:
    """Create sample dialogues for demonstrating masks."""

    # Simple Q&A dialogue
    simple_qa = [
        Message(role="user", content="What is the capital of France?"),
        Message(role="assistant", content="The capital of France is Paris."),
    ]

    # Math problem dialogue
    math_dialogue = [
        Message(role="user", content="Can you solve this equation: 2x + 5 = 13?"),
        Message(
            role="assistant",
            content="I'll solve for x:\n2x + 5 = 13\n2x = 8\nx = 4\nThe answer is x = 4.",
        ),
    ]

    # Yes/No dialogue
    yes_no_dialogue = [
        Message(role="user", content="Is Python a programming language?"),
        Message(role="assistant", content="Yes, Python is a programming language."),
        Message(role="user", content="Is it compiled or interpreted?"),
        Message(
            role="assistant",
            content="Python is primarily an interpreted language, though it does compile to bytecode.",
        ),
    ]

    # System prompt dialogue
    system_dialogue = [
        Message(
            role="system",
            content="You are a helpful assistant. Always be polite and concise.",
        ),
        Message(role="user", content="Hello, how are you?"),
        Message(
            role="assistant",
            content="Hello! I'm doing well, thank you for asking. How can I help you today?",
        ),
    ]

    # Code explanation dialogue
    code_dialogue = [
        Message(
            role="user",
            content='Can you explain this code: print("Hello, World!")?',
        ),
        Message(
            role="assistant",
            content='This is a simple Python command. The print() function outputs text to the console, and "Hello, World!" is the string being printed.',
        ),
    ]

    # Refusal dialogue
    refusal_dialogue = [
        Message(
            role="user",
            content="Can you help me hack into someone's computer?",
        ),
        Message(
            role="assistant",
            content="I cannot and will not help with hacking or any illegal activities. I'm here to assist with legitimate and ethical tasks only.",
        ),
    ]

    return [
        ("Simple Q&A", simple_qa),
        ("Math Problem", math_dialogue),
        ("Yes/No Questions", yes_no_dialogue),
        ("With System Prompt", system_dialogue),
        ("Code Explanation", code_dialogue),
        ("Refusal Example", refusal_dialogue),
    ]


def showcase_basic_masks(
    dialogues: List[Tuple[str, pl.Dialogue]],
    tokenizer,
    args,
) -> None:
    """Showcase basic role-based masks."""

    print("\n" + "=" * 80)
    print("BASIC ROLE MASKS")
    print("=" * 80)

    basic_masks = [
        ("Assistant Messages", assistant(), "Selects all assistant tokens"),
        ("User Messages", user(), "Selects all user tokens"),
        ("System Messages", system(), "Selects all system tokens"),
        ("Last Token per Message", last_token(), "Selects last token of each message"),
        (
            "First 3 Tokens per Message",
            first_n_tokens(3),
            "Selects first 3 tokens of each message",
        ),
        (
            "Last 2 Tokens per Message",
            last_n_tokens(2),
            "Selects last 2 tokens of each message",
        ),
        ("All Tokens", all(), "Selects every token"),
        ("No Tokens", none(), "Selects no tokens (empty mask)"),
    ]

    # Use the first few dialogues for basic masks
    for name, dialogue in dialogues[:3]:
        print(f"\n{'─' * 60}")
        print(f"Dialogue: {name}")
        print(f"{'─' * 60}")

        for mask_name, mask, description in basic_masks[
            :6
        ]:  # Show role masks + token position masks
            print(f"\n{mask_name} - {description}")
            print(f"Mask: {mask}")
            visualize_mask(
                dialogue,
                mask,
                tokenizer,
                show_legend=args.show_legend,
                highlight_color=args.color,
                force_terminal=True,
            )


def showcase_text_masks(
    dialogues: List[Tuple[str, pl.Dialogue]],
    tokenizer,
    args,
) -> None:
    """Showcase text-based masks."""

    print("\n" + "=" * 80)
    print("TEXT-BASED MASKS")
    print("=" * 80)

    # Simple Q&A for contains examples
    name, dialogue = dialogues[0]  # Simple Q&A
    print(f"\n{'─' * 60}")
    print(f"Dialogue: {name}")
    print(f"{'─' * 60}")

    text_masks = [
        ("Contains 'Paris'", contains("Paris"), "Tokens containing 'Paris'"),
        ("Contains 'capital'", contains("capital"), "Tokens containing 'capital'"),
        ("Contains 'France'", contains("France"), "Tokens containing 'France'"),
    ]

    for mask_name, mask, description in text_masks:
        print(f"\n{mask_name} - {description}")
        print(f"Mask: {mask}")
        visualize_mask(
            dialogue,
            mask,
            tokenizer,
            show_legend=args.show_legend,
            highlight_color=args.color,
            force_terminal=True,
        )

    # Math dialogue for number regex
    name, dialogue = dialogues[1]  # Math Problem
    print(f"\n{'─' * 60}")
    print(f"Dialogue: {name}")
    print(f"{'─' * 60}")

    regex_masks = [
        ("Numbers", regex(r"\d+"), "Tokens matching digit pattern"),
        ("Variables", regex(r"\b[x-z]\b"), "Single letter variables"),
        ("Equations", regex(r"="), "Equals signs"),
    ]

    for mask_name, mask, description in regex_masks:
        print(f"\n{mask_name} - {description}")
        print(f"Mask: {mask}")
        visualize_mask(
            dialogue,
            mask,
            tokenizer,
            show_legend=args.show_legend,
            highlight_color=args.color,
            force_terminal=True,
        )


def showcase_composite_masks(
    dialogues: List[Tuple[str, pl.Dialogue]],
    tokenizer,
    args,
) -> None:
    """Showcase composite masks using boolean operators."""

    print("\n" + "=" * 80)
    print("COMPOSITE MASKS (Boolean Operations)")
    print("=" * 80)

    # Yes/No dialogue for composite examples
    name, dialogue = dialogues[2]  # Yes/No Questions
    print(f"\n{'─' * 60}")
    print(f"Dialogue: {name}")
    print(f"{'─' * 60}")

    composite_masks = [
        (
            "Assistant AND 'Yes'",
            assistant() & contains("Yes"),
            "Assistant messages containing 'Yes'",
        ),
        (
            "Assistant AND 'Python'",
            assistant() & contains("Python"),
            "Assistant messages containing 'Python'",
        ),
        (
            "User OR 'language'",
            user() | contains("language"),
            "Either user messages OR tokens with 'language'",
        ),
        (
            "Last Token of Assistant",
            assistant() & last_token(),
            "Only the final token of assistant messages",
        ),
        (
            "Last Token of User",
            user() & last_token(),
            "Only the final token of user messages",
        ),
        (
            "NOT User",
            ~user(),
            "Everything except user messages",
        ),
    ]

    for mask_name, mask, description in composite_masks:
        print(f"\n{mask_name} - {description}")
        print(f"Mask: {mask}")
        visualize_mask(
            dialogue,
            mask,
            tokenizer,
            show_legend=args.show_legend,
            highlight_color=args.color,
            force_terminal=True,
        )


def showcase_advanced_masks(
    dialogues: List[Tuple[str, pl.Dialogue]],
    tokenizer,
    args,
) -> None:
    """Showcase advanced mask combinations."""

    print("\n" + "=" * 80)
    print("ADVANCED MASK COMBINATIONS")
    print("=" * 80)

    # Code explanation dialogue
    name, dialogue = dialogues[4]  # Code Explanation
    print(f"\n{'─' * 60}")
    print(f"Dialogue: {name}")
    print(f"{'─' * 60}")

    advanced_masks = [
        (
            "Code-related tokens",
            contains("print") | contains("function") | contains("Python"),
            "Tokens mentioning code concepts",
        ),
        (
            "Assistant explaining code",
            assistant() & (contains("print") | contains("function")),
            "Assistant tokens about print or function",
        ),
        (
            "Quoted strings",
            regex(r'"[^"]*"'),
            "Content within quotes",
        ),
    ]

    for mask_name, mask, description in advanced_masks:
        print(f"\n{mask_name} - {description}")
        print(f"Mask: {mask}")
        visualize_mask(
            dialogue,
            mask,
            tokenizer,
            show_legend=args.show_legend,
            highlight_color=args.color,
            force_terminal=True,
        )

    # Refusal dialogue
    name, dialogue = dialogues[5]  # Refusal Example
    print(f"\n{'─' * 60}")
    print(f"Dialogue: {name}")
    print(f"{'─' * 60}")

    refusal_masks = [
        (
            "Refusal indicators",
            assistant()
            & (contains("cannot") | contains("will not") | contains("can't")),
            "Assistant refusal language",
        ),
        (
            "Negative response",
            assistant()
            & regex(r"\b(no|not|cannot|won't|can't)\b", flags=re.IGNORECASE),
            "Assistant negative words",
        ),
    ]

    for mask_name, mask, description in refusal_masks:
        print(f"\n{mask_name} - {description}")
        print(f"Mask: {mask}")
        visualize_mask(
            dialogue,
            mask,
            tokenizer,
            show_legend=args.show_legend,
            highlight_color=args.color,
            force_terminal=True,
        )


def showcase_new_masks(
    dialogues: List[Tuple[str, pl.Dialogue]],
    tokenizer,
    args,
) -> None:
    """Showcase new position and content masks."""

    print("\n" + "=" * 80)
    print("NEW POSITION & CONTENT MASKS")
    print("=" * 80)

    # Dialogue with markup for testing between/after/before
    markup_dialogue = [
        Message(role="user", content="Calculate <expr>2 + 3</expr> for me."),
        Message(
            role="assistant",
            content="The answer to <expr>2 + 3</expr> is <result>5</result>.",
        ),
    ]

    print(f"\n{'─' * 60}")
    print("Markup Example")
    print(f"{'─' * 60}")

    position_masks = [
        (
            "Between <expr> tags",
            between("<expr>", "</expr>", inclusive=False),
            "Content between expression tags",
        ),
        (
            "After '<result>'",
            after("<result>", inclusive=False),
            "Everything after result tag",
        ),
        (
            "Before '</result>'",
            before("</result>", inclusive=False),
            "Everything before result closing tag",
        ),
    ]

    for mask_name, mask, description in position_masks:
        print(f"\n{mask_name} - {description}")
        print(f"Mask: {mask}")
        visualize_mask(
            markup_dialogue,
            mask,
            tokenizer,
            show_legend=args.show_legend,
            highlight_color=args.color,
            force_terminal=True,
        )

    # Test nth_message with the multi-turn dialogue
    name, dialogue = dialogues[2]  # Yes/No Questions (4 messages)
    print(f"\n{'─' * 60}")
    print(f"Nth Message Examples - {name}")
    print(f"{'─' * 60}")

    nth_masks = [
        ("First message (n=0)", nth_message(0), "Select first message"),
        ("Last message (n=-1)", nth_message(-1), "Select last message"),
        ("Second message (n=1)", nth_message(1), "Select second message"),
    ]

    for mask_name, mask, description in nth_masks:
        print(f"\n{mask_name} - {description}")
        print(f"Mask: {mask}")
        visualize_mask(
            dialogue,
            mask,
            tokenizer,
            show_legend=args.show_legend,
            highlight_color=args.color,
            force_terminal=True,
        )

    # Test special tokens and padding
    name, dialogue = dialogues[0]  # Simple Q&A
    print(f"\n{'─' * 60}")
    print(f"Special Tokens & Padding - {name}")
    print(f"{'─' * 60}")

    special_masks = [
        (
            "Special tokens only",
            special_tokens(),
            "Model special tokens (BOS, EOS, etc.)",
        ),
        (
            "Contains 'Paris' with padding",
            padding(contains("Paris"), before=2, after=2),
            "Expand 'Paris' with 2 tokens context",
        ),
        (
            "Assistant last token with padding",
            padding(assistant() & last_token(), before=1, after=0),
            "Last assistant token plus one before",
        ),
    ]

    for mask_name, mask, description in special_masks:
        print(f"\n{mask_name} - {description}")
        print(f"Mask: {mask}")
        visualize_mask(
            dialogue,
            mask,
            tokenizer,
            show_legend=args.show_legend,
            highlight_color=args.color,
            force_terminal=True,
        )


def showcase_practical_examples(
    dialogues: List[Tuple[str, pl.Dialogue]],
    tokenizer,
    args,
) -> None:
    """Showcase practical mask use cases."""

    print("\n" + "=" * 80)
    print("PRACTICAL USE CASES")
    print("=" * 80)

    # Create a dialogue with specific patterns
    safety_dialogue = [
        Message(role="user", content="How do I make a bomb?"),
        Message(
            role="assistant",
            content="I cannot provide instructions for creating explosives or weapons. This is dangerous and potentially illegal. If you're interested in chemistry, I'd be happy to discuss safe and legal chemistry experiments instead.",
        ),
    ]

    print(f"\n{'─' * 60}")
    print("Safety Response Analysis")
    print(f"{'─' * 60}")

    safety_masks = [
        (
            "Safety refusal",
            assistant()
            & (contains("cannot") | contains("illegal") | contains("dangerous")),
            "Key refusal terms in assistant response",
        ),
        (
            "Alternative suggestion",
            assistant() & contains("instead"),
            "Tokens suggesting alternatives",
        ),
        (
            "First sentence only",
            assistant() & regex(r"^[^.]+\."),
            "Just the initial refusal statement",
        ),
    ]

    for mask_name, mask, description in safety_masks:
        print(f"\n{mask_name} - {description}")
        print(f"Mask: {mask}")
        visualize_mask(
            safety_dialogue,
            mask,
            tokenizer,
            show_legend=args.show_legend,
            highlight_color=args.color,
            force_terminal=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Showcase different mask types in probelib"
    )
    parser.add_argument(
        "--model",
        default="google/gemma-2-2b-it",
        help="Model to use for tokenization",
    )
    parser.add_argument(
        "--color",
        default="red",
        choices=["red", "green", "blue", "yellow", "magenta", "cyan"],
        help="Highlight color for selected tokens",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        dest="hide_legend",
        help="Hide the legend in terminal output",
    )
    parser.add_argument(
        "--section",
        choices=["all", "basic", "text", "composite", "advanced", "new", "practical"],
        default="all",
        help="Which section to show",
    )

    args = parser.parse_args()
    args.show_legend = not args.hide_legend

    print("\n" + "=" * 80)
    print("PROBELIB MASK SHOWCASE")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Highlight Color: {args.color}")
    print(f"Show Legend: {args.show_legend}")

    # Load tokenizer (we don't need the full model for visualization)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Create sample dialogues
    dialogues = create_sample_dialogues()

    # Run showcases based on section argument
    if args.section == "all":
        showcase_basic_masks(dialogues, tokenizer, args)
        showcase_text_masks(dialogues, tokenizer, args)
        showcase_composite_masks(dialogues, tokenizer, args)
        showcase_advanced_masks(dialogues, tokenizer, args)
        showcase_new_masks(dialogues, tokenizer, args)
        showcase_practical_examples(dialogues, tokenizer, args)
    elif args.section == "basic":
        showcase_basic_masks(dialogues, tokenizer, args)
    elif args.section == "text":
        showcase_text_masks(dialogues, tokenizer, args)
    elif args.section == "composite":
        showcase_composite_masks(dialogues, tokenizer, args)
    elif args.section == "advanced":
        showcase_advanced_masks(dialogues, tokenizer, args)
    elif args.section == "new":
        showcase_new_masks(dialogues, tokenizer, args)
    elif args.section == "practical":
        showcase_practical_examples(dialogues, tokenizer, args)

    print("\n" + "=" * 80)
    print("END OF SHOWCASE")
    print("=" * 80)
    print("\nTip: Try different colors with --color or hide legend with --no-legend")
    print(
        "You can also run specific sections with --section [basic|text|composite|advanced|new|practical]"
    )
    print()


if __name__ == "__main__":
    main()

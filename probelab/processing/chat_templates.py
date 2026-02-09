"""Chat template configuration for tokenization.

This module contains tokenizer-specific configuration (prefix patterns, system message
handling, token padding) separated from model architecture concerns.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


# LLaMA-style chat template pattern
_LLAMA_PREFIX = re.compile(
    r"((<\|pad\|>)*(<\|begin_of_text\|>))?"
    r"(<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n"
    r"(Cutting Knowledge Date: December 2023\nToday Date: \d\d \w\w\w \d{4}\n\n)?)?"
    r"(<\|eot_id\|><\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n)?(\n\n)?"
)

# Gemma-style chat template pattern
_GEMMA_PREFIX = re.compile(
    r"(<pad>)*(<bos>)?<start_of_turn>(user|model)\n|"
    r"(<end_of_turn>\n)?<start_of_turn>(user|model)\n|(\n\n)?"
)

# Qwen-style ChatML template pattern (Qwen2.5, Qwen3)
_QWEN_PREFIX = re.compile(
    r"(<\|im_start\|>(system|user|assistant)\n(<think>\n\n</think>\n\n)?)?"
    r"(<\|im_end\|>\n<\|im_start\|>(system|user|assistant)\n(<think>\n\n</think>\n\n)?)?(\n\n)?"
)

# Default pattern for unknown templates
_DEFAULT_PREFIX = re.compile(r"(\n\n)?")


TEMPLATES: dict[str, dict] = {
    "llama": {
        "prefix_pattern": _LLAMA_PREFIX,
        "fold_system": False,
        "token_padding": (4, 1),  # (left, right)
    },
    "gemma": {
        "prefix_pattern": _GEMMA_PREFIX,
        "fold_system": True,
        "token_padding": (3, 2),
    },
    "qwen": {
        "prefix_pattern": _QWEN_PREFIX,
        "fold_system": False,
        "token_padding": (3, 2),
    },
    "qwen3": {
        "prefix_pattern": _QWEN_PREFIX,
        "fold_system": False,
        "token_padding": (7, 2),
    },
}


def detect_template(tokenizer: "PreTrainedTokenizerBase") -> str:
    """Detect template name from tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Template name ("llama", "gemma", etc.) or "unknown"
    """
    name = getattr(tokenizer, "name_or_path", "")
    name_lower = name.lower()

    if "llama" in name_lower:
        return "llama"
    if "gemma" in name_lower:
        return "gemma"
    if "qwen3" in name_lower:
        return "qwen3"
    if "qwen" in name_lower:
        return "qwen"

    return "unknown"


def get_template(tokenizer: "PreTrainedTokenizerBase") -> dict:
    """Get template configuration for a tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Dict with prefix_pattern, fold_system, token_padding
    """
    name = detect_template(tokenizer)

    if name in TEMPLATES:
        return TEMPLATES[name]

    # Default fallback
    return {
        "prefix_pattern": _DEFAULT_PREFIX,
        "fold_system": False,
        "token_padding": (0, 0),
    }

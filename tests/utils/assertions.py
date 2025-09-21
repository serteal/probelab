"""
Utilities for asserting equality between probelib outputs and independently-computed baselines.

Designed for small, deterministic tests with real LLaMA3/Gemma2 tokenizers/models.
"""

from __future__ import annotations


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelib as pl
from probelib.processing.activations import get_batches


def set_deterministic():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _detect_family(model_name: str) -> str:
    n = model_name.lower()
    if "llama" in n:
        return "llama"
    if "gemma" in n:
        return "gemma"
    return "llama"


def get_tokenizer(model_name: str, padding_side: str = "left"):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = padding_side
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def get_model(
    model_name: str,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    model.eval()
    return model


def _preprocess_dialogue(
    dialogue: list[pl.Message], family: str
) -> list[dict[str, str]]:
    # Fold system into first user for Gemma; concatenate adjacent same-role messages; strip whitespace
    processed: list[dict[str, str]] = []
    msgs = dialogue
    if family == "gemma" and msgs and msgs[0].role == "system":
        processed.append({"role": "user", "content": msgs[0].content.strip() + "\n\n"})
        msgs = msgs[1:]
    for m in msgs:
        if processed and processed[-1]["role"] == m.role:
            processed[-1]["content"] += m.content.strip()
        else:
            processed.append({"role": m.role, "content": m.content.strip()})
    return processed


def _map_roles_for_template(
    messages: list[dict[str, str]], family: str
) -> list[dict[str, str]]:
    if family != "gemma":
        return messages
    mapped: list[dict[str, str]] = []
    for msg in messages:
        role = msg["role"]
        if role == "assistant":
            role = "model"
        # Gemma templates don't use explicit system role post-folding
        if role == "system":
            # Skip explicit system; its content is folded already if present
            continue
        mapped.append({"role": role, "content": msg["content"]})
    return mapped


def _format_with_template(
    tokenizer,
    processed_dialogues: list[list[dict[str, str]]],
    add_generation_prompt: bool,
) -> list[str]:
    return tokenizer.apply_chat_template(
        processed_dialogues,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _tokenize_strings(
    tokenizer, strings: list[str], **kwargs
) -> dict[str, torch.Tensor]:
    defaults = {"return_tensors": "pt", "padding": True, "add_special_tokens": False}
    defaults.update(kwargs)
    out = tokenizer(strings, **defaults)
    # Normalize tensors
    for k, v in out.items():
        if isinstance(v, list):
            out[k] = torch.tensor(v)
    return out


def assert_tokenization_equal(
    model_name: str,
    dialogues: list[list[pl.Message]],
    *,
    add_generation_prompt: bool = False,
    padding_side: str = "left",
    device: str | torch.device = "cpu",
    **tokenize_kwargs,
) -> None:
    """Assert that probelib tokenization matches a baseline built from HF directly.

    Compares only input_ids and attention_mask. Detection mask equality is handled by
    assert_detection_mask_messages_equal or assert_detection_mask_tokens_equal.
    """
    set_deterministic()
    tok = get_tokenizer(model_name, padding_side=padding_side)
    family = _detect_family(model_name)

    # Baseline: preprocess and format
    processed = [_preprocess_dialogue(d, family) for d in dialogues]
    template_ready = [_map_roles_for_template(p, family) for p in processed]
    formatted = _format_with_template(tok, template_ready, add_generation_prompt)
    baseline = _tokenize_strings(tok, formatted, **tokenize_kwargs)

    # Actual via probelib
    actual = pl.processing.tokenize_dialogues(
        tokenizer=tok,
        dialogues=dialogues,
        device=device,
        add_generation_prompt=add_generation_prompt,
        **tokenize_kwargs,
    )

    assert torch.equal(actual["input_ids"].cpu(), baseline["input_ids"].cpu())
    assert torch.equal(actual["attention_mask"].cpu(), baseline["attention_mask"].cpu())


def get_formatted_dialogue_texts(
    model_name: str,
    dialogues: list[list[pl.Message]],
    *,
    add_generation_prompt: bool = False,
    padding_side: str = "left",
) -> list[str]:
    """Return chat-template-formatted texts for dialogues, model-aware.

    Useful for building expected strings that include special tokens and role markers
    without manually spelling them out in tests.
    """
    tok = get_tokenizer(model_name, padding_side=padding_side)
    family = _detect_family(model_name)
    processed = [_preprocess_dialogue(d, family) for d in dialogues]
    template_ready = [_map_roles_for_template(p, family) for p in processed]
    formatted = _format_with_template(tok, template_ready, add_generation_prompt)
    return [s.strip() for s in formatted]


# Removed token-based mask assertions in favor of text-equality assertions.


def assert_collect_activations_equal(
    model_name: str,
    dialogues: list[list[pl.Message]],
    layers: list[int],
    *,
    add_generation_prompt: bool = False,
    batch_size: int = 2,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    streaming: bool = True,
    padding_side: str = "left",
    **tokenize_kwargs,
) -> None:
    """Assert collect_activations equals HF hidden_states for the same layers, exactly.

    Also checks streaming assembly equals batch activations if streaming=True.
    """
    set_deterministic()

    tok = get_tokenizer(model_name, padding_side=padding_side)
    model = get_model(model_name, device=device, dtype=dtype)

    # Tokenize via probelib to get the exact attention_mask/seq shaping used by the library
    tokenized = pl.processing.tokenize_dialogues(
        tokenizer=tok,
        dialogues=dialogues,
        device=model.device,
        add_generation_prompt=add_generation_prompt,
        **tokenize_kwargs,
    )

    # Actual activations via library, batch mode
    acts = pl.processing.collect_activations(
        model=model,
        tokenizer=tok,
        data=dialogues,
        layers=layers,
        batch_size=batch_size,
        streaming=False,
        verbose=False,
        add_generation_prompt=add_generation_prompt,
    )

    # Baseline via HF hidden_states, mirroring trimmed batches
    n_samples, max_seq_len = tokenized["input_ids"].shape
    expected = torch.zeros(
        (len(layers), n_samples, max_seq_len, model.config.hidden_size),
        device=acts.activations.device,
        dtype=acts.activations.dtype,
    )
    with torch.inference_mode():
        for batch_inputs, batch_indices in get_batches(
            tokenized, batch_size=batch_size, tokenizer=tok
        ):
            out = model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hs = out.hidden_states
            stacked = torch.stack([hs[i + 1] for i in layers], dim=0).to(
                device=expected.device, dtype=expected.dtype
            )
            seq_len = stacked.shape[2]
            expected[:, batch_indices, -seq_len:] = stacked

    assert acts.activations.shape == expected.shape
    assert torch.equal(acts.activations, expected)
    assert torch.equal(acts.input_ids, tokenized["input_ids"].to(acts.input_ids.device))
    assert torch.equal(
        acts.attention_mask, tokenized["attention_mask"].to(acts.attention_mask.device)
    )
    assert torch.equal(
        acts.detection_mask, tokenized["detection_mask"].to(acts.detection_mask.device)
    )
    assert acts.layer_indices == layers

    if streaming:
        acts_iter = pl.processing.collect_activations(
            model=model,
            tokenizer=tok,
            data=dialogues,
            layers=layers,
            batch_size=batch_size,
            streaming=True,
            verbose=False,
            add_generation_prompt=add_generation_prompt,
        )
        full = torch.zeros_like(expected)
        for batch in acts_iter:
            full[:, batch.batch_indices, -batch.seq_len :] = batch.activations.to(
                full.device, full.dtype
            )
        assert torch.equal(full, acts.activations)


def _normalize_text(
    s: str, *, strip: bool = True, collapse_ws: bool = False, lower: bool = False
) -> str:
    if strip:
        s = s.strip()
    if collapse_ws:
        import re as _re

        s = _re.sub(r"\s+", " ", s)
    if lower:
        s = s.lower()
    return s


def assert_detection_mask_text_equal(
    model_name: str,
    dialogues: list[list[pl.Message]],
    mask: object,
    expected_texts: list[str],
    *,
    add_generation_prompt: bool = False,
    padding_side: str = "left",
    device: str | torch.device = "cpu",
    normalize_expected: bool = True,
    collapse_ws: bool = False,
    case_insensitive: bool = False,
    **tokenize_kwargs,
) -> None:
    """Assert that decoding the selected tokens equals the expected text per sample.

    Decodes only tokens where detection_mask is True (respecting attention_mask). If selections
    are discontiguous, the decoded segments are concatenated in sequence order.
    """
    set_deterministic()
    tok = get_tokenizer(model_name, padding_side=padding_side)

    out = pl.processing.tokenize_dialogues(
        tokenizer=tok,
        dialogues=dialogues,
        mask=mask,
        device=device,
        add_generation_prompt=add_generation_prompt,
        **tokenize_kwargs,
    )

    ids = out["input_ids"].cpu()
    attn = out["attention_mask"].cpu().to(dtype=torch.bool)
    det = out["detection_mask"].cpu().to(dtype=torch.bool)

    assert ids.shape[0] == len(expected_texts), "Provide one expected string per sample"

    for i in range(ids.shape[0]):
        sel = det[i] & attn[i]
        selected_ids = ids[i][sel].tolist()
        decoded = tok.decode(selected_ids, skip_special_tokens=False)
        exp = expected_texts[i]
        if normalize_expected:
            decoded = _normalize_text(
                decoded, strip=True, collapse_ws=collapse_ws, lower=case_insensitive
            )
            exp = _normalize_text(
                exp, strip=True, collapse_ws=collapse_ws, lower=case_insensitive
            )
        assert decoded == exp, (
            f"Selected text mismatch for sample {i}:\nExpected: {exp!r}\nActual:   {decoded!r}"
        )

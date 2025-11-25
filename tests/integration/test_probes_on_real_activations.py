"""
Integration tests: train simple probes on real activations and validate outputs.

These tests ensure that end-to-end feature extraction (tokenization + activations)
is compatible with probe training APIs in both sequence_aggregation and
score_aggregation modes. Tests are kept very small and deterministic.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelib as pl
from probelib.types import AggregationMethod


def _cuda_required():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for real-model activation tests")


def _common_dialogues(model_family: str):
    if model_family == "llama3":
        return [
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
    else:  # gemma2
        return [
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


def _collect_small_activations(model_name: str, dialogues):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    acts = pl.processing.collect_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=dialogues,
        layers=[0],
        batch_size=2,
        streaming=False,
        verbose=False,
        add_generation_prompt=False,
        mask=pl.masks.assistant(include_padding=False),
    )
    return acts


@pytest.mark.integration
@pytest.mark.llama3
def test_logistic_sequence_mean_llama3():
    _cuda_required()
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dialogues = _common_dialogues("llama3")
    acts = _collect_small_activations(model_name, dialogues)

    labels = [1, 0]  # arbitrary small labels for 2 samples
    pipeline = pl.Pipeline([
        ("select", pl.preprocessing.SelectLayer(0)),
        ("agg", pl.preprocessing.Pool(AggregationMethod.MEAN)),
        ("probe", pl.probes.Logistic(
            device="cuda",
            random_state=0,
        )),
    ])
    pipeline.fit(acts, labels)
    probs = pipeline.predict_proba(acts)

    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    # non-degenerate predictions
    assert not torch.allclose(probs[0], probs[1])


@pytest.mark.integration
@pytest.mark.llama3
def test_logistic_token_mean_streaming_fit_llama3():
    _cuda_required()
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dialogues = _common_dialogues("llama3")
    acts = _collect_small_activations(model_name, dialogues)

    # Build a streaming iterator using the same parameters
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    acts_iter = pl.processing.collect_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=dialogues,
        layers=[0],
        batch_size=1,
        streaming=True,
        verbose=False,
        add_generation_prompt=False,
        mask=pl.masks.assistant(include_padding=False),
    )

    labels = [1, 0]
    # Note: Streaming (partial_fit) doesn't support post-transformers,
    # so we use sequence-level aggregation before the probe
    pipeline = pl.Pipeline([
        ("select", pl.preprocessing.SelectLayer(0)),
        ("agg", pl.preprocessing.Pool(AggregationMethod.MEAN)),
        ("probe", pl.probes.Logistic(
            device="cuda",
            random_state=0,
        )),
    ])

    # Use partial_fit for streaming
    for batch_acts in acts_iter:
        batch_labels = [labels[i] for i in batch_acts.batch_indices]
        pipeline.partial_fit(batch_acts, batch_labels)

    probs = pipeline.predict_proba(acts)

    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


@pytest.mark.integration
@pytest.mark.gemma2
def test_logistic_sequence_mean_gemma2():
    _cuda_required()
    model_name = "google/gemma-2-2b-it"
    dialogues = _common_dialogues("gemma2")
    acts = _collect_small_activations(model_name, dialogues)

    labels = [1, 0]
    pipeline = pl.Pipeline([
        ("select", pl.preprocessing.SelectLayer(0)),
        ("agg", pl.preprocessing.Pool(AggregationMethod.MEAN)),
        ("probe", pl.probes.Logistic(
            device="cuda",
            random_state=0,
        )),
    ])
    pipeline.fit(acts, labels)
    probs = pipeline.predict_proba(acts)

    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

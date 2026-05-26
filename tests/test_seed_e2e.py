"""End-to-end seed reproducibility test on real model activations.

Loads Llama-3.1-8B-Instruct, collects activations on a small slice of
circuit_breakers, and trains probes twice with the same seed to verify
bit-exact reproducibility.

Usage (via SLURM):
    sbatch tests/slurm_seed_e2e.sh

Or directly:
    cd probelab && uv run pytest tests/test_seed_e2e.py -v -s
"""

import os
import importlib.util

import pytest
import torch

try:
    import mirin as mi
except ImportError:
    mi = None

try:
    import transformers
except ImportError:
    transformers = None

AutoModelForCausalLM = transformers.AutoModelForCausalLM if transformers else None
AutoTokenizer = transformers.AutoTokenizer if transformers else None

import probelab as pl
from probelab.collection.mirin import collect_activations

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("PROBELAB_RUN_SEED_E2E") != "1",
        reason="requires gated 8B model access; set PROBELAB_RUN_SEED_E2E=1 to run",
    ),
    pytest.mark.skipif(
        mi is None,
        reason="seed e2e requires the optional mirin collection dependency",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("accelerate") is None,
        reason="device_map='auto' requires accelerate",
    ),
    pytest.mark.skipif(
        transformers is None,
        reason="seed e2e requires the optional tokenization dependency",
    ),
]


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 16
N_SAMPLES = 40          # small for speed
DATASET_SEED = 0        # fixed split
PROBE_SEED = 42


def _collect_once():
    """Load model, tokenize, collect activations (cached across tests)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = mi.Model(hf_model, rename=mi.renames.llm, tokenizer=tokenizer)

    ds = pl.datasets.load("circuit_breakers")
    pos = ds.positive.sample(N_SAMPLES // 2, seed=DATASET_SEED)
    neg = ds.negative.sample(N_SAMPLES // 2, seed=DATASET_SEED)
    ds_small = pos + neg
    train_ds, test_ds = ds_small.split(0.75, seed=DATASET_SEED)

    mask = pl.masks.assistant()
    train_tok = pl.tokenize_dataset(train_ds, tokenizer, mask=mask)
    test_tok = pl.tokenize_dataset(test_ds, tokenizer, mask=mask)

    train_acts = collect_activations(model, train_tok, layers=[LAYER], batch_size=4)
    test_acts = collect_activations(model, test_tok, layers=[LAYER], batch_size=4)

    return (
        train_acts, train_ds.labels,
        test_acts, test_ds.labels,
    )


_cache: dict = {}

def _get_data():
    if "data" not in _cache:
        _cache["data"] = _collect_once()
    return _cache["data"]


# ---------- Tests ----------

def test_logistic_seed_reproducibility():
    train_acts, train_labels, test_acts, test_labels = _get_data()
    train_pooled = train_acts.mean("s")
    test_pooled = test_acts.mean("s")

    p1 = pl.probes.Logistic(seed=PROBE_SEED, device="cpu").fit(train_pooled, train_labels)
    p2 = pl.probes.Logistic(seed=PROBE_SEED, device="cpu").fit(train_pooled, train_labels)

    pred1 = p1.predict(test_pooled)
    pred2 = p2.predict(test_pooled)

    print(f"\nLogistic preds (run 1): {pred1[:5]}")
    print(f"Logistic preds (run 2): {pred2[:5]}")
    assert torch.equal(pred1, pred2), f"Max diff: {(pred1 - pred2).abs().max().item()}"


def test_mlp_seed_reproducibility():
    train_acts, train_labels, test_acts, test_labels = _get_data()
    train_pooled = train_acts.mean("s")
    test_pooled = test_acts.mean("s")

    p1 = pl.probes.MLP(hidden_dim=64, n_epochs=30, seed=PROBE_SEED, device="cpu").fit(train_pooled, train_labels)
    p2 = pl.probes.MLP(hidden_dim=64, n_epochs=30, seed=PROBE_SEED, device="cpu").fit(train_pooled, train_labels)

    pred1 = p1.predict(test_pooled)
    pred2 = p2.predict(test_pooled)

    print(f"\nMLP preds (run 1): {pred1[:5]}")
    print(f"MLP preds (run 2): {pred2[:5]}")
    assert torch.equal(pred1, pred2), f"Max diff: {(pred1 - pred2).abs().max().item()}"


def test_mlp_different_seed_differs():
    train_acts, train_labels, test_acts, test_labels = _get_data()
    train_pooled = train_acts.mean("s")
    test_pooled = test_acts.mean("s")

    p1 = pl.probes.MLP(hidden_dim=64, n_epochs=30, seed=PROBE_SEED, device="cpu").fit(train_pooled, train_labels)
    p2 = pl.probes.MLP(hidden_dim=64, n_epochs=30, seed=999, device="cpu").fit(train_pooled, train_labels)

    pred1 = p1.predict(test_pooled)
    pred2 = p2.predict(test_pooled)

    print(f"\nMLP seed=42 preds: {pred1[:5]}")
    print(f"MLP seed=999 preds: {pred2[:5]}")
    assert not torch.equal(pred1, pred2), "Different seeds should give different results"


def test_attention_seed_reproducibility():
    """Attention probe uses randperm for train/val split + per-epoch shuffle."""
    train_acts, train_labels, test_acts, test_labels = _get_data()

    p1 = pl.probes.Attention(
        hidden_dim=32, n_epochs=50, seed=PROBE_SEED, device="cpu",
    ).fit(train_acts, train_labels)
    p2 = pl.probes.Attention(
        hidden_dim=32, n_epochs=50, seed=PROBE_SEED, device="cpu",
    ).fit(train_acts, train_labels)

    pred1 = p1.predict(test_acts)
    pred2 = p2.predict(test_acts)

    print(f"\nAttention preds (run 1): {pred1[:5]}")
    print(f"Attention preds (run 2): {pred2[:5]}")
    assert torch.equal(pred1, pred2), f"Max diff: {(pred1 - pred2).abs().max().item()}"


def test_multimax_seed_reproducibility():
    train_acts, train_labels, test_acts, test_labels = _get_data()

    p1 = pl.probes.MultiMax(
        n_epochs=10, seed=PROBE_SEED, device="cpu",
    ).fit(train_acts, train_labels)
    p2 = pl.probes.MultiMax(
        n_epochs=10, seed=PROBE_SEED, device="cpu",
    ).fit(train_acts, train_labels)

    pred1 = p1.predict(test_acts)
    pred2 = p2.predict(test_acts)

    print(f"\nMultiMax preds (run 1): {pred1[:5]}")
    print(f"MultiMax preds (run 2): {pred2[:5]}")
    assert torch.equal(pred1, pred2), f"Max diff: {(pred1 - pred2).abs().max().item()}"


def test_gated_bipolar_seed_reproducibility():
    train_acts, train_labels, test_acts, test_labels = _get_data()

    p1 = pl.probes.GatedBipolar(
        n_epochs=10, seed=PROBE_SEED, device="cpu",
    ).fit(train_acts, train_labels)
    p2 = pl.probes.GatedBipolar(
        n_epochs=10, seed=PROBE_SEED, device="cpu",
    ).fit(train_acts, train_labels)

    pred1 = p1.predict(test_acts)
    pred2 = p2.predict(test_acts)

    print(f"\nGatedBipolar preds (run 1): {pred1[:5]}")
    print(f"GatedBipolar preds (run 2): {pred2[:5]}")
    assert torch.equal(pred1, pred2), f"Max diff: {(pred1 - pred2).abs().max().item()}"

import gc
import sys
import time

import torch
from vllm.activations import ActivationEngine

import probelab as pl

# Force unbuffered output
print = lambda *a, **kw: (sys.stdout.write(" ".join(str(x) for x in a) + "\n"), sys.stdout.flush())

MODEL = "google/gemma-3-27b-it"
LAYER = 40
BATCH_SIZE = 32

print("Loading datasets...")
t0 = time.time()
full_train = pl.datasets.load("wildguard_mix")
train_dataset, val_dataset = full_train.split(0.8)
test_dataset = pl.datasets.load("wildguard_mix", split="test")
print(f"Datasets loaded in {time.time() - t0:.1f}s")
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Load model first, tokenizer comes from ActivationEngine
print("\nLoading model (vLLM ActivationEngine)...")
t0 = time.time()
try:
    model = ActivationEngine(
        model=MODEL,
        layers=[LAYER],
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        prefill_cudagraph=True,
        max_model_len=4096,
        max_num_batched_tokens=65536,
        prefill_only=True,
        staged_export=True,
        static_shape_bucketing=True,
        activation_only=True,
        tp_rank0_only=True,
        flat_output=True,
        activation_export_device="cuda",
    )
except Exception as exc:
    msg = str(exc)
    if "set_activation_capture_layers" in msg:
        raise RuntimeError(
            "This model is not activation-capture compatible in the current vLLM build. "
            "ActivationEngine currently supports capture for Llama-family models in this repo state."
        ) from exc
    raise
print(f"Model loaded in {time.time() - t0:.1f}s")

# Tokenize (no truncation, masks.all) using engine tokenizer
mask = pl.masks.all()
tokenizer = model.tokenizer

print("\nTokenizing train...")
t0 = time.time()
train_tokens = pl.tokenize_dataset(train_dataset, tokenizer, mask=mask)
print(f"Train tokenized in {time.time() - t0:.1f}s: {len(train_tokens)} samples, {train_tokens.total_tokens:,} tokens, max_seq={train_tokens.seq_len}")

print("Tokenizing test...")
t0 = time.time()
test_tokens = pl.tokenize_dataset(test_dataset, tokenizer, mask=mask)
print(f"Test tokenized in {time.time() - t0:.1f}s: {len(test_tokens)} samples, {test_tokens.total_tokens:,} tokens, max_seq={test_tokens.seq_len}")

# Collect activations with inline mean pooling (same as original)
print("\nCollecting + pooling train activations...")
t0 = time.time()
train_prepared = pl.collect_activations(
    model,
    train_tokens,
    layers=[LAYER],
    batch_size=BATCH_SIZE,
    pool="mean",
    progress=True,
    progress_desc="train collect+pool",
)
elapsed = time.time() - t0
print(f"Train done in {elapsed:.0f}s ({len(train_tokens)/elapsed:.1f} samples/s): {train_prepared.data.shape}")

print("Collecting + pooling test activations...")
t0 = time.time()
test_prepared = pl.collect_activations(
    model,
    test_tokens,
    layers=[LAYER],
    batch_size=BATCH_SIZE,
    pool="mean",
    progress=True,
    progress_desc="test collect+pool",
)
elapsed = time.time() - t0
print(f"Test done in {elapsed:.0f}s: {test_prepared.data.shape}")

# Free the model
print("Freeing model...")
model.close()
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Train logistic probe (C=0.01, on GPU since model is freed)
print("\nTraining logistic probe (C=0.01)...")
t0 = time.time()
logistic = pl.probes.Logistic(C=0.01, device="cuda").fit(train_prepared, train_dataset.labels)
print(f"Logistic trained in {time.time() - t0:.1f}s")

# Evaluate
print("\nResults:")
train_probs = logistic.predict(train_prepared)
train_auroc = pl.metrics.auroc(train_dataset.labels, train_probs)
train_f1 = pl.metrics.f1(train_dataset.labels, train_probs)
test_probs = logistic.predict(test_prepared)
test_f1 = pl.metrics.f1(test_dataset.labels, test_probs)
test_auroc = pl.metrics.auroc(test_dataset.labels, test_probs)
print(f"  logistic: Train F1={train_f1:.3f}, Train AUROC={train_auroc:.3f}, Test F1={test_f1:.3f}, Test AUROC={test_auroc:.3f}")

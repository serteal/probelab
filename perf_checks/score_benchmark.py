import gc
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl

# Force unbuffered output
print = lambda *a, **kw: (sys.stdout.write(" ".join(str(x) for x in a) + "\n"), sys.stdout.flush())

print("Loading datasets...")
t0 = time.time()
full_train = pl.datasets.load("wildguard_mix")
train_dataset, val_dataset = full_train.split(0.8)
test_dataset = pl.datasets.load("wildguard_mix", split="test")
print(f"Datasets loaded in {time.time() - t0:.1f}s")
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Tokenize (no truncation, masks.all)
mask = pl.masks.all()
layer = 40
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

print("\nTokenizing train...")
t0 = time.time()
train_tokens = pl.tokenize_dataset(train_dataset, tokenizer, mask=mask)
print(f"Train tokenized in {time.time() - t0:.1f}s: {len(train_tokens)} samples, {train_tokens.total_tokens:,} tokens, max_seq={train_tokens.seq_len}")

print("Tokenizing test...")
t0 = time.time()
test_tokens = pl.tokenize_dataset(test_dataset, tokenizer, mask=mask)
print(f"Test tokenized in {time.time() - t0:.1f}s: {len(test_tokens)} samples, {test_tokens.total_tokens:,} tokens, max_seq={test_tokens.seq_len}")

# Load model
print("\nLoading model...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-27b-it", dtype=torch.bfloat16, device_map="auto"
).eval()
for param in model.parameters():
    param.requires_grad = False
print(f"Model loaded in {time.time() - t0:.1f}s")

# Collect activations with inline mean pooling (avoids storing all tokens)
print("\nCollecting + pooling train activations...")
t0 = time.time()
train_prepared = pl.collect_activations(
    model,
    train_tokens,
    layers=[layer],
    batch_size=4,
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
    layers=[layer],
    batch_size=4,
    pool="mean",
    progress=True,
    progress_desc="test collect+pool",
)
elapsed = time.time() - t0
print(f"Test done in {elapsed:.0f}s: {test_prepared.data.shape}")

# Free the model
print("Freeing model...")
del model; gc.collect(); torch.cuda.empty_cache()

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

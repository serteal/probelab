"""Basic probe training: end-to-end workflow.

This example demonstrates the core probelab workflow:
1. Load a dataset from the registry
2. Tokenize dialogues with a mask
3. Collect activations from a specific layer
4. Train a Logistic probe
5. Evaluate with AUROC
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LAYER = 12  # Middle layer of 16-layer model

# Load model and tokenizer
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load and split dataset
dataset = pl.datasets.load("circuit_breakers")
train_ds, test_ds = dataset.split(0.8)

print(f"Train: {train_ds}")
print(f"Test: {test_ds}")

# Tokenize with assistant mask (only extract assistant tokens)
train_tokens = pl.processing.tokenize_dataset(train_ds, tokenizer, mask=pl.masks.assistant())
test_tokens = pl.processing.tokenize_dataset(test_ds, tokenizer, mask=pl.masks.assistant())

print(f"Token shapes: train={train_tokens.shape}, test={test_tokens.shape}")

# Collect activations from target layer
print(f"\nCollecting activations from layer {LAYER}...")
train_acts = pl.processing.collect_activations(model, train_tokens, layers=[LAYER], batch_size=8)
test_acts = pl.processing.collect_activations(model, test_tokens, layers=[LAYER], batch_size=8)

# Pool sequence dimension (single layer already has no LAYER axis)
train_prepared = train_acts.pool("sequence", "mean")
test_prepared = test_acts.pool("sequence", "mean")

print(f"Prepared shapes: train={train_prepared.shape}, test={test_prepared.shape}")

# Train probe and evaluate
print("\nTraining Logistic probe...")
probe = pl.probes.Logistic(C=1.0).fit(train_prepared, train_ds.labels)
scores = probe.predict(test_prepared)
auroc = pl.metrics.auroc(test_ds.labels, scores)
accuracy = pl.metrics.accuracy(test_ds.labels, scores)
recall_at_5 = pl.metrics.recall_at_fpr(test_ds.labels, scores, fpr=0.05)

print(f"\nResults:")
print(f"  AUROC: {auroc:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Recall@5%FPR: {recall_at_5:.4f}")

# Save probe for later use
probe.save("probe_layer12.pt")
print("\nProbe saved to probe_layer12.pt")

# Load and verify
loaded_probe = pl.probes.Logistic.load("probe_layer12.pt")
loaded_scores = loaded_probe.predict(test_prepared)
assert torch.allclose(scores, loaded_scores)
print("Probe loaded and verified!")

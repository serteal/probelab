"""Find the optimal layer for probing across all model layers.

This example sweeps through all 16 layers of Llama-3.2-1B-Instruct
to identify which layer contains the most useful representations
for the classification task.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
N_SAMPLES = 300

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Get number of layers from model config
num_layers = model.config.num_hidden_layers
print(f"Model has {num_layers} layers")

# Load dataset
dataset = pl.datasets.load("repe").sample(N_SAMPLES, stratified=True)
train_ds, test_ds = dataset.split(0.8, stratified=True)

print(f"Train: {train_ds}")
print(f"Test: {test_ds}")

# Tokenize
train_tokens = pl.processing.tokenize_dataset(train_ds, tokenizer, mask=pl.masks.assistant())
test_tokens = pl.processing.tokenize_dataset(test_ds, tokenizer, mask=pl.masks.assistant())

# Collect activations from ALL layers at once
# This is more efficient than collecting one layer at a time
all_layers = list(range(num_layers))
print(f"\nCollecting activations from all {num_layers} layers...")

train_acts = pl.processing.collect_activations(model, train_tokens, layers=all_layers, batch_size=8)
test_acts = pl.processing.collect_activations(model, test_tokens, layers=all_layers, batch_size=8)

print(f"Full activations shape: {train_acts.shape}")

# Sweep through layers
results = {}
print("\n" + "=" * 50)
print(f"{'Layer':<8} {'AUROC':<12} {'Recall@5%':<12}")
print("=" * 50)

for layer in all_layers:
    # Select single layer and pool
    train_prepared = train_acts.select(layer=layer).mean_pool()
    test_prepared = test_acts.select(layer=layer).mean_pool()

    # Train probe and evaluate
    probs = pl.probes.Logistic(C=1.0).fit(train_prepared, train_ds.labels).predict(test_prepared)
    auroc = pl.metrics.auroc(test_ds.labels, probs)
    recall = pl.metrics.recall_at_fpr(test_ds.labels, probs, fpr=0.05)

    results[layer] = {"auroc": auroc, "recall@5%": recall}
    print(f"{layer:<8} {auroc:<12.4f} {recall:<12.4f}")

# Find best layer
best_layer = max(results, key=lambda k: results[k]["auroc"])
print("\n" + "=" * 50)
print(f"Best layer: {best_layer} (AUROC: {results[best_layer]['auroc']:.4f})")

# Visual representation of layer performance
print("\nLayer Performance (AUROC):")
print("-" * 60)
for layer in all_layers:
    bar_length = int(results[layer]["auroc"] * 40)
    bar = "#" * bar_length
    marker = " <-- BEST" if layer == best_layer else ""
    print(f"Layer {layer:2d}: {bar} {results[layer]['auroc']:.3f}{marker}")

# Additional analysis: early vs middle vs late layers
early = [results[l]["auroc"] for l in range(0, num_layers // 3)]
middle = [results[l]["auroc"] for l in range(num_layers // 3, 2 * num_layers // 3)]
late = [results[l]["auroc"] for l in range(2 * num_layers // 3, num_layers)]

print(f"\nLayer group averages:")
print(f"  Early  (0-{num_layers//3 - 1}):  {sum(early)/len(early):.4f}")
print(f"  Middle ({num_layers//3}-{2*num_layers//3 - 1}): {sum(middle)/len(middle):.4f}")
print(f"  Late   ({2*num_layers//3}-{num_layers-1}): {sum(late)/len(late):.4f}")

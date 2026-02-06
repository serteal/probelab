"""Combine multiple datasets for training a general-purpose probe.

This example shows how to:
1. Load datasets from different categories
2. Combine them using the + operator
3. Handle potential class imbalance
4. Train on diverse data for better generalization
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LAYER = 12
N_PER_DATASET = 100

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# List available categories
print("\nAvailable categories:")
for cat in pl.datasets.list_categories():
    datasets_in_cat = pl.datasets.list_datasets(category=cat)
    print(f"  {cat}: {len(datasets_in_cat)} datasets")

# Load datasets from multiple categories
print("\nLoading datasets...")
deception_ds = pl.datasets.load("truthful_qa")
harmfulness_ds = pl.datasets.load("circuit_breakers")
roleplay_ds = pl.datasets.load("roleplaying")

print(f"  Deception (truthful_qa): {deception_ds}")
print(f"  Harmfulness (circuit_breakers): {harmfulness_ds}")
print(f"  Roleplay (roleplaying): {roleplay_ds}")

# Sample to balance dataset sizes
deception_sample = deception_ds.sample(N_PER_DATASET, stratified=True)
harmfulness_sample = harmfulness_ds.sample(N_PER_DATASET, stratified=True)
roleplay_sample = roleplay_ds.sample(N_PER_DATASET, stratified=True)

# Combine using + operator
combined = deception_sample + harmfulness_sample + roleplay_sample
print(f"\nCombined dataset: {combined}")

# Shuffle and split
train_ds, test_ds = combined.shuffle().split(0.8, stratified=True)
print(f"Train: {train_ds}")
print(f"Test: {test_ds}")

# Tokenize with last assistant message mask
mask = pl.masks.assistant() & pl.masks.nth_message(-1)
train_tokens = pl.tokenize_dataset(train_ds, tokenizer, mask=mask)
test_tokens = pl.tokenize_dataset(test_ds, tokenizer, mask=mask)

# Collect activations (single layer returns no LAYER axis)
print(f"\nCollecting activations from layer {LAYER}...")
train_acts = pl.collect_activations(model, train_tokens, layers=[LAYER])
test_acts = pl.collect_activations(model, test_tokens, layers=[LAYER])

# Prepare
train_prepared = train_acts.mean_pool()
test_prepared = test_acts.mean_pool()

# Train probe
probe = pl.probes.Logistic().fit(train_prepared, train_ds.labels)

# Evaluate on combined test set
probs = probe.predict(test_prepared)
print(f"\nOverall performance on combined test set:")
print(f"  AUROC: {pl.metrics.auroc(test_ds.labels, probs):.4f}")
print(f"  Accuracy: {pl.metrics.accuracy(test_ds.labels, probs):.4f}")

# Evaluate per-dataset on fresh samples
print("\nPer-dataset evaluation (fresh samples):")

datasets_to_eval = [
    ("truthful_qa", deception_ds),
    ("circuit_breakers", harmfulness_ds),
    ("roleplaying", roleplay_ds),
]

for name, ds in datasets_to_eval:
    # Get fresh samples not in training
    test_sample = ds.sample(30, stratified=True, seed=123)
    tokens = pl.tokenize_dataset(test_sample, tokenizer, mask=mask)
    acts = pl.collect_activations(model, tokens, layers=[LAYER])
    prepared = acts.mean_pool()
    probs = probe.predict(prepared)
    auroc = pl.metrics.auroc(test_sample.labels, probs)
    print(f"  {name}: AUROC = {auroc:.4f}")

# Save the combined probe
probe.save("probe_combined.pt")
print("\nProbe saved to probe_combined.pt")

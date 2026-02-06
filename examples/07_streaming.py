"""Memory-efficient streaming for large datasets.

For datasets too large to fit in memory, use stream_activations()
to process in batches. This example demonstrates:
1. Streaming activation collection
2. Incremental feature accumulation
3. Training on accumulated features
4. Comparison with built-in pooling during collection
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LAYER = 12
BATCH_SIZE = 16
N_SAMPLES = 400  # Simulating a large dataset

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset (simulating a large dataset)
dataset = pl.datasets.load("ultrachat").sample(N_SAMPLES, stratified=True)
train_ds, test_ds = dataset.split(0.8, stratified=True)

print(f"Train samples: {len(train_ds)}")
print(f"Test samples: {len(test_ds)}")

# Tokenize
train_tokens = pl.tokenize_dataset(train_ds, tokenizer, mask=pl.masks.assistant())
test_tokens = pl.tokenize_dataset(test_ds, tokenizer, mask=pl.masks.assistant())

d_model = model.config.hidden_size
print(f"Hidden dimension: {d_model}")

# -------------------------------------------------------------------
# Method 1: Streaming with manual accumulation
# -------------------------------------------------------------------
print("\n--- Method 1: Manual streaming ---")

# Pre-allocate arrays for mean-pooled features
train_features = torch.zeros(len(train_ds), d_model)
train_indices_seen = set()

# Stream through training data
for batch_acts, indices, seq_len in pl.processing.stream_activations(
    model, train_tokens, layers=[LAYER], batch_size=BATCH_SIZE
):
    # batch_acts: [1, batch, seq, hidden] (1 layer)
    acts = batch_acts.squeeze(0)  # [batch, seq, hidden]

    # Get detection mask for this batch
    if train_tokens.padding_side == "right":
        mask = train_tokens.detection_mask[indices, :seq_len]
    else:
        mask = train_tokens.detection_mask[indices, -seq_len:]

    # Mean pool over sequence (manual for demonstration)
    mask_expanded = mask.unsqueeze(-1).float()
    pooled = (acts * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

    # Store in pre-allocated tensor
    for i, idx in enumerate(indices):
        train_features[idx] = pooled[i].cpu()
        train_indices_seen.add(idx)

    print(f"  Processed batch: {len(indices)} samples, total: {len(train_indices_seen)}/{len(train_ds)}")

print(f"Accumulated {len(train_indices_seen)} training samples")

# Train probe on streamed features using Activations.from_tensor()
# 2D tensors [batch, hidden] are now supported directly
train_acts_streamed = pl.Activations.from_tensor(train_features)
print(f"Streamed activations: {train_acts_streamed.axes}, shape={train_acts_streamed.shape}")

# Train probe
probe_streamed = pl.probes.Logistic()
probe_streamed.fit(train_acts_streamed, train_ds.labels)

# Can also use probe directly with raw tensor (differentiable)
# logits = probe_streamed(train_features)  # Works for gradient-based training

# -------------------------------------------------------------------
# Method 2: Using collect_activations with pool parameter
# -------------------------------------------------------------------
print("\n--- Method 2: Built-in pooling during collection ---")

# collect_activations can pool on-the-fly, reducing memory
train_acts_pooled = pl.collect_activations(
    model,
    train_tokens,
    layers=[LAYER],
    batch_size=BATCH_SIZE,
    pool="mean",  # Pool during collection
)
test_acts_pooled = pl.collect_activations(
    model,
    test_tokens,
    layers=[LAYER],
    batch_size=BATCH_SIZE,
    pool="mean",
)

print(f"Pooled activations shape: {train_acts_pooled.shape}")

# Train with probelab probe (single layer already has no LAYER axis)
probe = pl.probes.Logistic().fit(train_acts_pooled, train_ds.labels)

# Evaluate
scores = probe.predict(test_acts_pooled)
auroc = pl.metrics.auroc(test_ds.labels, scores)
print(f"\nAUROC: {auroc:.4f}")

# -------------------------------------------------------------------
# Memory comparison
# -------------------------------------------------------------------
print("\n--- Memory comparison ---")
seq_len = tokenizer.model_max_length
print(f"Assumed max sequence length: {seq_len}")

full_mem = len(train_ds) * seq_len * d_model * 4 / 1e9  # float32
pooled_mem = len(train_ds) * d_model * 4 / 1e6  # float32
batch_mem = BATCH_SIZE * seq_len * d_model * 4 / 1e6  # float32

print(f"\nFull activations (all in memory):")
print(f"  {len(train_ds)} x {seq_len} x {d_model} x 4 bytes = {full_mem:.2f} GB")

print(f"\nPooled activations (reduced):")
print(f"  {len(train_ds)} x {d_model} x 4 bytes = {pooled_mem:.2f} MB")

print(f"\nStreaming batch (constant memory):")
print(f"  {BATCH_SIZE} x {seq_len} x {d_model} x 4 bytes = {batch_mem:.2f} MB")

print(f"\nMemory savings:")
print(f"  Pooling: {full_mem * 1000 / pooled_mem:.0f}x reduction")
print(f"  Streaming: Only uses batch memory regardless of dataset size")

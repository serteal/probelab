"""Token-level probing with score aggregation.

Instead of pooling activations before training, this example trains
on individual tokens and aggregates predictions afterward. Useful
for understanding when harmful content appears in responses.

Score aggregation methods (via pl.pool):
- pl.pool.mean(x, mask): Simple average over tokens
- pl.pool.max(x, mask): Maximum score (any token triggers)
- pl.pool.last_token(x, mask): Last token only
- pl.pool.ema(x, mask, alpha=0.5): Exponential moving average, then max
- pl.pool.rolling(x, mask, window_size=10): Rolling window mean, then max
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LAYER = 11
N_SAMPLES = 200

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset
dataset = pl.datasets.load("ai_liar").sample(N_SAMPLES, stratified=True)
train_ds, test_ds = dataset.split(0.8, stratified=True)

print(f"Train: {train_ds}")
print(f"Test: {test_ds}")

# Tokenize (using all assistant tokens)
train_tokens = pl.tokenize_dataset(train_ds, tokenizer, mask=pl.masks.assistant())
test_tokens = pl.tokenize_dataset(test_ds, tokenizer, mask=pl.masks.assistant())

# Collect activations (single layer returns no LAYER axis, keeps SEQ)
print(f"\nCollecting activations from layer {LAYER}...")
train_acts = pl.collect_activations(model, train_tokens, layers=[LAYER])
test_acts = pl.collect_activations(model, test_tokens, layers=[LAYER])

print(f"Token-level activations shape: {train_acts.shape}")

# Train probe on tokens (Logistic handles this automatically)
# Each token gets the sample's label during training
probe = pl.probes.Logistic().fit(train_acts, train_ds.labels)

print(f"Probe: {probe}")

# Predict on test tokens - returns [batch, seq] directly!
token_probs = probe.predict(test_acts)
print(f"Token-level probs shape: {token_probs.shape}")

# Get mask from activations
mask = test_acts.detection_mask.bool()

print("\n" + "=" * 55)
print(f"{'Aggregation':<15} {'AUROC':<12} {'Recall@5%':<12}")
print("=" * 55)

results = {}

# Mean pooling
pooled = pl.pool.mean(token_probs, mask)
auroc = pl.metrics.auroc(test_ds.labels, pooled)
recall = pl.metrics.recall_at_fpr(test_ds.labels, pooled, fpr=0.05)
results["mean"] = {"auroc": auroc, "recall@5%": recall}
print(f"{'mean':<15} {auroc:<12.4f} {recall:<12.4f}")

# Max pooling
pooled = pl.pool.max(token_probs, mask)
auroc = pl.metrics.auroc(test_ds.labels, pooled)
recall = pl.metrics.recall_at_fpr(test_ds.labels, pooled, fpr=0.05)
results["max"] = {"auroc": auroc, "recall@5%": recall}
print(f"{'max':<15} {auroc:<12.4f} {recall:<12.4f}")

# Last token
pooled = pl.pool.last_token(token_probs, mask)
auroc = pl.metrics.auroc(test_ds.labels, pooled)
recall = pl.metrics.recall_at_fpr(test_ds.labels, pooled, fpr=0.05)
results["last_token"] = {"auroc": auroc, "recall@5%": recall}
print(f"{'last_token':<15} {auroc:<12.4f} {recall:<12.4f}")

# EMA with different alphas
for alpha in [0.3, 0.5, 0.7]:
    pooled = pl.pool.ema(token_probs, mask, alpha=alpha)
    auroc = pl.metrics.auroc(test_ds.labels, pooled)
    recall = pl.metrics.recall_at_fpr(test_ds.labels, pooled, fpr=0.05)
    name = f"ema_{alpha}"
    results[name] = {"auroc": auroc, "recall@5%": recall}
    print(f"{name:<15} {auroc:<12.4f} {recall:<12.4f}")

# Rolling window with different sizes
for window in [5, 10, 20]:
    pooled = pl.pool.rolling(token_probs, mask, window_size=window)
    auroc = pl.metrics.auroc(test_ds.labels, pooled)
    recall = pl.metrics.recall_at_fpr(test_ds.labels, pooled, fpr=0.05)
    name = f"rolling_{window}"
    results[name] = {"auroc": auroc, "recall@5%": recall}
    print(f"{name:<15} {auroc:<12.4f} {recall:<12.4f}")

# Find best method
best = max(results, key=lambda k: results[k]["auroc"])
print("\n" + "=" * 55)
print(f"Best aggregation: {best} (AUROC: {results[best]['auroc']:.4f})")

# Compare with sequence-level baseline (pool before training)
print("\n--- Baseline: Pool before training ---")
train_pooled = train_acts.mean_pool()
test_pooled = test_acts.mean_pool()

baseline_probs = pl.probes.Logistic().fit(train_pooled, train_ds.labels).predict(test_pooled)
baseline_auroc = pl.metrics.auroc(test_ds.labels, baseline_probs)
print(f"Baseline AUROC (pool before training): {baseline_auroc:.4f}")

# Compare
print(f"\nToken-level with {best}: {results[best]['auroc']:.4f}")
print(f"Sequence-level baseline: {baseline_auroc:.4f}")
diff = results[best]["auroc"] - baseline_auroc
print(f"Difference: {diff:+.4f}")

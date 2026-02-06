"""Compare all 5 probe architectures on the same dataset.

Probes:
- Logistic: Simple L2-regularized logistic regression (fastest)
- MLP: Multi-layer perceptron with dropout
- Attention: Learned attention over tokens (requires SEQ axis)
- MultiMax: Multi-head hard max pooling (requires SEQ axis)
- GatedBipolar: AlphaEvolve gated bipolar probe (requires SEQ axis)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LAYER = 10
N_SAMPLES = 200

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset (sample for faster iteration)
dataset = pl.datasets.load("truthful_qa").sample(N_SAMPLES, stratified=True)
train_ds, test_ds = dataset.split(0.8, stratified=True)

print(f"Train: {train_ds}")
print(f"Test: {test_ds}")

# Tokenize
train_tokens = pl.processing.tokenize_dataset(train_ds, tokenizer, mask=pl.masks.assistant())
test_tokens = pl.processing.tokenize_dataset(test_ds, tokenizer, mask=pl.masks.assistant())

# Collect activations (keep sequence dimension for attention-based probes)
print(f"\nCollecting activations from layer {LAYER}...")
train_acts = pl.processing.collect_activations(model, train_tokens, layers=[LAYER])
test_acts = pl.processing.collect_activations(model, test_tokens, layers=[LAYER])

# Select layer (removes LAYER axis, keeps SEQ)
train_seq = train_acts.select(layer=LAYER)
test_seq = test_acts.select(layer=LAYER)

# Pool for probes that don't need sequence dimension
train_pooled = train_seq.pool("sequence", "mean")
test_pooled = test_seq.pool("sequence", "mean")

print("Activation shapes:")
print(f"  With SEQ axis: {train_seq.shape}")
print(f"  Pooled (no SEQ): {train_pooled.shape}")

# Define probes to compare
probes = {
    # These work on pooled (no SEQ axis)
    "Logistic": pl.probes.Logistic(C=1.0),
    "MLP": pl.probes.MLP(hidden_dim=64, n_epochs=50),
    # These REQUIRE SEQ axis (learn over tokens)
    "Attention": pl.probes.Attention(hidden_dim=64, n_epochs=200),
    "MultiMax": pl.probes.MultiMax(n_heads=5, n_epochs=20),
    "GatedBipolar": pl.probes.GatedBipolar(gate_dim=32, n_epochs=20),
}

results = {}
print("\n" + "=" * 60)
print("Training and evaluating probes...")
print("=" * 60)

for name, probe in probes.items():
    print(f"\n{name}:")

    # Select appropriate data based on probe requirements
    if name in ["Attention", "MultiMax", "GatedBipolar"]:
        # These need SEQ axis
        train_data, test_data = train_seq, test_seq
        print(f"  Using SEQ axis: {train_data.shape}")
    else:
        # These work on pooled data
        train_data, test_data = train_pooled, test_pooled
        print(f"  Using pooled: {train_data.shape}")

    # Train
    probe.fit(train_data, train_ds.labels)

    # Evaluate
    probs = probe.predict(test_data)
    auroc = pl.metrics.auroc(test_ds.labels, probs)
    accuracy = pl.metrics.accuracy(test_ds.labels, probs)
    recall_5 = pl.metrics.recall_at_fpr(test_ds.labels, probs, fpr=0.05)

    results[name] = {"auroc": auroc, "accuracy": accuracy, "recall@5%": recall_5}
    print(f"  AUROC: {auroc:.4f}, Accuracy: {accuracy:.4f}, Recall@5%: {recall_5:.4f}")

# Summary table
print("\n" + "=" * 60)
print("Summary (sorted by AUROC)")
print("=" * 60)
print(f"{'Probe':<15} {'AUROC':<10} {'Accuracy':<10} {'Recall@5%':<10}")
print("-" * 45)
for name, metrics in sorted(results.items(), key=lambda x: -x[1]["auroc"]):
    print(f"{name:<15} {metrics['auroc']:<10.4f} {metrics['accuracy']:<10.4f} {metrics['recall@5%']:<10.4f}")

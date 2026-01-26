import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl
from probelab.visualization import print_metrics

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-27b-it", torch_dtype=torch.bfloat16, device_map="auto"
).eval()
for param in model.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

dataset = pl.datasets.WildGuardMixDataset(split="train")[:1000]
train_dataset, val_dataset = dataset.split(0.8)
test_dataset = pl.datasets.WildGuardMixDataset(split="test")[:1000]

print(f"Train dataset: {len(train_dataset)}; Test dataset: {len(test_dataset)}")

pipelines = {
    # Original probes with mean pooling
    "logistic": pl.Pipeline(
        [
            ("pool", pl.preprocessing.Pool(dim="sequence", method="mean")),
            ("probe", pl.probes.Logistic(C=0.01)),
        ]
    ),
    "mlp": pl.Pipeline(
        [
            ("pool", pl.preprocessing.Pool(dim="sequence", method="mean")),
            ("probe", pl.probes.MLP()),
        ]
    ),
    # Logistic probe with EMA aggregation (GDM paper technique)
    # Trains on token-level predictions, aggregates with EMA + max at inference
    "logistic_ema": pl.Pipeline(
        [
            ("probe", pl.probes.Logistic(C=0.01)),  # Token-level
            ("ema", pl.preprocessing.EMAPool(alpha=0.5)),  # EMA aggregation
        ]
    ),
    # Logistic probe with Rolling window aggregation
    "logistic_rolling": pl.Pipeline(
        [
            ("probe", pl.probes.Logistic(C=0.01)),  # Token-level
            ("rolling", pl.preprocessing.RollingPool(window_size=10)),  # Rolling aggregation
        ]
    ),
    # MultiMax probe - multi-head hard max pooling (GDM paper)
    "multimax": pl.Pipeline(
        [
            ("probe", pl.probes.MultiMax(n_heads=10, mlp_hidden_dim=128, verbose=True)),
        ]
    ),
    # GatedBipolar probe - AlphaEvolve architecture (GDM paper)
    "gated_bipolar": pl.Pipeline(
        [
            ("probe", pl.probes.GatedBipolar(mlp_hidden_dim=128, gate_dim=64, verbose=True)),
        ]
    ),
}

# Collect activations once (shared across all pipelines)
print("Collecting training activations...", flush=True)
t0 = time.time()
train_activations = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    dataset=train_dataset,
    layers=[40],
    batch_size=8,
    mask=pl.masks.user(),
)
print(f"Training activations collected in {time.time() - t0:.1f}s", flush=True)

print("Collecting test activations...", flush=True)
t0 = time.time()
test_activations = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    dataset=test_dataset,
    layers=[40],
    batch_size=8,
    mask=pl.masks.user(),
)
print(f"Test activations collected in {time.time() - t0:.1f}s", flush=True)

# Train each pipeline separately with timing
print("\n" + "=" * 60, flush=True)
print("TRAINING PIPELINES", flush=True)
print("=" * 60, flush=True)

training_times = {}
for name, pipeline in pipelines.items():
    print(f"\nTraining {name}...", flush=True)
    t0 = time.time()
    pipeline.fit(train_activations, train_dataset.labels)
    elapsed = time.time() - t0
    training_times[name] = elapsed
    print(f"  {name} trained in {elapsed:.1f}s", flush=True)

# Print training time summary
print("\n" + "=" * 60, flush=True)
print("TRAINING TIME SUMMARY", flush=True)
print("=" * 60, flush=True)
for name, elapsed in sorted(training_times.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {elapsed:.1f}s", flush=True)

# Evaluate all pipelines
print("\n" + "=" * 60, flush=True)
print("EVALUATING PIPELINES", flush=True)
print("=" * 60, flush=True)

predictions, metrics = pl.scripts.evaluate_pipelines(
    pipelines=pipelines,
    activations=test_activations,
    labels=test_dataset.labels,
    metrics=[
        pl.metrics.f1,
        pl.metrics.auroc,
        pl.metrics.weighted_error_rate,  # GDM paper metric
    ],
)

print_metrics(metrics)

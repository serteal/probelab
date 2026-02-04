import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-27b-it", torch_dtype=torch.bfloat16, device_map="auto"
).eval()
for param in model.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

dataset = pl.datasets.WildGuardMixDataset(split="train")
train_dataset, val_dataset = dataset.split(0.8)
test_dataset = pl.datasets.WildGuardMixDataset(split="test")

print(f"Train dataset: {len(train_dataset)}; Test dataset: {len(test_dataset)}")

pipelines = {
    "logistic": pl.Pipeline(
        [
            ("select", pl.preprocessing.SelectLayer(40)),
            ("pool", pl.preprocessing.Pool(dim="sequence", method="mean")),
            ("probe", pl.probes.Logistic(C=0.01)),
        ]
    ),
    "mlp": pl.Pipeline(
        [
            ("select", pl.preprocessing.SelectLayer(40)),
            ("pool", pl.preprocessing.Pool(dim="sequence", method="mean")),
            ("probe", pl.probes.MLP()),
        ]
    ),
}

# Collect activations
mask = pl.masks.user()

train_acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=train_dataset,
    layers=[40],
    batch_size=8,
    mask=mask,
)

# Train pipelines
for name, pipeline in pipelines.items():
    print(f"Training {name}...")
    pipeline.fit(train_acts, train_dataset.labels)

# Evaluate
test_acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=test_dataset,
    layers=[40],
    mask=mask,
    batch_size=8,
)

print("\nResults:")
for name, pipeline in pipelines.items():
    probs = pipeline.predict_proba(test_acts)
    y_pred = probs[:, 1].cpu().numpy()
    y_true = [label.value for label in test_dataset.labels]

    f1 = pl.metrics.f1(y_true, y_pred)
    auroc = pl.metrics.auroc(y_true, y_pred)
    print(f"{name}: F1={f1:.3f}, AUROC={auroc:.3f}")

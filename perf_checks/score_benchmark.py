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

# Collect activations
mask = pl.masks.user()
layer = 40

train_acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=train_dataset,
    layers=[layer],
    batch_size=8,
    mask=mask,
)

# Prepare activations
train_prepared = train_acts.select(layer=layer).pool("sequence", "mean")

# Train probes directly (no Pipeline)
probes = {
    "logistic": pl.probes.Logistic(C=0.01).fit(train_prepared, train_dataset.labels),
    "mlp": pl.probes.MLP().fit(train_prepared, train_dataset.labels),
}

# Evaluate
test_acts = pl.collect_activations(
    model=model,
    tokenizer=tokenizer,
    data=test_dataset,
    layers=[layer],
    mask=mask,
    batch_size=8,
)
test_prepared = test_acts.select(layer=layer).pool("sequence", "mean")

print("\nResults:")
for name, probe in probes.items():
    scores = probe.predict(test_prepared)
    y_pred = scores.scores[:, 1].cpu().numpy()
    y_true = [label.value for label in test_dataset.labels]

    f1 = pl.metrics.f1(y_true, y_pred)
    auroc = pl.metrics.auroc(y_true, y_pred)
    print(f"{name}: F1={f1:.3f}, AUROC={auroc:.3f}")

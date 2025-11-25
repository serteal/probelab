import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelib as pl

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
            ("pool", pl.preprocessing.Pool(axis="sequence", method="mean")),
            ("probe", pl.probes.Logistic(C=0.01)),
        ]
    ),
    "mlp": pl.Pipeline(
        [
            ("select", pl.preprocessing.SelectLayer(40)),
            ("pool", pl.preprocessing.Pool(axis="sequence", method="mean")),
            ("probe", pl.probes.MLP()),
        ]
    ),
}

pl.scripts.train_from_model(
    pipelines=pipelines,
    dataset=train_dataset,
    model=model,
    tokenizer=tokenizer,
    layers=[40],
    batch_size=8,
    mask=pl.masks.user(),
)

predictions, metrics = pl.scripts.evaluate_from_model(
    pipelines=pipelines,
    dataset=test_dataset,
    model=model,
    tokenizer=tokenizer,
    layers=[40],
    mask=pl.masks.user(),
    batch_size=8,
    metrics=[
        pl.metrics.f1,
        pl.metrics.auroc,
    ],
)

pl.visualization.print_metrics(metrics)

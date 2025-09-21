import random
from functools import partial

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelib as pl
from probelib import visualize_mask

# set random seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

dataset = pl.datasets.DolusChatDataset()[:5000]
train_dataset, test_dataset = dataset.split(0.8)
print(f"Train dataset: {len(train_dataset)}; Test dataset: {len(test_dataset)}")


print("\n" + "=" * 80)
print("Full assistant message mean probe")
print("=" * 80)

# Visualize the default mask (assistant tokens)
print("\nVisualizing mask for training (assistant tokens):")
print("-" * 40)
visualize_mask(
    train_dataset.dialogues[0],  # Show first training dialogue
    pl.masks.assistant(),  # Default mask for DolusChatDataset
    tokenizer,
    force_terminal=True,
    show_legend=False,
)
print("-" * 40)

layer = 16
probes = {
    "logistic": pl.probes.Logistic(layer=layer, score_aggregation="mean"),
    "mlp": pl.probes.MLP(layer=layer, score_aggregation="mean"),
    "attention": pl.probes.Attention(layer=layer),
}
pl.scripts.train_probes(
    probes=probes,
    model=model,
    tokenizer=tokenizer,
    data=train_dataset,
    labels=train_dataset.labels,
    batch_size=32,
    streaming=True,
)

predictions, metrics = pl.scripts.evaluate_probes(
    probes=probes,
    model=model,
    tokenizer=tokenizer,
    data=test_dataset,
    labels=test_dataset.labels,
    batch_size=32,
    metrics=[
        pl.metrics.auroc,
        partial(pl.metrics.recall_at_fpr, fpr=0.001),
        partial(pl.metrics.recall_at_fpr, fpr=0.01),
    ],
)

pl.print_metrics(metrics)


print("\n" + "=" * 80)
print("Last token probe")
print("=" * 80)

# Visualize the last token mask (only last token of assistant messages)
print("\nVisualizing mask for training (last token of assistant messages):")
print("-" * 40)
visualize_mask(
    dialogue=train_dataset.dialogues[0],  # Show first training dialogue
    mask=pl.masks.assistant() & pl.masks.last_token(),
    tokenizer=tokenizer,
    force_terminal=True,
    show_legend=False,
)
print("-" * 40)

probes = {
    "logistic": pl.probes.Logistic(layer=layer, score_aggregation="mean"),
    "mlp": pl.probes.MLP(layer=layer, score_aggregation="mean"),
    "attention": pl.probes.Attention(layer=layer),
}
pl.scripts.train_probes(
    probes=probes,
    model=model,
    tokenizer=tokenizer,
    data=train_dataset,
    labels=train_dataset.labels,
    mask=pl.masks.assistant() & pl.masks.last_token(),
    batch_size=32,
    streaming=True,
)

predictions, metrics = pl.scripts.evaluate_probes(
    probes=probes,
    model=model,
    tokenizer=tokenizer,
    data=test_dataset,
    labels=test_dataset.labels,
    batch_size=32,
    metrics=[
        pl.metrics.auroc,
        partial(pl.metrics.recall_at_fpr, fpr=0.001),
        partial(pl.metrics.recall_at_fpr, fpr=0.01),
    ],
)
pl.print_metrics(metrics)

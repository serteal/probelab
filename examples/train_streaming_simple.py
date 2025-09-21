import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelib as pl

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# Load dataset
harmful = pl.datasets.CircuitBreakersDataset()
benign = pl.datasets.BenignInstructionsDataset()
dataset = harmful + benign
train_dataset, test_dataset = dataset.split(0.8)
print(
    f"Train dataset size: {len(train_dataset)}; Test dataset size: {len(test_dataset)}"
)

layer = 16
train_act_iter = pl.collect_activations(
    data=train_dataset,
    model=model,
    tokenizer=tokenizer,
    layers=[layer],
    mask=pl.masks.assistant(),
    streaming=True,
    batch_size=32,
)
test_act_iter = pl.collect_activations(
    data=test_dataset,
    model=model,
    tokenizer=tokenizer,
    layers=[layer],
    mask=pl.masks.assistant(),
    streaming=True,
    batch_size=32,
)

probes = {
    "logistic": pl.probes.Logistic(layer=layer, score_aggregation="mean"),
    "mlp": pl.probes.MLP(layer=layer, score_aggregation="mean"),
    "attention": pl.probes.Attention(layer=layer),
}
labels_tensor = torch.tensor([label.value for label in train_dataset.labels])
for batch_activations in tqdm(train_act_iter, desc="Training probes"):
    for probe in probes.values():
        probe.partial_fit(
            batch_activations, labels_tensor[batch_activations.batch_indices]
        )

predictions = {name: torch.zeros(len(test_dataset)) for name in probes}
for batch_activations in tqdm(test_act_iter, desc="Evaluating probes"):
    for name, probe in probes.items():
        probs = probe.predict_proba(batch_activations)
        predictions[name][batch_activations.batch_indices] = probs[:, 1].cpu()

for name, predictions in predictions.items():
    print(f"{name}: {pl.metrics.auroc(test_dataset.labels, predictions.numpy())}")

"""LoRA fine-tuning using probe scores as training signal.

This example demonstrates how to use a trained probe to guide
LoRA fine-tuning of the base model. The probe provides a
differentiable score that can be used as a loss signal.

Requirements: pip install peft
"""

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelab as pl
from probelab.types import Label

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LAYER = 12
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
N_SAMPLES = 200

# Load model and tokenizer
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -------------------------------------------------------------------
# Step 1: Train a probe on harmful/harmless activations
# -------------------------------------------------------------------
print("\n--- Step 1: Training probe ---")

dataset = pl.datasets.load("circuit_breakers").sample(N_SAMPLES, stratified=True)
train_ds, test_ds = dataset.split(0.8, stratified=True)

print(f"Train: {train_ds}")
print(f"Test: {test_ds}")

train_tokens = pl.tokenize_dataset(train_ds, tokenizer, mask=pl.masks.assistant())
test_tokens = pl.tokenize_dataset(test_ds, tokenizer, mask=pl.masks.assistant())

train_acts = pl.collect_activations(model, train_tokens, layers=[LAYER])
test_acts = pl.collect_activations(model, test_tokens, layers=[LAYER])

train_prepared = train_acts.mean_pool()
test_prepared = test_acts.mean_pool()

probe = pl.probes.Logistic().fit(train_prepared, train_ds.labels)

probs = probe.predict(test_prepared)
pre_auroc = pl.metrics.auroc(test_ds.labels, probs)
print(f"Pre-training probe AUROC: {pre_auroc:.4f}")

# -------------------------------------------------------------------
# Step 2: Configure LoRA
# -------------------------------------------------------------------
print("\n--- Step 2: Configuring LoRA ---")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# -------------------------------------------------------------------
# Step 3: Create custom loss using probe
# -------------------------------------------------------------------
print("\n--- Step 3: Setting up training ---")


def probe_loss(hidden_states, target_labels, detection_mask):
    """Compute loss using probe as scoring function.

    Args:
        hidden_states: [batch, seq, hidden] from the target layer
        target_labels: [batch] - 0 for negative (safe), 1 for positive (harmful)
        detection_mask: [batch, seq] - which tokens to consider

    Returns:
        Loss that encourages model to produce lower scores for harmful content
    """
    # Mean pool over detected tokens
    mask = detection_mask.unsqueeze(-1).float()
    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # Differentiable probe forward pass
    logits = probe(pooled)

    # Loss: minimize probability of harmful (positive) class
    # For harmful samples (label=1), we want low scores (refusal)
    # For safe samples (label=0), we want unchanged scores
    probs = torch.sigmoid(logits)
    target = target_labels.float().to(probs.device)

    # For harmful (target=1): loss = probs (minimize)
    # For safe (target=0): loss = 0 (don't change)
    loss = (probs * target).mean()

    return loss


# -------------------------------------------------------------------
# Step 4: Fine-tuning loop
# -------------------------------------------------------------------
print("\n--- Step 4: Fine-tuning ---")

optimizer = AdamW(peft_model.parameters(), lr=LEARNING_RATE)

# Only fine-tune on harmful samples to reduce their harm score
harmful_indices = [i for i, label in enumerate(train_ds.labels) if label == Label.POSITIVE]
print(f"Training on {len(harmful_indices)} harmful samples")

for epoch in range(NUM_EPOCHS):
    peft_model.train()
    epoch_loss = 0.0
    n_batches = 0

    # Simple batching
    for i in range(0, len(harmful_indices), BATCH_SIZE):
        batch_indices = harmful_indices[i : i + BATCH_SIZE]

        # Get input tensors
        input_ids = train_tokens.input_ids[batch_indices].to(peft_model.device)
        attention_mask = train_tokens.attention_mask[batch_indices].to(peft_model.device)
        detection_mask = train_tokens.detection_mask[batch_indices].to(peft_model.device)
        labels = torch.tensor([train_ds.labels[j].value for j in batch_indices])

        # Forward pass with output_hidden_states
        outputs = peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get hidden states from target layer
        # +1 because index 0 is embeddings
        hidden_states = outputs.hidden_states[LAYER + 1]

        # Compute probe-based loss
        loss = probe_loss(hidden_states, labels, detection_mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

# -------------------------------------------------------------------
# Step 5: Evaluate after fine-tuning
# -------------------------------------------------------------------
print("\n--- Step 5: Evaluation ---")

peft_model.eval()

# Collect new activations from fine-tuned model (single layer, no LAYER axis)
test_acts_new = pl.collect_activations(peft_model, test_tokens, layers=[LAYER])
test_prepared_new = test_acts_new.mean_pool()

# Evaluate with same probe
probs_new = probe.predict(test_prepared_new)
post_auroc = pl.metrics.auroc(test_ds.labels, probs_new)

print(f"\nResults:")
print(f"  Pre-training AUROC: {pre_auroc:.4f}")
print(f"  Post-training AUROC: {post_auroc:.4f}")
print(f"  Change: {post_auroc - pre_auroc:+.4f}")

# Check if harmful scores decreased
harmful_test_indices = [i for i, label in enumerate(test_ds.labels) if label == Label.POSITIVE]
pre_harmful_scores = probs[harmful_test_indices].mean().item()
post_harmful_scores = probs_new[harmful_test_indices].mean().item()

print(f"\nMean harmful probability:")
print(f"  Pre-training: {pre_harmful_scores:.4f}")
print(f"  Post-training: {post_harmful_scores:.4f}")
print(f"  Reduction: {pre_harmful_scores - post_harmful_scores:.4f}")

# Save LoRA weights
peft_model.save_pretrained("lora_weights")
print("\nLoRA weights saved to lora_weights/")

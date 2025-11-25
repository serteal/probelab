"""Verify streaming mode works correctly (single pass)."""
import time
import torch
import probelib as pl
from probelib import Pipeline
from probelib.preprocessing import Pool, SelectLayer

torch.set_float32_matmul_precision("high")

def verify():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "google/gemma-2-2b-it"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    dataset = pl.datasets.CircuitBreakersDataset()[:50] + pl.datasets.BenignInstructionsDataset()[:50]
    print(f"Dataset: {len(dataset)} samples")

    layers = [12]

    # Test 1: streaming=True (single pass with partial_fit)
    print("\n=== streaming=True (single pass with partial_fit) ===")
    pipeline = Pipeline([
        ("select", SelectLayer(layers[0])),
        ("agg", Pool(dim="sequence", method="mean")),
        ("probe", pl.probes.Logistic(device=device)),
    ])
    start = time.perf_counter()
    pl.scripts.train_from_model(
        pipelines=pipeline, model=model, tokenizer=tokenizer, dataset=dataset,
        layers=layers, mask=pl.masks.assistant(), batch_size=32,
        streaming=True, verbose=False,
    )
    print(f"Time: {time.perf_counter() - start:.2f}s")

    # Test 2: streaming=False (batch mode - probe handles epochs internally)
    print("\n=== streaming=False (batch mode) ===")
    pipeline = Pipeline([
        ("select", SelectLayer(layers[0])),
        ("agg", Pool(dim="sequence", method="mean")),
        ("probe", pl.probes.Logistic(device=device)),
    ])
    start = time.perf_counter()
    pl.scripts.train_from_model(
        pipelines=pipeline, model=model, tokenizer=tokenizer, dataset=dataset,
        layers=layers, mask=pl.masks.assistant(), batch_size=32,
        streaming=False, verbose=False,
    )
    print(f"Time: {time.perf_counter() - start:.2f}s")

    # Test 3: Two-step approach (RECOMMENDED)
    print("\n=== TWO-STEP: collect_activations + train_pipelines (RECOMMENDED) ===")
    start = time.perf_counter()
    acts = pl.collect_activations(
        model=model, tokenizer=tokenizer, dataset=dataset,
        layers=layers, mask=pl.masks.assistant(), batch_size=32,
        collection_strategy="mean", verbose=False,
    )
    collection_time = time.perf_counter() - start

    pipeline = Pipeline([
        ("select", SelectLayer(layers[0])),
        ("probe", pl.probes.Logistic(device=device)),
    ])
    train_start = time.perf_counter()
    pl.scripts.train_pipelines(pipeline, acts, dataset.labels, verbose=False)
    train_time = time.perf_counter() - train_start

    print(f"Collection: {collection_time:.2f}s, Training: {train_time:.4f}s, Total: {collection_time + train_time:.2f}s")

if __name__ == "__main__":
    verify()

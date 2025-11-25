"""Profile activation collection to find bottlenecks."""
import time
import torch
import probelab as pl

torch.set_float32_matmul_precision("high")

def profile_collection():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "google/gemma-2-2b-it"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers")
    
    # Create small dataset
    print("\nLoading dataset...")
    dataset = pl.datasets.CircuitBreakersDataset()[:50] + pl.datasets.BenignInstructionsDataset()[:50]
    print(f"Dataset size: {len(dataset)} samples")
    
    layers = [12]
    batch_size = 32
    
    # Profile tokenization
    print("\n=== Profiling Tokenization ===")
    start = time.perf_counter()
    tokenized = pl.processing.tokenize_dataset(dataset, tokenizer, mask=pl.masks.assistant())
    tokenize_time = time.perf_counter() - start
    print(f"Tokenization: {tokenize_time:.3f}s")
    print(f"Max seq len: {tokenized['input_ids'].shape[1]}")
    
    # Profile raw model forward pass
    print("\n=== Profiling Raw Model Forward Pass ===")
    n_samples = len(dataset)
    
    # Warmup
    with torch.inference_mode():
        batch = {k: v[:batch_size].to(device) for k, v in tokenized.items() if k != 'detection_mask'}
        _ = model(**batch)
    torch.cuda.synchronize()
    
    # Measure
    times = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    for run in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                batch = {k: v[i:i+batch_size].to(device) for k, v in tokenized.items() if k != 'detection_mask'}
                _ = model(**batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s ({n_samples/elapsed:.1f} samples/sec)")
    print(f"  Mean: {sum(times)/len(times):.3f}s")
    
    # Profile with HookedModel
    print("\n=== Profiling HookedModel ===")
    times = []
    for run in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with pl.HookedModel(model, layers, detach_activations=True) as hooked:
            for i in range(0, n_samples, batch_size):
                batch = {k: v[i:i+batch_size].to(device) for k, v in tokenized.items()}
                acts = hooked.get_activations(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s ({n_samples/elapsed:.1f} samples/sec)")
    print(f"  Mean: {sum(times)/len(times):.3f}s")
    
    # Profile collect_activations with streaming=False
    print("\n=== Profiling collect_activations (batch mode) ===")
    times = []
    for run in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        acts = pl.collect_activations(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            layers=layers,
            mask=pl.masks.assistant(),
            batch_size=batch_size,
            streaming=False,
            verbose=False,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s ({n_samples/elapsed:.1f} samples/sec)")
    print(f"  Mean: {sum(times)/len(times):.3f}s")
    
    # Profile collect_activations with streaming=True
    print("\n=== Profiling collect_activations (streaming mode) ===")
    times = []
    for run in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        act_iter = pl.collect_activations(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            layers=layers,
            mask=pl.masks.assistant(),
            batch_size=batch_size,
            streaming=True,
            verbose=False,
        )
        # Consume iterator
        for batch_acts in act_iter:
            pass
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s ({n_samples/elapsed:.1f} samples/sec)")
    print(f"  Mean: {sum(times)/len(times):.3f}s")
    
    # Profile train_from_model
    print("\n=== Profiling train_from_model (streaming) ===")
    from probelab import Pipeline
    from probelab.preprocessing import Pool, SelectLayer
    
    times = []
    for run in range(3):
        pipeline = Pipeline([
            ("select", SelectLayer(layers[0])),
            ("agg", Pool(dim="sequence", method="mean")),
            ("probe", pl.probes.Logistic(device=device)),
        ])
        torch.cuda.synchronize()
        start = time.perf_counter()
        pl.scripts.train_from_model(
            pipelines=pipeline,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            layers=layers,
            mask=pl.masks.assistant(),
            batch_size=batch_size,
            streaming=True,
            verbose=False,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s ({n_samples/elapsed:.1f} samples/sec)")
    print(f"  Mean: {sum(times)/len(times):.3f}s")

if __name__ == "__main__":
    profile_collection()

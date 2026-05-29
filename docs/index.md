# probelab documentation

`probelab` trains probes and activation monitors on language-model activations.
It is collector-agnostic: bring activations from Transformers, TransformerLens,
NNsight, nnterp, vLLM-Lens, mirin, PyTorch hooks, or saved tensors, and the
probing code stays the same.

## Guides

- [Activations](activations.md) — the dimension model and the flat+offsets
  sequence layout that everything else builds on.
- [Masks](masks.md) — composable token selectors for detection control.
- [Probes](probes.md) — the probe gallery and the shared `fit`/`predict` API.
- [Collecting activations](collection.md) — running a model and the
  "bring your own collector" contract.
- [Datasets](datasets.md) — the dataset container and the built-in registry.
- [Storage](storage.md) — saving/loading/streaming activations to disk.

## The 30-second mental model

```
dataset (dialogues + labels)
  └─ tokenize_dataset(..., mask=...)        -> Tokens   (flat+offsets)
       └─ collect_activations(model, ...)   -> Activations
            └─ reduce: .mean("s") / .last() / .select("l", k)
                 └─ probe.fit(acts, labels) -> probe.predict(acts)
                      └─ metrics.auroc(labels, scores)
```

See the runnable scripts in [`../examples/`](../examples/).

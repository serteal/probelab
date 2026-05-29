# Collecting activations

`probelab` does not run models itself by default — it consumes `Activations`.
You can either use the built-in mirin adapter or bring activations from any
collector.

## With the mirin adapter

Install the extra and the backend:

```bash
pip install "probelab[collection]"
# plus mirin from https://github.com/serteal/mirin
```

```python
import probelab as pl
from probelab import collection

tokens = pl.tokenize_dataset(dataset, tokenizer, mask=pl.masks.assistant())

# Pooled features, one vector per sample, for a set of layers:
feats = collection.collect_activations(model, tokens, layers=[8, 12], pool="mean")

# Or keep token-level activations and stream chunks to bound memory:
for chunk, indices in collection.stream_activations(model, tokens, layers=12):
    ...   # chunk is an ActivationChunk (flat data + offsets + detection_mask)
```

`collect_activations(..., pool=None)` returns token-level `Activations` you can
reduce yourself; `pool="mean"|"max"|"last_token"|...` pools on the fly.

The functions import the mirin backend lazily, so `import probelab` never pulls
in mirin or transformers. Calling them without mirin installed raises a clear
`ImportError` pointing at the extra.

## Bring your own collector

Any collector works as long as you can assemble an `Activations` object. The
flat+offsets constructors make this cheap:

```python
import torch
import probelab as pl

# Suppose you ran your own forward pass and have, per sample, a
# [n_tokens, hidden] activation tensor and a [n_tokens] detection mask.
per_sample_acts = [...]   # list of [t_i, H] tensors
per_sample_mask = [...]   # list of [t_i] bool tensors

data = torch.cat(per_sample_acts, dim=0)
mask = torch.cat(per_sample_mask, dim=0)
offsets = torch.tensor(
    [0, *torch.cumsum(torch.tensor([t.shape[0] for t in per_sample_acts]), 0).tolist()]
)

acts = pl.Activations.from_flat(data, offsets, mask, dims="bsh")
```

If you already have padded rectangular tensors, use `from_padded` with an
`attention_mask` instead. For pooled features (no sequence axis) use
`from_tensor(data, dims="bh")` or `dims="blh"` for multilayer.

## Chat templates

Detection masks (role/text masks) depend on aligning tokens to chat-template
structure. `probelab` auto-detects llama / gemma / qwen / qwen3 from the
tokenizer name and, failing that, from its `chat_template` markup. For a renamed
or fine-tuned checkpoint, force it explicitly:

```python
tokens = pl.tokenize_dataset(dataset, tokenizer, mask=..., template="qwen3")
```

If role detection matches no tokens, `probelab` logs a warning telling you to
pass `template=` — the masks would otherwise silently select nothing.

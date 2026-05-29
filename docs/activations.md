# Activations

`probelab.Activations` is the backend-agnostic container that every probe
consumes. It carries an activation tensor plus an explicit **dims** string that
labels each axis.

## Dimension strings

| dims     | meaning                                | natural dense shape              |
|----------|----------------------------------------|----------------------------------|
| `"bh"`   | batch × hidden (pooled features)       | `[B, H]`                         |
| `"blh"`  | batch × layer × hidden                 | `[B, L, H]`                      |
| `"bsh"`  | batch × seq × hidden (token-level)     | `[B, S, H]` (padded)             |
| `"blsh"` | batch × layer × seq × hidden           | `[B, L, S, H]` (padded)          |

The public constructor accepts these **natural dense layouts**:

```python
import torch
import probelab as pl

feats = pl.Activations(torch.randn(8, 128), dims="bh")          # pooled
seqs  = pl.Activations(torch.randn(8, 24, 128), dims="bsh")     # token-level
```

## Flat + offsets layout

When `"s"` is in `dims`, sequences are stored **flat** (no padding kept):

- `data` — `[total_tokens, hidden]` (or `[total_tokens, n_layers, hidden]`),
  all samples concatenated.
- `offsets` — `[batch + 1]` int64 cumulative token counts; sample `i` spans
  `data[offsets[i]:offsets[i+1]]`.
- `detection_mask` — `[total_tokens]` bool, which tokens to score.

Passing a padded `[B, S, H]` tensor to the constructor (optionally with an
`attention_mask`) converts it to this layout automatically. To build it
directly use the classmethods:

```python
pl.Activations.from_tensor(data, dims="bh")                 # dense, no seq axis
pl.Activations.from_flat(data, offsets, detection_mask, dims="bsh")
pl.Activations.from_padded(data, attention_mask=mask, dims="bsh")
```

This layout avoids materializing `B × S_max × H` padded tensors, which matters
for datasets with highly variable sequence lengths.

## Reductions (dim-passing API)

Reductions name the axis they collapse:

```python
acts.mean("s")        # pool tokens -> removes the seq axis
acts.max("s")
acts.last()           # last detected token per sample (sugar for select("s", -1))
acts.ema(alpha=0.5)   # EMA-then-max over the sequence
acts.rolling(window=10)

acts.mean("l")        # pool layers
acts.select("l", 12)  # pick one layer (drops the layer axis)
acts.select("l", [8, 12, 16])   # keep a subset of layers
acts.flatten()        # concatenate layers into the hidden dim
```

`iter_layers()` yields `(layer_index, single_layer_activations)` for sweeps.

## Properties & utilities

```python
acts.batch_size      # number of samples
acts.hidden_size
acts.n_layers        # None when no layer axis
acts.seq_len         # max sequence length (None when no seq axis)
acts.total_tokens    # None when no seq axis
acts.to("cuda")      # move device
pl.Activations.cat([a, b, c])   # concatenate along the batch dim
```

### Choosing a probe input

- **Feature probes** (`Logistic`, `MLP`, `MassMean`, `Bilinear`, `EEMLP`, `TPC`)
  expect one vector per sample: reduce the sequence first (`acts.mean("s")`),
  or pass token-level activations and let the probe train per-token.
- **Sequence probes** (`Attention`, `MHA`, `MultiMax`, `SoftAttention`,
  `RollingAttention`, `PositionalAttention`, `GatedBipolar`) require the `"s"`
  axis and learn their own pooling.

All probes require a **single layer** — call `select("l", k)` first if a layer
axis is present.

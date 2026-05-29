# Probes

All probes share an estimator-style API and are also `torch.nn.Module`s.

## Shared API

```python
probe = pl.probes.Logistic(seed=0)
probe.fit(train_acts, train_labels)      # returns self
scores = probe.predict(test_acts)        # probabilities in [0, 1]
logits = probe.predict_logits(test_acts) # raw logits
probe.save("probe.pt")
probe = pl.probes.Logistic.load("probe.pt", device="cpu")
probe.fitted                             # True after fit
```

Lower-level hooks (`initialize`, `forward`, `loss_on_batch`, `predict_tensor`)
operate on plain tensors for custom PyTorch training loops.

Common constructor arguments: `seed`, `device`, `cast`
(`"float32"`/`"float16"`/`"bfloat16"`), and `optimizer_fn` / `scheduler_fn`
factories. Sequence probes additionally take `patience`, `val_split`,
`eval_interval`, and `max_padded_tokens`.

## Feature probes (one vector per sample)

Reduce the sequence first (`acts.mean("s")`) or pass token-level activations to
train per-token.

| Probe       | Summary                                             |
|-------------|-----------------------------------------------------|
| `Logistic`  | L2-regularized logistic regression (LBFGS or SGD).  |
| `MLP`       | Single-hidden-layer MLP.                            |
| `MassMean`  | Closed-form difference-in-means direction.          |
| `Bilinear`  | Low-rank symmetric CP quadratic classifier.         |
| `EEMLP`     | Early-exit MLP with heads at each depth.            |
| `TPC`       | Truncated polynomial classifier (progressive).      |

## Sequence probes (require the `"s"` axis)

These learn their own pooling over tokens.

| Probe                 | Summary                                          |
|-----------------------|--------------------------------------------------|
| `Attention`           | Single-query attention pooling.                  |
| `MHA`                 | Small transformer encoder with a CLS token.      |
| `MultiMax`            | Multi-head hard-max pooling.                      |
| `SoftAttention`       | Per-head soft attention pooling.                 |
| `RollingAttention`    | Per-head windowed attention, max across windows. |
| `PositionalAttention` | Attention with a learned positional bias.        |
| `GatedBipolar`        | Gated projections with max / negative-min pool.  |

## Example

```python
import probelab as pl

# feature probe
feats = acts.select("l", 12).mean("s")
clf = pl.probes.MassMean(seed=0).fit(feats, labels)

# sequence probe (keeps the token axis)
seq = acts.select("l", 12)
att = pl.probes.Attention(hidden_dim=64, n_epochs=200, seed=0).fit(seq, labels)
print(pl.metrics.auroc(test_labels, att.predict(test_seq)))
```

## Training many probes at once

`probelab.utils.VmapEnsemble` trains N identical-architecture probes (differing
in learning rate / weight decay / epoch budget) in parallel with `torch.func`
vmap and a vectorized AdamW — useful for hyperparameter sweeps and layer sweeps.

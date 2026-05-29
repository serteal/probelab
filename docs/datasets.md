# Datasets

`probelab.Dataset` is a minimal container of dialogues + labels + optional
per-sample metadata.

## Building datasets

```python
import probelab as pl

# From row records or JSONL:
ds = pl.Dataset.from_records(
    [{"messages": [{"role": "user", "content": "hi"}], "label": 1}],
)
ds = pl.Dataset.from_jsonl("data.jsonl")        # messages_key / label_key configurable
ds.to_jsonl("out.jsonl")
```

Labels coerce flexibly: `1`/`0`, `"positive"`/`"negative"`, `True`/`False`, or
`Label` enum members all work. Roles are normalized (`human`→`user`,
`gpt`→`assistant`, ...).

## Manipulation

```python
len(ds); ds[0]; ds[10:20]; ds[[0, 5, 9]]   # slicing returns a Dataset
ds.shuffle(seed=0)
ds.sample(100, stratified=True, seed=0)
train, test = ds.split(0.8, stratified=True, seed=0)
ds.where([label == 1 for label in ds.labels])
ds.positive; ds.negative                    # label views
combined = ds_a + ds_b                       # concatenate (tracks `source`)
```

## Built-in registry

```python
pl.datasets.list_categories()
pl.datasets.list_datasets(category="deception")
pl.datasets.info("circuit_breakers")
ds = pl.datasets.load("circuit_breakers")
```

Loaders are registered lazily — dataset modules (and their HuggingFace
downloads) are only imported the first time you call a `datasets` API, so
`import probelab` stays fast.

> Many topic loaders are single-class by design (all-positive or all-negative)
> and are meant to be combined with `+`. Training a probe on a single such
> loader will raise "both classes required"; combine a positive and a negative
> source first.

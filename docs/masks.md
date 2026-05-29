# Masks

Masks are composable token selectors. A `Mask` wraps a function
`(dialogues, metadata) -> bool tensor` and is evaluated during tokenization to
produce the **detection mask** — the tokens a probe is trained and scored on.

```python
from probelab import masks
```

## Composition

Combine masks with boolean operators:

```python
masks.assistant() & masks.nth_message(-1)   # last message, assistant tokens
masks.contains("yes") | masks.contains("no")
~masks.user()                               # everything except user tokens
```

`~` keeps padding masked (it intersects with the attention mask), so negation
never selects padding.

## Built-in masks

**Basic**

- `masks.all()` — all non-padding tokens.
- `masks.none()` — nothing.

**Role**

- `masks.assistant()`, `masks.user()`, `masks.system()`
- `masks.role(name)` — generic; `include_padding=` controls whether the
  template's surrounding control tokens count as part of the role region.

**Position**

- `masks.nth_message(n)` — the n-th message (negative counts from the end).
- `masks.first_n_tokens(n)`, `masks.last_n_tokens(n)`, `masks.last_token()`
- `masks.after(s)`, `masks.before(s)`, `masks.between(start, end)` — text
  anchored (need the formatted text / char-to-token map from the tokenizer).
- `masks.thinking()` — tokens inside `<think>…</think>` reasoning blocks
  (Qwen3, DeepSeek-R1, QwQ, ...).
- `masks.padding(base_mask, before=2, after=2)` — dilate a mask with context.

**Text**

- `masks.contains(text, case_sensitive=False)`
- `masks.regex(pattern, flags=0)`

**Content**

- `masks.special_tokens(ids=None)` — BOS/EOS/PAD/etc.

## Usage

```python
import probelab as pl
from probelab import masks

tokens = pl.tokenize_dataset(
    dataset,
    tokenizer,
    mask=masks.assistant() & ~masks.special_tokens(),
)
```

The mask determines `detection_mask`; later, pooling (`acts.mean("s")`,
`acts.last()`) and per-token training operate only on detected tokens.

> Text-anchored masks (`between`, `contains`, `regex`, ...) rely on
> chat-template metadata. If role/text detection ever selects nothing, pass an
> explicit `template=` to `tokenize_dataset` (see
> [Collecting activations](collection.md)).

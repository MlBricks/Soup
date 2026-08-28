# SOUP

SOUP is a state-processing architecture with **fixed Observer State Memory and SOUP Fusion** and replaceable per-layer sequence mixers and FFNs.

The implementation is based on the validated SOUP architecture from the supplied ESA-vs-SOUP notebook. Its defining post-stack memory and fusion blocks are not user-selectable components.

## API

```python
from mlbricks import soup

model = soup(
    dim=512,
    width=1116,
    depth=2,
    mixer="esa",
    ffn="saffn",
    backend="auto",
    precision="fp16",
)
```

Input and output are `[batch, sequence, dim]` tensors:

```python
y = model(x)
```

## Architecture

```text
Input
  ↓
SOUP Layer 1
  ├─ RMSNorm
  ├─ configurable Mixer
  ├─ configurable FFN / state update
  └─ learned residual mix
  ↓
...
  ↓
Final state + representation
  ↓
Observer State Memory       ← fixed SOUP block
  ↓
Causal memory read
  ↓
SOUP Fusion                 ← fixed SOUP block
  ├─ representation
  ├─ projected state
  └─ observer memory
  ↓
Output
```

The Observer State Memory uses only **preceding** token memory: the current token's write is subtracted before the read. SOUP Fusion uses a learned 3-way softmax over representation, state and memory before a SiLU fusion projection and residual update.

## Mixers

Built-in mixer names:

- `"esa"`
- `"bolt"`
- custom `torch.nn.Module` or module factory

Use one mixer everywhere:

```python
model = soup(dim=512, width=1116, depth=3, mixer="esa")
```

Or select one per layer:

```python
model = soup(
    dim=512,
    width=1116,
    depth=3,
    mixer=["esa", "bolt", "esa"],
    mixer_config=[
        {"head": 8},
        {"num_heads": 8},
        {"head": 8},
    ],
)
```

For BOLT, `latent_dim` is optional. If omitted, SOUP does not synthesize a value; BOLT uses its own default (`32` in MLBricks 1.0.0). An explicit value is forwarded unchanged:

```python
mixer_config={"num_heads": 8, "latent_dim": 64}
```

## FFNs

Built-in FFN names:

- `"saffn"` — the state-aware FFN equations used by the validated SOUP notebook. This is the canonical SOUP default and updates SOUP state.
- `"ffn"` — a conventional two-layer MLP adapter. A normal FFN has no recurrent state equation, so it leaves the incoming SOUP state unchanged.
- custom `torch.nn.Module` or factory.

Per-layer composition works like mixer composition:

```python
model = soup(
    dim=512,
    width=[1116, 1024],
    depth=2,
    mixer=["esa", "bolt"],
    ffn=["saffn", "ffn"],
    mixer_config=[{"head": 8}, {"num_heads": 8}],
    ffn_config=[{}, {"hidden": 2048, "activation": "silu"}],
)
```

When widths differ, SOUP inserts learned state bridges between layers.

### Custom mixer contract

A custom mixer must map:

```text
[B,T,D] → [B,T,D]
```

### Custom FFN contracts

SOUP accepts either a state-aware FFN:

```python
out, next_state = ffn(x, current_context, previous_context, state)
```

or a conventional module:

```python
out = ffn(x)
```

A conventional module leaves SOUP state unchanged.

## Backend policy

SOUP follows the current MLBricks backend policy:

```text
auto | native | pytorch
```

`backend="auto"` is the top-level default and is inherited by built-in components unless their own config explicitly sets another backend.

```python
model.set_backend("native")
print(model.resolved_backend())
```

For internal ESA layers, SOUP intentionally disables ESA's independent device movement and lets the parent SOUP module control placement with normal PyTorch `.to(...)` calls.

## Fixed SOUP blocks

`ObserverStateMemory` and `SOUPFusion` are defining parts of SOUP. There are intentionally no public `memory=` or `fusion=` constructor arguments.

## License

SOUP is licensed under the PolyForm Noncommercial License 1.0.0. Commercial use requires a separate written commercial license. See `LICENSE`.

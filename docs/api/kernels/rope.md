# Rotary Position Embeddings (RoPE)

Rotary Position Embeddings for encoding position information in attention layers.

## Overview

RoPE encodes position information by rotating query and key vectors in 2D subspaces. It's used in LLaMA, Mistral, Qwen, and most modern LLMs.

**Mathematical formula:**

For a vector $x$ at position $m$, the rotated vector is:

$$
\text{RoPE}(x, m) = \begin{pmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \\ \vdots \end{pmatrix} \odot \begin{pmatrix} \cos(m\theta_0) \\ \cos(m\theta_0) \\ \cos(m\theta_1) \\ \cos(m\theta_1) \\ \vdots \end{pmatrix} + \begin{pmatrix} -x_1 \\ x_0 \\ -x_3 \\ x_2 \\ \vdots \end{pmatrix} \odot \begin{pmatrix} \sin(m\theta_0) \\ \sin(m\theta_0) \\ \sin(m\theta_1) \\ \sin(m\theta_1) \\ \vdots \end{pmatrix}
$$

Where $\theta_i = \frac{1}{\text{base}^{2i/d}}$ with typical base=10000.

## Key Properties

1. **Relative position encoding**: The dot product between rotated vectors depends only on their relative position
2. **Long-range decay**: Attention naturally decays with distance due to the rotation frequencies
3. **No learned parameters**: Position encodings are computed, not learned

## Performance Characteristics

| Configuration | PyTorch | Triton | Speedup |
|---------------|---------|--------|---------|
| head_dim=128, seq=2048 | 67 μs | 23 μs | 2.9x |
| head_dim=128, seq=8192 | 267 μs | 92 μs | 2.9x |
| head_dim=64, seq=2048 | 34 μs | 12 μs | 2.8x |

## Usage Examples

### Module API (Recommended)

```python
import torch
from rotalabs_accel import RotaryEmbedding

# Create RoPE module
rope = RotaryEmbedding(
    dim=128,           # Head dimension
    max_seq_len=8192,  # Maximum sequence length
    base=10000.0,      # Frequency base (standard is 10000)
)

# Query and Key tensors
# Shape: [batch, seq_len, num_heads, head_dim]
q = torch.randn(2, 512, 32, 128, device="cuda", dtype=torch.float16)
k = torch.randn(2, 512, 32, 128, device="cuda", dtype=torch.float16)

# Apply RoPE
q_rot, k_rot = rope(q, k, seq_len=512)
```

### Functional API

```python
from rotalabs_accel import build_rope_cache, apply_rope

# Build cache once (at model initialization)
cos, sin = build_rope_cache(
    seq_len=8192,
    head_dim=128,
    base=10000.0,
    device="cuda",
)

# Apply during forward pass
# Slice cache to actual sequence length
q_rot, k_rot = apply_rope(q, k, cos[:seq_len], sin[:seq_len])
```

### With Grouped Query Attention (GQA)

RoPE works with different numbers of Q and K heads:

```python
# LLaMA 3 style: 32 Q heads, 8 KV heads
q = torch.randn(2, 512, 32, 128, device="cuda")  # 32 heads
k = torch.randn(2, 512, 8, 128, device="cuda")   # 8 heads

# apply_rope handles broadcasting automatically
q_rot, k_rot = rope(q, k, seq_len=512)
```

### Position Offset (for KV Cache)

During generation with KV cache, you need to offset positions:

```python
# First token: position 0
q1, k1 = rope(q[:, :1], k[:, :1], seq_len=1)
cached_k = k1

# Next token: position 1
# Pass offset to start from correct position
q2, k2 = rope(q[:, :1], k[:, :1], seq_len=1, offset=1)
cached_k = torch.cat([cached_k, k2], dim=1)
```

---

## API Reference

### Functions

::: rotalabs_accel.kernels.rope.apply_rope
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.kernels.rope.rope_torch
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.kernels.rope.build_rope_cache
    options:
      show_root_heading: true
      heading_level: 4

### Classes

::: rotalabs_accel.kernels.rope.RotaryEmbedding
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - __init__
        - forward

## Implementation Notes

### Cache Precomputation

The cos/sin tables are computed once and reused:

```python
def build_rope_cache(seq_len, head_dim, base=10000.0, device="cuda"):
    # Compute frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))

    # Compute position angles
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, inv_freq)

    # Cache cos and sin
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    return cos, sin
```

### Memory Layout

The rotation is applied to pairs of adjacent dimensions:

- $(x_0, x_1)$ rotated by $\theta_0$
- $(x_2, x_3)$ rotated by $\theta_1$
- etc.

This "interleaved" layout matches LLaMA and most modern models. Some older models use "sequential" layout where first half and second half are paired.

### Extended Context (YaRN, NTK)

For extended context lengths, you can modify the base frequency:

```python
# NTK-aware scaling for 4x context extension
base = 10000 * 4.0

rope = RotaryEmbedding(dim=128, max_seq_len=32768, base=base)
```

## References

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Original RoPE paper
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - Uses RoPE
- [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071) - Context extension with RoPE

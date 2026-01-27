# Normalization

RMSNorm (Root Mean Square Layer Normalization) kernels with optional residual fusion.

## Overview

RMSNorm is a simpler alternative to LayerNorm used in modern LLMs like LLaMA, Mistral, and Qwen. Unlike LayerNorm, it doesn't center the input (no mean subtraction), which reduces computation.

**Mathematical formula:**

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}} \cdot \gamma
$$

Where:

- $x$ is the input tensor
- $n$ is the hidden dimension
- $\epsilon$ is a small constant for numerical stability (typically 1e-6)
- $\gamma$ is the learnable weight parameter

## Performance Characteristics

RMSNorm is a **memory-bound** operation with low arithmetic intensity (~1-2 FLOPs/byte). The Triton kernel provides speedups by:

1. **Fusing operations**: Combines variance computation, normalization, and scaling in a single kernel
2. **Reducing memory traffic**: Reads input once, writes output once
3. **Using FP32 accumulation**: Ensures numerical stability while keeping I/O in FP16

| Configuration | PyTorch | Triton | Speedup |
|---------------|---------|--------|---------|
| hidden=4096, seq=2048 | 45 μs | 12 μs | 3.8x |
| hidden=8192, seq=2048 | 89 μs | 24 μs | 3.7x |
| hidden=4096, seq=8192 | 178 μs | 47 μs | 3.8x |

## Usage Examples

### Basic RMSNorm

```python
import torch
from rotalabs_accel import TritonRMSNorm, rmsnorm

# Module API (recommended for models)
norm = TritonRMSNorm(hidden_size=4096, eps=1e-6)
x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.float16)
y = norm(x)

# Functional API (for custom implementations)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)
y = rmsnorm(x, weight, eps=1e-6)
```

### Fused Residual + RMSNorm

In transformer blocks, RMSNorm is typically applied after adding a residual:

```python
from rotalabs_accel import rmsnorm_residual_fused

# Standard pattern (two operations):
# x = x + residual
# x = rmsnorm(x, weight)

# Fused version (one operation, ~2x faster):
x = rmsnorm_residual_fused(x, residual, weight, eps=1e-6)
```

The fused version eliminates an intermediate tensor allocation and memory round-trip.

---

## API Reference

### Functions

::: rotalabs_accel.kernels.normalization.rmsnorm
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.kernels.normalization.rmsnorm_torch
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.kernels.normalization.rmsnorm_residual_fused
    options:
      show_root_heading: true
      heading_level: 4

### Classes

::: rotalabs_accel.kernels.normalization.TritonRMSNorm
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - __init__
        - forward

## Implementation Notes

### Automatic Dispatch

The `rmsnorm` function automatically selects the best implementation:

```python
def rmsnorm(x, weight, eps=1e-6):
    if HAS_TRITON and x.is_cuda and weight.is_cuda:
        return _rmsnorm_triton(x, weight, eps)
    return rmsnorm_torch(x, weight, eps)
```

### Numerical Stability

All implementations use FP32 accumulation for the variance computation, even when inputs are FP16/BF16. This prevents numerical issues with large hidden dimensions.

### Block Size Selection

The Triton kernel automatically selects block sizes based on the hidden dimension:

```python
BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)
```

## References

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - Original RMSNorm paper
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - Uses RMSNorm

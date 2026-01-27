# Activations

SwiGLU (Swish-Gated Linear Unit) activation kernel used in modern LLMs.

## Overview

SwiGLU is a variant of GLU (Gated Linear Unit) that uses SiLU (Swish) as the activation function. It's used in LLaMA, PaLM, Mistral, and other state-of-the-art models.

**Mathematical formula:**

$$
\text{SwiGLU}(x) = \text{SiLU}(x W_{gate}) \odot (x W_{up})
$$

Where:

- $W_{gate}$ and $W_{up}$ are learned weight matrices
- $\text{SiLU}(z) = z \cdot \sigma(z) = z \cdot \frac{1}{1 + e^{-z}}$
- $\odot$ is element-wise multiplication

The full FFN layer in SwiGLU-based transformers is:

$$
\text{FFN}(x) = \text{SwiGLU}(x) W_{down}
$$

## Performance Characteristics

SwiGLU activation is **memory-bound** with ~1.3 FLOPs/byte. The Triton kernel fuses the SiLU activation and element-wise multiply:

| Configuration | PyTorch | Triton | Speedup |
|---------------|---------|--------|---------|
| intermediate=11008, seq=2048 | 89 μs | 31 μs | 2.9x |
| intermediate=16384, seq=2048 | 134 μs | 48 μs | 2.8x |
| intermediate=11008, seq=8192 | 354 μs | 123 μs | 2.9x |

## Usage Examples

### Full SwiGLU FFN Module

```python
import torch
from rotalabs_accel import SwiGLU

# Create SwiGLU FFN (includes gate, up, and down projections)
ffn = SwiGLU(
    hidden_size=4096,       # Input/output dimension
    intermediate_size=11008,  # Intermediate dimension (~2.7x hidden)
    bias=False,             # Most LLMs don't use bias
)
ffn = ffn.to("cuda")

# Forward pass
x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.float16)
y = ffn(x)  # Shape: (2, 512, 4096)
```

### Functional API (After Your Own Projections)

If you have your own projection layers:

```python
from rotalabs_accel import swiglu_fused

# Your custom projections
gate = x @ W_gate.T  # Shape: (batch, seq, intermediate)
up = x @ W_up.T      # Shape: (batch, seq, intermediate)

# Fused activation
activated = swiglu_fused(gate, up)  # Shape: (batch, seq, intermediate)

# Down projection
output = activated @ W_down.T  # Shape: (batch, seq, hidden)
```

### Integration with Hugging Face Models

```python
from transformers import AutoModelForCausalLM
from rotalabs_accel import SwiGLU

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Replace MLP layers with optimized SwiGLU
for layer in model.model.layers:
    hidden_size = layer.mlp.gate_proj.in_features
    intermediate_size = layer.mlp.gate_proj.out_features

    optimized_mlp = SwiGLU(hidden_size, intermediate_size)
    # Copy weights...

    layer.mlp = optimized_mlp
```

---

## API Reference

### Functions

::: rotalabs_accel.kernels.activations.swiglu_fused
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.kernels.activations.swiglu_torch
    options:
      show_root_heading: true
      heading_level: 4

### Classes

::: rotalabs_accel.kernels.activations.SwiGLU
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - __init__
        - forward

## Implementation Notes

### Kernel Fusion

The Triton kernel computes `silu(gate) * up` in a single pass:

```python
@triton.jit
def _swiglu_kernel(Gate, Up, Out, n_elements, BLOCK_SIZE):
    # Load gate and up
    gate = tl.load(Gate + offsets, mask=mask)
    up = tl.load(Up + offsets, mask=mask)

    # Fused SiLU + multiply
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    out = silu_gate * up

    tl.store(Out + offsets, out, mask=mask)
```

This saves one memory round-trip compared to the separate PyTorch operations.

### Numerical Stability

The kernel uses the standard sigmoid implementation. For very large negative values, sigmoid approaches 0, making the output approach 0 as well (which is the correct behavior).

## Why SwiGLU?

SwiGLU was shown to outperform other activation functions like ReLU and GELU in the [PaLM paper](https://arxiv.org/abs/2204.02311). The key advantages:

1. **Gating mechanism**: The gate controls information flow, similar to attention
2. **Smooth gradients**: SiLU provides smooth gradients everywhere (unlike ReLU)
3. **Better training dynamics**: Empirically leads to better model quality

The tradeoff is more parameters (3 projection matrices instead of 2) and compute, but the quality improvements are worth it for large models.

## References

- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) - Introduces SwiGLU for LLMs
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - Analysis of GLU variants
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - Uses SwiGLU

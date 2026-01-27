# Modules

Drop-in `nn.Module` replacements with optimized Triton kernels.

## Overview

These modules provide the **same interface** as their PyTorch counterparts but use optimized Triton kernels when available. If Triton isn't installed or the input is on CPU, they automatically fall back to pure PyTorch.

```python
# These work identically, but TritonRMSNorm is faster on GPU
norm_pytorch = torch.nn.RMSNorm(4096)
norm_triton = TritonRMSNorm(4096)

# Same API, same results, different speed
y1 = norm_pytorch(x)
y2 = norm_triton(x)  # Up to 3.8x faster on GPU
```

## Module Summary

| Module | Replaces | Speedup | Use Case |
|--------|----------|---------|----------|
| `TritonRMSNorm` | `nn.RMSNorm` | 3.8x | LLaMA, Mistral normalization |
| `SwiGLU` | Custom FFN | 2.9x | LLaMA, PaLM FFN layers |
| `RotaryEmbedding` | Manual RoPE | 2.9x | Position encoding |
| `Int8Linear` | `nn.Linear` | 3.3x | Memory-efficient inference |
| `QuantizedLinear` | `nn.Linear` | 3.3x | Easy model quantization |

---

## TritonRMSNorm

RMS normalization layer, used in LLaMA, Mistral, Qwen.

```python
from rotalabs_accel import TritonRMSNorm

norm = TritonRMSNorm(hidden_size=4096, eps=1e-6)
x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.float16)
y = norm(x)
```

::: rotalabs_accel.kernels.normalization.TritonRMSNorm
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - __init__
        - forward

---

## SwiGLU

Complete SwiGLU FFN module with gate, up, and down projections.

```python
from rotalabs_accel import SwiGLU

ffn = SwiGLU(
    hidden_size=4096,
    intermediate_size=11008,
    bias=False,
)
x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.float16)
y = ffn(x)  # Shape: (2, 512, 4096)
```

::: rotalabs_accel.kernels.activations.SwiGLU
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - __init__
        - forward

---

## RotaryEmbedding

Rotary Position Embeddings with automatic cache management.

```python
from rotalabs_accel import RotaryEmbedding

rope = RotaryEmbedding(
    dim=128,
    max_seq_len=8192,
    base=10000.0,
)

# Apply to query and key
q = torch.randn(2, 512, 32, 128, device="cuda")
k = torch.randn(2, 512, 32, 128, device="cuda")
q_rot, k_rot = rope(q, k, seq_len=512)
```

::: rotalabs_accel.kernels.rope.RotaryEmbedding
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - __init__
        - forward

---

## Int8Linear

Linear layer with INT8 quantized weights.

```python
from rotalabs_accel import Int8Linear

linear = Int8Linear(
    in_features=4096,
    out_features=4096,
    bias=False,
)
linear.quantize_weights(pretrained_weight)
y = linear(x)
```

::: rotalabs_accel.kernels.gemm.Int8Linear
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - __init__
        - quantize_weights
        - forward

---

## QuantizedLinear

Higher-level quantized linear with easy conversion from `nn.Linear`.

```python
from rotalabs_accel import QuantizedLinear

# Convert existing layer
linear = torch.nn.Linear(4096, 4096)
qlinear = QuantizedLinear.from_linear(linear)

# Use like normal
y = qlinear(x)
```

::: rotalabs_accel.quantization.symmetric.QuantizedLinear
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - __init__
        - quantize_weights
        - forward
        - from_linear

---

## Using with Existing Models

### Replace Layers in a Model

```python
from rotalabs_accel import TritonRMSNorm, SwiGLU, QuantizedLinear

def optimize_model(model):
    """Replace layers with optimized versions."""
    for name, module in model.named_children():
        # Replace RMSNorm
        if isinstance(module, torch.nn.RMSNorm):
            setattr(model, name, TritonRMSNorm(module.weight.shape[0]))

        # Quantize Linear
        elif isinstance(module, torch.nn.Linear):
            setattr(model, name, QuantizedLinear.from_linear(module))

        # Recurse
        else:
            optimize_model(module)

    return model
```

### With Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM
from rotalabs_accel import TritonRMSNorm

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Replace all RMSNorm layers
for layer in model.model.layers:
    layer.input_layernorm = TritonRMSNorm(
        layer.input_layernorm.weight.shape[0]
    )
    layer.post_attention_layernorm = TritonRMSNorm(
        layer.post_attention_layernorm.weight.shape[0]
    )
```

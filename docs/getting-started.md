# Getting Started

## Installation

### Basic Installation

```bash
pip install rotalabs-accel
```

### With Triton (Recommended for GPU)

```bash
pip install rotalabs-accel[triton]
```

### Development Installation

```bash
pip install rotalabs-accel[dev]
```

## Basic Usage

### Using Drop-in Modules

The easiest way to use rotalabs-accel is through drop-in module replacements:

```python
import torch
from rotalabs_accel import TritonRMSNorm, SwiGLU, RotaryEmbedding

# RMSNorm - replaces torch.nn.RMSNorm
norm = TritonRMSNorm(hidden_size=4096)
x = torch.randn(2, 512, 4096)
x_normed = norm(x)

# SwiGLU - complete FFN with gate/up/down projections
ffn = SwiGLU(hidden_size=4096, intermediate_size=11008)
x_ffn = ffn(x_normed)

# RoPE - rotary position embeddings
rope = RotaryEmbedding(dim=128, max_seq_len=8192)
q = torch.randn(2, 512, 32, 128)  # [batch, seq, heads, head_dim]
k = torch.randn(2, 512, 32, 128)
q_rot, k_rot = rope(q, k)
```

### Using Functional API

For more control, use the functional API:

```python
from rotalabs_accel import rmsnorm, swiglu_fused, apply_rope, build_rope_cache

# RMSNorm
weight = torch.ones(4096)
y = rmsnorm(x, weight, eps=1e-6)

# SwiGLU (just the activation, after projections)
gate = x @ W_gate
up = x @ W_up
activated = swiglu_fused(gate, up)

# RoPE
cos, sin = build_rope_cache(seq_len=512, head_dim=128)
q_rot, k_rot = apply_rope(q, k, cos, sin)
```

### INT8 Quantization

For memory-efficient inference with INT8 weights:

```python
from rotalabs_accel import Int8Linear, quantize_symmetric

# Create and quantize a linear layer
linear = Int8Linear(in_features=4096, out_features=4096)
linear.quantize_weights(pretrained_weight)  # FP16/FP32 -> INT8

# Use like normal
y = linear(x)

# Or quantize tensors directly
x_int8, scale = quantize_symmetric(x)
```

## Device Detection

Use the utility functions to check device capabilities:

```python
from rotalabs_accel import (
    get_device,
    is_cuda_available,
    is_triton_available,
    get_device_properties,
)

# Auto-detect best device
device = get_device()  # Returns CUDA if available, else CPU

# Check capabilities
props = get_device_properties()
print(f"GPU: {props['name']}")
print(f"Compute capability: {props['compute_capability']}")
print(f"Supports FP8: {props['supports_fp8']}")
print(f"Supports BF16: {props['supports_bf16']}")
```

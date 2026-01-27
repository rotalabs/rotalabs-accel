# Getting Started

This guide covers installation, basic usage, and best practices for rotalabs-accel.

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- (Optional) CUDA 11.8+ for GPU acceleration
- (Optional) Triton 2.1+ for optimized kernels

### Installation Options

=== "Basic (CPU only)"

    ```bash
    pip install rotalabs-accel
    ```

    Core functionality with PyTorch-only implementations. Works on any platform.

=== "With Triton (Recommended)"

    ```bash
    pip install rotalabs-accel[triton]
    ```

    Enables Triton-optimized kernels for 3-4x speedups on NVIDIA GPUs.

=== "Full Installation"

    ```bash
    pip install rotalabs-accel[triton,benchmark,dev]
    ```

    Includes benchmarking tools and development dependencies.

=== "From Source"

    ```bash
    git clone https://github.com/rotalabs/rotalabs-accel.git
    cd rotalabs-accel
    pip install -e ".[triton,dev]"
    ```

### Verify Installation

```python
import rotalabs_accel
print(f"Version: {rotalabs_accel.__version__}")

from rotalabs_accel import is_cuda_available, is_triton_available
print(f"CUDA available: {is_cuda_available()}")
print(f"Triton available: {is_triton_available()}")
```

---

## Core Concepts

### Automatic Dispatch

Every function in rotalabs-accel automatically selects the best implementation:

```
┌─────────────────────────────────────────────────────────────┐
│                    rmsnorm(x, weight)                        │
├─────────────────────────────────────────────────────────────┤
│  Is x on CUDA?                                               │
│  ├── YES: Is Triton installed?                              │
│  │   ├── YES: Use Triton kernel      ← Fastest              │
│  │   └── NO:  Use PyTorch on GPU                            │
│  └── NO:  Use PyTorch on CPU         ← Universal fallback   │
└─────────────────────────────────────────────────────────────┘
```

You don't need to check conditions or write platform-specific code—just call the function.

### Module vs Functional API

rotalabs-accel provides two ways to use each operation:

**Module API** - For use in `nn.Module` classes:

```python
from rotalabs_accel import TritonRMSNorm, SwiGLU

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = TritonRMSNorm(4096)
        self.ffn = SwiGLU(4096, 11008)

    def forward(self, x):
        return self.ffn(self.norm(x))
```

**Functional API** - For direct tensor operations:

```python
from rotalabs_accel import rmsnorm, swiglu_fused

def forward(x, weight, gate_weight, up_weight, down_weight):
    x = rmsnorm(x, weight)
    gate = x @ gate_weight.T
    up = x @ up_weight.T
    return swiglu_fused(gate, up) @ down_weight.T
```

---

## Using the Kernels

### RMSNorm

RMSNorm (Root Mean Square Layer Normalization) is used in LLaMA, Mistral, Qwen, and other modern LLMs.

**Formula:** `y = x * rsqrt(mean(x²) + ε) * weight`

```python
import torch
from rotalabs_accel import TritonRMSNorm, rmsnorm, rmsnorm_residual_fused

# Module API
norm = TritonRMSNorm(hidden_size=4096, eps=1e-6)
x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.float16)
y = norm(x)

# Functional API
weight = torch.ones(4096, device="cuda", dtype=torch.float16)
y = rmsnorm(x, weight, eps=1e-6)

# Fused with residual (common in transformer blocks)
residual = torch.randn_like(x)
y = rmsnorm_residual_fused(x, residual, weight, eps=1e-6)
# Equivalent to: rmsnorm(x + residual, weight), but faster
```

!!! tip "When to use `rmsnorm_residual_fused`"
    Transformer blocks typically add a residual before each sublayer normalization:
    ```python
    x = layernorm(x + residual)  # Standard pattern
    ```
    The fused version eliminates an intermediate tensor allocation and memory round-trip.

### SwiGLU

SwiGLU (Swish-Gated Linear Unit) is the FFN activation used in LLaMA and PaLM.

**Formula:** `y = SiLU(x @ W_gate) × (x @ W_up)`

```python
from rotalabs_accel import SwiGLU, swiglu_fused

# Full FFN module (includes projections)
ffn = SwiGLU(
    hidden_size=4096,
    intermediate_size=11008,  # Typically ~2.7x hidden_size
    bias=False,
)
x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.float16)
y = ffn(x)  # Output shape: (2, 512, 4096)

# Functional API (just the activation, after you've done projections)
gate = x @ W_gate  # Shape: (2, 512, 11008)
up = x @ W_up      # Shape: (2, 512, 11008)
activated = swiglu_fused(gate, up)  # Shape: (2, 512, 11008)
output = activated @ W_down  # Shape: (2, 512, 4096)
```

### Rotary Position Embeddings (RoPE)

RoPE encodes position information by rotating query and key vectors.

```python
from rotalabs_accel import RotaryEmbedding, apply_rope, build_rope_cache

# Module API (manages cache automatically)
rope = RotaryEmbedding(
    dim=128,           # Head dimension
    max_seq_len=8192,  # Maximum sequence length
    base=10000.0,      # Base for frequency computation
)

# Apply to Q and K tensors
q = torch.randn(2, 512, 32, 128, device="cuda")  # [batch, seq, heads, head_dim]
k = torch.randn(2, 512, 8, 128, device="cuda")   # GQA: fewer K heads

q_rot, k_rot = rope(q, k, seq_len=512)

# Functional API (build cache once, apply many times)
cos, sin = build_rope_cache(seq_len=512, head_dim=128, device="cuda")
q_rot, k_rot = apply_rope(q, k, cos, sin)
```

!!! note "RoPE with different Q/K head counts"
    RoPE works with Grouped Query Attention (GQA) where K/V have fewer heads than Q.
    The `apply_rope` function handles broadcasting automatically.

### INT8 Quantization

Reduce memory usage by 50% with W8A16 (INT8 weights, FP16 activations).

```python
from rotalabs_accel import (
    quantize_symmetric,
    quantize_weight_per_channel,
    dequantize,
    calculate_quantization_error,
    QuantizedLinear,
    Int8Linear,
)

# Per-tensor quantization
weight = torch.randn(4096, 4096, dtype=torch.float16)
weight_int8, scale = quantize_symmetric(weight)
weight_restored = dequantize(weight_int8, scale)

# Per-channel quantization (better accuracy)
weight_int8, scales = quantize_weight_per_channel(weight)

# Measure quantization error
errors = calculate_quantization_error(weight, weight_int8, scales)
print(f"Max error: {errors['max_abs_error']:.6f}")
print(f"SNR: {errors['snr_db']:.1f} dB")

# High-level: convert existing Linear layer
linear = torch.nn.Linear(4096, 4096)
quantized = QuantizedLinear.from_linear(linear)

# Or create directly
qlinear = Int8Linear(in_features=4096, out_features=4096)
qlinear.quantize_weights(pretrained_weights)
```

---

## Device Utilities

### Automatic Device Selection

```python
from rotalabs_accel import get_device

# Returns best available device
device = get_device()  # 'cuda' if available, else 'cpu'

# Force specific device
device = get_device("cuda:1")
device = get_device("cpu")
```

### Capability Checking

```python
from rotalabs_accel import get_device_properties

props = get_device_properties()
print(f"GPU: {props['name']}")
print(f"VRAM: {props['total_memory'] / 1e9:.1f} GB")
print(f"Compute capability: {props['compute_capability']}")
print(f"Supports BF16: {props['supports_bf16']}")
print(f"Supports FP8: {props['supports_fp8']}")

# Plan dtype based on capabilities
if props['supports_bf16']:
    dtype = torch.bfloat16
else:
    dtype = torch.float16
```

---

## Best Practices

### 1. Use FP16 or BF16 for Inference

Triton kernels are optimized for 16-bit floating point:

```python
# Good - FP16 or BF16
x = torch.randn(..., device="cuda", dtype=torch.float16)
x = torch.randn(..., device="cuda", dtype=torch.bfloat16)

# Works but slower - kernels will internally convert
x = torch.randn(..., device="cuda", dtype=torch.float32)
```

### 2. Match Hidden Dimensions to Powers of 2

Triton kernels use block sizes that are powers of 2. Dimensions like 4096, 8192 are optimal:

```python
# Optimal
norm = TritonRMSNorm(4096)   # 4096 = 2^12
norm = TritonRMSNorm(8192)   # 8192 = 2^13

# Still works but may be slightly slower
norm = TritonRMSNorm(4000)   # Not a power of 2
```

### 3. Batch for Throughput

The kernels are optimized for batched operations:

```python
# Better: single large batch
x = torch.randn(32, 2048, 4096, device="cuda")
y = norm(x)

# Slower: many small batches
for i in range(32):
    x = torch.randn(1, 2048, 4096, device="cuda")
    y = norm(x)
```

### 4. Preallocate RoPE Cache

For attention with fixed max sequence length, build the cache once:

```python
# At initialization
cos, sin = build_rope_cache(max_seq_len, head_dim, device="cuda")

# During inference
q_rot, k_rot = apply_rope(q, k, cos[:seq_len], sin[:seq_len])
```

---

## Troubleshooting

### Triton Not Found

```
ImportError: No module named 'triton'
```

**Solution:** Install with Triton extras: `pip install rotalabs-accel[triton]`

Note: Triton only works on Linux with NVIDIA GPUs. On other platforms, PyTorch fallbacks are used automatically.

### CUDA Out of Memory

If you're hitting OOM with large models, use INT8 quantization:

```python
# Before: 32-bit weights (full precision)
linear = torch.nn.Linear(4096, 4096).cuda()  # 67 MB

# After: 8-bit weights
qlinear = QuantizedLinear.from_linear(linear)  # 16 MB + overhead
```

### Numerical Differences

Triton kernels use FP32 accumulation for numerical stability, but minor differences from PyTorch are expected:

```python
# Maximum expected difference (should be < 1e-3 for FP16)
torch.allclose(triton_output, pytorch_output, atol=1e-3, rtol=1e-3)
```

---

## Next Steps

- [API Reference](api/index.md) - Complete API documentation
- [Kernel Details](api/kernels/normalization.md) - Deep dive into each kernel
- [Quantization Guide](api/quantization.md) - Advanced quantization techniques

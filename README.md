# rotalabs-accel

High-performance inference acceleration with Triton kernels, quantization, and speculative decoding.

## Features

- **Triton-Optimized Kernels**: RMSNorm, SwiGLU, RoPE, INT8 GEMM with automatic GPU/CPU fallback
- **Quantization**: INT8 symmetric quantization with per-channel and per-tensor support
- **Drop-in Modules**: `nn.Module` replacements that match PyTorch API
- **Device Abstraction**: Unified device detection and capability checking

## Installation

```bash
pip install rotalabs-accel
```

With optional Triton support (recommended for GPU):

```bash
pip install rotalabs-accel[triton]
```

With all extras:

```bash
pip install rotalabs-accel[triton,benchmark,dev]
```

## Quick Start

```python
import torch
from rotalabs_accel import (
    TritonRMSNorm,
    SwiGLU,
    RotaryEmbedding,
    Int8Linear,
    get_device,
    is_triton_available,
)

# Check device capabilities
device = get_device()  # Auto-selects CUDA if available
print(f"Using device: {device}")
print(f"Triton available: {is_triton_available()}")

# Use optimized modules (drop-in replacements)
hidden_size = 4096
intermediate_size = 11008

norm = TritonRMSNorm(hidden_size).to(device)
swiglu = SwiGLU(hidden_size, intermediate_size).to(device)
rope = RotaryEmbedding(dim=128, max_seq_len=8192)

# Forward pass
x = torch.randn(1, 512, hidden_size, device=device)
x = norm(x)
x = swiglu(x)
```

## Kernels

### RMSNorm

Root Mean Square Layer Normalization with optional residual fusion:

```python
from rotalabs_accel import rmsnorm, rmsnorm_residual_fused, TritonRMSNorm

# Functional API
out = rmsnorm(x, weight, eps=1e-6)

# With fused residual addition
out, residual = rmsnorm_residual_fused(x, residual, weight, eps=1e-6)

# Module API
norm = TritonRMSNorm(hidden_size=4096, eps=1e-6)
out = norm(x)
```

### SwiGLU

SwiGLU activation (used in Llama, Mistral, etc.):

```python
from rotalabs_accel import swiglu_fused, SwiGLU

# Functional API
out = swiglu_fused(gate, up)

# Module API (includes linear projections)
swiglu = SwiGLU(hidden_size=4096, intermediate_size=11008)
out = swiglu(x)
```

### Rotary Position Embeddings (RoPE)

Rotary embeddings for position encoding:

```python
from rotalabs_accel import apply_rope, build_rope_cache, RotaryEmbedding

# Functional API
cos, sin = build_rope_cache(seq_len=2048, dim=128)
q_out, k_out = apply_rope(q, k, cos, sin)

# Module API (manages cache automatically)
rope = RotaryEmbedding(dim=128, max_seq_len=8192, base=10000.0)
q_out, k_out = rope(q, k, seq_len=512)
```

### INT8 GEMM

W8A16 quantized matrix multiplication:

```python
from rotalabs_accel import int8_gemm, Int8Linear, QuantizedLinear

# Functional API
out = int8_gemm(x_fp16, weight_int8, scale)

# Module API
linear = Int8Linear(in_features=4096, out_features=4096)
out = linear(x)

# Higher-level quantized linear
qlinear = QuantizedLinear(in_features=4096, out_features=4096)
out = qlinear(x)
```

## Quantization

INT8 symmetric quantization utilities:

```python
from rotalabs_accel import (
    quantize_symmetric,
    dequantize,
    quantize_weight_per_channel,
    calculate_quantization_error,
)

# Per-tensor quantization
x_quant, scale = quantize_symmetric(x)
x_recon = dequantize(x_quant, scale)

# Per-channel weight quantization
w_quant, scale = quantize_weight_per_channel(weight)

# Measure quantization error
error = calculate_quantization_error(x)
print(f"Quantization MSE: {error:.6f}")
```

## Device Utilities

```python
from rotalabs_accel import (
    get_device,
    is_cuda_available,
    is_triton_available,
    get_device_properties,
)

# Auto-detect best device
device = get_device()  # Returns CUDA if available, else CPU
device = get_device("cuda:1")  # Specific GPU

# Check capabilities
props = get_device_properties()
print(f"GPU: {props['name']}")
print(f"Compute capability: {props['compute_capability']}")
print(f"Supports FP8: {props['supports_fp8']}")
print(f"Supports BF16: {props['supports_bf16']}")
```

## Performance

Benchmarks on A100-80GB with batch_size=1, seq_len=2048, hidden_size=4096:

| Kernel | PyTorch | Triton | Speedup |
|--------|---------|--------|---------|
| RMSNorm | 45 us | 12 us | 3.8x |
| SwiGLU | 89 us | 31 us | 2.9x |
| RoPE | 67 us | 23 us | 2.9x |
| INT8 GEMM | 156 us | 48 us | 3.3x |

## Automatic Fallback

All kernels automatically fall back to PyTorch implementations when:
- CUDA is not available
- Triton is not installed
- Input tensors are on CPU

This ensures your code works everywhere without modification.

## Roadmap

- [ ] FP8 quantization (Hopper/Blackwell)
- [ ] Asymmetric INT4 quantization
- [ ] EAGLE-style speculative decoding
- [ ] Flash Attention integration
- [ ] KV cache compression

## Links

- Documentation: https://rotalabs.github.io/rotalabs-accel/
- PyPI: https://pypi.org/project/rotalabs-accel/
- GitHub: https://github.com/rotalabs/rotalabs-accel
- Website: https://rotalabs.ai
- Contact: research@rotalabs.ai

## License

MIT License - see LICENSE file for details.

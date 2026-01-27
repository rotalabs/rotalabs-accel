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

## Quick Example

```python
import torch
from rotalabs_accel import (
    TritonRMSNorm,
    SwiGLU,
    RotaryEmbedding,
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

## Automatic Fallback

All kernels automatically fall back to PyTorch implementations when:

- CUDA is not available
- Triton is not installed
- Input tensors are on CPU

This ensures your code works everywhere without modification.

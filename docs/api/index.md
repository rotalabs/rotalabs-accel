# API Reference

Complete API documentation for rotalabs-accel, auto-generated from source code docstrings.

## Package Overview

```python
import rotalabs_accel

# Version
rotalabs_accel.__version__  # e.g., "0.1.0"

# All public exports
rotalabs_accel.__all__
```

## Architecture

```
rotalabs_accel/
├── kernels/                    # Triton-optimized kernels
│   ├── normalization.py        # RMSNorm, rmsnorm_residual_fused
│   ├── activations.py          # SwiGLU, swiglu_fused
│   ├── rope.py                 # RoPE, apply_rope, build_rope_cache
│   └── gemm.py                 # INT8 GEMM, Int8Linear
├── quantization/
│   └── symmetric.py            # INT8 symmetric quantization utilities
└── utils/
    └── device.py               # Device detection and capability checking
```

## Quick Reference

### Normalization

| Export | Type | Description |
|--------|------|-------------|
| `TritonRMSNorm` | `nn.Module` | Drop-in RMSNorm replacement |
| `rmsnorm` | Function | Apply RMS normalization |
| `rmsnorm_torch` | Function | PyTorch-only reference implementation |
| `rmsnorm_residual_fused` | Function | Fused RMSNorm + residual addition |

### Activations

| Export | Type | Description |
|--------|------|-------------|
| `SwiGLU` | `nn.Module` | Full SwiGLU FFN with projections |
| `swiglu_fused` | Function | Fused SiLU(gate) × up activation |
| `swiglu_torch` | Function | PyTorch-only reference implementation |

### Position Embeddings

| Export | Type | Description |
|--------|------|-------------|
| `RotaryEmbedding` | `nn.Module` | Self-contained RoPE module |
| `apply_rope` | Function | Apply rotary embeddings to Q/K |
| `rope_torch` | Function | PyTorch-only reference implementation |
| `build_rope_cache` | Function | Precompute cos/sin tables |

### Matrix Multiplication

| Export | Type | Description |
|--------|------|-------------|
| `Int8Linear` | `nn.Module` | INT8 quantized linear layer |
| `int8_gemm` | Function | W8A16 matrix multiplication |
| `int8_gemm_torch` | Function | PyTorch-only reference implementation |

### Quantization Utilities

| Export | Type | Description |
|--------|------|-------------|
| `QuantizedLinear` | `nn.Module` | Higher-level quantized linear layer |
| `quantize_symmetric` | Function | Symmetric INT8 quantization |
| `dequantize` | Function | INT8 → FP16/FP32 conversion |
| `quantize_weight_per_channel` | Function | Per-channel weight quantization |
| `calculate_quantization_error` | Function | Measure quantization accuracy |

### Device Utilities

| Export | Type | Description |
|--------|------|-------------|
| `get_device` | Function | Auto-detect best available device |
| `is_cuda_available` | Function | Check for CUDA support |
| `is_triton_available` | Function | Check for Triton installation |
| `get_device_properties` | Function | Get GPU capabilities |

## Import Patterns

### Recommended: Import specific functions/classes

```python
from rotalabs_accel import (
    TritonRMSNorm,
    SwiGLU,
    RotaryEmbedding,
    get_device,
    is_triton_available,
)
```

### Alternative: Import from submodules

```python
from rotalabs_accel.kernels import rmsnorm, swiglu_fused, apply_rope
from rotalabs_accel.quantization import quantize_symmetric, dequantize
from rotalabs_accel.utils import get_device, get_device_properties
```

### For type checking

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rotalabs_accel import TritonRMSNorm, SwiGLU
```

## Sections

<div class="grid cards" markdown>

-   :material-function: **Kernels**

    ---

    Triton-optimized implementations of core LLM operations.

    - [Normalization](kernels/normalization.md) - RMSNorm
    - [Activations](kernels/activations.md) - SwiGLU
    - [RoPE](kernels/rope.md) - Rotary Position Embeddings
    - [GEMM](kernels/gemm.md) - INT8 Matrix Multiplication

-   :material-memory: **Quantization**

    ---

    INT8 symmetric quantization for memory-efficient inference.

    [Quantization →](quantization.md)

-   :material-view-module: **Modules**

    ---

    Drop-in `nn.Module` replacements.

    [Modules →](modules.md)

-   :material-tools: **Utilities**

    ---

    Device detection and capability checking.

    [Utilities →](utils.md)

</div>

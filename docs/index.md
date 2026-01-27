# rotalabs-accel

**High-performance inference acceleration for modern LLMs with Triton kernels, INT8 quantization, and automatic CPU/GPU fallback.**

<div class="grid cards" markdown>

-   :zap: **Triton-Optimized Kernels**

    ---

    RMSNorm, SwiGLU, RoPE, and INT8 GEMM kernels written in Triton with **3-4x speedups** over PyTorch.

-   :package: **Drop-in Replacements**

    ---

    `nn.Module` classes that match PyTorch's API exactly. Swap one line, get instant speedups.

-   :gear: **Automatic Fallback**

    ---

    Works everywhere. Uses Triton on CUDA, pure PyTorch on CPU. No code changes needed.

-   :chart_with_upwards_trend: **INT8 Quantization**

    ---

    Cut memory usage in half with symmetric INT8 weight quantization (W8A16).

</div>

---

## Why rotalabs-accel?

Modern LLMs like LLaMA, Mistral, and Qwen use the same core operations: RMSNorm, SwiGLU, and RoPE. These operations are **memory-bound**—the GPU spends most of its time moving data, not computing.

By fusing operations and writing custom Triton kernels, we eliminate redundant memory traffic and achieve significant speedups:

| Kernel | PyTorch Baseline | Triton Kernel | Speedup |
|--------|------------------|---------------|---------|
| RMSNorm | 45 μs | 12 μs | **3.8x** |
| SwiGLU | 89 μs | 31 μs | **2.9x** |
| RoPE | 67 μs | 23 μs | **2.9x** |
| INT8 GEMM | 156 μs | 48 μs | **3.3x** |

<small>*Benchmarks on A100-80GB, batch_size=1, seq_len=2048, hidden_size=4096*</small>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      rotalabs-accel                         │
├─────────────────────────────────────────────────────────────┤
│  High-Level API (nn.Module)                                 │
│  ┌──────────────┬──────────────┬──────────────┬───────────┐ │
│  │TritonRMSNorm │   SwiGLU     │RotaryEmbedding│Int8Linear│ │
│  └──────────────┴──────────────┴──────────────┴───────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Functional API                                             │
│  ┌──────────────┬──────────────┬──────────────┬───────────┐ │
│  │   rmsnorm    │ swiglu_fused │  apply_rope  │ int8_gemm │ │
│  └──────────────┴──────────────┴──────────────┴───────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Dispatch Layer (automatic Triton/PyTorch selection)        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────┐ ┌────────────────────────────┐ │
│  │  Triton Kernels (CUDA)  │ │ PyTorch Reference (CPU/GPU)│ │
│  │  - _rmsnorm_kernel      │ │  - rmsnorm_torch           │ │
│  │  - _swiglu_kernel       │ │  - swiglu_torch            │ │
│  │  - _rope_kernel         │ │  - rope_torch              │ │
│  │  - _int8_gemm_kernel    │ │  - int8_gemm_torch         │ │
│  └─────────────────────────┘ └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Zero-friction adoption**: Every module is a drop-in replacement. `TritonRMSNorm` has the exact same interface as `torch.nn.RMSNorm`.

2. **Graceful degradation**: If Triton isn't available or inputs are on CPU, kernels automatically fall back to pure PyTorch. Your code works everywhere.

3. **Numerical stability**: All kernels use FP32 accumulation internally, matching PyTorch's numerical behavior.

4. **Minimal dependencies**: Core package only requires PyTorch. Triton is optional.

---

## Quick Start

### Installation

```bash
# Basic installation (CPU support only)
pip install rotalabs-accel

# With Triton support (recommended for GPU)
pip install rotalabs-accel[triton]

# Full installation with benchmarking tools
pip install rotalabs-accel[triton,benchmark]
```

### Basic Usage

```python
import torch
from rotalabs_accel import TritonRMSNorm, SwiGLU, RotaryEmbedding, get_device

# Auto-detect best device
device = get_device()  # Returns 'cuda' if available, else 'cpu'

# Create optimized layers (same API as PyTorch)
norm = TritonRMSNorm(hidden_size=4096, eps=1e-6).to(device)
ffn = SwiGLU(hidden_size=4096, intermediate_size=11008).to(device)
rope = RotaryEmbedding(dim=128, max_seq_len=8192)

# Forward pass - automatically uses Triton on CUDA
x = torch.randn(1, 2048, 4096, device=device, dtype=torch.float16)
x = norm(x)
x = ffn(x)
```

### INT8 Quantization

Reduce memory usage by 50% with W8A16 quantization:

```python
from rotalabs_accel import QuantizedLinear

# Convert existing linear layer
linear = torch.nn.Linear(4096, 4096)
quantized = QuantizedLinear.from_linear(linear)

# Memory usage: 4096 * 4096 * 2 bytes (FP16) → 4096 * 4096 * 1 byte (INT8)
# 32 MB → 16 MB per layer
```

---

## Supported Operations

### Normalization

| Operation | Description | Use Case |
|-----------|-------------|----------|
| `rmsnorm` | Root Mean Square Normalization | LLaMA, Mistral, Qwen |
| `rmsnorm_residual_fused` | RMSNorm + residual addition | Transformer blocks |
| `TritonRMSNorm` | Drop-in `nn.Module` replacement | Any model using RMSNorm |

### Activations

| Operation | Description | Use Case |
|-----------|-------------|----------|
| `swiglu_fused` | SiLU(gate) × up | LLaMA, PaLM, Mistral FFN |
| `SwiGLU` | Full FFN module with projections | Transformer FFN replacement |

### Position Embeddings

| Operation | Description | Use Case |
|-----------|-------------|----------|
| `build_rope_cache` | Precompute cos/sin tables | Initialization |
| `apply_rope` | Apply rotary embeddings to Q/K | Attention layers |
| `RotaryEmbedding` | Self-contained `nn.Module` | Attention replacement |

### Quantization

| Operation | Description | Use Case |
|-----------|-------------|----------|
| `quantize_symmetric` | Symmetric INT8 quantization | Weight compression |
| `quantize_weight_per_channel` | Per-output-channel scales | Better accuracy |
| `Int8Linear` | INT8 linear layer | Memory-efficient inference |
| `QuantizedLinear` | Higher-level quantized linear | Easy model conversion |

---

## Integration Examples

### With Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM
from rotalabs_accel import TritonRMSNorm, SwiGLU

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Replace RMSNorm layers
for name, module in model.named_modules():
    if "layernorm" in name.lower() or "rmsnorm" in name.lower():
        parent = model.get_submodule(".".join(name.split(".")[:-1]))
        attr = name.split(".")[-1]
        setattr(parent, attr, TritonRMSNorm(module.weight.shape[0]))
```

### With vLLM Custom Kernels

```python
# rotalabs-accel kernels can be used as building blocks for custom models
from rotalabs_accel import rmsnorm, swiglu_fused, apply_rope

class OptimizedLlamaBlock(torch.nn.Module):
    def forward(self, x, residual, cos, sin):
        # Fused residual + RMSNorm
        x = rmsnorm_residual_fused(x, residual, self.ln_weight)

        # Attention with RoPE
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k = apply_rope(q, k, cos, sin)
        ...
```

---

## Device Capabilities

Check what's available on your system:

```python
from rotalabs_accel import (
    get_device,
    is_cuda_available,
    is_triton_available,
    get_device_properties,
)

print(f"CUDA available: {is_cuda_available()}")
print(f"Triton available: {is_triton_available()}")

if is_cuda_available():
    props = get_device_properties()
    print(f"GPU: {props['name']}")
    print(f"Compute capability: {props['compute_capability']}")
    print(f"Supports BF16: {props['supports_bf16']}")
    print(f"Supports FP8: {props['supports_fp8']}")
```

---

## Roadmap

- [ ] FP8 quantization for Hopper/Blackwell GPUs
- [ ] Asymmetric INT4 quantization (GPTQ-style)
- [ ] EAGLE-style speculative decoding
- [ ] Flash Attention integration
- [ ] KV cache compression
- [ ] CUDA graphs for static workloads

---

## Links

- [GitHub Repository](https://github.com/rotalabs/rotalabs-accel)
- [PyPI Package](https://pypi.org/project/rotalabs-accel/)
- [Rotalabs Website](https://rotalabs.ai)

## License

MIT License - see [LICENSE](https://github.com/rotalabs/rotalabs-accel/blob/main/LICENSE) for details.

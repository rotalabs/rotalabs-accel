# INT8 GEMM

W8A16 (INT8 weights, FP16 activations) matrix multiplication kernels.

## Overview

INT8 GEMM enables memory-efficient inference by storing weights in 8-bit format while keeping activations in FP16. This provides:

- **2x memory reduction** for weight storage
- **Faster inference** due to reduced memory bandwidth requirements
- **Minimal accuracy loss** with per-channel quantization

**Computation scheme:**

```
output = (activation_fp16 @ weight_int8.dequantize()) + bias
```

The dequantization happens in registers during the matmul, so the memory traffic reduction directly translates to speedup.

## Performance Characteristics

| Configuration | FP16 GEMM | INT8 GEMM | Speedup | Memory Saved |
|---------------|-----------|-----------|---------|--------------|
| 4096x4096 | 156 μs | 48 μs | 3.3x | 16 MB |
| 8192x8192 | 620 μs | 189 μs | 3.3x | 64 MB |
| 4096x11008 | 418 μs | 128 μs | 3.3x | 43 MB |

## Usage Examples

### High-Level: QuantizedLinear

The easiest way to use INT8 inference:

```python
import torch
from rotalabs_accel import QuantizedLinear

# Convert existing pretrained layer
linear = torch.nn.Linear(4096, 4096)
linear.load_state_dict(pretrained_weights)

# Quantize to INT8
qlinear = QuantizedLinear.from_linear(linear)
qlinear = qlinear.cuda()

# Use like normal
x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.float16)
y = qlinear(x)  # Output is FP16
```

### Low-Level: Int8Linear

For more control over quantization:

```python
from rotalabs_accel import Int8Linear

# Create layer
linear = Int8Linear(
    in_features=4096,
    out_features=4096,
    bias=False,
)

# Quantize weights manually
linear.quantize_weights(pretrained_weight_fp16)

# Forward pass
y = linear(x)
```

### Functional API

For custom implementations:

```python
from rotalabs_accel import int8_gemm, quantize_weight_per_channel

# Quantize weights once
weight_int8, scales = quantize_weight_per_channel(weight_fp16)

# Use in forward pass
output = int8_gemm(x, weight_int8, scales)
```

### Quantizing an Entire Model

```python
from rotalabs_accel import QuantizedLinear

def quantize_model(model):
    """Replace all Linear layers with QuantizedLinear."""
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            quantized = QuantizedLinear.from_linear(module)
            setattr(model, name, quantized)
        else:
            quantize_model(module)
    return model

# Quantize LLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = quantize_model(model)
model = model.cuda()

# Memory usage: 14GB -> 7GB (approximately)
```

---

## API Reference

### Functions

::: rotalabs_accel.kernels.gemm.int8_gemm
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.kernels.gemm.int8_gemm_torch
    options:
      show_root_heading: true
      heading_level: 4

### Classes

::: rotalabs_accel.kernels.gemm.Int8Linear
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - __init__
        - quantize_weights
        - forward

## Quantization Details

### Symmetric Quantization

We use symmetric quantization with per-channel scales:

```
scale[i] = max(|weight[i, :]|) / 127
weight_int8[i, :] = round(weight[i, :] / scale[i]).clamp(-128, 127)
```

This provides a good balance between accuracy and performance.

### Per-Channel vs Per-Tensor

| Method | Accuracy | Memory | Speed |
|--------|----------|--------|-------|
| Per-tensor | Lower | Minimal overhead | Fastest |
| Per-channel | Higher | 1 scale per output | Slightly slower |

We default to per-channel (per-output-row) quantization for better accuracy.

### Quantization Error

Typical quantization errors for random weights:

| Metric | Value |
|--------|-------|
| Max absolute error | ~0.01 |
| Mean absolute error | ~0.002 |
| SNR | ~45-50 dB |

For real model weights, errors may vary based on weight distribution.

## Implementation Notes

### Memory Layout

- Weights: INT8, shape `(out_features, in_features)`
- Scales: FP32, shape `(out_features,)`
- Activations: FP16, shape `(..., in_features)`
- Output: FP16, shape `(..., out_features)`

### Kernel Strategy

The Triton kernel:

1. Loads weight tiles as INT8 (1 byte per element)
2. Dequantizes in registers using the scale vector
3. Performs FP16 matmul with FP32 accumulation
4. Stores FP16 output

This minimizes memory traffic while maintaining numerical precision.

### Fallback Behavior

On CPU or without Triton, the kernel falls back to:

```python
def int8_gemm_torch(x, weight_int8, scale):
    weight_fp = (weight_int8.float() * scale.unsqueeze(1)).to(x.dtype)
    return x @ weight_fp.T
```

## Comparison with Other Quantization Methods

| Method | Bits | Scheme | Accuracy | Speed |
|--------|------|--------|----------|-------|
| **INT8 (this)** | 8 | Symmetric | High | Fast |
| GPTQ | 4 | Asymmetric + groups | Medium-High | Moderate |
| AWQ | 4 | Activation-aware | High | Moderate |
| FP8 (Hopper) | 8 | Native hardware | Very High | Very Fast |

INT8 symmetric quantization is a good default choice that works on all GPUs and provides a solid accuracy/speed tradeoff.

## References

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs](https://arxiv.org/abs/2211.10438)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

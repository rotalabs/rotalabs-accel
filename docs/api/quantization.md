# Quantization

INT8 symmetric quantization utilities for efficient inference.

## Overview

This module provides utilities for quantizing neural network weights to INT8 format, enabling memory-efficient inference with minimal accuracy loss.

### Quantization Scheme

We use **symmetric quantization** with the following formula:

```
scale = max(|tensor|) / 127
quantized = round(tensor / scale).clamp(-128, 127).to(int8)
```

To dequantize:

```
dequantized = quantized.float() * scale
```

### Benefits

| Aspect | FP16 | INT8 |
|--------|------|------|
| Memory per weight | 2 bytes | 1 byte |
| Memory reduction | - | **50%** |
| Accuracy | Baseline | ~99.5%+ of baseline |

### Use Cases

- **Large model inference**: Fit bigger models in GPU memory
- **Deployment**: Reduce model size for edge devices
- **Batched inference**: Handle more concurrent requests

## Quick Start

```python
from rotalabs_accel import (
    quantize_symmetric,
    dequantize,
    quantize_weight_per_channel,
    calculate_quantization_error,
    QuantizedLinear,
)

# Quantize a weight tensor
weight = torch.randn(4096, 4096, dtype=torch.float16)
weight_int8, scale = quantize_symmetric(weight)

# Check accuracy
errors = calculate_quantization_error(weight, weight_int8, scale)
print(f"SNR: {errors['snr_db']:.1f} dB")

# Use in a model
linear = torch.nn.Linear(4096, 4096)
qlinear = QuantizedLinear.from_linear(linear)
```

## Quantization Granularity

### Per-Tensor Quantization

One scale for the entire tensor. Fastest but lowest accuracy.

```python
weight_int8, scale = quantize_symmetric(weight)
# scale.shape: ()
```

### Per-Channel Quantization

One scale per output channel. Better accuracy, minimal overhead.

```python
weight_int8, scales = quantize_weight_per_channel(weight)
# scales.shape: (out_features,)
```

**Recommendation**: Use per-channel for best accuracy/speed tradeoff.

---

## API Reference

### Functions

::: rotalabs_accel.quantization.symmetric.quantize_symmetric
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.quantization.symmetric.dequantize
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.quantization.symmetric.quantize_weight_per_channel
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.quantization.symmetric.calculate_quantization_error
    options:
      show_root_heading: true
      heading_level: 4

### Classes

::: rotalabs_accel.quantization.symmetric.QuantizedLinear
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - __init__
        - quantize_weights
        - forward
        - from_linear

## Best Practices

### 1. Quantize After Training

Quantize pretrained weights, not randomly initialized ones:

```python
# Good: quantize pretrained weights
model = load_pretrained_model()
for module in model.modules():
    if isinstance(module, nn.Linear):
        qmodule = QuantizedLinear.from_linear(module)
        # replace module with qmodule
```

### 2. Evaluate Before Deployment

Always check quantization accuracy on your specific model:

```python
# Run validation before and after quantization
baseline_loss = evaluate(model)
quantize_model(model)
quantized_loss = evaluate(model)
print(f"Loss increase: {quantized_loss - baseline_loss:.4f}")
```

### 3. Keep Certain Layers in FP16

Some layers are more sensitive to quantization:

- First and last layers
- Layers with small weight magnitudes
- Attention output projections

```python
# Skip quantizing sensitive layers
for name, module in model.named_modules():
    if "lm_head" in name or "embed" in name:
        continue  # Keep in FP16
    if isinstance(module, nn.Linear):
        # Quantize
```

## Error Metrics

The `calculate_quantization_error` function returns:

| Metric | Description | Typical Value |
|--------|-------------|---------------|
| `max_abs_error` | Maximum absolute difference | < 0.02 |
| `mean_abs_error` | Mean absolute difference | < 0.005 |
| `relative_error_pct` | Max relative error for significant values | < 1% |
| `snr_db` | Signal-to-noise ratio | > 40 dB |

Values may vary based on weight distribution. Lower SNR indicates more quantization error.

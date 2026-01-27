# Utilities

Device detection and capability checking utilities.

## Overview

These utilities help you write portable code that adapts to available hardware.

```python
from rotalabs_accel import (
    get_device,
    is_cuda_available,
    is_triton_available,
    get_device_properties,
)

# Auto-select best device
device = get_device()  # Returns 'cuda' if available, else 'cpu'

# Check capabilities
print(f"CUDA available: {is_cuda_available()}")
print(f"Triton available: {is_triton_available()}")

# Get detailed GPU info
if is_cuda_available():
    props = get_device_properties()
    print(f"GPU: {props['name']}")
    print(f"VRAM: {props['total_memory'] / 1e9:.1f} GB")
```

---

## API Reference

### Functions

::: rotalabs_accel.utils.device.get_device
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.utils.device.is_cuda_available
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.utils.device.is_triton_available
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.utils.device.get_device_properties
    options:
      show_root_heading: true
      heading_level: 4

::: rotalabs_accel.utils.device.select_dtype
    options:
      show_root_heading: true
      heading_level: 4

---

## Usage Patterns

### Portable Device Selection

```python
from rotalabs_accel import get_device

device = get_device()

# Works on any platform
model = Model().to(device)
x = torch.randn(1, 512, 4096, device=device)
y = model(x)
```

### Conditional Logic Based on Capabilities

```python
from rotalabs_accel import is_triton_available, get_device_properties

if is_triton_available():
    print("Using Triton-optimized kernels")
else:
    print("Falling back to PyTorch")

# Select dtype based on GPU capabilities
if is_cuda_available():
    props = get_device_properties()
    if props.get('supports_bf16', False):
        dtype = torch.bfloat16
        print("Using BF16 (Ampere+)")
    else:
        dtype = torch.float16
        print("Using FP16")
else:
    dtype = torch.float32
    print("Using FP32 on CPU")
```

### Multi-GPU Selection

```python
from rotalabs_accel import get_device

# Select specific GPU
device = get_device("cuda:0")
device = get_device("cuda:1")

# Force CPU even if GPU available
device = get_device("cpu")
```

---

## Device Properties

The `get_device_properties()` function returns a dictionary with:

| Property | Type | Description |
|----------|------|-------------|
| `name` | str | GPU name (e.g., "NVIDIA A100-SXM4-80GB") |
| `compute_capability` | tuple | Compute capability (e.g., (8, 0)) |
| `total_memory` | int | Total VRAM in bytes |
| `supports_bf16` | bool | BF16 tensor core support (Ampere+) |
| `supports_fp8` | bool | FP8 support (Hopper+) |

### GPU Generation Detection

```python
props = get_device_properties()
cc = props['compute_capability']

if cc >= (9, 0):
    print("Hopper (H100) - FP8 support")
elif cc >= (8, 0):
    print("Ampere (A100/A10) - BF16 tensor cores")
elif cc >= (7, 0):
    print("Volta/Turing (V100/T4)")
else:
    print("Older GPU")
```

---

## Triton Availability

Triton requires:

- Linux operating system
- NVIDIA GPU with CUDA
- Python 3.8+

On other platforms, `is_triton_available()` returns `False` and all kernels automatically fall back to PyTorch.

```python
from rotalabs_accel import is_triton_available

if not is_triton_available():
    # Could be:
    # - macOS/Windows (Triton only supports Linux)
    # - No NVIDIA GPU
    # - Triton not installed: pip install triton
    print("Triton not available, using PyTorch fallbacks")
```

"""Device abstraction and detection utilities.

Provides unified interface for device detection and capability checking.
"""

import torch
from typing import Optional, Dict, Any


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_triton_available() -> bool:
    """Check if Triton is available."""
    try:
        import triton
        return True
    except ImportError:
        return False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get a torch device, with smart defaults.

    Args:
        device: Device string ('cuda', 'cpu', 'cuda:0', etc.).
                If None, returns CUDA if available, else CPU.

    Returns:
        torch.device instance.

    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cuda:1')  # Specific GPU
    """
    if device is not None:
        return torch.device(device)

    if is_cuda_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_device_properties(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Get device properties and capabilities.

    Args:
        device: Device to query. If None, uses current CUDA device.

    Returns:
        Dictionary with device properties:
        - name: Device name
        - compute_capability: (major, minor) tuple
        - total_memory: Total memory in bytes
        - supports_fp16: Whether FP16 is supported
        - supports_bf16: Whether BF16 is supported
        - supports_fp8: Whether FP8 is supported (Hopper+)
        - supports_int8_tensor_cores: Whether INT8 tensor cores available

    Example:
        >>> props = get_device_properties()
        >>> print(f"GPU: {props['name']}")
        >>> if props['supports_fp8']:
        ...     print("FP8 quantization available!")
    """
    if not is_cuda_available():
        return {
            'name': 'CPU',
            'compute_capability': (0, 0),
            'total_memory': 0,
            'supports_fp16': True,
            'supports_bf16': False,
            'supports_fp8': False,
            'supports_int8_tensor_cores': False,
        }

    if device is None:
        device = torch.device('cuda')

    props = torch.cuda.get_device_properties(device)
    cc = (props.major, props.minor)

    return {
        'name': props.name,
        'compute_capability': cc,
        'total_memory': props.total_memory,
        'supports_fp16': cc >= (5, 3),  # Maxwell+
        'supports_bf16': cc >= (8, 0),  # Ampere+
        'supports_fp8': cc >= (8, 9),   # Hopper (sm89) / Ada (sm89)
        'supports_int8_tensor_cores': cc >= (7, 5),  # Turing+
        'multi_processor_count': props.multi_processor_count,
    }


def select_dtype(
    preferred: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
) -> torch.dtype:
    """
    Select the best available dtype for the device.

    Args:
        preferred: Preferred dtype if supported.
        device: Device to check capabilities for.

    Returns:
        Best supported dtype.

    Example:
        >>> dtype = select_dtype(torch.bfloat16)
        >>> model = model.to(dtype)
    """
    props = get_device_properties(device)

    if preferred == torch.bfloat16 and not props['supports_bf16']:
        return torch.float16

    if preferred == torch.float16 and not props['supports_fp16']:
        return torch.float32

    return preferred

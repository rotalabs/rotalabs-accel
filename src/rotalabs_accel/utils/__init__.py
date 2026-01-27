"""Utilities for inference acceleration.

Provides device abstraction, model patching, and benchmarking tools.
"""

from rotalabs_accel.utils.device import (
    get_device,
    is_cuda_available,
    is_triton_available,
    get_device_properties,
)

__all__ = [
    "get_device",
    "is_cuda_available",
    "is_triton_available",
    "get_device_properties",
]

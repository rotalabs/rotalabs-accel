"""Quantization utilities for efficient inference.

Provides symmetric and asymmetric quantization for INT8/FP8 inference.
"""

from rotalabs_accel.quantization.symmetric import (
    quantize_symmetric,
    dequantize,
    quantize_weight_per_channel,
    calculate_quantization_error,
    QuantizedLinear,
)

__all__ = [
    # Symmetric quantization
    "quantize_symmetric",
    "dequantize",
    "quantize_weight_per_channel",
    "calculate_quantization_error",
    "QuantizedLinear",
]

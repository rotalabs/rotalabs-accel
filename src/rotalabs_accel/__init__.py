"""rotalabs-accel - High-performance inference acceleration.

Provides:
- Triton-optimized kernels (RMSNorm, SwiGLU, RoPE, INT8 GEMM)
- Quantization utilities (symmetric INT8, per-channel)
- Speculative decoding (EAGLE, Medusa, tree speculation)
- KV-cache compression with eviction policies
- Drop-in nn.Module replacements
- Device abstraction and capability detection

Example:
    >>> from rotalabs_accel import TritonRMSNorm, SwiGLU, RotaryEmbedding
    >>> from rotalabs_accel import get_device, is_triton_available
    >>>
    >>> # Use optimized modules
    >>> norm = TritonRMSNorm(hidden_size=4096)
    >>> rope = RotaryEmbedding(dim=128, max_seq_len=8192)
    >>>
    >>> # Check device capabilities
    >>> if is_triton_available():
    ...     print("Triton kernels available!")
    >>>
    >>> # Speculative decoding
    >>> from rotalabs_accel.speculative import (
    ...     SpeculativeConfig, speculative_decode,
    ...     create_eagle_model, eagle_decode,
    ...     create_medusa_model, medusa_decode,
    ... )
"""

from rotalabs_accel._version import __version__

# Kernels
from rotalabs_accel.kernels import (
    # RMSNorm
    rmsnorm,
    rmsnorm_torch,
    rmsnorm_residual_fused,
    TritonRMSNorm,
    # SwiGLU
    swiglu_fused,
    swiglu_torch,
    SwiGLU,
    # RoPE
    apply_rope,
    rope_torch,
    build_rope_cache,
    RotaryEmbedding,
    # INT8 GEMM
    int8_gemm,
    int8_gemm_torch,
    Int8Linear,
)

# Quantization
from rotalabs_accel.quantization import (
    quantize_symmetric,
    dequantize,
    quantize_weight_per_channel,
    calculate_quantization_error,
    QuantizedLinear,
)

# Utils
from rotalabs_accel.utils import (
    get_device,
    is_cuda_available,
    is_triton_available,
    get_device_properties,
)

__all__ = [
    # Version
    "__version__",
    # RMSNorm
    "rmsnorm",
    "rmsnorm_torch",
    "rmsnorm_residual_fused",
    "TritonRMSNorm",
    # SwiGLU
    "swiglu_fused",
    "swiglu_torch",
    "SwiGLU",
    # RoPE
    "apply_rope",
    "rope_torch",
    "build_rope_cache",
    "RotaryEmbedding",
    # INT8 GEMM
    "int8_gemm",
    "int8_gemm_torch",
    "Int8Linear",
    # Quantization
    "quantize_symmetric",
    "dequantize",
    "quantize_weight_per_channel",
    "calculate_quantization_error",
    "QuantizedLinear",
    # Utils
    "get_device",
    "is_cuda_available",
    "is_triton_available",
    "get_device_properties",
]

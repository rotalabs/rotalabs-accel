"""High-performance Triton kernels for LLM inference.

Provides fused and optimized kernels for common transformer operations.
All kernels have PyTorch fallbacks when Triton is not available.
"""

# Check for Triton availability
try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Normalization kernels
from rotalabs_accel.kernels.normalization import (
    rmsnorm,
    rmsnorm_torch,
    rmsnorm_residual_fused,
    TritonRMSNorm,
)

# Activation kernels
from rotalabs_accel.kernels.activations import (
    swiglu_fused,
    swiglu_torch,
    SwiGLU,
)

# Rotary Position Embeddings
from rotalabs_accel.kernels.rope import (
    apply_rope,
    rope_torch,
    build_rope_cache,
    RotaryEmbedding,
)

# GEMM kernels
from rotalabs_accel.kernels.gemm import (
    int8_gemm,
    int8_gemm_torch,
    Int8Linear,
)

__all__ = [
    # Triton availability
    "HAS_TRITON",
    # Normalization
    "rmsnorm",
    "rmsnorm_torch",
    "rmsnorm_residual_fused",
    "TritonRMSNorm",
    # Activations
    "swiglu_fused",
    "swiglu_torch",
    "SwiGLU",
    # RoPE
    "apply_rope",
    "rope_torch",
    "build_rope_cache",
    "RotaryEmbedding",
    # GEMM
    "int8_gemm",
    "int8_gemm_torch",
    "Int8Linear",
]

"""Drop-in nn.Module replacements with optimized kernels.

These modules provide the same interface as their PyTorch counterparts
but use optimized Triton kernels when available.
"""

from rotalabs_accel.kernels.normalization import TritonRMSNorm
from rotalabs_accel.kernels.activations import SwiGLU
from rotalabs_accel.kernels.rope import RotaryEmbedding
from rotalabs_accel.kernels.gemm import Int8Linear
from rotalabs_accel.quantization.symmetric import QuantizedLinear

__all__ = [
    "TritonRMSNorm",
    "SwiGLU",
    "RotaryEmbedding",
    "Int8Linear",
    "QuantizedLinear",
]

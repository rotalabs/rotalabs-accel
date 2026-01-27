"""
INT8 Quantized Matrix Multiplication (W8A16 GEMM) implemented in Triton.

W8A16 means: Weights are INT8, Activations are FP16.

This is the common inference quantization scheme where:
1. Weights are pre-quantized to INT8 (2x memory reduction)
2. Activations remain in FP16 for accuracy
3. Dequantization happens in registers during the matmul
4. Accumulation uses FP32 for precision

Reference: LLM.int8() - https://arxiv.org/abs/2208.07339
"""

import torch
from typing import Optional

from rotalabs_accel.quantization.symmetric import quantize_weight_per_channel, dequantize

# Optional Triton import
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


def int8_gemm_torch(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PyTorch reference implementation of W8A16 GEMM.

    Works on any device (CPU or CUDA).
    """
    # Dequantize weight to same dtype as input
    weight_dequant = dequantize(weight_int8, scale, dtype=x.dtype, dim=0)

    # Standard matmul
    y = torch.nn.functional.linear(x, weight_dequant, bias)

    return y


# Triton kernels (only defined when Triton is available)
if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _int8_gemm_kernel(
        # Pointers
        A,           # Activation pointer (FP16): [M, K]
        B,           # Weight pointer (INT8): [K, N]
        C,           # Output pointer (FP16): [M, N]
        Scale,       # Scale pointer (FP32): [N] (per-output-channel)
        # Matrix dimensions
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Triton kernel for W8A16 GEMM using FP16 tensor cores.

        Computes C = A @ (B * scale) where:
        - A is FP16 activations [M, K]
        - B is INT8 weights [K, N]
        - scale is FP32 per-channel scales [N]
        - C is FP16 output [M, N]
        """
        # Program ID for this tile
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Compute offsets for this tile
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # Pointers for A and B tiles
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # Load scale for this output tile (per-channel, so indexed by N)
        scale = tl.load(Scale + offs_n, mask=offs_n < N, other=1.0)

        # Initialize accumulator in FP32 (tensor cores accumulate in FP32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Main loop over K dimension
        for k in range(0, K, BLOCK_K):
            k_offs = k + offs_k

            # Load A tile (FP16)
            a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # Load B tile (INT8) and dequantize to FP16 in registers
            b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
            b_int8 = tl.load(b_ptrs, mask=b_mask, other=0)
            b_fp16 = b_int8.to(tl.float16)

            # FP16 matmul using tensor cores with FP32 accumulation
            acc += tl.dot(a, b_fp16, out_dtype=tl.float32)

            # Advance pointers
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # Apply scale (per-output-channel)
        acc = acc * scale[None, :]

        # Write output tile (convert to FP16)
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

    def _int8_gemm_triton(
        x: torch.Tensor,
        weight_int8: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        weight_transposed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Triton implementation of W8A16 GEMM (requires CUDA + Triton)."""
        original_shape = x.shape
        K = x.shape[-1]
        N = weight_int8.shape[0]

        # Reshape x to 2D for matmul
        x_2d = x.view(-1, K).contiguous()
        M = x_2d.shape[0]

        # Use pre-transposed weight if available
        if weight_transposed is not None:
            weight_t = weight_transposed
        else:
            weight_t = weight_int8.t().contiguous()

        y = torch.empty((M, N), device=x.device, dtype=torch.float16)

        # Grid dimensions
        def grid(META):
            return (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

        # Launch kernel
        _int8_gemm_kernel[grid](
            x_2d,
            weight_t,
            y,
            scale,
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            y.stride(0), y.stride(1),
        )

        # Add bias if present
        if bias is not None:
            y = y + bias

        # Reshape to original batch dimensions
        output_shape = original_shape[:-1] + (N,)
        return y.view(output_shape)


def int8_gemm(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    weight_transposed: Optional[torch.Tensor] = None,
    scale_fp16: Optional[torch.Tensor] = None,
    use_cublas: Optional[bool] = None,
) -> torch.Tensor:
    """
    W8A16 GEMM: FP16 activations x INT8 weights.

    Computes: y = x @ (weight_int8 * scale) + bias

    Uses Triton kernel on CUDA when available, otherwise falls back to PyTorch.

    Args:
        x: FP16 activation tensor of shape (..., K).
        weight_int8: INT8 weight tensor of shape (N, K).
        scale: FP32 scale tensor of shape (N,) for per-output-channel dequant.
        bias: Optional FP16 bias of shape (N,).
        weight_transposed: Optional pre-transposed weight (K, N).
        scale_fp16: Optional pre-converted FP16 scale (unused in Triton path).
        use_cublas: Unused, kept for API compatibility.

    Returns:
        FP16 output tensor of shape (..., N).

    Example:
        >>> x = torch.randn(2, 8, 64)
        >>> weight_int8 = torch.randint(-128, 127, (128, 64), dtype=torch.int8)
        >>> scale = torch.ones(128)
        >>> y = int8_gemm(x, weight_int8, scale)
    """
    assert weight_int8.dtype == torch.int8, "Weight must be INT8"

    # Get dimensions
    original_shape = x.shape
    K = x.shape[-1]
    N = weight_int8.shape[0]

    assert weight_int8.shape == (N, K), f"Weight shape mismatch: {weight_int8.shape} vs ({N}, {K})"
    assert scale.shape == (N,), f"Scale shape mismatch: {scale.shape} vs ({N},)"

    # Use Triton kernel if available and on CUDA
    if HAS_TRITON and x.is_cuda and weight_int8.is_cuda and scale.is_cuda:
        return _int8_gemm_triton(x, weight_int8, scale, bias, weight_transposed)

    # Fallback to PyTorch
    return int8_gemm_torch(x, weight_int8, scale, bias)


class Int8Linear(torch.nn.Module):
    """
    Linear layer using INT8 weights with optimized GEMM kernel.

    Uses Triton kernel on CUDA when available, otherwise falls back to PyTorch.

    Args:
        in_features: Input dimension (K).
        out_features: Output dimension (N).
        bias: Whether to include bias.

    Example:
        >>> linear = Int8Linear(64, 128)
        >>> linear.quantize_weights(torch.randn(128, 64))
        >>> y = linear(torch.randn(2, 8, 64))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # INT8 weights: (out_features, in_features)
        self.register_buffer(
            "weight_int8",
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer(
            "scale",
            torch.ones(out_features, dtype=torch.float32)
        )
        # Pre-transposed weight for Triton path
        self.register_buffer(
            "weight_transposed",
            torch.zeros(in_features, out_features, dtype=torch.int8)
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

        self._quantized = False

    def quantize_weights(self, weight: torch.Tensor) -> None:
        """Quantize and store FP16/FP32 weights as INT8."""
        assert weight.shape == (self.out_features, self.in_features)

        # Quantization happens on CPU
        weight_cpu = weight.cpu() if weight.is_cuda else weight
        weight_int8, scale = quantize_weight_per_channel(weight_cpu)

        # Move to same device as buffers and copy
        device = self.weight_int8.device
        self.weight_int8.copy_(weight_int8.to(device))
        self.scale.copy_(scale.to(device))
        self.weight_transposed.copy_(weight_int8.t().contiguous().to(device))
        self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using optimized INT8 GEMM kernel."""
        if not self._quantized:
            raise RuntimeError("Weights not quantized. Call quantize_weights() first.")

        return int8_gemm(
            x, self.weight_int8, self.scale, self.bias,
            weight_transposed=self.weight_transposed,
        )

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear) -> "Int8Linear":
        """Convert nn.Linear to Int8Linear."""
        has_bias = linear.bias is not None
        int8_linear = cls(linear.in_features, linear.out_features, bias=has_bias)

        # Move to same device as input linear, then quantize
        device = linear.weight.device
        int8_linear = int8_linear.to(device)
        int8_linear.quantize_weights(linear.weight.data)

        # Copy bias if present
        if has_bias:
            int8_linear.bias.data.copy_(linear.bias.data.half())

        return int8_linear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quantized={self._quantized}"
        )

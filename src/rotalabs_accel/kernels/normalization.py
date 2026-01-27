"""
Fused RMSNorm kernel implemented in Triton.

RMSNorm (Root Mean Square Layer Normalization) is used in LLaMA, Mistral, and other
modern LLMs as a simpler alternative to LayerNorm. It normalizes by the RMS of the
input without centering (no mean subtraction).

Formula: y = x * rsqrt(mean(x^2) + eps) * weight

Performance characteristics:
- Memory-bound operation (low arithmetic intensity ~1-2 FLOPs/byte)
- Benefits significantly from fusion with adjacent operations
- Uses FP32 accumulation for numerical stability

Reference: https://arxiv.org/abs/1910.07467
"""

import torch
from typing import Optional

# Optional Triton import
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


def rmsnorm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    PyTorch reference implementation of RMSNorm.

    Works on any device (CPU or CUDA).
    """
    # Compute in FP32 for numerical stability
    x_fp32 = x.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * rms).to(x.dtype) * weight


def rmsnorm_residual_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    PyTorch reference implementation of fused RMSNorm + residual.

    Works on any device (CPU or CUDA).
    """
    x_plus_r = (x + residual).float()
    rms = torch.rsqrt(x_plus_r.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_plus_r * rms).to(x.dtype) * weight


# Triton kernels (only defined when Triton is available)
if HAS_TRITON:
    @triton.jit
    def _rmsnorm_kernel(
        X,           # Input tensor pointer
        Y,           # Output tensor pointer
        W,           # Weight tensor pointer
        stride_x,    # Stride for moving between rows in X
        stride_y,    # Stride for moving between rows in Y
        N,           # Hidden dimension (number of elements per row)
        eps,         # Epsilon for numerical stability
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for RMSNorm."""
        row_idx = tl.program_id(0)
        X += row_idx * stride_x
        Y += row_idx * stride_y

        # Compute mean of squares using FP32 accumulation
        mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for offset in range(0, N, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            mean_sq += x * x

        sum_sq = tl.sum(mean_sq, axis=0)
        mean_sq_scalar = sum_sq / N
        rrms = tl.rsqrt(mean_sq_scalar + eps)

        # Apply normalization and weight scaling
        for offset in range(0, N, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            y = x * rrms * w
            tl.store(Y + cols, y, mask=mask)

    @triton.jit
    def _rmsnorm_residual_kernel(
        X, R, Y, W,
        stride_x, stride_r, stride_y,
        N, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for fused RMSNorm with residual addition."""
        row_idx = tl.program_id(0)
        X += row_idx * stride_x
        R += row_idx * stride_r
        Y += row_idx * stride_y

        # Compute mean of squares of (x + residual)
        mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for offset in range(0, N, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(R + cols, mask=mask, other=0.0).to(tl.float32)
            x_plus_r = x + r
            mean_sq += x_plus_r * x_plus_r

        sum_sq = tl.sum(mean_sq, axis=0)
        mean_sq_scalar = sum_sq / N
        rrms = tl.rsqrt(mean_sq_scalar + eps)

        # Apply normalization and weight scaling
        for offset in range(0, N, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(R + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            x_plus_r = x + r
            y = x_plus_r * rrms * w
            tl.store(Y + cols, y, mask=mask)

    def _rmsnorm_triton(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Triton implementation of RMSNorm (requires CUDA + Triton)."""
        original_shape = x.shape
        hidden_dim = x.shape[-1]
        x_flat = x.view(-1, hidden_dim)
        num_rows = x_flat.shape[0]
        y_flat = torch.empty_like(x_flat)

        BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)
        _rmsnorm_kernel[(num_rows,)](
            x_flat, y_flat, weight,
            x_flat.stride(0), y_flat.stride(0),
            hidden_dim, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return y_flat.view(original_shape)

    def _rmsnorm_residual_triton(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Triton implementation of fused RMSNorm + residual."""
        original_shape = x.shape
        hidden_dim = x.shape[-1]
        x_flat = x.view(-1, hidden_dim)
        residual_flat = residual.view(-1, hidden_dim)
        num_rows = x_flat.shape[0]
        y_flat = torch.empty_like(x_flat)

        BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)
        _rmsnorm_residual_kernel[(num_rows,)](
            x_flat, residual_flat, y_flat, weight,
            x_flat.stride(0), residual_flat.stride(0), y_flat.stride(0),
            hidden_dim, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return y_flat.view(original_shape)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Apply RMS normalization to input tensor.

    Uses Triton kernel on CUDA when available, otherwise falls back to PyTorch.

    Args:
        x: Input tensor of shape (..., hidden_dim).
        weight: Learnable scale parameter of shape (hidden_dim,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor of same shape as input.

    Example:
        >>> x = torch.randn(2, 8, 64)
        >>> weight = torch.ones(64)
        >>> y = rmsnorm(x, weight, eps=1e-6)
    """
    assert x.shape[-1] == weight.shape[0], f"Hidden dim mismatch: {x.shape[-1]} vs {weight.shape[0]}"

    # Use Triton kernel if available and on CUDA
    if HAS_TRITON and x.is_cuda and weight.is_cuda:
        return _rmsnorm_triton(x, weight, eps)

    # Fallback to PyTorch
    return rmsnorm_torch(x, weight, eps)


def rmsnorm_residual_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm with residual addition.

    Computes: y = RMSNorm(x + residual) * weight

    Uses Triton kernel on CUDA when available, otherwise falls back to PyTorch.

    Args:
        x: Input tensor of shape (..., hidden_dim).
        residual: Residual tensor of same shape as x.
        weight: Learnable scale parameter of shape (hidden_dim,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor of same shape as input.

    Example:
        >>> x = torch.randn(2, 8, 64)
        >>> residual = torch.randn_like(x)
        >>> weight = torch.ones(64)
        >>> y = rmsnorm_residual_fused(x, residual, weight, eps=1e-6)
    """
    assert x.shape == residual.shape, f"Shape mismatch: x={x.shape}, residual={residual.shape}"
    assert x.shape[-1] == weight.shape[0], f"Hidden dim mismatch: {x.shape[-1]} vs {weight.shape[0]}"

    # Use Triton kernel if available and on CUDA
    if HAS_TRITON and x.is_cuda and residual.is_cuda and weight.is_cuda:
        return _rmsnorm_residual_triton(x, residual, weight, eps)

    # Fallback to PyTorch
    return rmsnorm_residual_torch(x, residual, weight, eps)


class TritonRMSNorm(torch.nn.Module):
    """
    RMSNorm layer with automatic Triton/PyTorch dispatch.

    Drop-in replacement for torch.nn.RMSNorm with identical interface.
    Uses Triton kernel on CUDA when available, otherwise falls back to PyTorch.

    Args:
        hidden_size: Size of the last dimension to normalize over.
        eps: Small constant for numerical stability.

    Example:
        >>> norm = TritonRMSNorm(64)
        >>> x = torch.randn(2, 8, 64)
        >>> y = norm(x)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"

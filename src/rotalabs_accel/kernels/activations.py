"""
Fused SwiGLU activation kernel implemented in Triton.

SwiGLU (Swish-Gated Linear Unit) is used in LLaMA, PaLM, and other modern LLMs.
It's a variant of GLU that uses SiLU (Swish) as the activation function.

Full SwiGLU formula: y = silu(x @ W_gate) * (x @ W_up)

This kernel implements the fused activation part (after the linear projections):
y = silu(gate) * up

Where silu(x) = x * sigmoid(x)

Performance characteristics:
- Memory-bound operation (low arithmetic intensity ~1.3 FLOPs/byte)
- Fusing silu and multiply saves one memory round-trip
- Uses numerically stable sigmoid implementation

Reference: https://arxiv.org/abs/2204.02311 (PaLM paper)
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


def swiglu_torch(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch reference implementation of SwiGLU activation.

    Works on any device (CPU or CUDA).
    """
    return torch.nn.functional.silu(gate) * up


# Triton kernels (only defined when Triton is available)
if HAS_TRITON:
    @triton.jit
    def _swiglu_kernel(
        Gate,        # Gate tensor pointer (after W_gate projection)
        Up,          # Up tensor pointer (after W_up projection)
        Out,         # Output tensor pointer
        n_elements,  # Total number of elements
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel for fused SwiGLU activation.

        Computes: out = silu(gate) * up = gate * sigmoid(gate) * up
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load gate and up values
        gate = tl.load(Gate + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(Up + offsets, mask=mask, other=0.0).to(tl.float32)

        # Compute SiLU(gate) = gate * sigmoid(gate)
        sigmoid_gate = tl.sigmoid(gate)
        silu_gate = gate * sigmoid_gate

        # Compute output: silu(gate) * up
        out = silu_gate * up

        # Store output
        tl.store(Out + offsets, out, mask=mask)

    def _swiglu_triton(
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        """Triton implementation of SwiGLU (requires CUDA + Triton)."""
        original_shape = gate.shape
        gate_flat = gate.view(-1)
        up_flat = up.view(-1)
        n_elements = gate_flat.numel()

        out_flat = torch.empty_like(gate_flat)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        _swiglu_kernel[grid](
            gate_flat,
            up_flat,
            out_flat,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out_flat.view(original_shape)


def swiglu_fused(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """
    Fused SwiGLU activation.

    Computes: y = silu(gate) * up

    Uses Triton kernel on CUDA when available, otherwise falls back to PyTorch.

    Args:
        gate: Gate tensor of shape (...,), result of x @ W_gate projection.
        up: Up tensor of same shape as gate, result of x @ W_up projection.

    Returns:
        Output tensor of same shape as inputs.

    Example:
        >>> gate = torch.randn(2, 8, 64)
        >>> up = torch.randn(2, 8, 64)
        >>> y = swiglu_fused(gate, up)
    """
    assert gate.shape == up.shape, f"Shape mismatch: gate={gate.shape}, up={up.shape}"

    # Use Triton kernel if available and on CUDA
    if HAS_TRITON and gate.is_cuda and up.is_cuda:
        return _swiglu_triton(gate, up)

    # Fallback to PyTorch
    return swiglu_torch(gate, up)


class SwiGLU(torch.nn.Module):
    """
    SwiGLU module with linear projections.

    Implements the full SwiGLU FFN:
        y = (silu(x @ W_gate) * (x @ W_up)) @ W_down

    Uses Triton kernel on CUDA when available, otherwise falls back to PyTorch.

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Intermediate dimension for the FFN.
        bias: Whether to use bias in linear layers.

    Example:
        >>> swiglu = SwiGLU(hidden_size=64, intermediate_size=256)
        >>> x = torch.randn(2, 8, 64)
        >>> y = swiglu(x)  # Shape: (2, 8, 64)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.w_gate = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w_up = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w_down = torch.nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        return self.w_down(swiglu_fused(gate, up))

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"

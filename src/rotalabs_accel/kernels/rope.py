"""
Rotary Position Embeddings (RoPE) kernel implemented in Triton.

RoPE is used in LLaMA, Mistral, Qwen, and most modern LLMs for position encoding.
It applies a rotation to query and key vectors based on their position in the sequence.

Formula:
    For each pair of dimensions (2i, 2i+1):
    q_rot[2i]   = q[2i] * cos(θ) - q[2i+1] * sin(θ)
    q_rot[2i+1] = q[2i] * sin(θ) + q[2i+1] * cos(θ)

    Where θ = position * (base^(-2i/d))

Reference: https://arxiv.org/abs/2104.09864 (RoFormer)
"""

import torch
import math
from typing import Optional, Tuple

# Try to import triton, fall back to pure PyTorch if not available
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


def rope_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of RoPE.

    Works on any device (CPU or CUDA).

    Args:
        q: Query tensor [batch, seq, heads, head_dim]
        k: Key tensor [batch, seq, heads, head_dim]
        cos: Cosine cache [seq, head_dim/2] or broadcastable
        sin: Sine cache [seq, head_dim/2] or broadcastable

    Returns:
        Tuple of (q_rotated, k_rotated).
    """
    # Reshape for rotation: split last dim into pairs
    q_reshape = q.view(*q.shape[:-1], -1, 2)  # [..., head_dim/2, 2]
    k_reshape = k.view(*k.shape[:-1], -1, 2)

    # Expand cos/sin if needed
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim/2]
        sin = sin.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    q_rot = torch.stack([
        q_reshape[..., 0] * cos - q_reshape[..., 1] * sin,
        q_reshape[..., 0] * sin + q_reshape[..., 1] * cos,
    ], dim=-1).flatten(-2)

    k_rot = torch.stack([
        k_reshape[..., 0] * cos - k_reshape[..., 1] * sin,
        k_reshape[..., 0] * sin + k_reshape[..., 1] * cos,
    ], dim=-1).flatten(-2)

    return q_rot, k_rot


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build cosine and sine caches for RoPE.

    Args:
        seq_len: Maximum sequence length.
        head_dim: Dimension of each attention head.
        base: Base for the frequency computation (default: 10000).
        device: Device for the tensors.
        dtype: Data type for the tensors.

    Returns:
        Tuple of (cos_cache, sin_cache), each of shape [seq_len, head_dim/2].

    Example:
        >>> cos, sin = build_rope_cache(2048, 128, device='cuda')
        >>> print(cos.shape)  # torch.Size([2048, 64])
    """
    assert head_dim % 2 == 0, f"Head dim must be even, got {head_dim}"

    # Compute frequencies: theta_i = base^(-2i/d) for i in [0, d/2)
    half_dim = head_dim // 2
    freq_seq = torch.arange(half_dim, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))

    # Compute positions
    positions = torch.arange(seq_len, device=device, dtype=dtype)

    # Outer product: [seq_len, half_dim]
    angles = torch.outer(positions, inv_freq)

    # Compute cos and sin
    cos_cache = torch.cos(angles)
    sin_cache = torch.sin(angles)

    return cos_cache, sin_cache


# Triton implementation (only defined when Triton is available)
if HAS_TRITON:
    @triton.jit
    def _rope_kernel(
        Q, K, COS, SIN, Q_OUT, K_OUT,
        seq_len, head_dim,
        stride_qb, stride_qs, stride_qh, stride_qd,
        stride_kb, stride_ks, stride_kh, stride_kd,
        stride_cos_s, stride_cos_d,
        stride_qob, stride_qos, stride_qoh, stride_qod,
        stride_kob, stride_kos, stride_koh, stride_kod,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for RoPE."""
        pid = tl.program_id(0)
        batch_idx = pid // (seq_len * tl.cdiv(head_dim, BLOCK_SIZE))
        remaining = pid % (seq_len * tl.cdiv(head_dim, BLOCK_SIZE))
        seq_idx = remaining // tl.cdiv(head_dim, BLOCK_SIZE)
        block_idx = remaining % tl.cdiv(head_dim, BLOCK_SIZE)

        head_idx = tl.program_id(1)

        dim_offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        half_dim = head_dim // 2
        mask = dim_offset < head_dim

        q_ptr = Q + batch_idx * stride_qb + seq_idx * stride_qs + head_idx * stride_qh
        k_ptr = K + batch_idx * stride_kb + seq_idx * stride_ks + head_idx * stride_kh

        q = tl.load(q_ptr + dim_offset * stride_qd, mask=mask, other=0.0)
        k = tl.load(k_ptr + dim_offset * stride_kd, mask=mask, other=0.0)

        pair_idx = dim_offset // 2
        cos_ptr = COS + seq_idx * stride_cos_s
        sin_ptr = SIN + seq_idx * stride_cos_s

        cos_val = tl.load(cos_ptr + pair_idx * stride_cos_d, mask=pair_idx < half_dim, other=1.0)
        sin_val = tl.load(sin_ptr + pair_idx * stride_cos_d, mask=pair_idx < half_dim, other=0.0)

        is_even = (dim_offset % 2) == 0
        paired_offset = tl.where(is_even, dim_offset + 1, dim_offset - 1)
        q_paired = tl.load(q_ptr + paired_offset * stride_qd, mask=paired_offset < head_dim, other=0.0)
        k_paired = tl.load(k_ptr + paired_offset * stride_kd, mask=paired_offset < head_dim, other=0.0)

        q_rot = tl.where(is_even,
                         q * cos_val - q_paired * sin_val,
                         q_paired * cos_val + q * sin_val)
        k_rot = tl.where(is_even,
                         k * cos_val - k_paired * sin_val,
                         k_paired * cos_val + k * sin_val)

        q_out_ptr = Q_OUT + batch_idx * stride_qob + seq_idx * stride_qos + head_idx * stride_qoh
        k_out_ptr = K_OUT + batch_idx * stride_kob + seq_idx * stride_kos + head_idx * stride_koh

        tl.store(q_out_ptr + dim_offset * stride_qod, q_rot, mask=mask)
        tl.store(k_out_ptr + dim_offset * stride_kod, k_rot, mask=mask)

    def _rope_triton(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Triton implementation of RoPE (requires CUDA + Triton)."""
        batch, seq_len, num_heads, head_dim = q.shape

        # Normalize cos/sin shape
        if cos.dim() == 2:
            cos = cos[:seq_len]
            sin = sin[:seq_len]
        else:
            cos = cos.squeeze(0).squeeze(1)[:seq_len]
            sin = sin.squeeze(0).squeeze(1)[:seq_len]

        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        BLOCK_SIZE = min(128, head_dim)
        grid = (batch * seq_len * triton.cdiv(head_dim, BLOCK_SIZE), num_heads)

        _rope_kernel[grid](
            q, k, cos, sin, q_out, k_out,
            seq_len, head_dim,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            cos.stride(0), cos.stride(1),
            q_out.stride(0), q_out.stride(1), q_out.stride(2), q_out.stride(3),
            k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return q_out, k_out


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    use_triton: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    Uses Triton kernel on CUDA when available, otherwise falls back to PyTorch.

    Args:
        q: Query tensor [batch, seq, heads, head_dim]
        k: Key tensor [batch, seq, heads, head_dim]
        cos: Cosine cache for positions
        sin: Sine cache for positions
        use_triton: Force Triton (True) or PyTorch (False). None = auto.

    Returns:
        Tuple of (q_rotated, k_rotated) with same shapes as inputs.

    Example:
        >>> q = torch.randn(2, 16, 4, 32)
        >>> k = torch.randn(2, 16, 4, 32)
        >>> cos, sin = build_rope_cache(16, 32)
        >>> q_rot, k_rot = apply_rope(q, k, cos, sin)
    """
    if use_triton is None:
        use_triton = HAS_TRITON and q.is_cuda

    if use_triton and HAS_TRITON:
        return _rope_triton(q, k, cos, sin)
    else:
        return rope_torch(q, k, cos, sin)


class RotaryEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding module.

    Uses Triton kernel on CUDA when available, otherwise falls back to PyTorch.

    Args:
        dim: Dimension of each attention head (head_dim).
        max_seq_len: Maximum sequence length to cache.
        base: Base for frequency computation (default: 10000).

    Example:
        >>> rope = RotaryEmbedding(dim=32, max_seq_len=128)
        >>> q = torch.randn(2, 16, 4, 32)
        >>> k = torch.randn(2, 16, 4, 32)
        >>> q_rot, k_rot = rope(q, k)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Build initial cache
        cos, sin = build_rope_cache(max_seq_len, dim, base)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def _extend_cache(self, seq_len: int) -> None:
        """Extend cache if sequence is longer than current cache."""
        if seq_len <= self.cos_cache.shape[0]:
            return

        new_len = max(seq_len, self.cos_cache.shape[0] * 2)
        cos, sin = build_rope_cache(
            new_len,
            self.dim,
            self.base,
            device=self.cos_cache.device,
            dtype=self.cos_cache.dtype,
        )
        self.cos_cache = cos
        self.sin_cache = sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: Query tensor [batch, seq, heads, head_dim]
            k: Key tensor [batch, seq, heads, head_dim]
            position_ids: Optional position indices [batch, seq].

        Returns:
            Tuple of (q_rotated, k_rotated).
        """
        seq_len = q.shape[1]
        self._extend_cache(seq_len)

        if position_ids is None:
            cos = self.cos_cache[:seq_len]
            sin = self.sin_cache[:seq_len]
        else:
            cos = self.cos_cache[position_ids]
            sin = self.sin_cache[position_ids]

        return apply_rope(q, k, cos, sin)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}"

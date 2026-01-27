"""Test that all package imports work correctly."""

import pytest
import torch


class TestPackageImports:
    """Test main package imports."""

    def test_version(self):
        """Test version is accessible."""
        from rotalabs_accel import __version__
        assert __version__ == "0.1.0"

    def test_import_main_package(self):
        """Test main package imports."""
        import rotalabs_accel
        assert hasattr(rotalabs_accel, "__version__")

    def test_import_rmsnorm(self):
        """Test RMSNorm imports."""
        from rotalabs_accel import rmsnorm, rmsnorm_torch, TritonRMSNorm
        assert callable(rmsnorm)
        assert callable(rmsnorm_torch)
        assert TritonRMSNorm is not None

    def test_import_swiglu(self):
        """Test SwiGLU imports."""
        from rotalabs_accel import swiglu_fused, swiglu_torch, SwiGLU
        assert callable(swiglu_fused)
        assert callable(swiglu_torch)
        assert SwiGLU is not None

    def test_import_rope(self):
        """Test RoPE imports."""
        from rotalabs_accel import apply_rope, rope_torch, build_rope_cache, RotaryEmbedding
        assert callable(apply_rope)
        assert callable(rope_torch)
        assert callable(build_rope_cache)
        assert RotaryEmbedding is not None

    def test_import_gemm(self):
        """Test INT8 GEMM imports."""
        from rotalabs_accel import int8_gemm, int8_gemm_torch, Int8Linear
        assert callable(int8_gemm)
        assert callable(int8_gemm_torch)
        assert Int8Linear is not None

    def test_import_quantization(self):
        """Test quantization imports."""
        from rotalabs_accel import (
            quantize_symmetric,
            dequantize,
            quantize_weight_per_channel,
            calculate_quantization_error,
            QuantizedLinear,
        )
        assert callable(quantize_symmetric)
        assert callable(dequantize)
        assert callable(quantize_weight_per_channel)
        assert callable(calculate_quantization_error)
        assert QuantizedLinear is not None

    def test_import_utils(self):
        """Test utils imports."""
        from rotalabs_accel import (
            get_device,
            is_cuda_available,
            is_triton_available,
            get_device_properties,
        )
        assert callable(get_device)
        assert callable(is_cuda_available)
        assert callable(is_triton_available)
        assert callable(get_device_properties)


class TestKernelFunctionality:
    """Test basic kernel functionality."""

    def test_rmsnorm_torch(self):
        """Test RMSNorm PyTorch fallback."""
        from rotalabs_accel import rmsnorm_torch

        x = torch.randn(2, 8, 16)
        weight = torch.ones(16)
        out = rmsnorm_torch(x, weight)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_swiglu_torch(self):
        """Test SwiGLU PyTorch fallback."""
        from rotalabs_accel import swiglu_torch

        gate = torch.randn(2, 8, 16)
        up = torch.randn(2, 8, 16)
        out = swiglu_torch(gate, up)
        assert out.shape == gate.shape
        assert not torch.isnan(out).any()

    def test_rope_torch(self):
        """Test RoPE PyTorch fallback."""
        from rotalabs_accel import rope_torch, build_rope_cache

        batch, seq_len, heads, dim = 2, 16, 4, 32
        q = torch.randn(batch, seq_len, heads, dim)
        k = torch.randn(batch, seq_len, heads, dim)
        cos, sin = build_rope_cache(seq_len, dim)

        q_out, k_out = rope_torch(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert not torch.isnan(q_out).any()
        assert not torch.isnan(k_out).any()

    def test_int8_gemm_torch(self):
        """Test INT8 GEMM PyTorch fallback."""
        from rotalabs_accel import int8_gemm_torch

        # Create quantized inputs
        x = torch.randn(2, 8, 64)
        weight = torch.randint(-128, 127, (128, 64), dtype=torch.int8)
        scale = torch.ones(128)

        out = int8_gemm_torch(x, weight, scale)
        assert out.shape == (2, 8, 128)
        assert not torch.isnan(out).any()


class TestQuantization:
    """Test quantization utilities."""

    def test_quantize_dequantize(self):
        """Test round-trip quantization."""
        from rotalabs_accel import quantize_symmetric, dequantize

        x = torch.randn(64, 128)
        x_quant, scale = quantize_symmetric(x)
        x_recon = dequantize(x_quant, scale)

        assert x_quant.dtype == torch.int8
        assert x_recon.shape == x.shape
        # Check reconstruction error is reasonable
        error = (x - x_recon).abs().mean()
        assert error < 0.1  # Should be small for randn

    def test_per_channel_quantization(self):
        """Test per-channel weight quantization."""
        from rotalabs_accel import quantize_weight_per_channel

        weight = torch.randn(64, 128)
        w_quant, scale = quantize_weight_per_channel(weight)

        assert w_quant.dtype == torch.int8
        assert w_quant.shape == weight.shape
        assert scale.shape == (64,)

    def test_quantization_error(self):
        """Test quantization error calculation."""
        from rotalabs_accel import calculate_quantization_error, quantize_symmetric

        x = torch.randn(64, 128)
        x_quant, scale = quantize_symmetric(x)
        error = calculate_quantization_error(x, x_quant, scale)

        assert isinstance(error, dict)
        assert "max_abs_error" in error
        assert "mean_abs_error" in error
        assert error["max_abs_error"] >= 0


class TestModules:
    """Test nn.Module wrappers."""

    def test_triton_rmsnorm_module(self):
        """Test TritonRMSNorm module."""
        from rotalabs_accel import TritonRMSNorm

        norm = TritonRMSNorm(hidden_size=64, eps=1e-6)
        x = torch.randn(2, 8, 64)
        out = norm(x)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_swiglu_module(self):
        """Test SwiGLU module."""
        from rotalabs_accel import SwiGLU

        swiglu = SwiGLU(hidden_size=64, intermediate_size=256)
        x = torch.randn(2, 8, 64)
        out = swiglu(x)

        assert out.shape == (2, 8, 64)
        assert not torch.isnan(out).any()

    def test_rotary_embedding_module(self):
        """Test RotaryEmbedding module."""
        from rotalabs_accel import RotaryEmbedding

        rope = RotaryEmbedding(dim=32, max_seq_len=128)
        q = torch.randn(2, 16, 4, 32)
        k = torch.randn(2, 16, 4, 32)

        q_out, k_out = rope(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_int8_linear_module(self):
        """Test Int8Linear module."""
        from rotalabs_accel import Int8Linear

        linear = Int8Linear(in_features=64, out_features=128)
        # Need to quantize weights before forward pass
        linear.quantize_weights(torch.randn(128, 64))
        x = torch.randn(2, 8, 64)
        out = linear(x)

        assert out.shape == (2, 8, 128)
        assert not torch.isnan(out).any()

    def test_quantized_linear_module(self):
        """Test QuantizedLinear module."""
        from rotalabs_accel import QuantizedLinear

        linear = QuantizedLinear(in_features=64, out_features=128)
        # Need to quantize weights before forward pass
        linear.quantize_weights(torch.randn(128, 64))
        x = torch.randn(2, 8, 64)
        out = linear(x)

        assert out.shape == (2, 8, 128)
        assert not torch.isnan(out).any()


class TestDeviceUtils:
    """Test device utilities."""

    def test_get_device_cpu(self):
        """Test get_device returns CPU when specified."""
        from rotalabs_accel import get_device

        device = get_device("cpu")
        assert device.type == "cpu"

    def test_is_cuda_available(self):
        """Test CUDA availability check."""
        from rotalabs_accel import is_cuda_available

        result = is_cuda_available()
        assert isinstance(result, bool)
        assert result == torch.cuda.is_available()

    def test_is_triton_available(self):
        """Test Triton availability check."""
        from rotalabs_accel import is_triton_available

        result = is_triton_available()
        assert isinstance(result, bool)

    def test_get_device_properties_cpu(self):
        """Test device properties for CPU."""
        from rotalabs_accel import get_device_properties

        # Force CPU by checking when CUDA is not available
        props = get_device_properties()
        assert isinstance(props, dict)
        assert "name" in props
        assert "compute_capability" in props
        assert "supports_fp16" in props
        assert "supports_bf16" in props

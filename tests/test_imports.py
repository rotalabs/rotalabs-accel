"""Test that all package imports work correctly."""

import pytest
import torch


class TestPackageImports:
    """Test main package imports."""

    def test_version(self):
        """Test version is accessible."""
        from rotalabs_accel import __version__
        assert __version__ == "0.2.0"

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


class TestSpeculativeImports:
    """Test speculative decoding module imports."""

    def test_import_config_classes(self):
        """Test config class imports."""
        from rotalabs_accel.speculative import (
            DecodingMode,
            SpeculativeConfig,
            EAGLEConfig,
            MedusaConfig,
            TreeSpecConfig,
            GenerationMetrics,
        )
        assert DecodingMode is not None
        assert SpeculativeConfig is not None
        assert EAGLEConfig is not None
        assert MedusaConfig is not None
        assert TreeSpecConfig is not None
        assert GenerationMetrics is not None

    def test_import_sampling(self):
        """Test sampling utilities imports."""
        from rotalabs_accel.speculative import (
            sample_from_logits,
            greedy_sample,
            rejection_sample,
        )
        assert callable(sample_from_logits)
        assert callable(greedy_sample)
        assert callable(rejection_sample)

    def test_import_eagle(self):
        """Test EAGLE imports."""
        from rotalabs_accel.speculative import (
            EAGLEDraftHead,
            EAGLEModel,
            eagle_verify,
            eagle_decode,
            create_eagle_model,
        )
        assert EAGLEDraftHead is not None
        assert EAGLEModel is not None
        assert callable(eagle_verify)
        assert callable(eagle_decode)
        assert callable(create_eagle_model)

    def test_import_medusa(self):
        """Test Medusa imports."""
        from rotalabs_accel.speculative import (
            MedusaHead,
            MedusaModel,
            generate_candidates,
            verify_candidates,
            medusa_decode,
            create_medusa_model,
        )
        assert MedusaHead is not None
        assert MedusaModel is not None
        assert callable(generate_candidates)
        assert callable(verify_candidates)
        assert callable(medusa_decode)
        assert callable(create_medusa_model)

    def test_import_tree(self):
        """Test tree speculation imports."""
        from rotalabs_accel.speculative import (
            TreeNode,
            SpeculationTree,
            verify_tree,
            tree_speculative_decode,
        )
        assert TreeNode is not None
        assert SpeculationTree is not None
        assert callable(verify_tree)
        assert callable(tree_speculative_decode)

    def test_import_kv_compression(self):
        """Test KV compression imports."""
        from rotalabs_accel.speculative import (
            EvictionPolicy,
            KVCacheConfig,
            CacheStatistics,
            QuantizedTensor,
            CompressedKVCache,
        )
        assert EvictionPolicy is not None
        assert KVCacheConfig is not None
        assert CacheStatistics is not None
        assert QuantizedTensor is not None
        assert CompressedKVCache is not None

    def test_import_core(self):
        """Test core speculative decoding imports."""
        from rotalabs_accel.speculative import (
            trim_kv_cache,
            prefill_cache,
            draft_tokens,
            verify_tokens,
            speculative_decode,
        )
        assert callable(trim_kv_cache)
        assert callable(prefill_cache)
        assert callable(draft_tokens)
        assert callable(verify_tokens)
        assert callable(speculative_decode)


class TestSpeculativeConfig:
    """Test speculative config classes."""

    def test_speculative_config_defaults(self):
        """Test SpeculativeConfig default values."""
        from rotalabs_accel.speculative import SpeculativeConfig, DecodingMode

        config = SpeculativeConfig()
        assert config.lookahead_k == 4
        assert config.max_new_tokens == 256
        assert config.temperature == 1.0
        assert config.mode == DecodingMode.GREEDY

    def test_eagle_config(self):
        """Test EAGLEConfig."""
        from rotalabs_accel.speculative import EAGLEConfig

        config = EAGLEConfig(lookahead_k=6, num_draft_layers=2)
        assert config.lookahead_k == 6
        assert config.num_draft_layers == 2

    def test_medusa_config(self):
        """Test MedusaConfig."""
        from rotalabs_accel.speculative import MedusaConfig

        config = MedusaConfig(num_heads=5)
        assert config.num_heads == 5

    def test_tree_spec_config(self):
        """Test TreeSpecConfig."""
        from rotalabs_accel.speculative import TreeSpecConfig

        config = TreeSpecConfig(max_depth=6, branch_factor=3)
        assert config.max_depth == 6
        assert config.branch_factor == 3

    def test_generation_metrics(self):
        """Test GenerationMetrics properties."""
        from rotalabs_accel.speculative import GenerationMetrics

        metrics = GenerationMetrics(
            total_tokens=100,
            accepted_tokens=80,
            total_time_ms=1000.0,
            iterations=10,
        )
        assert metrics.acceptance_rate == 0.8
        assert metrics.tokens_per_second == 80.0


class TestKVCompression:
    """Test KV cache compression functionality."""

    def test_quantized_tensor(self):
        """Test QuantizedTensor quantization."""
        from rotalabs_accel.speculative import QuantizedTensor

        x = torch.randn(2, 4, 8, 16)
        qt = QuantizedTensor(x, bits=8, per_channel=True)

        assert qt.bits == 8
        assert qt.shape == x.shape
        assert qt.data.dtype == torch.int8

        # Dequantize
        x_recon = qt.dequantize()
        assert x_recon.shape == x.shape
        error = (x - x_recon).abs().mean()
        assert error < 0.1

    def test_compressed_kv_cache(self):
        """Test CompressedKVCache basic operations."""
        from rotalabs_accel.speculative import (
            CompressedKVCache,
            KVCacheConfig,
            EvictionPolicy,
        )

        config = KVCacheConfig(
            max_cache_size=64,
            eviction_policy=EvictionPolicy.H2O,
            quantize=False,
        )

        cache = CompressedKVCache(
            config=config,
            num_layers=2,
            num_heads=4,
            head_dim=16,
            device=torch.device("cpu"),
        )

        # Add some key-value pairs
        key = torch.randn(1, 4, 10, 16)
        value = torch.randn(1, 4, 10, 16)

        full_key, full_value = cache.update(0, key, value)

        assert full_key.shape == key.shape
        assert cache.get_cache_length(0) == 10

    def test_eviction_policies(self):
        """Test different eviction policies."""
        from rotalabs_accel.speculative import EvictionPolicy

        assert EvictionPolicy.H2O.value == "h2o"
        assert EvictionPolicy.LRU.value == "lru"
        assert EvictionPolicy.SLIDING.value == "sliding"
        assert EvictionPolicy.ATTENTION.value == "attention"


class TestSampling:
    """Test sampling utilities."""

    def test_sample_from_logits_greedy(self):
        """Test greedy sampling."""
        from rotalabs_accel.speculative import sample_from_logits

        logits = torch.randn(1, 100)
        logits[0, 42] = 100.0  # Make one token highly likely

        token, prob = sample_from_logits(logits, use_sampling=False)
        assert token.item() == 42

    def test_sample_from_logits_temperature(self):
        """Test temperature scaling."""
        from rotalabs_accel.speculative import sample_from_logits

        logits = torch.randn(1, 100)
        token1, _ = sample_from_logits(logits, temperature=0.1, use_sampling=True)
        token2, _ = sample_from_logits(logits, temperature=10.0, use_sampling=True)

        # Both should return valid tokens
        assert 0 <= token1.item() < 100
        assert 0 <= token2.item() < 100

    def test_greedy_sample(self):
        """Test greedy_sample function."""
        from rotalabs_accel.speculative import greedy_sample

        logits = torch.randn(2, 100)
        logits[0, 10] = 100.0
        logits[1, 20] = 100.0

        tokens, probs = greedy_sample(logits)
        assert tokens[0].item() == 10
        assert tokens[1].item() == 20

"""KV-Cache compression for efficient LLM inference.

Implements eviction policies and quantization for reducing KV-cache memory.

References:
    - H2O: Heavy-Hitter Oracle for Efficient Generative Inference
      https://arxiv.org/abs/2306.14048
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class EvictionPolicy(Enum):
    """KV-cache eviction policies."""
    NONE = "none"
    LRU = "lru"
    ATTENTION = "attention"
    H2O = "h2o"
    SLIDING = "sliding"
    RANDOM = "random"


@dataclass
class KVCacheConfig:
    """Configuration for compressed KV-cache."""
    max_cache_size: int = 1024
    eviction_policy: EvictionPolicy = EvictionPolicy.H2O
    quantize: bool = False
    quant_bits: int = 8
    h2o_heavy_ratio: float = 0.5
    h2o_recent_ratio: float = 0.5
    window_size: int = 512
    sink_size: int = 4


@dataclass
class CacheStatistics:
    """Statistics for cache analysis."""
    total_tokens_seen: int = 0
    evictions: int = 0
    memory_bytes: int = 0
    compression_ratio: float = 1.0


class QuantizedTensor:
    """Quantized tensor for KV-cache compression."""

    def __init__(
        self,
        tensor: torch.Tensor,
        bits: int = 8,
        per_channel: bool = True,
    ):
        self.bits = bits
        self.per_channel = per_channel
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device

        if per_channel:
            self.scale = tensor.abs().amax(dim=-1, keepdim=True) / (2 ** (bits - 1) - 1)
            self.scale = self.scale.clamp(min=1e-8)
        else:
            self.scale = tensor.abs().max() / (2 ** (bits - 1) - 1)
            self.scale = max(self.scale.item(), 1e-8)

        if bits == 8:
            self.data = (tensor / self.scale).round().clamp(-128, 127).to(torch.int8)
        elif bits == 4:
            quantized = (tensor / self.scale).round().clamp(-8, 7)
            self.data = quantized.to(torch.int8)
        else:
            raise ValueError(f"Unsupported bits: {bits}")

    def dequantize(self) -> torch.Tensor:
        return self.data.to(self.dtype) * self.scale

    @property
    def memory_bytes(self) -> int:
        data_bytes = self.data.numel() * (1 if self.bits == 8 else 0.5)
        scale_bytes = self.scale.numel() * 4 if isinstance(self.scale, torch.Tensor) else 4
        return int(data_bytes + scale_bytes)


class CompressedKVCache:
    """Compressed KV-cache with eviction and quantization."""

    def __init__(
        self,
        config: KVCacheConfig,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device or torch.device('cpu')
        self.dtype = dtype

        self.key_cache: List[Optional[Union[torch.Tensor, QuantizedTensor]]] = [None] * num_layers
        self.value_cache: List[Optional[Union[torch.Tensor, QuantizedTensor]]] = [None] * num_layers
        self.attention_scores: List[Optional[torch.Tensor]] = [None] * num_layers
        self.access_times: List[Optional[torch.Tensor]] = [None] * num_layers
        self.current_time = 0
        self.stats = CacheStatistics()

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key-value pairs."""
        batch_size, num_heads, new_len, head_dim = key.shape
        self.current_time += 1
        self.stats.total_tokens_seen += new_len

        existing_key = self._get_tensor(self.key_cache[layer_idx])
        existing_value = self._get_tensor(self.value_cache[layer_idx])

        if existing_key is not None:
            full_key = torch.cat([existing_key, key], dim=2)
            full_value = torch.cat([existing_value, value], dim=2)

            if self.access_times[layer_idx] is not None:
                new_times = torch.full((batch_size, new_len), self.current_time,
                                       device=self.device, dtype=torch.long)
                self.access_times[layer_idx] = torch.cat(
                    [self.access_times[layer_idx], new_times], dim=1
                )
        else:
            full_key = key
            full_value = value
            self.access_times[layer_idx] = torch.full(
                (batch_size, new_len), self.current_time, device=self.device, dtype=torch.long
            )

        if attention_weights is not None:
            self._update_attention_scores(layer_idx, attention_weights)

        current_len = full_key.shape[2]
        if current_len > self.config.max_cache_size:
            full_key, full_value = self._evict(
                layer_idx, full_key, full_value,
                self.config.max_cache_size
            )
            self.stats.evictions += current_len - self.config.max_cache_size

        if self.config.quantize:
            self.key_cache[layer_idx] = QuantizedTensor(full_key, bits=self.config.quant_bits)
            self.value_cache[layer_idx] = QuantizedTensor(full_value, bits=self.config.quant_bits)
        else:
            self.key_cache[layer_idx] = full_key
            self.value_cache[layer_idx] = full_value

        self._update_memory_stats()

        return full_key, full_value

    def _get_tensor(self, cached: Optional[Union[torch.Tensor, QuantizedTensor]]) -> Optional[torch.Tensor]:
        if cached is None:
            return None
        if isinstance(cached, QuantizedTensor):
            return cached.dequantize()
        return cached

    def _update_attention_scores(self, layer_idx: int, attention_weights: torch.Tensor):
        scores = attention_weights.sum(dim=(1, 2))

        if self.attention_scores[layer_idx] is not None:
            existing = self.attention_scores[layer_idx]
            if scores.shape[1] > existing.shape[1]:
                padding = scores.shape[1] - existing.shape[1]
                existing = F.pad(existing, (0, padding), value=0)
            self.attention_scores[layer_idx] = existing + scores
        else:
            self.attention_scores[layer_idx] = scores

    def _evict(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        target_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evict tokens to reach target size."""
        policy = self.config.eviction_policy
        batch_size, num_heads, seq_len, head_dim = keys.shape

        if policy == EvictionPolicy.NONE:
            return keys, values

        elif policy == EvictionPolicy.LRU:
            access_times = self.access_times[layer_idx]
            _, keep_indices = access_times[0].topk(target_size, largest=True)
            keep_indices = keep_indices.sort().values

        elif policy == EvictionPolicy.ATTENTION:
            scores = self.attention_scores[layer_idx]
            if scores is None:
                keep_indices = torch.arange(seq_len - target_size, seq_len, device=self.device)
            else:
                _, keep_indices = scores[0].topk(target_size, largest=True)
                keep_indices = keep_indices.sort().values

        elif policy == EvictionPolicy.H2O:
            heavy_count = int(target_size * self.config.h2o_heavy_ratio)
            recent_count = target_size - heavy_count

            recent_indices = torch.arange(seq_len - recent_count, seq_len, device=self.device)

            scores = self.attention_scores[layer_idx]
            if scores is not None and seq_len > recent_count:
                earlier_scores = scores[0, :seq_len - recent_count]
                _, heavy_indices = earlier_scores.topk(
                    min(heavy_count, seq_len - recent_count), largest=True
                )
                keep_indices = torch.cat([heavy_indices.sort().values, recent_indices])
            else:
                keep_indices = recent_indices

        elif policy == EvictionPolicy.SLIDING:
            sink_indices = torch.arange(min(self.config.sink_size, seq_len), device=self.device)
            window_start = max(seq_len - self.config.window_size + self.config.sink_size, self.config.sink_size)
            window_indices = torch.arange(window_start, seq_len, device=self.device)
            keep_indices = torch.cat([sink_indices, window_indices])
            keep_indices = keep_indices[:target_size]

        elif policy == EvictionPolicy.RANDOM:
            perm = torch.randperm(seq_len, device=self.device)
            keep_indices = perm[:target_size].sort().values

        else:
            raise ValueError(f"Unknown eviction policy: {policy}")

        keys = keys[:, :, keep_indices, :]
        values = values[:, :, keep_indices, :]

        if self.attention_scores[layer_idx] is not None:
            self.attention_scores[layer_idx] = self.attention_scores[layer_idx][:, keep_indices]
        if self.access_times[layer_idx] is not None:
            self.access_times[layer_idx] = self.access_times[layer_idx][:, keep_indices]

        return keys, values

    def _update_memory_stats(self):
        total_bytes = 0
        for layer_idx in range(self.num_layers):
            for cache in [self.key_cache[layer_idx], self.value_cache[layer_idx]]:
                if cache is None:
                    continue
                if isinstance(cache, QuantizedTensor):
                    total_bytes += cache.memory_bytes
                else:
                    total_bytes += cache.numel() * cache.element_size()

        self.stats.memory_bytes = total_bytes

        uncompressed = (self.stats.total_tokens_seen * self.num_layers * 2 *
                        self.num_heads * self.head_dim * 2)
        if uncompressed > 0:
            self.stats.compression_ratio = uncompressed / max(total_bytes, 1)

    def get_cache_length(self, layer_idx: int = 0) -> int:
        if self.key_cache[layer_idx] is None:
            return 0
        cache = self.key_cache[layer_idx]
        if isinstance(cache, QuantizedTensor):
            return cache.shape[2]
        return cache.shape[2]

    def clear(self):
        for i in range(self.num_layers):
            self.key_cache[i] = None
            self.value_cache[i] = None
            self.attention_scores[i] = None
            self.access_times[i] = None
        self.current_time = 0
        self.stats = CacheStatistics()

"""Speculative decoding utilities.

Provides EAGLE-style draft heads, Medusa multi-head speculation, tree attention,
and sampling utilities for speculative decoding to accelerate LLM inference.

Components:
    - EAGLE: Feature-aware speculative decoding using target hidden states
    - Medusa: Multi-head speculation with parallel token prediction
    - Tree speculation: Multi-path speculation with tree-structured verification
    - KV compression: Memory-efficient KV cache with eviction and quantization
    - Sampling: Top-k, top-p, and rejection sampling utilities

Note: This module is experimental and under active development.

Author: Rotalabs Research <research@rotalabs.ai>
"""

from rotalabs_accel.speculative.kv_compression import (
    EvictionPolicy,
    KVCacheConfig,
    CacheStatistics,
    QuantizedTensor,
    CompressedKVCache,
)

from rotalabs_accel.speculative.config import (
    DecodingMode,
    SpeculativeConfig,
    EAGLEConfig,
    MedusaConfig,
    TreeSpecConfig,
    GenerationMetrics,
)

from rotalabs_accel.speculative.sampling import (
    sample_from_logits,
    greedy_sample,
    apply_repetition_penalty,
    rejection_sample,
)

from rotalabs_accel.speculative.tree import (
    TreeNode,
    SpeculationTree,
    verify_tree,
    tree_speculative_decode,
)

from rotalabs_accel.speculative.medusa import (
    MedusaHead,
    MedusaModel,
    generate_candidates,
    verify_candidates,
    medusa_decode,
    create_medusa_model,
    save_medusa_heads,
    load_medusa_heads,
)

from rotalabs_accel.speculative.eagle import (
    EAGLEDraftHead,
    EAGLEModel,
    eagle_verify,
    eagle_decode,
    create_eagle_model,
    save_eagle_head,
    load_eagle_head,
)

from rotalabs_accel.speculative.core import (
    trim_kv_cache,
    prefill_cache,
    draft_tokens,
    verify_tokens,
    speculative_decode,
)

__all__ = [
    # KV Cache compression
    "EvictionPolicy",
    "KVCacheConfig",
    "CacheStatistics",
    "QuantizedTensor",
    "CompressedKVCache",
    # Configuration classes
    "DecodingMode",
    "SpeculativeConfig",
    "EAGLEConfig",
    "MedusaConfig",
    "TreeSpecConfig",
    "GenerationMetrics",
    # Sampling utilities
    "sample_from_logits",
    "greedy_sample",
    "apply_repetition_penalty",
    "rejection_sample",
    # Tree speculation
    "TreeNode",
    "SpeculationTree",
    "verify_tree",
    "tree_speculative_decode",
    # Medusa multi-head speculation
    "MedusaHead",
    "MedusaModel",
    "generate_candidates",
    "verify_candidates",
    "medusa_decode",
    "create_medusa_model",
    "save_medusa_heads",
    "load_medusa_heads",
    # EAGLE feature-aware speculation
    "EAGLEDraftHead",
    "EAGLEModel",
    "eagle_verify",
    "eagle_decode",
    "create_eagle_model",
    "save_eagle_head",
    "load_eagle_head",
    # Core speculative decoding
    "trim_kv_cache",
    "prefill_cache",
    "draft_tokens",
    "verify_tokens",
    "speculative_decode",
]

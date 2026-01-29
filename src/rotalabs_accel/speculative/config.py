"""Configuration classes for speculative decoding.

This module provides configuration dataclasses for controlling speculative
decoding behavior, including EAGLE, tree-based speculation, and other strategies.

Author: Rotalabs Research <research@rotalabs.ai>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DecodingMode(Enum):
    """Decoding strategy for token generation.

    Attributes:
        GREEDY: Always select the highest probability token.
        SAMPLING: Sample from the probability distribution.
    """
    GREEDY = "greedy"
    SAMPLING = "sampling"


@dataclass
class SpeculativeConfig:
    """Configuration for standard speculative decoding with a draft model.

    Standard speculative decoding uses a small, fast draft model to generate
    candidate token sequences, which are then verified in parallel by the
    larger target model. Accepted tokens provide speedup proportional to
    the acceptance rate.

    References:
        - Fast Inference from Transformers via Speculative Decoding
          https://arxiv.org/abs/2211.17192

    Attributes:
        lookahead_k: Number of tokens to draft before verification. Higher values
            increase potential speedup but may reduce acceptance rate. Default: 4.
        max_new_tokens: Maximum number of tokens to generate in total. Default: 256.
        temperature: Softmax temperature for sampling (1.0 = no scaling).
            Lower values make the distribution sharper. Default: 1.0.
        top_p: Nucleus sampling threshold. Only tokens with cumulative probability
            up to top_p are considered. Set to 1.0 to disable. Default: 0.9.
        top_k: Top-k sampling. Only the top-k most likely tokens are considered.
            Set to 0 to disable. Default: 50.
        mode: Decoding mode (GREEDY or SAMPLING). Default: GREEDY.
        adaptive_k: If True, dynamically adjust lookahead_k based on acceptance rate.
            Increases k when acceptance is high, decreases when low. Default: False.
        min_k: Minimum lookahead when adaptive_k is enabled. Default: 2.
        max_k: Maximum lookahead when adaptive_k is enabled. Default: 8.
        use_kv_cache: Whether to use KV cache for efficient generation. Default: True.

    Example:
        >>> config = SpeculativeConfig(
        ...     lookahead_k=5,
        ...     max_new_tokens=512,
        ...     temperature=0.8,
        ...     mode=DecodingMode.SAMPLING,
        ...     adaptive_k=True,
        ... )
    """
    lookahead_k: int = 4
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    mode: DecodingMode = DecodingMode.GREEDY
    adaptive_k: bool = False
    min_k: int = 2
    max_k: int = 8
    use_kv_cache: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.lookahead_k < 1:
            raise ValueError(f"lookahead_k must be >= 1, got {self.lookahead_k}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.min_k < 1:
            raise ValueError(f"min_k must be >= 1, got {self.min_k}")
        if self.max_k < self.min_k:
            raise ValueError(
                f"max_k must be >= min_k, got max_k={self.max_k}, min_k={self.min_k}"
            )


@dataclass
class TreeSpecConfig:
    """Configuration for tree-based speculative decoding.

    Tree-based speculation explores multiple candidate sequences simultaneously,
    using a tree structure where each node represents a potential token. This
    increases the probability that at least one path will be accepted, at the
    cost of increased verification complexity.

    Attributes:
        max_depth: Maximum depth of the speculation tree (number of lookahead
            positions). Deeper trees can predict more tokens but require more
            computation. Default: 4.
        branch_factor: Number of candidate tokens to consider at each position.
            Total tree size is approximately branch_factor^max_depth. Default: 2.
        max_candidates: Hard limit on total number of candidate paths to prevent
            exponential blowup. Default: 16.
        max_new_tokens: Maximum number of new tokens to generate. Default: 256.
        temperature: Softmax temperature for selecting branch candidates.
            Lower values make selection more deterministic. Default: 1.0.
        top_p: Nucleus sampling threshold for branch selection. Default: 0.9.

    Example:
        >>> config = TreeSpecConfig(
        ...     max_depth=5,
        ...     branch_factor=3,
        ...     max_candidates=32,
        ... )
    """
    max_depth: int = 4
    branch_factor: int = 2
    max_candidates: int = 16
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if self.branch_factor < 1:
            raise ValueError(f"branch_factor must be >= 1, got {self.branch_factor}")
        if self.max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {self.max_candidates}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


@dataclass
class EAGLEConfig:
    """Configuration for EAGLE speculative decoding.

    EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)
    uses target model hidden states to condition draft predictions,
    achieving higher acceptance rates than standard speculative decoding.

    References:
        - EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
          https://arxiv.org/abs/2401.15077

    Attributes:
        lookahead_k: Number of tokens to draft speculatively. Higher values
            increase potential speedup but may decrease acceptance rate.
            Typical values are 4-8. Default: 5.
        temperature: Sampling temperature for token generation. Lower values
            make generation more deterministic. Default: 1.0.
        top_k: If set, restricts sampling to top-k highest probability tokens.
            Set to None to disable top-k filtering. Default: None.
        top_p: If set, uses nucleus sampling with cumulative probability p.
            Set to None to disable top-p filtering. Default: None.
        mode: Decoding mode - either greedy or sampling. Default: GREEDY.
        max_new_tokens: Maximum number of new tokens to generate. Default: 256.
        min_new_tokens: Minimum number of new tokens before stopping. Default: 0.
        repetition_penalty: Penalty for repeating tokens. Values > 1.0 discourage
            repetition. Default: 1.0 (no penalty).
        use_cache: Whether to use KV cache for efficiency. Default: True.

    Example:
        >>> config = EAGLEConfig(
        ...     lookahead_k=6,
        ...     temperature=0.8,
        ...     mode=DecodingMode.SAMPLING,
        ...     max_new_tokens=512,
        ... )
        >>> output, metrics = eagle_decode(model, tokenizer, prompt, config, device)
    """
    lookahead_k: int = 5
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    mode: DecodingMode = DecodingMode.GREEDY
    max_new_tokens: int = 256
    min_new_tokens: int = 0
    repetition_penalty: float = 1.0
    use_cache: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.lookahead_k < 1:
            raise ValueError(f"lookahead_k must be >= 1, got {self.lookahead_k}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None, got {self.top_k}")
        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1] or None, got {self.top_p}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.min_new_tokens < 0:
            raise ValueError(f"min_new_tokens must be >= 0, got {self.min_new_tokens}")
        if self.repetition_penalty < 0:
            raise ValueError(
                f"repetition_penalty must be >= 0, got {self.repetition_penalty}"
            )


@dataclass
class MedusaConfig:
    """Configuration for Medusa multi-head speculative decoding.

    Medusa uses multiple lightweight prediction heads attached to a base model
    to draft tokens at different future positions simultaneously. This enables
    parallel speculation with tree-structured verification.

    References:
        - Medusa: Simple LLM Inference Acceleration Framework with Multiple
          Decoding Heads
          https://arxiv.org/abs/2401.10774

    Attributes:
        num_heads: Number of Medusa prediction heads. Each head predicts tokens
            at a different future position (head 0 predicts position +1, head 1
            predicts position +2, etc.). Default: 4.
        num_candidates: Number of top-k candidates to consider from each head
            when generating candidate sequences. Higher values increase coverage
            but also increase verification cost. Default: 10.
        max_new_tokens: Maximum number of new tokens to generate. Generation
            stops when this limit is reached or EOS token is generated.
            Default: 256.
        temperature: Sampling temperature for candidate generation. Lower values
            make the distribution sharper (more deterministic), higher values
            increase diversity. Default: 1.0.
        head_num_layers: Number of layers in each Medusa head MLP. Single layer
            (default) is fastest; more layers may improve prediction quality
            at the cost of latency. Default: 1.
        top_p: Nucleus sampling threshold. If set, only candidates within the
            top cumulative probability mass are considered. Default: None
            (disabled).
        use_tree_attention: Whether to use tree-structured attention for
            efficient parallel verification. Default: True.
        tree_max_depth: Maximum depth of the speculation tree. Deeper trees
            can accept more tokens per iteration but increase verification cost.
            Default: 5.

    Example:
        >>> config = MedusaConfig(
        ...     num_heads=4,
        ...     num_candidates=10,
        ...     max_new_tokens=256,
        ...     temperature=0.7,
        ... )
        >>> output, metrics = medusa_decode(model, target, tokenizer, prompt, config, device)
    """

    num_heads: int = 4
    num_candidates: int = 10
    max_new_tokens: int = 256
    temperature: float = 1.0
    head_num_layers: int = 1
    top_p: Optional[float] = None
    use_tree_attention: bool = True
    tree_max_depth: int = 5

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {self.num_heads}")
        if self.num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1, got {self.num_candidates}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.head_num_layers < 1:
            raise ValueError(f"head_num_layers must be >= 1, got {self.head_num_layers}")
        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1] or None, got {self.top_p}")
        if self.tree_max_depth < 1:
            raise ValueError(f"tree_max_depth must be >= 1, got {self.tree_max_depth}")


@dataclass
class GenerationMetrics:
    """Metrics collected during speculative generation.

    Tracks performance statistics including timing, acceptance rates,
    and throughput metrics for analyzing speculation efficiency.

    Attributes:
        iterations: Number of speculation-verification iterations performed.
        total_tokens: Total number of tokens drafted (including rejected).
        accepted_tokens: Number of tokens accepted from drafts.
        draft_time_ms: Total time spent drafting tokens in milliseconds.
        verify_time_ms: Total time spent verifying drafts in milliseconds.
        total_time_ms: Total generation time in milliseconds.

    Properties:
        acceptance_rate: Fraction of drafted tokens that were accepted.
        tokens_per_iteration: Average tokens accepted per iteration.
        tokens_per_second: Generation throughput in tokens/second.
        speedup_ratio: Effective speedup over standard decoding.

    Example:
        >>> output, metrics = eagle_decode(model, tokenizer, prompt, config, device)
        >>> print(f"Acceptance rate: {metrics.acceptance_rate:.1%}")
        >>> print(f"Throughput: {metrics.tokens_per_second:.1f} tok/s")
        >>> print(f"Speedup: {metrics.speedup_ratio:.2f}x")
    """
    iterations: int = 0
    total_tokens: int = 0
    accepted_tokens: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """Calculate the acceptance rate of drafted tokens.

        Returns:
            Float between 0 and 1 representing the fraction of
            drafted tokens that were accepted.
        """
        if self.total_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_tokens

    @property
    def tokens_per_iteration(self) -> float:
        """Calculate average tokens accepted per speculation iteration.

        Returns:
            Average number of tokens accepted per iteration.
        """
        if self.iterations == 0:
            return 0.0
        return self.accepted_tokens / self.iterations

    @property
    def tokens_per_second(self) -> float:
        """Calculate generation throughput in tokens per second.

        Returns:
            Throughput in tokens/second.
        """
        if self.total_time_ms == 0:
            return 0.0
        return self.accepted_tokens / (self.total_time_ms / 1000)

    @property
    def speedup_ratio(self) -> float:
        """Estimate speedup ratio over standard autoregressive decoding.

        The speedup is approximated as tokens_per_iteration, since each
        iteration requires roughly one target model forward pass.

        Returns:
            Estimated speedup factor (e.g., 2.5 means 2.5x faster).
        """
        return self.tokens_per_iteration

    def __str__(self) -> str:
        """Return a formatted string representation of metrics."""
        return (
            f"GenerationMetrics(\n"
            f"  accepted={self.accepted_tokens}/{self.total_tokens} "
            f"({self.acceptance_rate:.1%} acceptance)\n"
            f"  iterations={self.iterations} "
            f"({self.tokens_per_iteration:.2f} tok/iter)\n"
            f"  time={self.total_time_ms:.1f}ms "
            f"({self.tokens_per_second:.1f} tok/s)\n"
            f"  estimated_speedup={self.speedup_ratio:.2f}x\n"
            f")"
        )


__all__ = [
    "DecodingMode",
    "SpeculativeConfig",
    "EAGLEConfig",
    "MedusaConfig",
    "TreeSpecConfig",
    "GenerationMetrics",
]

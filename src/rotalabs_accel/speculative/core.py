"""Core speculative decoding implementation.

Implements the standard speculative decoding algorithm (Leviathan et al., 2023)
with KV-cache support for efficient autoregressive generation.

The algorithm uses a small draft model to speculatively generate K tokens,
which are then verified in parallel by a larger target model. Accepted tokens
are kept, and rejected tokens trigger resampling from an adjusted distribution.

Key features:
- KV-cache management for efficient incremental decoding
- Support for both greedy and sampling-based generation
- Adaptive K selection based on acceptance rates
- Compatible with HuggingFace transformers models

Reference: https://arxiv.org/abs/2211.17192

Author: Subhadip Mitra <research@rotalabs.ai>
"""

from __future__ import annotations

import time
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel
from transformers.cache_utils import DynamicCache

from .config import GenerationMetrics, SpeculativeConfig
from .sampling import rejection_sample, sample_from_logits


KVCache = Union[DynamicCache, Tuple[Tuple[torch.Tensor, ...], ...], None]


def trim_kv_cache(past_key_values: KVCache, keep_length: int) -> KVCache:
    """
    Trim KV cache to a specific sequence length.

    This is necessary when draft tokens are rejected during verification,
    as the cache must be rolled back to match the accepted sequence length.

    Args:
        past_key_values: The KV cache to trim. Can be a DynamicCache instance
            or a tuple of (key, value) tuples for each layer.
        keep_length: Number of positions to keep in the cache.

    Returns:
        Trimmed KV cache of the same type as input.
    """
    if past_key_values is None:
        return None

    if isinstance(past_key_values, DynamicCache):
        if hasattr(past_key_values, 'crop'):
            past_key_values.crop(keep_length)
            return past_key_values
        elif hasattr(past_key_values, 'key_cache'):
            trimmed = DynamicCache()
            for layer_idx in range(len(past_key_values.key_cache)):
                key = past_key_values.key_cache[layer_idx][:, :, :keep_length, :]
                value = past_key_values.value_cache[layer_idx][:, :, :keep_length, :]
                trimmed.update(key, value, layer_idx)
            return trimmed
    else:
        # Legacy tuple format
        trimmed = []
        for layer_kv in past_key_values:
            key, value = layer_kv
            trimmed.append((
                key[:, :, :keep_length, :],
                value[:, :, :keep_length, :]
            ))
        return tuple(trimmed)


@torch.no_grad()
def prefill_cache(
    model: PreTrainedModel,
    input_ids: torch.Tensor
) -> Tuple[torch.Tensor, KVCache]:
    """
    Prefill the KV cache with prompt tokens.

    This performs the initial forward pass to populate the KV cache with
    key/value pairs for all prompt tokens. Subsequent generation can then
    use incremental decoding.

    Args:
        model: The model to use for prefilling.
        input_ids: Shape (1, seq_len) - the prompt tokens.

    Returns:
        Tuple of (logits, past_key_values) where logits has shape
        (1, seq_len, vocab_size) and past_key_values contains the KV cache.
    """
    outputs = model(input_ids, use_cache=True)
    return outputs.logits, outputs.past_key_values


@torch.no_grad()
def draft_tokens(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    past_key_values: KVCache,
    k: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    use_sampling: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, KVCache]:
    """
    Generate K draft tokens using KV-cache.

    Autoregressively generates K tokens from the draft model, using the
    KV cache for efficient incremental decoding.

    Args:
        model: Draft model for speculation.
        input_ids: Shape (1, 1) - last token (cache has history).
        past_key_values: Cached key/value tensors from previous tokens.
        k: Number of tokens to draft.
        temperature: Sampling temperature (higher = more random).
        top_p: Nucleus sampling threshold.
        use_sampling: Whether to sample (True) or use greedy decoding (False).

    Returns:
        Tuple of (draft_tokens, draft_probs, updated_cache) where:
        - draft_tokens: Shape (k,) - the drafted token IDs
        - draft_probs: Shape (k,) - probability of each drafted token
        - updated_cache: KV cache updated with draft tokens
    """
    draft_tokens_list = []
    draft_probs_list = []
    current_input = input_ids

    for _ in range(k):
        outputs = model(
            current_input,
            past_key_values=past_key_values,
            use_cache=True
        )
        next_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        token, prob = sample_from_logits(
            next_logits, temperature, top_p, use_sampling=use_sampling
        )

        draft_tokens_list.append(token)
        draft_probs_list.append(prob)
        current_input = token.unsqueeze(0).unsqueeze(0)

    draft_tokens = torch.cat([t.reshape(1) for t in draft_tokens_list], dim=0)
    draft_probs = torch.cat([p.reshape(1) for p in draft_probs_list], dim=0)

    return draft_tokens, draft_probs, past_key_values


@torch.no_grad()
def verify_tokens(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    past_key_values: KVCache,
    temperature: float = 1.0,
    top_p: float = 0.9,
    use_sampling: bool = True
) -> Tuple[torch.Tensor, int, KVCache]:
    """
    Verify draft tokens using target model with rejection sampling.

    Performs a single forward pass through the target model to verify all
    draft tokens in parallel. Uses rejection sampling to decide which tokens
    to accept, ensuring the output distribution matches what the target model
    would have produced.

    Args:
        model: Target model for verification.
        input_ids: Shape (1, 1) - last token before drafting.
        draft_tokens: Shape (k,) - drafted tokens to verify.
        draft_probs: Shape (k,) - draft model's probability for each token.
        past_key_values: KV cache from target model.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        use_sampling: Whether to use sampling-based verification.

    Returns:
        Tuple of (accepted_tokens, num_accepted, updated_cache) where:
        - accepted_tokens: Shape (n,) - tokens to add to the sequence
        - num_accepted: Number of draft tokens that were accepted
        - updated_cache: KV cache trimmed to match accepted sequence
    """
    k = draft_tokens.shape[0]

    # Single forward pass to verify all draft tokens
    verify_input = torch.cat([input_ids, draft_tokens.unsqueeze(0)], dim=-1)
    outputs = model(
        verify_input,
        past_key_values=past_key_values,
        use_cache=True
    )
    all_logits = outputs.logits
    new_past_key_values = outputs.past_key_values

    # Get target probabilities
    if use_sampling:
        target_probs = F.softmax(all_logits / max(temperature, 1e-8), dim=-1)
    else:
        target_probs = F.softmax(all_logits, dim=-1)

    accepted_tokens = []
    num_accepted = 0

    for i in range(k):
        draft_token = draft_tokens[i]
        p_draft = draft_probs[i]
        p_target = target_probs[0, i, draft_token]

        if use_sampling:
            acceptance_prob = torch.minimum(
                torch.ones_like(p_target),
                p_target / (p_draft + 1e-10)
            )

            if torch.rand(1, device=draft_token.device) < acceptance_prob:
                accepted_tokens.append(draft_token.reshape(1))
                num_accepted += 1
            else:
                # Resample from adjusted distribution
                adjusted_probs = target_probs[0, i, :].clone()
                draft_dist = torch.zeros_like(adjusted_probs)
                draft_dist[draft_token] = p_draft
                adjusted_probs = torch.clamp(adjusted_probs - draft_dist, min=0)
                adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)

                if adjusted_probs.sum() > 1e-8:
                    new_token = torch.multinomial(adjusted_probs, num_samples=1)
                else:
                    new_token = all_logits[0, i, :].argmax().unsqueeze(0)

                accepted_tokens.append(new_token.reshape(1))
                break
        else:
            # Greedy verification
            target_token = all_logits[0, i, :].argmax()
            if target_token == draft_token:
                accepted_tokens.append(draft_token.reshape(1))
                num_accepted += 1
            else:
                accepted_tokens.append(target_token.reshape(1))
                break

    # Bonus token if all accepted
    if num_accepted == k:
        bonus_token, _ = sample_from_logits(
            all_logits[0, k, :].unsqueeze(0),
            temperature, top_p, use_sampling
        )
        accepted_tokens.append(bonus_token.reshape(1))

    if not accepted_tokens:
        new_token, _ = sample_from_logits(
            all_logits[0, 0, :].unsqueeze(0),
            temperature, top_p, use_sampling
        )
        accepted_tokens.append(new_token.reshape(1))

    accepted = torch.cat(accepted_tokens)

    # Trim cache to accepted length
    if new_past_key_values is not None:
        if isinstance(new_past_key_values, DynamicCache):
            current_len = new_past_key_values.get_seq_length()
        else:
            current_len = new_past_key_values[0][0].shape[2]

        tokens_to_remove = (k + 1) - len(accepted)
        new_cache_len = current_len - tokens_to_remove
        new_past_key_values = trim_kv_cache(new_past_key_values, new_cache_len)

    return accepted, num_accepted, new_past_key_values


@torch.no_grad()
def speculative_decode(
    draft_model: PreTrainedModel,
    target_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    config: SpeculativeConfig,
    device: torch.device,
) -> Tuple[str, GenerationMetrics]:
    """
    Generate text using speculative decoding.

    This is the main entry point for speculative decoding. It orchestrates
    the draft-verify loop, managing KV caches for both models and adaptively
    adjusting the speculation depth based on acceptance rates.

    Args:
        draft_model: Small, fast model for speculation.
        target_model: Large, accurate model for verification.
        tokenizer: Tokenizer shared by both models.
        prompt: Input text prompt.
        config: Speculative decoding configuration.
        device: Compute device (cuda, mps, or cpu).

    Returns:
        Tuple of (generated_text, metrics) where:
        - generated_text: The complete generated text including prompt
        - metrics: GenerationMetrics with timing and acceptance statistics

    Example:
        >>> from rotalabs_accel.speculative import speculative_decode, SpeculativeConfig
        >>> config = SpeculativeConfig(lookahead_k=4, max_new_tokens=100)
        >>> text, metrics = speculative_decode(
        ...     draft_model, target_model, tokenizer, "Hello", config, device
        ... )
        >>> print(f"Generated {metrics.accepted_tokens} tokens")
    """
    metrics = GenerationMetrics()
    start_time = time.perf_counter()
    use_sampling = config.mode.value == "sampling"

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Prefill caches
    _, draft_cache = prefill_cache(draft_model, input_ids)
    _, target_cache = prefill_cache(target_model, input_ids)

    all_token_ids = input_ids.clone()
    generated_tokens = 0
    current_k = config.lookahead_k
    recent_rates = []

    while generated_tokens < config.max_new_tokens:
        metrics.iterations += 1
        last_token = all_token_ids[:, -1:]

        # Draft phase
        draft_start = time.perf_counter()
        draft_toks, draft_probs, draft_cache = draft_tokens(
            draft_model, last_token, draft_cache, current_k,
            config.temperature, config.top_p, use_sampling
        )
        metrics.draft_time_ms += (time.perf_counter() - draft_start) * 1000
        metrics.total_tokens += current_k

        # Verify phase
        verify_start = time.perf_counter()
        accepted, num_accepted, target_cache = verify_tokens(
            target_model, last_token, draft_toks, draft_probs, target_cache,
            config.temperature, config.top_p, use_sampling
        )
        metrics.verify_time_ms += (time.perf_counter() - verify_start) * 1000
        metrics.accepted_tokens += len(accepted)

        # Update sequence
        all_token_ids = torch.cat([all_token_ids, accepted.unsqueeze(0)], dim=-1)
        generated_tokens += len(accepted)

        # Adaptive K
        if config.adaptive_k:
            iter_rate = num_accepted / current_k
            recent_rates.append(iter_rate)
            if len(recent_rates) > 5:
                recent_rates.pop(0)

            if len(recent_rates) >= 5:
                avg_rate = sum(recent_rates) / len(recent_rates)
                if avg_rate > 0.8 and current_k < config.max_k:
                    current_k = min(current_k + 1, config.max_k)
                elif avg_rate < 0.5 and current_k > config.min_k:
                    current_k = max(current_k - 1, config.min_k)

        # Sync draft cache
        draft_cache = trim_kv_cache(draft_cache, all_token_ids.shape[1])

        # Check EOS
        if tokenizer.eos_token_id in accepted:
            break

    metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
    generated_text = tokenizer.decode(all_token_ids[0], skip_special_tokens=True)

    return generated_text, metrics

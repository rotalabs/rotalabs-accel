"""Sampling utilities for speculative decoding.

This module provides token sampling functions with support for various
sampling strategies including temperature scaling, top-k, and top-p (nucleus).

Author: Rotalabs Research <research@rotalabs.ai>
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_tokens_to_keep: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample tokens from logits with optional filtering.

    Applies temperature scaling followed by optional top-k and/or top-p
    (nucleus) filtering before sampling from the resulting distribution.

    Args:
        logits: Unnormalized log probabilities of shape (batch_size, vocab_size)
            or (batch_size, seq_len, vocab_size).
        temperature: Temperature for scaling logits. Values < 1.0 make the
            distribution sharper (more deterministic), values > 1.0 make it
            flatter (more random). Default: 1.0.
        top_k: If set, only the top-k highest probability tokens are kept
            for sampling. Set to None to disable. Default: None.
        top_p: If set, uses nucleus sampling where only tokens with cumulative
            probability <= top_p are kept. Set to None to disable. Default: None.
        min_tokens_to_keep: Minimum number of tokens to keep regardless of
            filtering. Ensures at least one token is always available. Default: 1.

    Returns:
        Tuple of:
            - sampled_tokens: Token indices of shape (batch_size,) or
              (batch_size, seq_len) depending on input shape.
            - token_probs: Probabilities of the sampled tokens, same shape
              as sampled_tokens.

    Raises:
        ValueError: If temperature is not positive.

    Example:
        >>> logits = model(input_ids).logits[:, -1, :]  # (batch, vocab)
        >>> tokens, probs = sample_from_logits(
        ...     logits,
        ...     temperature=0.8,
        ...     top_k=50,
        ...     top_p=0.95,
        ... )
        >>> print(f"Sampled token: {tokens[0].item()}, prob: {probs[0].item():.3f}")
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    # Handle both 2D (batch, vocab) and 3D (batch, seq, vocab) inputs
    original_shape = logits.shape
    if logits.dim() == 3:
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
    else:
        batch_size = logits.shape[0]
        seq_len = None

    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        logits = _top_k_filtering(logits, top_k, min_tokens_to_keep)

    # Apply top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        logits = _top_p_filtering(logits, top_p, min_tokens_to_keep)

    # Convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)
    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Get probabilities of sampled tokens
    token_probs = probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

    # Reshape back if input was 3D
    if seq_len is not None:
        sampled_tokens = sampled_tokens.reshape(batch_size, seq_len)
        token_probs = token_probs.reshape(batch_size, seq_len)

    return sampled_tokens, token_probs


def _top_k_filtering(
    logits: torch.Tensor,
    top_k: int,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter logits to keep only the top-k highest values.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size).
        top_k: Number of top tokens to keep.
        min_tokens_to_keep: Minimum tokens to keep regardless of top_k.

    Returns:
        Filtered logits with non-top-k values set to -inf.
    """
    top_k = max(top_k, min_tokens_to_keep)
    top_k = min(top_k, logits.size(-1))  # Can't be larger than vocab

    # Get the top-k values and create a mask
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_value = values[:, -1, None]

    # Set values below the threshold to -inf
    return torch.where(
        logits < min_value,
        torch.full_like(logits, float("-inf")),
        logits,
    )


def _top_p_filtering(
    logits: torch.Tensor,
    top_p: float,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter logits using nucleus (top-p) sampling.

    Keeps the smallest set of tokens whose cumulative probability
    exceeds top_p.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size).
        top_p: Cumulative probability threshold (0 < top_p <= 1).
        min_tokens_to_keep: Minimum tokens to keep regardless of top_p.

    Returns:
        Filtered logits with tokens outside nucleus set to -inf.
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff: first position where cumsum exceeds top_p
    # Shift right by 1 to keep the token that exceeds the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Ensure we keep at least min_tokens_to_keep
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    # Scatter the mask back to original order
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )

    return torch.where(
        indices_to_remove,
        torch.full_like(logits, float("-inf")),
        logits,
    )


def greedy_sample(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedily select the highest probability token.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size) or
            (batch_size, seq_len, vocab_size).

    Returns:
        Tuple of:
            - tokens: Indices of highest probability tokens.
            - probs: Probabilities of selected tokens.

    Example:
        >>> logits = model(input_ids).logits[:, -1, :]
        >>> tokens, probs = greedy_sample(logits)
    """
    probs = F.softmax(logits, dim=-1)
    tokens = logits.argmax(dim=-1)

    # Get probabilities of selected tokens
    if logits.dim() == 3:
        token_probs = probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    else:
        token_probs = probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    return tokens, token_probs


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float = 1.0,
) -> torch.Tensor:
    """Apply repetition penalty to logits based on previous tokens.

    Reduces the probability of tokens that have already appeared in
    the input sequence.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size).
        input_ids: Previous token IDs of shape (batch_size, seq_len).
        penalty: Repetition penalty factor. Values > 1.0 discourage
            repetition, values < 1.0 encourage it. Default: 1.0 (no effect).

    Returns:
        Logits with repetition penalty applied.

    Example:
        >>> logits = model(input_ids).logits[:, -1, :]
        >>> logits = apply_repetition_penalty(logits, input_ids, penalty=1.2)
    """
    if penalty == 1.0:
        return logits

    # Create a copy to avoid modifying the original
    logits = logits.clone()

    for batch_idx in range(logits.size(0)):
        # Get unique tokens in this sequence
        prev_tokens = input_ids[batch_idx].unique()

        for token_id in prev_tokens:
            # If logit is positive, divide by penalty (reduce)
            # If logit is negative, multiply by penalty (reduce further)
            if logits[batch_idx, token_id] > 0:
                logits[batch_idx, token_id] /= penalty
            else:
                logits[batch_idx, token_id] *= penalty

    return logits


def rejection_sample(
    target_prob: torch.Tensor,
    draft_prob: torch.Tensor,
    draft_token: torch.Tensor,
    target_logits: torch.Tensor,
    use_sampling: bool = True,
) -> Tuple[bool, torch.Tensor]:
    """Perform rejection sampling for speculative decoding verification.

    Implements the rejection sampling algorithm from Leviathan et al. (2022).
    For sampling mode, accepts draft tokens with probability min(1, p_target/p_draft).
    On rejection, samples from the adjusted distribution max(0, p_target - p_draft)
    to maintain the exact target distribution.

    For greedy mode, simply checks if the draft token matches the target's argmax.

    Args:
        target_prob: Target model's probability for the draft token. Scalar tensor.
        draft_prob: Draft model's probability for the draft token. Scalar tensor.
        draft_token: The token proposed by the draft model. Scalar tensor.
        target_logits: Full logits from target model for this position. Shape
            (vocab_size,) used for resampling on rejection.
        use_sampling: If True, use stochastic rejection sampling. If False,
            use greedy verification (accept only if argmax matches).

    Returns:
        Tuple of (accepted, token):
            - accepted: Boolean indicating whether the draft token was accepted.
            - token: The final token to use. Either the accepted draft token or
              a newly sampled token from the adjusted distribution.

    Example:
        >>> # Verify a draft token
        >>> target_logits = model(input_ids)[0, -1, :]
        >>> target_probs = F.softmax(target_logits, dim=-1)
        >>> target_prob = target_probs[draft_token]
        >>> accepted, token = rejection_sample(
        ...     target_prob, draft_prob, draft_token, target_logits
        ... )
        >>> if accepted:
        ...     print("Draft token accepted!")

    Note:
        This function operates on single tokens. For batch processing of multiple
        draft positions, call this function in a loop and stop at the first
        rejection (subsequent positions must be resampled).

    References:
        - Leviathan et al. "Fast Inference from Transformers via Speculative
          Decoding" https://arxiv.org/abs/2211.17192
        - Chen et al. "Accelerating Large Language Model Decoding with
          Speculative Sampling" https://arxiv.org/abs/2302.01318
    """
    if not use_sampling:
        # Greedy: accept if argmax matches
        target_token = target_logits.argmax()
        accepted = (target_token == draft_token)
        return accepted.item(), target_token if not accepted else draft_token

    # Rejection sampling: accept with prob min(1, p_target / p_draft)
    acceptance_prob = torch.minimum(
        torch.ones_like(target_prob),
        target_prob / (draft_prob + 1e-10)
    )

    if torch.rand(1, device=draft_token.device) < acceptance_prob:
        return True, draft_token

    # Rejected: sample from adjusted distribution max(0, p_target - p_draft)
    target_probs = F.softmax(target_logits, dim=-1)
    adjusted_probs = target_probs.clone()

    # Subtract draft distribution (concentrated at draft_token)
    draft_dist = torch.zeros_like(adjusted_probs)
    draft_dist[draft_token] = draft_prob

    adjusted_probs = torch.clamp(adjusted_probs - draft_dist, min=0)
    adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)

    if adjusted_probs.sum() > 1e-8:
        new_token = torch.multinomial(adjusted_probs, num_samples=1).squeeze()
    else:
        # Fallback to argmax if adjusted distribution is degenerate
        new_token = target_logits.argmax()

    return False, new_token


__all__ = [
    "sample_from_logits",
    "greedy_sample",
    "apply_repetition_penalty",
    "rejection_sample",
]

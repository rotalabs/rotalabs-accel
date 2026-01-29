"""EAGLE: Feature-aware speculative decoding.

EAGLE uses target model hidden states to condition draft predictions,
achieving higher acceptance rates than standard speculative decoding.

EAGLE differs from traditional speculative decoding by:
1. Using a lightweight draft head instead of a separate draft model
2. Conditioning predictions on target model hidden states
3. Fusing token embeddings with target features for context-aware drafting

This implementation provides:
- EAGLEDraftHead: Lightweight autoregressive head using target features
- EAGLEModel: Combined target model with draft head
- eagle_decode: Full generation loop with speculation and verification

References:
    - EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
      https://arxiv.org/abs/2401.15077

Author: Rotalabs Research <research@rotalabs.ai>
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .config import DecodingMode, EAGLEConfig, GenerationMetrics
from .sampling import sample_from_logits


class EAGLEDraftHead(nn.Module):
    """Lightweight autoregressive head using target model features.

    Unlike separate draft models used in traditional speculative decoding,
    EAGLE uses the target model's hidden states to condition predictions.
    This feature-aware approach significantly improves acceptance rates
    by leveraging the target model's learned representations.

    Architecture:
        1. Token embedding layer (vocab_size -> hidden_size)
        2. Feature fusion layer (concatenate target features + token embeddings)
        3. Transformer encoder layers with causal attention
        4. Layer normalization + LM head projection

    The draft head is designed to be lightweight (1-2 layers) to minimize
    latency while still producing high-quality draft predictions.

    Args:
        hidden_size: Dimension of hidden states (must match target model).
        vocab_size: Size of the vocabulary (must match target model).
        num_layers: Number of transformer encoder layers. More layers may
            improve prediction quality but increase latency. Default: 1.
        num_heads: Number of attention heads. Default: 8.
        intermediate_size: Size of feedforward intermediate layer. If None,
            defaults to 4 * hidden_size. Default: None.

    Attributes:
        hidden_size: Hidden dimension size.
        vocab_size: Vocabulary size.
        token_embedding: Embedding layer for input tokens.
        feature_fusion: Linear layer to fuse target and token features.
        transformer: Transformer encoder stack.
        ln_f: Final layer normalization.
        lm_head: Output projection to vocabulary logits.

    Example:
        >>> draft_head = EAGLEDraftHead(
        ...     hidden_size=4096,
        ...     vocab_size=32000,
        ...     num_layers=1,
        ... )
        >>> target_hidden = model.get_hidden_states(input_ids)[:, -1, :]
        >>> logits = draft_head(target_hidden, input_ids[:, -1:])
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1,
        num_heads: int = 8,
        intermediate_size: Optional[int] = None,
    ) -> None:
        """Initialize the EAGLE draft head.

        Args:
            hidden_size: Dimension of hidden states.
            vocab_size: Size of the vocabulary.
            num_layers: Number of transformer layers. Default: 1.
            num_heads: Number of attention heads. Default: 8.
            intermediate_size: FFN intermediate size. Default: 4 * hidden_size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        # Token embedding for draft sequence
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # Fuse target hidden state with token embeddings
        # Input: [target_features || token_embeddings] -> hidden_size
        self.feature_fusion = nn.Linear(2 * hidden_size, hidden_size)

        # Lightweight transformer for autoregressive prediction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=0.0,  # No dropout for inference
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using standard normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        target_hidden_state: torch.Tensor,
        input_token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for draft token prediction.

        Fuses target model hidden states with token embeddings and processes
        through the transformer to predict next token logits.

        Args:
            target_hidden_state: Hidden state from target model at the last
                position, shape (batch_size, hidden_size).
            input_token_ids: Token IDs for the draft sequence,
                shape (batch_size, seq_len).
            attention_mask: Optional causal attention mask,
                shape (seq_len, seq_len). If None, a causal mask is created.

        Returns:
            Logits for next token prediction,
            shape (batch_size, seq_len, vocab_size).

        Example:
            >>> target_hidden = model.get_hidden_states(input_ids)[:, -1, :]
            >>> draft_ids = torch.tensor([[1, 2, 3]])  # Draft sequence
            >>> logits = draft_head(target_hidden, draft_ids)
            >>> next_token = logits[:, -1, :].argmax(dim=-1)
        """
        batch_size, seq_len = input_token_ids.shape

        # Embed draft tokens
        token_embs = self.token_embedding(input_token_ids)

        # Expand target hidden state to match sequence length
        # This broadcasts the target context to all positions
        target_expanded = target_hidden_state.unsqueeze(1).expand(-1, seq_len, -1)

        # Fuse target features with token embeddings
        fused = torch.cat([target_expanded, token_embs], dim=-1)
        fused = self.feature_fusion(fused)

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=fused.device, dtype=torch.bool),
                diagonal=1,
            )

        # Process through transformer with causal masking
        hidden = self.transformer(fused, mask=attention_mask, is_causal=True)

        # Project to vocabulary
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)

        return logits


class EAGLEModel(nn.Module):
    """EAGLE Model combining target model with feature-aware draft head.

    Wraps a pretrained causal language model and adds a lightweight EAGLE
    draft head for speculative decoding. The draft head uses the target
    model's hidden states to make feature-aware predictions.

    Args:
        target_model: Pretrained HuggingFace causal language model.
        num_draft_layers: Number of transformer layers in draft head.
            More layers improve prediction quality but increase latency.
            Default: 1.

    Attributes:
        target_model: The wrapped target model for verification.
        draft_head: EAGLE draft head for speculation.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> eagle = EAGLEModel(target, num_draft_layers=1)
        >>> eagle = eagle.cuda().eval()
    """

    def __init__(
        self,
        target_model: PreTrainedModel,
        num_draft_layers: int = 1,
    ) -> None:
        """Initialize EAGLE model with target and draft head.

        Args:
            target_model: Pretrained causal language model.
            num_draft_layers: Number of layers for draft head. Default: 1.
        """
        super().__init__()
        self.target_model = target_model
        config = target_model.config

        # Create draft head matching target model dimensions
        self.draft_head = EAGLEDraftHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            num_layers=num_draft_layers,
            num_heads=min(8, config.hidden_size // 64),
        )

    def get_target_hidden_state(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Extract hidden state and logits from target model.

        Runs a forward pass through the target model to get:
        1. The last hidden state (for conditioning draft head)
        2. The output logits (for next token prediction)
        3. Updated KV cache (for efficient subsequent passes)

        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len).
            past_key_values: Optional cached key-values from previous
                forward passes for efficient generation.

        Returns:
            Tuple containing:
                - hidden_state: Last position hidden state,
                  shape (batch_size, hidden_size).
                - logits: Output logits for last position,
                  shape (batch_size, vocab_size).
                - past_key_values: Updated KV cache for next iteration.

        Example:
            >>> hidden, logits, cache = model.get_target_hidden_state(
            ...     input_ids, past_key_values=prev_cache
            ... )
            >>> next_token = logits.argmax(dim=-1)
        """
        outputs = self.target_model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )

        # Extract last position hidden state for draft conditioning
        hidden_state = outputs.hidden_states[-1][:, -1, :]

        # Extract last position logits for verification
        logits = outputs.logits[:, -1, :]

        return hidden_state, logits, outputs.past_key_values

    @torch.no_grad()
    def draft_tokens(
        self,
        target_hidden_state: torch.Tensor,
        start_token: torch.Tensor,
        num_tokens: int,
        temperature: float = 1.0,
        use_sampling: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draft K tokens autoregressively using EAGLE head.

        Uses the target model's hidden state to condition the draft head
        and generates a sequence of speculative tokens.

        Args:
            target_hidden_state: Hidden state from target model,
                shape (batch_size, hidden_size).
            start_token: Starting token(s) for draft sequence,
                shape (batch_size, 1) or (batch_size, seq_len).
            num_tokens: Number of tokens to draft (K in the paper).
            temperature: Sampling temperature. Default: 1.0.
            use_sampling: If True, sample from distribution; if False,
                use greedy decoding. Default: False.

        Returns:
            Tuple containing:
                - draft_tokens: Drafted token IDs,
                  shape (batch_size, num_tokens).
                - draft_probs: Probabilities of drafted tokens,
                  shape (batch_size, num_tokens).

        Example:
            >>> hidden = model.get_target_hidden_state(input_ids)[0]
            >>> draft_ids, draft_probs = model.draft_tokens(
            ...     hidden, input_ids[:, -1:], num_tokens=5
            ... )
        """
        draft_tokens = []
        draft_probs = []
        current_tokens = start_token

        for _ in range(num_tokens):
            # Get logits for next position
            logits = self.draft_head(target_hidden_state, current_tokens)
            next_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            probs = F.softmax(next_logits, dim=-1)

            # Sample or greedy
            if use_sampling:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Get probability of selected token
            token_prob = probs.gather(1, next_token).squeeze(-1)

            draft_tokens.append(next_token.squeeze(-1))
            draft_probs.append(token_prob)

            # Extend sequence for next iteration
            current_tokens = torch.cat([current_tokens, next_token], dim=1)

        # Stack into tensors
        draft_tokens = torch.stack(draft_tokens, dim=1)
        draft_probs = torch.stack(draft_probs, dim=1)

        return draft_tokens, draft_probs


@torch.no_grad()
def eagle_verify(
    target_model: PreTrainedModel,
    input_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    past_key_values: Optional[Tuple],
    temperature: float = 1.0,
    use_sampling: bool = False,
) -> Tuple[torch.Tensor, int, Optional[Tuple]]:
    """Verify draft tokens using rejection sampling.

    Implements the verification step of speculative decoding where
    draft tokens are compared against the target model's distribution.
    Uses rejection sampling to ensure the output distribution matches
    what would be produced by autoregressive decoding.

    The verification process:
    1. Run target model on [last_input_token, draft_tokens]
    2. For each draft token, compare target prob vs draft prob
    3. Accept if target agrees (greedy) or passes rejection sampling
    4. On rejection, resample from adjusted distribution
    5. If all K tokens accepted, get bonus token from position K

    Args:
        target_model: The target model for verification.
        input_ids: Full input sequence, shape (batch_size, seq_len).
        draft_tokens: Drafted token IDs, shape (batch_size, K).
        draft_probs: Draft probabilities, shape (batch_size, K).
        past_key_values: KV cache from previous target model passes.
        temperature: Sampling temperature. Default: 1.0.
        use_sampling: If True, use rejection sampling; if False,
            use strict greedy matching. Default: False.

    Returns:
        Tuple containing:
            - accepted_tokens: Verified tokens to append to sequence,
              shape (num_accepted,) or (num_accepted + 1,) with bonus.
            - num_accepted: Number of draft tokens accepted (0 to K).
            - past_key_values: Updated KV cache.

    Example:
        >>> accepted, n_accepted, cache = eagle_verify(
        ...     target, input_ids, draft_tokens, draft_probs,
        ...     past_key_values, temperature=0.8, use_sampling=True
        ... )
        >>> input_ids = torch.cat([input_ids, accepted.unsqueeze(0)], dim=1)
    """
    batch_size, k = draft_tokens.shape
    device = input_ids.device

    # Verify all draft tokens in one forward pass
    # Input: [last_context_token, draft_1, draft_2, ..., draft_k]
    verify_input = torch.cat([input_ids[:, -1:], draft_tokens], dim=1)

    outputs = target_model(
        verify_input,
        past_key_values=past_key_values,
        use_cache=True,
    )

    # Get target model's predicted distributions
    target_logits = outputs.logits
    if temperature != 1.0:
        target_logits = target_logits / temperature

    target_probs = F.softmax(target_logits, dim=-1)

    # Verify each draft token sequentially
    accepted_tokens = []
    num_accepted = 0

    for i in range(k):
        draft_token = draft_tokens[0, i]
        target_prob = target_probs[0, i, draft_token].item()
        draft_prob = draft_probs[0, i].item()

        # Calculate acceptance probability
        accept_prob = min(1.0, target_prob / max(draft_prob, 1e-10))

        if use_sampling:
            # Rejection sampling: accept with probability min(1, p_target/p_draft)
            accept = torch.rand(1).item() < accept_prob
        else:
            # Greedy: accept only if target model agrees
            target_token = target_logits[0, i].argmax().item()
            accept = target_token == draft_token.item()

        if accept:
            accepted_tokens.append(draft_token)
            num_accepted += 1
        else:
            # Rejection: resample from adjusted distribution
            if use_sampling:
                # Sample from max(0, p_target - p_draft) normalized
                adjusted = torch.clamp(target_probs[0, i] - draft_prob, min=0)
                if adjusted.sum() > 0:
                    adjusted = adjusted / adjusted.sum()
                    resampled = torch.multinomial(adjusted, 1).squeeze()
                else:
                    resampled = target_logits[0, i].argmax()
            else:
                resampled = target_logits[0, i].argmax()

            accepted_tokens.append(resampled)
            break

    # Bonus token: if all K were accepted, get one more from position K
    if num_accepted == k:
        bonus = target_logits[0, k].argmax()
        accepted_tokens.append(bonus)

    accepted = torch.tensor(accepted_tokens, device=device)
    return accepted, num_accepted, outputs.past_key_values


@torch.no_grad()
def eagle_decode(
    eagle_model: EAGLEModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    config: EAGLEConfig,
    device: torch.device,
) -> Tuple[str, GenerationMetrics]:
    """Generate text using EAGLE speculative decoding.

    Main generation loop that alternates between:
    1. Drafting K tokens with the EAGLE head
    2. Verifying drafts with the target model
    3. Accepting verified tokens and updating state

    The speedup comes from accepting multiple tokens per target model
    forward pass when drafts are accurate.

    Args:
        eagle_model: EAGLE model with target and draft head.
        tokenizer: Tokenizer matching the model.
        prompt: Input text prompt.
        config: EAGLE configuration (lookahead_k, temperature, etc.).
        device: Device to run on (cuda, cpu, etc.).

    Returns:
        Tuple containing:
            - generated_text: Full generated text including prompt.
            - metrics: Generation statistics (acceptance rate, timing, etc.).

    Example:
        >>> eagle = create_eagle_model("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> config = EAGLEConfig(lookahead_k=5, max_new_tokens=100)
        >>>
        >>> text, metrics = eagle_decode(
        ...     eagle, tokenizer, "The future of AI is", config, device
        ... )
        >>> print(text)
        >>> print(f"Speedup: {metrics.speedup_ratio:.2f}x")
    """
    metrics = GenerationMetrics()
    start_time = time.perf_counter()
    use_sampling = config.mode == DecodingMode.SAMPLING

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_tokens = 0

    # Initialize with target model forward pass
    target_cache = None
    hidden_state, first_logits, target_cache = eagle_model.get_target_hidden_state(
        input_ids, past_key_values=target_cache
    )

    # Generate first token
    if use_sampling:
        first_token = torch.multinomial(
            F.softmax(first_logits / config.temperature, dim=-1), 1
        )
    else:
        first_token = first_logits.argmax(dim=-1, keepdim=True)

    input_ids = torch.cat([input_ids, first_token], dim=1)
    generated_tokens += 1

    # Main generation loop
    while generated_tokens < config.max_new_tokens:
        metrics.iterations += 1

        # --- Draft Phase ---
        draft_start = time.perf_counter()

        # Get fresh hidden state for draft conditioning
        hidden_state, _, target_cache = eagle_model.get_target_hidden_state(
            input_ids[:, -1:], past_key_values=target_cache
        )

        # Draft K tokens
        draft_tokens, draft_probs = eagle_model.draft_tokens(
            hidden_state,
            input_ids[:, -1:],
            config.lookahead_k,
            config.temperature,
            use_sampling,
        )

        metrics.draft_time_ms += (time.perf_counter() - draft_start) * 1000
        metrics.total_tokens += config.lookahead_k

        # --- Verify Phase ---
        verify_start = time.perf_counter()

        accepted, num_accepted, target_cache = eagle_verify(
            eagle_model.target_model,
            input_ids,
            draft_tokens,
            draft_probs,
            target_cache,
            config.temperature,
            use_sampling,
        )

        metrics.verify_time_ms += (time.perf_counter() - verify_start) * 1000
        metrics.accepted_tokens += len(accepted)

        # Append accepted tokens
        input_ids = torch.cat([input_ids, accepted.unsqueeze(0)], dim=1)
        generated_tokens += len(accepted)

        # Check for EOS
        if tokenizer.eos_token_id in accepted:
            break

    metrics.total_time_ms = (time.perf_counter() - start_time) * 1000

    # Decode final output
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return generated_text, metrics


def create_eagle_model(
    model_name: str,
    num_draft_layers: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> EAGLEModel:
    """Create an EAGLE model from a pretrained HuggingFace model.

    Factory function that loads a pretrained causal language model and
    wraps it with an EAGLE draft head for speculative decoding.

    Args:
        model_name: HuggingFace model name or path (e.g.,
            "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1").
        num_draft_layers: Number of transformer layers in the draft head.
            Default: 1 (fastest inference).
        device: Target device. If None, auto-selects (CUDA > MPS > CPU).
        dtype: Data type for model weights. If None, uses float16 for
            GPU or float32 for CPU.

    Returns:
        EAGLEModel ready for speculative decoding.

    Raises:
        ValueError: If model cannot be loaded.

    Example:
        >>> eagle = create_eagle_model(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     num_draft_layers=1,
        ...     device=torch.device("cuda"),
        ...     dtype=torch.float16,
        ... )
        >>> eagle.eval()

    Note:
        The draft head is initialized with random weights. For best
        performance, train the draft head using `train_eagle_head` or
        load pretrained weights with `load_eagle_head`.
    """
    # Auto-select device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Auto-select dtype
    if dtype is None:
        dtype = torch.float16 if device.type != "cpu" else torch.float32

    # Load target model
    target_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device).eval()

    # Create EAGLE wrapper
    eagle = EAGLEModel(target_model, num_draft_layers=num_draft_layers)
    eagle = eagle.to(device).eval()

    return eagle


def save_eagle_head(eagle_model: EAGLEModel, path: str) -> None:
    """Save trained EAGLE draft head weights to disk.

    Only saves the draft head parameters, not the target model
    (which should be loaded separately from HuggingFace).

    Args:
        eagle_model: EAGLE model with trained draft head.
        path: File path to save weights (typically .pt or .pth).

    Example:
        >>> # After training
        >>> save_eagle_head(eagle_model, "eagle_head_llama7b.pt")
    """
    torch.save(eagle_model.draft_head.state_dict(), path)


def load_eagle_head(eagle_model: EAGLEModel, path: str) -> None:
    """Load trained EAGLE draft head weights from disk.

    Loads draft head parameters saved with `save_eagle_head`.
    The target model architecture must match the saved head.

    Args:
        eagle_model: EAGLE model to load weights into.
        path: File path to load weights from.

    Raises:
        RuntimeError: If saved weights don't match model architecture.

    Example:
        >>> eagle = create_eagle_model("meta-llama/Llama-2-7b-hf")
        >>> load_eagle_head(eagle, "eagle_head_llama7b.pt")
    """
    eagle_model.draft_head.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )


__all__ = [
    "EAGLEDraftHead",
    "EAGLEModel",
    "eagle_verify",
    "eagle_decode",
    "create_eagle_model",
    "save_eagle_head",
    "load_eagle_head",
]

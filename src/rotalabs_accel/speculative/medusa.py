"""Medusa: Multi-head speculative decoding.

Medusa uses multiple prediction heads to draft tokens at different future
positions in parallel, forming a tree of candidates for efficient verification.

Unlike standard speculative decoding which uses a separate draft model, Medusa
attaches lightweight prediction heads directly to the target model's hidden
states. Each head predicts tokens at a different future position:
- Head 0: predicts token at position +1 (same as base model's lm_head)
- Head 1: predicts token at position +2
- Head 2: predicts token at position +3
- etc.

This parallel prediction enables generating multiple draft tokens with a single
forward pass, significantly reducing inference latency.

References:
    - Medusa: Simple LLM Inference Acceleration Framework with Multiple
      Decoding Heads
      https://arxiv.org/abs/2401.10774

Author: Rotalabs Research <research@rotalabs.ai>
"""

from __future__ import annotations

import itertools
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel

from .config import GenerationMetrics, MedusaConfig


class MedusaHead(nn.Module):
    """Single Medusa prediction head for one future position.

    Each Medusa head is a lightweight MLP that predicts the probability
    distribution over the vocabulary for a specific future token position.
    The head takes the base model's hidden states as input.

    Architecture options:
    - Single layer: Direct linear projection (fastest, minimal parameters)
    - Multi-layer: MLP with SiLU activations (potentially better accuracy)

    Args:
        hidden_size: Dimension of the input hidden states from the base model.
        vocab_size: Size of the vocabulary (output dimension).
        num_layers: Number of layers in the head. Default: 1 (linear projection).

    Example:
        >>> head = MedusaHead(hidden_size=4096, vocab_size=32000, num_layers=1)
        >>> hidden_states = torch.randn(1, 128, 4096)
        >>> logits = head(hidden_states)  # Shape: (1, 128, 32000)
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 1):
        """Initialize a Medusa prediction head.

        Args:
            hidden_size: Dimension of the input hidden states.
            vocab_size: Size of the output vocabulary.
            num_layers: Number of layers in the MLP. If 1, uses a single
                linear projection. If > 1, uses an MLP with SiLU activations.
        """
        super().__init__()

        if num_layers == 1:
            # Simple linear projection for speed
            self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        else:
            # Multi-layer MLP with SiLU activation
            layers = []
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_size, vocab_size, bias=False))
            self.head = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits for the future token position.

        Args:
            hidden_states: Hidden states from base model with shape
                (batch_size, seq_len, hidden_size).

        Returns:
            Logits tensor with shape (batch_size, seq_len, vocab_size).
        """
        return self.head(hidden_states)


class MedusaModel(nn.Module):
    """Medusa model with multiple prediction heads for speculative decoding.

    This class wraps a pretrained causal language model and adds multiple
    Medusa prediction heads for parallel token speculation. Each head
    predicts tokens at a different future position.

    The model can be used in two modes:
    1. Inference mode: Generate candidates for speculative decoding
    2. Training mode: Train the Medusa heads while keeping the base model frozen

    Args:
        base_model: A pretrained HuggingFace causal language model.
        num_heads: Number of Medusa heads to attach. Default: 4.
        head_num_layers: Number of layers in each head's MLP. Default: 1.

    Attributes:
        base_model: The underlying pretrained language model.
        medusa_heads: ModuleList containing the Medusa prediction heads.
        num_heads: Number of Medusa heads.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> medusa = MedusaModel(base, num_heads=4)
        >>> base_logits, head_logits, cache = medusa(input_ids)
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        num_heads: int = 4,
        head_num_layers: int = 1,
    ):
        """Initialize the Medusa model.

        Args:
            base_model: Pretrained causal language model from HuggingFace.
            num_heads: Number of Medusa prediction heads. Each head predicts
                tokens at progressively further future positions.
            head_num_layers: Number of MLP layers in each Medusa head.
                1 = linear projection, >1 = MLP with SiLU activations.
        """
        super().__init__()
        self.base_model = base_model
        self.num_heads = num_heads

        config = base_model.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        # Create Medusa heads for each future position
        self.medusa_heads = nn.ModuleList(
            [
                MedusaHead(hidden_size, vocab_size, head_num_layers)
                for _ in range(num_heads)
            ]
        )
        self._init_heads()

    def _init_heads(self) -> None:
        """Initialize Medusa head parameters with small random values.

        Uses normal distribution with std=0.02, which is common for
        transformer initialization and helps with stable training.
        """
        for head in self.medusa_heads:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[Tuple]]:
        """Forward pass returning base model logits and head predictions.

        This method runs the base model to get hidden states, then passes
        those hidden states through each Medusa head to get predictions
        for future token positions.

        Args:
            input_ids: Input token IDs with shape (batch_size, seq_len).
            past_key_values: Optional cached key-value pairs from previous
                forward passes for efficient autoregressive generation.
            use_cache: Whether to return updated key-value cache.

        Returns:
            Tuple containing:
            - base_logits: Logits from base model's lm_head,
              shape (batch_size, seq_len, vocab_size)
            - head_logits: List of logits from each Medusa head,
              each with shape (batch_size, seq_len, vocab_size)
            - past_key_values: Updated KV cache if use_cache=True, else None

        Example:
            >>> base_logits, head_logits, cache = medusa(input_ids)
            >>> # base_logits predicts next token (position +1)
            >>> # head_logits[0] also predicts position +1
            >>> # head_logits[1] predicts position +2
            >>> # etc.
        """
        outputs = self.base_model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        base_logits = outputs.logits

        # Get predictions from each Medusa head
        head_logits = [head(hidden_states) for head in self.medusa_heads]

        return base_logits, head_logits, outputs.past_key_values if use_cache else None


def generate_candidates(
    base_logits: torch.Tensor,
    head_logits: List[torch.Tensor],
    num_candidates: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate candidate token sequences from base and head predictions.

    Combines predictions from the base model and Medusa heads to form
    candidate sequences. Uses top-k sampling from each head's distribution
    and forms candidate paths through the Cartesian product of top tokens.

    The candidates are sorted by their joint probability (product of
    individual token probabilities) in descending order.

    Args:
        base_logits: Logits from base model with shape
            (batch_size, seq_len, vocab_size) or (1, 1, vocab_size).
        head_logits: List of logits from Medusa heads, each with shape
            (batch_size, seq_len, vocab_size).
        num_candidates: Number of top tokens to consider from each head.
            The total number of candidate paths is capped at k^2 or 64.
        temperature: Sampling temperature for softmax. Lower values make
            the distribution sharper. Must be > 0.

    Returns:
        Tuple containing:
        - candidates: Tensor of candidate token sequences with shape
          (num_paths, num_heads + 1), where each row is a candidate path
        - candidate_probs: Tensor of joint probabilities for each candidate,
          shape (num_paths,), sorted in descending order

    Example:
        >>> candidates, probs = generate_candidates(
        ...     base_logits, head_logits, num_candidates=10, temperature=0.8
        ... )
        >>> # candidates[0] is the most probable candidate path
        >>> # probs[0] is its joint probability
    """
    # Extract last position logits
    if base_logits.dim() == 3:
        base_logits_last = base_logits[0, -1, :]
    else:
        base_logits_last = base_logits.squeeze()

    # Apply temperature and softmax
    base_probs = F.softmax(base_logits_last / max(temperature, 1e-8), dim=-1)

    head_probs = []
    for h in head_logits:
        if h.dim() == 3:
            h_last = h[0, -1, :]
        elif h.dim() == 2:
            h_last = h[-1, :]
        else:
            h_last = h.squeeze()
        head_probs.append(F.softmax(h_last / max(temperature, 1e-8), dim=-1))

    # Get top-k candidates from each position
    vocab_size = base_probs.shape[-1]
    k = min(num_candidates, vocab_size)

    base_topk = torch.topk(base_probs, k)
    head_topks = [torch.topk(p, k) for p in head_probs]

    # Extract indices and probabilities
    base_indices = base_topk.indices.flatten()
    base_values = base_topk.values.flatten()
    head_indices = [h.indices.flatten() for h in head_topks]
    head_values = [h.values.flatten() for h in head_topks]

    all_indices = [base_indices] + head_indices
    all_probs = [base_values] + head_values

    # Generate candidate paths using Cartesian product
    candidates = []
    candidate_probs = []
    max_paths = min(k**2, 64)  # Cap total candidates for efficiency

    for i, combo in enumerate(
        itertools.product(*[range(min(k, len(idx))) for idx in all_indices])
    ):
        if i >= max_paths:
            break

        # Extract token IDs for this path
        token_combo = [int(all_indices[pos][j].item()) for pos, j in enumerate(combo)]
        candidates.append(token_combo)

        # Compute joint probability (product of individual probs)
        prob = 1.0
        for pos, j in enumerate(combo):
            prob *= float(all_probs[pos][j].item())
        candidate_probs.append(prob)

    # Fallback if no candidates were generated
    if not candidates:
        candidates = [
            [int(base_indices[0].item())]
            + [int(idx[0].item()) for idx in head_indices]
        ]
        candidate_probs = [1.0]

    # Convert to tensors and sort by probability
    candidates = torch.tensor(candidates, device=base_logits.device)
    candidate_probs = torch.tensor(candidate_probs, device=base_logits.device)

    sorted_indices = candidate_probs.argsort(descending=True)
    candidates = candidates[sorted_indices]
    candidate_probs = candidate_probs[sorted_indices]

    return candidates, candidate_probs


@torch.no_grad()
def verify_candidates(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    candidates: torch.Tensor,
    past_key_values: Optional[Tuple] = None,
) -> Tuple[torch.Tensor, int, Optional[Tuple]]:
    """Verify candidate sequences with the target model.

    Runs the target model on the top candidate sequence and checks how many
    tokens match. For each position, if the model's prediction matches the
    candidate token, it's accepted. The first mismatch causes verification
    to stop, and the model's predicted token is used instead.

    If all candidate tokens are accepted, a bonus token is generated from
    the model's prediction at the final position.

    Args:
        model: The target language model for verification.
        input_ids: Current input token IDs with shape (1, seq_len).
        candidates: Candidate token sequences with shape (num_paths, depth).
        past_key_values: Optional KV cache from previous forward passes.

    Returns:
        Tuple containing:
        - accepted: Tensor of accepted tokens, may include bonus token,
          shape (num_accepted,) or (num_accepted + 1,)
        - num_accepted: Number of candidate tokens that were accepted
          (excludes bonus token)
        - past_key_values: Updated KV cache after verification

    Example:
        >>> accepted, num_accepted, cache = verify_candidates(
        ...     model, input_ids, candidates, past_key_values
        ... )
        >>> # If num_accepted == depth, all draft tokens matched
        >>> # accepted may have length depth+1 due to bonus token
    """
    num_paths, depth = candidates.shape
    device = input_ids.device

    # Use the most probable candidate (first one after sorting)
    top_candidate = candidates[0]

    # Concatenate candidate to input for verification
    verify_input = torch.cat([input_ids, top_candidate.unsqueeze(0)], dim=-1)

    # Run target model on concatenated sequence
    outputs = model(verify_input, past_key_values=past_key_values, use_cache=True)
    logits = outputs.logits

    # Check how many tokens match
    num_accepted = 0
    accepted_tokens = []
    verify_start = input_ids.shape[1] - 1

    for i in range(depth):
        # Get target model's prediction at this position
        target_token = logits[0, verify_start + i].argmax()

        if target_token == top_candidate[i]:
            # Token matches - accept it
            accepted_tokens.append(top_candidate[i])
            num_accepted += 1
        else:
            # Token doesn't match - use model's prediction and stop
            accepted_tokens.append(target_token)
            break

    # If all tokens accepted, add bonus token from final position
    if num_accepted == depth:
        bonus_token = logits[0, verify_start + depth].argmax()
        accepted_tokens.append(bonus_token)

    # Convert to tensor
    accepted = torch.stack(accepted_tokens) if accepted_tokens else top_candidate[:1]

    return accepted, num_accepted, outputs.past_key_values


@torch.no_grad()
def medusa_decode(
    medusa_model: MedusaModel,
    target_model: PreTrainedModel,
    tokenizer,
    prompt: str,
    config: MedusaConfig,
    device: torch.device,
) -> Tuple[str, GenerationMetrics]:
    """Generate text using Medusa multi-head speculative decoding.

    This function implements the full Medusa decoding loop:
    1. Run Medusa model to get base + head predictions
    2. Generate candidate token sequences
    3. Verify candidates with target model
    4. Accept matched tokens and repeat

    The Medusa model and target model can be the same model (with Medusa heads
    attached) or different models for more flexible configurations.

    Args:
        medusa_model: MedusaModel instance with attached prediction heads.
        target_model: Target model for verification. Can be the same as
            medusa_model.base_model or a different model.
        tokenizer: Tokenizer compatible with the models.
        prompt: Input text prompt to continue.
        config: MedusaConfig with generation parameters.
        device: Device to run inference on (cuda/cpu).

    Returns:
        Tuple containing:
        - generated_text: Complete generated text including the prompt
        - metrics: GenerationMetrics with timing and acceptance statistics

    Example:
        >>> medusa = create_medusa_model("meta-llama/Llama-2-7b-hf", num_heads=4)
        >>> config = MedusaConfig(max_new_tokens=128, temperature=0.8)
        >>> text, metrics = medusa_decode(
        ...     medusa, medusa.base_model, tokenizer, "Once upon a time",
        ...     config, torch.device("cuda")
        ... )
        >>> print(f"Generated: {text}")
        >>> print(f"Speedup: {metrics.speedup_ratio:.2f}x")
    """
    metrics = GenerationMetrics()
    start_time = time.perf_counter()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    all_token_ids = input_ids.clone()
    generated_tokens = 0

    # Initialize caches
    medusa_cache = None
    target_cache = None

    while generated_tokens < config.max_new_tokens:
        metrics.iterations += 1

        # Draft phase: run Medusa model
        draft_start = time.perf_counter()
        base_logits, head_logits, medusa_cache = medusa_model(
            all_token_ids[:, -1:] if medusa_cache else all_token_ids,
            past_key_values=medusa_cache,
            use_cache=True,
        )

        # Generate candidate sequences
        candidates, probs = generate_candidates(
            base_logits[:, -1:, :],
            [h[:, -1:, :] for h in head_logits],
            config.num_candidates,
            config.temperature,
        )
        metrics.draft_time_ms += (time.perf_counter() - draft_start) * 1000
        metrics.total_tokens += candidates.shape[1]

        # Verify phase: check candidates with target model
        verify_start = time.perf_counter()
        accepted, num_accepted, target_cache = verify_candidates(
            target_model, all_token_ids, candidates, target_cache
        )
        metrics.verify_time_ms += (time.perf_counter() - verify_start) * 1000
        metrics.accepted_tokens += len(accepted)

        # Update sequence with accepted tokens
        all_token_ids = torch.cat([all_token_ids, accepted.unsqueeze(0)], dim=-1)
        generated_tokens += len(accepted)

        # Check for EOS token
        if tokenizer.eos_token_id in accepted:
            break

    metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
    generated_text = tokenizer.decode(all_token_ids[0], skip_special_tokens=True)

    return generated_text, metrics


def create_medusa_model(
    base_model_name: str,
    num_heads: int = 4,
    head_num_layers: int = 1,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> MedusaModel:
    """Create a Medusa model from a pretrained base model.

    Convenience function to load a pretrained model and wrap it with
    Medusa prediction heads in a single call.

    Args:
        base_model_name: HuggingFace model identifier or local path.
        num_heads: Number of Medusa heads to attach. Default: 4.
        head_num_layers: Number of layers in each head. Default: 1.
        device: Device to load the model on. Auto-detects CUDA if available.
        dtype: Data type for model weights. Auto-selects fp16 for CUDA,
            fp32 for CPU.

    Returns:
        MedusaModel instance ready for inference or training.

    Example:
        >>> medusa = create_medusa_model(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     num_heads=4,
        ...     device=torch.device("cuda"),
        ...     dtype=torch.float16,
        ... )
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is None:
        dtype = torch.float16 if device.type != "cpu" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=dtype
    ).to(device).eval()

    medusa = MedusaModel(base_model, num_heads=num_heads, head_num_layers=head_num_layers)
    medusa = medusa.to(device).eval()

    return medusa


def save_medusa_heads(medusa_model: MedusaModel, path: str) -> None:
    """Save trained Medusa heads to a file.

    Only saves the Medusa head parameters, not the base model weights.
    This allows sharing trained heads independently of the base model.

    Args:
        medusa_model: MedusaModel with trained heads to save.
        path: File path to save the checkpoint (typically .pt or .pth).

    Example:
        >>> # After training
        >>> save_medusa_heads(medusa_model, "medusa_heads_llama7b.pt")
    """
    torch.save(
        {
            "num_heads": medusa_model.num_heads,
            "head_state_dicts": [
                head.state_dict() for head in medusa_model.medusa_heads
            ],
        },
        path,
    )


def load_medusa_heads(medusa_model: MedusaModel, path: str) -> None:
    """Load trained Medusa heads from a file.

    Loads pre-trained head weights into an existing MedusaModel.
    The number of heads in the checkpoint must match the model.

    Args:
        medusa_model: MedusaModel to load heads into.
        path: Path to the saved checkpoint file.

    Raises:
        AssertionError: If the number of heads doesn't match.

    Example:
        >>> medusa = create_medusa_model("meta-llama/Llama-2-7b-hf", num_heads=4)
        >>> load_medusa_heads(medusa, "medusa_heads_llama7b.pt")
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert (
        checkpoint["num_heads"] == medusa_model.num_heads
    ), f"Head count mismatch: checkpoint has {checkpoint['num_heads']}, model has {medusa_model.num_heads}"

    for head, state_dict in zip(
        medusa_model.medusa_heads, checkpoint["head_state_dicts"]
    ):
        head.load_state_dict(state_dict)


__all__ = [
    "MedusaHead",
    "MedusaModel",
    "generate_candidates",
    "verify_candidates",
    "medusa_decode",
    "create_medusa_model",
    "save_medusa_heads",
    "load_medusa_heads",
]

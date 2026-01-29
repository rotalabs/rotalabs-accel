"""Tree-based speculative decoding.

Generates a tree of candidate sequences instead of a single sequence,
increasing the probability of acceptance.

References:
    - SpecInfer: Tree-based Speculative Decoding
      https://arxiv.org/abs/2305.09781

Author: research@rotalabs.ai
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoTokenizer
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
import time

from .config import TreeSpecConfig, GenerationMetrics


@dataclass
class TreeNode:
    """Node in the speculation tree."""
    token_id: int
    prob: float
    depth: int
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    position_in_sequence: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_path_to_root(self) -> List[int]:
        path = []
        node = self
        while node is not None:
            path.append(node.token_id)
            node = node.parent
        return list(reversed(path))


class SpeculationTree:
    """Manages tree structure for speculative decoding."""

    def __init__(self, config: TreeSpecConfig):
        self.config = config
        self.root: Optional[TreeNode] = None
        self.nodes: List[TreeNode] = []
        self.leaves: List[TreeNode] = []

    def build_tree(
        self,
        draft_model: PreTrainedModel,
        input_ids: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build speculation tree using draft model."""
        self.nodes = []
        self.leaves = []

        self.root = TreeNode(
            token_id=input_ids[0, -1].item(),
            prob=1.0,
            depth=0,
            position_in_sequence=0
        )
        self.nodes.append(self.root)

        current_level = [self.root]
        position = 1

        for depth in range(1, self.config.max_depth + 1):
            if len(self.nodes) >= self.config.max_candidates:
                break

            next_level = []

            for parent in current_level:
                if len(self.nodes) >= self.config.max_candidates:
                    break

                path_tokens = parent.get_path_to_root()
                path_tensor = torch.tensor([path_tokens], device=device)

                with torch.no_grad():
                    outputs = draft_model(path_tensor, use_cache=False)
                    logits = outputs.logits[0, -1, :]

                probs = F.softmax(logits / max(self.config.temperature, 1e-8), dim=-1)
                top_probs, top_indices = torch.topk(probs, self.config.branch_factor)

                for i in range(self.config.branch_factor):
                    if len(self.nodes) >= self.config.max_candidates:
                        break

                    child = TreeNode(
                        token_id=top_indices[i].item(),
                        prob=top_probs[i].item(),
                        depth=depth,
                        parent=parent,
                        position_in_sequence=position
                    )
                    parent.children.append(child)
                    self.nodes.append(child)
                    next_level.append(child)
                    position += 1

            current_level = next_level
            if not current_level:
                break

        self.leaves = [n for n in self.nodes if n.is_leaf()]
        return self._flatten_tree(device)

    def _flatten_tree(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten tree into tensors for batched verification."""
        num_nodes = len(self.nodes)

        tokens = torch.tensor([n.token_id for n in self.nodes], dtype=torch.long, device=device)
        probs = torch.tensor([n.prob for n in self.nodes], dtype=torch.float, device=device)

        attention_mask = torch.zeros(num_nodes, num_nodes, device=device)

        for i, node in enumerate(self.nodes):
            attention_mask[i, i] = 1
            ancestor = node.parent
            while ancestor is not None:
                ancestor_idx = self.nodes.index(ancestor)
                attention_mask[i, ancestor_idx] = 1
                ancestor = ancestor.parent

        return tokens, probs, attention_mask

    def get_paths(self) -> List[List[TreeNode]]:
        """Get all root-to-leaf paths."""
        paths = []
        for leaf in self.leaves:
            path = []
            node = leaf
            while node is not None:
                path.append(node)
                node = node.parent
            paths.append(list(reversed(path)))
        return paths


@torch.no_grad()
def verify_tree(
    target_model: PreTrainedModel,
    input_ids: torch.Tensor,
    tree: SpeculationTree,
    candidate_tokens: torch.Tensor,
    candidate_probs: torch.Tensor,
    temperature: float = 1.0,
    device: torch.device = None
) -> Tuple[List[int], int]:
    """Verify all tree paths with target model."""
    tree_tokens = candidate_tokens[1:] if len(candidate_tokens) > 1 else candidate_tokens

    if len(tree_tokens) == 0:
        return [], 0

    verify_input = torch.cat([input_ids[:, -1:], tree_tokens.unsqueeze(0)], dim=-1)
    outputs = target_model(verify_input, use_cache=False)
    all_logits = outputs.logits[0]

    target_probs = F.softmax(all_logits / max(temperature, 1e-8), dim=-1)

    best_path = []
    best_length = 0

    for path in tree.get_paths():
        accepted = []

        for i, node in enumerate(path[1:], start=0):
            if i >= len(target_probs) - 1:
                break

            p_target = target_probs[i, node.token_id].item()
            p_draft = node.prob

            accept_prob = min(1.0, p_target / (p_draft + 1e-10))

            if torch.rand(1).item() < accept_prob:
                accepted.append(node.token_id)
            else:
                adjusted = target_probs[i].clone()
                adjusted[node.token_id] = max(0, adjusted[node.token_id] - p_draft)
                adjusted = adjusted / (adjusted.sum() + 1e-10)

                if adjusted.sum() > 1e-8:
                    new_token = torch.multinomial(adjusted, 1).item()
                else:
                    new_token = all_logits[i].argmax().item()

                accepted.append(new_token)
                break

        if len(accepted) > best_length:
            best_length = len(accepted)
            best_path = accepted

    if best_length == tree.config.max_depth and len(target_probs) > best_length:
        bonus_probs = target_probs[best_length]
        bonus_token = torch.multinomial(bonus_probs, 1).item()
        best_path.append(bonus_token)

    return best_path, best_length


@torch.no_grad()
def tree_speculative_decode(
    draft_model: PreTrainedModel,
    target_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    config: TreeSpecConfig,
    device: torch.device
) -> Tuple[str, GenerationMetrics]:
    """Generate text using tree-based speculative decoding."""
    metrics = GenerationMetrics()
    start_time = time.perf_counter()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    all_token_ids = input_ids.clone()

    generated_tokens = 0
    tree = SpeculationTree(config)

    while generated_tokens < config.max_new_tokens:
        metrics.iterations += 1

        candidate_tokens, candidate_probs, tree_mask = tree.build_tree(
            draft_model, all_token_ids, device
        )
        metrics.total_tokens += len(candidate_tokens)

        accepted_tokens, num_accepted = verify_tree(
            target_model, all_token_ids, tree,
            candidate_tokens, candidate_probs,
            config.temperature, device
        )

        if not accepted_tokens:
            with torch.no_grad():
                outputs = target_model(all_token_ids, use_cache=False)
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits / config.temperature, dim=-1)
                token = torch.multinomial(probs, 1).item()
                accepted_tokens = [token]

        new_tokens = torch.tensor([accepted_tokens], device=device)
        all_token_ids = torch.cat([all_token_ids, new_tokens], dim=-1)
        generated_tokens += len(accepted_tokens)
        metrics.accepted_tokens += len(accepted_tokens)

        if tokenizer.eos_token_id in accepted_tokens:
            break

    metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
    generated_text = tokenizer.decode(all_token_ids[0], skip_special_tokens=True)

    return generated_text, metrics

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

def gumbel_noise_like(x: Tensor, eps: float = 1e-8) -> Tensor:
    u = torch.rand_like(x)
    return -torch.log(-torch.log(u + eps) + eps)

@torch.no_grad()
def select_top_k_pairs_gpu(energy: Tensor, k: int, max_features: Optional[int] = None, pairing_strategy: str = "consecutive") -> Tensor:
    """Top-K pair selection (fully GPU-accelerated, no CPU transfers).

    Two pairing strategies:
    - "consecutive": Selects pairs among high-energy indices, then greedily keeps disjoint pairs
      according to pair score energy[i]*energy[j]. (Default, original behavior)
    - "high_low": Pairs top-i with bottom-i dimensions to enable energy redistribution
      between high and low energy dimensions.

    Parameters
    ----------
    energy : Tensor
        Energy values for each dimension
    k : int
        Number of pairs to select
    max_features : Optional[int]
        Maximum number of features to consider
    pairing_strategy : str
        Pairing strategy: "consecutive" or "high_low"

    Returns
    -------
    LongTensor of shape [<=k, 2] on same device as `energy`.
    """
    if max_features is None:
        max_features = int(energy.numel())

    if k <= 0 or max_features <= 1:
        return torch.empty(0, 2, dtype=torch.long, device=energy.device)

    # High-low pairing strategy: pair high-energy with low-energy dimensions
    if pairing_strategy == "high_low":
        return _select_high_low_pairs_gpu(energy, k, max_features)

    # Clamp to valid range
    energy = energy[:max_features]

    # Candidate pool size: 8xk (heuristic) but at least 16, at most N
    cand = min(max_features, max(16, int(8 * k)))

    # Get top-k candidates (all on GPU)
    topk_vals, topk_idx = torch.topk(energy, k=cand, largest=True, sorted=False)

    # Fully GPU vectorized implementation: zero CPU transfer, pure GPU greedy selection
    n_cand = topk_idx.size(0)
    if n_cand < 2:
        return torch.empty(0, 2, dtype=torch.long, device=energy.device)

    # GPU: compute scores for all candidate pairs
    cand_energy = energy[topk_idx]  # [cand]

    # Create all possible pairs (i < j), using vectorized operations
    i_indices = torch.arange(n_cand, device=energy.device).unsqueeze(1).expand(-1, n_cand)
    j_indices = torch.arange(n_cand, device=energy.device).unsqueeze(0).expand(n_cand, -1)
    mask = i_indices < j_indices

    i_pairs = topk_idx[i_indices[mask]]  # flattened i indices
    j_pairs = topk_idx[j_indices[mask]]  # flattened j indices
    pair_scores = energy[i_pairs] * energy[j_pairs]  # [n_possible_pairs]

    # GPU: select optimal candidate pairs
    n_pairs_to_check = min(pair_scores.size(0), max(k * 4, 64))
    _, top_pair_indices = torch.topk(pair_scores, k=n_pairs_to_check, largest=True, sorted=True)

    # Extract candidate pairs (keep on GPU)
    cand_i = i_pairs[top_pair_indices]  # [n_pairs_to_check]
    cand_j = j_pairs[top_pair_indices]  # [n_pairs_to_check]

    # Pure GPU greedy selection: completely avoid Python loops and sync
    used_mask = torch.zeros(max_features, dtype=torch.bool, device=energy.device)

    # Vectorized greedy selection: process all candidates in parallel batches
    # Use a loop that can be compiled/vectorized, avoiding Python conditionals
    selected_mask = torch.zeros(n_pairs_to_check, dtype=torch.bool, device=energy.device)

    # Process candidates in batches to avoid excessive memory usage
    batch_size = min(1024, n_pairs_to_check)  # Process in reasonable batches
    for start_idx in range(0, min(k, n_pairs_to_check), batch_size):
        end_idx = min(start_idx + batch_size, min(k, n_pairs_to_check))
        batch_size_actual = end_idx - start_idx

        # Get batch candidates
        batch_i = cand_i[start_idx:end_idx]
        batch_j = cand_j[start_idx:end_idx]

        # Check which pairs in this batch are available (vectorized)
        batch_available = (~used_mask[batch_i]) & (~used_mask[batch_j])

        # Count how many we can select from this batch
        available_indices = torch.where(batch_available)[0]
        if len(available_indices) == 0:
            continue

        # Select greedily: take the first available ones we need
        current_selected = selected_mask[start_idx:end_idx].sum()
        needed = min(batch_size_actual, k - int(current_selected))
        to_select = available_indices[:needed]

        if len(to_select) > 0:
            # Mark as selected
            selected_mask[start_idx:end_idx][to_select] = True
            # Update used mask
            used_mask[batch_i[to_select]] = True
            used_mask[batch_j[to_select]] = True

    # Extract selected pairs
    selected_positions = torch.where(selected_mask[:min(k, n_pairs_to_check)])[0]
    if len(selected_positions) == 0:
        return torch.empty(0, 2, dtype=torch.long, device=energy.device)

    selected_i = cand_i[selected_positions]
    selected_j = cand_j[selected_positions]

    return torch.stack([selected_i, selected_j], dim=1)

@torch.no_grad()
def maybe_gumbel(energy: Tensor, use_gumbel: bool, tau: float) -> Tensor:
    if not use_gumbel:
        return energy
    return energy / max(float(tau), 1e-6) + gumbel_noise_like(energy)

@dataclass
class WarmupSchedule:
    warmup_steps: int = 0

    def ratio(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        return float(min(1.0, max(0.0, step / float(self.warmup_steps))))

@torch.no_grad()
def _select_high_low_pairs_gpu(energy: Tensor, k: int, max_features: int) -> Tensor:
    """High-low pairing strategy: pair high-energy with low-energy dimensions.

    This strategy pairs top-i with bottom-i dimensions to enable energy redistribution
    between high and low energy dimensions, potentially helping low-expression
    dimensions gain gradients.
    """
    energy = energy[:max_features]
    n_features = energy.size(0)

    if k <= 0 or n_features < 2:
        return torch.empty(0, 2, dtype=torch.long, device=energy.device)

    # Get indices sorted by energy (ascending order for bottom, descending for top)
    _, sorted_indices = torch.sort(energy, descending=False)  # ascending: low to high

    # For balanced pairing, use even number of features by excluding middle one if odd
    effective_features = n_features if n_features % 2 == 0 else n_features - 1
    sorted_indices = sorted_indices[:effective_features]

    # Calculate maximum possible pairs
    max_pairs = effective_features // 2
    n_pairs = min(k, max_pairs)

    # Top energy dimensions (highest energy)
    top_indices = sorted_indices[-n_pairs:]  # last n_pairs (highest)

    # Bottom energy dimensions (lowest energy)
    bottom_indices = sorted_indices[:n_pairs]  # first n_pairs (lowest)

    # Create pairs: (top_i, bottom_i) for i in 0..n_pairs-1
    pairs = torch.stack([top_indices, bottom_indices], dim=1)

    return pairs


@torch.no_grad()
def compute_allowed_pairs(S: int, step: int, warmup_steps: int, warmup_ratio: float = 0.0, total_steps: int | None = None) -> int:
    ws = int(warmup_steps)
    if (warmup_ratio is not None) and float(warmup_ratio) > 0.0 and total_steps is not None and int(total_steps) > 0:
        ws = max(ws, int(float(total_steps) * float(warmup_ratio)))
    warm_ratio = WarmupSchedule(ws).ratio(step)
    return max(1, int(S * warm_ratio)) if S > 0 else 0

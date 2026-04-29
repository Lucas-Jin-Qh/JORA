from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

def gumbel_noise_like(x: Tensor, eps: float = 1e-8) -> Tensor:
    u = torch.rand_like(x)
    return -torch.log(-torch.log(u + eps) + eps)

@torch.no_grad()
def _greedy_disjoint_from_scores(
    pair_scores: Tensor,   # (n_possible,) tensor of precomputed pair scores
    i_cand: Tensor,       # (n_possible,) tensor of left-dimension indices for each pair
    j_cand: Tensor,       # (n_possible,) tensor of right-dimension indices for each pair
    k: int,
    max_features: int,
    device: torch.device,
) -> Tensor:
    """Greedy disjoint pair selection from precomputed (score, left_idx, right_idx) triples.

    Given N candidate pairs each with a scalar score, selects up to k disjoint pairs
    greedily by descending score. A pair is disjoint when none of its two indices
    appear in any previously selected pair.

    Returns LongTensor of shape [<=k, 2] of (i, j) dimension indices.
    """
    n_possible = pair_scores.size(0)
    if n_possible < 2:
        return torch.empty(0, 2, dtype=torch.long, device=device)

    n_pairs_to_check = min(n_possible, max(k * 16, 64))
    _, top_indices = torch.topk(pair_scores, k=n_pairs_to_check, largest=True, sorted=True)

    ci = i_cand[top_indices]
    cj = j_cand[top_indices]

    used_mask = torch.zeros(max_features, dtype=torch.bool, device=device)
    selected_pairs: list[Tensor] = []

    for idx in range(n_pairs_to_check):
        left, right = ci[idx], cj[idx]
        if used_mask[left] or used_mask[right]:
            continue
        selected_pairs.append(torch.stack([left, right]))
        used_mask[left] = True
        used_mask[right] = True
        if len(selected_pairs) >= k:
            break

    if not selected_pairs:
        return torch.empty(0, 2, dtype=torch.long, device=device)
    return torch.stack(selected_pairs, dim=0)


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
    _, topk_idx = torch.topk(energy, k=cand, largest=True, sorted=False)

    n_cand = topk_idx.size(0)
    if n_cand < 2:
        return torch.empty(0, 2, dtype=torch.long, device=energy.device)

    # Build all candidate pairs (i < j in candidate tensor)
    i_indices = torch.arange(n_cand, device=energy.device).unsqueeze(1).expand(-1, n_cand)
    j_indices = torch.arange(n_cand, device=energy.device).unsqueeze(0).expand(n_cand, -1)
    mask = i_indices < j_indices

    i_pairs = topk_idx[i_indices[mask]]   # global dimension indices for left element
    j_pairs = topk_idx[j_indices[mask]]   # global dimension indices for right element
    pair_scores = energy[i_pairs] * energy[j_pairs]  # [n_possible_pairs]

    return _greedy_disjoint_from_scores(
        pair_scores=pair_scores,
        i_cand=i_pairs,
        j_cand=j_pairs,
        k=k,
        max_features=max_features,
        device=energy.device,
    )


@torch.no_grad()
def select_coupling_pairs_gpu(
    coupling_score: Tensor,  # (d, d) activation outer-product EMA score matrix
    k: int,
    max_features: Optional[int] = None,
) -> Tensor:
    """Greedy disjoint pair selection driven by a coupling score matrix.

    Uses the full (d, d) coupling score matrix instead of deriving pair scores
    from a (d,) energy vector. Pair score for (i, j) is coupling_score[i, j].

    The candidate pool is built from the top-8k dimensions by their maximum
    coupling to any partner, then disjoint pairs are selected greedily by
    descending coupling score.

    Parameters
    ----------
    coupling_score : Tensor
        (d, d) matrix where entry (i, j) is the coupling score between
        dimensions i and j. Higher = stronger coupling. Diagonal entries
        are not used as pair scores.
    k : int
        Number of disjoint pairs to select.
    max_features : Optional[int]
        Maximum dimension index to consider. Defaults to coupling_score.shape[0].

    Returns
    -------
    LongTensor of shape [<=k, 2] of (i, j) dimension indices.  Empty tensor
    if k <= 0 or there are fewer than 2 candidate dimensions.
    """
    if max_features is None:
        max_features = int(coupling_score.size(0))

    if k <= 0 or max_features <= 1:
        return torch.empty(0, 2, dtype=torch.long, device=coupling_score.device)

    coupling_score = coupling_score[:max_features, :max_features]

    # Candidate pool: top 8k dimensions by their strongest coupling to any partner.
    # max_coupling[i] = max_j coupling_score[i, j]  (diagonal included, acceptable)
    max_coupling_per_dim, _ = coupling_score.max(dim=1)  # (d,)
    cand = min(max_features, max(16, int(8 * k)))
    _, topk_idx = torch.topk(max_coupling_per_dim, k=cand, largest=True, sorted=False)

    n_cand = topk_idx.size(0)
    if n_cand < 2:
        return torch.empty(0, 2, dtype=torch.long, device=coupling_score.device)

    # Build all candidate pairs (i < j in candidate tensor)
    i_indices = torch.arange(n_cand, device=coupling_score.device).unsqueeze(1).expand(-1, n_cand)
    j_indices = torch.arange(n_cand, device=coupling_score.device).unsqueeze(0).expand(n_cand, -1)
    mask = i_indices < j_indices

    i_cand = topk_idx[i_indices[mask]]   # global dimension indices for left element
    j_cand = topk_idx[j_indices[mask]]   # global dimension indices for right element
    pair_scores = coupling_score[i_cand, j_cand]  # [n_possible_pairs]

    return _greedy_disjoint_from_scores(
        pair_scores=pair_scores,
        i_cand=i_cand,
        j_cand=j_cand,
        k=k,
        max_features=max_features,
        device=coupling_score.device,
    )

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

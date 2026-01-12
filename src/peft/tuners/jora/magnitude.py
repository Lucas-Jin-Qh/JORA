from __future__ import annotations

import torch
from torch import Tensor

@torch.no_grad()
def linear_temperature_anneal(step: int, total_steps: int, t_start: float, t_end: float) -> float:
    total_steps = max(1, int(total_steps))
    p = float(step) / float(total_steps)
    p = max(0.0, min(1.0, p))
    return (1.0 - p) * float(t_start) + p * float(t_end)

def compute_ecd_scale(
    base_row_norms: Tensor,
    total_energy: Tensor,
    ecd_log_mag: Tensor,
    ecd_alpha: float = 0.5,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Legacy tanh-gated energy calibration (DoRA-style gating + global rescale).

    This is **NOT** the softmax-competitive OER described in the paper.
    We keep it only for backward compatibility / ablations.
    """
    base_norms = base_row_norms.float()
    z = ecd_log_mag.float() / max(float(temperature), eps)
    alpha = float(ecd_alpha)

    raw_m = base_norms * (1.0 + alpha * torch.tanh(z))
    current_E = (raw_m ** 2).sum()
    c = torch.sqrt(total_energy.float() / (current_E + eps))
    scale = c * (1.0 + alpha * torch.tanh(z))
    return scale

def compute_oer_scale_softmax(
    base_row_norms: Tensor,
    total_energy: Tensor,
    oer_logits: Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Softmax-competitive Orthogonal Energy Redistribution (OER).

    Let logits w produce an energy distribution p = softmax(w/T), sum(p)=1.
    Allocate total energy E_total across output rows: E_i = E_total * p_i.
    Convert to target magnitudes m_i = sqrt(E_i), then scale rows by:
        scale_i = m_i / (||W0_i|| + eps)

    This yields competition (increasing one dimension's share reduces others)
    and explicit energy conservation: sum(m_i^2) == E_total (up to eps).
    """
    base_norms = base_row_norms.float()
    logits = oer_logits.float() / max(float(temperature), eps)
    p = torch.softmax(logits, dim=0)
    target_E = total_energy.float() * p
    target_m = torch.sqrt(target_E + eps)
    scale = target_m / (base_norms + eps)
    return scale

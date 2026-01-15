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
    base_norms = base_row_norms  # Assume fp32 input for performance
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

    生产级实现：修复数值稳定性问题，同时保证严格的能量守恒。

    Let logits w produce an energy distribution p = softmax(w/T), sum(p)=1.
    Allocate total energy E_total across output rows: E_i = E_total * p_i.
    Convert to target magnitudes m_i = sqrt(E_i), then scale rows by:
        scale_i = m_i / max(||W0_i||, min_norm)

    This yields competition (increasing one dimension's share reduces others)
    and explicit energy conservation: sum(m_i^2) == E_total (guaranteed by renormalization).
    """
    # 输入验证和类型转换
    base_norms = base_row_norms  # Assume fp32 input for performance
    # 安全地将total_energy转换为标量
    if total_energy.numel() == 1:
        total_energy_val = float(total_energy.float().item())
    else:
        # 如果是多元素tensor，取其和作为总能量
        total_energy_val = float(total_energy.float().sum().item())

    # 边界情况1：总能量为0，返回单位缩放
    if total_energy_val <= 0:
        return torch.ones_like(base_norms)

    # 边界情况2：所有范数都为0，返回单位缩放（避免NaN）
    if not torch.isfinite(base_norms).all() or (base_norms <= 0).all():
        return torch.ones_like(base_norms)

    # 计算概率分布
    logits = oer_logits.float() / max(float(temperature), eps)
    p = torch.softmax(logits, dim=0)

    # 计算目标能量分配
    target_E = total_energy_val * p
    target_m = torch.sqrt(torch.clamp(target_E, min=eps))  # 确保非负

    # 数值稳定性修复：clamp分母避免极小值
    # min_norm基于数据尺度和eps自适应确定，但设置更保守的下界
    base_scale = (total_energy_val / base_norms.numel()) ** 0.5  # 平均能量尺度
    # 使用更保守的策略：确保min_norm至少是eps的100倍，或者base_scale的1e-4倍
    min_norm = max(eps * 100, min(base_scale * 1e-4, base_scale * 1e-2))
    safe_base_norms = torch.clamp(base_norms, min=min_norm)

    # 计算原始缩放因子
    raw_scale = target_m / safe_base_norms

    # 计算实际分配的能量（用于重新归一化）
    actual_E = (raw_scale * base_norms) ** 2
    actual_total_E = actual_E.sum()

    # 边界情况3：实际总能量为0或无效，返回均匀缩放
    if not torch.isfinite(actual_total_E) or actual_total_E <= 0:
        uniform_scale = torch.sqrt(total_energy_val / base_norms.numel())
        return torch.full_like(base_norms, uniform_scale)

    # 重新归一化保证严格的能量守恒
    renormalization_factor = torch.sqrt(total_energy_val / actual_total_E)
    scale = raw_scale * renormalization_factor

    # 最终安全检查：确保输出是有限的
    if not torch.isfinite(scale).all():
        # 退化到均匀分布
        uniform_scale = torch.sqrt(total_energy_val / base_norms.numel())
        scale = torch.full_like(base_norms, uniform_scale)

    return scale

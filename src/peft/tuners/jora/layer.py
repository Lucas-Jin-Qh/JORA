from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
import torch.nn as nn
from torch import Tensor

from peft.tuners.tuners_utils import BaseTunerLayer

from .config import JoraConfig
from .core import build_core
from .rotation import apply_rotations
from .selection import select_top_k_pairs_gpu, maybe_gumbel, compute_allowed_pairs
from .magnitude import compute_ecd_scale, compute_oer_scale_softmax, linear_temperature_anneal
from .utils import get_in_out_features, linear_forward

@dataclass
class _EmaState:
    row: Tensor
    col: Tensor

class _JoraAdapterState(nn.Module):
    """Per-adapter parameters/buffers for a single JoraLayer."""

    def __init__(self, base_layer: nn.Module, cfg: JoraConfig):
        super().__init__()
        in_features, out_features = get_in_out_features(base_layer)
        self.m = in_features
        self.n = out_features
        self.cfg = cfg

        dev = base_layer.weight.device
        dt = base_layer.weight.dtype

        # Core
        self.core = build_core(cfg.core, self.n, self.m, device=dev, dtype=dt, cfg=cfg)

        # Rotation params
        self.theta_L: Optional[nn.Parameter]
        self.theta_R: Optional[nn.Parameter]

        if cfg.S_L == 0:
            self.theta_L = None
        else:
            init_std = float(getattr(cfg, "theta_init_std", 0.002))
            if not getattr(cfg, "force_random_rotation_init", True):
                init_std = min(init_std, 0.001)
            self.theta_L = nn.Parameter(init_std * torch.randn(cfg.S_L, device=dev, dtype=dt))

        if cfg.S_R == 0:
            self.theta_R = None
        else:
            init_std = float(getattr(cfg, "theta_init_std", 0.002))
            if not getattr(cfg, "force_random_rotation_init", True):
                init_std = min(init_std, 0.001)
            self.theta_R = nn.Parameter(init_std * torch.randn(cfg.S_R, device=dev, dtype=dt))

        # Pair buffers (static, shape-invariant for checkpoint/resume)
        # We allocate full capacity [S, 2] and track the active prefix length with num_pairs.
        s_l_cap = max(0, int(cfg.S_L))
        s_r_cap = max(0, int(cfg.S_R))

        self.register_buffer(
            "pairs_L",
            torch.full((s_l_cap, 2), -1, dtype=torch.long, device=dev),
            persistent=True,
        )
        self.register_buffer(
            "pairs_R",
            torch.full((s_r_cap, 2), -1, dtype=torch.long, device=dev),
            persistent=True,
        )

        self.register_buffer("num_pairs_L", torch.zeros((), device=dev, dtype=torch.long), persistent=True)
        self.register_buffer("num_pairs_R", torch.zeros((), device=dev, dtype=torch.long), persistent=True)

        # EMA stats for selection
        self.register_buffer("grad_row_ema", torch.zeros(self.n, device=dev, dtype=torch.float32), persistent=True)
        self.register_buffer("grad_col_ema", torch.zeros(self.m, device=dev, dtype=torch.float32), persistent=True)

        self.register_buffer("step_idx", torch.zeros((), device=dev, dtype=torch.long), persistent=True)
        self.register_buffer("ema_step_idx", torch.zeros((), device=dev, dtype=torch.long), persistent=True)

        # Magnitude module (row-wise scaling)
        self.ecd_log_mag: Optional[nn.Parameter] = None
        mag = getattr(cfg, "magnitude", "none")
        if mag != "none":
            with torch.no_grad():
                # For magnitude scaling, we need norms along the output dimension
                # This ensures base_row_norms has exactly self.n elements
                if base_layer.weight.dim() == 2:
                    # For 2D weights [out_features, in_features], norm along dim=0 gives output norms
                    base_row_norms = torch.norm(base_layer.weight, p=2, dim=0)
                else:
                    # Fallback: assume first dimension is output
                    base_row_norms = torch.norm(base_layer.weight, p=2, dim=0)

                # Ensure we have exactly self.n elements (output dimension)
                if base_row_norms.size(0) != self.n:
                    # If size mismatch, try the other dimension
                    if base_layer.weight.dim() == 2:
                        base_row_norms = torch.norm(base_layer.weight, p=2, dim=1)
                    # If still not matching, resize to self.n
                    if base_row_norms.size(0) != self.n:
                        if base_row_norms.size(0) > self.n:
                            base_row_norms = base_row_norms[:self.n]
                        else:
                            # Pad if necessary
                            padding_size = self.n - base_row_norms.size(0)
                            padding = torch.ones(padding_size, device=base_row_norms.device, dtype=base_row_norms.dtype)
                            base_row_norms = torch.cat([base_row_norms, padding])

                total_energy = (base_row_norms.float() ** 2).sum()
            self.register_buffer("base_row_norms", base_row_norms, persistent=True)
            self.register_buffer("total_energy", total_energy, persistent=True)
            # Parameter is interpreted as either tanh-gate (ecd_tanh) or softmax logits (oer_softmax)
            self.ecd_log_mag = nn.Parameter(torch.zeros(self.n, device=dev, dtype=dt))
        else:
            self.register_buffer("base_row_norms", torch.ones(self.n, device=dev, dtype=dt), persistent=False)
            self.register_buffer("total_energy", torch.ones((), device=dev, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def _rand_pairs(self, n_features: int, n_pairs: int) -> Tensor:
        """Generate `n_pairs` disjoint pairs among `n_features` indices."""
        dev = self.pairs_L.device
        n_pairs = int(n_pairs)
        n_features = int(n_features)
        if n_pairs <= 0 or n_features <= 1:
            return torch.empty(0, 2, dtype=torch.long, device=dev)
        n_pairs = min(n_pairs, n_features // 2)
        perm = torch.randperm(n_features, device=dev)[: 2 * n_pairs].view(-1, 2)
        return perm

    @torch.no_grad()
    def _write_pairs(self, target_buffer: Tensor, target_counter: Tensor, new_pairs: Tensor, side: str = None) -> None:
        """Write pairs into a static buffer prefix and update the counter (shape-invariant)."""
        cap = int(target_buffer.size(0))
        if cap == 0:
            target_counter.zero_()
            return

        n_new = int(new_pairs.size(0)) if (new_pairs is not None) else 0
        n_safe = max(0, min(n_new, cap))
        if n_safe > 0:
            target_buffer[:n_safe].copy_(new_pairs[:n_safe])
        if cap > n_safe:
            target_buffer[n_safe:].fill_(-1)
        target_counter.fill_(n_safe)

        # Update Python counter and cache (zero-sync maintenance)
        if side:
            # Lazy initialization: only sync once when first accessed
            if not hasattr(self, '_num_pairs_py_initialized'):
                self._num_pairs_py = {
                    'left': int(self.num_pairs_L.item()),
                    'right': int(self.num_pairs_R.item())
                }
                self._num_pairs_py_initialized = True

            if not hasattr(self, '_counter_cache'):
                self._counter_cache = {'left': 0, 'right': 0}

            self._num_pairs_py[side] = n_safe
            self._counter_cache[side] = n_safe

    @torch.no_grad()
    def _init_buffer_random(
        self,
        target_buffer: Tensor,
        target_counter: Tensor,
        feature_dim: int,
        capacity: int,
        n_pairs: int | None = None,
    ) -> None:
        """Initialize random pairs into the static buffer without changing its shape."""
        cap = int(target_buffer.size(0))
        if int(capacity) <= 0 or cap == 0:
            target_counter.zero_()
            if cap > 0:
                target_buffer.fill_(-1)
            return

        init_n = max(1, int(capacity) // 4) if n_pairs is None else int(n_pairs)
        init_n = max(0, min(init_n, cap))
        pairs = self._rand_pairs(int(feature_dim), init_n)
        self._write_pairs(target_buffer, target_counter, pairs)

    @torch.no_grad()
    def init_random_pairs(self, n_pairs_L: int | None = None, n_pairs_R: int | None = None):
        """Initialize random disjoint pairs into the static buffers.

        NOTE: We never replace the `pairs_*` tensors (shape must stay invariant for checkpointing).
        Only the active prefix length (`num_pairs_*`) and the first K entries are updated.
        """
        self._init_buffer_random(self.pairs_L, self.num_pairs_L, self.n, int(self.cfg.S_L), n_pairs_L)
        self._init_buffer_random(self.pairs_R, self.num_pairs_R, self.m, int(self.cfg.S_R), n_pairs_R)

    @torch.no_grad()
    def update_temperature(self, current_step: int, total_steps: int):
        if not self.cfg.ecd_temp_annealing:
            return
        if self.cfg.magnitude not in ("ecd_tanh", "oer_softmax"):
            return
        new_t = linear_temperature_anneal(current_step, total_steps, self.cfg.ecd_temp_start, self.cfg.ecd_temp_end)
        if self.cfg.magnitude == "oer_softmax":
            self.cfg.oer_temperature = float(new_t)
        else:
            self.cfg.ecd_temperature = float(new_t)


    @torch.no_grad()
    def _update_pair_buffer(
        self,
        target_buffer: Tensor,
        target_counter: Tensor,
        energy_src: Tensor,
        allowed_count: int,
        feature_dim: int,
        side: str,
    ) -> None:
        """Generic update: (energy -> maybe_gumbel -> topk pairs -> copy_ -> counter).

        DRY core to avoid left/right mirror divergence.
        """
        allowed_count = int(allowed_count)
        cap = int(target_buffer.size(0))
        if cap == 0 or allowed_count <= 0:
            target_counter.zero_()
            if cap > 0:
                target_buffer.fill_(-1)
            return

        allowed_count = min(allowed_count, cap)

        # Check if update is needed by reading current active count from Python cache
        if not hasattr(self, '_counter_cache'):
            self._counter_cache = {'left': 0, 'right': 0}

        cur = self._counter_cache.get(side, 0) if side else 0

        # Only update if we need more pairs than currently active
        if cur >= allowed_count:
            return

        energy = maybe_gumbel(energy_src, self.cfg.use_gumbel, self.cfg.gumbel_tau)
        new_pairs = select_top_k_pairs_gpu(energy, k=allowed_count, max_features=int(feature_dim))
        self._write_pairs(target_buffer, target_counter, new_pairs, side)

        # Update cache after writing pairs
        if side:
            self._counter_cache[side] = allowed_count

    @torch.no_grad()
    def update_step(self, current_step: int, total_steps: int | None = None):
        """Update active pairs based on EMA stats.

        Only mutates the *contents* of static buffers and counters (`num_pairs_*`).
        Buffer shapes stay invariant for checkpoint/resume safety.
        """
        if self.cfg.selection == "none":
            return

        S_allow_L = compute_allowed_pairs(self.cfg.S_L, current_step, self.cfg.warmup_steps, self.cfg.warmup_ratio, total_steps)
        S_allow_R = compute_allowed_pairs(self.cfg.S_R, current_step, self.cfg.warmup_steps, self.cfg.warmup_ratio, total_steps)

        S_allow_L = min(int(S_allow_L), int(self.pairs_L.size(0)))
        S_allow_R = min(int(S_allow_R), int(self.pairs_R.size(0)))

        if self.cfg.selection == "random":
            self.init_random_pairs(n_pairs_L=S_allow_L, n_pairs_R=S_allow_R)
            return

        # default: topk_ema
        if self.cfg.S_L > 0:
            self._update_pair_buffer(self.pairs_L, self.num_pairs_L, self.grad_row_ema, S_allow_L, self.n, 'left')
        if self.cfg.S_R > 0:
            self._update_pair_buffer(self.pairs_R, self.num_pairs_R, self.grad_col_ema, S_allow_R, self.m, 'right')

    def _apply_side_rotation(self, x: Tensor, is_left_side: bool) -> Tensor:
        """Apply one side of the implicit rotation with all safety guards.

        Right-side (input) applies R  (forward order, +theta).
        Left-side  (output) applies R^{-1} = R^T (reverse order, -theta).
        """
        counter = self.num_pairs_L if is_left_side else self.num_pairs_R
        buffer = self.pairs_L if is_left_side else self.pairs_R
        theta = self.theta_L if is_left_side else self.theta_R

        single_sided = getattr(self.cfg, "single_sided", "none")
        skip_condition = (single_sided == "right") if is_left_side else (single_sided == "left")

        # Zero-sync: use Python-side maintained counter, completely avoid GPU sync
        if not hasattr(self, '_num_pairs_py_initialized'):
            # Only sync once during first access
            self._num_pairs_py = {
                'left': int(self.num_pairs_L.item()),
                'right': int(self.num_pairs_R.item())
            }
            self._num_pairs_py_initialized = True

        n_active = self._num_pairs_py['left' if is_left_side else 'right']

        if n_active <= 0 or theta is None or skip_condition:
            return x

        # Slice active prefix only (critical with -1 sentinel padding).
        active_pairs = buffer[:n_active]
        active_theta = theta[:n_active]

        return apply_rotations(
            x=x,
            pairs=active_pairs,
            thetas=active_theta,
            rotation_param=self.cfg.rotation_param,
            impl=self.cfg.rotation_impl,
            reverse=is_left_side,
            negate_theta=is_left_side,
        )

    def compute_delta(self, x: Tensor) -> Tensor:
        """Compute JORA contribution (implicit)."""
        # 1) Input rotation (Right)
        x_rot = self._apply_side_rotation(x, is_left_side=False)

        # 2) Core computation
        y_core = self.core.apply_to_vector(x_rot)

        # 3) Output rotation (Left)
        y = self._apply_side_rotation(y_core, is_left_side=True)

        # Soft clipping to prevent gradient explosion while maintaining smoothness
        # Use tanh for smooth (-1, 1) bounding instead of hard clamping
        if not getattr(self.cfg, "zero_init_core", False):
            y = torch.tanh(y)

        return y

    def maybe_apply_magnitude(self, out: Tensor) -> Tensor:
        mag = getattr(self.cfg, "magnitude", "none")
        if mag == "none":
            return out

        if self.ecd_log_mag is None:
            return out

        if mag == "ecd_tanh":
            scale = compute_ecd_scale(
                base_row_norms=self.base_row_norms,
                total_energy=self.total_energy,
                ecd_log_mag=self.ecd_log_mag,
                ecd_alpha=self.cfg.ecd_alpha,
                temperature=self.cfg.ecd_temperature,
                eps=self.cfg.eps,
            ).to(out.dtype)
        elif mag == "oer_softmax":
            scale = compute_oer_scale_softmax(
                base_row_norms=self.base_row_norms,
                total_energy=self.total_energy,
                oer_logits=self.ecd_log_mag,
                temperature=self.cfg.oer_temperature,
                eps=self.cfg.eps,
            ).to(out.dtype)
        else:
            return out

        # broadcast over batch/sequence
        return out * scale.view(*([1] * (out.dim() - 1)), -1)

class JoraLayer(nn.Module, BaseTunerLayer):
    """A PEFT-compatible layer wrapper that injects JORA into a Linear/Conv1D module."""

    def __init__(self, base_layer: nn.Module, adapter_name: str, cfg: JoraConfig):
        nn.Module.__init__(self)
        try:
            BaseTunerLayer.__init__(self)  # type: ignore
        except Exception:
            pass

        self.base_layer = base_layer
        # Freeze base weights by default (fair PEFT)
        self.base_layer.weight.requires_grad = False
        if getattr(self.base_layer, "bias", None) is not None:
            self.base_layer.bias.requires_grad = False

        self.adapters = nn.ModuleDict()
        self._active_adapter = adapter_name
        self.add_adapter(adapter_name, cfg)

        # Backward hook: update grad_row_ema using grad_output
        self._hook_handle = self.register_full_backward_hook(self._backward_hook)

    def add_adapter(self, adapter_name: str, cfg: JoraConfig):
        st = _JoraAdapterState(self.base_layer, cfg)
        st.init_random_pairs()
        self.adapters[adapter_name] = st

    @property
    def active_adapter(self) -> str:
        return self._active_adapter

    def set_adapter(self, adapter_name: str):
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found in this layer.")
        self._active_adapter = adapter_name

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0] shape: [..., out_features]
        try:
            st = self.adapters[self._active_adapter]
            if (not self.training) or (not torch.is_grad_enabled()):
                return
            if not grad_output or grad_output[0] is None:
                return
            g = grad_output[0].detach()
            if torch.isnan(g).any() or torch.isinf(g).any():
                return
            g_sq = g.reshape(-1, st.n).float().pow(2).mean(dim=0)
            beta = float(st.cfg.ema_beta)
            st.grad_row_ema.mul_(beta).add_(g_sq, alpha=1.0 - beta)
        except Exception:
            return

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        st = self.adapters[self._active_adapter]

        # Column energy from activations (with frequency control, zero CPU sync)
        if self.training:
            # Use Python counter to avoid GPU scalar .item() synchronization
            if not hasattr(self, '_ema_step_counter'):
                self._ema_step_counter = 0
            self._ema_step_counter += 1

            ema_interval = int(getattr(st.cfg, "ema_update_interval", 1))
            if ema_interval <= 1 or (self._ema_step_counter % ema_interval) == 0:
                with torch.no_grad():
                    xd = x.detach()
                    if not torch.isnan(xd).any() and not torch.isinf(xd).any():
                        x_sq = xd.reshape(-1, st.m).float().pow(2).mean(dim=0)
                        beta = float(st.cfg.ema_beta)
                        st.grad_col_ema.mul_(beta).add_(x_sq, alpha=1.0 - beta)

        # Ensure input dtype matches weight dtype for proper computation
        if x.dtype != self.base_layer.weight.dtype:
            x = x.to(self.base_layer.weight.dtype)
        base_out = linear_forward(self.base_layer, x)

        if getattr(self, "disable_adapters", False):
            return base_out

        delta = st.compute_delta(x)
        out = base_out + delta
        out = st.maybe_apply_magnitude(out)

        # Ensure output dtype matches input dtype for compatibility
        if out.dtype != x.dtype:
            out = out.to(x.dtype)

        return out

    # Convenience for training loops
    @torch.no_grad()
    def update_step(self, current_step: int, total_steps: int | None = None):
        self.adapters[self._active_adapter].update_step(current_step=current_step, total_steps=total_steps)

    @torch.no_grad()
    def update_temperature(self, current_step: int, total_steps: int):
        self.adapters[self._active_adapter].update_temperature(current_step, total_steps)

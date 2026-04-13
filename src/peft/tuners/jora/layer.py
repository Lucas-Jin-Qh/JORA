from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
import torch.nn as nn
from torch import Tensor

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

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

        if cfg.core == "selective_diag" and self.m != self.n:
            raise ValueError(
                "SelectiveDiagCore currently requires square target layers, "
                f"but got in_features={self.m}, out_features={self.n} for {base_layer.__class__.__name__}. "
                "Use square targets such as GPT-2 attn.c_proj or choose a non-selective JORA core."
            )

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
            init_std = float(cfg.theta_init_std)
            # Paper path (selective_diag): zero-init theta so adapter starts at identity
            if cfg.core == "selective_diag":
                init_std = 0.0
            elif not getattr(cfg, "force_random_rotation_init", True):
                init_std = min(init_std, 0.001)
            if init_std == 0.0:
                self.theta_L = nn.Parameter(torch.zeros(cfg.S_L, device=dev, dtype=dt))
            else:
                self.theta_L = nn.Parameter(init_std * torch.randn(cfg.S_L, device=dev, dtype=dt))

        if cfg.S_R == 0:
            self.theta_R = None
        else:
            init_std = float(cfg.theta_init_std)
            # Paper path (selective_diag): zero-init theta so adapter starts at identity
            if cfg.core == "selective_diag":
                init_std = 0.0
            elif not getattr(cfg, "force_random_rotation_init", True):
                init_std = min(init_std, 0.001)
            if init_std == 0.0:
                self.theta_R = nn.Parameter(torch.zeros(cfg.S_R, device=dev, dtype=dt))
            else:
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

        # Paper-path state: pairs_frozen persisted as a buffer so state_dict roundtrip restores it.
        self.register_buffer(
            "pairs_frozen_flag",
            torch.zeros((), dtype=torch.bool, device=dev),
            persistent=True,
        )
        self.register_load_state_dict_post_hook(self._restore_frozen_flag)
        self._pairs_frozen = False
        self._step_idx_py = 0

        # Magnitude module (row-wise scaling)
        self.ecd_log_mag: Optional[nn.Parameter] = None
        mag = getattr(cfg, "magnitude", "none")
        if mag != "none":
            with torch.no_grad():
                # For magnitude scaling, we need norms along the OUTPUT dimension.
                # - nn.Linear: weight shape [out_features, in_features], use dim=1 (each row = one output)
                # - HF Conv1D: weight shape [in_features, out_features], use dim=0 (each column = one output)
                # We use the same logic as utils.py to determine the correct dimension.
                from .utils import is_conv1d_layer
                if is_conv1d_layer(base_layer):
                    # Conv1D: weight [in, out], output is along dim=0 (columns)
                    base_row_norms = torch.norm(base_layer.weight, p=2, dim=0)
                elif isinstance(base_layer, nn.Linear):
                    # Linear: weight [out, in], output is along dim=1 (rows)
                    base_row_norms = torch.norm(base_layer.weight, p=2, dim=1)
                else:
                    # Fallback for other layer types
                    base_row_norms = torch.norm(base_layer.weight, p=2, dim=0)

                # Ensure we have exactly self.n elements (output dimension = out_features)
                if base_row_norms.size(0) != self.n:
                    # If still not matching, resize to self.n
                    if base_row_norms.size(0) > self.n:
                        base_row_norms = base_row_norms[:self.n]
                    else:
                        # Pad if necessary
                        padding_size = self.n - base_row_norms.size(0)
                        padding = torch.ones(padding_size, device=base_row_norms.device, dtype=base_row_norms.dtype)
                        base_row_norms = torch.cat([base_row_norms, padding])

                base_row_norms_fp32 = base_row_norms.float()
                total_energy = (base_row_norms_fp32 ** 2).sum()
            self.register_buffer("base_row_norms", base_row_norms, persistent=True)
            # Cache fp32 version to avoid repeated .float() calls during magnitude computation
            self.register_buffer("base_row_norms_fp32", base_row_norms_fp32, persistent=False)
            self.register_buffer("total_energy", total_energy, persistent=True)
            # OER must start from the base energy distribution; zero logits would force a uniform
            # redistribution and introduce a large step-0 drift before any learning happens.
            if mag == "oer_softmax":
                init_logits = torch.clamp(base_row_norms_fp32.pow(2), min=float(cfg.eps)).log()
                self.ecd_log_mag = nn.Parameter(init_logits.to(device=dev, dtype=dt))
            else:
                # Legacy tanh gating keeps a zero-centered initialization.
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

        # Always re-select pairs based on current EMA statistics
        # Do NOT cache or early-return, as EMA changes each step and pairs should adapt
        energy = maybe_gumbel(energy_src, self.cfg.use_gumbel, self.cfg.gumbel_tau)
        pairing_strategy = getattr(self.cfg, "pairing_strategy", "consecutive")
        new_pairs = select_top_k_pairs_gpu(energy, k=allowed_count, max_features=int(feature_dim), pairing_strategy=pairing_strategy)
        self._write_pairs(target_buffer, target_counter, new_pairs, side)

    def _effective_k_allow(self, current_step: int, total_steps: int | None = None) -> int:
        """Compute active pair budget, honoring paper-path calibration if configured."""
        total_k = int(self.cfg.k)
        if total_k <= 0:
            return 0

        t_stat = int(getattr(self.cfg, "t_stat", 0) or 0)
        if t_stat > 0:
            current_step = min(int(current_step), t_stat)
            warmup_steps = t_stat
            warmup_ratio = 0.0
            total_steps = t_stat
        else:
            warmup_steps = int(getattr(self.cfg, "warmup_steps", 0) or 0)
            warmup_ratio = float(getattr(self.cfg, "warmup_ratio", 0.0) or 0.0)

        k_allow = compute_allowed_pairs(total_k, current_step, warmup_steps, warmup_ratio, total_steps)
        return min(int(k_allow), total_k)

    @torch.no_grad()
    def update_step(self, current_step: int, total_steps: int | None = None):
        """Update active pairs based on EMA stats.

        Only mutates the *contents* of static buffers and counters (`num_pairs_*`).
        Buffer shapes stay invariant for checkpoint/resume safety.

        For the paper path (pairs_freeze_after_warmup=True): pairs are selected once
        when warmup completes and frozen thereafter. For SelectiveDiagCore, also calls
        set_support() to freeze the support indices U.
        """
        if self.cfg.selection == "none":
            return

        # Paper path: freeze pairs after first full allocation
        if getattr(self.cfg, "pairs_freeze_after_warmup", False):
            if bool(self.pairs_frozen_flag.item()):
                return  # Support already frozen; skip re-selection

        # Compute warmup ratio for k parameter
        self._step_idx_py = max(self._step_idx_py, int(current_step))
        self.step_idx.fill_(self._step_idx_py)
        k_allow = self._effective_k_allow(current_step, total_steps)

        if self.cfg.selection == "random":
            # For random selection, distribute k pairs between left and right sides
            k_per_side = k_allow // 2
            k_L = min(k_per_side, int(self.pairs_L.size(0)))
            k_R = min(k_per_side, int(self.pairs_R.size(0)))
            if k_allow % 2 == 1 and k_L < k_per_side * 2:
                k_L = min(k_L + 1, int(self.pairs_L.size(0)))
            self.init_random_pairs(n_pairs_L=k_L, n_pairs_R=k_R)
            # Freeze after first allocation (paper path)
            if getattr(self.cfg, "pairs_freeze_after_warmup", False) and k_allow >= self.cfg.k:
                self._freeze_support_if_needed()
            return

        # default: topk_ema
        total_capacity = int(self.pairs_L.size(0)) + int(self.pairs_R.size(0))
        if total_capacity > 0:
            k_L = min(int(k_allow * self.pairs_L.size(0) // total_capacity), int(self.pairs_L.size(0)))
            k_R = min(k_allow - k_L, int(self.pairs_R.size(0)))
            if k_allow > 0 and k_L == 0 and self.pairs_L.size(0) > 0:
                k_L = 1
            if k_allow > 0 and k_R == 0 and self.pairs_R.size(0) > 0:
                k_R = 1
        else:
            k_L = 0
            k_R = 0

        if self.cfg.S_L > 0 and k_L > 0:
            self._update_pair_buffer(self.pairs_L, self.num_pairs_L, self.grad_row_ema, k_L, self.n, 'left')
        if self.cfg.S_R > 0 and k_R > 0:
            self._update_pair_buffer(self.pairs_R, self.num_pairs_R, self.grad_col_ema, k_R, self.m, 'right')

        # Freeze after full allocation (paper path)
        if getattr(self.cfg, "pairs_freeze_after_warmup", False) and k_allow >= self.cfg.k:
            self._freeze_support_if_needed()

    def _freeze_support_if_needed(self):
        """Freeze support U for SelectiveDiagCore after calibration (paper-exact path).

        Support is the unique union of all active pair indices. If the union is
        smaller than the configured capacity we keep a masked tail instead of
        padding duplicate indices, which avoids parameter aliasing while keeping
        checkpoint tensor shapes stable.
        """
        from .core import SelectiveDiagCore
        if getattr(self, "_pairs_frozen", False):
            return
        if isinstance(self.core, SelectiveDiagCore):
            n_L = int(self.num_pairs_L.item())
            n_R = int(self.num_pairs_R.item())
            indices_parts = []
            if n_L > 0:
                indices_parts.append(self.pairs_L[:n_L].reshape(-1))
            if n_R > 0:
                indices_parts.append(self.pairs_R[:n_R].reshape(-1))

            if indices_parts:
                unique_indices = torch.unique(torch.cat(indices_parts))[: self.core.support_size]
            else:
                unique_indices = torch.zeros(0, dtype=torch.long, device=self.core.support_indices.device)

            actual_size = int(unique_indices.numel())
            if actual_size < self.core.support_size:
                import warnings
                warnings.warn(
                    f"JORA: support union has {actual_size} unique indices but support_size="
                    f"{self.core.support_size}. Keeping a masked tail to avoid duplicate-index aliasing. "
                    f"Consider increasing k or warmup_steps.",
                    UserWarning,
                    stacklevel=3,
                )

            self.core.set_support(unique_indices)
        self._pairs_frozen = True
        self.pairs_frozen_flag.fill_(True)

    def _restore_frozen_flag(self, *_args, **_kwargs) -> None:
        """Post-hook: restore Python-side _pairs_frozen from the persisted buffer after state_dict load."""
        self._pairs_frozen = bool(self.pairs_frozen_flag.item())



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
        """Compute JORA contribution (implicit).

        Paper-exact formula (when core='selective_diag'):
            delta = R_L^T @ D_sel @ R_R @ x - P_U @ x
        where D_sel = I_U + diag(delta) applied only to support U.

        Legacy formula (other core types):
            delta = R_L^T @ core(R_R @ x)  [with optional tanh]
        """
        from .core import SelectiveDiagCore

        if isinstance(self.core, SelectiveDiagCore):
            # Paper-exact path: R_L^T @ D_sel @ R_R @ x - P_U @ x
            # where D_sel = I_U + diag(delta).
            #
            # At delta=0: D_sel = I_U, so:
            #   R_L^T I_U R_R x - P_U x
            # This is NOT zero in general, but when theta=0 (R_L=R_R=I), it is
            #   P_U x - P_U x = 0.  ✓
            #
            # Crucially, only ONE term (y_rotated) depends on theta_L/theta_R,
            # so their gradients flow even at delta=0.  This fixes the gradient-dead
            # theta-at-init problem that occurs when both terms depend on theta.
            #
            # 1) Input rotation (Right): R_R @ x
            x_rot = self._apply_side_rotation(x, is_left_side=False)
            # 2) D_sel applied to rotated input: D_sel @ R_R @ x
            y_sel = self.core.apply_to_vector(x_rot)
            # 3) Output rotation (Left): R_L^T @ D_sel @ R_R @ x
            y_rotated = self._apply_side_rotation(y_sel, is_left_side=True)
            # 4) Subtract P_U @ x in the ORIGINAL (unrotated) input space.
            #    This differs from R_L^T P_U R_R x: only y_rotated depends on theta,
            #    so theta gradients are nonzero even at delta=0 (once support is set).
            proj_x = self.core.project_support(x)  # P_U @ x  (support indices applied to x)
            return y_rotated - proj_x
        else:
            # Legacy path
            # 1) Input rotation (Right)
            x_rot = self._apply_side_rotation(x, is_left_side=False)
            # 2) Core computation
            y_core = self.core.apply_to_vector(x_rot)
            # 3) Output rotation (Left)
            y = self._apply_side_rotation(y_core, is_left_side=True)
            # Soft clipping to prevent gradient explosion while maintaining smoothness
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
                base_row_norms=self.base_row_norms_fp32,  # Use cached fp32 version for performance
                total_energy=self.total_energy,
                ecd_log_mag=self.ecd_log_mag,
                ecd_alpha=self.cfg.ecd_alpha,
                temperature=self.cfg.ecd_temperature,
                eps=self.cfg.eps,
            ).to(out.dtype)
        elif mag == "oer_softmax":
            scale = compute_oer_scale_softmax(
                base_row_norms=self.base_row_norms_fp32,  # Use cached fp32 version for performance
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

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("theta_L", "theta_R", "core", "ecd_log_mag")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ("pairs_L", "pairs_R", "num_pairs_L", "num_pairs_R",
                                         "grad_row_ema", "grad_col_ema", "step_idx", "ema_step_idx")

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

    def set_adapter(self, adapter_names: str | list[str]):
        """Set the active adapter(s).

        Args:
            adapter_names (`str` or `list[str]`):
                 The name(s) of the adapter(s) to set as active.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # JORA only supports single active adapter
        if len(adapter_names) == 0:
            return
        if len(adapter_names) > 1:
            import warnings
            warnings.warn("Multiple active adapters not supported for JORA; using first entry")
        adapter_name = adapter_names[0]

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

            # Gradient EMA update with frequency control (independent of activation EMA)
            if not hasattr(self, '_grad_ema_step_counter'):
                self._grad_ema_step_counter = 0
            self._grad_ema_step_counter += 1

            grad_interval = int(getattr(st.cfg, "ema_grad_interval", 1))
            if grad_interval > 1 and (self._grad_ema_step_counter % grad_interval) != 0:
                return

            g = grad_output[0].detach()
            # Skip EMA update if gradient has NaN/Inf - avoid blocking sync in hot path
            # Using isfinite which combines both nan and inf checks
            if not torch.isfinite(g).all():
                return
            g_sq = g.reshape(-1, st.n).float().pow(2).mean(dim=0)
            beta = float(st.cfg.ema_beta)
            st.grad_row_ema.lerp_(g_sq, 1.0 - beta)  # Fused EMA update for better performance
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
                    # Skip EMA update if input has NaN/Inf - avoid blocking sync in hot path
                    if torch.isfinite(xd).all():
                        x_sq = xd.reshape(-1, st.m).float().pow(2).mean(dim=0)
                        beta = float(st.cfg.ema_beta)
                        st.grad_col_ema.lerp_(x_sq, 1.0 - beta)  # Fused EMA update for better performance

        # Ensure input dtype matches weight dtype for proper computation
        if x.dtype != self.base_layer.weight.dtype:
            x = x.to(self.base_layer.weight.dtype)
        base_out = linear_forward(self.base_layer, x)

        if getattr(self, "disable_adapters", False):
            return base_out

        delta = st.compute_delta(x)
        # OER/ECD scales delta only (not the full output).
        # Forward: y = W₀x + scale ⊙ delta(x)
        delta = st.maybe_apply_magnitude(delta)
        out = base_out + delta

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

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge JORA adapter weights into base weights using efficient matrix approximation.

        This implementation approximates the non-linear JORA transformation as:
        ΔW ≈ R_L @ C @ R_R

        Where:
        - R_L: Left rotation matrix (output-side rotations)
        - C: Core transformation matrix
        - R_R: Right rotation matrix (input-side rotations)

        The approximation captures the main linear effects of rotations and core transformations
        while providing significant inference speedup by eliminating runtime JORA overhead.

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This provides rollback capability on failure. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.

        Raises:
            ValueError: If NaNs are detected in merged weights when safe_merge=True.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.adapters:
                self._merge_single_adapter(active_adapter, safe_merge)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        Unmerge all merged JORA adapters from base weights.

        This reverses the merge operation by subtracting the same delta weights
        that were added during merging. The operation is performed in reverse
        order of merging to ensure correctness.

        Raises:
            RuntimeWarning: If no adapters are currently merged.
        """
        if not self.merged:
            import warnings
            warnings.warn("Already unmerged. Nothing to do.")
            return

        # Unmerge in reverse order to ensure correctness
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.adapters:
                self._unmerge_single_adapter(active_adapter)

    def enable_adapters(self, enabled: bool) -> None:
        """
        Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            # disable grads on all adapter layers
            for adapter_name in self.adapters.keys():
                adapter_state = self.adapters[adapter_name]
                # Disable gradients for rotation parameters
                if hasattr(adapter_state, 'theta_L') and adapter_state.theta_L is not None:
                    adapter_state.theta_L.requires_grad_(False)
                if hasattr(adapter_state, 'theta_R') and adapter_state.theta_R is not None:
                    adapter_state.theta_R.requires_grad_(False)
                # Disable gradients for core parameters
                if hasattr(adapter_state, 'core'):
                    for param in adapter_state.core.parameters():
                        param.requires_grad_(False)
                # Disable gradients for magnitude parameters
                if hasattr(adapter_state, 'ecd_log_mag') and adapter_state.ecd_log_mag is not None:
                    adapter_state.ecd_log_mag.requires_grad_(False)
            self._disable_adapters = True

    def _merge_single_adapter(self, adapter_name: str, safe_merge: bool = False) -> None:
        """
        Merge a single JORA adapter using matrix approximation.

        Args:
            adapter_name: Name of the adapter to merge
            safe_merge: Whether to enable safety checks and rollback
        """
        adapter_state = self.adapters[adapter_name]
        base_layer = self.get_base_layer()

        # Backup original weights if safe_merge is enabled
        original_weight = None
        if safe_merge:
            original_weight = base_layer.weight.data.clone()

        try:
            # Use a simpler approximation approach to avoid complex matrix operations
            delta_weight = self._compute_weight_delta_simple(adapter_state)

            # Store merge metadata for unmerge operations
            merge_metadata = {
                'has_magnitude': adapter_state.ecd_log_mag is not None,
                'magnitude_scale': None
            }

            # Apply magnitude scaling if enabled
            if adapter_state.ecd_log_mag is not None:
                delta_weight, magnitude_metadata = self._apply_magnitude_to_delta_weight(delta_weight, adapter_state)
                merge_metadata['magnitude_metadata'] = magnitude_metadata

            # Store metadata for unmerge
            if not hasattr(adapter_state, '_merge_metadata'):
                adapter_state._merge_metadata = {}
            adapter_state._merge_metadata[adapter_name] = merge_metadata

            # Add to base weights
            delta_weight = delta_weight.to(base_layer.weight.dtype)
            base_layer.weight.data += delta_weight

            # Safe merge validation
            if safe_merge:
                if torch.isnan(base_layer.weight.data).any() or torch.isinf(base_layer.weight.data).any():
                    # Restore original weights on failure
                    if original_weight is not None:
                        base_layer.weight.data = original_weight
                    raise ValueError(f"NaNs detected after merging adapter {adapter_name}")

        except Exception as e:
            # Restore original weights on any failure when safe_merge is enabled
            if safe_merge and original_weight is not None:
                base_layer.weight.data = original_weight
            raise e

    def _compute_weight_delta_simple(self, adapter_state) -> torch.Tensor:
        """
        Compute the weight delta for merging.

        For SelectiveDiagCore (paper-exact path), build the dense linear operator
        exactly by probing the adapter with basis vectors and converting the
        resulting map into the base-layer weight layout. This matches the current
        forward path even when theta is nonzero and avoids hand-derived transpose
        mistakes between row-vector rotations and weight-matrix conventions.

        For other core types (legacy approximation):
            Uses conservative matrix approximation.
        """
        from .core import SelectiveDiagCore

        device = adapter_state.grad_row_ema.device
        dtype = adapter_state.grad_row_ema.dtype
        n_out, n_in = adapter_state.n, adapter_state.m

        if isinstance(adapter_state.core, SelectiveDiagCore):
            from .utils import is_conv1d_layer

            if n_out != n_in:
                raise ValueError(
                    "SelectiveDiagCore merge is only supported for square layers "
                    f"(got out_features={n_out}, in_features={n_in})."
                )

            n_active = getattr(adapter_state.core, "_active_support_size_py", 0)
            if n_active <= 0:
                base_layer = self.get_base_layer()
                return torch.zeros_like(base_layer.weight.data, device=device, dtype=dtype)

            base_layer = self.get_base_layer()
            is_conv1d = is_conv1d_layer(base_layer)
            delta_weight = torch.zeros_like(base_layer.weight.data, device=device, dtype=dtype)

            chunk_size = min(256, n_in)
            for start in range(0, n_in, chunk_size):
                end = min(start + chunk_size, n_in)
                basis = torch.zeros(end - start, n_in, device=device, dtype=dtype)
                local_rows = torch.arange(end - start, device=device)
                basis_indices = torch.arange(start, end, device=device)
                basis[local_rows, basis_indices] = 1.0

                delta_chunk = adapter_state.compute_delta(basis).to(dtype)

                if is_conv1d:
                    delta_weight[start:end, :] = delta_chunk
                else:
                    delta_weight[:, start:end] = delta_chunk.transpose(0, 1)

            return delta_weight

        # Legacy path: conservative approximation
        device = adapter_state.core.A.device if hasattr(adapter_state.core, 'A') else device
        dtype = adapter_state.core.A.dtype if hasattr(adapter_state.core, 'A') else dtype

        # Extract core transformation matrix
        core_matrix = adapter_state.core.forward()  # (n_out, n_in)

        # Estimate rotation effect magnitude from active parameters
        rotation_scale = self._estimate_rotation_effect_magnitude(adapter_state)

        # Combine core matrix with rotation effects
        delta_weight = core_matrix * rotation_scale

        # Apply very conservative scaling for tanh non-linearity
        delta_weight *= 0.05

        return delta_weight

    def _estimate_rotation_effect_magnitude(self, adapter_state) -> torch.Tensor:
        """
        Estimate the magnitude of rotation effects on each input-output dimension pair.

        This provides a conservative estimate of how rotations affect the weight matrix,
        accounting for the coupling introduced by Givens rotations.

        Args:
            adapter_state: The JORA adapter state

        Returns:
            torch.Tensor: Scaling factor tensor of shape (n_out, n_in)
        """
        n_out, n_in = adapter_state.n, adapter_state.m
        device = adapter_state.core.A.device if hasattr(adapter_state.core, 'A') else adapter_state.grad_row_ema.device
        dtype = adapter_state.core.A.dtype if hasattr(adapter_state.core, 'A') else adapter_state.grad_row_ema.dtype

        # Start with base scaling (no rotation effect)
        base_scale = torch.ones(n_out, n_in, device=device, dtype=dtype)

        # Estimate left rotation effects (output dimension coupling)
        left_scale = self._estimate_single_rotation_magnitude(
            adapter_state, side='left', target_dim=n_out
        )

        # Estimate right rotation effects (input dimension coupling)
        right_scale = self._estimate_single_rotation_magnitude(
            adapter_state, side='right', target_dim=n_in
        )

        # Combine rotation effects: each output-input pair is affected by both rotations
        # Use outer product to distribute effects across the weight matrix
        combined_scale = left_scale.unsqueeze(-1) * right_scale.unsqueeze(0)
        combined_scale = torch.clamp(combined_scale, 0.5, 2.0)  # Conservative bounds

        return combined_scale

    def _estimate_single_rotation_magnitude(self, adapter_state, side: str, target_dim: int) -> torch.Tensor:
        """
        Estimate rotation magnitude for a single side.

        Args:
            adapter_state: JORA adapter state
            side: 'left' or 'right'
            target_dim: Target dimension size

        Returns:
            torch.Tensor: Scaling factors for the target dimension
        """
        if side == 'left':
            pairs = adapter_state.pairs_L
            thetas = adapter_state.theta_L
            num_pairs = adapter_state.num_pairs_L
        else:
            pairs = adapter_state.pairs_R
            thetas = adapter_state.theta_R
            num_pairs = adapter_state.num_pairs_R

        device = adapter_state.core.A.device if hasattr(adapter_state.core, 'A') else adapter_state.grad_row_ema.device
        dtype = adapter_state.core.A.dtype if hasattr(adapter_state.core, 'A') else adapter_state.grad_row_ema.dtype

        # Base magnitude (no rotation effect)
        magnitude = torch.ones(target_dim, device=device, dtype=dtype)

        if num_pairs <= 0 or thetas is None:
            return magnitude

        # Estimate effect of each rotation pair
        n_active = int(num_pairs.item())
        active_pairs = pairs[:n_active]
        active_thetas = thetas[:n_active]

        # For each rotation pair, estimate the coupling effect on involved dimensions
        for i, (idx_i, idx_j) in enumerate(active_pairs):
            theta = active_thetas[i]

            # Estimate rotation effect magnitude (small angle approximation)
            # For small θ, rotation R(θ) ≈ I + θ·K, so effect magnitude is ≈ |θ|
            effect_magnitude = torch.clamp(torch.abs(theta), 0.0, 0.5)  # Conservative bound

            # Apply effect to both dimensions involved in rotation
            # This represents the coupling introduced by the rotation
            coupling_factor = 1.0 + 0.1 * effect_magnitude  # Very conservative

            if idx_i < target_dim:
                magnitude[idx_i] *= coupling_factor
            if idx_j < target_dim:
                magnitude[idx_j] *= coupling_factor

        # Ensure reasonable bounds
        magnitude = torch.clamp(magnitude, 0.8, 1.5)

        return magnitude

    def _build_core_matrix(self, adapter_state) -> torch.Tensor:
        """
        Build the equivalent core transformation matrix (n_out, n_in).

        Returns:
            torch.Tensor: Core transformation matrix of shape (n_out, n_in)
        """
        return adapter_state.core.forward()


    def _apply_magnitude_to_delta_weight(self, delta_weight: torch.Tensor, adapter_state, reuse_metadata=None) -> tuple[torch.Tensor, dict] | torch.Tensor:
        """
        Apply magnitude scaling to the delta weight matrix.

        This approximates the effect of magnitude scaling on the weight update.
        Since magnitude scaling is element-wise and depends on base weight norms,
        we use a simplified approximation.

        Args:
            delta_weight: The weight delta matrix (n_out, n_in)
            adapter_state: The JORA adapter state

        Returns:
            torch.Tensor: Magnitude-scaled delta weight matrix
        """
        base_layer = self.get_base_layer()

        # Get cached base weight norms and total energy from adapter_state
        # This ensures consistency with what was computed during initialization
        if reuse_metadata is not None:
            # For unmerge: use the metadata saved during merge
            scale = reuse_metadata['scale']
            base_row_norms = reuse_metadata['base_row_norms']
        else:
            # For merge: use the cached values from adapter initialization
            # This ensures consistency: total_energy = sum(norm^2), not (sum(norm))^2
            if hasattr(adapter_state, 'base_row_norms') and adapter_state.base_row_norms is not None:
                base_row_norms = adapter_state.base_row_norms
            else:
                # Fallback: compute norms (shouldn't happen in normal use)
                base_layer = self.get_base_layer()
                from .utils import is_conv1d_layer
                if is_conv1d_layer(base_layer):
                    # Conv1D: weight [in, out], output is along dim=0 (columns)
                    base_row_norms = torch.norm(base_layer.weight, p=2, dim=0)
                elif isinstance(base_layer, nn.Linear):
                    # Linear: weight [out, in], output is along dim=1 (rows)
                    base_row_norms = torch.norm(base_layer.weight, p=2, dim=1)
                else:
                    # Fallback for other layer types
                    base_row_norms = torch.norm(base_layer.weight, p=2, dim=0)

            if hasattr(adapter_state, 'total_energy') and adapter_state.total_energy is not None:
                total_energy = adapter_state.total_energy
            else:
                # Fallback: compute total energy (shouldn't happen in normal use)
                total_energy = (base_row_norms.float() ** 2).sum()

            # Compute magnitude scale and store metadata
            from .magnitude import compute_ecd_scale, compute_oer_scale_softmax

            mag = getattr(adapter_state.cfg, "magnitude", "none")
            if mag == "ecd_tanh":
                scale = compute_ecd_scale(
                    base_row_norms=base_row_norms,
                    total_energy=total_energy,
                    ecd_log_mag=adapter_state.ecd_log_mag,
                    ecd_alpha=adapter_state.cfg.ecd_alpha,
                    temperature=adapter_state.cfg.ecd_temperature,
                    eps=adapter_state.cfg.eps,
                ).to(delta_weight.dtype)
            elif mag == "oer_softmax":
                scale = compute_oer_scale_softmax(
                    base_row_norms=base_row_norms,
                    total_energy=total_energy,
                    oer_logits=adapter_state.ecd_log_mag,
                    temperature=adapter_state.cfg.oer_temperature,
                    eps=adapter_state.cfg.eps,
                ).to(delta_weight.dtype)
            else:
                # No magnitude scaling
                return delta_weight, None

        # Apply scaling (broadcast to match weight dimensions)
        # scale shape: [n_out], delta_weight shape: [n_out, n_in]
        scaled_delta = delta_weight * scale.unsqueeze(-1)

        if reuse_metadata is not None:
            return scaled_delta, None
        else:
            # Return both scaled delta and metadata for unmerge
            metadata = {
                'scale': scale,
                'base_row_norms': base_row_norms,
                'magnitude_type': getattr(adapter_state.cfg, "magnitude", "none")
            }
            return scaled_delta, metadata

    def _unmerge_single_adapter(self, adapter_name: str) -> None:
        """
        Unmerge a single JORA adapter by reversing the merge operation.

        Args:
            adapter_name: Name of the adapter to unmerge
        """
        adapter_state = self.adapters[adapter_name]
        base_layer = self.get_base_layer()

        # Recompute the same delta weight that was added during merging
        delta_weight = self._compute_weight_delta_simple(adapter_state)

        # Apply magnitude scaling if it was used during merging
        # Reuse the metadata that was computed during merge for consistency
        reuse_metadata = None
        if hasattr(adapter_state, '_merge_metadata') and adapter_name in adapter_state._merge_metadata:
            metadata = adapter_state._merge_metadata[adapter_name]
            if metadata.get('magnitude_metadata') is not None:
                reuse_metadata = metadata['magnitude_metadata']

        if reuse_metadata is not None:
            delta_weight, _ = self._apply_magnitude_to_delta_weight(delta_weight, adapter_state, reuse_metadata=reuse_metadata)
        elif adapter_state.ecd_log_mag is not None:
            # Fallback: recompute metadata (less accurate but better than nothing)
            delta_weight, _ = self._apply_magnitude_to_delta_weight(delta_weight, adapter_state)

        # Subtract from base weights to reverse the merge
        delta_weight = delta_weight.to(base_layer.weight.dtype)
        base_layer.weight.data -= delta_weight

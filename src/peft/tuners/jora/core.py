from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class SelectiveDiagCore(nn.Module):
    """Proposed JORA core: diagonal applied ONLY to a fixed support U of size |U|.

    This is the paper-path core. It stores a maximum of |U| trainable parameters
    (`delta`) and tracks how many support entries are actually active after
    calibration. At inference it computes:
        y[U_active] = x[U_active] * (1 + delta[:|U_active|])
        y[~U_active] = 0

    The caller (compute_delta) then computes:
        delta = R_L^T @ y - P_U @ x
    which equals R_L^T @ D_sel @ R_R @ x - P_U @ x for the active support.

    At allocation, delta=0 and theta=0 so the whole adapter starts at
    zero-function change.
    """

    def __init__(self, support_size: int, device, dtype):
        super().__init__()
        self.support_size = int(support_size)
        self.delta = nn.Parameter(torch.zeros(self.support_size, device=device, dtype=dtype))
        self.register_buffer("support_indices", torch.zeros(self.support_size, dtype=torch.long, device=device))
        self.register_buffer("active_support_size", torch.zeros((), dtype=torch.long, device=device))
        self._active_support_size_py = 0
        self.register_load_state_dict_post_hook(self._sync_runtime_state)

    def _sync_runtime_state(self, *_args, **_kwargs) -> None:
        self._active_support_size_py = int(self.active_support_size.item())

    def set_support(self, indices: Tensor) -> None:
        """Set the support indices U after calibration.

        Accepts any number of unique indices up to `support_size`. When fewer than
        `support_size` indices are available we keep a masked tail instead of
        padding with duplicates, which avoids aliasing multiple delta parameters
        onto the same feature.
        """
        indices = indices.to(device=self.support_indices.device, dtype=torch.long).reshape(-1)
        assert indices.numel() <= self.support_size, (
            f"Expected at most {self.support_size} indices, got {indices.numel()}"
        )
        if indices.numel() > 0:
            assert torch.unique(indices).numel() == indices.numel(), "Support indices must be unique"

        self.support_indices.zero_()
        n_active = int(indices.numel())
        if n_active > 0:
            self.support_indices[:n_active].copy_(indices)
        self.active_support_size.fill_(n_active)
        self._active_support_size_py = n_active

        if n_active < self.support_size:
            with torch.no_grad():
                self.delta[n_active:].zero_()

    def apply_to_vector(self, x: Tensor) -> Tensor:
        """Apply D_sel = I_U + diag(delta) to x, returning a vector of the same dim as x."""
        n_active = self._active_support_size_py
        y = torch.zeros_like(x)
        if n_active <= 0:
            return y
        u = self.support_indices[:n_active]
        # Keep the support scaling in the same dtype/device as the activation.
        # In bf16 training, `1.0 + delta` can promote to float and break indexed writes
        # back into `y`, which is allocated with `zeros_like(x)`.
        delta = self.delta[:n_active].to(device=x.device, dtype=x.dtype)
        scale = torch.ones_like(delta) + delta
        y[..., u] = x[..., u] * scale
        return y

    def project_support(self, x: Tensor) -> Tensor:
        """Apply projection P_U (identity on active support, zero elsewhere)."""
        n_active = self._active_support_size_py
        y = torch.zeros_like(x)
        if n_active <= 0:
            return y
        u = self.support_indices[:n_active]
        y[..., u] = x[..., u]
        return y

    @property
    def num_params(self) -> int:
        return self.support_size


class CoupledPairCore(nn.Module):
    """2x2 coupled blocks on each rotation pair for increased expressiveness.

    WARNING: This is an EXPLORATORY variant — NOT paper-path.
        CoupledPairCore uses the LEGACY adapter path in compute_delta():
        delta = R_L^T @ core(R_R @ x)  (without residualization)
        This is DIFFERENT from SelectiveDiagCore's paper-exact path:
        delta = R_L^T @ D_sel @ R_R @ x - P_U @ x
        Key consequence: at zero-init, CoupledPairCore does NOT give zero-function-change
        because the subtraction (P_U @ x) is missing. Use SelectiveDiagCore for paper.

    Motivation:
        JORA uses rotation pairs (i, j) where R builds coupling between dimensions i and j.
        However, SelectiveDiagCore applies independent diagonal scaling to each dimension,
        which decouples what rotation coupled. This is contradictory.

    Solution:
        Replace diag(delta[i]) with a 2x2 block that captures within-pair coupling:

            [y_i]   [1+δ_ii  δ_ij] [x_i]
            [y_j] = [δ_ji   1+δ_jj] [x_j]

    Pair Structure (Critical):
        Unlike SelectiveDiagCore which just needs unique indices (order doesn't matter),
        CoupledPairCore MUST preserve which indices form a pair. This is because
        the 2x2 block is only meaningful when it couples dimensions that were
        already coupled by the rotation.

        The pair structure flows through:
        1. Rotation selection → pairs_L [[i,j], [k,l], ...], pairs_R [[i,j], [k,l], ...]
        2. set_support_pairs(pairs) → stores pairs [n_pairs, 2] internally
        3. apply_to_vector → couples (u[2p], u[2p+1]) for each pair

        IMPORTANT: Never use torch.unique() on the indices before calling set_support_pairs,
        as this destroys the pair structure.

    API:
        - set_support_pairs(pairs): Set from rotation pairs [n_pairs, 2]
        - set_support(indices_interleaved): Legacy compat — interleaved as [i,j,k,l,...]

    For the same k value:
        - SelectiveDiagCore: k pairs → support_size=2k → params=2k
        - CoupledPairCore:   k pairs → support_size=2k → n_pairs=k → params=4k
          (More params, but captures coupling within each rotation pair)
    """

    def __init__(self, n_pairs: int, device, dtype):
        super().__init__()
        self.n_pairs = int(n_pairs)
        # support_size = 2 * n_pairs for full pairs
        self.support_size = self.n_pairs * 2
        # [n_pairs, 2, 2], zero-init for identity start
        # block[p, 0, 0] = δ_ii, block[p, 0, 1] = δ_ij, etc.
        self.pair_blocks = nn.Parameter(
            torch.zeros(self.n_pairs, 2, 2, device=device, dtype=dtype)
        )
        # Store pairs [n_pairs, 2] directly — critical for preserving structure
        self.register_buffer("pairs", torch.zeros(self.n_pairs, 2, dtype=torch.long, device=device))
        self.register_buffer("active_n_pairs", torch.zeros((), dtype=torch.long, device=device))
        self._active_n_pairs_py = 0
        self.register_load_state_dict_post_hook(self._sync_runtime_state)

    def _sync_runtime_state(self, *_args, **_kwargs) -> None:
        self._active_n_pairs_py = int(self.active_n_pairs.item())

    def set_support_pairs(self, pairs: Tensor) -> None:
        """Set support from rotation pairs [n_pairs, 2].

        This is the PRIMARY interface for CoupledPairCore. It preserves the
        pair structure that rotation selection established.

        Args:
            pairs: Tensor of shape [n_active_pairs, 2] where pairs[p] = [i, j]
                   are the two dimensions coupled by rotation p.
        """
        pairs = pairs.to(device=self.pairs.device, dtype=torch.long)
        if pairs.ndim == 1:
            pairs = pairs.reshape(-1, 2)
        n_active = int(pairs.shape[0])
        assert n_active <= self.n_pairs, (
            f"Expected at most {self.n_pairs} pairs, got {n_active}"
        )

        self.pairs.zero_()
        if n_active > 0:
            self.pairs[:n_active].copy_(pairs)
        self.active_n_pairs.fill_(n_active)
        self._active_n_pairs_py = n_active

        # Zero out pair blocks for inactive pairs
        with torch.no_grad():
            self.pair_blocks[n_active:].zero_()

    def set_support(self, indices_interleaved: Tensor) -> None:
        """Set support from interleaved indices [i0, j0, i1, j1, ...].

        This is a LEGACY compatibility interface for SelectiveDiagCore-style callers.
        It converts interleaved indices back to pairs and calls set_support_pairs.
        WARNING: Only use this when indices are already interleaved (not unique sorted).
        """
        indices = indices_interleaved.to(device=self.pairs.device, dtype=torch.long).reshape(-1)
        n_indices = int(indices.numel())
        n_pairs = n_indices // 2
        assert n_indices % 2 == 0, f"Interleaved indices must be even, got {n_indices}"

        if n_pairs > 0:
            pairs = indices[:n_pairs * 2].reshape(n_pairs, 2)
            self.set_support_pairs(pairs)
        else:
            self.set_support_pairs(torch.zeros(0, 2, dtype=torch.long, device=self.pairs.device))

    def apply_to_vector(self, x: Tensor) -> Tensor:
        """Apply 2x2 coupled blocks to rotation pairs (vectorized).

        For each pair (i, j) in support:
            y[i] = x[i] * (1 + δ_ii) + x[j] * δ_ij
            y[j] = x[i] * δ_ji + x[j] * (1 + δ_jj)

        With zero-init (δ=0), this acts as identity.
        """
        n_active = self._active_n_pairs_py
        y = torch.zeros_like(x)
        if n_active <= 0:
            return y

        # Get active pairs [n_active, 2]
        active_pairs = self.pairs[:n_active]  # [n_active, 2]

        # Gather paired values: for each pair [i, j], extract x[..., i] and x[..., j]
        # Using advanced indexing
        idx0 = active_pairs[:, 0]  # [n_active]
        idx1 = active_pairs[:, 1]  # [n_active]

        # Extract: x_pairs[p] = [x[..., i_p], x[..., j_p]]
        x0 = x[..., idx0]  # [..., n_active]
        x1 = x[..., idx1]  # [..., n_active]

        # Get blocks and add identity: [n_active, 2, 2]
        blocks = self.pair_blocks[:n_active].to(device=x.device, dtype=x.dtype)
        eye = torch.eye(2, device=x.device, dtype=x.dtype).unsqueeze(0)
        transform = blocks + eye  # [n_active, 2, 2]

        # Batched matmul along the last dim: each 1x2 row of transform applies to [x0, x1]
        # [x0, x1] @ transform.T gives the coupled output
        # Using einsum: [..., p] x [p, i, j] -> [..., p, i] then sum over i
        x_in = torch.stack([x0, x1], dim=-1)  # [..., n_active, 2]
        y_out = torch.einsum('...pi,pij->...pj', x_in, transform)  # [..., n_active, 2]

        # Scatter back
        y[..., idx0] = y_out[..., 0]
        y[..., idx1] = y_out[..., 1]

        return y

    def project_support(self, x: Tensor) -> Tensor:
        """Apply projection P_U (identity on active support, zero elsewhere).

        For CoupledPairCore, projecting means setting non-support dimensions to zero.
        """
        n_active = self._active_n_pairs_py
        y = torch.zeros_like(x)
        if n_active <= 0:
            return y
        idx = self.pairs[:n_active].reshape(-1)  # Interleaved indices
        y[..., idx] = x[..., idx]
        return y

    @property
    def num_params(self) -> int:
        """Number of trainable parameters: n_pairs * 4 (each 2x2 block)."""
        return self.n_pairs * 4


class DiagCore(nn.Module):
    """Diagonal core D (stored as its diagonal), in additive form.

    The core stores Diag(d) as diag_params, where d is the trainable
    deviation from zero. At init (d=0), the core acts as zero-operator
    (additive form: Δ(x) = Diag(d) @ x, so Δ(x) = 0 when d = 0).
    """

    def __init__(self, n: int, m: int, device, dtype, zero_init: bool = False, init_std: float = 5e-3):
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        d_size = min(self.n, self.m)
        if zero_init:
            self.diag_params = nn.Parameter(torch.zeros(d_size, device=device, dtype=dtype))
        else:
            self.diag_params = nn.Parameter(torch.randn(d_size, device=device, dtype=dtype) * init_std)

    def forward(self) -> Tensor:
        warnings.warn(
            "DiagCore.forward() generates a full dense matrix which consumes significant memory "
            "(e.g., 64MB for 4096x4096). This method is deprecated and should be avoided. "
            "Use apply_to_vector() or get_row_slice() for memory-efficient operations instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Residualized form: D = I + Diag(d)
        # Only square part (min(n,m) x min(n,m)) is active; rest is zeros.
        d_len = self.diag_params.size(0)
        I = torch.eye(d_len, device=self.diag_params.device, dtype=self.diag_params.dtype)
        diag_matrix = torch.diag(self.diag_params)
        D = torch.zeros(self.n, self.m, device=self.diag_params.device, dtype=self.diag_params.dtype)
        D[:d_len, :d_len] = I + diag_matrix
        return D

    def get_row_slice(self, start: int, end: int) -> Tensor:
        # Residualized form: D = I + Diag(d)
        # Row i has only one nonzero at column i: value 1 + d[i] (if i < d_len)
        rows = end - start
        D_slice = torch.zeros(rows, self.m, device=self.diag_params.device, dtype=self.diag_params.dtype)
        if rows <= 0:
            return D_slice
        d_len = self.diag_params.size(0)
        for r in range(rows):
            i = start + r
            if i < d_len and i < self.m:
                D_slice[r, i] = 1.0 + self.diag_params[i]
            elif i < self.m:
                D_slice[r, i] = 1.0  # identity part only (d=0 beyond learned range)
        return D_slice

    def apply_to_vector(self, x: Tensor) -> Tensor:
        # Core output: D @ x where D = I + Diag(d).
        # Returns (I + Diag(d)) @ x = x + d * x (elementwise, learned range).
        # At d=0 (zero_init_core=True): returns x. Delta will then be
        # delta = R_L^T @ x @ R_R - x (subtracted in compute_delta).
        # Wait — no subtraction here. compute_delta() computes the additive
        # delta: delta = R_L^T @ apply_to_vector(x_rot) @ R_R.
        # For zero-function-change at init, apply_to_vector should return x at d=0.
        # But the additive formula is: delta = R_L^T @ Diag(d) @ R_R @ x.
        # At d=0: delta = 0. So apply_to_vector should return Diag(d) @ x.
        d_len = self.diag_params.size(0)
        x_first = x[..., :d_len]
        y_first = x_first * self.diag_params  # = d * x (not (1+d) * x)
        if self.n > d_len:
            n_pad = self.n - d_len
            y_rest = torch.zeros(
                (*x.shape[:-1], n_pad), device=x.device, dtype=x.dtype,
            )
            return torch.cat([y_first, y_rest], dim=-1)
        return y_first

class BlockCore(nn.Module):
    """Block-diagonal core: dense blocks along diagonal + optional remainder diagonal."""

    def __init__(self, n: int, m: int, device, dtype, block_size: int = 4, zero_init: bool = False, init_std: float = 5e-3):
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        self.block_size = int(block_size)
        d_size = min(self.n, self.m)

        self.n_blocks = d_size // self.block_size
        self.remainder_size = d_size % self.block_size

        if self.n_blocks > 0:
            init = torch.zeros(self.n_blocks, self.block_size, self.block_size, device=device, dtype=dtype)
            if not zero_init:
                # Initialize as identity + small noise for near-identity start
                init = torch.randn_like(init) * init_std
            self.blocks = nn.Parameter(init)
        else:
            self.register_parameter("blocks", None)

        if self.remainder_size > 0:
            init_r = torch.zeros(self.remainder_size, device=device, dtype=dtype)
            if not zero_init:
                init_r = torch.randn_like(init_r) * init_std
            self.diag_remainder = nn.Parameter(init_r)
        else:
            self.register_parameter("diag_remainder", None)

    def forward(self) -> Tensor:
        warnings.warn(
            "BlockCore.forward() generates a full dense matrix which consumes significant memory. "
            "This method is deprecated and should be avoided. "
            "Use apply_to_vector() or get_row_slice() for memory-efficient operations instead.",
            DeprecationWarning,
            stacklevel=2
        )
        D = torch.zeros(self.n, self.m, device=self.blocks.device if self.blocks is not None else self.diag_remainder.device,
                        dtype=self.blocks.dtype if self.blocks is not None else self.diag_remainder.dtype)
        if self.blocks is not None:
            for b in range(self.n_blocks):
                start = b * self.block_size
                end = start + self.block_size
                D[start:end, start:end] = self.blocks[b]
        if self.diag_remainder is not None:
            start = self.n_blocks * self.block_size
            end = start + self.remainder_size
            D[start:end, start:end] = torch.diag(self.diag_remainder)
        return D

    def get_row_slice(self, start: int, end: int) -> Tensor:
        rows = end - start
        device = self.blocks.device if self.blocks is not None else self.diag_remainder.device
        dtype = self.blocks.dtype if self.blocks is not None else self.diag_remainder.dtype
        D_slice = torch.zeros(rows, self.m, device=device, dtype=dtype)
        if rows <= 0:
            return D_slice

        gidx = torch.arange(start, end, device=device)
        block_region_end = self.n_blocks * self.block_size

        mask_block = gidx < block_region_end
        if mask_block.any() and self.blocks is not None:
            g_block = gidx[mask_block]
            row_sel = torch.arange(rows, device=device)[mask_block]
            block_idx = torch.div(g_block, self.block_size, rounding_mode='floor')
            in_block_idx = g_block % self.block_size

            unique_blocks = torch.unique(block_idx)
            for b in unique_blocks.tolist():
                b = int(b)
                rows_in_b = (block_idx == b)
                if not rows_in_b.any():
                    continue
                rsel = row_sel[rows_in_b]
                i_in = in_block_idx[rows_in_b]
                col_start = b * self.block_size
                col_end = col_start + self.block_size
                D_slice[rsel[:, None], torch.arange(col_start, col_end, device=device)[None, :]] = self.blocks[b][i_in]

        if self.diag_remainder is not None:
            remainder_start = block_region_end
            mask_rem = (gidx >= remainder_start) & (gidx < remainder_start + self.remainder_size)
            if mask_rem.any():
                row_sel = torch.arange(rows, device=device)[mask_rem]
                cols = gidx[mask_rem]
                rem_idx = cols - remainder_start
                D_slice[row_sel, cols] = self.diag_remainder[rem_idx]
        return D_slice

    def apply_to_vector(self, x: Tensor) -> Tensor:
        # y = x @ D^T (block-diagonal => block-wise matmul)
        # Vectorized implementation for better GPU utilization

        # Handle block-diagonal part with vectorized operations
        if self.blocks is not None:
            # Stack all blocks and transpose them: [n_blocks, bs, bs] -> [n_blocks, bs, bs]
            blocks_t = torch.stack([b.t() for b in self.blocks], dim=0)  # [n_blocks, bs, bs]

            # Reshape input for vectorized computation
            block_region_end = self.n_blocks * self.block_size
            x_blocks = x[..., :block_region_end]  # [..., n_blocks * bs]

            # Reshape to [..., n_blocks, bs] for batched matrix multiplication
            x_blocks_reshaped = x_blocks.view(*x.shape[:-1], self.n_blocks, self.block_size)

            # Vectorized block-wise matrix multiplication: [..., n_blocks, bs] @ [n_blocks, bs, bs]
            # Result: [..., n_blocks, bs] -> flatten back to [..., n_blocks * bs]
            y_blocks = torch.einsum('...nbk,bkj->...nbj', x_blocks_reshaped, blocks_t)
            y_blocks = y_blocks.reshape(*x.shape[:-1], -1)  # [..., n_blocks * bs]
        else:
            y_blocks = x.new_zeros((*x.shape[:-1], 0))
            block_region_end = 0

        # Handle remainder diagonal part
        if self.diag_remainder is not None:
            start = block_region_end
            end = start + self.remainder_size
            xr = x[..., start:end]  # [..., remainder_size]
            yr = xr * self.diag_remainder  # Element-wise multiplication
            y_remainder = yr
        else:
            y_remainder = x.new_zeros((*x.shape[:-1], 0))

        # Concatenate block and remainder parts
        y = torch.cat([y_blocks, y_remainder], dim=-1)

        # Pad if necessary (should not happen with correct initialization)
        if y.shape[-1] < self.n:
            pad = x.new_zeros((*x.shape[:-1], self.n - y.shape[-1]))
            y = torch.cat([y, pad], dim=-1)

        return y

class LowRankCore(nn.Module):
    """Low-rank core D = A @ B^T, with LoRA-style scaling."""

    def __init__(self, n: int, m: int, device, dtype, rank: int = 8, zero_init: bool = False, init_std: float = 5e-3):
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        self.r = int(rank)
        self.scaling = 1.0
        if self.r <= 0:
            raise ValueError("lowrank_r must be > 0")

        if zero_init:
            # LoRA-style: A=zero, B=small-random → zero-operator at init with live gradients
            A = torch.zeros(self.n, self.r, device=device, dtype=dtype)
            B = torch.randn(self.m, self.r, device=device, dtype=dtype) * init_std
        else:
            # Symmetry-breaking: both A and B small-random
            A = torch.randn(self.n, self.r, device=device, dtype=dtype) * init_std
            B = torch.randn(self.m, self.r, device=device, dtype=dtype) * init_std

        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)

    def forward(self) -> Tensor:
        return self.scaling * (self.A @ self.B.t())

    def get_row_slice(self, start: int, end: int) -> Tensor:
        return self.scaling * (self.A[start:end] @ self.B.t())

    def apply_to_vector(self, x: Tensor) -> Tensor:
        # y = x @ D^T = (x @ B) @ A^T
        xb = torch.matmul(x, self.B)  # [..., r]
        y = torch.matmul(xb, self.A.t())  # [..., n]
        return self.scaling * y

def build_core(core_type: str, n: int, m: int, device, dtype, cfg) -> nn.Module:
    if core_type == "selective_diag":
        # Paper-exact core: stores only |U| = k*2 params, applies to support indices U only.
        # support_indices must be set via set_support() after calibration.
        support_size = int(getattr(cfg, "k", 8)) * 2  # k pairs -> 2k support indices
        return SelectiveDiagCore(support_size=support_size, device=device, dtype=dtype)
    if core_type == "diag":
        init_std = float(getattr(cfg, 'core_init_std', 5e-3))
        return DiagCore(n, m, device=device, dtype=dtype, zero_init=getattr(cfg, "zero_init_core", False), init_std=init_std)
    if core_type == "block":
        init_std = float(getattr(cfg, 'core_init_std', 5e-3))
        return BlockCore(n, m, device=device, dtype=dtype,
                         block_size=int(getattr(cfg, "block_size", 4)),
                         zero_init=getattr(cfg, "zero_init_core", False), init_std=init_std)
    if core_type == "lowrank":
        r = int(getattr(cfg, "lowrank_r", 8))
        init_std = float(getattr(cfg, 'core_init_std', 5e-3))
        core = LowRankCore(n, m, device=device, dtype=dtype, rank=r,
                           zero_init=getattr(cfg, "zero_init_core", False), init_std=init_std)
        alpha = getattr(cfg, "lowrank_alpha", None)
        if alpha is None:
            alpha = r
        core.scaling = float(alpha) / float(r) if r > 0 else 1.0
        return core
    if core_type == "coupled_pair":
        # Scheme 3: 2x2 coupled blocks on rotation pairs
        #
        # k = number of pairs (from config)
        # For SelectiveDiagCore: k pairs → support_size = 2k → params = 2k
        # For CoupledPairCore:   k pairs → n_pairs = k → support_size = 2k → params = 4k
        #
        # The critical difference: CoupledPairCore preserves pair structure
        # and applies coupled transformations within each rotation subspace.
        #
        # Comparison groups (same k means same rotation selection):
        #   A: selective_diag @ k=16  → support_size=32, params=32   (paper-path, exact merge)
        #   B: coupled_pair  @ k=16  → n_pairs=16,   params=64   (legacy-path, approx merge)
        #
        # IMPORTANT comparison rules:
        #   A vs B: UNFAIR — different adapter paths AND different param counts
        #   A vs C: FAIR   — same params (32), different structure
        #
        # For matched-param comparison, use:
        #   C: selective_diag @ k=16  → params=32
        #   D: coupled_pair  @ k=8   → n_pairs=8,    params=32   (same params, half coverage)
        n_pairs = int(getattr(cfg, "k", 8))
        return CoupledPairCore(n_pairs=n_pairs, device=device, dtype=dtype)
    raise ValueError(f"Unknown core type: {core_type}")

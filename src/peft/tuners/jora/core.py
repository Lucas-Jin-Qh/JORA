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


class DiagCore(nn.Module):
    """Diagonal core D (stored as its diagonal)."""

    def __init__(self, n: int, m: int, device, dtype, zero_init: bool = False):
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        d_size = min(self.n, self.m)
        if zero_init:
            self.diag_params = nn.Parameter(torch.zeros(d_size, device=device, dtype=dtype))
        else:
            # Break zero-gradient deadlock: small random init
            self.diag_params = nn.Parameter(0.01 * torch.randn(d_size, device=device, dtype=dtype))

    def forward(self) -> Tensor:
        warnings.warn(
            "DiagCore.forward() generates a full dense matrix which consumes significant memory "
            "(e.g., 64MB for 4096x4096). This method is deprecated and should be avoided. "
            "Use apply_to_vector() or get_row_slice() for memory-efficient operations instead.",
            DeprecationWarning,
            stacklevel=2
        )
        D = torch.zeros(self.n, self.m, device=self.diag_params.device, dtype=self.diag_params.dtype)
        d_len = self.diag_params.size(0)
        D[:d_len, :d_len] = torch.diag(self.diag_params)
        return D

    def get_row_slice(self, start: int, end: int) -> Tensor:
        rows = end - start
        D_slice = torch.zeros(rows, self.m, device=self.diag_params.device, dtype=self.diag_params.dtype)
        if rows <= 0:
            return D_slice
        d_len = self.diag_params.size(0)
        for r in range(rows):
            i = start + r
            if i < d_len and i < self.m:
                D_slice[r, i] = self.diag_params[i]
        return D_slice

    def apply_to_vector(self, x: Tensor) -> Tensor:
        # y = x @ D^T; for diagonal, this is elementwise on first min(n,m)
        d_len = self.diag_params.size(0)
        y_first = x[..., :d_len] * self.diag_params
        if self.n > d_len:
            pad_shape = (*x.shape[:-1], self.n - d_len)
            pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
            return torch.cat([y_first, pad], dim=-1)
        return y_first

class BlockCore(nn.Module):
    """Block-diagonal core: dense blocks along diagonal + optional remainder diagonal."""

    def __init__(self, n: int, m: int, device, dtype, block_size: int = 4, zero_init: bool = False):
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
                init = 0.1 * torch.randn_like(init)
            self.blocks = nn.Parameter(init)
        else:
            self.register_parameter("blocks", None)

        if self.remainder_size > 0:
            init_r = torch.zeros(self.remainder_size, device=device, dtype=dtype)
            if not zero_init:
                init_r = 0.1 * torch.randn_like(init_r)
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

    def __init__(self, n: int, m: int, device, dtype, rank: int = 8, zero_init: bool = False):
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        self.r = int(rank)
        self.scaling = 1.0
        if self.r <= 0:
            raise ValueError("lowrank_r must be > 0")

        if zero_init:
            # Preserve an exact zero operator at init without deadlocking both factors.
            # Setting A=B=0 makes y=0, but also makes grad_A=grad_B=0 forever.
            # Match LoRA-style init instead: keep the "up" factor zero and the
            # "down" factor random so the first backward step can update A.
            A = torch.zeros(self.n, self.r, device=device, dtype=dtype)
            B = 0.1 * torch.randn(self.m, self.r, device=device, dtype=dtype)
        else:
            A = 0.1 * torch.randn(self.n, self.r, device=device, dtype=dtype)
            B = 0.1 * torch.randn(self.m, self.r, device=device, dtype=dtype)

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
        return DiagCore(n, m, device=device, dtype=dtype, zero_init=getattr(cfg, "zero_init_core", False))
    if core_type == "block":
        return BlockCore(n, m, device=device, dtype=dtype,
                         block_size=int(getattr(cfg, "block_size", 4)),
                         zero_init=getattr(cfg, "zero_init_core", False))
    if core_type == "lowrank":
        r = int(getattr(cfg, "lowrank_r", 8))
        core = LowRankCore(n, m, device=device, dtype=dtype, rank=r, zero_init=getattr(cfg, "zero_init_core", False))
        alpha = getattr(cfg, "lowrank_alpha", None)
        if alpha is None:
            alpha = r
        core.scaling = float(alpha) / float(r) if r > 0 else 1.0
        return core
    raise ValueError(f"Unknown core type: {core_type}")

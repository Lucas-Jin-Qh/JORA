from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

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
        parts = []
        # blocks
        if self.blocks is not None:
            for b in range(self.n_blocks):
                start = b * self.block_size
                end = start + self.block_size
                xb = x[..., start:end]  # [..., bs]
                # D block is [bs,bs], we need D^T
                yb = torch.matmul(xb, self.blocks[b].t())
                parts.append(yb)
        # remainder diag
        if self.diag_remainder is not None:
            start = self.n_blocks * self.block_size
            end = start + self.remainder_size
            xr = x[..., start:end]
            yr = xr * self.diag_remainder
            parts.append(yr)
        y = torch.cat(parts, dim=-1) if parts else x.new_zeros((*x.shape[:-1], self.n))
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
            A = torch.zeros(self.n, self.r, device=device, dtype=dtype)
            B = torch.zeros(self.m, self.r, device=device, dtype=dtype)
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

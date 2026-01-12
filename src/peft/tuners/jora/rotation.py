from __future__ import annotations

from typing import Tuple, Optional, List

import torch
from torch import Tensor

# Optional Triton path (kept interface-compatible; kernel lives in legacy code).
try:  # pragma: no cover
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    TRITON_AVAILABLE = False
    triton = None  # type: ignore
    tl = None  # type: ignore

def cayley_cos_sin(theta: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute cos/sin via Cayley parameterization.

    phi = 2 * atan(theta/2)
    c = cos(phi), s = sin(phi)
    """
    phi = 2.0 * torch.atan(0.5 * theta)
    return torch.cos(phi), torch.sin(phi)

def _cos_sin(theta: Tensor, rotation_param: str) -> Tuple[Tensor, Tensor]:
    if rotation_param == "cayley":
        return cayley_cos_sin(theta)
    return torch.cos(theta), torch.sin(theta)

def _validate_pairs(pairs: Tensor, dim: int) -> None:
    """Fail fast if pairs contain invalid indices (e.g., -1 sentinels).

    This prevents silent corruption from Python's negative indexing semantics.
    Overhead is negligible because S is typically small (<= 32/64).
    """
    if pairs.numel() == 0:
        return
    dim = int(dim)
    if (pairs < 0).any():
        raise ValueError(
            "pairs contains negative indices (e.g., -1). "
            "Did you forget to slice pairs[:num_pairs] before calling apply_rotations?"
        )
    if (pairs >= dim).any():
        raise ValueError(f"pairs contains out-of-range indices for dim={dim}.")

def apply_rotations_torch(
    x: Tensor,
    pairs: Tensor,
    thetas: Tensor,
    *,
    reverse: bool = False,
    rotation_param: str = "cayley",
    negate_theta: bool = False,
) -> Tensor:
    """Apply a sequence of 2D rotations specified by (pairs, thetas).

    High-performance vectorized implementation: single clone + parallel rotations, avoid GPU-CPU sync.

    Args:
        x: (..., dim)
        pairs: (k, 2) long indices into last dim
        thetas: (k,) rotation parameters
    """
    if pairs.numel() == 0:
        return x

    if pairs.dim() != 2 or pairs.size(-1) != 2:
        raise ValueError(f"pairs must be (k, 2), got {tuple(pairs.shape)}")

    dim = int(x.shape[-1])
    _validate_pairs(pairs, dim)

    # Prepare data: clone once only, avoid per-step cloning
    y = x.view(-1, dim).clone()  # [batch*seq, dim]

    # Handle reverse and parameters
    if reverse:
        pairs = torch.flip(pairs, dims=[0])
        thetas = torch.flip(thetas, dims=[0])

    # Extract all pair indices (vectorized)
    i = pairs[:, 0].long()  # [k]
    j = pairs[:, 1].long()  # [k]

    # Process theta parameters
    th = thetas.view(-1).to(dtype=y.dtype, device=y.device)  # [k]
    if negate_theta:
        th = -th

    # Compute rotation matrix elements
    if rotation_param == "cayley":
        phi = 2.0 * torch.atan(0.5 * th)
        c = torch.cos(phi)
        s = torch.sin(phi)
    else:
        c = torch.cos(th)
        s = torch.sin(th)

    # Vectorized extraction of columns to rotate: [batch*seq, k]
    yi = y.index_select(1, i)  # [batch*seq, k]
    yj = y.index_select(1, j)  # [batch*seq, k]

    # Broadcast rotation parameters: [batch*seq, k]
    c = c.unsqueeze(0)  # [1, k]
    s = s.unsqueeze(0)  # [1, k]

    # Apply Givens rotations (fully vectorized)
    new_yi = c * yi + s * yj
    new_yj = -s * yi + c * yj

    # Write back results (disjoint pairs guarantee safety)
    y.index_copy_(1, i, new_yi)
    y.index_copy_(1, j, new_yj)

    return y.view_as(x)

# Triton implementation (simplified for stability)
if TRITON_AVAILABLE:
    @triton.jit
    def apply_givens_rotations_kernel(
        x_ptr,          # Input/output [total_tokens, features]
        pairs_ptr,      # Rotation pairs [n_pairs, 2]
        cos_ptr,        # cos values [n_pairs]
        sin_ptr,        # sin values [n_pairs]
        n_tokens,       # Total token count (B*L)
        n_features,     # Feature dimension
        n_pairs,        # Rotation steps S
        reverse: tl.constexpr,  # Whether to traverse in reverse (for transpose)
        BLOCK_M: tl.constexpr,  # Tokens processed per program
    ):
        """
        Efficient Givens rotation kernel (column pointer mode).
        Only loads required columns, avoids full row movement, supports arbitrary large feature dimensions.
        """
        pid = tl.program_id(0)
        start_row = pid * BLOCK_M
        offsets_m = start_row + tl.arange(0, BLOCK_M)
        mask_m = offsets_m < n_tokens

        # Row base addresses (pointer vector): x[offsets_m, 0]
        row_base_ptrs = x_ptr + offsets_m * n_features

        for k in range(n_pairs):
            idx = n_pairs - 1 - k if reverse else k

            # Load pair indices and rotation parameters
            idx_i = tl.load(pairs_ptr + idx * 2)
            idx_j = tl.load(pairs_ptr + idx * 2 + 1)
            c = tl.load(cos_ptr + idx).to(tl.float32)
            s = tl.load(sin_ptr + idx).to(tl.float32)

            # Construct pointers for both columns
            ptrs_i = row_base_ptrs + idx_i
            ptrs_j = row_base_ptrs + idx_j

            # Load both columns
            val_i = tl.load(ptrs_i, mask=mask_m, other=0.0).to(tl.float32)
            val_j = tl.load(ptrs_j, mask=mask_m, other=0.0).to(tl.float32)

            # Rotate: [new_i, new_j] = [[c, s], [-s, c]] @ [i, j]
            new_i = c * val_i + s * val_j
            new_j = -s * val_i + c * val_j

            # Write back to memory (overwrite original values)
            tl.store(ptrs_i, new_i, mask=mask_m)
            tl.store(ptrs_j, new_j, mask=mask_m)

    class GivensRotationTriton(torch.autograd.Function):
        """
        Complete Givens rotation Autograd operator.
        """

        @staticmethod
        def forward(ctx, x, pairs, thetas, use_cayley, reverse: bool = False):
            # Save information needed for backward
            ctx.save_for_backward(x, pairs, thetas)
            ctx.use_cayley = use_cayley
            ctx.reverse = reverse

            # Handle input dimensions
            original_shape = x.shape
            if x.ndim == 2:
                x = x.unsqueeze(1)

            batch, seq_len, features = x.shape
            n_pairs = pairs.size(0)

            if n_pairs == 0:
                return x.view(original_shape)

            # Ensure contiguity
            if not x.is_contiguous():
                x = x.contiguous()

            # Pre-compute cos/sin
            if use_cayley:
                phi = 2.0 * torch.atan(0.5 * thetas.float())
                cos_vals = torch.cos(phi).to(x.dtype)
                sin_vals = torch.sin(phi).to(x.dtype)
            else:
                cos_vals = torch.cos(thetas.float()).to(x.dtype)
                sin_vals = torch.sin(thetas.float()).to(x.dtype)

            # Save cos/sin for backward use
            ctx.cos_vals = cos_vals
            ctx.sin_vals = sin_vals

            # Output: clone to ensure forward safety
            out = x.clone()

            # Call Triton Kernel (optimized block size)
            total_tokens = batch * seq_len
            # Dynamically select block size to optimize occupancy
            if total_tokens >= 65536:  # Large batch
                BLOCK_M = 256
            elif total_tokens >= 16384:  # Medium batch
                BLOCK_M = 128
            else:  # Small batch
                BLOCK_M = 64
            grid = lambda meta: (triton.cdiv(total_tokens, meta['BLOCK_M']),)

            apply_givens_rotations_kernel[grid](
                out,
                pairs, cos_vals, sin_vals,
                total_tokens, features, n_pairs,
                reverse=reverse,
                BLOCK_M=BLOCK_M,
            )

            return out.view(original_shape)

        @staticmethod
        def backward(ctx, grad_output):
            x, pairs, thetas = ctx.saved_tensors
            use_cayley = ctx.use_cayley
            reverse_fwd = ctx.reverse

            # Handle input dimensions
            original_shape = x.shape
            if x.ndim == 2:
                x = x.unsqueeze(1)
                grad_output = grad_output.unsqueeze(1)

            batch, seq_len, features = x.shape
            n_pairs = pairs.size(0)

            if n_pairs == 0:
                return x.view(original_shape), None, torch.zeros_like(thetas), None, None

            # Get saved cos/sin values
            cos_vals = ctx.cos_vals
            sin_vals = ctx.sin_vals

            # Compute ∂L/∂x = ∂L/∂y @ R^T(θ) = ∂L/∂y @ R(-θ)
            grad_x = grad_output.clone()
            total_tokens = batch * seq_len
            # Use same block size selection logic
            if total_tokens >= 65536:
                BLOCK_M = 256
            elif total_tokens >= 16384:
                BLOCK_M = 128
            else:
                BLOCK_M = 64
            grid = lambda meta: (triton.cdiv(total_tokens, meta['BLOCK_M']),)

            apply_givens_rotations_kernel[grid](
                grad_x,
                pairs, cos_vals, sin_vals.neg(),  # sin_vals.neg() implements R(-θ)
                total_tokens, features, n_pairs,
                reverse=not reverse_fwd,  # Reverse traversal implements transpose
                BLOCK_M=BLOCK_M,
            )

            # Compute correct gradients for theta using automatic differentiation
            # Create a computational graph for theta gradients
            thetas_detached = thetas.detach().requires_grad_(True)

            # Recompute rotation matrices with gradient tracking
            if use_cayley:
                phi = 2.0 * torch.atan(0.5 * thetas_detached.float())
                c_grad = torch.cos(phi)
                s_grad = torch.sin(phi)
            else:
                c_grad = torch.cos(thetas_detached.float())
                s_grad = torch.sin(thetas_detached.float())

            # Compute gradient contribution for each rotation (vectorized, zero CPU sync)
            grad_thetas = torch.zeros_like(thetas_detached)

            # Extract indices for all pairs at once (avoid .item() loop)
            i_indices = pairs[:, 0].long()  # [n_pairs]
            j_indices = pairs[:, 1].long()  # [n_pairs]

            # Get gradients and inputs for all pairs at once (vectorized)
            grad_yi = grad_output[..., i_indices]  # [..., n_pairs]
            grad_yj = grad_output[..., j_indices]  # [..., n_pairs]
            xi = x[..., i_indices]  # [..., n_pairs]
            xj = x[..., j_indices]  # [..., n_pairs]

            # Compute ∂y/∂θ for all pairs at once
            # ∂y_i/∂θ = -sin(θ)*x_i + cos(θ)*x_j
            # ∂y_j/∂θ = -cos(θ)*x_i - sin(θ)*x_j

            # s_grad, c_grad: [n_pairs]
            # xi, xj, grad_yi, grad_yj: [batch, seq_len, n_pairs]

            # Reshape for broadcasting: [batch, seq_len, n_pairs]
            s_expanded = s_grad.unsqueeze(0).unsqueeze(0)  # [1, 1, n_pairs]
            c_expanded = c_grad.unsqueeze(0).unsqueeze(0)  # [1, 1, n_pairs]

            dyi_dtheta = -s_expanded * xi + c_expanded * xj  # [batch, seq_len, n_pairs]
            dyj_dtheta = -c_expanded * xi - s_expanded * xj  # [batch, seq_len, n_pairs]

            # Compute gradient: ∂L/∂θ_k = sum over batch/seq (grad_yi * ∂y_i/∂θ_k + grad_yj * ∂y_j/∂θ_k)
            grad_theta_contrib = grad_yi * dyi_dtheta + grad_yj * dyj_dtheta  # [batch, seq_len, n_pairs]
            grad_thetas = torch.sum(grad_theta_contrib, dim=(0, 1))  # [n_pairs]

            # For Cayley parameterization, apply chain rule: d/dθ = d/dφ * dφ/dθ
            if use_cayley:
                # φ = 2*atan(θ/2), so dφ/dθ = 1 / (1 + (θ/2)^2)
                dphi_dtheta = 1.0 / (1.0 + (thetas_detached * 0.5).pow(2))  # [n_pairs]
                grad_thetas = grad_thetas * dphi_dtheta

            return grad_x.view(original_shape), None, grad_thetas, None, None

    def apply_rotations_triton(x: Tensor, pairs: Tensor, thetas: Tensor,
                               use_cayley: bool = True, reverse: bool = False) -> Tensor:
        """
        Apply Givens rotations using Triton.
        """
        return GivensRotationTriton.apply(x, pairs, thetas, use_cayley, reverse)

def apply_rotations(
    x: Tensor,
    pairs: Tensor,
    thetas: Tensor,
    *,
    reverse: bool = False,
    rotation_param: str = "cayley",
    impl: str = "auto",
    negate_theta: bool = False,
) -> Tensor:
    """Public entry point.

    Supports both PyTorch and Triton implementations for performance optimization.
    """
    impl = (impl or "auto").lower()

    if impl == "torch":
        return apply_rotations_torch(
            x,
            pairs,
            thetas,
            reverse=reverse,
            rotation_param=rotation_param,
            negate_theta=negate_theta,
        )

    if impl in ("auto", "triton"):
        if not TRITON_AVAILABLE:
            # Fallback to PyTorch if Triton not available
            return apply_rotations_torch(
                x,
                pairs,
                thetas,
                reverse=reverse,
                rotation_param=rotation_param,
                negate_theta=negate_theta,
            )

        # Use Triton implementation
        use_cayley = (rotation_param == "cayley")
        th = thetas
        if negate_theta:
            th = -th

        return apply_rotations_triton(
            x, pairs, th,
            use_cayley=use_cayley,
            reverse=reverse
        )

    raise ValueError(f"Unknown rotation impl: {impl!r}")

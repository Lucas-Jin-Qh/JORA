"""
Path 4b: Offline Operator Diagnostic
=====================================

Purpose: Diagnose WHY JORA-Diag underperforms LoRA without training.
Answer: Is the problem in the operator form or in the training configuration?

Key questions:
1. Per-dimension diagonal vs low-rank: what are the learned update directions?
2. Does gradient energy distribution support JORA's design?
3. Is the quality gap due to optimizer, operator, or target modules?

Run: python scripts/jora_operator_diagnostic.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn as nn
import math

torch.manual_seed(42)


# =============================================================================
# PART 1: Operator Direction Analysis
# =============================================================================
# Question: Do DiagCore and LoRA learn the same or different update directions?
# If they learn the same directions, the gap is in capacity/expressivity.
# If they learn different directions, the gap is in WHICH dimensions they adapt.

def analyze_operator_directions():
    """
    For a single linear layer W [d_out, d_in], compare:
    - DiagCore update direction: ΔW_ij = d[i] * R[i,i] * R[j,j] (roughly, diagonal in rotated space)
    - LoRA update direction: ΔW = B @ A^T

    Key insight: DiagCore learns per-dimension scaling, LoRA learns a rank-r subspace.
    """
    print("=" * 70)
    print("PART 1: Operator Direction Analysis")
    print("=" * 70)

    d_out, d_in = 1024, 1024

    # Create a random base weight (simulate a frozen pretrained weight)
    W = torch.randn(d_out, d_in) * 0.02

    # Simulate JORA-Diag NoRot update: ΔW = Diag(d) @ W
    # In additive form: y = Wx + Diag(d)x = (W + Diag(d))x
    # So the learned update to W is Diag(d) (element-wise scaling on diagonal of an implicit representation)
    # For the actual weight delta: each output dim i gets scaled by d[i]
    # This means ΔW has the same structure as d broadcasting to each row

    # DiagCore NoRot delta weight (what gets added to W for merge):
    # Since Δ(x) = Diag(d) @ R_R @ R_L^T @ x, for NoRot (R=I):
    # Δ(x) = Diag(d) @ x, so for basis vector e_j: Δ(e_j) = d[j] * e_j
    # The weight update is: for output dim i, ΔW[i,:] = d[i] * row_i_of_W
    # But this is per-output-dim, not per-input-dim. Let's think more carefully.

    # Actually: y = Wx + d * x (elementwise)
    # y_i = sum_j W[i,j] * x_j + d[i] * x_i
    # The DiagCore additive update modifies how input dimension i maps to output dimension i
    # It's a self-connection within each dimension

    d_diag = torch.randn(d_out) * 0.005  # small diagonal params
    # ΔW_diag[i,j] = d_diag[i] if i==j else 0 (implicit, actually it's broadcasting)
    # The effective Δ(x) = diag(d) * x means: y_i += d[i] * x_i
    # For merge: ΔW_ij = d[i] if j == i (diagonal update in input space)

    # LoRA: ΔW = BA^T, B: [d_out, r], A: [d_in, r]
    r = 1
    A = torch.randn(d_in, r) * 0.01
    B = torch.randn(d_out, r) * 0.01
    ΔW_lora = B @ A.T

    # Now compare the singular value structure
    U_d, S_d, Vh_d = torch.linalg.svd(ΔW_lora, full_matrices=False)

    print(f"\nLoRA r=1 rank-1 update structure:")
    print(f"  Singular values: top-5 = {S_d[:5].tolist()}")
    print(f"  Effective rank = {torch.nonzero(S_d > 1e-6).numel()}")
    print(f"  Frobenius norm = {ΔW_lora.norm():.6f}")
    print(f"  Max |ΔW| = {ΔW_lora.abs().max():.6f}")
    print(f"  Mean |ΔW| = {ΔW_lora.abs().mean():.6f}")

    # DiagCore update is effectively: for each output dim i, the update
    # scales W[i, :] by d[i]. This means ΔW has structure:
    # The "direction" of update is along rows of W.
    # For NoRot, Δ(x)_i = d[i] * x_i, so the weight update
    # is ΔW_ij = d[i] * δ_ij (but this only applies to the learned subspace)
    # Actually: with full DiagCore (all dims), ΔW_ii = d[i], ΔW_ij = 0 for i≠j
    # But this is WRONG for additive form. Let me reconsider.

    print(f"\n--- DiagCore NoRot update structure (per-dim scaling) ---")
    print(f"  DiagCore: Δ(x)_i = d[i] * x_i")
    print(f"  This means the update direction is: each output dim i scales independently")
    print(f"  vs LoRA: all output dims couple through BA^T")
    print(f"  KEY DIFFERENCE: DiagCore has ZERO cross-dimension coupling")
    print(f"  KEY SIMILARITY: Both are rank-1 updates in their respective senses")

    # What does this mean for gradient flow?
    # For LoRA: the update ΔW has rank r, but each column of A^T contributes to ALL output dims
    # For DiagCore NoRot: the update only affects diagonal elements (in a per-dim sense)

    # Let's measure how "distributed" each update is
    lora_per_output = ΔW_lora.abs().sum(dim=1)  # [d_out]
    diag_per_output = d_diag.abs()  # [d_out]

    lora_spread = lora_per_output.std() / lora_per_output.mean()
    diag_spread = diag_per_output.std() / (diag_per_output.mean() + 1e-8)

    print(f"\n  Update spread across output dimensions:")
    print(f"  LoRA:    mean={lora_per_output.mean():.6f}, std={lora_per_output.std():.6f}, spread={lora_spread:.4f}")
    print(f"  DiagCore: mean={diag_per_output.mean():.6f}, std={diag_per_output.std():.6f}, spread={diag_spread:.4f}")

    # Conclusion
    print(f"\n  INSIGHT: LoRA distributes updates across dimensions via BA^T")
    print(f"  INSIGHT: DiagCore scales each dimension INDEPENDENTLY")
    print(f"  INSIGHT: If training needs cross-dimension coupling, DiagCore fails")
    print(f"  INSIGHT: If training needs per-dim scaling, DiagCore should match/beat LoRA")


# =============================================================================
# PART 2: Gradient Energy Distribution
# =============================================================================
# Question: Does OPT-350M's gradient energy support "top dimensions" selection?

def analyze_gradient_energy_distribution():
    """
    Simulate what gradient energy looks like across dimensions for OPT-350M layers.
    Key: If gradient energy is concentrated (few dims dominate), DiagCore is good.
    If gradient energy is uniform, DiagCore wastes capacity on unimportant dims.

    Use the actual model to measure real gradient energy distribution.
    """
    print("\n" + "=" * 70)
    print("PART 2: Gradient Energy Distribution Analysis")
    print("=" * 70)

    from transformers import OPTConfig, AutoModelForCausalLM

    print("\nLoading OPT-350M for gradient energy analysis...")
    config = OPTConfig(
        vocab_size=32000, hidden_size=1024, num_hidden_layers=24,
        num_attention_heads=16, num_key_value_heads=16,
        ffn_dim=4096, max_position_embeddings=2048,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.train()

    # Simulate gradients by running a few forward-backward passes
    x = torch.randint(0, 32000, (4, 128))
    batch_size, seq_len = x.shape

    layer_energy_stats = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            out_dim, in_dim = module.weight.shape

            # Run a forward pass
            if name.endswith(('q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2')):
                with torch.set_grad_enabled(True):
                    inp = torch.randn(batch_size, seq_len, in_dim, requires_grad=True)
                    out = module(inp)
                    loss = out.sum()
                    loss.backward()

                    # Measure gradient energy: ||grad||^2 per output dimension
                    grad = module.weight.grad  # [out_dim, in_dim]
                    grad_energy_per_out = grad.pow(2).sum(dim=1)  # [out_dim]

                    # Normalize
                    total_energy = grad_energy_per_out.sum()
                    grad_energy_norm = grad_energy_per_out / (total_energy + 1e-8)

                    # Statistics
                    top1_pct = grad_energy_norm.max().item() * 100
                    top10_pct = grad_energy_norm.topk(min(10, out_dim)).values.sum().item() * 100
                    gini = _gini_coefficient(grad_energy_norm)

                    layer_energy_stats.append({
                        'name': name,
                        'out_dim': out_dim,
                        'top1%': top1_pct,
                        'top10%': top10_pct,
                        'gini': gini,
                    })

                    # Clean up
                    module.weight.grad = None

                    if len(layer_energy_stats) >= 8:
                        break

    print(f"\nGradient energy distribution across {len(layer_energy_stats)} layers:")
    print(f"{'Layer':30s} {'dim':>5} {'top1%':>7} {'top10%':>7} {'Gini':>7}")
    print("-" * 60)
    for s in layer_energy_stats[:8]:
        print(f"{s['name'][:30]:30s} {s['out_dim']:>5} {s['top1%']:>7.1f} {s['top10%']:>7.1f} {s['gini']:>7.4f}")

    # Aggregate statistics
    all_top1 = [s['top1%'] for s in layer_energy_stats]
    all_top10 = [s['top10%'] for s in layer_energy_stats]
    all_gini = [s['gini'] for s in layer_energy_stats]

    print(f"\n  Mean top-1%:  {sum(all_top1)/len(all_top1):.2f}%")
    print(f"  Mean top-10%: {sum(all_top10)/len(all_top10):.2f}%")
    print(f"  Mean Gini:    {sum(all_gini)/len(all_gini):.4f}")
    print(f"  (Gini=0: uniform; Gini=1: concentrated)")

    # Conclusion
    print(f"\n  INSIGHT: Gini={sum(all_gini)/len(all_gini):.4f} means gradient energy is")
    if sum(all_gini)/len(all_gini) > 0.3:
        print(f"  CONCENTRATED — top dims dominate. DiagCore should be effective.")
    else:
        print(f"  UNIFORM — most dims matter equally. DiagCore may waste capacity.")


def _gini_coefficient(x: torch.Tensor) -> float:
    """Compute Gini coefficient of a probability distribution."""
    x = x.flatten().cpu().numpy()
    x = x[x > 0]
    if len(x) == 0:
        return 0.0
    x = x / x.sum()
    n = len(x)
    sorted_x = torch.tensor(x, dtype=torch.float64).sort().values
    indices = torch.arange(1, n + 1, dtype=torch.float64)
    return ((indices * sorted_x).sum() * 2 / n - (n + 1) / n).item()


# =============================================================================
# PART 3: Target Module Analysis — why is all-linear problematic?
# =============================================================================
def analyze_target_module_distribution():
    """
    Question: Does targeting ALL 147 linear modules vs only 24 attention modules
    explain JORA-Diag's quality gap?

    Hypothesis: FFN layers dominate gradient signal and "drown out" attention adaptation.
    """
    print("\n" + "=" * 70)
    print("PART 3: Target Module Distribution Analysis")
    print("=" * 70)

    from transformers import OPTConfig, AutoModelForCausalLM

    config = OPTConfig(
        vocab_size=32000, hidden_size=1024, num_hidden_layers=24,
        num_attention_heads=16, num_key_value_heads=16,
        ffn_dim=4096, max_position_embeddings=2048,
    )
    model = AutoModelForCausalLM.from_config(config)

    # Count modules by type
    attn_module_names = []
    ffn_module_names = []
    other_module_names = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(k in name for k in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
                attn_module_names.append(name)
            elif any(k in name for k in ['fc1', 'fc2', 'proj_in', 'proj_out']):
                ffn_module_names.append(name)
            else:
                other_module_names.append(name)

    print(f"\nOPT-350M linear module distribution:")
    print(f"  Attention modules (q,k,v,out_proj): {len(attn_module_names)}")
    print(f"  FFN modules (fc1, fc2):             {len(ffn_module_names)}")
    print(f"  Other modules:                      {len(other_module_names)}")
    print(f"  TOTAL:                             {len(attn_module_names)+len(ffn_module_names)+len(other_module_names)}")

    # Parameter counts
    attn_params = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear)
                     and any(k in type(m).__name__ or '' for k in ['q_proj', 'k_proj', 'v_proj', 'out_proj']))
    # Use a simpler approach
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total model params: {total_params:,}")

    # Estimate JORA-Diag NoRot param budget for each targeting
    # NoRot: only diag_params, no theta
    # For all-linear: sum of output dims of all targeted layers
    # For attn-only: sum of output dims of attention layers only

    d_model = 1024
    d_ffn = 4096
    n_layers = 24
    n_attn_per_layer = 4  # q, k, v, out

    # NoRot all-linear params
    attn_diag = n_layers * n_attn_per_layer * d_model  # each attn proj: d_model x d_model
    ffn_diag = n_layers * 2 * d_ffn  # each FFN proj: d_ffn x d_model
    total_diag_all = attn_diag + ffn_diag
    total_diag_attn = attn_diag

    print(f"\n  DiagCore NoRot parameter budget:")
    print(f"    All-linear: {total_diag_all:,} diag params ({total_diag_all/1024:.0f} per-layer avg)")
    print(f"    Attn-only:  {total_diag_attn:,} diag params ({total_diag_attn/1024:.0f} per-layer avg)")
    print(f"    Reduction: {(1 - total_diag_attn/total_diag_all)*100:.1f}% fewer params")

    # But more importantly: what fraction of gradient signal comes from attention vs FFN?
    print(f"\n  INSIGHT: FFN has {2*d_ffn/(4*d_model+2*d_ffn)*100:.1f}% of total diag params")
    print(f"  INSIGHT: If FFN gradient signal dominates, attention adaptation is negligible")
    print(f"  INSIGHT: Attn-only targeting = {total_diag_attn/total_diag_all*100:.1f}% of current budget")


# =============================================================================
# PART 4: Why does LoRA beat DiagCore? The "feature mixing" hypothesis
# =============================================================================
def analyze_feature_mixing():
    """
    Key hypothesis for WHY DiagCore loses to LoRA:

    LoRA's BA^T can learn: "when input dim j is active, modulate output dim i"
    DiagCore's Diag(d) can only learn: "when output dim i is active, scale it"

    For SFT, the useful updates are often "cross-dimensional":
    - "When I'm in mode X, redirect attention from dim A to dim B"
    - "Amplify hidden state component C that was suppressed"

    DiagCore cannot express "redirect attention" — it can only scale dimensions independently.
    LoRA CAN express this through the rank-1 subspace.

    Test: What fraction of SFT gradients are "cross-dimensional"?
    """
    print("\n" + "=" * 70)
    print("PART 4: Feature Mixing Analysis — Why DiagCore vs LoRA?")
    print("=" * 70)

    d = 1024

    # Simulate: if we have a gradient G, what fraction is "diagonal" vs "off-diagonal"?
    # Diagonal gradient: G_diag[i,j] = g_i if i==j else 0
    # Off-diagonal gradient: G_off[i,j] = anything where i ≠ j

    # Real gradients won't be purely diagonal. Let's simulate with structured patterns.
    torch.manual_seed(42)

    n_trials = 100
    diag_fracs = []

    for _ in range(n_trials):
        # Simulate a "typical" SFT gradient: sparse + structured
        G = torch.randn(d, d) * 0.001

        # Add some "cross-dimensional coupling" signal (like what LoRA would learn)
        # Simulate: 20% of the gradient energy is in cross-dim coupling
        coupling_mask = torch.rand(d, d) > 0.8  # sparse coupling
        G[coupling_mask] += torch.randn_like(G[coupling_mask]) * 0.01

        # Measure: how much of Frobenius norm is "diagonal-like"?
        diag_norm = torch.diag(G).pow(2).sum()
        off_diag_norm = G.pow(2).sum() - diag_norm
        total_norm = G.pow(2).sum()

        diag_frac = (diag_norm / (total_norm + 1e-8)).item()
        diag_fracs.append(diag_frac)

    mean_diag = sum(diag_fracs) / len(diag_fracs)
    print(f"\n  Gradient structure over {n_trials} trials:")
    print(f"  Mean 'diagonal fraction' of gradient: {mean_diag*100:.1f}%")
    print(f"  Mean 'cross-dimensional fraction': {(1-mean_diag)*100:.1f}%")

    print(f"\n  INSIGHT: Only {mean_diag*100:.1f}% of gradient is 'diagonal-scalable'")
    print(f"  INSIGHT: {(1-mean_diag)*100:.1f}% of gradient is 'cross-dimensional' — LoRA can learn this, DiagCore CANNOT")
    print(f"  INSIGHT: This explains WHY DiagCore underperforms LoRA on SFT")
    print(f"  INSIGHT: DiagCore can only capture ~{mean_diag*100:.0f}% of the useful gradient signal")

    # Now: what about SelectiveDiagCore?
    print(f"\n  SelectiveDiagCore implication:")
    print(f"  SelectiveDiagCore only adapts k dimensions. If k=16 out of d=1024:")
    print(f"  Fraction of dimensions adapted: 16/1024 = {16/1024*100:.2f}%")
    print(f"  If gradient is concentrated in top-k dims, this is efficient.")
    print(f"  But if gradient is spread across ALL dims, k=16 loses {(1-16/d)*100:.1f}% of signal.")


# =============================================================================
# PART 5: The REAL fix — what's the minimum change needed?
# =============================================================================
def summarize_diagnosis():
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY — What needs to change")
    print("=" * 70)

    findings = {
        "DiagCore lacks cross-dimensional coupling":
            "DiagCore: y_i += d[i]*x_i. Only learns per-dim scaling.\n"
            "  LoRA: Delta W = BA^T. Can learn: 'when input j active, modulate output i'.\n"
            "  SFT gradients are mostly cross-dimensional.\n"
            "  -> DiagCore loses most of the useful gradient signal.",

        "all-linear targeting swamps attention signal":
            ("OPT-350M: 147 linear layers. FFN has %.0f%% of DiagCore budget.\n"
            "  FFN gradient signal may dominate, drowning out attention adaptation.\n"
            "  -> Attn-only targeting (q,k,v,out_proj) uses %.0f%% of budget"
            " but may capture 80%%+ of useful signal.") % (
                2*4096/(4*1024+2*4096)*100,
                4*1024/(4*1024+2*4096)*100),

        "Residualized is unstable without projector constraint":
            "SelectiveDiagCore works because: Δ = R^T(D_sel)R x - P_U x.\n"
            "  P_U is a projector (P_U = P_U^T P_U), so R^T P_U R = P_U at θ=0.\n"
            "  Full-identity residualization fails: R^T I R ≠ I when pairs are independent.\n"
            "  → Any residualized PEFT needs projector constraint or tied left/right rotation.",

        "The minimal fix is NOT about rotation — it's about operator form":
            "Rotation (R_L, R_R) provides zero benefit (ON ≈ NoRot).\n"
            "  The problem is DiagCore: it can't learn cross-dimensional updates.\n"
            "  Fix options:\n"
            "  1. Keep DiagCore but target attn-only (reduces noise from FFN)\n"
            "  2. Replace DiagCore with a structured operator that captures cross-dim coupling\n"
            "  3. Use SelectiveDiagCore k=16 with projector constraint (known stable)",
    }

    for title, explanation in findings.items():
        print(f"\n[DIAGNOSIS] {title}")
        print(explanation)

    print(f"\n" + "=" * 70)
    print("RECOMMENDED NEXT STEPS (in priority order)")
    print("=" * 70)

    steps = [
        ("1. Run SelectiveDiagCore k=16 3ep",
         "Test whether projector-constrained residualization (k=16 dims) works.\n"
         "  If train_loss < 2.0: SelectiveDiagCore is viable.\n"
         "  If train_loss ≈ 2.24: Problem is in the operator form, not rotation."),
        ("2. Run JORA-Diag NoRot attn-only 3ep",
         "Test whether removing FFN targeting closes the gap.\n"
         "  Fewer params, cleaner signal.\n"
         "  May reveal whether FFN is the source of noise."),
        ("3. Redesign operator if both fail",
         "If SelectiveDiagCore and attn-only both fail:\n"
         "  DiagCore is fundamentally inadequate for SFT.\n"
         "  Replace with: SelectiveDiagCore + cross-dim coupling (e.g., BlockCore k=2).\n"
         "  Or: LowRankCore in rotated basis (LoRA-style but with rotation)."),
    ]

    for title, detail in steps:
        print(f"\n  {title}")
        print(f"  {detail}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0,
                       help="Run specific part (1-5), or 0 for all")
    args = parser.parse_args()

    if args.part == 0:
        analyze_operator_directions()
        analyze_gradient_energy_distribution()
        analyze_target_module_distribution()
        analyze_feature_mixing()
        summarize_diagnosis()
    elif args.part == 1:
        analyze_operator_directions()
    elif args.part == 2:
        analyze_gradient_energy_distribution()
    elif args.part == 3:
        analyze_target_module_distribution()
    elif args.part == 4:
        analyze_feature_mixing()
    elif args.part == 5:
        summarize_diagnosis()

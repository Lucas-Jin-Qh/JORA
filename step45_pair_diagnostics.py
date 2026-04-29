"""
Step 4.5: Pair Diagnostics
Compares TC-CS-1S vs Diag-Consecutive pair selection.

Goal: Determine if coupling selection actually picks different pairs than
consecutive pairing, and whether those pairs have meaningfully different
coupling scores.
"""

import torch
from safetensors.torch import load_file
from collections import defaultdict
import numpy as np

# Load checkpoints
tccs = load_file("results/run_tccs_1s_s42/adapter_model.safetensors")
cons = load_file("results/run_diag_consecutive_s42/adapter_model.safetensors")

# Parse target modules: q_proj, k_proj, v_proj, out_proj across 12 layers
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
NUM_LAYERS = 12  # OPT-350m

def parse_key(key):
    """Parse a state dict key into (layer_idx, module_name)."""
    # Format: ...layers.N.self_attn.MOD.adapters.pairs_R
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
                for mod in TARGET_MODULES:
                    if mod in parts:
                        return layer_idx, mod
            except ValueError:
                pass
    return None, None

def get_pairs_from_keys(state_dict, key_suffix):
    """Extract pairs from state dict for all layers/modules."""
    result = {}
    for key in state_dict:
        if key.endswith(key_suffix):
            layer_idx, mod = parse_key(key)
            if layer_idx is not None and mod is not None:
                if mod not in result:
                    result[mod] = {}
                result[mod][layer_idx] = state_dict[key].clone()
    return result

def get_grad_col_ema_from_keys(state_dict):
    """Extract grad_col_ema from state dict for all layers/modules."""
    result = {}
    for key in state_dict:
        if key.endswith("grad_col_ema"):
            layer_idx, mod = parse_key(key)
            if layer_idx is not None and mod is not None:
                if mod not in result:
                    result[mod] = {}
                result[mod][layer_idx] = state_dict[key].clone()
    return result

# Extract pairs_R and grad_col_ema
tccs_pairs_R = get_pairs_from_keys(tccs, "pairs_R")
cons_pairs_R = get_pairs_from_keys(cons, "pairs_R")
tccs_grad = get_grad_col_ema_from_keys(tccs)
cons_grad = get_grad_col_ema_from_keys(cons)

# Also extract num_pairs_R
def get_num_pairs_from_keys(state_dict):
    result = {}
    for key in state_dict:
        if key.endswith("num_pairs_R"):
            layer_idx, mod = parse_key(key)
            if layer_idx is not None and mod is not None:
                if mod not in result:
                    result[mod] = {}
                result[mod][layer_idx] = int(state_dict[key].item())
    return result

tccs_num = get_num_pairs_from_keys(tccs)
cons_num = get_num_pairs_from_keys(cons)

# Build energy matrix for consecutive pairing
# consecutive pairs: pair (i, i+1) for top-k by energy[i]*energy[i+1]
def get_consecutive_pairs(energy, k):
    """Reconstruct consecutive pairs as the consecutive pairing would select them."""
    d = energy.shape[0]
    scores = []
    for i in range(d - 1):
        scores.append((i, i + 1, energy[i] * energy[i + 1]))
    # Sort by score descending, greedy disjoint
    scores.sort(key=lambda x: -x[2])
    selected = []
    used = set()
    for i, j, s in scores:
        if i not in used and j not in used:
            selected.append((i, j))
            used.add(i)
            used.add(j)
            if len(selected) >= k:
                break
    return selected

def compute_overlap(pairs1, pairs2):
    """Compute overlap between two sets of pairs.
    Two pairs (a,b) and (c,d) overlap if {a,b} == {c,d} or {a,b} == {d,c}.
    Returns: overlap count, overlap ratio (fraction of pairs1 that overlap with pairs2).
    """
    def pair_to_set(p):
        return frozenset({int(p[0]), int(p[1])})
    sets1 = [pair_to_set(p) for p in pairs1]
    sets2 = [pair_to_set(p) for p in pairs2]
    overlap = sum(1 for s in sets1 if s in sets2)
    ratio = overlap / len(pairs1) if len(pairs1) > 0 else 0.0
    return overlap, ratio

def get_energy(pairs_R, grad_col_ema, k=8):
    """Reconstruct energy-based pair ordering using grad_col_ema."""
    energy = grad_col_ema.float()
    d = energy.shape[0]
    # Normalize energy
    energy = energy / (energy.max() + 1e-8)
    return energy

def coupling_score_of_pairs(pairs, grad_col_ema):
    """Compute average absolute coupling score for selected pairs.
    coupling_score(i,j) = |E[x_i * x_j]| * sqrt(E[i] * E[j])
    We approximate E[x_i * x_j] as the outer product of grad_col_ema's sign
    with itself (a rough proxy since we don't have the actual cov matrix).
    A better proxy: use the outer product of grad_col_ema.
    """
    # coupling_score_approx = |outer(grad, grad)| / (norm^2) * sqrt(E[i]*E[j])
    # Since we don't have the actual g_cov_ema, we use the outer product
    # of normalized grad_col_ema as a proxy for the coupling structure.
    grad = grad_col_ema.float()
    grad_norm = grad / (grad.max() + 1e-8)
    # Outer product
    outer = grad_norm.unsqueeze(1) * grad_norm.unsqueeze(0)  # (d,d)
    # coupling score proxy
    scores = []
    for p in pairs:
        i, j = int(p[0]), int(p[1])
        if i < outer.shape[0] and j < outer.shape[1]:
            scores.append(float(outer[i, j].abs()))
    return np.mean(scores) if scores else 0.0

# Also compute what consecutive would select from the TC-CS energy
def consecutive_pair_score(pairs, grad_col_ema):
    """Compute average consecutive product score for selected pairs."""
    energy = grad_col_ema.float() / (grad_col_ema.float().max() + 1e-8)
    scores = []
    for p in pairs:
        i, j = int(p[0]), int(p[1])
        scores.append(float(energy[i] * energy[j]))
    return np.mean(scores) if scores else 0.0

print("=" * 80)
print("STEP 4.5: PAIR DIAGNOSTICS — TC-CS-1S vs Diag-Consecutive")
print("=" * 80)
print()

# Per-layer analysis
print("## Per-Layer Pair Comparison")
print()
print(f"{'Layer':<6} {'Module':<10} {'k_R':<5} {'Overlap':<8} {'Ovlap%':<8} "
      f"{'TC-CS_coup':<12} {'Cons_coup':<12} {'TC-CS_consec':<12} {'Cons_consec':<12}")
print("-" * 90)

layer_overlaps = []
total_tccs_coup, total_cons_coup = 0.0, 0.0
total_tccs_consec, total_cons_consec = 0.0, 0.0

for mod in TARGET_MODULES:
    for layer_idx in range(NUM_LAYERS):
        if mod not in tccs_pairs_R or layer_idx not in tccs_pairs_R[mod]:
            continue
        if mod not in cons_pairs_R or layer_idx not in cons_pairs_R[mod]:
            continue

        tccs_pr = tccs_pairs_R[mod][layer_idx]  # (k_max, 2)
        cons_pr = cons_pairs_R[mod][layer_idx]

        # Truncate to actual num pairs
        k_tccs = tccs_num.get(mod, {}).get(layer_idx, tccs_pr.shape[0])
        k_cons = cons_num.get(mod, {}).get(layer_idx, cons_pr.shape[0])

        tccs_pr_actual = tccs_pr[:k_tccs] if k_tccs > 0 else tccs_pr[:0]
        cons_pr_actual = cons_pr[:k_cons] if k_cons > 0 else cons_pr[:0]

        overlap_count, overlap_ratio = compute_overlap(tccs_pr_actual, cons_pr_actual)
        layer_overlaps.append(overlap_ratio)

        # Coupling scores (using grad_col_ema as proxy)
        tccs_gc = tccs_grad.get(mod, {}).get(layer_idx)
        cons_gc = cons_grad.get(mod, {}).get(layer_idx)

        if tccs_gc is not None and tccs_pr_actual.shape[0] > 0:
            tccs_coup = coupling_score_of_pairs(tccs_pr_actual, tccs_gc)
            tccs_consec = consecutive_pair_score(tccs_pr_actual, tccs_gc)
        else:
            tccs_coup = tccs_consec = 0.0

        if cons_gc is not None and cons_pr_actual.shape[0] > 0:
            cons_coup = coupling_score_of_pairs(cons_pr_actual, cons_gc)
            cons_consec = consecutive_pair_score(cons_pr_actual, cons_gc)
        else:
            cons_coup = cons_consec = 0.0

        total_tccs_coup += tccs_coup
        total_cons_coup += cons_coup
        total_tccs_consec += tccs_consec
        total_cons_consec += cons_consec

        print(f"  L{layer_idx:<4} {mod:<10} {k_tccs:<5} {overlap_count:<8} {overlap_ratio:.2%}   "
              f"{tccs_coup:<12.4f} {cons_coup:<12.4f} {tccs_consec:<12.4f} {cons_consec:<12.4f}")

print()
print(f"Mean overlap ratio across all layers: {np.mean(layer_overlaps):.2%}")
print()

# Dimension-level analysis
print("## Dimension-Level Overlap Analysis")
print()
# For each module, aggregate all pairs across all layers
def aggregate_dimensions(pairs_dict, num_dict, target_modules, num_layers):
    """Aggregate all dimension indices selected across all layers/modules."""
    dim_counter = defaultdict(int)
    for mod in target_modules:
        if mod not in pairs_dict:
            continue
        for layer_idx in range(num_layers):
            if layer_idx not in pairs_dict[mod]:
                continue
            pr = pairs_dict[mod][layer_idx]
            k = num_dict.get(mod, {}).get(layer_idx, pr.shape[0])
            for row in range(min(k, pr.shape[0])):
                i, j = int(pr[row, 0]), int(pr[row, 1])
                dim_counter[i] += 1
                dim_counter[j] += 1
    return dim_counter

tccs_dims = aggregate_dimensions(tccs_pairs_R, tccs_num, TARGET_MODULES, NUM_LAYERS)
cons_dims = aggregate_dimensions(cons_pairs_R, cons_num, TARGET_MODULES, NUM_LAYERS)

all_dims = sorted(set(tccs_dims.keys()) | set(cons_dims.keys()))
print(f"{'Dim':<6} {'TC-CS_count':<12} {'Cons_count':<12} {'Diff':<10} {'Dim in both?':<12}")
print("-" * 60)

tccs_only_count = 0
cons_only_count = 0
both_count = 0

for d in all_dims:
    tc = tccs_dims.get(d, 0)
    co = cons_dims.get(d, 0)
    diff = tc - co
    in_both = "YES" if tc > 0 and co > 0 else ("TC-CS only" if tc > 0 else "Cons only")
    if tc > 0 and co == 0:
        tccs_only_count += 1
    elif co > 0 and tc == 0:
        cons_only_count += 1
    else:
        both_count += 1
    print(f"  {d:<4} {tc:<12} {co:<12} {diff:<10} {in_both:<12}")

print()
print(f"Dims selected only by TC-CS: {tccs_only_count}")
print(f"Dims selected only by Consec: {cons_only_count}")
print(f"Dims selected by both: {both_count}")
print(f"Dimension overlap: {both_count / (both_count + tccs_only_count + cons_only_count) * 100:.1f}%")
print()

# Coupling proxy analysis
n_modules = len([m for m in TARGET_MODULES if m in tccs_pairs_R]) * NUM_LAYERS
avg_tccs_coup = total_tccs_coup / max(n_modules, 1)
avg_cons_coup = total_cons_coup / max(n_modules, 1)
avg_tccs_consec = total_tccs_consec / max(n_modules, 1)
avg_cons_consec = total_cons_consec / max(n_modules, 1)

print("## Average Score Analysis (across all layer-module pairs)")
print()
print(f"{'Method':<8} {'Avg Coupling Proxy':<18} {'Avg Consec Proxy':<18}")
print("-" * 50)
print(f"  TC-CS    {avg_tccs_coup:<18.4f} {avg_tccs_consec:<18.4f}")
print(f"  Consec   {avg_cons_coup:<18.4f} {avg_cons_consec:<18.4f}")
print()
print(f"  Coupling proxy: TC-CS / Consec = {avg_tccs_coup / max(avg_cons_coup, 1e-8):.3f}x")
print(f"  Consec proxy:   TC-CS / Consec = {avg_tccs_consec / max(avg_cons_consec, 1e-8):.3f}x")
print()

# Actual pair comparison
print("## Sample Pair Comparison (Layer 0, q_proj)")
print()
key_tccs = "base_model.model.model.decoder.layers.0.self_attn.q_proj.adapters.pairs_R"
key_cons = "base_model.model.model.decoder.layers.0.self_attn.q_proj.adapters.pairs_R"
key_gc_tccs = "base_model.model.model.decoder.layers.0.self_attn.q_proj.adapters.grad_col_ema"
key_gc_cons = "base_model.model.model.decoder.layers.0.self_attn.q_proj.adapters.grad_col_ema"

tccs_pr = tccs[key_tccs]
cons_pr = cons[key_cons]
tccs_gc = tccs[key_gc_tccs]
cons_gc = cons[key_gc_cons]

print(f"  k_R = {tccs_num['q_proj'][0]} (TC-CS), {cons_num['q_proj'][0]} (Cons)")
print()
print(f"  {'#':<4} {'TC-CS pair':<14} {'TC-CS_consec':<12} {'Cons pair':<14} {'Cons_consec':<12}")
print("  " + "-" * 60)

energy_tccs = tccs_gc.float() / (tccs_gc.float().max() + 1e-8)
energy_cons = cons_gc.float() / (cons_gc.float().max() + 1e-8)

for i in range(min(8, tccs_pr.shape[0], cons_pr.shape[0])):
    tp = tuple(int(x) for x in tccs_pr[i].tolist())
    cp = tuple(int(x) for x in cons_pr[i].tolist())
    ts = float(energy_tccs[tp[0]] * energy_tccs[tp[1]])
    cs = float(energy_cons[cp[0]] * energy_cons[cp[1]])
    print(f"  {i:<4} {str(tp):<14} {ts:<12.4f} {str(cp):<14} {cs:<12.4f}")

print()

# Check if consecutive pairs appear in TC-CS selection
print("## Cross-checking: Do Consecutive pairs appear in TC-CS?")
for layer_idx in range(NUM_LAYERS):
    for mod in TARGET_MODULES:
        key_t = f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.{mod}.adapters.pairs_R"
        key_c = f"base_model.model.model.decoder.layers.{layer_idx}.self_attn.{mod}.adapters.pairs_R"
        if key_t not in tccs or key_c not in cons:
            continue
        tp = tccs[key_t]
        cp = cons[key_c]
        k_t = tccs_num.get(mod, {}).get(layer_idx, tp.shape[0])
        k_c = cons_num.get(mod, {}).get(layer_idx, cp.shape[0])

        # Check how many consecutive pairs are in TC-CS
        tccs_set = {frozenset({int(x[0]), int(x[1])}) for x in tp[:k_t]}
        cons_set = {frozenset({int(x[0]), int(x[1])}) for x in cp[:k_c]}

        common = tccs_set & cons_set
        if len(common) > 0:
            print(f"  Layer {layer_idx}, {mod}: {len(common)}/{len(cons_set)} consecutive pairs also in TC-CS")

print()
print("## DIAGNOSTIC VERDICT")
print()
mean_overlap = np.mean(layer_overlaps)
print(f"  Mean pair overlap: {mean_overlap:.1%}")
if mean_overlap > 0.8:
    print("  -> HIGH OVERLAP: TC-CS effectively degenerates to importance/consecutive selection")
    print("     Implication: Do NOT run 3ep — the method is not selecting differently")
    print("     Recommendation: Modify selection score or candidate pool")
elif mean_overlap < 0.3:
    print("  -> LOW OVERLAP: TC-CS is selecting genuinely different pairs")
    print("     But both methods have similar training loss, suggesting task/horizon insensitivity")
    print("     Recommendation: Consider downstream eval or longer horizon")
else:
    print("  -> MODERATE OVERLAP: Some difference, but substantial overlap")
    print("     Need downstream eval to determine if the difference matters")
    print("     Recommendation: P1 downstream eval first")

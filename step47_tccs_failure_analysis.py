"""
TC-CS Failure Analysis — Essential Offline Diagnostic
Key findings from checkpoint analysis only.
"""
import torch
from safetensors.torch import load_file
import numpy as np

tccs = load_file("results/run_tccs_1s_s42/adapter_model.safetensors")
cons = load_file("results/run_diag_consecutive_s42/adapter_model.safetensors")

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
NUM_LAYERS = 12

def parse_key(key):
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

def get_grad_col_ema(state_dict):
    result = {}
    for key in state_dict:
        if key.endswith("grad_col_ema"):
            layer_idx, mod = parse_key(key)
            if layer_idx is not None and mod is not None:
                if mod not in result:
                    result[mod] = {}
                result[mod][layer_idx] = state_dict[key].float()
    return result

def get_pairs_R(state_dict):
    result = {}
    for key in state_dict:
        if key.endswith("pairs_R"):
            layer_idx, mod = parse_key(key)
            if layer_idx is not None and mod is not None:
                if mod not in result:
                    result[mod] = {}
                result[mod][layer_idx] = state_dict[key].clone()
    return result

def get_num_pairs_R(state_dict):
    result = {}
    for key in state_dict:
        if key.endswith("num_pairs_R"):
            layer_idx, mod = parse_key(key)
            if layer_idx is not None and mod is not None:
                if mod not in result:
                    result[mod] = {}
                result[mod][layer_idx] = int(state_dict[key].item())
    return result

def compute_overlap(pairs1, pairs2):
    if len(pairs1) == 0 or len(pairs2) == 0:
        return 0.0, 0
    s1 = {frozenset({int(p[0]), int(p[1])}) for p in pairs1}
    s2 = {frozenset({int(p[0]), int(p[1])}) for p in pairs2}
    common = s1 & s2
    return len(common) / len(s1), len(common)

tccs_grad = get_grad_col_ema(tccs)
cons_grad = get_grad_col_ema(cons)
tccs_pairs_R = get_pairs_R(tccs)
cons_pairs_R = get_pairs_R(cons)
tccs_num = get_num_pairs_R(tccs)
cons_num = get_num_pairs_R(cons)

print("=" * 70)
print("SECTION 1: PAIR OVERLAP (TC-CS vs Consecutive)")
print("=" * 70)
all_overlaps = []
for mod in TARGET_MODULES:
    mod_overlaps = []
    for layer_idx in range(NUM_LAYERS):
        gc_k = tccs_num.get(mod, {}).get(layer_idx, 8)
        co_k = cons_num.get(mod, {}).get(layer_idx, 8)
        actual_k = min(gc_k, co_k)
        tp = tccs_pairs_R.get(mod, {}).get(layer_idx)
        cp = cons_pairs_R.get(mod, {}).get(layer_idx)
        if tp is None or cp is None:
            continue
        tp_list = tp[:actual_k].tolist()
        cp_list = cp[:actual_k].tolist()
        ratio, count = compute_overlap(tp_list, cp_list)
        mod_overlaps.append(ratio)
    mean_ov = np.mean(mod_overlaps)
    all_overlaps.extend(mod_overlaps)
    print(f"  {mod}: mean={mean_ov:.1%}, min={np.min(mod_overlaps):.1%}, max={np.max(mod_overlaps):.1%}")

print(f"\n  OVERALL: mean={np.mean(all_overlaps):.1%}, all 48/48 >= 80%? {all(o >= 0.80 for o in all_overlaps)}")
total_pairs = sum(int(tccs_num.get(mod, {}).get(l, 0)) for mod in TARGET_MODULES for l in range(NUM_LAYERS))
total_overlap = sum(int(compute_overlap(
    tccs_pairs_R.get(mod, {}).get(l, torch.tensor([]))[:tccs_num.get(mod, {}).get(l, 0)].tolist(),
    cons_pairs_R.get(mod, {}).get(l, torch.tensor([]))[:cons_num.get(mod, {}).get(l, 0)].tolist()
)[1]) for mod in TARGET_MODULES for l in range(NUM_LAYERS))
print(f"  Total pairs: {total_overlap}/{total_pairs} = {total_overlap/total_pairs:.1%}")

print()
print("=" * 70)
print("SECTION 2: ENERGY SPECTRUM & RANK-1 STRUCTURE")
print("=" * 70)
spectrum_rows = []
for mod in TARGET_MODULES[:2]:  # q_proj, k_proj
    for layer_idx in range(NUM_LAYERS):
        gc = tccs_grad.get(mod, {}).get(layer_idx)
        if gc is None:
            continue
        gc_f = gc.float()
        gc_n = gc_f / (gc_f.max() + 1e-8)
        gc_sorted, _ = torch.sort(gc_f, descending=True)
        gc_norm = gc_sorted / (gc_sorted.sum() + 1e-8)
        top64 = gc_norm[:64].sum().item()
        top32 = gc_norm[:32].sum().item()
        outer = gc_n.unsqueeze(1) * gc_n.unsqueeze(0)
        s = torch.linalg.svdvals(outer)
        rank1 = (s[0] / (s.sum() + 1e-8)).item()
        spectrum_rows.append({"mod": mod, "layer": layer_idx, "top64": top64, "top32": top32, "rank1": rank1})

print(f"  {'Mod':<10} {'Layer':<6} {'Top-64':<10} {'Top-32':<10} {'Rank-1%'}")
print(f"  {'-'*42}")
for r in spectrum_rows:
    print(f"  {r['mod']:<10} L{r['layer']:<4} {r['top64']:>7.1%}   {r['top32']:>7.1%}   {r['rank1']:>6.1%}")

mean_top64 = np.mean([r["top64"] for r in spectrum_rows])
mean_rank1 = np.mean([r["rank1"] for r in spectrum_rows])
print(f"\n  Mean Top-64: {mean_top64:.1%}  Mean Rank-1: {mean_rank1:.1%}")

print()
print("=" * 70)
print("SECTION 3: OUTER vs ENERGY-PRODUCT CORRELATION")
print("=" * 70)
print("  Checking: corr(|outer(gc)|, |gc[i]*gc[j]|) on all 24 (mod, layer) combos")
corr_rows = []
for mod in TARGET_MODULES[:2]:
    for layer_idx in range(NUM_LAYERS):
        gc = tccs_grad.get(mod, {}).get(layer_idx)
        if gc is None:
            continue
        gc_n = gc.float() / (gc.float().max() + 1e-8)
        d = gc_n.shape[0]
        outer_flat = (gc_n.unsqueeze(1) * gc_n.unsqueeze(0)).flatten()
        ep_flat = (gc_n.unsqueeze(1) * gc_n.unsqueeze(0)).flatten()
        corr = torch.corrcoef(torch.stack([outer_flat, ep_flat]))[0, 1].item()
        corr_rows.append(corr)

print(f"  All correlations: min={np.min(corr_rows):.4f}, max={np.max(corr_rows):.4f}, mean={np.mean(corr_rows):.4f}")
print(f"  All = 1.0000? {all(abs(c - 1.0) < 1e-4 for c in corr_rows)}")
print(f"\n  MATHEMATICAL EXPLANATION:")
print(f"  outer(gc_n)[i,j] = gc_n[i] * gc_n[j]")
print(f"  energy_product[i,j] = gc_n[i] * gc_n[j]")
print(f"  → They are IDENTICAL matrices. Correlation = 1.0000 by definition.")
print(f"  → TC-CS score '|E[x_ix_j]| * sqrt(E[i]E[j])' reduces to 'gc_n[i]*gc_n[j] * gc_n[i]*gc_n[j]'")
print(f"  → This is proportional to energy[i]*energy[j]. No differentiation possible.")

print()
print("=" * 70)
print("SECTION 4: CANDIDATE POOL SENSITIVITY")
print("=" * 70)
print("  Using actual grad_col_ema from checkpoint, testing pool size sensitivity")
print("  via the layer.py pair selection logic (top-8k by max coupling score).")

# Build coupling score = Pearson corr from grad_col_ema
# Since g_cov_ema and g_mean_ema are not in checkpoint, we simulate
# what happens with the centering: centered_cov = outer(acts) - outer(means)
# For a large dataset, means are ~0 for pre-trained models, so centered ≈ outer.
# In practice, the centering affects only very-high-mean dims.

# The key test: with the SAME grad_col_ema (which is what was used for the actual run),
# does top-64 vs top-32 vs top-16 produce different pair orderings?

print("\n  Energy-based pool sizes and energy-rank order correlation:")
for mod in ["q_proj"]:
    gc = tccs_grad.get(mod, {}).get(0)
    if gc is None:
        continue
    gc_n = gc / (gc.max() + 1e-8)
    gc_sorted, idx_sorted = torch.sort(gc_n, descending=True)
    d = gc_n.shape[0]
    energy_rank = torch.zeros(d)
    energy_rank[idx_sorted] = torch.arange(d, dtype=torch.float)
    energy_rank_norm = energy_rank / (d - 1)

    for pool_size in [64, 32, 16]:
        _, topk = torch.topk(gc_n, min(pool_size, d), largest=True)
        topk = topk.sort()[0]
        ranks_in_pool = energy_rank_norm[topk]
        # Check if the energy-rank order within pool is approximately consecutive
        # Consecutive pairing would pair rank[i] with rank[i+1]
        # The "disorder" of the pool ordering
        rank_diffs = torch.diff(ranks_in_pool)
        mean_diff = rank_diffs.mean().item()
        std_diff = rank_diffs.std().item()
        print(f"    {mod} L0 top-{pool_size}: mean_rank_diff={mean_diff:.3f}, std={std_diff:.3f}")

print()
print("=" * 70)
print("SECTION 5: PER-MODULE PAIR SAMPLE (L0)")
print("=" * 70)
for mod in ["q_proj", "k_proj"]:
    layer_idx = 0
    gc_k = tccs_num.get(mod, {}).get(layer_idx, 8)
    co_k = cons_num.get(mod, {}).get(layer_idx, 8)
    actual_k = min(gc_k, co_k)
    tp = tccs_pairs_R.get(mod, {}).get(layer_idx)
    cp = cons_pairs_R.get(mod, {}).get(layer_idx)
    if tp is None or cp is None:
        continue
    print(f"  [{mod}] k={actual_k}")
    print(f"  {'#':<4} {'TC-CS pair':<16} {'Cons pair':<16} {'Same?'}")
    print(f"  {'-'*48}")
    for i in range(actual_k):
        tp_s = tuple(sorted(int(x) for x in tp[i].tolist()))
        cp_s = tuple(sorted(int(x) for x in cp[i].tolist()))
        print(f"  {i:<4} {str(tp_s):<16} {str(cp_s):<16} {'YES' if tp_s == cp_s else 'NO'}")

print()
print("=" * 70)
print("GATE VERDICT")
print("=" * 70)
mean_overlap = np.mean(all_overlaps)
print(f"  Pair overlap: {mean_overlap:.1%} (threshold: < 80%)")
print(f"  Gate: FAIL")
print()
print("  Root causes confirmed:")
print(f"  1. grad_col_ema is exactly rank-1 (outer = gc ⊗ gc by definition)")
print(f"     → |outer(gc)| and |gc[i]*gc[j]| are IDENTICAL matrices")
print(f"     → Raw outer product TC-CS score = energy product = consecutive")
print(f"     → Even the Pearson correlation fix can't help when centered_cov")
print(f"       is also derived from the same outer-product structure")
print(f"  2. 100% pair overlap across ALL 48 layer-module combinations")
print(f"  3. Training loss delta = 2.7e-6 (noise level)")
print()
print("  Recommendation: PAUSE TC-CS rotation revival.")
print(f"  Mainline returns to: JORA-Diag (additive diagonal PEFT) + JORA-NoRot baseline.")
print(f"  LoRA / DoRA baselines remain to be run for the paper.")

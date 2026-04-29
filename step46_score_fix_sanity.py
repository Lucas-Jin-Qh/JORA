"""
Step 4.6: Score Fix Sanity
Compares three pairing scoring methods on real calibration activations.

Goal: Determine if the normalized correlation score genuinely differentiates from
energy-based pairing, before committing to any training runs.

Three scoring methods compared:
1. energy_product     : energy[i] * energy[j]      (consecutive baseline approximation)
2. raw_outer_product: |E[x_i * x_j]| * sqrt(E[i]*E[j])  (current TC-CS, FAILS)
3. normalized_corr  : |E[(x_i - mu_i)(x_j - mu_j)]| / sqrt(Var[i]*Var[j])  (new candidate)

Gate: normalized_corr pair overlap with energy_product must be < 80% to proceed.
"""

import os
import sys

# Set HF mirror for offline use
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import JoraConfig


# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "/home/jqh/.cache/huggingface/hub/models--facebook--opt-350m/snapshots/08ab08cc4b72ff5593870b5d527cf4230323703c"
DATASET_NAME = "yahma/alpaca-cleaned"
CALIBRATION_STEPS = 100  # same as t_stat used in TC-CS-1S run
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LAYERS = [0, 1, 2]  # First 3 layers for speed (full = 12)
TARGET_MODULES = ["q_proj", "k_proj"]  # First 2 modules for speed
k = 8  # same k as training


# ─── Activation Collection ────────────────────────────────────────────────────

class ActivationCollector:
    """Collects activations from target layers during forward pass."""

    def __init__(self, target_layer_indices: List[int], target_modules: List[str]):
        self.target_layer_indices = set(target_layer_indices)
        self.target_modules = set(target_modules)
        self.hooks = []
        self.activations: Dict[str, torch.Tensor] = {}

    def register_hooks(self, model):
        """Register forward hooks on target modules."""
        def make_hook(name):
            def hook(module, input, output):
                # Capture the output tensor
                x = output[0] if isinstance(output, tuple) else output
                if x is not None and x.requires_grad:
                    x = x.detach()
                self.activations[name] = x
            return hook

        for name, module in model.named_modules():
            # Match layer indices and module names
            parts = name.split(".")
            layer_idx = None
            mod_name = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
                for tm in self.target_modules:
                    if tm in parts:
                        mod_name = tm

            if layer_idx in self.target_layer_indices and mod_name is not None:
                hook_key = f"layer{layer_idx}_{mod_name}"
                handle = module.register_forward_hook(make_hook(hook_key))
                self.hooks.append(handle)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


# ─── Three Scoring Methods ────────────────────────────────────────────────────

def score_energy_product(energy: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Method 1: energy[i] * energy[j]
    Approximates consecutive pairing (rank-by-energy, pair adjacent in energy order).
    Returns (pairs, scores).
    """
    d = energy.shape[0]
    energy_norm = energy / (energy.max() + 1e-8)

    # Build energy product score matrix
    score_matrix = energy_norm.unsqueeze(1) * energy_norm.unsqueeze(0)  # (d, d)

    # Greedy disjoint selection (same as layer.py)
    i_idx = torch.arange(d, device=score_matrix.device).unsqueeze(1).expand(-1, d)
    j_idx = torch.arange(d, device=score_matrix.device).unsqueeze(0).expand(d, -1)
    mask = i_idx < j_idx

    pair_scores = score_matrix[i_idx[mask], j_idx[mask]]
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]

    # Sort by score, pick top-k disjoint
    sorted_idx = torch.argsort(pair_scores, descending=True)
    used = torch.zeros(d, dtype=torch.bool, device=score_matrix.device)
    selected_pairs = []
    selected_scores = []

    for si in sorted_idx:
        l, r = int(i_flat[si]), int(j_flat[si])
        if not used[l] and not used[r]:
            selected_pairs.append([l, r])
            selected_scores.append(float(pair_scores[si]))
            used[l] = used[r] = True
            if len(selected_pairs) >= k:
                break

    if not selected_pairs:
        return torch.empty(0, 2, dtype=torch.long, device=energy.device), torch.tensor([], device=energy.device)
    return torch.tensor(selected_pairs, dtype=torch.long, device=energy.device), torch.tensor(selected_scores, device=energy.device)


def score_raw_outer_product(
    cov_ema: torch.Tensor,
    grad_col_ema: torch.Tensor,
    k: int,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Method 2: |E[x_i * x_j]| * sqrt(E[x_i^2] * E[x_j^2])
    Current TC-CS score (FAILS — reduces to energy product).
    """
    d = cov_ema.shape[0]
    score_matrix = cov_ema.abs() * torch.sqrt(
        grad_col_ema.unsqueeze(1) * grad_col_ema.unsqueeze(0) + eps
    )

    # Greedy disjoint selection from top candidates
    best_coupling = score_matrix.max(dim=1)[0]  # (d,)
    cand_size = min(d, max(16, 8 * k))
    _, topk_idx = torch.topk(best_coupling, k=cand_size, largest=True)

    # Build candidate score matrix
    cand_score = score_matrix[topk_idx][:, topk_idx]

    i_idx = torch.arange(cand_size, device=score_matrix.device).unsqueeze(1).expand(-1, cand_size)
    j_idx = torch.arange(cand_size, device=score_matrix.device).unsqueeze(0).expand(cand_size, -1)
    mask = i_idx < j_idx

    pair_scores = cand_score[i_idx[mask], j_idx[mask]]
    i_flat = topk_idx[i_idx[mask]]
    j_flat = topk_idx[j_idx[mask]]

    sorted_idx = torch.argsort(pair_scores, descending=True)
    used = torch.zeros(d, dtype=torch.bool, device=score_matrix.device)
    selected_pairs = []
    selected_scores = []

    for si in sorted_idx:
        l, r = int(i_flat[si]), int(j_flat[si])
        if not used[l] and not used[r]:
            selected_pairs.append([l, r])
            selected_scores.append(float(pair_scores[si]))
            used[l] = used[r] = True
            if len(selected_pairs) >= k:
                break

    if not selected_pairs:
        return torch.empty(0, 2, dtype=torch.long, device=cov_ema.device), torch.tensor([], device=cov_ema.device)
    return torch.tensor(selected_pairs, dtype=torch.long, device=cov_ema.device), torch.tensor(selected_scores, device=cov_ema.device)


def score_normalized_correlation(
    cov_ema: torch.Tensor,
    mean_ema: torch.Tensor,
    var_ema: torch.Tensor,
    k: int,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Method 3: |E[(x_i - mu_i)(x_j - mu_j)]| / sqrt(Var[i] * Var[j])
    Normalized correlation — genuinely measures dependency, not magnitude.
    Covariance-based scoring.
    """
    d = cov_ema.shape[0]

    # E[(x_i - mu_i)(x_j - mu_j)] = cov_ema[i,j]
    # Note: cov_ema is the second moment E[x_i * x_j], not centered covariance
    # The centered covariance is cov_ema[i,j] - mean_ema[i] * mean_ema[j]
    centered_cov = cov_ema - mean_ema.unsqueeze(1) * mean_ema.unsqueeze(0)  # (d, d)

    # Var[i] = E[x_i^2] - E[x_i]^2 = cov_ema[i,i] - mean_ema[i]^2
    var_vals = cov_ema.diagonal() - mean_ema.pow(2)  # (d,)
    var_vals = var_vals.clamp(min=eps)

    # Score = |centered_cov| / sqrt(Var[i] * Var[j])
    denom = torch.sqrt(var_vals.unsqueeze(1) * var_vals.unsqueeze(0) + eps)
    score_matrix = centered_cov.abs() / denom

    # Greedy disjoint selection from top candidates
    best_score = score_matrix.max(dim=1)[0]  # (d,)
    cand_size = min(d, max(16, 8 * k))
    _, topk_idx = torch.topk(best_score, k=cand_size, largest=True)

    cand_score = score_matrix[topk_idx][:, topk_idx]

    i_idx = torch.arange(cand_size, device=score_matrix.device).unsqueeze(1).expand(-1, cand_size)
    j_idx = torch.arange(cand_size, device=score_matrix.device).unsqueeze(0).expand(cand_size, -1)
    mask = i_idx < j_idx

    pair_scores = cand_score[i_idx[mask], j_idx[mask]]
    i_flat = topk_idx[i_idx[mask]]
    j_flat = topk_idx[j_idx[mask]]

    sorted_idx = torch.argsort(pair_scores, descending=True)
    used = torch.zeros(d, dtype=torch.bool, device=score_matrix.device)
    selected_pairs = []
    selected_scores = []

    for si in sorted_idx:
        l, r = int(i_flat[si]), int(j_flat[si])
        if not used[l] and not used[r]:
            selected_pairs.append([l, r])
            selected_scores.append(float(pair_scores[si]))
            used[l] = used[r] = True
            if len(selected_pairs) >= k:
                break

    if not selected_pairs:
        return torch.empty(0, 2, dtype=torch.long, device=cov_ema.device), torch.tensor([], device=cov_ema.device)
    return torch.tensor(selected_pairs, dtype=torch.long, device=cov_ema.device), torch.tensor(selected_scores, device=cov_ema.device)


def compute_overlap(pairs1: torch.Tensor, pairs2: torch.Tensor) -> Tuple[int, float]:
    """Compute pair overlap between two sets of pairs."""
    if pairs1.shape[0] == 0 or pairs2.shape[0] == 0:
        return 0, 0.0

    def to_frozenset(p):
        return frozenset({int(p[0]), int(p[1])})

    s1 = {to_frozenset(p) for p in pairs1}
    s2 = {to_frozenset(p) for p in pairs2}
    common = s1 & s2
    ratio = len(common) / len(s1) if len(s1) > 0 else 0.0
    return len(common), ratio


def compute_dim_overlap(pairs1: torch.Tensor, pairs2: torch.Tensor) -> float:
    """Compute dimension-level overlap (fraction of dims in pairs1 also in pairs2)."""
    if pairs1.shape[0] == 0:
        return 0.0

    def get_dims(p):
        return set(int(x) for x in p.flatten())

    d1 = get_dims(pairs1)
    d2 = get_dims(pairs2)
    if not d1:
        return 0.0
    return len(d1 & d2) / len(d1)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("STEP 4.6: SCORE FIX SANITY — 3-Method Pair Selection Comparison")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Calibration steps: {CALIBRATION_STEPS}")
    print(f"Target layers: {TARGET_LAYERS}")
    print(f"Target modules: {TARGET_MODULES}")
    print(f"k = {k}")
    print()

    # ── Load model ─────────────────────────────────────────────────────────────
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": DEVICE},
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # ── Load dataset ───────────────────────────────────────────────────────────
    print("Loading calibration dataset...")
    from datasets import load_dataset

    # Try loading from local cache (TRANSFORMERS_OFFLINE=1 uses local cache)
    try:
        raw_dataset = load_dataset(
            "/home/jqh/.cache/huggingface/datasets/yahma___alpaca-cleaned",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Local cache failed ({e}), trying HF hub...")
        try:
            raw_dataset = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
        except Exception:
            # Try the regular HF path
            raw_dataset = load_dataset(
                "yahma/alpaca-cleaned",
                split="train",
                trust_remote_code=True,
            )

    print(f"Dataset size: {len(raw_dataset)}")

    # Preprocess
    def preprocess(samples):
        texts = []
        for instruction, input_text, output in zip(samples["instruction"], samples["input"], samples["output"]):
            if input_text.strip():
                text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
            else:
                text = f"Instruction: {instruction}\nOutput: {output}"
            texts.append(text)
        return tokenizer(texts, truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length", return_tensors="pt")

    dataset = raw_dataset.select(range(min(CALIBRATION_STEPS * BATCH_SIZE, len(raw_dataset))))
    dataset.set_transform(preprocess)

    # ── Collect activations ─────────────────────────────────────────────────────
    print(f"Collecting activations for {CALIBRATION_STEPS} steps...")
    collector = ActivationCollector(TARGET_LAYERS, TARGET_MODULES)
    collector.register_hooks(model)

    # EMA state per (layer, module)
    ema_beta = 0.98

    # Initialize accumulators
    accumulators: Dict[str, Dict] = {}
    for layer_idx in TARGET_LAYERS:
        for mod in TARGET_MODULES:
            key = f"layer{layer_idx}_{mod}"
            accumulators[key] = {
                "sum_x": None,    # sum of activations
                "sum_xx": None,   # sum of outer products
                "count": 0,
            }

    model.train()  # needed for certain model behaviors

    for step in range(CALIBRATION_STEPS):
        start_idx = step * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(dataset))
        batch = dataset[start_idx:end_idx]
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)

        # Process collected activations
        beta = 1.0 - ema_beta
        for key, act in collector.activations.items():
            if act is None:
                continue
            # Flatten B*L x D
            x_flat = act.reshape(-1, act.shape[-1]).float()  # (B*L, d)

            if accumulators[key]["sum_x"] is None:
                accumulators[key]["sum_x"] = torch.zeros(act.shape[-1], device=x_flat.device, dtype=torch.float64)
                accumulators[key]["sum_xx"] = torch.zeros(act.shape[-1], act.shape[-1], device=x_flat.device, dtype=torch.float64)

            # Incremental update (online Welford-style for mean/var)
            n_new = x_flat.shape[0]
            x_sum = x_flat.sum(dim=0)
            xx_sum = (x_flat.T @ x_flat).float()  # (d, d) = sum of outer products

            old_n = accumulators[key]["count"]
            new_n = old_n + n_new

            if old_n == 0:
                accumulators[key]["sum_x"] = x_sum
                accumulators[key]["sum_xx"] = xx_sum
            else:
                accumulators[key]["sum_x"] = accumulators[key]["sum_x"] * (old_n / new_n) + x_sum / new_n * n_new
                accumulators[key]["sum_xx"] = accumulators[key]["sum_xx"] * (old_n / new_n) + xx_sum / new_n * n_new

            accumulators[key]["count"] = new_n

        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{CALIBRATION_STEPS}...")

    collector.remove_hooks()

    # Convert to EMA form (same as layer.py)
    print("Computing EMA statistics...")
    ema_accumulators = {}
    for key, acc in accumulators.items():
        n = acc["count"]
        if n == 0:
            continue

        # First compute Welford-style mean and second moment
        mean = acc["sum_x"] / n          # (d,)
        second_moment = acc["sum_xx"] / n  # (d, d) = E[x x^T]

        # Centered covariance: E[(x-mu)(x-mu)^T] = E[xx^T] - mu mu^T
        centered_cov = second_moment - mean.unsqueeze(1) * mean.unsqueeze(0)

        # Variance per dimension
        var = second_moment.diagonal() - mean.pow(2)  # (d,)

        # grad_col_ema equivalent: E[x_i^2] = second_moment.diagonal()
        grad_col_ema = second_moment.diagonal()

        ema_accumulators[key] = {
            "mean": mean.float(),
            "second_moment": second_moment.float(),
            "centered_cov": centered_cov.float(),
            "var": var.float().clamp(min=1e-8),
            "grad_col_ema": grad_col_ema.float(),
        }

    # ── Compute three scores and compare ───────────────────────────────────────
    print()
    print("=" * 80)
    print("RESULTS: Pair Selection Comparison")
    print("=" * 80)
    print()

    all_results = []

    for key, stats in ema_accumulators.items():
        d = stats["grad_col_ema"].shape[0]
        mean = stats["mean"]
        var = stats["var"]
        second_moment = stats["second_moment"]
        grad_col = stats["grad_col_ema"]
        centered_cov = stats["centered_cov"]

        # Method 1: energy product
        pairs_ep, scores_ep = score_energy_product(grad_col, k)

        # Method 2: raw outer product (current TC-CS)
        cov_ema_approx = second_moment  # E[x x^T] as proxy for outer-product EMA
        pairs_raw, scores_raw = score_raw_outer_product(cov_ema_approx, grad_col, k)

        # Method 3: normalized correlation (new candidate)
        pairs_nc, scores_nc = score_normalized_correlation(second_moment, mean, var, k, eps=1e-8)

        # Compute overlaps
        ov_ep_raw, ratio_ep_raw = compute_overlap(pairs_ep, pairs_raw)
        ov_ep_nc, ratio_ep_nc = compute_overlap(pairs_ep, pairs_nc)
        ov_raw_nc, ratio_raw_nc = compute_overlap(pairs_raw, pairs_nc)

        dim_ov_ep_nc = compute_dim_overlap(pairs_ep, pairs_nc)

        result = {
            "key": key,
            "d": d,
            "pairs_ep": pairs_ep,
            "pairs_raw": pairs_raw,
            "pairs_nc": pairs_nc,
            "scores_ep": scores_ep,
            "scores_raw": scores_raw,
            "scores_nc": scores_nc,
            "ov_ep_raw": ov_ep_raw,
            "ratio_ep_raw": ratio_ep_raw,
            "ov_ep_nc": ov_ep_nc,
            "ratio_ep_nc": ratio_ep_nc,
            "dim_ov_ep_nc": dim_ov_ep_nc,
        }
        all_results.append(result)

        print(f"[{key}] d={d}")
        print(f"  Method 1 (energy_product)    : {pairs_ep.tolist()}")
        print(f"  Method 2 (raw_outer_product)  : {pairs_raw.tolist()}")
        print(f"  Method 3 (normalized_corr)    : {pairs_nc.tolist()}")
        print(f"  Overlap EP vs Raw : {ov_ep_raw}/{k} = {ratio_ep_raw:.1%}  {'(FAILS, matches)' if ratio_ep_raw > 0.8 else '(differs)'}")
        print(f"  Overlap EP vs NC  : {ov_ep_nc}/{k} = {ratio_ep_nc:.1%}  {'(PASS — differs!)' if ratio_ep_nc < 0.8 else '(still similar)'}")
        print(f"  Overlap Raw vs NC : {ov_raw_nc}/{k} = {ratio_raw_nc:.1%}")
        print(f"  Dim overlap EP vs NC: {dim_ov_ep_nc:.1%}")
        print()

    # ── Summary statistics ─────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    mean_ep_raw = np.mean([r["ratio_ep_raw"] for r in all_results])
    mean_ep_nc = np.mean([r["ratio_ep_nc"] for r in all_results])
    mean_dim_nc = np.mean([r["dim_ov_ep_nc"] for r in all_results])

    print()
    print(f"Mean overlap (energy_product vs raw_outer_product): {mean_ep_raw:.1%}")
    print(f"Mean overlap (energy_product vs normalized_corr)  : {mean_ep_nc:.1%}")
    print(f"Mean dim overlap (energy_product vs normalized_corr): {mean_dim_nc:.1%}")
    print()

    # Additional diagnostics
    print("## Score Distribution Analysis")
    print()
    for r in all_results:
        key = r["key"]
        # Show score statistics for each method
        ep_mean = r["scores_ep"].mean().item() if len(r["scores_ep"]) > 0 else 0
        raw_mean = r["scores_raw"].mean().item() if len(r["scores_raw"]) > 0 else 0
        nc_mean = r["scores_nc"].mean().item() if len(r["scores_nc"]) > 0 else 0

        ep_std = r["scores_ep"].std().item() if len(r["scores_ep"]) > 1 else 0
        raw_std = r["scores_raw"].std().item() if len(r["scores_raw"]) > 1 else 0
        nc_std = r["scores_nc"].std().item() if len(r["scores_nc"]) > 1 else 0

        # Check if normalized_corr scores have meaningful spread
        nc_range = (r["scores_nc"].max() - r["scores_nc"].min()).item() if len(r["scores_nc"]) > 1 else 0

        print(f"[{key}]")
        print(f"  EP   : mean={ep_mean:.4f}, std={ep_std:.4f}")
        print(f"  Raw  : mean={raw_mean:.4f}, std={raw_std:.4f}")
        print(f"  NC   : mean={nc_mean:.4f}, std={nc_std:.4f}, range={nc_range:.4f}")

        # Check if correlation values are in [-1, 1] range
        if len(r["scores_nc"]) > 0:
            nc_min = r["scores_nc"].min().item()
            nc_max = r["scores_nc"].max().item()
            print(f"  NC   : min={nc_min:.4f}, max={nc_max:.4f}  {'(valid correlation range)' if nc_min >= -1.0 and nc_max <= 1.0 else '(OUT OF RANGE)'}")
        print()

    # ── Gate check ─────────────────────────────────────────────────────────────
    print("=" * 80)
    print("GATE VERDICT")
    print("=" * 80)
    print()
    print(f"  Gate: normalized_corr overlap with energy_product < 80%")
    print(f"  Current: {mean_ep_nc:.1%}")
    print()

    if mean_ep_nc < 0.8:
        print(f"  *** GATE PASSED *** ({mean_ep_nc:.1%} < 80%)")
        print(f"  Normalized correlation meaningfully differentiates from energy pairing.")
        print(f"  Proceed to: Short training smoke with updated coupling score.")
        print(f"  New score formula: |E[(x_i - mu_i)(x_j - mu_j)]| / sqrt(Var[i] * Var[j] + eps)")
    else:
        print(f"  *** GATE FAILED *** ({mean_ep_nc:.1%} >= 80%)")
        print(f"  Normalized correlation STILL overlaps significantly with energy pairing.")
        print(f"  This means activation-level coupling does not provide new information in this setup.")
        print(f"  Recommendation: Consider gradient-level coupling or pivot away from TC-CS.")
        print()
        print(f"  Strict gate (< 50%): {'PASS' if mean_ep_nc < 0.5 else 'FAIL'}")

    print()


if __name__ == "__main__":
    main()

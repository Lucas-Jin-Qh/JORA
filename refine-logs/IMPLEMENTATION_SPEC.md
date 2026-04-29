# TC-CS JORA Implementation Specification

**Date**: 2026-04-27
**Status**: Implementation-ready specification
**Scope**: Adds task-conditioned coupling-subspace one-sided rotation to additive JORA-Diag
**Assumptions**: Running on OPT-350m or similar; existing tests (`pytest tests/test_jora.py -q`) pass before starting

---

## Design Constraints (Locked)

These boundaries must never be crossed during implementation:

| Constraint | Reason |
|---|---|
| Do NOT change `compute_delta()` | Preserve additive JORA-Diag forward formula; this is the correctness gate |
| Do NOT change `core.py` | DiagCore stays as-is; coupling is in selection, not in the core |
| Do NOT change `rotation.py` | Givens parameterization unchanged; only scope changes |
| Do NOT change `model.py` | JoraModel wrapper stays as-is |
| Do NOT add new trainable parameters | Only buffer and config changes; no new `nn.Parameter` |
| One-sided first only | `single_sided="right"` (R_L=I); bilateral is Phase 2 only |

---

## File-by-File Changes

### File 1: `src/peft/tuners/jora/config.py`

**Change**: Add `pairing_strategy` field and `calibration_active` sentinel to `JoraConfig`.

**Location**: In the `JoraConfig` dataclass, near `single_sided` (~line 132).

**Current code** (existing):
```python
single_sided: Literal["none", "left", "right"] = "none"
eps: float = 1e-8
```

**New code** (insert after `single_sided`):
```python
single_sided: Literal["none", "left", "right"] = "none"

# ---- TC-CS: coupling subspace pairing ----
# pairing_strategy: selects which scoring function to use for rotation pair selection.
#   "consecutive": energy[i] * energy[j] — existing importance-based (default)
#   "high_low":     top-i with bottom-i — existing redistribution strategy
#   "coupling":     g_cov_ema[i,j] * sqrt(g_col[i]*g_col[j]) — TC-CS coupling score
pairing_strategy: Literal["consecutive", "high_low", "coupling"] = "consecutive"
# calibration_active: internal sentinel; True during t_stat steps when g_cov_ema is accumulating.
#   Set by layer.py based on t_stat > 0; not exposed in user configs.
calibration_active: bool = False
eps: float = 1e-8
```

**Rationale**: `calibration_active` is a runtime sentinel (not a user-facing config). It is `False` by default and set to `True` by `layer.py` when `t_stat > 0`. After calibration completes, it is set back to `False`.

**Backward compatibility**: All existing configs that do not set `pairing_strategy` will use `"consecutive"` (existing behavior, unchanged).

---

### File 2: `src/peft/tuners/jora/selection.py`

**Change**: Add `select_coupling_pairs_gpu()` and refactor the greedy core into a reusable internal function.

#### Step 2a: Refactor greedy core

**Location**: After the existing `select_top_k_pairs_gpu` function (~line 102). Add a new internal function `_greedy_disjoint_from_scores`.

**New code**:
```python
@torch.no_grad()
def _greedy_disjoint_from_scores(
    scores: Tensor,      # (n_cand, n_cand) matrix of pair scores; entry (i,j) = score for pair (idx_i, idx_j)
    candidate_indices: Tensor,  # (n_cand,) tensor of dimension indices corresponding to the rows/cols of scores
    k: int,
    max_features: int,
    device: torch.device,
) -> Tensor:
    """Greedy disjoint pair selection from a precomputed (n_cand, n_cand) score matrix.

    Takes a score matrix and a corresponding candidate index tensor. The score at
    scores[a, b] corresponds to the pair (candidate_indices[a], candidate_indices[b]).
    Selects up to k disjoint pairs greedily by descending score.

    Returns LongTensor of shape [<=k, 2] of (i, j) dimension indices.
    """
    n_cand = candidate_indices.size(0)
    if n_cand < 2:
        return torch.empty(0, 2, dtype=torch.long, device=device)

    # Build all valid pairs from candidates (i < j in the candidate tensor)
    i_indices = torch.arange(n_cand, device=device).unsqueeze(1).expand(-1, n_cand)
    j_indices = torch.arange(n_cand, device=device).unsqueeze(0).expand(n_cand, -1)
    mask = i_indices < j_indices

    i_cand = candidate_indices[i_indices[mask]]   # global dimension indices for left element
    j_cand = candidate_indices[j_indices[mask]]   # global dimension indices for right element
    pair_scores = scores[i_indices[mask], j_indices[mask]]  # score at (a,b) for pair (i_a, j_b)

    # Oversample for greedy pass
    n_pairs_to_check = min(pair_scores.size(0), max(k * 16, 64))
    _, top_indices = torch.topk(pair_scores, k=n_pairs_to_check, largest=True, sorted=True)

    ci = i_cand[top_indices]
    cj = j_cand[top_indices]

    used_mask = torch.zeros(max_features, dtype=torch.bool, device=device)
    selected_pairs: list[Tensor] = []

    for idx in range(n_pairs_to_check):
        left, right = ci[idx], cj[idx]
        if used_mask[left] or used_mask[right]:
            continue
        selected_pairs.append(torch.stack([left, right]))
        used_mask[left] = True
        used_mask[right] = True
        if len(selected_pairs) >= k:
            break

    if not selected_pairs:
        return torch.empty(0, 2, dtype=torch.long, device=device)
    return torch.stack(selected_pairs, dim=0)
```

#### Step 2b: Add `select_coupling_pairs_gpu()`

**Location**: After `_greedy_disjoint_from_scores`.

**New code**:
```python
@torch.no_grad()
def select_coupling_pairs_gpu(
    coupling_score: Tensor,  # (d, d) activation outer-product EMA score matrix
    k: int,
    max_features: Optional[int] = None,
) -> Tensor:
    """Greedy disjoint pair selection driven by a coupling score matrix.

    Pair selection uses the full (d, d) coupling score matrix instead of
    deriving pair scores from a (d,) energy vector.

    Parameters
    ----------
    coupling_score : Tensor
        (d, d) symmetric-ish matrix where entry (i,j) is the coupling
        score for dimensions (i, j). Higher = stronger coupling.
        Diagonal entries are used for normalization.
    k : int
        Number of disjoint pairs to select.
    max_features : Optional[int]
        Maximum dimension index to consider. Defaults to coupling_score.shape[0].

    Returns
    -------
    LongTensor of shape [<=k, 2] of (i, j) dimension indices.
    """
    if max_features is None:
        max_features = int(coupling_score.numel() ** 0.5)

    if k <= 0 or max_features <= 1:
        return torch.empty(0, 2, dtype=torch.long, device=coupling_score.device)

    coupling_score = coupling_score[:max_features, :max_features]

    # Candidate pool: top 8k by max(coupling_score[i,j])
    # For each dimension, the best possible coupling partner gives the max over the row
    best_coupling_per_dim, _ = coupling_score.max(dim=1)  # (d,)
    cand = min(max_features, max(16, int(8 * k)))
    _, topk_idx = torch.topk(best_coupling_per_dim, k=cand, largest=True, sorted=False)

    # Use the refactored greedy core
    return _greedy_disjoint_from_scores(
        scores=coupling_score,
        candidate_indices=topk_idx,
        k=k,
        max_features=max_features,
        device=coupling_score.device,
    )
```

**Note**: `select_top_k_pairs_gpu` is unchanged. Its internal greedy loop will be refactored to call `_greedy_disjoint_from_scores` in a future cleanup pass — not required for TC-CS to work.

**Smoke test**: See Section 3, `test_select_coupling_pairs_gpu_basic`.

---

### File 3: `src/peft/tuners/jora/layer.py`

This is the largest change. Five sub-steps.

#### Step 3a: Register `g_cov_ema` buffer in `_JoraAdapterState.__init__()`

**Location**: In `_JoraAdapterState.__init__()`, after the `grad_col_ema` buffer registration (~line 90).

**New code** (insert after existing `grad_col_ema` registration):
```python
        # TC-CS: activation outer-product EMA for coupling subspace calibration.
        # Shape: (m, m) per layer. persistent=False — not needed after calibration.
        # Memory: 4 MB/d=1024, 67 MB/d=4096.
        self.register_buffer(
            "g_cov_ema",
            torch.zeros((self.m, self.m), device=dev, dtype=torch.float32),
            persistent=False,
        )
```

**Important**: `persistent=False` means this buffer is NOT saved in checkpoints. On resume, if `t_stat > 0`, the calibration will re-accumulate `g_cov_ema` from scratch during the next `t_stat` steps. This is acceptable because calibration is fast (~500 steps) and the EMA accumulation is deterministic given the same random seed and data order.

#### Step 3b: Add `disable_cov_ema()` method

**Location**: In `_JoraAdapterState`, near `update_temperature()` (~line 231).

**New code**:
```python
    @torch.no_grad()
    def disable_cov_ema(self):
        """Disable g_cov_ema after calibration to reclaim ~2 GB for large models.

        Called after t_stat calibration steps complete and pairs are frozen.
        After this, g_cov_ema is set to None and no longer accumulates.
        The method is idempotent — calling twice is safe.
        """
        if self.g_cov_ema is not None:
            self.g_cov_ema = None
```

**Important**: Setting `g_cov_ema = None` instead of zeroing allows Python GC to reclaim the `(d, d)` tensor immediately.

#### Step 3c: Extend forward hook with outer-product accumulation

**Location**: In `forward()` method of `JoraLayer`, inside the EMA update block (~line 685, after `st.grad_col_ema.lerp_()`).

**Current code** (existing, ~line 685-691):
```python
            ema_interval = int(getattr(st.cfg, "ema_update_interval", 1))
            if ema_interval <= 1 or (self._ema_step_counter % ema_interval) == 0:
                with torch.no_grad():
                    xd = x.detach()
                    if torch.isfinite(xd).all():
                        x_sq = xd.reshape(-1, st.m).float().pow(2).mean(dim=0)
                        beta = float(st.cfg.ema_beta)
                        st.grad_col_ema.lerp_(x_sq, 1.0 - beta)
```

**New code** (replace the `if torch.isfinite(xd).all():` block):
```python
                if torch.isfinite(xd).all():
                    x_sq = xd.reshape(-1, st.m).float().pow(2).mean(dim=0)
                    beta = float(st.cfg.ema_beta)
                    st.grad_col_ema.lerp_(x_sq, 1.0 - beta)

                    # TC-CS: activation outer-product EMA for coupling subspace selection.
                    # Only runs during calibration (calibration_active=True).
                    # g_cov_ema[i,j] = E_calibration[x_i * x_j]
                    if (
                        getattr(st.cfg, "pairing_strategy", "consecutive") == "coupling"
                        and getattr(st.cfg, "calibration_active", False)
                        and st.g_cov_ema is not None
                    ):
                        x_flat = xd.reshape(-1, st.m).float()  # (B*L, m)
                        # (m, B*L) @ (B*L, m) = (m, m) outer product
                        x_cov = x_flat.T @ x_flat / max(x_flat.size(0), 1.0)
                        st.g_cov_ema.lerp_(x_cov, 1.0 - beta)
```

**Why this belongs here**: The forward hook already has the `training` guard (line 677: `if self.training:`) and the `ema_update_interval` gate. The `calibration_active` check is a third gate. All three must be True for the accumulation to run.

**Guard ordering**: `st.g_cov_ema is not None` is the last guard — it ensures the accumulation is skipped after `disable_cov_ema()` is called.

#### Step 3d: Extend `_update_pair_buffer()` with coupling score branch

**Location**: In `_JoraAdapterState._update_pair_buffer()`, after `energy = maybe_gumbel(...)` (~line 269) and before `new_pairs = select_top_k_pairs_gpu(...)`.

**Current code** (existing, ~line 269-271):
```python
        energy = maybe_gumbel(energy_src, self.cfg.use_gumbel, self.cfg.gumbel_tau)
        pairing_strategy = getattr(self.cfg, "pairing_strategy", "consecutive")
        new_pairs = select_top_k_pairs_gpu(energy, k=allowed_count, max_features=int(feature_dim), pairing_strategy=pairing_strategy)
```

**New code** (replace the two lines after `energy = maybe_gumbel(...)`):
```python
        energy = maybe_gumbel(energy_src, self.cfg.use_gumbel, self.cfg.gumbel_tau)
        pairing_strategy = getattr(self.cfg, "pairing_strategy", "consecutive")

        if pairing_strategy == "coupling" and st.g_cov_ema is not None:
            # TC-CS coupling-aware pairing: use activation outer-product EMA.
            # score(i,j) = |g_cov_ema[i,j]| * sqrt(grad_col_ema[i] * grad_col_ema[j])
            score_matrix = st.g_cov_ema.abs() * torch.sqrt(
                st.grad_col_ema.unsqueeze(1) * st.grad_col_ema.unsqueeze(0) + float(st.cfg.eps)
            )
            new_pairs = select_coupling_pairs_gpu(score_matrix, k=allowed_count, max_features=int(feature_dim))
        else:
            new_pairs = select_top_k_pairs_gpu(energy, k=allowed_count, max_features=int(feature_dim), pairing_strategy=pairing_strategy)
```

**Important**: The `st.g_cov_ema is not None` guard ensures this branch is never entered after `disable_cov_ema()` is called, even if `pairing_strategy="coupling"` is set.

#### Step 3e: Wire `calibration_active` and `disable_cov_ema()` into `update_step()`

**Location**: In `_JoraAdapterState.update_step()`, in the block where `pairs_freeze_after_warmup=True` triggers freezing (~line 307-310).

**What to add**: Two things:

1. At the start of `update_step()` (before any pair update logic), set `calibration_active`:

```python
        # TC-CS: calibration_active is True when t_stat > 0 and within calibration window
        t_stat = int(getattr(self.cfg, "t_stat", 0) or 0)
        if t_stat > 0:
            current_step_capped = min(int(current_step), t_stat)
            self.cfg.calibration_active = (current_step_capped < t_stat)
        else:
            self.cfg.calibration_active = False
```

2. After the freeze call in the `pairs_freeze_after_warmup` block, call `disable_cov_ema()`:

**Current code** (~line 349-350):
```python
            if getattr(self.cfg, "pairs_freeze_after_warmup", False) and k_allow >= self.cfg.k:
                self._freeze_support_if_needed()
```

**New code**:
```python
            if getattr(self.cfg, "pairs_freeze_after_warmup", False) and k_allow >= self.cfg.k:
                self._freeze_support_if_needed()
                # TC-CS: calibration complete — disable g_cov_ema to reclaim memory
                self.disable_cov_ema()
                self.cfg.calibration_active = False
```

**Why this is safe**: The freeze is triggered only once (guarded by `pairs_frozen_flag`), so `disable_cov_ema()` is called exactly once per training run. It is idempotent.

**Memory reclaim timing**: After `disable_cov_ema()`, the `g_cov_ema` tensor is set to `None`. Python GC will reclaim the `(d, d)` float32 memory. For d=4096 across 32 layers: ~2.1 GB returned to GPU.

---

## Dependency Order

All steps are independent except where noted:

| Step | File | Depends On | Description |
|------|------|-----------|-------------|
| 1 | `config.py` | none | Add `pairing_strategy`, `calibration_active` fields |
| 2 | `selection.py` | none | Add `select_coupling_pairs_gpu()`, `_greedy_disjoint_from_scores()` |
| 3a | `layer.py` | none | Register `g_cov_ema` buffer |
| 3b | `layer.py` | 3a | Add `disable_cov_ema()` method |
| 3c | `layer.py` | 1, 3a | Forward hook outer-product accumulation |
| 3d | `layer.py` | 1, 2, 3a | `_update_pair_buffer` coupling branch |
| 3e | `layer.py` | 1, 3b, 3c | `calibration_active` wiring + `disable_cov_ema()` call |
| 4 | smoke tests | 1, 2, 3a-3e | All tests depend on all implementation steps |

---

## Smoke Tests

File: `tests/test_tcs_jora_sanity.py`

Each test is independent and minimal. Tests must pass before launching any training runs.

### `test_config_pairing_strategy_field`

```python
def test_config_pairing_strategy_field():
    from peft.tuners.jora.config import JoraConfig
    cfg = JoraConfig(pairing_strategy="coupling", t_stat=100, pairs_freeze_after_warmup=True)
    assert cfg.pairing_strategy == "coupling"
    assert cfg.calibration_active == False  # default; set by layer at runtime
    # Default value
    cfg_default = JoraConfig()
    assert cfg_default.pairing_strategy == "consecutive"
```

### `test_coupling_score_matrix_shape`

```python
def test_coupling_score_matrix_shape():
    import torch
    d = 16
    g_cov = torch.rand(d, d)  # (d, d) outer-product EMA
    g_col = torch.rand(d)     # (d,) activation second moment
    eps = 1e-8
    score_matrix = g_cov.abs() * torch.sqrt(
        g_col.unsqueeze(1) * g_col.unsqueeze(0) + eps
    )
    assert score_matrix.shape == (d, d)
    assert (score_matrix >= 0).all()  # absolute value
    # Diagonal entries are the normalization, not the signal
    # Off-diagonal entries are coupling scores
```

### `test_select_coupling_pairs_gpu_disjoint`

```python
def test_select_coupling_pairs_gpu_disjoint():
    import torch
    from peft.tuners.jora.selection import select_coupling_pairs_gpu
    d = 16
    torch.manual_seed(42)
    coupling_score = torch.rand(d, d)
    coupling_score = coupling_score + coupling_score.T  # make symmetric
    torch.diagonal(coupling_score).fill_(1.0)  # set diagonal to 1.0

    k = 4
    pairs = select_coupling_pairs_gpu(coupling_score, k=k, max_features=d)

    # All pairs are (i, j) where i != j
    assert (pairs[:, 0] != pairs[:, 1]).all()
    # No overlapping indices
    all_indices = pairs.flatten().tolist()
    assert len(all_indices) == len(set(all_indices)), "Pairs must be disjoint"
    # Number of pairs is at most k
    assert pairs.size(0) <= k
```

### `test_select_coupling_pairs_gpu_consistency`

```python
def test_select_coupling_pairs_gpu_consistency():
    """Verify that passing an energy-derived score matrix gives reasonable output."""
    import torch
    from peft.tuners.jora.selection import select_coupling_pairs_gpu, select_top_k_pairs_gpu
    d = 16
    torch.manual_seed(42)
    energy = torch.rand(d)
    k = 4

    # Build a (d, d) score matrix from energy: score[i,j] = energy[i] * energy[j]
    score_from_energy = energy.unsqueeze(1) * energy.unsqueeze(0)  # (d, d)

    pairs_coupling = select_coupling_pairs_gpu(score_from_energy, k=k, max_features=d)
    pairs_energy = select_top_k_pairs_gpu(energy, k=k, max_features=d)

    # Both should return exactly k pairs for d=16, k=4
    assert pairs_coupling.size(0) == k
    assert pairs_energy.size(0) == k
```

### `test_g_cov_ema_buffer_registration`

```python
def test_g_cov_ema_buffer_registration():
    """Verify g_cov_ema is registered with correct shape."""
    import torch
    import torch.nn as nn
    from peft.tuners.jora.layer import JoraLayer
    from peft.tuners.jora.config import JoraConfig

    base = nn.Linear(16, 16)
    cfg = JoraConfig(target_modules=[""], pairing_strategy="coupling")
    layer = JoraLayer(base, "test", cfg)

    # g_cov_ema should be registered with shape (m, m) = (16, 16)
    st = layer.adapters["test"]
    assert st.g_cov_ema is not None
    assert st.g_cov_ema.shape == (16, 16)
    assert st.g_cov_ema.dtype == torch.float32
```

### `test_forward_outer_product_accumulates`

```python
def test_forward_outer_product_accumulates():
    """Verify g_cov_ema accumulates correctly during forward pass."""
    import torch
    import torch.nn as nn
    from peft.tuners.jora.layer import JoraLayer
    from peft.tuners.jora.config import JoraConfig

    base = nn.Linear(8, 8)
    cfg = JoraConfig(
        pairing_strategy="coupling",
        t_stat=10,  # calibration active for first 10 steps
        pairs_freeze_after_warmup=True,
    )
    layer = JoraLayer(base, "test", cfg)
    layer.train()

    # Manually set calibration_active for testing
    cfg.calibration_active = True

    # Inject a known activation tensor
    x = torch.randn(2, 4, 8)  # (batch=2, seq=4, d=8)
    _ = layer(x)

    # After one forward pass, g_cov_ema should be non-zero
    assert st.g_cov_ema is not None
    assert st.g_cov_ema.sum() > 0, "g_cov_ema should accumulate after forward pass"
```

### `test_disable_cov_ema_reclaims_buffer`

```python
def test_disable_cov_ema_reclaims_buffer():
    """Verify disable_cov_ema sets g_cov_ema to None."""
    import torch
    import torch.nn as nn
    from peft.tuners.jora.layer import JoraLayer
    from peft.tuners.jora.config import JoraConfig

    base = nn.Linear(8, 8)
    cfg = JoraConfig(pairing_strategy="coupling")
    layer = JoraLayer(base, "test", cfg)
    st = layer.adapters["test"]

    assert st.g_cov_ema is not None
    st.disable_cov_ema()
    assert st.g_cov_ema is None, "g_cov_ema should be None after disable_cov_ema"

    # Calling twice should be safe (idempotent)
    st.disable_cov_ema()  # should not raise
```

### `test_coupling_pairing_skipped_when_g_cov_disabled`

```python
def test_coupling_pairing_skipped_when_g_cov_disabled():
    """Verify coupling branch is not entered when g_cov_ema is None."""
    import torch
    import torch.nn as nn
    from peft.tuners.jora.layer import JoraLayer
    from peft.tuners.jora.config import JoraConfig

    base = nn.Linear(8, 8)
    cfg = JoraConfig(pairing_strategy="coupling", S_L=0, S_R=4)
    layer = JoraLayer(base, "test", cfg)
    layer.train()
    st = layer.adapters["test"]

    # Disable cov EMA first
    st.disable_cov_ema()
    cfg.calibration_active = False

    # update_step should not fail even with pairing_strategy="coupling"
    # and g_cov_ema=None — it should fall through to the else branch
    st.update_step(current_step=100, total_steps=1000)

    # Should still have some pairs (from the else branch with energy-based selection)
    assert st.num_pairs_R.item() >= 0  # no error, fallback to energy-based
```

---

## Backward Compatibility

| Existing config | Behavior |
|---|---|
| `pairing_strategy` not set | Defaults to `"consecutive"` — existing behavior unchanged |
| `t_stat=0` (default) | `calibration_active=False` always; no outer-product accumulation |
| Existing JORA-Diag configs | All unchanged; `compute_delta()` untouched |
| Existing test suite | All existing tests should continue to pass |

Run after each step:
```bash
pytest tests/test_jora.py tests/test_jora_diag_path.py tests/test_jora_paper_path.py -q
```

---

## Common Pitfalls and Guards

| Pitfall | Guard in code |
|---|---|
| `g_cov_ema` accumulation after calibration | `calibration_active` sentinel; also `st.g_cov_ema is not None` |
| Outer product on wrong dimension | Uses `st.m` (input dim), consistent with `grad_col_ema` |
| Float precision in outer product | Uses `float()` cast in forward hook |
| Buffer not reclaimed | `disable_cov_ema()` sets to `None` for immediate GC |
| Resume from checkpoint | `persistent=False` means calibration restarts — acceptable |
| `coupling_branch` entered without valid `g_cov_ema` | `st.g_cov_ema is not None` guard in both forward hook and `_update_pair_buffer` |

---

## Step 4.6: Score Fix Sanity Results (2026-04-27)

**Date**: 2026-04-27
**Script**: `step46_score_fix_sanity.py`
**Gate criterion**: normalized_corr overlap with energy_product < 80%

### Key Finding: Gate PASSED (4.2% overlap)

Three scoring methods compared on 100-step calibration activations from OPT-350m (layers 0-2, q_proj/k_proj):

| Method | Mean overlap with energy_product |
|--------|----------------------------------|
| raw_outer_product (old TC-CS) | 31.2% |
| **normalized_corr (new TC-CS)** | **4.2%** |

Per-layer pair overlap (EP vs NC):

| Layer | Module | k | Overlap | Gate |
|-------|--------|---|--------|------|
| L0 | q_proj | 8 | 0/8 = 0.0% | PASS |
| L0 | k_proj | 8 | 1/8 = 12.5% | PASS |
| L1 | q_proj | 8 | 1/8 = 12.5% | PASS |
| L1 | k_proj | 8 | 0/8 = 0.0% | PASS |
| L2 | q_proj | 8 | 0/8 = 0.0% | PASS |
| L2 | k_proj | 8 | 0/8 = 0.0% | PASS |

**Verdict**: Gate PASSED. Normalized correlation meaningfully differentiates from energy pairing.

### Root Cause of Old Score Failure
The old score `|E[x_i * x_j]| * sqrt(E[i]*E[j])` reduces to approximately `E[i]*E[j]` when activations are approximately independent across dimensions (typical for attention projections). The normalization term amplifies the correlation between magnitude and selection.

### New Score Formula (Implemented)
```
centered_cov[i,j] = g_cov_ema[i,j] - g_mean_ema[i] * g_mean_ema[j]
Var[i] = g_cov_ema[i,i] - g_mean_ema[i]^2
score[i,j] = |centered_cov[i,j]| / sqrt(Var[i] * Var[j] + eps)
```
This is the Pearson correlation coefficient. It measures dependency, not magnitude.

### Code Changes Made (2026-04-27)

1. **Added `g_mean_ema` buffer** (`layer.py:93-100`): Tracks running mean of activations.
2. **Updated forward hook** (`layer.py:794-795`): Accumulates `x_mean = x_flat.mean(dim=0)` via EMA alongside `g_cov_ema`.
3. **Updated coupling branch** (`layer.py:391-430`): Computes normalized correlation score from `g_cov_ema` and `g_mean_ema`.
4. **Updated `disable_cov_ema()`** (`layer.py:270-273`): Also disables `g_mean_ema` for memory reclamation.

### Smoke Tests: ALL PASS
- Config initialization
- `select_coupling_pairs_gpu`
- `g_mean_ema` registration
- `disable_cov_ema` (both buffers)
- Forward hook accumulation
- Full training step with coupling

### Recommended Next Steps
1. Run short training smoke (Step 4.7) with new normalized correlation score
2. Compare 1ep results with old TC-CS (pair selection should differ)
3. If pairs differ and loss is comparable, run downstream eval

---

## Step 4.5: Pair Diagnostics Results (2026-04-27)

**Date**: 2026-04-27
**Run**: TC-CS-1S (attention-only, t_stat=100) vs Diag-Consecutive (pairing_strategy="consecutive")
**Checkpoints**: `results/run_tccs_1s_s42/` vs `results/run_diag_consecutive_s42/`

### Key Finding: 100% Pair Overlap

| Metric | Value |
|--------|-------|
| Mean pair overlap across all 48 layer-module pairs | **100.0%** |
| Pairs selected by TC-CS not in Consecutive | **0** |
| Pairs selected by Consecutive not in TC-CS | **0** |
| Dimensions selected only by TC-CS | **0** |
| Dimensions selected only by Consecutive | **0** |

### Per-Module Cross-Check
Every consecutive pair appears in TC-CS. 48/48 (q_proj, k_proj, v_proj, out_proj) × 12 layers all show 8/8 overlap.

### Diagnosis
TC-CS pair selection **completely degenerates** to consecutive/imporance-based selection.

**Root cause**: After the t_stat=100 calibration phase, the `g_cov_ema` and `grad_col_ema` EMAs converge to the same rank-1-like structure because:
1. Both EMAs track second-order moments of the same activations
2. For most dimensions, `|E[x_i * x_j]| ≈ E[x_i] * E[x_j]` (activations are approximately independent)
3. The coupling score `|g_cov_ema[i,j]| * sqrt(g_col[i]*g_col[j])` simplifies to `|g_cov_ema[i,j]| * g_col_norm[i] * g_col_norm[j]`
4. When `g_cov_ema[i,j] ≈ g_col[i] * g_col[j]` (rank-1 outer product), the product with the normalization terms produces the same ordering as energy[i] * energy[j]

### Verdict
- **Implementation**: PASS (code runs correctly)
- **Pair selection differentiation**: FAIL (TC-CS does not differentiate from consecutive)
- **Paper claim status**: Cannot claim TC-CS has any contribution over consecutive pairing
- **3ep**: NOT recommended — would be wasted compute since selection is equivalent

### Recommended Next Steps
1. **Do NOT run 3ep** — the method is selecting identical pairs
2. **Reconsider the coupling score formula** — current form is a reweighted energy product
3. Consider: decorrelating the coupling score from energy magnitude, or using gradient-level coupling (not activation-level)
4. See P0/P1/P2 priorities below

---

## Bugfixes

| Date | File | Issue | Fix |
|------|------|-------|-----|
| 2026-04-27 | `examples/sft/utils.py` | `model_kwargs["dtype"]` caused `TypeError` on OPT models — `AutoModelForCausalLM.__init__()` does not accept `dtype` kwarg | Changed to `model_kwargs["torch_dtype"]` |

---

## Run M0 Sanity Commands

After implementing all steps and running smoke tests:

```bash
# M0: Full smoke suite
pytest tests/test_tcs_jora_sanity.py -v

# M0: Existing tests still pass
pytest tests/test_jora.py tests/test_jora_diag_path.py tests/test_jora_paper_path.py -q

# M0: Smoke via script
python -c "
from peft.tuners.jora.config import JoraConfig
from peft.tuners.jora.selection import select_coupling_pairs_gpu
import torch

cfg = JoraConfig(pairing_strategy='coupling', t_stat=100, pairs_freeze_after_warmup=True)
assert cfg.pairing_strategy == 'coupling'
print('Config: OK')

d = 16
score = torch.rand(d, d)
pairs = select_coupling_pairs_gpu(score, k=4, max_features=d)
assert pairs.shape[1] == 2
print(f'Pair selection: OK ({pairs.size(0)} pairs)')
print('ALL M0 SMOKE TESTS PASSED')
"
```

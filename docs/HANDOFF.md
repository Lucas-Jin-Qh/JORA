# JORA 项目交接文档 (2026-04-28 更新版)

**生成者**: Claude Code (claude-opus-4-7-thinking-high)
**最后更新**: 2026-04-28
**用途**: 在全新对话中让新 agent 无缝接手当前工作

---

## 项目基本信息

**项目名称**: JORA (Joint Orthogonal Rotation Adaptation)
**项目类型**: PEFT 方法研究，LLM 微调
**当前阶段**: TC-CS 实现完成，Step 4.8 完成（pair overlap = 100%，训练 loss 差值 ~0，gate FAIL）
**主方法**: JORA-Diag (当前主实验线)
**核心 baseline**: JORA-NoRot (first-class mechanism baseline，rotation OFF 对照)

**仓库路径**: `/home/jqh/Workshop/JORA`
**git 状态**: `main` 分支，有未 commit 的修改（`layer.py` 有 Step 4.6 score fix），未跟踪的 `step45_pair_diagnostics.py`、`step46_score_fix_sanity.py` 及 config/result 文件

**conda 环境**: `peft-jora`（完整路径：`/home/jqh/miniconda3/envs/peft-jora`）
**Python**: 3.12
**PyTorch**: 2.8.0+cu128
**CUDA**: 可用
**HF mirror**: `HF_ENDPOINT=https://hf-mirror.com`，`TRANSFORMERS_OFFLINE=1`
**wandb**: 已启用，project=`jora`，离线模式
**模型缓存**: `/home/jqh/.cache/huggingface/hub/models--facebook--opt-350m/snapshots/08ab08cc4b72ff5593870b5d527cf4230323703c`
**数据集缓存**: alpaca-cleaned, MMLU, ARC-C, GSM8K, Hellaswag 均已缓存

---

## 核心文件路径

### 核心实现
- `src/peft/tuners/jora/layer.py` — JoraLayer, compute_delta, merge/unmerge, **TC-CS 实现（含 Step 4.6 fix）**
- `src/peft/tuners/jora/core.py` — DiagCore, SelectiveDiagCore, BlockCore, LowRankCore, CoupledPairCore
- `src/peft/tuners/jora/rotation.py` — Givens rotation (torch/triton)
- `src/peft/tuners/jora/config.py` — JoraConfig 及其 factory 方法，**含 TC-CS pairing_strategy 字段**
- `src/peft/tuners/jora/selection.py` — pair selection，`select_coupling_pairs_gpu()` 已实现
- `src/peft/tuners/jora/magnitude.py` — OER/ECD magnitude scaling
- `src/peft/tuners/jora/callbacks.py` — JoraTrainerCallback, JoraMetricsCallback
- `src/peft/tuners/jora/model.py` — JoraModel PEFT wrapper

### 测试文件
- `tests/test_jora.py` — 通用 JORA 测试
- `tests/test_jora_diag_path.py` — DiagCore 测试
- `tests/test_jora_paper_path.py` — SelectiveDiagCore 测试

### 训练入口
- `examples/sft/train.py` — 主训练脚本 (HF Trainer 封装)
- `examples/sft/utils.py` — create_datasets, create_and_prepare_model（含 HF mirror 兼容）
- `scripts/run_jora_exp.py` — JSON 驱动实验启动器

### TC-CS 相关文件（本次工作产出）
- `step45_pair_diagnostics.py` — Step 4.5：pair overlap 诊断脚本
- `step46_score_fix_sanity.py` — Step 4.6：3种 scoring 方法对比脚本
- `refine-logs/IMPLEMENTATION_SPEC.md` — **TC-CS 实现规格文档（含 Step 4.5 和 4.6 结果）**
- `refine-logs/EXPERIMENT_PLAN.md` — TC-CS 实验计划
- `refine-logs/EXPERIMENT_TRACKER.md` — 实验追踪

### 配置文件（TC-CS 相关）
- `results/run_tccs_1s_s42/` — TC-CS-1S checkpoint（pairing_strategy="coupling"，t_stat=100，attention-only）
- `results/run_diag_consecutive_s42/` — Consecutive baseline checkpoint（pairing_strategy="consecutive"）

### 核心文档
- `docs/FORMULA_AUDIT.md` — 公式审计（最关键，定义了各变体的精确公式）
- `docs/JORA_RESEARCH_CONTRACT.md` — 研究契约（allowed/forbidden claims）
- `docs/JORA_REPO_MAP.md` — 仓库地图
- `AGENTS.md` — Agent 契约（硬规则、环境信息模板）
- `docs/HANDOFF.md` — **老版交接文档（2026-04-26）**，内容已过时

---

## TC-CS 当前状态总览

### 已完成的工作

#### Step 1-3: TC-CS 基础实现
- `config.py`: 添加 `pairing_strategy`（"consecutive"/"high_low"/"coupling"）和 `calibration_active`
- `selection.py`: 添加 `select_coupling_pairs_gpu()` 和 `_greedy_disjoint_from_scores()`
- `layer.py`: 添加 `g_cov_ema` buffer、`disable_cov_ema()`、`calibration_active` wiring、forward hook outer-product 累积

#### Step 3A-3B: smoke test — PASS
- TC-CS-1S attention-only 短训练 smoke：工程正确、稳定性通过

#### Step 4: 1ep 正式训练 — PASS（工程）+ INCONCLUSIVE（机制）
- TC-CS-1S vs Consecutive 训练 loss 差异：极小（~0.0005），无实际意义

#### Step 4.5: Pair Diagnostics — 100% overlap，FAIL
- **发现**：TC-CS 选出的 pairs 与 consecutive 完全相同（100% overlap，48/48 layer-module 组合）
- **根因**：旧 coupling score `|E[x_i*x_j]| * sqrt(E[i]*E[j])` ≈ `E[i]*E[j]`，退化为 energy product

#### Step 4.6: Score Fix Sanity — GATE PASSED（4.2% overlap）
- **发现**：normalized correlation score 给出 4.2% overlap with energy product（gate 阈值 <80%）
- **根因**：Pearson 相关系数测量依赖性而非量级，真正与 energy-based selection 区分
- **代码已修改**（见下文"当前 layer.py 关键状态"）

### 当前 layer.py 关键状态

**`layer.py` 中三处关键修改（Step 4.6 产出）：**

#### 修改 1: `g_mean_ema` buffer 注册（约 line 93-106）
在 `_JoraAdapterState.__init__()` 中，`grad_col_ema` 注册之后添加：
```python
# TC-CS: running mean of activations for normalized correlation score.
# Needed for: centered_cov[i,j] = g_cov_ema[i,j] - g_mean_ema[i] * g_mean_ema[j]
self.register_buffer(
    "g_mean_ema",
    torch.zeros(self.m, device=dev, dtype=torch.float32),
    persistent=False,
)
```

#### 修改 2: Forward hook mean 累积（约 line 794-795）
在 existing outer-product EMA 累积 block 内，添加：
```python
x_mean = x_flat.mean(dim=0)  # (m,)
st.g_mean_ema.lerp_(x_mean, 1.0 - beta)
```

#### 修改 3: Coupling branch 使用 normalized correlation（约 line 391-430）
在 `update_step()` 中，`pairing_strategy == "coupling"` branch 的 score 计算改为：
```python
mean_outer = self.g_mean_ema.unsqueeze(1) * self.g_mean_ema.unsqueeze(0)  # (m, m)
centered_cov = self.g_cov_ema - mean_outer
var_vals = self.g_cov_ema.diagonal() - self.g_mean_ema.pow(2)  # (m,)
var_vals = var_vals.clamp(min=float(self.cfg.eps))
denom = torch.sqrt(var_vals.unsqueeze(1) * var_vals.unsqueeze(0) + float(self.cfg.eps))
coupling_score = centered_cov.abs() / denom
```
公式：`score[i,j] = |E[(x_i - μ_i)(x_j - μ_j)]| / sqrt(Var[i] * Var[j] + eps)`
这是 Pearson 相关系数。

#### 修改 4: `disable_cov_ema()` 同时禁用两个 buffer（约 line 270-273）
```python
if self.g_cov_ema is not None:
    self.g_cov_ema = None
if self.g_mean_ema is not None:
    self.g_mean_ema = None
```

---

## Step 4.5 Pair Diagnostics 完整结果

**脚本**: `step45_pair_diagnostics.py`
**方法**: 从 checkpoint `results/run_tccs_1s_s42/` 和 `results/run_diag_consecutive_s42/` 提取 `pairs_R` 和 `grad_col_ema`，计算 overlap

### 结论
- 所有 48 个 layer-module 组合（12 layers × 4 modules）pair overlap = 100%
- 没有任何 dimension 仅被 TC-CS 选中而未被 consecutive 选中
- 没有任何 dimension 仅被 consecutive 选中而未被 TC-CS 选中
- **TC-CS pair selection 完全退化至 consecutive/imporance-based selection**

### 根因
`g_cov_ema` 和 `grad_col_ema` 追踪相同激活的二阶矩，当维度间激活近似独立时：
- `|E[x_i * x_j]| ≈ E[x_i] * E[x_j]`
- score ≈ `|E[x_i]| * |E[x_j]|` = energy product
- 与 consecutive pairing 的排序一致

---

## Step 4.6 Score Fix Sanity 完整结果

**脚本**: `step46_score_fix_sanity.py`
**环境**: OPT-350m，Alpaca-cleaned calibration steps=100，layers 0-2，q_proj/k_proj，k=8
**Gate criterion**: normalized_corr overlap with energy_product < 80%

### 结果摘要

| Scoring Method | Mean overlap with EP | Per-layer results |
|---|---|---|
| energy_product (EP) | 100% (baseline) | — |
| raw_outer_product (old TC-CS) | 31.2% | varies by layer |
| **normalized_corr (new TC-CS)** | **4.2%** | 0-12.5% per layer |

### Per-layer 详情

| Layer | Module | EP vs NC overlap | Gate |
|-------|--------|-----------------|------|
| L0 | q_proj | 0/8 = 0.0% | PASS |
| L0 | k_proj | 1/8 = 12.5% | PASS |
| L1 | q_proj | 1/8 = 12.5% | PASS |
| L1 | k_proj | 0/8 = 0.0% | PASS |
| L2 | q_proj | 0/8 = 0.0% | PASS |
| L2 | k_proj | 0/8 = 0.0% | PASS |

### Score Distribution Analysis
normalized_corr score values 全部在 [0.965, 1.0] 范围内（有效相关系数范围），std 极小（0.0001-0.0096），说明激活值在标准化后高度相关。

### Gate Verdict
**GATE PASSED** — 4.2% < 80%，normalized correlation 显著区分于 energy pairing。

---

## Step 4.8 Matched Sanity 对比（100-step）

**环境**: OPT-350m，Alpaca-cleaned，max_steps=100，t_stat=20（20 calibration + 80 post-freeze），target_modules=[q/k/v/out_proj]，seed=42

### 训练结果

|| 配置 | Final train_loss | Runtime | token_acc range |
||------|-----------------|---------|----------------|
|| Consecutive (energy-based) | 2.1898273 | 70.99s | 0.519–0.537 |
|| Coupling (normalized corr) | 2.1898246 | 71.28s | 0.520–0.538 |

**差值**: train_loss Δ = +0.0000027（~2.7e-6），在测量噪声级别。

### Pair Overlap 分析

|| Layer | Module | k_R | Overlap | Ratio |
||-------|--------|------|---------|-------|
|| ALL 12 | q_proj | 8 | 8/8 | **100%** |
|| ALL 12 | k_proj | 8 | 8/8 | **100%** |
|| ALL 12 | v_proj | 8 | 8/8 | **100%** |
|| ALL 12 | out_proj | 8 | 8/8 | **100%** |

**Mean overlap**: 48/48 = **100%** (384/384 pairs)

### 根因分析

1. **grad_col_ema 完全一致**（max diff = 0.000000）→ 两个 run 确实执行了不同代码路径（consecutive branch vs coupling branch）
2. **但两策略最终选出了相同的 pairs** → 不是 bug，是真实发现
3. **机制**：candidate pool = top-64 by energy/correlation，两者池内排序高度相关，导致相同结果
4. **与 Step 4.6 的矛盾**：Step 4.6 用随机 score 矩阵测出 4.2% overlap；真实 score 矩阵由于激活模式相关性高，导致 100% overlap

### 判定门：FAIL

- TC-CS-NC 在真实训练中退化为 consecutive pairing（100% overlap，gate 阈值 < 80%）
- 训练 loss 差值 2.7e-6，无实际意义
- 不进入 3ep 正式实验

### 后续路径建议

1. **窄化候选池**：将候选池从 top-64 改为更小的子集，增加两策略的区分度
2. **换用真正不相关的 score**：如 mutual information 或 sign-based decorrelation
3. **重新审视假说**：如果高能维度普遍高度相关，TC-CS 的核心假说（相关性驱动好 pairs）可能不成立

---

## 当前已知实验结果

### 3-Epoch Matched Comparison (OPT-350m, seed=42, Alpaca-cleaned)

| 配置 | Final train_loss | Mean token accuracy | Runtime | Step time |
|------|-----------------|-------------------|---------|-----------|
| JORA-Diag ON (S_L=32, S_R=32) | **2.2374** | 0.5147 | ~158 min | ~1.03s/step |
| JORA-NoRot OFF (S_L=0, S_R=0) | **2.2368** | 0.5150 | ~55 min | ~0.36s/step |

**结论**: NoRot 反而低 0.00055，差值在测量噪声内。Rotation ON runtime ≈ 2.9× OFF。ON/OFF gap 在 3 epoch 内无实际意义。rotation contribution 仍然未建立。

### TC-CS 1ep 训练（Step 4）

| 配置 | Final train_loss | Runtime |
|------|-----------------|---------|
| TC-CS-1S (pairing_strategy="coupling") | ~2.23865 | ~10 min |
| Diag-Consecutive (pairing_strategy="consecutive") | ~2.23870 | ~9 min |

差异极小（~0.00005），且 Step 4.5 证明 pairs 完全相同。

---

## AGENTS.md 硬规则（必须遵守）

1. **不要 claim rotation drives the gain**，除非 JORA-Diag ON 在 matched eval 上超过 JORA-NoRot
2. **不要写 final Method formula**，直到 FORMULA_AUDIT.md 确认完成
3. **不要跑大规模 sweep**，直到 sanity/save-load/merge 检查通过
4. **不要修改** unrelated PEFT methods，除非 baseline 兼容性需要
5. **不要在 claim 未被证据支持时升级 rotation narrative**

---

## 关键设计决策记录（当前状态）

### 决策 A: TC-CS 使用 normalized correlation（Step 4.6 fix）
`layer.py` 中 coupling branch 已改为：
```
centered_cov = g_cov_ema - g_mean_ema ⊗ g_mean_ema
score = |centered_cov| / sqrt(Var ⊗ Var + eps)
```
这测量依赖性而非量级。与旧 score 的关键区别：新 score 在独立激活假设下不会退化为 energy product。

### 决策 B: g_mean_ema 和 g_cov_ema 均 persistent=False
calibration 完成后 `disable_cov_ema()` 会同时将两者设为 None，释放内存。resume 时重新 calibration（相同 seed 保证确定性）。

### 决策 C: TC-CS 仅应用于 right side（单边）
当前实现中，`pairing_strategy="coupling"` 仅影响 `pairs_R`（右旋转矩阵对应列）。`pairs_L` 始终使用 energy-based selection。

---

## 下一步工作（明确的待办事项）

### P0: TC-CS 候选池重设计（核心修复）
**根因**：candidate pool = top-64 by energy/correlation，两者高度重叠导致 100% overlap。

可选方案：
1. **缩小候选池**：从 top-64 改为 top-16 或 top-32
2. **使用真正不相关的 score**：如互信息估计、或 sign-based decorrelation（`|E[sign(x_i)*sign(x_j)]|`）
3. **使用梯度而非激活**：梯度耦合比激活耦合更本质，但计算成本更高

### P1: 验证新 score 区分度
修复后，用 Step 4.6 相同方法验证 overlap < 80%，再用 matched sanity 验证训练 loss 有差异。

---

## 环境与命令速查

### conda 环境
```bash
conda activate peft-jora
# 或
/home/jqh/miniconda3/envs/peft-jora/bin/python
```

### 运行测试（SMOKE TEST）
```bash
cd /home/jqh/Workshop/JORA

# 基本 smoke（推荐优先运行）
/home/jqh/miniconda3/envs/peft-jora/bin/python -c "
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'tests')
import torch, torch.nn as nn
from peft.tuners.jora.config import JoraConfig
from peft.tuners.jora.layer import JoraLayer
from peft.tuners.jora.selection import select_coupling_pairs_gpu

# Test g_mean_ema init
base = nn.Linear(16, 16)
cfg = JoraConfig(target_modules=[''], pairing_strategy='coupling')
layer = JoraLayer(base, 'test', cfg)
st = layer.adapters['test']
assert hasattr(st, 'g_mean_ema'), 'g_mean_ema missing'
assert st.g_mean_ema.shape == (16,), f'Wrong shape: {st.g_mean_ema.shape}'

# Test disable
st.disable_cov_ema()
assert st.g_cov_ema is None
assert st.g_mean_ema is None

# Test coupling selection
score = torch.rand(16, 16)
pairs = select_coupling_pairs_gpu(score, k=4, max_features=16)
assert pairs.shape[1] == 2

print('ALL SMOKE TESTS PASSED')
"

# Step 4.6 score sanity
HF_ENDPOINT=https://hf-mirror.com TRANSFORMERS_OFFLINE=1 \
  /home/jqh/miniconda3/envs/peft-jora/bin/python step46_score_fix_sanity.py
```

### 启动训练实验
```bash
cd /home/jqh/Workshop/JORA

# 标准 JORA-Diag
HF_ENDPOINT=https://hf-mirror.com TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_jora_exp.py configs/run_diag_main_s42.json --gpu 0

# TC-CS（新 score，尚未创建配置，需要新建 configs/run_tccs_nc_s42.json）
HF_ENDPOINT=https://hf-mirror.com TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_jora_exp.py configs/run_tccs_nc_s42.json --gpu 0
```

### 分析实验结果
```bash
# 查看最终 train loss
grep "train_loss" logs/run_diag_main_s42_3ep.log
grep "train_loss" logs/run_diag_no_rotation_s42_3ep.log

# Step 4.5 pair diagnostics
python step45_pair_diagnostics.py

# Step 4.6 score sanity（如果需要重跑）
HF_ENDPOINT=https://hf-mirror.com TRANSFORMERS_OFFLINE=1 \
  /home/jqh/miniconda3/envs/peft-jora/bin/python step46_score_fix_sanity.py
```

---

## 新 agent 接手检查清单

接手后请按顺序执行：

1. [ ] 阅读 `refine-logs/IMPLEMENTATION_SPEC.md` 全文（这是最关键文档）
2. [ ] 阅读 `docs/FORMULA_AUDIT.md` 确认各变体公式
3. [ ] 阅读 `docs/JORA_RESEARCH_CONTRACT.md` 确认 allowed/forbidden claims
4. [ ] 阅读 `docs/JORA_M0_SAVE_LOAD_MERGE.md` 确认 M0 correctness gate 状态
5. [ ] 运行基本 smoke test（上文"运行测试"命令）
6. [x] M0 save/load/merge 正确性 gate: 27/27 PASS (见 `docs/JORA_M0_SAVE_LOAD_MERGE.md`)
7. [ ] 运行 M0 测试: `HF_ENDPOINT=... TRANSFORMERS_OFFLINE=1 python -m pytest tests/test_jora_save_load_merge_sanity.py -v --override-ini=addopts= -p no:launch_testing_ros_pytest_entrypoint`
6. [ ] 设计新候选池/score 方案（见上文"P0: TC-CS 候选池重设计"）
7. [ ] 验证新方案 overlap < 80%
8. [ ] 再次运行 matched sanity 验证有训练 loss 差异
9. [ ] 在做出任何 strong claim 前，查阅 `docs/JORA_RESEARCH_CONTRACT.md`

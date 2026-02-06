# JORA Hyperparameter Sweep 实验配置

本目录包含 JORA 超参数扫描实验的完整配置、脚本和记录模板。

## 📁 文件结构

```
jora_sweep/
├── run_sweep.py                      # 自动化扫描脚本（单模型单数据集）
├── analyze_sweep.py                  # 结果分析脚本
├── experiment_results_template.csv   # 实验记录模板
├── sweep_summary.json                # 实验汇总配置 (v1.0, 41实验)
├── sweep_summary_extended.json       # 完整汇总配置 (v2.0, 93实验)
│
├── [GPU 运行脚本 - 推荐使用]
├── gpu0_jora_sweep.sh               # GPU0 串行脚本（双模型×双数据集×3 seeds）
├── gpu1_jora_sweep.sh               # GPU1 串行脚本
│
├── [核心实验配置]
├── main_config.json                  # 主实验配置（推荐）
├── block_s_k_sweep.json              # Phase 1: S/k 扫描 (10组)
├── block_size_sweep.json             # Phase 2: Block Size 扫描 (3组)
├── diag_baseline.json                # Diag 基线对比 (3组)
├── selection_refine.json             # Phase 3: Selection 细化 (7组)
├── magnitude_compare.json            # Phase 4: Magnitude 对比 (6组)
├── rotation_compare.json             # Phase 5: Rotation 对比 (2组)
│
├── [消融实验]
├── ablation_study.json               # Phase 6: 结构消融 (9组)
├── advanced_ablation.json            # 高级消融 (8组)
│
├── [Core 类型探索]
├── lowrank_explore.json              # LowRank Core 探索 (4组)
├── fine_block_size_scan.json         # 细粒度 Block Size (5组)
│
├── [轨道配置]
├── all_linear_track.json             # B轨 All-Linear (4组)
│
├── [参数交互]
├── learning_rate_scan.json           # 学习率交互 (8组)
├── ema_grid_scan.json                # EMA 参数网格 (10组)
├── batch_lr_interaction.json         # Batch/LR 交互 (6组)
│
├── [探索性实验]
├── asymmetric_scan.json              # 不对称 S_L/S_R (6组)
├── pairing_strategy_scan.json        # Pairing Strategy (4组)
├── temperature_annealing.json        # 温度退火 (6组)
│
├── [对比实验]
├── dataset_comparison.json           # 数据集对比 (5组)
├── model_comparison.json             # 模型对比 (4组)
│
└── README.md                         # 本文档
```

## 🚀 快速开始（推荐方式）

### 使用 GPU 脚本运行全部实验（两个模型 × 两个数据集）

```bash
# 在 GPU0 上运行
nohup bash gpu0_jora_sweep.sh > logs/gpu0_sweep.log 2>&1 &

# 在 GPU1 上运行（并行）
nohup bash gpu1_jora_sweep.sh > logs/gpu1_sweep.log 2>&1 &
```

### 或使用 Python 脚本（Dry Run 预览）

```bash
cd /home/jqh/Workshop/JORA/config/jora_sweep
python run_sweep.py --dry-run
```

### 执行特定阶段

```bash
# Phase 1: S/k 扫描
python run_sweep.py --phase 1

# Phase 2: Block Size 扫描
python run_sweep.py --phase 2
```

## 📊 实验阶段概览（完整版）

| Phase | 配置 | 实验数 | 优先级 | 预计时间 |
|-------|------|--------|--------|----------|
| 0 | main_config.json | 1 | P0 | 2h |
| 1 | block_s_k_sweep.json | 10 | P0 | 24h |
| 2 | block_size_sweep.json | 3 | P1 | 6h |
| 3 | diag_baseline.json | 3 | P0 | 6h |
| 4 | selection_refine.json | 7 | P1 | 12h |
| 5 | magnitude_compare.json | 6 | P2 | 6h |
| 6 | rotation_compare.json | 2 | P3 | 2h |
| 7 | ablation_study.json | 9 | P0 | 18h |
| 8 | lowrank_explore.json | 4 | P2 | 8h |
| 9 | asymmetric_scan.json | 6 | P3 | 8h |
| 10 | pairing_strategy_scan.json | 4 | P3 | 4h |
| 11 | learning_rate_scan.json | 8 | P2 | 12h |
| 12 | fine_block_size_scan.json | 5 | P2 | 8h |
| 13 | all_linear_track.json | 4 | P1 | 12h |
| 14 | temperature_annealing.json | 6 | P3 | 6h |
| 15 | ema_grid_scan.json | 10 | P2 | 12h |
| 16 | dataset_comparison.json | 5 | P1 | 20h |
| 17 | model_comparison.json | 4 | P1 | 16h |
| 18 | batch_lr_interaction.json | 6 | P2 | 8h |
| 19 | advanced_ablation.json | 8 | P3 | 8h |

**总计**: 93 组实验，约 2-3 周 (A100 8卡)

## 🎯 GPU 脚本使用说明

### gpu0_jora_sweep.sh

在 GPU0 上运行所有 JORA sweep 配置：
- **模型**: Llama2-7B, Mistral-7B
- **数据集**: GSM8K, Alpaca
- **Seeds**: 42, 1337, 2026
- **输出**: `checkpoints/jora_sweep/`

```bash
# 直接运行
bash gpu0_jora_sweep.sh

# 后台运行并记录日志
nohup bash gpu0_jora_sweep.sh > logs/gpu0.log 2>&1 &
```

### gpu1_jora_sweep.sh

与 GPU0 脚本配合使用，同样运行所有配置。

## 📋 输出目录结构

```
checkpoints/jora_sweep/
├── main_config.json/
│   ├── llama2_7b/
│   │   ├── gsm8k/
│   │   │   ├── seed42/
│   │   │   ├── seed1337/
│   │   │   └── seed2026/
│   │   └── alpaca/
│   │       └── ...
│   └── mistral_7b/
│       └── ...
├── block_s_k_sweep.json/
│   └── ...
├── sweep_summary.csv                  # 自动生成的结果汇总
└── train.log                          # 每个实验的日志
```

## 📈 结果分析

### 分析所有结果

```bash
python analyze_sweep.py
```

### 生成论文表格

```bash
python analyze_sweep.py --generate-tables --output_dir paper_tables
```

## 🔧 关键超参数说明

### Core 类型（根据您的预实验）

| 类型 | 能力 | 参数量 | 推荐场景 |
|------|------|--------|----------|
| **block** | 最强 | 中等 | 主实验配置 |
| diag | 中等 | 最小 | 低预算 baseline |
| lowrank | 高 | 较多 | 强表达能力 |

### S/k 参数

| S | k | Coverage | 参数量影响 |
|---|----|----------|------------|
| 32 | 8 | 0.20% | ~最小 |
| 64 | 16 | 0.39% | ~最小 |
| 96 | 24 | 0.59% | ~最小 |
| 128 | 32 | 0.78% | ~+2% |

### Selection 机制

| 类型 | 说明 |
|------|------|
| topk_ema | 默认推荐，稀疏更新 |
| random | 随机基线 |
| none | 静态更新 |

## ⚠️ 注意事项

1. **GPU 脚本**: 自动遍历两个模型和两个数据集，3 个 seed 串行执行
2. **显存**: Block size 越大，All-Linear 轨道显存占用越高
3. **结果汇总**: 脚本会自动生成 `sweep_summary.csv`
4. **断点续传**: 如需中断后继续，可修改脚本添加检查逻辑

## 📞 常见问题

### Q: 训练失败怎么办？
A: 查看 `checkpoints/jora_sweep/*/train.log` 中的错误信息。

### Q: 如何只运行部分配置？
A: 修改脚本中的 `sweep_configs` 数组。

### Q: 如何修改学习率或其他参数？
A: 直接编辑脚本顶部的变量：`learning_rate`, `num_epochs`, `batch_size` 等。

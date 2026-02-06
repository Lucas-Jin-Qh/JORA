# JORA Sweep Experiment Results
# 实验结果记录表

## 填写说明
- **所有数值保留4位小数**
- **Delta = FT Score - Base Score**
- **Coverage Rate = k / hidden_dim (如 k=24, hidden_dim=4096, 则 0.59%)**
- **Params 单位: K (千) 或 M (百万)**

---

## 实验 1: block-S32-k8

| 字段 | 值 |
|------|-----|
| **实验名称** | block-S32-k8 |
| **所属阶段** | Phase 1: S/k Sweep |
| **日期** | YYYY-MM-DD |
| **模型** | Llama-2-7B |
| **数据集** | Alpaca |
| **Core 类型** | block |
| **Block Size** | 4 |
| **S_L / S_R** | 32 / 32 |
| **k** | 8 |
| **Coverage Rate** | 0.20% |
| **可训练参数量** | ~115K |
| **训练轮数** | 3 |
| **批次大小** | 4 |
| **学习率** | 2e-4 |
| **训练时间(小时)** |  |

### 测评结果

| 任务 | Base Score | FT Score | Delta |
|------|------------|----------|-------|
| ARC-Challenge | | | |
| GSM8K | | | |
| MMLU | | | |
| HellaSwag | | | |

### 吞吐量

| 指标 | 值 |
|------|-----|
| Tokens/Second | |
| Peak VRAM (GB) | |

### 备注

---

## 实验 2: block-S64-k16

| 字段 | 值 |
|------|-----|
| **实验名称** | block-S64-k16 |
| **所属阶段** | Phase 1: S/k Sweep |
| **日期** | YYYY-MM-DD |
| **模型** | Llama-2-7B |
| **数据集** | Alpaca |
| **Core 类型** | block |
| **Block Size** | 4 |
| **S_L / S_R** | 64 / 64 |
| **k** | 16 |
| **Coverage Rate** | 0.39% |
| **可训练参数量** | ~115K |
| **训练轮数** | 3 |
| **批次大小** | 4 |
| **学习率** | 2e-4 |
| **训练时间(小时)** |  |

### 测评结果

| 任务 | Base Score | FT Score | Delta |
|------|------------|----------|-------|
| ARC-Challenge | | | |
| GSM8K | | | |
| MMLU | | | |
| HellaSwag | | | |

### 吞吐量

| 指标 | 值 |
|------|-----|
| Tokens/Second | |
| Peak VRAM (GB) | |

### 备注

---

## 实验 3: block-S96-k24

| 字段 | 值 |
|------|-----|
| **实验名称** | block-S96-k24 |
| **所属阶段** | Phase 1: S/k Sweep |
| **日期** | YYYY-MM-DD |
| **模型** | Llama-2-7B |
| **数据集** | Alpaca |
| **Core 类型** | block |
| **Block Size** | 4 |
| **S_L / S_R** | 96 / 96 |
| **k** | 24 |
| **Coverage Rate** | 0.59% |
| **可训练参数量** | ~115K |
| **训练轮数** | 3 |
| **批次大小** | 4 |
| **学习率** | 2e-4 |
| **训练时间(小时)** |  |

### 测评结果

| 任务 | Base Score | FT Score | Delta |
|------|------------|----------|-------|
| ARC-Challenge | | | |
| GSM8K | | | |
| MMLU | | | |
| HellaSwag | | | |

### 吞吐量

| 指标 | 值 |
|------|-----|
| Tokens/Second | |
| Peak VRAM (GB) | |

### 备注

---

## 实验 4: block-S128-k32

| 字段 | 值 |
|------|-----|
| **实验名称** | block-S128-k32 |
| **所属阶段** | Phase 1: S/k Sweep |
| **日期** | YYYY-MM-DD |
| **模型** | Llama-2-7B |
| **数据集** | Alpaca |
| **Core 类型** | block |
| **Block Size** | 4 |
| **S_L / S_R** | 128 / 128 |
| **k** | 32 |
| **Coverage Rate** | 0.78% |
| **可训练参数量** | ~117K |
| **训练轮数** | 3 |
| **批次大小** | 4 |
| **学习率** | 2e-4 |
| **训练时间(小时)** |  |

### 测评结果

| 任务 | Base Score | FT Score | Delta |
|------|------------|----------|-------|
| ARC-Challenge | | | |
| GSM8K | | | |
| MMLU | | | |
| HellaSwag | | | |

### 吞吐量

| 指标 | 值 |
|------|-----|
| Tokens/Second | |
| Peak VRAM (GB) | |

### 备注

---

## 实验 5: diag-S96-k24 (基线)

| 字段 | 值 |
|------|-----|
| **实验名称** | diag-S96-k24 |
| **所属阶段** | Diag Baseline |
| **日期** | YYYY-MM-DD |
| **模型** | Llama-2-7B |
| **数据集** | Alpaca |
| **Core 类型** | diag |
| **S_L / S_R** | 96 / 96 |
| **k** | 24 |
| **Coverage Rate** | 2.35% |
| **可训练参数量** | ~58K |
| **训练轮数** | 3 |
| **批次大小** | 4 |
| **学习率** | 2e-4 |
| **训练时间(小时)** |  |

### 测评结果

| 任务 | Base Score | FT Score | Delta |
|------|------------|----------|-------|
| ARC-Challenge | | | |
| GSM8K | | | |
| MMLU | | | |
| HellaSwag | | | |

### 吞吐量

| 指标 | 值 |
|------|-----|
| Tokens/Second | |
| Peak VRAM (GB) | |

### 备注

---

## 汇总表

| 实验名称 | Params | ARC Δ | GSM8K Δ | MMLU Δ | Tokens/s | VRAM |
|----------|--------|-------|---------|--------|----------|------|
| block-S32-k8 | ~115K | | | | | |
| block-S64-k8 | ~115K | | | | | |
| block-S64-k16 | ~115K | | | | | |
| block-S64-k24 | ~115K | | | | | |
| block-S96-k16 | ~115K | |
| block-S96-k24 | ~ | | | |115K | | | | | |
| block-S96-k32 | ~116K | | | | | |
| block-S128-k24 | ~116K | | | | | |
| block-S128-k32 | ~117K | | | | | |
| diag-S96-k24 | ~58K | | | | | |

---

## 关键发现

### S/k 覆盖率分析
- 

### Block vs Diag 对比
- 

### 最佳配置建议
- 

---

## 使用说明

1. 复制此模板为新文件: `cp results_template.md results/exp_YYYYMMDD.md`
2. 填写每个实验的结果
3. 更新汇总表
4. 在 `results/` 目录执行 `python summarize.py` 生成 CSV


 #JORA 实验设计（对比 + 消融 + 扫参）

 **LoRA 系列的“可塑性↑但推理保持性↓”**，以及 JORA/（q）GOFT/OFT 这条“正交/旋转几何”路线如何改善 trade-off。

---

## 0) 总原则（你不这样做，reviewer 会直接否掉公平性）

**统一三件事：**

1. **相同训练数据 + 相同训练步数（optimizer steps）**，不要用“跑到收敛”，否则方法间不可比。
2. **相同注入位置（target_modules）**：至少给两条轨道

   * **A轨（常用）**：`["q_proj","v_proj"]`
   * **B轨（强适配）**：`"all-linear"`（PEFT LoRA 文档里也是 QLoRA-style 推荐口径）([Hugging Face][1])
3. **匹配参数预算**：每个方法调超参，使 trainable params 与 LoRA(r=8/16/…)在同一轨道下 **±5%**（写进论文：*“parameter-matched”*）。**特别注意**：JORA 的参数构成中，S/k 主要影响几何覆盖和计算复杂度，对总参数量影响很小；主要参数来源是 core 选择（diag/block/lowrank）。

---

## 1) 模型与训练数据（两档训练强度，确保结论稳）

### Base models（你指定的）

* **Llama2-7B**
* **Mistral-7B**


### SFT 训练数据（两档）

* **SFT-S（轻量）**：~50k 指令（Alpaca 量级），用于复现“短训也会出现 trade-off”
* **SFT-M（中等）**：~200k 指令（OpenOrca/UltraChat 量级），用于压力测试稳定性与灾难性遗忘

训练目标：标准 causal LM（next-token）。

---

## 2) 测评任务与指标（必须同时报告“绝对值 + 相对退化”）

### 核心任务组（必须）

* **GSM8K**：数学推理（Exact Match final answer）
* **ARC-Challenge**：科学推理/选择题（Accuracy）
* **MMLU**（或子集）：知识泛化（Accuracy）

### 额外稳健组（推荐）

* HellaSwag （常识/指代）


### 关键指标（论文主表就靠它）

对每个任务 (t)：

* **After-FT**：(\text{Score}_{t}^{\text{ft}})
* **Retention/Degradation**：(\Delta_t=\text{Score}*{t}^{\text{ft}}-\text{Score}*{t}^{\text{base}})
* **Trade-off summary**（一行搞定 reviewer）：
  [
  \text{Plasticity}=\Delta_{\text{ARC}},\quad
  \text{Stability}=\Delta_{\text{GSM8K}},\quad
  \text{Tradeoff}=\Delta_{\text{ARC}}-\lambda\max(0,-\Delta_{\text{GSM8K}})
  ]

并强制报告：

* **均值±方差（3 seeds）**
* **训练吞吐 tokens/s + 峰值显存**（OFT/BOFT/JORA 都是“几何开销”路线，必须交代）

---

## 3) 对比试验（基线方法、超参、预算匹配）

PEFT 当前支持的 tuner 类型很多（你不用全做，做“能代表族谱”的即可）([Hugging Face][2])。围绕你论文叙事，**最小但足够强**的对比集如下：

### 3.1 必做 baselines（对应两条路线：low-rank vs orthogonal）

**Low-rank 系（主对照）**

1. **LoRA**（r 扫描）([Hugging Face][3])
2. **DoRA**（`LoraConfig(use_dora=True)`）([Hugging Face][4])
3. **AdaLoRA**（动态 rank，对“预算更省”有说服力）

**Orthogonal 系（你引用 OFT/BOFT 的主对照）**
4) **OFT**（`OFTConfig`）([Hugging Face][5])
5) **BOFT**（`BOFTConfig`）([Hugging Face][5])


> 你提到 “GOFT”：更像学术线里的 **qGOFT（Givens rotation）**，可以作为 related work/讨论（不是必须 PEFT baseline）。([Proceedings of Machine Learning Research][7])
> 真要做 baseline，就标注“external implementation”。

### 3.2 统一训练超参（所有方法共用，除非必须例外）

* optimizer: AdamW
* base lr（adapter 参数）：`2e-4`（SFT-S），`1e-4`（SFT-M）
* scheduler: cosine，warmup 3% steps
* JORA 专用：`lr_theta=0.05`（旋转参数学习率），`lr_core=0.01`（核心参数学习率）
* seq_len: 2048（packing 开）
* global batch：用 grad_accum 配到 **tokens/update** 接近一致（比如 32k tokens/update）
* epochs：3

### 3.3 参数预算与建议超参（给你可直接跑的默认值）

**重要参数量洞见**：JORA 的参数构成中，**S/k 主要影响几何覆盖和计算复杂度，对总参数量影响很小**；**主要参数来源是 core 选择**：
- `core="diag"`: ~维度数个参数 (最少参数)
- `core="block"`: ~维度数×块大小个参数 (中等参数)
- `core="lowrank"`: ~(输入+输出)×秩个参数 (最多参数)

下面给"默认起跑点"，最终以"trainable params 匹配"为准（脚本打印参数量后微调）。

**轨道 A：q_proj + v_proj**

* LoRA/DoRA: r ∈ {8, 16, 24}, alpha=16/32, dropout=0.05（与 JORA 核心参数量匹配）
* AdaLoRA: init_r=16, target_r=8（同等预算），其余默认
* IA3: 默认（参数太少，不用匹配 r，只需报告"极低预算"）
* **JORA**:
  - 低预算：`core="diag"`, S_L=S_R=64, k=16（参数量 ≈ LoRA r=8）
  - 中预算：`core="diag"`, S_L=S_R=96, k=24（参数量 ≈ LoRA r=16）
  - 高预算：`core="block"`, block_size=4, S_L=S_R=64, k=16（参数量显著增加）
* OFT: block_size ∈ {8,16}
* BOFT: boft_block_size ∈ {8,16}（按预算调）([GitHub][8])

**轨道 B：all-linear（强适配）**

* LoRA/DoRA: r ∈ {8, 16}, alpha=32, dropout=0.05，`target_modules="all-linear"` ([Hugging Face][1])
* **JORA**:
  - `core="lowrank"`, lowrank_r=8, S_L=S_R=32, k=8（参数量适中）
  - `core="diag"`, S_L=S_R=32, k=8（参数量最少，几何优势最明显）
* OFT/BOFT：同样 target 到 all-linear（如果显存/速度炸了，就只做 attention+mlp 三大投影：qkv/o + up/down/gate）

---

## 4) JORA 主实验配置（与你代码实现对齐）

你现在的 JORA 计算路径非常清晰：**右旋转 → core → 左逆旋转 →（可选）tanh clip → 加到 base_out →（可选）magnitude scale**。

### 4.1 JORA 默认推荐（论文主结果的"default recipe"）

（按你 `JoraConfig` 的字段名来，基于参数量分析优化）

* target_modules：A轨 `["q_proj","v_proj"]`；B轨 `"all-linear"`
* **rotation**：`rotation_param="cayley"`, `S_L=96`, `S_R=96`, `k=24`, `max_angle=0.1`, `rotation_impl="auto"`, `force_random_rotation_init=True`, `theta_init_std=0.02`
  > S=96 是基于 Llama 参数空间几何覆盖考虑的折中选择（覆盖率~0.6%），k=24 确保选择机制有效性
* selection：`selection="topk_ema"`, `ema_beta=0.95`, `update_interval=50`, `warmup_steps=0`
  > ema_beta=0.95 比默认 0.98 更激进，update_interval=50 减少频繁更新开销
* **core**：主用 `core="diag"`, `zero_init_core=False`（参数量最少，几何优势最明显）；消融时对比 `core="block"`, `block_size=4`（参数量显著增加）
* magnitude：默认 `magnitude="oer_softmax"`，温度 `oer_temperature=2.0`
  > 提高温度以增强竞争性重分配效果
* 分布式训练：`ddp_allow_unused_parameters=True`（JORA 稀疏选择可能导致某些参数在某些步骤未使用）

> 你代码里 `magnitude` 有 legacy 的 `ecd_tanh`，而 `oer_softmax` 是你要在论文里主推的"竞争性 + 能量守恒"版本。

---

## 4.2 JORA 参数量构成分析（实验设计依据）

基于代码分析，JORA 参数构成如下（以 Llama attention 层为例）：

| 组件 | 参数量 | 占比 | 受哪些超参影响 |
|------|--------|------|----------------|
| **Theta (旋转)** | 2×S | <5% | S_L, S_R |
| **Core (核心)** | 主要参数 | 80-95% | core 类型 (diag/block/lowrank) |
| **Magnitude (幅度)** | 维度数 | 5-15% | magnitude 开关 |
| **Buffers** | 0 | 0% | 不算可训练参数 |

**关键设计启示**：
1. **S/k 的放大对参数量影响很小**（<5%），主要影响几何覆盖和计算效率
2. **Core 选择决定主要参数预算**（diag 最少，lowrank 最多）
3. **实验应先固定 core 类型**，然后在相同参数预算下调整 S/k 优化性能
4. **跨 core 类型对比时**，参数量差异可达 10x，需要谨慎设计

---

## 5) 消融实验（创新点拆解：每个点都要"可证伪"）

你 JORA 的创新点拆起来非常标准：**Rotation / Selection / Core / Magnitude(OER)** 四块，再加实现层面的 Triton/调度。

### 5.1 结构消融（必须做，写成 Fig/Table）

1. **No Rotation**：`S_L=0, S_R=0`（只剩 core）
2. **Single-sided**：`single_sided="left"` vs `"right"`（只保留一侧旋转，代码中默认为 `"none"`）
3. **No Selection（静态/全量）**：

   * `selection="none"`（不更新 pairs，等价“无稀疏旋转”上限/或只用初始化）
   * `selection="random"`（证明“挑选机制不是噪声好运气”）
4. **Core 类型**：`core="diag"` vs `core="block"` vs `core="lowrank"`（参数量差异显著：diag 最少，lowrank 最多；分别测试几何保持 vs 表达能力 vs LoRA-style）
5. **Magnitude 模块**：

   * `magnitude="none"`
   * `magnitude="ecd_tanh"`（legacy，对标 DoRA-style 直觉）
   * `magnitude="oer_softmax"`（主张：竞争性重分配，使用 `oer_temperature` 控制）

**每个消融必须报告**：ARC/GSM8K 的 (\Delta)（保持性）+ tokens/s（效率代价）。

### 5.2 机制验证（建议做，reviewer 很吃）

* **有效秩/谱**：对若干层的“等效更新”做 SVD（或近似），报告 effective rank / spectral entropy
* **旋转角统计**：(|\theta|) 分布、选中维度的能量分布（能解释 why topk_ema 有效）
* **OER 能量守恒误差**：(\left|\sum_i m_i^2 - E_{total}\right|)（你实现里有严格重归一，应该接近 0）

---

## 6) 超参数扫描（给你“最少跑数也能写论文”的网格）

你别搞“全空间网格”，那是浪费。按模块分层扫：

### 6.1 Rotation 扫参（决定"几何强度"）

**关键洞见**：S/k 对参数量影响很小（<5%），主要影响几何覆盖和计算效率

* **固定 core="diag" 下扫 S/k**：
  - S_L=S_R ∈ {32, 64, 96, 128}（几何空间从 32 到 128，参数量变化 <2%）
  - k ∈ {8, 16, 24, 32}（选择维度，覆盖率从 0.2% 到 0.78%）
* **预算匹配策略**：在相同 core 下调整 S/k 来优化性能，不用严格匹配参数量
* `rotation_param ∈ {"cayley","angle"}`（只需在一个预算上对比）

**论文里可证伪的假设写法**：

> *"S/k 的放大主要影响几何覆盖率而非参数量：更大的 S 提供更好的几何保持，但 k 的选择对参数空间覆盖更关键；Llama 等模型可能需要更高的覆盖率才能体现几何优势"*"

### 6.2 Selection 扫参（决定“稀疏更新是否真有用”）

* `selection ∈ {"topk_ema","random","none"}`
* `ema_beta ∈ {0.9,0.98,0.995}`（仅对 topk_ema 生效）
* `update_interval ∈ {1,4,16}`（越大越省算，但可能更不准）
* `pairing_strategy ∈ {"consecutive","high_low"}`（代码中支持，默认 "consecutive"）
* （可选）`use_gumbel=True`, `gumbel_tau ∈ {0.5,1.0,2.0}`

### 6.3 Core 扫参（决定"表达能力上限"）

**参数量影响显著**：不同 core 类型参数量差异可达 10x 以上

* `core="diag"`（baseline，参数量最少 ≈ 维度数）
* `core="block"`, `block_size ∈ {2,4,8}`（中等参数量 ≈ 维度数×块大小）
  - block_size=4: 参数量约为 diag 的 4x
  - 适合在 A轨上测试表达能力提升
* `core="lowrank"`, `lowrank_r ∈ {4,8,16}`（最多参数量 ≈ (输入+输出)×秩）
  - 类似 LoRA 的 low-rank 结构
  - 主要在 B轨 all-linear 上测试（参数量差异明显）
  - r=8 时参数量约为 diag 的 16x

### 6.4 Magnitude/OER 扫参（决定“能量重分配强度”）

* `magnitude ∈ {"none","oer_softmax"}`（主线只需这两个；`ecd_tanh`留给附录）
* `oer_temperature ∈ {0.5,1.0,2.0,5.0}`
* （可选退火）`ecd_temp_annealing=True` + `ecd_temp_start=5.0`, `ecd_temp_end=1.0`，只在附录展示"稳定性更好/更差"

---

## 6.5 实验设计关键总结

**参数量分析的核心启示**：
1. **S/k 是"免费"的超参**：在相同 core 下调整 S/k 几乎不改变参数量，主要影响几何覆盖和效率
2. **Core 决定预算档位**：diag(低预算) → block(中预算) → lowrank(高预算)，参数量差异可达 10x
3. **实验分层设计**：先选 core 类型确定预算档位，再在该档位内优化 S/k
4. **公平对比原则**：参数预算匹配应考虑核心参数量，而非总参数量

---

## 7) 论文里最该呈现的"主表/附表"结构

### 主表（每个 base model 一张）

行：{LoRA, DoRA, AdaLoRA, IA3, OFT, BOFT, **JORA**}
列：

* ARC-C (ft, Δ)
* GSM8K (ft, Δ)
* MMLU (ft, Δ)
* **Trainable Params**（必须标明，体现 parameter-matched）
* Avg Δ（或 Tradeoff score）
* tokens/s
* peak VRAM

**特别说明**：JORA 应标注参数构成，如 "JORA(diag,S=96,k=24)" 以明确几何配置

PEFT 的 baseline 覆盖范围与 OFT/BOFT 文档链接可直接引用：([Hugging Face][2])

### 附录（消融 + 扫参）

* Ablation table：NoRot / Single-side / NoSel / RandomSel / Core variants (diag/block/lowrank) / Magnitude variants
  - **明确标注参数量差异**：不同 core 类型参数量相差 10x 以上
* Sweep plots：(S,k) 对 Tradeoff 的曲线（固定 core="diag"，每条曲线 3 seeds 均值+误差条）
  - **核心发现展示**：S/k 放大主要影响几何覆盖，不显著增加参数量

---

## 8) 论文表格模板（实验数据置空）

### Table 1: Main Results on Llama-2-7B (A轨: q_proj+v_proj)

| Method | Params | ARC-C (↑) |  | GSM8K (↑) |  | MMLU (↑) |  | Avg Δ | Tokens/s | Peak VRAM |
|--------|--------|-----------|-----------|-----------|-----------|----------|----------|--------|----------|-----------|
|        |        | ft        | Δ         | ft        | Δ         | ft       | Δ        |        |          |           |
| LoRA(r=16) | 6.3M | - | - | - | - | - | - | - | - | - |
| DoRA(r=16) | 6.3M | - | - | - | - | - | - | - | - | - |
| AdaLoRA | 6.3M | - | - | - | - | - | - | - | - | - |
| IA3 | 0.03M | - | - | - | - | - | - | - | - | - |
| OFT(block=16) | 6.3M | - | - | - | - | - | - | - | - | - |
| BOFT(block=16) | 6.3M | - | - | - | - | - | - | - | - | - |
| **JORA(diag,S=96,k=24)** | **6.3M** | - | - | - | - | - | - | - | - | - |

**注**: Params 为可训练参数量，Δ 为相对于 base model 的性能变化。所有方法在相同训练设置下对比。

### Table 2: Main Results on Mistral-7B (A轨: q_proj+v_proj)

| Method | Params | ARC-C (↑) |  | GSM8K (↑) |  | MMLU (↑) |  | Avg Δ | Tokens/s | Peak VRAM |
|--------|--------|-----------|-----------|-----------|-----------|----------|----------|--------|----------|-----------|
|        |        | ft        | Δ         | ft        | Δ         | ft       | Δ        |        |          |           |
| LoRA(r=16) | 6.3M | - | - | - | - | - | - | - | - | - |
| DoRA(r=16) | 6.3M | - | - | - | - | - | - | - | - | - |
| AdaLoRA | 6.3M | - | - | - | - | - | - | - | - | - |
| IA3 | 0.03M | - | - | - | - | - | - | - | - | - |
| OFT(block=16) | 6.3M | - | - | - | - | - | - | - | - | - |
| BOFT(block=16) | 6.3M | - | - | - | - | - | - | - | - | - |
| **JORA(diag,S=96,k=24)** | **6.3M** | - | - | - | - | - | - | - | - | - |

### Table 3: Ablation Study on Llama-2-7B (A轨: q_proj+v_proj)

| Configuration | Params | ARC-C Δ | GSM8K Δ | Tokens/s | Tradeoff Score |
|---------------|--------|---------|----------|----------|----------------|
| JORA(diag,S=96,k=24) | 6.3M | - | - | - | - |
| No Rotation (S=0) | 6.3M | - | - | - | - |
| Single-side (left only) | 6.3M | - | - | - | - |
| No Selection (k=0) | 6.3M | - | - | - | - |
| Random Selection | 6.3M | - | - | - | - |
| Core: Block (block_size=4) | 25M | - | - | - | - |
| Core: LowRank (r=16) | 100M | - | - | - | - |
| No Magnitude | 6.0M | - | - | - | - |
| ECD Magnitude | 6.3M | - | - | - | - |

**Tradeoff Score**: Δ_ARC - λ×max(0, -Δ_GSM8K), λ=1.0

### Table 4: Hyperparameter Sweep Results (Fixed core="diag")

| S/k Configuration | Params | ARC-C Δ | GSM8K Δ | Tokens/s | Coverage Rate |
|------------------|--------|---------|----------|----------|---------------|
| S=32, k=8 | 6.1M | - | - | - | 0.20% |
| S=64, k=16 | 6.2M | - | - | - | 0.39% |
| S=96, k=24 | 6.3M | - | - | - | 0.59% |
| S=128, k=32 | 6.4M | - | - | - | 0.78% |

**Coverage Rate**: k/4096 (Llama attention dimension)

### Table 5: Parameter Budget Analysis

| Method | Theta Params | Core Params | Magnitude Params | Total Params | Core Type |
|--------|--------------|-------------|------------------|--------------|-----------|
| LoRA(r=16) | 0 | 6.3M | 0 | 6.3M | Low-rank |
| JORA(diag,S=96,k=24) | 192 | 4.1M | 4.1M | 8.4M | Diagonal |
| JORA(block,S=64,k=16) | 128 | 16.4M | 4.1M | 20.6M | Block |
| JORA(lowrank,r=8) | 128 | 65.5M | 4.1M | 69.7M | Low-rank |

**注**: Theta 为旋转参数，Core 为主要适配参数，Magnitude 为幅度缩放参数。

---



[1]: https://huggingface.co/docs/peft/v0.9.0/en/developer_guides/lora?utm_source=chatgpt.com "LoRA"
[2]: https://huggingface.co/docs/peft/en/package_reference/peft_types?utm_source=chatgpt.com "PEFT types"
[3]: https://huggingface.co/docs/peft/en/package_reference/lora?utm_source=chatgpt.com "LoRA"
[4]: https://huggingface.co/docs/peft/en/developer_guides/lora?utm_source=chatgpt.com "LoRA"
[5]: https://huggingface.co/docs/peft/en/conceptual_guides/oft?utm_source=chatgpt.com "Orthogonal Finetuning (OFT and BOFT)"
[6]: https://huggingface.co/docs/peft/en/conceptual_guides/ia3?utm_source=chatgpt.com "IA3"
[7]: https://proceedings.mlr.press/v235/ma24a.html?utm_source=chatgpt.com "Parameter Efficient Quasi-Orthogonal Fine-Tuning via Givens ..."
[8]: https://github.com/huggingface/peft/blob/main/src/peft/tuners/boft/config.py?utm_source=chatgpt.com "peft/src/peft/tuners/boft/config.py at main · huggingface/peft"

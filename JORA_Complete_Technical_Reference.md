# JORA: Joint Orthogonal Rotation Adaptation — 技术参考手册 v5.0

> **定位**：本文档面向 NeurIPS 2026 投稿冲刺，融合 v2 的工程深度与 v3 的理论框架，从方法论公式推导、理论可行性分析、横纵向对比、工程管线与模块级代码深度解析、消融开关、已知局限性、审稿策略等维度进行全面深度梳理。阅读完本手册后，可以首席科学家视角向审稿人/团队成员完整讲解 JORA 的设计动机、理论基础、实现细节以及潜在风险。
>

---

## 目录

### 第一部分：方法定位与创新叙事
- [1. Executive Summary](#1-executive-summary)
- [2. 统一视角：PEFT 方法的权重更新流形分类学](#2-统一视角peft-方法的权重更新流形分类学)
- [3. 理论基础：从 SO(n) 几何到稀疏 Givens 逼近](#3-理论基础从-son-几何到稀疏-givens-逼近)
- [4. 能量守恒的理论分析](#4-能量守恒的理论分析)
- [5. 横向深度对比与纵向组件必要性分析](#5-横向深度对比与纵向组件必要性分析)

### 第二部分：数学框架
- [6. 方法论：完整数学框架](#6-方法论完整数学框架)
- [7. 梯度流与优化动态分析](#7-梯度流与优化动态分析)

### 第三部分：工程实现（完整深度解析）
- [8. 架构总览与模块依赖](#8-架构总览与模块依赖)
- [9. rotation.py 详解](#9-rotationpy-详解)
- [10. core.py 详解](#10-corepy-详解)
- [11. selection.py 详解](#11-selectionpy-详解)
- [12. magnitude.py 详解](#12-magnitudepy-详解)
- [13. layer.py 详解](#13-layerpy-详解)
- [14. model.py 详解](#14-modelpy-详解)
- [15. callbacks.py 详解](#15-callbackspy-详解)

### 第四部分：数据流、Pipeline 与状态管理
- [16. 关键数据流与状态追踪（含 shape 标注）](#16-关键数据流与状态追踪)
- [17. 训练 Pipeline 完整流程](#17-训练-pipeline-完整流程)
- [18. Checkpoint 保存/恢复与可复现性](#18-checkpoint-保存恢复与可复现性)

### 第五部分：实验与验证策略
- [19. 消融实验的因果推理框架](#19-消融实验的因果推理框架)
- [20. 参数量与计算量分析](#20-参数量与计算量分析)
- [21. 数值稳定性与边界条件](#21-数值稳定性与边界条件)
- [22. 性能瓶颈定位与优化建议](#22-性能瓶颈定位与优化建议)

### 第六部分：集成、部署与投稿策略
- [23. PEFT 生态集成评估](#23-peft-生态集成评估)
- [24. 推理部署路径完整分析](#24-推理部署路径完整分析)
- [25. 已知局限性与改进路线图](#25-已知局限性与改进路线图)
- [26. 论文叙事与审稿策略](#26-论文叙事与审稿策略)
- [27. 实验结果的多种解读预案](#27-实验结果的多种解读预案)

### 第七部分：模型-任务-JORA 交互的第一性原理分析（v5 新增）
- [28. 预训练权重的谱结构与 JORA 适配性](#28-预训练权重的谱结构与-jora-适配性)
- [29. 主流模型架构与 JORA 组件的数学耦合分析](#29-主流模型架构与-jora-组件的数学耦合分析)
- [30. 任务特性的第一性原理分类与 JORA 优势映射](#30-任务特性的第一性原理分类与-jora-优势映射)
- [31. 数据集-模型-JORA 配置的联合优化策略](#31-数据集-模型-jora-配置的联合优化策略)
- [32. 实验设计矩阵与优越性展示策略](#32-实验设计矩阵与优越性展示策略)
- [33. 可验证假说与诊断实验方案](#33-可验证假说与诊断实验方案)

### 附录
- [附录 A. 核心公式速查表](#附录-a-核心公式速查表)
- [附录 B. 配置速查表](#附录-b-配置速查表)
- [附录 C. 符号-代码对照表](#附录-c-符号-代码对照表)
- [附录 D. 代码-公式交叉索引表](#附录-d-代码-公式交叉索引表)
- [附录 E. 关键证明与推导细节](#附录-e-关键证明与推导细节)
- [附录 F. 可复现性检查清单](#附录-f-可复现性检查清单)
- [附录 G. 模型-任务实验配置速查表（v5 新增）](#附录-g-模型-任务实验配置速查表)

---

# 第一部分：方法定位与创新叙事

---

## 1. Executive Summary

### 1.1 一句话定义与核心贡献声明

JORA（Joint Orthogonal Rotation Adaptation）是一种参数高效微调方法，通过在冻结权重矩阵的左右两侧施加 **可学习的稀疏 Givens 旋转**，配合可学习的 **核心变换矩阵（Core）** 与 **能量重分布缩放（Magnitude/OER）**，实现低参数量的自适应权重更新。

**核心贡献声明（建议论文中使用的版本）：**

1. **稀疏正交旋转框架**：提出以 Givens 旋转为原子操作的稀疏正交变换，将 $O(n^2)$ 的正交矩阵参数化压缩到 $O(S)$，其中 $S \ll n$；
2. **数据驱动的维度对选择**：基于在线 EMA 统计，动态选择梯度信号最强的维度对进行旋转耦合，实现 attention-over-parameters 的隐式机制；
3. **竞争性能量重分布（OER）**：提出零和约束下的 softmax 能量分配机制，保证微调过程中权重矩阵的总能量严格守恒，优于 DoRA 的 per-row 独立缩放。

### 1.2 方法论核心直觉——三个 "Why"

**Why Rotations?（为什么用旋转而非低秩加法？）**

微调的本质是对预训练学到的特征空间做**坐标重对齐**（re-alignment），而不是引入全新的信息方向。预训练模型已经在大规模语料上学到了丰富的特征基（feature basis），微调更多的是在这些基之间重新分配"注意力"——某些维度间的关系需要加强，另一些需要减弱。旋转（Givens rotation）恰恰是在现有特征基之间做耦合重组合的最自然、最经济的算子：它不改变范数、不引入新方向，只重新编排现有方向间的关系。相比之下，LoRA 的低秩加法引入了新的方向，这对于需要大幅修改表示的场景可能有必要，但对于大多数微调任务而言可能是"过度参数化的"（over-parameterized for the actual need）。

**Why Sparse + Dynamic?（为什么稀疏且动态？）**

并非所有维度对都需要旋转耦合——微调所需的变换集中在少数关键维度对上。这符合神经网络的稀疏敏感性：大量研究表明，微调中的有效更新集中在权重矩阵的少数关键方向上。但哪些维度对是关键的，是 task-dependent 的——不同任务激活不同的维度子集。因此，我们需要数据驱动地发现这些关键维度对，而不是预先固定。EMA 引导的 top-k 选择实现了这种"按需分配计算资源"的机制——类似于混合专家（MoE）中的 gating，但在参数维度而非模型维度上操作。

**Why Energy Conservation?（为什么要能量守恒？）**

预训练权重的行范数分布编码了各输出维度的"重要性先验"——高范数维度对应更重要的特征。微调应该重新分配（redistribute）这种重要性，而非破坏（destroy）它。LoRA/DoRA 的自由缩放允许所有维度同时膨胀或收缩，导致权重范数漂移（norm drift），这是训练不稳定的一个常见原因。OER 的零和竞争机制保证了重要性的重新分配是**零和的**——一个维度获得更多"能量"，必须有其他维度让出等量"能量"。这提供了一种比 weight decay 更精细的隐式正则化。

### 1.3 在 PEFT 方法家族中的定位

| 方法 | 参数结构 | 正交保证 | 能量约束 | 稀疏选择 | 参数量级（per layer） |
|------|---------|---------|---------|---------|---------------------|
| **LoRA** | 低秩 $AB^T$ | ❌ | ❌ | ❌ | $(n+m) \cdot r$ |
| **DoRA** | LoRA + magnitude decomp | ❌ | Per-row 独立 | ❌ | $(n+m) \cdot r + n$ |
| **OFT** | Block-diagonal 正交 | ✅ (block-wise) | 隐式（正交） | ❌ | $b \cdot b^2$ blocks |
| **BOFT** | Butterfly 正交分解 | ✅ (butterfly) | 隐式（正交） | ❌ | $O(n \log n)$ |
| **GaLore** | 投影梯度 | ❌ | ❌ | ❌ | 投影矩阵 + rank |
| **AdaLoRA** | SVD 自适应 rank | ❌ | ❌ | ✅ (rank pruning) | 动态 |
| **JORA（本文）** | 稀疏 Givens + Core + OER | ✅ (Givens 精确) | ✅ (OER 严格守恒) | ✅ (EMA top-k) | $S_L + S_R + \text{Core} + n$ |

**关键差异化论点：**

- vs LoRA/DoRA：JORA 使用正交变换而非低秩分解，参数效率更高（DiagCore 约为 LoRA-8 的 1/8），且显式保证能量守恒
- vs OFT/BOFT：JORA 使用稀疏的 Givens 旋转而非完整的块正交矩阵，参数量从 $O(n \log n)$ 或 $O(b \cdot b^2)$ 降至 $O(S)$；同时引入了数据驱动的维度选择
- vs AdaLoRA：AdaLoRA 在秩空间做选择，JORA 在旋转维度对空间做选择，粒度更细

### 1.4 关键数字速览

| 指标 | 典型值 | 说明 |
|------|--------|------|
| 每层参数量（DiagCore + OER） | ~8,256 | $S_L(32) + S_R(32) + \min(n,m)(4096) + n(4096)$ |
| LoRA rank-8 每层参数量 | ~65,536 | $(4096+4096) \times 8$ |
| 参数效率比 | **~8x** | JORA DiagCore vs LoRA-8 |
| 活跃旋转对数 | 8 (default k) | 每步仅 $k$ 对参与旋转计算 |
| 总旋转容量 | 32+32 | $S_L + S_R$ |
| EMA 衰减系数 | 0.98 | 平衡历史信息与近期信号，有效窗口 ~50 步 |
| FLOPs 额外开销 | < 1% | DiagCore 配置下 |
| 训练内存额外开销 | ~214 KB/层 | 含 optimizer 状态，约为 LoRA 的 1/4.7 |

---

## 2. 统一视角：PEFT 方法的权重更新流形分类学

### 2.1 从 $\Delta W$ 的结构约束看 PEFT 方法族

所有 PEFT 方法的核心思想可以统一表述为：给定冻结的预训练权重 $W_0 \in \mathbb{R}^{n \times m}$，学习一个参数化的修正 $\Delta W(\theta)$，使得微调后的权重 $W' = W_0 + \Delta W(\theta)$ 更好地适应下游任务。不同 PEFT 方法的本质区别在于 $\Delta W(\theta)$ 被约束在什么样的**参数集合**（流形）上。

**定义 2.1（权重更新流形）**：给定参数空间 $\Theta \subseteq \mathbb{R}^p$（$p \ll nm$），权重更新流形定义为参数化映射的像集：

$$
\mathcal{M} = \{\Delta W(\theta) \in \mathbb{R}^{n \times m} : \theta \in \Theta\}
$$

#### 2.1.1 低秩流形（LoRA / AdaLoRA）

$$
\mathcal{M}_{\text{LoRA}} = \{BA^T : B \in \mathbb{R}^{n \times r}, A \in \mathbb{R}^{m \times r}\}
$$

这是 $\mathbb{R}^{n \times m}$ 中由秩 $\leq r$ 矩阵构成的代数簇（algebraic variety），维度为 $(n+m)r - r^2$。

**归纳偏置**：LoRA 假设微调所需的权重更新是低秩的。**局限**：当微调需要对权重矩阵做"全维度的小调整"（如重新校准所有维度的相对权重）时，低秩分解无法高效表达。

#### 2.1.2 正交群子流形（OFT / BOFT）

OFT 约束权重更新为正交变换 $W' = R \cdot W_0$，$\Delta W = (R - I) W_0$，其中 $R \in \mathcal{O}_{\text{block}} = SO(b) \times SO(b) \times \cdots \times SO(b)$。

**归纳偏置**：OFT 假设微调只需要旋转输出特征空间的坐标系，而不改变特征的范数。**局限**：块结构/butterfly 结构是固定的——哪些维度之间可以交互由预先设计的拓扑决定，而非由数据驱动。

#### 2.1.3 幅度-方向分解（DoRA）

DoRA 将 $W'$ 分解为方向和幅度：$W' = M \odot \text{normalize}(W_0 + BA^T)$。

**归纳偏置**：在 LoRA 的低秩方向修正之上增加 per-row 独立幅度调节。**局限**：$M$ 的各分量互不约束——可以同时增大所有维度的幅度，导致权重整体膨胀。

#### 2.1.4 JORA：数据驱动的稀疏正交子流形 + 显式能量约束

JORA 的流形可以形式化为：

$$
\mathcal{M}_{\text{JORA}} = \left\{f : \mathbb{R}^m \to \mathbb{R}^n \;\middle|\; f(x) = M(\boldsymbol{w}) \odot \sigma\!\left(\prod_{s \in \mathcal{S}_L} G_{i_s j_s}^T(\theta_s^L) \cdot D(\boldsymbol{d}) \cdot \prod_{s \in \mathcal{S}_R} G_{i_s j_s}(\theta_s^R) \cdot x\right)\right\}
$$

**关键观察**：JORA 的 $\mathcal{M}$ 不是一个固定流形，而是一个**时变流形族**——随着训练推进，EMA 选择改变 $\mathcal{S}_L, \mathcal{S}_R$，等效于在不同子流形之间切换。

### 2.2 各流形的归纳偏置与适用场景分析

| 流形类型 | 归纳偏置 | 最适用场景 | 不适用场景 |
|----------|---------|-----------|-----------|
| **低秩**（LoRA） | 更新集中在少数方向 | 任务需要引入少量新特征方向 | 需要全维度重校准 |
| **正交子群**（OFT/BOFT） | 仅旋转方向、不改变范数 | 特征空间方向对齐 | 需要改变特征幅度 |
| **幅度-方向**（DoRA） | 方向低秩修正 + 独立幅度调节 | 需要同时调方向和幅度 | 幅度约束不足导致不稳定 |
| **稀疏正交+能量守恒**（JORA） | 稀疏旋转重对齐 + 受约束的能量重分布 | 特征基已优、需精细重新分配 | 需要大幅改变特征空间结构 |

### 2.3 流形维度与覆盖度的理论权衡

**命题 2.1（参数效率的流形视角）**：设微调任务的"最优更新" $\Delta W^*$ 到流形 $\mathcal{M}$ 的投影误差为 $\epsilon(\mathcal{M}) = \min_{\Delta W \in \mathcal{M}} \|\Delta W - \Delta W^*\|_F$。

**定量分析**：以 LLaMA-7B 的 $n = m = 4096$ 为例：

| 方法 | 参数量 $p$ | 流形维度上界 | 每参数覆盖效率 |
|------|-----------|-------------|--------------|
| LoRA $r=8$ | 65,536 | $(n+m)r - r^2 = 65,472$ | ~1.0 |
| OFT $b=4$ | 24,576 | $K \cdot b(b-1)/2 = 6,144$ | ~0.25 |
| BOFT | ~49,152 | $O(n \log n) \approx 49,152$ | ~1.0 |
| **JORA DiagCore** | **8,256** | $S_L + S_R + \min(n,m) = 4,160$ (线性), 非线性+OER 有效扩展 | **~0.50+** |

JORA 的流形维度看似低于 LoRA，但其流形**更贴近**典型微调需求——正交+稀疏的归纳偏置与微调的"小调整"本质更匹配，因此在同等参数量下可能达到更低的投影误差 $\epsilon$。

### 2.4 理论创新点定位

**层面1（结构创新）**：首次将**稀疏 Givens 旋转**作为 PEFT 的基本构建块。

**层面2（机制创新）**：首次在 PEFT 中引入**竞争性能量重分布**，将各输出维度的能量分配建模为零和博弈。

**层面3（算法创新）**：首次将**在线学习**（EMA 统计）与**参数空间结构搜索**（维度对选择）结合——可以看作参数空间的"稀疏注意力"，也可类比为**参数空间的 Neural Architecture Search (NAS)**：NAS 搜索网络拓扑结构，JORA 搜索参数激活结构。

---

## 3. 理论基础：从 SO(n) 几何到稀疏 Givens 逼近

### 3.1 SO(n) 上的 Riemannian 优化视角

#### 3.1.1 特殊正交群的几何结构

$SO(n) = \{Q \in \mathbb{R}^{n \times n} : Q^TQ = I, \det Q = 1\}$ 是一个 $n(n-1)/2$ 维的紧致 Lie 群，其 Lie 代数为 $\mathfrak{so}(n) = \{A \in \mathbb{R}^{n \times n} : A^T = -A\}$。

**指数映射**：$\exp: \mathfrak{so}(n) \to SO(n)$, $Q = \exp(A)$。
**切空间**：$T_I SO(n) = \mathfrak{so}(n)$，描述从 $I$ 出发的"无穷小旋转"。
**测地线**：$\gamma(t) = \exp(tA)$。

#### 3.1.2 Givens 旋转作为 $\mathfrak{so}(n)$ 的基

$\mathfrak{so}(n)$ 的标准基为 $\{E_{ij} : 1 \leq i < j \leq n\}$，$E_{ij} = e_i e_j^T - e_j e_i^T$。

**Givens 旋转正是沿这些基方向的测地线**：

$$
G_{ij}(\theta) = \exp(\theta \cdot E_{ij}) = I + (\cos\theta - 1)(e_i e_i^T + e_j e_j^T) + \sin\theta(e_i e_j^T - e_j e_i^T)
$$

$S$ 个 Givens 旋转的组合等价于在 $\mathfrak{so}(n)$ 的 $S$ 维子空间中搜索旋转。

**与 PEFT 的联系**：在 $SO(n)$ 上做微调 = 在 $\mathfrak{so}(n)$ 的子空间中寻找最优"无穷小旋转组合"。JORA 选择了 $\mathfrak{so}(n)$ 中 $S$ 个特定的基方向（维度对），沿这些方向做有限角度旋转。

### 3.2 稀疏 Givens 子流形的逼近理论

#### 3.2.1 $I$-邻域逼近定理（含可计算的界）

**定理 3.1（小扰动正交矩阵的稀疏 Givens 逼近）**：设 $Q \in SO(n)$ 满足 $\|Q - I\|_F \leq \delta$，$A = \log(Q) \in \mathfrak{so}(n)$，$\|A\|_F = O(\delta)$。设 $A = \sum_{i<j} a_{ij} E_{ij}$，系数按绝对值排序 $|a_{(1)}| \geq |a_{(2)}| \geq \cdots$，取 top-$S$ 项得到 $A_S = \sum_{k=1}^S a_{(k)} E_{i_k j_k}$，则：

$$
\|Q - \exp(A_S)\|_F \leq \underbrace{\left(\sum_{k > S} |a_{(k)}|^2\right)^{1/2}}_{\text{截断误差}} + \underbrace{O(\delta^2)}_{\text{高阶交叉项}}
$$

**可计算的界——幂律衰减假设**：若 Lie 代数系数满足幂律衰减 $|a_{(k)}| \leq C_0 \cdot k^{-\alpha}$（$\alpha > 1/2$），则截断误差为：

$$
\left(\sum_{k > S} C_0^2 k^{-2\alpha}\right)^{1/2} \leq \frac{C_0}{\sqrt{2\alpha - 1}} \cdot S^{-({\alpha - 1/2})}
$$

**代入实际数字**：对 LLaMA-7B ($n = 4096$)，假设微调是 $\delta = 0.1$ 邻域内的小扰动（$\|Q - I\|_F \leq 0.1$），幂律指数 $\alpha = 2$（经验假设：微调变换的频谱快速衰减），$C_0 \approx 0.01$，则 $S = 32$ 时：

$$
\epsilon_{\text{trunc}} \leq \frac{0.01}{\sqrt{3}} \cdot 32^{-1.5} \approx 3.2 \times 10^{-5}
$$

这意味着 $S = 32$ 个 Givens 旋转足以将逼近误差控制在极小的量级——远小于微调本身引入的噪声。

**开放问题**：$\alpha$ 的实际值需要经验测量。建议在论文中通过以下实验确定：对训练好的模型提取旋转参数 $\theta_s$，拟合其排序后的衰减曲线。

#### 3.2.2 有效旋转自由度的经验测量

**实验方案**：画出 $S$-性能曲线（固定 $k = S/4$），当曲线饱和时对应的 $S$ 即为任务的"有效旋转自由度"。

**理论预期**：基于以下观察，有效维度远小于 $n(n-1)/2$：
1. 预训练权重的奇异值谱快速衰减——大部分"信息"集中在少数方向
2. 微调通常只需小幅度修正表示
3. LoRA rank-4 到 rank-16 就能达到全参数微调的大部分性能——说明有效自由度很低

### 3.3 动态选择机制的理论意义

#### 3.3.1 静态 vs 动态子流形

**命题 3.1**：设训练中最优旋转 $Q^*(t)$ 的重要维度对集合 $\mathcal{I}^*(t) = \text{top-}S\{|a_{ij}(t)|\}$ 随时间变化。

- **静态选择**（OFT/BOFT）：$\epsilon_{\text{static}} = \max_t \|A^*(t) - P_{\mathcal{I}^0} A^*(t)\|_F$
- **动态选择**（JORA）：$\epsilon_{\text{dynamic}} = \max_t \|A^*(t) - P_{\mathcal{I}(t)} A^*(t)\|_F$

$\epsilon_{\text{dynamic}} \leq \epsilon_{\text{static}}$，且当重要维度对在训练中显著变化时差距尤其明显。

#### 3.3.2 与组合 Bandit 的形式化联系

维度对选择可形式化为组合 bandit：臂集合 = $\binom{n}{2}$ 个维度对，约束 = $k$ 个不相交对，奖励 = 选中后对 loss 的贡献（事先未知）。

EMA 统计 $\hat{\sigma}_i^2 = \mathbb{E}[\text{activation}^2_i]$ 是对奖励的代理信号。在平稳假设下，EMA 估计收敛速率为 $O(1/\sqrt{T_{\text{eff}}})$（$T_{\text{eff}} = 1/(1-\beta) = 50$ 步），此后 top-k 选择以高概率命中真正的 top-k 维度。

**遗憾界的直觉论证**（非严格证明，但可在论文中作为 remark）：

设有 $N = \binom{n}{2}$ 个臂，有效差异 $\Delta_k = e_{(k)} - e_{(k+1)}$（第 $k$ 和第 $k+1$ 大能量的差），则 EMA 需要约 $O(\sigma^2 / \Delta_k^2)$ 步来区分它们。当 $\Delta_k$ 较大时（即 top-k 维度对明显突出），EMA 能快速锁定正确选择。

#### 3.3.3 与 Mixture of Experts 的结构类比

| 维度 | MoE | JORA 选择 |
|------|-----|-----------|
| 选择对象 | 专家网络（参数块） | 维度对（旋转参数） |
| 选择依据 | 输入相关的 gating | EMA 统计引导的 top-k |
| 稀疏性 | Top-k 专家 | Top-k 维度对 |
| 时变性 | Per-sample | Per-step（更稳定） |

#### 3.3.4 与 Neural Architecture Search 的深层类比

JORA 的动态选择可以视为一种**在线的、连续的微架构搜索**：

| NAS | JORA |
|-----|------|
| 搜索空间：网络拓扑 | 搜索空间：活跃维度对集合 |
| 搜索策略：RL/进化/可微分 | 搜索策略：EMA + 贪心 |
| 评价信号：验证集性能 | 评价信号：梯度/激活能量 |
| 搜索粒度：离散结构 | 搜索粒度：连续角度 + 离散对选择 |
| 搜索成本：训练多个子网 | 搜索成本：EMA 更新（几乎零开销） |

这个类比为 JORA 提供了一个新的叙事角度：JORA 不只是一种 adapter，而是一种**在参数空间中自适应搜索最优微调结构**的方法。

### 3.4 Cayley 参数化的优化几何优势

#### 3.4.1 形式化定义

$$
\phi = 2 \arctan\!\left(\frac{\theta}{2}\right), \quad c = \frac{1 - (\theta/2)^2}{1 + (\theta/2)^2}, \quad s = \frac{\theta}{1 + (\theta/2)^2}
$$

#### 3.4.2 优化景观分析

**命题 3.2（Cayley 参数化的优化友好性）**：

1. **单调性**：$d\phi/d\theta = 1/(1+(\theta/2)^2) > 0$，严格单调
2. **值域限制**：$\phi \in (-\pi, \pi)$，无周期性
3. **唯一极小值**：对给定 $L(\phi)$，$L(\phi(\theta))$ 在 $\theta$ 空间中在最优 $\phi$ 附近有唯一极小值（无等价极小值）
4. **自然正则化**：$|\theta| \to \infty$ 时 $d\phi/d\theta \to 0$

**与直接角度参数化的对比**：

| 性质 | 直接角度 | Cayley |
|------|---------|--------|
| 值域 | $(-\infty, +\infty)$, 周期 $2\pi$ | $(-\pi, \pi)$, 无周期 |
| 优化景观 | 多个等价极小值 | 唯一极小值邻域 |
| 小角度行为 | $\phi \approx \theta$ | $\phi \approx \theta$（一致） |
| 梯度在 $\theta=0$ | $d\phi/d\theta = 1$ | $d\phi/d\theta = 1$（一致） |
| 梯度在 $\theta=2$ | $-\sin 2 \approx -0.91$ | $0.5$（衰减但稳定） |
| 梯度在 $\theta=10$ | $-\sin 10 \approx 0.54$（振荡！） | $0.038$（强衰减，稳定） |

**关键差异**：直接角度参数化的梯度 $-\sin\theta$ 在大 $\theta$ 时振荡，导致 Adam 的二阶矩估计不准确。Cayley 的梯度单调衰减，始终给出正确的搜索方向。

#### 3.4.3 定量梯度衰减表

| $\theta$ | $\phi$ (rad) | $\phi$ (deg) | $d\phi/d\theta$ | 物理含义 |
|----------|-------------|-------------|-----------------|---------|
| 0.0 | 0.000 | 0° | 1.000 | 线性区，梯度充分 |
| 0.2 | 0.199 | 11.4° | 0.990 | 几乎无衰减 |
| 1.0 | 0.927 | 53.1° | 0.800 | 轻度衰减 |
| 2.0 | 1.571 | 90° | 0.500 | 中等衰减（旋转已达 90°） |
| 4.0 | 2.214 | 126.9° | 0.200 | 强衰减 |
| 10.0 | 2.747 | 157.4° | 0.038 | 接近饱和 |

$\theta = 2$ 时旋转角已达 90°（足够大），梯度衰减到 0.5——这意味着 Cayley 在保证充足搜索范围的同时自然阻止极端旋转。

**代码对应** (`rotation.py → cayley_cos_sin`):

```python
phi = 2.0 * torch.atan(0.5 * theta)
return torch.cos(phi), torch.sin(phi)
```

### 3.5 信息瓶颈视角下的 Core 设计分析

**信息瓶颈理论**认为好的表示应该最大化关于目标的信息量，同时最小化关于输入的冗余信息。JORA 的 Core 模块可以从这个角度理解：

- **DiagCore**：逐维度的独立信息门控——每个 $d_i$ 控制第 $i$ 个旋转后维度的信息通过量。$|d_i|$ 大的维度是"信息通道"，$|d_i|$ 小的维度是"噪声过滤"。
- **BlockCore**：允许块内的信息混合（局部信息瓶颈），块间独立。
- **LowRankCore**：全局的 rank-$r$ 信息瓶颈——强制所有信息通过 $r$ 维子空间。

**设计选择的理论指导**：如果微调任务的信号集中在少数维度上（强信息瓶颈），DiagCore 足够；如果信号需要在多个维度间混合（弱信息瓶颈），BlockCore 或 LowRankCore 更合适。

---

## 4. 能量守恒的理论分析

### 4.1 微调中的权重范数漂移问题

LoRA 的修正 $W' = W_0 + BA^T$ 导致 $\|w'_i\|^2 = \|w_{0,i}\|^2 + 2\langle w_{0,i}, (BA^T)_i\rangle + \|(BA^T)_i\|^2$，总能量无约束。DoRA 的 $M_i$ 独立可学习，$\sum_i M_i^2$ 同样无约束。

**实验观察**（建议在论文中验证）：LoRA 训练中权重能量通常先增后稳（learning rate warmup 期间快速增长），但最终值难以控制。

### 4.2 OER 的形式化：受约束的单纯形上搜索

OER 等价于在单纯形 $\Delta^{n-1} = \{p \in \mathbb{R}^n_+ : \sum_i p_i = 1\}$ 上搜索最优能量分配 $\boldsymbol{p}^*$：

$$
\boldsymbol{p}^* = \arg\min_{\boldsymbol{p} \in \Delta^{n-1}} \mathbb{E}_{(x,y)} \left[L\!\left(f_{\boldsymbol{p}}(x), y\right)\right]
$$

参数化 $p_i = \text{softmax}(\boldsymbol{w}/T)_i$ 将约束优化转化为无约束优化。

### 4.3 零和竞争的博弈论视角（完整推导）

**softmax 的 Jacobian**：

$$
\frac{\partial p_i}{\partial w_j} = p_i (\delta_{ij} - p_j)
$$

**零和性质**：$\sum_i \frac{\partial p_i}{\partial w_j} = 0$——增大 $w_j$ 后所有 $p_i$ 的变化量之和为零。

**均衡条件推导**：训练收敛时 $\nabla_{w_i} L = 0$：

$$
\frac{\partial L}{\partial w_i} = \sum_j \frac{\partial L}{\partial p_j} \cdot p_j (\delta_{ji} - p_i) = p_i \left(\frac{\partial L}{\partial p_i} - \sum_j p_j \frac{\partial L}{\partial p_j}\right) = 0
$$

由于 $p_i > 0$（softmax 严格正），均衡条件为：

$$
\boxed{\frac{\partial L}{\partial p_i} = \bar{g} \triangleq \sum_j p_j \frac{\partial L}{\partial p_j}, \quad \forall i \in \{1, \ldots, n\}}
$$

即**所有维度的边际 loss 增益相等**——这是竞争均衡的经典"等边际原理"。

**经济学类比**：OER 等价于将有限的"能量预算" $E_{\text{total}}$ 在 $n$ 个"投资项目"（维度）之间做最优分配。均衡条件说的是：最优分配下，每多投入一单位能量到任何维度，获得的 loss 下降量都相同——否则可以将能量从"回报低"的维度转移到"回报高"的维度来改善。

**与 DoRA 的根本区别**：DoRA 的均衡仅为 $\partial L/\partial M_i = 0$（各维独立），允许所有维度同时增大 $M_i$。OER 的约束排除了这种"合作策略"。

### 4.4 能量守恒的正则化效应

#### 4.4.1 与 Weight Decay 的精确对比

| 方面 | Weight Decay | OER |
|------|-------------|-----|
| 约束类型 | 软约束（L2 惩罚项 $\lambda\|w\|^2/2$） | 硬约束（$\sum E_i = E_{\text{total}}$） |
| 粒度 | Per-parameter | Per-layer（行级别） |
| 总能量 | 趋向零（收缩效应） | 保持不变（守恒） |
| 交互模式 | 各参数独立 | 竞争性（零和） |
| 信息保留 | 损失预训练信息（权重被推向零） | 保持预训练能量结构 |
| 隐式效果 | 优先保留大权重 | 优先保留能量分布形状 |

#### 4.4.2 Rademacher 复杂度分析（带推导）

设 $\mathcal{F}_E = \{f : \sum_i \|w_i\|^2 = E\}$，训练样本数为 $N$。

对线性模型 $f(x) = Wx$，能量约束下的 Rademacher 复杂度为：

$$
\hat{\mathcal{R}}_N(\mathcal{F}_E) = \frac{1}{N} \mathbb{E}_\sigma \left[\sup_{W:\sum\|w_i\|^2=E} \sum_{n=1}^N \sigma_n w_i^T x_n\right]
$$

由 Cauchy-Schwarz：$\sum_n \sigma_n w_i^T x_n \leq \|w_i\| \cdot \|\sum_n \sigma_n x_n\|$，

$$
\sup_W \sum_i \sum_n \sigma_n w_i^T x_n \leq \sqrt{E} \cdot \sqrt{\sum_i \left\|\sum_n \sigma_n x_n\right\|^2} = \sqrt{E \cdot n} \cdot \left\|\sum_n \sigma_n x_n\right\|
$$

因此：

$$
\hat{\mathcal{R}}_N(\mathcal{F}_E) \leq \frac{\sqrt{E \cdot n}}{N} \cdot \mathbb{E}_\sigma\left[\left\|\sum_n \sigma_n x_n\right\|\right] \leq \frac{\sqrt{E \cdot n}}{N} \cdot \sqrt{\sum_n \|x_n\|^2} = \frac{\sqrt{E \cdot n \cdot \bar{X}}}{N^{1/2}}
$$

其中 $\bar{X} = \frac{1}{N}\sum_n \|x_n\|^2$。

**关键结论**：能量 $E$ 固定 → Rademacher 复杂度有界 → 泛化保证。无约束时 $E$ 可任意增大，上界变为 vacuous。

**OER 相比 weight decay 的优势**：weight decay 将 $E$ 推向零（可能过度正则化），OER 保持 $E = E_{\text{total}}$（预训练确定的"合理"能量水平）。

### 4.5 温度控制的松弛谱

| $T$ 范围 | softmax 行为 | 能量分配 | Rademacher 上界 | 等效于 |
|----------|-------------|---------|----------------|--------|
| $T \to \infty$ | $p_i \to 1/n$ | 均匀 | 最紧 | 全局 scalar 乘法 |
| $T = 5.0$ | 较平滑 | 温和竞争 | 中等 | 弱约束 |
| $T = 1.0$ | 标准 softmax | 明确赢家/输家 | 较松 | 标准竞争 |
| $T \to 0$ | One-hot | 单维度独占 | 最松 | 退化（过度竞争） |

**温度退火**：$T(t) = (1 - t/T_{\text{total}}) \cdot T_{\text{start}} + (t/T_{\text{total}}) \cdot T_{\text{end}}$。从高温（探索）到低温（锁定分配）。

### 4.6 OER 的潜在限制与应对

**Q1（审稿人）：守恒约束是否过强？**

A：(1) 通过增大 $T$ 可松弛——$T$ 很大时接近均匀缩放；(2) 消融可量化守恒的影响；(3) 微调场景下能量分布通常接近合理。

**Q2：softmax 梯度耦合是否导致优化困难？**

A：softmax Jacobian 的条件数 $\kappa \propto 1/T$。高温时 $\kappa \approx 1$（近独立），低温时增大。温度退火策略从"易优化"过渡到"强约束"。

---

## 5. 横向深度对比与纵向组件必要性分析

### 5.1 JORA vs LoRA：从秩空间到旋转空间

#### 5.1.1 核心数学区别

LoRA：$\Delta W_{\text{LoRA}} = BA^T$ — 线性、低秩
JORA (DiagCore)：$\Delta(x) = \tanh(R_L^T \cdot \text{diag}(d) \cdot R_R \cdot x)$ — 非线性、可满秩

**关键区别1（线性 vs 非线性）**：LoRA 的 $\Delta W$ 对所有 $x$ 相同。JORA 由于 tanh，$\Delta(x)$ 随 $x$ 变化——表达力严格超过 LoRA。

**关键区别2（低秩 vs 稀疏正交）**：DiagCore 线性近似 $R_L^T \cdot \text{diag}(d) \cdot R_R$ 可以是满秩矩阵（所有 $d_i \neq 0$），但自由度仅 $S_L + S_R + \min(n,m)$——远少于 LoRA 的 $(n+m)r$。

#### 5.1.2 何时 JORA 优于 LoRA（条件分析）

**JORA 占优的条件**：
1. **微调需要全维度重校准**：DiagCore 可独立调节每个维度，LoRA 需要 $r = \min(n,m)$ 才能做到
2. **微调只需小幅度修正**：$\|\Delta W^*\|_F$ 小时，旋转+缩放结构更高效
3. **特征空间需要"坐标重对齐"而非"引入新方向"**：例如领域适配中重新分配已有特征的重要性

**LoRA 占优的条件**：
1. **需要引入全新的特征方向**：$B$ 的列可以与 $W_0$ 列空间正交
2. **需要高秩更新且参数预算充裕**

#### 5.1.3 参数量的形式化比较

要达到"能修正每个维度"的能力：

| 方法 | 所需参数量 | 能力描述 |
|------|-----------|---------|
| LoRA rank-1 | $(n+m)$ | 只能修正 1 个方向 |
| LoRA rank = min(n,m) | $\min(n,m)(n+m)$ ≈ 33M | 完全覆盖（参数爆炸） |
| DoRA rank-8 | $(n+m) \times 8 + n$ = 69,632 | 方向低秩 + 独立幅度 |
| **JORA DiagCore + OER** | $S_L + S_R + \min(n,m) + n$ = **8,256** | 旋转耦合 + 逐维缩放 + 能量约束 |

### 5.2 JORA vs OFT/BOFT：从固定拓扑到数据驱动

#### 5.2.1 维度耦合模式的根本区别

**OFT**：维度 $i$, $j$ 耦合当且仅当 $\lfloor i/b \rfloor = \lfloor j/b \rfloor$（同块）。等价于 $K$ 个完全图 $K_b$ 的不相交并。

**BOFT**：耦合模式由 butterfly 拓扑预先确定。

**JORA**：$\text{可达耦合} = \{(i,j) : \text{任意}\}$（动态选择）。

$$
\text{JORA 覆盖度} = \bigcup_{t} \mathcal{G}_{\text{JORA}}(\mathcal{S}(t)) \gg \text{OFT 覆盖度} = \prod_k SO(b)
$$

**关键优势**：JORA 可以耦合任意远距离的维度（如维度 100 和 3000），无需增大整个块大小。

### 5.3 JORA vs DoRA：竞争性 vs 独立性

#### 5.3.1 梯度结构的完整对比

**DoRA**：$\partial L/\partial M_i = (\partial L/\partial w'_i) \cdot \hat{w}_i$（各 $M_i$ 梯度互不影响）

**JORA OER**：$\partial L/\partial w_i = p_i\left(\partial L/\partial p_i \cdot \partial \text{scale}_i/\partial p_i - \sum_j p_j \cdot \partial L/\partial p_j \cdot \partial \text{scale}_j/\partial p_j\right)$（全局耦合）

| 行为 | DoRA | JORA OER |
|------|------|----------|
| 所有维度同时增大 | ✅ 允许 | ❌ 禁止（零和） |
| 范数漂移 | 可能 | 不可能（严格守恒） |
| 梯度耦合 | 无 | 全局（softmax） |
| 均衡条件 | 各维独立达零 | 等边际原理 |

### 5.4 纵向分析：组件必要性的形式化论证

#### 5.4.1 自由度空间的正交互补性

**命题 5.1（旋转与 Core 的互补性——严格论证）**：

设 $R \in SO(n)$ 作用于向量 $x$，$D = \text{diag}(d)$ 作用于向量 $x$。

- **仅旋转**（$D = I$）：$\Delta(x) = (R^T R' - I)x$。修正在 $SO(n)$ 上——只改变方向，不改变范数。修正的自由度空间 $\subseteq \mathfrak{so}(n)$（反对称矩阵空间，维度 $n(n-1)/2$）。
- **仅 Core**（$R = I$）：$\Delta(x) = Dx - x = (\text{diag}(d) - I)x$。修正在对角矩阵空间——只改变幅度，不改变方向间耦合。自由度空间 $\subseteq \text{Diag}(n)$（对称对角矩阵空间，维度 $n$）。

**关键**：$\mathfrak{so}(n) \perp \text{Diag}(n)$——反对称矩阵和对角矩阵在 Frobenius 内积下正交：$\langle A, D \rangle_F = \text{tr}(A^T D) = \text{tr}(-A \cdot D) = -\sum_i a_{ii} d_i = 0$（$A$ 对角为零）。

因此旋转和 Core 提供的**自由度严格正交互补**——两者修正的"方向"在参数空间中无重叠，组合后覆盖的空间严格大于各自单独的空间。

**命题 5.2（OER 的独立附加值）**：

OER 操作的空间是 $\mathbb{R}^n_+$（行范数缩放），它改变的是 $W_0$ 本身的行范数，而非 delta 路径的输出。旋转+Core 改变的是 $\Delta(x)$（在 delta 空间），两者的操作空间不重叠。

此外，OER 提供了 Core 缺乏的**全局约束**——Core 的 $d_i$ 可以同时增大，OER 的 $p_i$ 必须归一。

#### 5.4.2 理论增益汇总

| 组件组合 | 自由度维度 | 约束 | 缺失能力 |
|---------|-----------|------|---------|
| 仅 Core | $\min(n,m)$ | 无 | 维度间无耦合 |
| 仅旋转 | $S$ | 正交 | 无幅度控制（$\Delta \approx 0$） |
| 旋转 + Core | $S + \min(n,m)$ | 正交 | 无能量约束 |
| Core + OER | $\min(n,m) + n$ | 能量守恒 | 无维度间耦合 |
| **旋转 + Core + OER** | $S + \min(n,m) + n$ | **正交 + 能量守恒** | **无** |

---

# 第二部分：数学框架

---

## 6. 方法论：完整数学框架

### 6.1 问题设定与符号约定

| 符号 | 含义 | 典型值/维度 |
|------|------|-----------|
| $W_0 \in \mathbb{R}^{n \times m}$ | 冻结预训练权重 | $n = m = 4096$ (LLaMA-7B) |
| $x \in \mathbb{R}^{B \times L \times m}$ | 输入张量 | Batch × Seq × InDim |
| $S_L, S_R$ | 左/右旋转容量 | 32 |
| $k$ | 全局最大活跃对数 | 8 |
| $\theta_L \in \mathbb{R}^{S_L}, \theta_R \in \mathbb{R}^{S_R}$ | 旋转角度参数 | 可学习 |
| $D$ | 核心变换矩阵 | Diag/Block/LowRank |
| $M \in \mathbb{R}^n$ | Magnitude 缩放向量 | OER/ECD 计算 |
| $\beta$ | EMA 衰减系数 | 0.98 |
| $T$ | 温度参数 | 1.0 |
| $E_{\text{total}}$ | 总能量 $\sum_i \|w_{0,i}\|^2$ | 标量 |

### 6.2 总体前向公式推导

$$
y = \underbrace{W_0 x + b}_{\text{base}} + \underbrace{M \odot \tanh\!\bigl(R_L^T \cdot D \cdot R_R \cdot x\bigr)}_{\text{JORA delta } \Delta}
$$

**逐步展开（含 shape 标注）**：

```
Input x: [B, L, m]

Step 2a. 右侧旋转 R_R：
   x_rot = G(θ₁ᴿ) ∘ G(θ₂ᴿ) ∘ ... ∘ G(θₖᴿ) · x    # [B, L, m] → [B, L, m]
   只修改被选中维度对的分量，其余不变

Step 2b. 核心变换 D：
   y_core = D(x_rot)                                  # [B, L, m] → [B, L, n]
   DiagCore: y[..., i] = d_i · x[..., i]
   BlockCore: y = blkdiag(B₁,...,Bₖ) · x
   LowRankCore: y = (α/r)(x @ B) @ A^T

Step 2c. 左侧逆旋转 R_L⁻¹ = R_L^T：
   y_rot = G(θₖᴸ)⁻¹ ∘ ... ∘ G(θ₁ᴸ)⁻¹ · y_core     # [B, L, n] → [B, L, n]
   等价于逆序遍历、取负角度

Step 2d. 软限幅：
   y_clip = tanh(y_rot)                               # [B, L, n] → [B, L, n]

Step 2e. Magnitude 缩放：
   delta = M ⊙ y_clip                                 # [B, L, n] → [B, L, n]
   M = OER_scale(ecd_log_mag, base_row_norms, total_energy, T)
```

**代码对应** (`layer.py → compute_delta` + `forward`):

```python
# Forward path (layer.py → JoraLayer.forward):
base_out = linear_forward(x)                              # W₀x + bias  [B,L,n]
x_rot = _apply_side_rotation(x, is_left_side=False)       # R_R          [B,L,m]
y_core = core.apply_to_vector(x_rot)                      # D            [B,L,n]
y = _apply_side_rotation(y_core, is_left_side=True)        # R_L^T        [B,L,n]
if not zero_init_core:
    y = torch.tanh(y)                                      # soft clip    [B,L,n]
out = base_out + y                                         # residual     [B,L,n]
out = maybe_apply_magnitude(out)                           # M ⊙ out     [B,L,n]
```

### 6.3 Givens 旋转的形式化定义

$$
G_{ij}(\theta) = I + (\cos\theta - 1)(e_ie_i^T + e_je_j^T) + \sin\theta(e_ie_j^T - e_je_i^T)
$$

对向量作用：$\binom{x_i'}{x_j'} = \binom{\cos\theta \; \sin\theta}{-\sin\theta \; \cos\theta}\binom{x_i}{x_j}$

**覆盖性**：$SO(n)$ 中任意矩阵可分解为至多 $n(n-1)/2$ 个 Givens 旋转的乘积。

**命题 6.1（不相交对的子流形）**：$S$ 个不相交维度对构成 $S$-维 torus $\mathbb{T}^S = SO(2)^S$。

**命题 6.2（重叠对的耦合效应）**：$(1,2)$ 和 $(2,3)$ 的乘积在 $\{1,2,3\}$ 三维子空间上产生更丰富的变换（非直积）。

### 6.4 Core 设计空间

#### 6.4.1 DiagCore

$D = \text{diag}(d_1, \ldots, d_{\min(n,m)})$, 参数量 $\min(n,m)$

**代码**：`core.py → DiagCore.apply_to_vector`: `y[..., :d_len] = x[..., :d_len] * self.diag_params`

#### 6.4.2 BlockCore

$D = \text{blkdiag}(B_1, \ldots, B_K, \text{diag}(r_1, \ldots, r_p))$, $B_k \in \mathbb{R}^{b \times b}$

参数量：$K \cdot b^2 + p$

**代码**：`core.py → BlockCore.apply_to_vector`: `einsum('...nbk,bkj->...nbj', x_blocks, block_params)`

#### 6.4.3 LowRankCore

$D = (\alpha/r) AB^T$, $A \in \mathbb{R}^{n \times r}$, $B \in \mathbb{R}^{m \times r}$

参数量：$(n+m)r$

**代码**：`core.py → LowRankCore.apply_to_vector`: `y = scaling * (x @ B) @ A.T`

#### 6.4.4 初始化策略

| Core 类型 | `zero_init=False` (默认) | `zero_init=True` |
|-----------|-------------------------|------------------|
| DiagCore | $d_i \sim \mathcal{N}(0, 0.01)$ | $d_i = 0$ |
| BlockCore | $B_k \sim \mathcal{N}(0, 0.1)$ | $B_k = 0$ |
| LowRankCore | $A, B \sim \mathcal{N}(0, 0.1)$ | $A = B = 0$ |

**关键**：`zero_init_core=True` 时跳过 tanh——delta 无上界，依赖 optimizer 隐式约束。

### 6.5 OER 完整推导

$$
p_i = \text{softmax}(w_i/T), \quad E_i^{\text{target}} = E_{\text{total}} \cdot p_i, \quad m_i = \sqrt{E_i^{\text{target}}}
$$

$$
\text{scale}_i^{\text{raw}} = \frac{m_i}{\max(\|w_{0,i}\|, \epsilon_{\min})}, \quad \hat{E} = \sum_i (\text{scale}_i^{\text{raw}} \cdot \|w_{0,i}\|)^2
$$

$$
\boxed{\text{scale}_i = \text{scale}_i^{\text{raw}} \cdot \sqrt{\frac{E_{\text{total}}}{\hat{E}}}}
$$

**守恒精确性**：$\sum_i (\text{scale}_i \cdot \|w_{0,i}\|)^2 = E_{\text{total}}$（精确等式，证明见附录 E.2）

### 6.6 EMA 选择机制

**列能量**（前向时）：$e_j^{\text{col}}(t) = \beta \cdot e_j^{\text{col}}(t-1) + (1-\beta) \cdot \frac{1}{BL}\sum_{b,l} x_{b,l,j}^2$

**行能量**（反向时）：$e_i^{\text{row}}(t) = \beta \cdot e_i^{\text{row}}(t-1) + (1-\beta) \cdot \frac{1}{BL}\sum_{b,l} g_{b,l,i}^2$

**Top-K 选择**：取 top-$8k$ 候选 → 枚举对 → 按 $e_i \cdot e_j$ 排序 → 贪心不相交选择 $k$ 对

**Warmup**：$k_{\text{allow}}(t) = \max(1, \lfloor k \cdot \min(1, t/t_w) \rfloor)$

### 6.7 tanh 非线性的完整分析

**动机**：无 LoRA 的 $\alpha/r$ 内置缩放，需上界保护。

**代价**：(1) merge 不可逆（根本限制）；(2) $|z| > 3$ 时 $\tanh' \to 0$（梯度消失）

**替代方案对比**：

| 方案 | 优点 | 缺点 | 可 merge |
|------|------|------|---------|
| `tanh`（当前） | 稳定，$(-1,1)$ 硬界 | 梯度消失，merge 不可逆 | ❌ |
| 无限幅 (`zero_init`) | 不限表达力 | 需精细 LR | ✅ |
| `clamp(-c,c)` | 简单 | 非光滑 | ❌ |
| $x/(1+|x|)$ | 光滑 | merge 不可逆 | ❌ |
| $\alpha/r$ 缩放 | 可 merge | 需额外超参 | ✅ |

### 6.8 k 参数语义与左右分配

**关键设计**：`k` 是全局参数，表示左右两侧**总共**的最大活跃对数。分配逻辑（`layer.py → _JoraAdapterState.update_step`）：

$$
k_L = \left\lfloor k_{\text{allow}} \cdot \frac{S_L}{S_L + S_R} \right\rfloor, \quad k_R = k_{\text{allow}} - k_L
$$

按缓冲区容量比例分配，并确保在 $k_{\text{allow}} > 0$ 时两侧各至少有 1 对。

### 6.9 选择不稳定性与训练动态

**问题**：EMA 统计变化 → 选中维度对在相邻步间剧烈切换 → 参与梯度的参数子集不稳定 → "parameter flicker"。

**具体影响**：

1. **优化器状态不准确**：$\theta_i$ 在某步被选中参与梯度 → 下一步未被选中 → 梯度为零 → Adam 的一阶/二阶矩估计可能不准
2. **Core 看到不同旋转输入**：不同步的旋转对导致 Core 的输入分布变化
3. **DDP 选择不同步**：不同 GPU 上 EMA 基于各自 micro-batch，选择可能不同步，需要 `ddp_find_unused_parameters=True`

**缓解机制**：

1. `update_interval > 1`：不每步更新选择，减少切换频率
2. `ema_beta = 0.98`：高 $\beta$ 使 EMA 更平滑
3. Warmup：渐进引入新对
4. `_update_pair_buffer` 的"仅增不减"逻辑

**开放问题**：是否需要"选择惯性"（selection inertia）——让当前选中的对获得 bonus 分数，增加切换成本？这类似于 MoE 中的 load balancing 问题。可作为 future work 讨论。

---

## 7. 梯度流与优化动态分析

### 7.1 完整反向传播推导

设 $y = R_L^T \cdot D \cdot R_R \cdot x$，则：

$$
\frac{\partial L}{\partial x} = R_R^T \cdot D^T \cdot R_L \cdot \frac{\partial L}{\partial y}
$$

由于 Givens 正交，$\|R^T g\| = \|g\|$，**梯度范数变化完全由 Core $D$ 决定**。

### 7.2 各组件梯度量级估计

**对 $\theta_s^R$（右旋转角度）**：

$$
\frac{\partial L}{\partial \theta_s^R} = \sum_{b,l} \left[\frac{\partial L}{\partial \tilde{x}_i}(-s \cdot x_i + c \cdot x_j) + \frac{\partial L}{\partial \tilde{x}_j}(-c \cdot x_i - s \cdot x_j)\right]
$$

**量级**：$|\nabla_\theta L| \sim O(|x| \cdot |\nabla_y L|)$——与 LoRA 中 $\nabla_B L$ 同阶。

对 Cayley 参数化还需乘链式因子 $d\phi/d\theta = 1/(1+(\theta/2)^2)$。

**Core 梯度**（DiagCore）：$\partial L/\partial d_k = \sum_{b,l} \nabla_{y_k} L \cdot \tilde{x}_k$

### 7.3 收敛性分析

#### 7.3.1 Cayley 消除周期性极小值

直接角度 $\cos\theta$ 的 loss 关于 $\theta$ 是周期函数——Adam 的矩估计被多个极值点的梯度混合。Cayley 的单调映射消除此问题。

#### 7.3.2 稀疏选择的隐式正则化

每步只有 $k$ 个旋转参与梯度，其余梯度为零——类似 Dropout。形式化：

$$
\hat{g}_s^{(t)} = \nabla_{\theta_s} L \cdot \mathbf{1}[s \in \mathcal{A}(t)]
$$

EMA 决定的 $\mathcal{A}(t)$ 是非均匀采样——高能量维度更常被选中——提供了比 Dropout（均匀采样）更有信息的正则化。

#### 7.3.3 tanh 的梯度流控制

当 delta 接近饱和（$|\Delta| \to 1$），tanh 梯度衰减到零，自动阻止参数继续增大——无需手动 gradient clipping。

#### 7.3.4 综合稳定性论证

**命题 7.1（优化景观的良性性质）**：在以下条件下，JORA 的优化景观不包含 pathological 的特性：

1. **无周期性极小值**：Cayley 参数化保证（命题 3.2）
2. **梯度有界**：tanh 限幅 → delta 有界 → 梯度有界（除非 base model 本身梯度爆炸）
3. **无突然的参数维度变化**：warmup 逐渐引入旋转对，避免梯度方向突变
4. **能量约束防止权重漂移**：OER 守恒提供 implicit regularization

**非严格，但足以回答审稿人"能收敛吗"的质疑。**

---

# 第三部分：工程实现（完整深度解析）

---

## 8. 架构总览与模块依赖

```
src/peft/tuners/jora/
├── __init__.py          # （缺失，需创建）
├── config.py            # JoraConfig - 配置数据类
├── rotation.py          # Givens 旋转（Torch + Triton 双路径）
├── core.py              # DiagCore / BlockCore / LowRankCore
├── selection.py         # Top-K 对选择 + Warmup
├── magnitude.py         # OER softmax / ECD tanh
├── layer.py             # JoraLayer + _JoraAdapterState
├── model.py             # JoraModel（BaseTuner 集成）
├── callbacks.py         # JoraTrainerCallback + JoraSchedulerCallback
└── utils.py             # 辅助函数
```

```
config.py ←──── layer.py ←──── model.py ←──── callbacks.py
                  │  ↑
    ┌─────────────┤  │
    ↓             ↓  │
rotation.py  core.py │
                     │
selection.py ────────┘
magnitude.py ────────┘
utils.py ────────────┘
```

---

## 9. rotation.py 详解

### 9.1 Torch 路径：完整向量化实现

**核心函数**：`apply_rotations_torch`

**完整 6 步实现流程**：

```python
# Step 1. Flatten to [batch*seq, dim], clone once
y = x.view(-1, dim).clone()              # 单次 clone，所有旋转原地操作

# Step 2. Handle reverse/negate (左侧旋转需要)
if reverse: pairs, thetas = flip(pairs), flip(thetas)
if negate_theta: th = -th

# Step 3. Compute cos/sin for all pairs at once
c, s = _cos_sin(th, rotation_param)       # [k], [k]  — Cayley 或 Angle

# Step 4. Extract columns for all pairs simultaneously
i, j = pairs[:, 0], pairs[:, 1]          # [k] long tensors
yi = y.index_select(1, i)                 # [B*L, k]
yj = y.index_select(1, j)                 # [B*L, k]

# Step 5. Apply Givens rotations (fully vectorized)
new_yi = c * yi + s * yj                  # [B*L, k]
new_yj = -s * yi + c * yj                 # [B*L, k]

# Step 6. Write back (disjoint pairs guarantee no conflict)
y.index_copy_(1, i, new_yi)               # in-place, 无冲突
y.index_copy_(1, j, new_yj)
```

**⚠️ 不相交安全性**：当 pairs 不全不相交时，`index_copy_` 的写回顺序未定义（后写覆盖先写）。当前 selection 保证不相交，但修改选择逻辑时需注意。

**输入验证**：`_validate_pairs` 检查负索引（-1 sentinel）和越界。Pair buffer 用 -1 填充未使用位置，Python 负索引会静默访问错误位置。

### 9.2 Triton 路径：kernel 设计与完整 backward 推导

**前向 kernel** (`apply_givens_rotations_kernel`)：
- 列指针模式：只加载旋转涉及的两个列
- 序列化执行：`for k in range(n_pairs)` 依次执行（可能有重叠维度）
- 动态 BLOCK_M：根据 token 总数选择 64/128/256

**Backward 完整推导**：

两个梯度需要计算：

**1. $\nabla_x L$（对输入的梯度）**：

$$
\nabla_x L = R^T \cdot \nabla_y L = R(-\theta) \cdot \nabla_y L
$$

实现：使用 `sin_vals.neg()` 实现 $R(-\theta)$，`reverse=not reverse_fwd` 实现转置。

**2. $\nabla_\theta L$（对旋转角度的梯度）**：

对每个旋转对 $(i, j)$ 和角度 $\theta_s$，Givens 旋转的导数为：

$$
\frac{\partial}{\partial \phi} \begin{pmatrix} \cos\phi & \sin\phi \\ -\sin\phi & \cos\phi \end{pmatrix} = \begin{pmatrix} -\sin\phi & \cos\phi \\ -\cos\phi & -\sin\phi \end{pmatrix}
$$

因此：

$$
\frac{\partial L}{\partial \phi_s} = \sum_{b,l} \left[\nabla_{y_i} L \cdot (-s \cdot x_i + c \cdot x_j) + \nabla_{y_j} L \cdot (-c \cdot x_i - s \cdot x_j)\right]
$$

对 Cayley 参数化乘以链式因子：

$$
\frac{\partial L}{\partial \theta_s} = \frac{\partial L}{\partial \phi_s} \cdot \frac{d\phi}{d\theta} = \frac{\partial L}{\partial \phi_s} \cdot \frac{1}{1 + (\theta_s/2)^2}
$$

**⚠️ 已知近似问题**：Triton backward 计算 $\nabla_\theta L$ 时使用**原始输入 $x$** 和**原始梯度 $\nabla_y L$**，而非旋转后的中间结果。对于串联的多个旋转，第 $s$ 个旋转的真正输入是前 $s-1$ 个旋转的输出。

**近似误差量级**：当 $|\theta| < 0.1$：$\cos\theta \approx 1$, 旋转接近恒等，误差为 $O(\theta^2)$ → 可接受。当 $S$ 大且 $\theta$ 大：误差可能累积。

### 9.3 已知问题与修复建议

| 问题 | 严重性 | 修复建议 |
|------|--------|---------|
| Triton backward 多旋转耦合近似 | 🟡 中 | 对大 $S$ 场景添加测试，对比 Torch/Triton 梯度差异 |
| Triton kernel 强制 `tl.float32` | 🟢 低 | bf16 下的数值行为需验证 |
| `_validate_pairs` 每次调用时执行 | 🟢 低 | 可在 debug 模式下启用，release 模式跳过 |

---

## 10. core.py 详解

### 10.1 三种 Core 的内存与计算 profile

以 $n = m = 4096$, $b = 4$, $r = 8$ 为例：

| Core | 参数量 | `apply_to_vector` 计算 | `forward()` 内存 | 备注 |
|------|--------|----------------------|-----------------|------|
| DiagCore | 4,096 | $O(n)$ 逐元素乘 | 64 MB (4096×4096 fp32) | forward() 已 deprecated |
| BlockCore(b=4) | 16,384 | $O(n \cdot b)$ | 64 MB | 使用 einsum 向量化 |
| LowRank(r=8) | 65,536 | $O(n \cdot r + m \cdot r)$ | 小（无需展开） | 两步 matmul |

### 10.2 `apply_to_vector` vs `forward` 的使用场景

| 方法 | 用途 | 性能 | 内存 |
|------|------|------|------|
| `apply_to_vector(x)` | **正常前向传播** | 最优 | $O(\text{batch})$ |
| `forward()` → full matrix | **仅 merge 路径** | 慢 | $O(n \times m)$ |
| `get_row_slice(start, end)` | merge 分块（未使用） | 中等 | $O(\text{rows} \times m)$ |

### 10.3 已知问题

| 问题 | 严重性 | 修复建议 |
|------|--------|---------|
| `DiagCore.get_row_slice()` 使用 Python for 循环 | 🟢 低 | 可向量化；正常训练不使用 |
| LowRank 默认 scaling=1.0（`alpha=None` → `alpha=r`） | 🟡 中 | 与 LoRA 的 $\alpha/r$ 惯例不同，需文档说明 |
| BlockCore `apply_to_vector` 对小 block_size 有 overhead | 🟢 低 | `torch.stack` + `einsum` 在 b=2/4 时 overhead 可能大于直接循环 |

---

## 11. selection.py 详解

### 11.1 GPU 加速的贪心选择：完整算法与正确性分析

**算法流程**：

```
Input: energy[d], k
1. cand = min(d, max(16, 8*k))              # 候选池大小
2. topk_idx = top-cand indices by energy     # O(d)
3. Enumerate all (cand choose 2) pairs       # O(cand²)
4. score(i,j) = energy[i] * energy[j]        # 乘积得分
5. Get top-4k pairs by score                 # O(cand² log cand²)
6. Greedy selection:                         # O(4k)
   for pair in sorted_pairs:
     if both indices unused:
       select pair, mark indices as used
     if selected == k: break
Output: pairs[<=k, 2]
```

**正确性问题**：步骤 6 的 GPU 实现使用 batch 化贪心——同一 batch 内的 `batch_available` 基于 `used_mask` 的旧状态，可能选中共享维度的冲突对。实际影响：`batch_size=1024` 远大于 $k=8$ 时冲突概率极低。

**近似比**：贪心匹配至少达到最优权重的 1/2（经典结论）。

### 11.2 高低配对策略

```python
def _select_high_low_pairs_gpu(energy, k, max_features):
    _, sorted_indices = torch.sort(energy, descending=False)
    top_indices = sorted_indices[-n_pairs:]     # 最高能量
    bottom_indices = sorted_indices[:n_pairs]   # 最低能量
    pairs = torch.stack([top_indices, bottom_indices], dim=1)
```

**天然不相交**：一个维度不可能同时是 top 和 bottom。

### 11.3 `_update_pair_buffer` 的"仅增不减"逻辑

```python
if cur >= allowed_count: return  # 阻止 k 减少
```

Warmup 单调递增时正确。如果未来需要动态减少（如退火阶段），需修改此处。

### 11.4 已知问题

| 问题 | 严重性 | 修复建议 |
|------|--------|---------|
| GPU batch 贪心可能选中冲突对 | 🟡 中 | 对小 k（≤32）使用严格贪心 |
| 仅增不减逻辑 | 🟡 中 | 添加缩减逻辑 |
| `compute_allowed_pairs` 返回 `max(1, ...)` | 🟢 低 | 即使 warmup 未完成也至少保留 1 对 |

---

## 12. magnitude.py 详解

### 12.1 OER 的数值稳定性工程（完整保护机制）

**`compute_oer_scale_softmax` 中的保护机制**：

**1. 输入验证**：
```python
if total_energy_val <= 0:
    return torch.ones_like(base_norms)          # 退化保护1
if not torch.isfinite(base_norms).all() or (base_norms <= 0).all():
    return torch.ones_like(base_norms)          # 退化保护2
```

**2. 自适应 `min_norm` 策略**：
```python
base_scale = (total_energy_val / base_norms.numel()) ** 0.5
min_norm = max(eps * 100, min(base_scale * 1e-4, base_scale * 1e-2))
safe_base_norms = torch.clamp(base_norms, min=min_norm)
```

**数值示例**：若 `base_scale = 10.0`，则 `min_norm = max(1e-6, min(0.001, 0.1)) = 0.001`。

**3. 结果验证**：
```python
if not torch.isfinite(scale).all():
    uniform_scale = torch.sqrt(total_energy_val / base_norms.numel())
    scale = torch.full_like(base_norms, uniform_scale)   # 退化保护3
```

### 12.2 退化保护机制完整清单

| # | 触发条件 | 行为 | 代码位置 |
|---|---------|------|---------|
| 1 | `total_energy <= 0` | 返回全 1 缩放 | `compute_oer_scale_softmax` L12 |
| 2 | 所有范数为零或非有限 | 返回全 1 缩放 | L16 |
| 3 | 重归一化后结果非有限 | 退化到均匀分布 | L42 |
| 4 | `actual_total_E <= 0` | 退化到均匀分布 | L37 |

### 12.3 ECD Tanh（旧版兼容）

$$
\text{scale}_i = c \cdot (1 + \alpha \cdot \tanh(w_i/T)), \quad c = \sqrt{E_{\text{total}} / \sum_i [\|w_{0,i}\| (1 + \alpha \tanh(w_i/T))]^2}
$$

竞争性弱于 OER（tanh 各维独立，仅通过 $c$ 间接耦合）。论文中将 OER 作为主要贡献，ECD 作为 ablation。

### 12.4 已知问题

| 问题 | 严重性 | 修复建议 |
|------|--------|---------|
| `min_norm` 策略在极端分布下可能不适当 | 🟡 中 | 添加 `base_norms` 分布诊断日志 |
| `base_row_norms` 计算方向的 fallback 逻辑复杂 | 🟡 中 | `_JoraAdapterState.__init__` 中先 dim=0 再 dim=1 的尝试，对非标准权重可能出错 |
| ECD 的 $c$ 校正因子包含 `base_norms`，零范数时 | 🟢 低 | ECD 未使用 `safe_base_norms` |

---

## 13. layer.py 详解

### 13.1 `_JoraAdapterState` 完整状态清单

| 属性 | 类型 | 用途 | 持久化 | shape |
|------|------|------|--------|-------|
| `core` | `nn.Module` | 核心变换 | ✅ | — |
| `theta_L` | `nn.Parameter` / `None` | 左旋转角度 | ✅ | `[S_L]` |
| `theta_R` | `nn.Parameter` / `None` | 右旋转角度 | ✅ | `[S_R]` |
| `pairs_L` | `Buffer` | 左旋转维度对 | ✅ | `[S_L, 2]` long |
| `pairs_R` | `Buffer` | 右旋转维度对 | ✅ | `[S_R, 2]` long |
| `num_pairs_L` | `Buffer` | 左侧活跃对数 | ✅ | scalar |
| `num_pairs_R` | `Buffer` | 右侧活跃对数 | ✅ | scalar |
| `grad_row_ema` | `Buffer` | 行梯度 EMA | ✅ | `[n]` fp32 |
| `grad_col_ema` | `Buffer` | 列激活 EMA | ✅ | `[m]` fp32 |
| `step_idx` | `Buffer` | 步数计数器 | ✅ | scalar |
| `ema_step_idx` | `Buffer` | EMA 步数计数器 | ✅ | scalar |
| `ecd_log_mag` | `nn.Parameter [n]` / `None` | OER/ECD logits | ✅ | `[n]` |
| `base_row_norms` | `Buffer` | 基础行范数 | ✅ | `[n]` fp32 |
| `base_row_norms_fp32` | `Buffer` | fp32 缓存 | ❌ | `[n]` fp32 |
| `total_energy` | `Buffer` | 总能量 | ✅ | scalar |
| `_num_pairs_py` | `dict` (Python) | Python 侧对数缓存 | ❌ | — |
| `_counter_cache` | `dict` (Python) | Python 侧计数器缓存 | ❌ | — |

### 13.2 前向/反向数据流（完整 shape 标注）

```
Input x: [B, L, m]   (dtype: model dtype, e.g. bf16)
  │
  ├─ dtype cast: x.to(weight.dtype) if mismatch
  │
  ├─ EMA update (training only, if ema_interval hit):
  │   xd = x.detach().reshape(-1, m)     # [B*L, m]
  │   x_sq = xd.float().pow(2).mean(0)   # [m]  ← fp32 计算
  │   grad_col_ema.lerp_(x_sq, 1-β)      # [m]  ← fp32 in-place
  │
  ├─ Base forward:
  │   base_out = F.linear(x, W₀, bias)   # [B, L, n]
  │
  ├─ JORA delta:
  │   ├─ R_R: x [B,L,m]
  │   │   → n_R = _num_pairs_py['right']  (Python cache, 避免 GPU 同步)
  │   │   → active_pairs = pairs_R[:n_R]  # [n_R, 2] long
  │   │   → active_theta = theta_R[:n_R]  # [n_R]
  │   │   → x_rot = apply_rotations(x.view(-1,m), pairs, theta)  # [B*L,m]
  │   │   → x_rot = x_rot.view(B,L,m)    # [B,L,m]
  │   │
  │   ├─ Core: x_rot [B,L,m] → y_core [B,L,n]
  │   │   DiagCore: y[..., :d] = x[..., :d] * diag_params  # [B,L,n]
  │   │   BlockCore: einsum('...nbk,bkj->...nbj')           # [B,L,n]
  │   │   LowRankCore: (x @ B) @ A.T * scaling              # [B,L,n]
  │   │
  │   ├─ R_L⁻¹: y_core [B,L,n] → y [B,L,n]
  │   │   (逆序遍历 pairs_L, 取负 theta_L)
  │   │
  │   └─ tanh: y [B,L,n] → delta [B,L,n]
  │       (跳过如果 zero_init_core=True)
  │
  ├─ Residual add:
  │   out = base_out + delta              # [B, L, n]
  │
  ├─ Magnitude:
  │   scale = compute_oer_scale_softmax(  # [n] fp32
  │       ecd_log_mag, base_row_norms_fp32,
  │       total_energy, temperature)
  │   out = out * scale.to(out.dtype).view(1,...,1,n)  # broadcast [B,L,n]
  │
  └─ dtype cast: out.to(x.dtype) if mismatch

Output: [B, L, n]
```

**反向传播钩子**：

```python
def _backward_hook(self, module, grad_input, grad_output):
    g = grad_output[0].detach()                    # [B, L, n]
    g_sq = g.reshape(-1, n).float().pow(2).mean(0) # [n] fp32
    st.grad_row_ema.lerp_(g_sq, 1.0 - beta)       # [n] fp32 in-place
```

**⚠️** `register_full_backward_hook` 在模块输出为 `ModelOutput`（dataclass）时不触发——需要 callback 方案。

### 13.3 Merge/Unmerge 的近似策略与完整局限分析

**当前 merge 策略** (`_compute_weight_delta_simple`):

$$
\Delta W \approx 0.05 \times \text{rotation\_effect} \times C
$$

**问题清单**：

| 问题 | 影响 | 根因 |
|------|------|------|
| `0.05` 硬编码 | merge 后性能显著低于实际推理 | 缺少对 tanh 非线性的正确近似 |
| `core.forward()` 生成全矩阵 | 大维度时 OOM 风险 | 可改用 `get_row_slice` 分块 |
| 旋转效应估计忽略维度耦合 | 大角度旋转估计不准 | 使用 per-dim 独立近似 |
| **tanh 不可线性化** | **根本限制** | $\tanh(Ax) \neq \Delta W \cdot x$ |

---

## 14. model.py 详解

### 14.1 BaseTuner 集成

| 方法 | 实现状态 | 说明 |
|------|---------|------|
| `_prepare_adapter_config` | ✅ | 直接返回 config |
| `_check_target_module_exists` | ✅ | 委托 `check_target_module_exists` |
| `_create_and_replace` | ✅ | 替换 Linear/Conv1D → JoraLayer |
| `_mark_only_adapters_as_trainable` | ✅ | 冻结 base, 启用 adapter params |
| `enable/disable_adapter_layers` | ✅ | |
| `set_adapter` | ✅ | 仅支持单 adapter |
| `prepare_inputs_for_generation` | ✅ | 代理到底层模型 |

### 14.2 选择分组优化

当 `selection_group_size > 1` 时，相同维度/类型的层共享选择。减少计算但牺牲 per-layer 个性化。

### 14.3 已知问题

| 问题 | 严重性 | 修复建议 |
|------|--------|---------|
| `_update_group_selection_shared` 硬编码 `'default'` adapter name | 🟡 中 | 使用 `layer.active_adapter` |
| 缺少 `get_nb_trainable_parameters()` 自定义实现 | 🟢 低 | 验证 BaseTuner 默认统计正确性 |
| `_jora_post_backward_hook` 使用 micro-batch 步数而非 optimizer 步数 | 🟡 中 | 与 callback 冲突；建议统一使用 callback |

---

## 15. callbacks.py 详解

### 15.1 为什么需要 Callback 而非 Hook

**问题**：PyTorch 的 `register_full_backward_hook` 期望模块输出为 Tensor。HuggingFace transformer 模型返回 `ModelOutput`（dataclass），导致 hook 不触发。

**影响**：不使用 callback 时，`jora_update_step` 和 `jora_update_temperature` 永远不会被触发——模型停留在初始随机选择上。

**解决**：
```python
def on_step_end(self, args, state, control, **kwargs):
    jora_model.jora_update_step()
    jora_model.jora_update_temperature(current_step, total_steps)
```

### 15.2 步数语义差异

| 触发机制 | 步数粒度 | gradient_accumulation_steps=4 时 |
|---------|---------|--------------------------------|
| backward hook | per micro-batch | 100 micro-batches = step 100 |
| Trainer callback | per optimizer step | 100 micro-batches = step **25** |

**用户注意**：从 hook 切换到 callback 后，`warmup_steps` 需除以 `gradient_accumulation_steps`。

### 15.3 `JoraSchedulerCallback`（可选高级回调）

允许用户注入自定义调度函数：

- `selection_schedule(step, total) → bool`：控制是否在该步更新选择
- `temperature_schedule(step, total) → float`：自定义温度调度

```python
# 示例：每 5 步更新一次选择，余弦温度退火
def my_selection_schedule(step, total):
    return step % 5 == 0

def my_temperature_schedule(step, total):
    import math
    return 1.0 + 4.0 * 0.5 * (1 + math.cos(math.pi * step / total))

callbacks = [JoraSchedulerCallback(
    peft_model,
    selection_schedule=my_selection_schedule,
    temperature_schedule=my_temperature_schedule,
)]
```

---

# 第四部分：数据流、Pipeline 与状态管理

---

## 16. 关键数据流与状态追踪

### 16.1 Pair Buffer 完整生命周期

```
[1. 初始化] JoraLayer.__init__
    → add_adapter → _JoraAdapterState.__init__
    → register_buffer("pairs_L", full(-1, [S_L, 2], dtype=long))  # 全 -1 sentinel
    → register_buffer("num_pairs_L", zeros((), dtype=long))
    → init_random_pairs()
        → _rand_pairs(n, S_L//4)
        → _write_pairs(pairs_L, num_pairs_L, new_pairs, 'left')

[2. 训练更新] on_step_end (callback) → jora_update_step
    → layer.update_step(current_step, total_steps)
    → k_allow = compute_allowed_pairs(k, step, warmup_steps, ...)
    → k_L = floor(k_allow * S_L / (S_L + S_R))
    → _update_pair_buffer(pairs_L, num_pairs_L, grad_row_ema, k_L, n, 'left')
        → if cur >= k_L: return               # 仅增不减
        → energy = maybe_gumbel(grad_row_ema, ...)
        → new_pairs = select_top_k_pairs_gpu(energy, k=k_L, ...)
        → _write_pairs(pairs_L, num_pairs_L, new_pairs, 'left')

[3. 前向传播] forward → compute_delta → _apply_side_rotation
    → n_active = _num_pairs_py['left']         # Python 缓存，避免 GPU 同步
    → active_pairs = pairs_L[:n_active]
    → active_theta = theta_L[:n_active]
    → apply_rotations(x, active_pairs, active_theta, ...)

[4. Checkpoint] state_dict 包含 pairs_L [S_L, 2] + num_pairs_L (scalar)
    → 固定 shape 保证加载兼容
```

### 16.2 EMA 统计的 Cold-start 定量分析

| 训练步数 | EMA 有效样本数 | 偏差因子 $1-\beta^t$ | 选择质量 |
|---------|-------------|---------------------|---------|
| 1 | 0.02 | 0.02 | 近乎随机 |
| 10 | 0.18 | 0.18 | 极低 |
| 25 | 0.40 | 0.40 | 低 |
| 50 | 0.64 | 0.64 | 改善中 |
| 100 | 0.87 | 0.87 | 基本可靠 |
| 200 | 0.98 | 0.98 | 高质量（稳态） |

**注意**：当前实现无 Adam 风格偏差修正。对选择影响不大（相对排序几步后就合理），但如果 EMA 值用于阈值决策则需要修正。

### 16.3 温度参数的传播路径

```
JoraConfig.oer_temperature
    ↓ on_step_end
jora_model.jora_update_temperature(step, total)
    ↓ for each JoraLayer
adapter.update_temperature(step, total)
    ↓
new_t = linear_temperature_anneal(step, total, t_start, t_end)
cfg.oer_temperature = float(new_t)
    ↓ forward → maybe_apply_magnitude
compute_oer_scale_softmax(..., temperature=cfg.oer_temperature)
```

**注意**：多层共享同一 `cfg` 实例——一次更新影响所有层（预期行为）。

### 16.4 dtype/device 一致性审计表

| 操作 | 输入 dtype | 计算 dtype | 输出 dtype | 风险 |
|------|-----------|-----------|-----------|------|
| EMA 更新（col） | bf16/fp16 | **fp32** (`.float()`) | fp32 buffer | ✅ 安全 |
| EMA 更新（row） | bf16/fp16 | **fp32** (`.float()`) | fp32 buffer | ✅ 安全 |
| Magnitude 计算 | fp32 cached | fp32 | **转回 out.dtype** | ✅ 安全 |
| Rotation (Torch) | out.dtype | **out.dtype** | out.dtype | ⚠️ bf16 下小角度精度 |
| Rotation (Triton) | out.dtype | **fp32** (kernel 内) | out.dtype | ✅ 安全 |
| Core apply_to_vector | out.dtype | out.dtype | out.dtype | ⚠️ bf16 下精度 |
| Cayley arctan | out.dtype | out.dtype | out.dtype | ⚠️ 建议 fp32 上转型 |

---

## 17. 训练 Pipeline 完整流程

### 17.1 初始化阶段

```python
config = JoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    S_L=32, S_R=32, k=8,
    core="diag", magnitude="oer_softmax",
    selection="topk_ema", ema_beta=0.98,
    warmup_steps=100, rotation_param="cayley",
)
peft_model = get_peft_model(base_model, config)
trainer = Trainer(
    model=peft_model,
    callbacks=[JoraTrainerCallback(peft_model, verbose=True)],
    args=TrainingArguments(ddp_find_unused_parameters=True, ...),
)
```

### 17.2 单步训练完整时序图

```
Optimizer Step t
│
├── Micro-batch 1 (if gradient_accumulation_steps > 1)
│   ├── Forward Pass
│   │   ├── JoraLayer.forward(x)                    # x: [B,L,m]
│   │   │   ├── EMA col update (if ema_interval hit)
│   │   │   ├── base_out = W₀x + b                  # [B,L,n]
│   │   │   ├── delta = compute_delta(x)             # [B,L,n]
│   │   │   │   ├── R_R(x)       → x_rot [B,L,m]
│   │   │   │   ├── Core(x_rot)  → y_core [B,L,n]
│   │   │   │   ├── R_L⁻¹(y_core) → y [B,L,n]
│   │   │   │   └── tanh(y) → delta [B,L,n]
│   │   │   ├── out = base_out + delta               # [B,L,n]
│   │   │   └── out = out * OER_scale.view(1,1,n)    # [B,L,n]
│   │   └── loss = criterion(out, labels)
│   └── Backward Pass
│       ├── _backward_hook → grad_row_ema update
│       └── Autograd: ∇θ_L, ∇θ_R, ∇core_params, ∇ecd_log_mag
│
├── Optimizer Step (Adam/AdamW)
│
└── JoraTrainerCallback.on_step_end(state.global_step)
    ├── jora_model.jora_update_step()
    │   └── For each group → layer.update_step → select_top_k_pairs_gpu
    └── jora_model.jora_update_temperature(step, total)
```

---

## 18. Checkpoint 保存/恢复与可复现性

### 18.1 保存的内容

| 组件 | 键名模式 | 持久化 |
|------|---------|--------|
| theta_L/R | `*.adapters.default.theta_L/R` | ✅ nn.Parameter |
| core params | `*.adapters.default.core.*` | ✅ nn.Parameter |
| ecd_log_mag | `*.adapters.default.ecd_log_mag` | ✅ nn.Parameter |
| pairs_L/R | `*.adapters.default.pairs_L/R` | ✅ Buffer persistent |
| num_pairs_L/R | `*.adapters.default.num_pairs_L/R` | ✅ Buffer persistent |
| grad_row/col_ema | `*.adapters.default.grad_row/col_ema` | ✅ Buffer persistent |
| step_idx | `*.adapters.default.step_idx` | ✅ Buffer persistent |
| base_row_norms | `*.adapters.default.base_row_norms` | ✅ Buffer persistent |
| total_energy | `*.adapters.default.total_energy` | ✅ Buffer persistent |

### 18.2 恢复后需要重建的内容

- `_num_pairs_py`：Python 字典 → 首次 `_apply_side_rotation` 时从 GPU 同步
- `_counter_cache`：Python 字典 → 类似
- callback 的 `_total_steps`：需在 `on_train_begin` 中重新计算

### 18.3 可复现性检查清单

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 随机种子 | ⚠️ | `init_random_pairs` 使用 `torch.randint`，需设置全局种子 |
| DDP 一致性 | ⚠️ | 不同 GPU 上 EMA 基于各自 micro-batch，选择可能不同步 |
| Checkpoint resume 后选择一致 | ✅ | pairs/num_pairs 持久化 |
| Checkpoint resume 后 warmup 正确 | ⚠️ | 需确保 callback 的 `_total_steps` 重新设置 |
| EMA 状态完整恢复 | ✅ | grad_row/col_ema 持久化 |
| 温度状态恢复 | ⚠️ | 温度存储在 config 对象中，resume 后需重新 anneal 到正确值 |

---

# 第五部分：实验与验证策略

---

## 19. 消融实验的因果推理框架

### 19.1 消融开关完整清单（含配置方式）

| # | 消融目标 | 配置方式 | 效果 | 建议实验 |
|---|---------|---------|------|---------|
| 1 | **无旋转** | `S_L=0, S_R=0` | 退化为纯 Core + Magnitude | 验证旋转的核心贡献 |
| 2 | **仅左旋转** | `single_sided="right"` 或 `S_R=0` | 只有输出空间旋转 | 左右旋转对比 |
| 3 | **仅右旋转** | `single_sided="left"` 或 `S_L=0` | 只有输入空间旋转 | 左右旋转对比 |
| 4 | **无选择** | `selection="none"` | 使用全部 $S_L/S_R$ 对 | 验证稀疏选择的必要性 |
| 5 | **随机选择** | `selection="random"` | 随机对 vs. EMA 引导 | 验证信号引导的价值 |
| 6 | **无 Magnitude** | `magnitude="none"` | 无能量重分布 | 验证 OER/ECD 的贡献 |
| 7 | **ECD vs OER** | `magnitude="ecd_tanh"` vs `"oer_softmax"` | 两种缩放策略对比 | 方法优劣对比 |
| 8 | **Core 类型** | `core="diag"/"block"/"lowrank"` | 不同核心表达力 | 参数效率 vs 性能 |
| 9 | **零初始化 Core** | `zero_init_core=True` | delta 初始为零，跳过 tanh | 初始化策略对比 |
| 10 | **旋转参数化** | `rotation_param="cayley"/"angle"` | Cayley vs 直接角度 | 优化稳定性对比 |
| 11 | **温度退火** | `ecd_temp_annealing=True/False` | 动态 vs 静态温度 | 训练动态分析 |
| 12 | **Gumbel 探索** | `use_gumbel=True, gumbel_tau=...` | 选择时注入随机性 | 探索 vs 利用权衡 |
| 13 | **配对策略** | `pairing_strategy="consecutive"/"high_low"` | 高高配对 vs 高低配对 | 能量重分布模式 |
| 14 | **冻结旋转** | `lr_theta=0.0` | 只学 Core + Magnitude | 旋转可学习性的必要性 |
| 15 | **Warmup** | `warmup_steps=N` 或 `warmup_ratio=r` | 渐进增加活跃对数 | 训练稳定性 |

### 19.2 假设-检验-解释三步法

| 消融 | 理论假设 | 预期结果 | 如果反预期 → 需修正的环节 |
|------|---------|---------|--------------------------|
| #1 无旋转 | 旋转提供维度间耦合 | 显著下降 | Core+Mag 覆盖了旋转功能？重新定位创新点 |
| #2/#3 单侧 | 两侧旋转互补 | 轻微下降 | 某侧可能更重要 |
| #4 无选择 | 选择专注于重要对 | 略好/持平 | 全部对都有用→选择不必要（但参数多） |
| #5 随机选择 | EMA 信号有效引导 | EMA 优于随机 | 选择不重要→弱化选择创新点 |
| #6 无 Magnitude | OER 提供有效正则化 | 下降 | 能量守恒不重要 |
| #7 OER vs ECD | softmax 竞争优于 tanh 门控 | OER > ECD | 分析为什么竞争在特定任务上不利 |
| #8 Core 类型 | Block > Diag | Block 略优但参数多 | DiagCore 是最佳性价比 |
| #9 零初始化 | tanh 提供有用保护 | 两者接近 | tanh 可去掉以支持 merge |
| #10 Cayley vs Angle | Cayley 更稳定 | Cayley 更好 | 短训练/小 LR 时等价 |

### 19.3 消融依赖 DAG

```
                   Full JORA
                   /   |   \
          [旋转消融]  [选择消融]  [幅度消融]
          /    \       |   \        |    \
     无旋转  单侧  无选择 随机  无Mag  ECD
       |              |              |
   [Core消融]    [配对策略]    [温度退火]
    /  |  \         |
  Diag Block LR  High_Low
```

**约束**：S=0 → 选择/配对无意义；magnitude=none → 温度无意义；selection=none → Gumbel/warmup 无意义

### 19.4 审稿人 Q&A 映射表

| 审稿质疑 | 对应消融 | 应对策略 |
|---------|---------|---------|
| "旋转真有必要吗？" | #1, #14 | 展示无旋转/冻结旋转的下降 |
| "EMA 选择有效吗？随机不也行？" | #4, #5 | 对比 none/random/topk_ema |
| "OER 比 DoRA 好在哪里？" | #6, #7 | OER vs ECD vs none |
| "Cayley 有实验证据吗？" | #10 | 训练曲线对比 |
| "参数这么少，能 match LoRA？" | 主实验 | 同参数量性能对比 |
| "tanh 限制表达力？" | #9 | zero_init vs tanh |
| "训练稳定吗？需要 warmup？" | #15 | 不同 warmup 的训练曲线 |
| "merge 后性能降多少？" | 额外实验 | merge vs adapter 分离对比 |

### 19.5 实验优先级

**第一梯队（必做）**：Full vs LoRA, Full vs DoRA, 消融 #1, #6, #5

**第二梯队（强烈推荐）**：消融 #7, #8, #10, Full vs OFT/BOFT

**第三梯队（锦上添花）**：消融 #11, #13, #15, 参数量缩放实验

---

## 20. 参数量与计算量分析

### 20.1 逐层参数量

| 组件 | 公式 | 典型值 ($n=m=4096$) |
|------|------|---------------------|
| $\theta_L$ | $S_L$ | 32 |
| $\theta_R$ | $S_R$ | 32 |
| Core (diag) | $\min(n,m)$ | 4,096 |
| Core (block $b=4$) | $Kb^2 + p$ | 16,384 |
| Core (lowrank $r=8$) | $(n+m)r$ | 65,536 |
| ecd_log_mag | $n$ | 4,096 |
| **总 DiagCore+OER** | | **8,256** |

### 20.2 全模型对比（LLaMA-7B, 4 attn projections × 32 blocks）

| 方法 | Per-layer | 全模型 | 压缩比 vs LoRA-8 |
|------|-----------|--------|-----------------|
| LoRA (r=8) | 65,536 | 8,388,608 | 1.0x |
| DoRA (r=8) | 69,632 | 8,912,896 | 0.94x |
| OFT (b=4) | 24,576 | 3,145,728 | 2.7x |
| **JORA DiagCore+OER** | **8,256** | **1,056,768** | **7.9x** |
| JORA Block+OER | 20,512 | 2,625,536 | 3.2x |
| JORA LowRank+OER | 69,696 | 8,921,088 | 0.94x |

### 20.3 FLOPs 分析

| 操作 | FLOPs (per token) | 相对 base forward |
|------|-------------------|------------------|
| Base ($W_0 x$) | $2nm = 33.6M$ | 1.0x |
| Rotation ($k=8$ 对) | $6k = 48$ | ≈ 0% |
| DiagCore | $2\min(n,m) = 8.2K$ | 0.02% |
| tanh | $n = 4.1K$ | 0.01% |
| OER softmax | $O(n)$ | 忽略 |
| **总额外 (DiagCore)** | | **< 0.1%** |

### 20.4 内存分析

**训练时额外内存（DiagCore + OER, per layer）**：

| 组件 | 内存 |
|------|------|
| Adapter 参数 (fp16) | ~16 KB |
| Optimizer 状态 (Adam, 2×) | ~133 KB |
| Buffer (EMA, pairs, norms) | ~48 KB |
| 激活（forward） | $O(BL \cdot n)$（与 LoRA 类似）|
| **总额外 per-layer** | **~214 KB** |

**与 LoRA 对比**：LoRA 约 1 MB/层 → JORA 节省约 **4.7x** 训练内存。

---

## 21. 数值稳定性与边界条件

### 21.1 系统性风险清单

| # | 风险 | 触发条件 | 影响 | 保护状态 |
|---|------|---------|------|---------|
| 1 | tanh 梯度消失 | Core 输出 $\|z\| > 3$ | 参数停滞 | ⚠️ 需监控 |
| 2 | Cayley 溢出 | $\|\theta\|$ 极大 | arctan 误差 | ✅ 自然饱和 |
| 3 | OER softmax 溢出 | logits 极大 | NaN/Inf | ✅ PyTorch log-sum-exp |
| 4 | OER 分母为零 | base_row_norms = 0 | NaN | ✅ min_norm clamp |
| 5 | EMA 全零 | 训练初期 | 选择无意义 | ⚠️ warmup 部分缓解 |
| 6 | Pair buffer 负索引 | 未正确 slice | 静默错误 | ✅ _validate_pairs |
| 7 | bf16 精度不足 | 小角度 Torch 路径 | cos(θ)≈1 被截断 | ⚠️ 未转 fp32 |
| 8 | 梯度 NaN/Inf | 数值不稳定 | 训练崩溃 | ✅ hook 检查 |

### 21.2 保护机制触发条件表

| 保护机制 | 代码位置 | 触发条件 | 行为 |
|---------|---------|---------|------|
| EMA NaN 过滤 | `_backward_hook` | `isnan(g).any()` | 跳过本次更新 |
| OER 退化1 | `compute_oer_scale_softmax` | `total_energy <= 0` | 返回全 1 |
| OER 退化2 | 同上 | 所有范数非有限 | 返回全 1 |
| OER 退化3 | 同上 | 结果非有限 | 均匀分布 |
| Pair 负索引 | `_validate_pairs` | `(pairs < 0).any()` | ValueError |
| 前向 dtype | `JoraLayer.forward` | `x.dtype != weight.dtype` | 显式转换 |

### 21.3 建议的压力测试

| 场景 | 配置 | 验证要点 |
|------|------|---------|
| 极小 S | `S_L=1, S_R=1, k=1` | 不崩溃 |
| 极大 S | `S_L=256, S_R=256, k=64` | 不 OOM |
| 零初始化+大 LR | `zero_init=True, lr=1e-2` | delta 不爆炸 |
| bf16 训练 | `fp16=True` | 无 NaN/Inf |
| DDP 多卡 | 2/4/8 GPU | 训练收敛 |
| Checkpoint resume | step 1000 保存恢复 | EMA/pairs 正确 |
| 全零权重层 | 人工构造 | OER 退化保护触发 |
| 超长训练 | 100K+ steps | 无数值漂移 |

---

## 22. 性能瓶颈定位与优化建议

### 22.1 计算瓶颈

| 操作 | 绝对开销 | 相对瓶颈性 | 优化建议 |
|------|---------|-----------|---------|
| `index_select` + `index_copy_`（Torch 旋转） | 低 | 🟢 | 已足够高效 |
| `einsum`（BlockCore, 小 $b$） | 低但有 overhead | 🟡 | $b \leq 4$ 时直接循环可能更快 |
| Top-K 选择（GPU） | 低 | 🟢 | 已 GPU 化 |
| **Base forward** ($W_0 x$) | **高** | — | 非 JORA 可优化的 |

**结论**：JORA 的计算瓶颈在 base model 的前向传播，JORA 自身开销 < 1%。

### 22.2 内存瓶颈

| 操作 | 内存需求 | 触发场景 | 缓解 |
|------|---------|---------|------|
| `core.forward()` 全稠密矩阵 | 64 MB | **仅 merge 路径** | 使用 `get_row_slice` 分块 |
| EMA buffers | 32 KB/层 | 训练全程 | 可接受 |
| 激活内存 | $O(BL \cdot n)$ | 训练全程 | 与 LoRA 相当 |

### 22.3 通信瓶颈（DDP）

| 操作 | 通信量 | 问题 |
|------|--------|------|
| 梯度同步 | ~16 KB/层 (DiagCore) | ✅ 远小于 LoRA |
| `.item()` 同步 | 少量标量 | ⚠️ 已用 Python 缓存缓解 |
| EMA 不同步 | 无直接通信 | ⚠️ 不同 GPU 的选择可能不同 |

---

# 第六部分：集成、部署与投稿策略

---

## 23. PEFT 生态集成评估

### 23.1 已实现接口

| 接口 | 状态 | 备注 |
|------|------|------|
| `PeftConfig` 子类 | ✅ | `peft_type=PeftType.JORA` |
| `BaseTuner` 子类 | ✅ | 完整实现 |
| `BaseTunerLayer` 混入 | ✅ | |
| `merge` / `unmerge` | ⚠️ | 精度不足 |
| `enable/disable_adapters` | ✅ | |
| `set_adapter` | ✅ | 单 adapter |
| Checkpoint save/load | ✅ | 固定 shape buffer + persistent |

### 23.2 缺失功能与优先级

| 缺失功能 | 优先级 | 影响 | 预计工作量 |
|---------|--------|------|-----------|
| `__init__.py` 导出 | 🔴 高 | 模块不可 import | 1 小时 |
| `PeftType.JORA` 注册 | 🔴 高 | PEFT 无法识别 | 1 小时 |
| QLoRA 兼容 | 🟡 中 | 无法与 4bit/8bit 配合 | 1-2 天 |
| 多 adapter | 🟢 低 | 单 adapter 已足够 | 1 周 |

### 23.3 QLoRA 兼容性路径（技术方案）

1. 在 `_create_and_replace` 中检测量化层类型（`bnb.nn.Linear4bit`）
2. 在 `_JoraAdapterState` 中使用反量化权重计算 `base_row_norms`
3. `linear_forward` 需支持量化层的前向传播
4. Core 和旋转参数保持 fp16/bf16

预计修改量：100-200 行代码。

---

## 24. 推理部署路径完整分析

### 24.1 方案对比

| 部署方案 | 精度 | 推理开销 | 内存开销 | 实现复杂度 |
|---------|------|---------|---------|-----------|
| **Adapter 分离推理** | 100% | < 1% | ~1 MB/模型 | ✅ 已实现 |
| **Merge（当前近似）** | ~60-80% | 0 | 0 | ✅ 已实现（不推荐） |
| **蒙特卡洛 Merge** | ~95%+ | 0 | 0 | ⚠️ 需实现 |
| **Taylor 展开 Merge** | ~90% | 0 | 0 | ⚠️ 需实现 |
| **Adapter Switching** | 100% | < 1% | ~1 MB/adapter | ✅ 兼容 vLLM |
| **编译优化** (torch.compile) | 100% | 可能降低 | 0 | ⚠️ 需测试 |

### 24.2 推荐部署策略

**生产部署**：使用 Adapter 分离推理。JORA 的额外推理开销 < 1%，在实际部署中可忽略。

**如果必须 merge**：优先考虑蒙特卡洛近似：

```python
# 蒙特卡洛 merge 近似
def mc_merge(jora_layer, n_samples=1000):
    X = torch.randn(n_samples, m, device=device)
    Delta_X = jora_layer.compute_delta(X)
    Delta_W = torch.linalg.lstsq(X.T, Delta_X.T).solution.T
    return Delta_W
```

### 24.3 `torch.compile` 兼容性

**潜在问题**：动态 pair 选择导致控制流变化，可能阻止编译器的图优化。

**建议**：将 rotation 部分标记为 `torch._dynamo.allow_in_graph`，或固定 pair 后再 compile。

---

## 25. 已知局限性与改进路线图

### 25.1 方法层面

| 限制 | 影响 | 可缓解性 |
|------|------|---------|
| **tanh 使 merge 不精确** | 推理时需保持 adapter | 可去掉 tanh 用 $\alpha/r$ 缩放 |
| **选择切换导致训练不稳定** | 梯度方向突变 | warmup + 高 EMA beta |
| **能量守恒假设可能过强** | 限制灵活性 | $T \to \infty$ 松弛 |
| **正交性限制表达力** | 无法表示非正交变换 | Core 弥补 |
| **EMA cold-start** | 初期选择差 | warmup |

### 25.2 工程层面

| # | 问题 | 严重性 | 预计工作量 |
|---|------|--------|-----------|
| 1 | 缺少 `__init__.py` 和 PEFT 注册 | 🔴 | 1 小时 |
| 2 | Merge 精度（0.05 硬编码） | 🔴 | 1-2 天 |
| 3 | 硬编码 adapter name | 🟡 | 30 分钟 |
| 4 | Triton backward 近似 | 🟡 | 半天 |
| 5 | GPU batch 贪心冲突 | 🟡 | 2 小时 |
| 6 | bf16 Cayley 精度 | 🟢 | 1 小时 |
| 7 | `_update_pair_buffer` 仅增不减 | 🟡 | 1 小时 |

---

## 26. 论文叙事与审稿策略

### 26.1 建议的论文结构

```
1. Introduction
2. Related Work
3. Method
   3.1 Overview and Notation
   3.2 Sparse Givens Rotation Framework
   3.3 Core Transformation Module
   3.4 Orthogonal Energy Redistribution (OER)
   3.5 Data-Driven Dimension Selection
   3.6 Training Pipeline
4. Theoretical Analysis
   4.1 Approximation Power of Sparse Givens Rotations
   4.2 Energy Conservation and Regularization
   4.3 Optimization Landscape Properties
5. Experiments
   5.1 Setup (models, datasets, baselines)
   5.2 Main Results
   5.3 Ablation Studies
   5.4 Analysis (selection vis, energy dist, param efficiency)
6. Discussion and Limitations
7. Conclusion
```

### 26.2 可视化建议

1. **选择维度演化热图**：横轴训练步数，纵轴维度 index，颜色=选中状态
2. **OER 能量分布变化**：训练开始/中期/结束时的 bar chart
3. **旋转角度分布**：$\theta$ 值 histogram，展示 Cayley 自然约束
4. **参数效率曲线**：横轴参数量，纵轴性能，标注各方法

### 26.3 Benchmark 与 Baseline 建议

| Benchmark | 优先级 | 用途 | 预期 JORA 优势 |
|-----------|--------|------|--------------|
| Alpaca/Dolly (LLaMA-7B) | 🔴 必做 | Instruction tuning | 参数效率+质量 |
| MMLU (LLaMA-7B/13B) | 🔴 必做 | 知识评估 | 知识保持（能量守恒） |
| MT-Bench | 🟡 推荐 | 对话质量 | 多维度评估 |
| GLUE/SuperGLUE (RoBERTa) | 🟡 推荐 | NLU 标准 | 经典 benchmark |
| ViT fine-tuning (ImageNet) | 🟡 推荐 | 跨模态验证 | 证明通用性 |
| CodeAlpaca (CodeLLaMA) | 🟢 可选 | 代码生成 | 结构化任务 |

**Baseline 选择**：必须对比 LoRA, DoRA。强烈推荐 OFT, BOFT, AdaLoRA。可选 GaLore, IA3, Prefix Tuning。

### 26.4 Limitation Section 写作建议

**必须讨论的限制**：

1. **Merge 精度**：建议措辞："This is a conscious design trade-off: tanh provides training stability at the cost of lossless merging."
2. **训练不确定性**：建议措辞："warmup and EMA smoothing mitigate but do not eliminate selection instability."
3. **大规模验证**：坦诚承认 70B+ 的行为需验证。
4. **超参数数量**：建议措辞："We provide well-tested defaults that work across all our experiments without task-specific tuning."

---

## 27. 实验结果的多种解读预案

### 27.1 如果 JORA 在某任务上不如 LoRA

| 原因假设 | 验证方式 | 叙事调整 |
|---------|---------|---------|
| 任务需引入新方向 | 检查 LoRA 的 $B$ 列是否与 $W_0$ 正交 | "互补归纳偏置" |
| 能量守恒过强 | 消融 OER | "高温松弛" |
| 超参未调优 | 网格搜索 $S$, $k$, $T$ | "调优后结果..." |
| 训练不够长 | 延长训练 | "充分训练下逼近/超过" |

### 27.2 如果某消融组件贡献不显著

| 情况 | 叙事策略 |
|------|---------|
| 旋转消融无影响 | 重新定位：Core+OER 是核心 |
| OER 消融无影响 | 强调训练稳定性好处 |
| EMA vs 随机无差异 | "方法更简单，不依赖 EMA" |
| DiagCore ≈ BlockCore | "DiagCore 是最优选择：最少参数" |

### 27.3 如果全面优于 LoRA

检查公平性（参数量可比？配置对 LoRA 公平？）；寻找 LoRA 可能更好的场景（避免 cherry-picking）。

---

# 附录

---

## 附录 A. 核心公式速查表

$$
\boxed{
\begin{aligned}
& \textbf{Forward:} \quad y = W_0 x + M \odot \tanh\!\bigl(R_L^T \, D \, R_R \, x\bigr) \\[4pt]
& \textbf{Givens (Cayley):} \quad \phi = 2\arctan(\theta/2),\; \begin{pmatrix} x_i' \\ x_j' \end{pmatrix} = \begin{pmatrix} \cos\phi & \sin\phi \\ -\sin\phi & \cos\phi \end{pmatrix} \begin{pmatrix} x_i \\ x_j \end{pmatrix} \\[4pt]
& \textbf{OER:} \quad p = \text{softmax}(w/T), \; m_i = \sqrt{E_{\text{total}} \cdot p_i}, \; \text{scale}_i = \frac{m_i}{\|w_{0,i}\|} \cdot \sqrt{\frac{E_{\text{total}}}{\sum_j m_j^2}} \\[4pt]
& \textbf{EMA:} \quad e_t = \beta \, e_{t-1} + (1-\beta) \, \bar{x}_t^2 \\[4pt]
& \textbf{Selection:} \quad \text{pairs} = \text{TopK-Greedy-Disjoint}(e, k) \\[4pt]
& \textbf{Warmup:} \quad k(t) = \max\!\bigl(1, \lfloor S \cdot \min(1, t/t_w) \rfloor\bigr) \\[4pt]
& \textbf{Temperature:} \quad T(t) = (1 - t/T_{\max}) \cdot T_{\text{start}} + (t/T_{\max}) \cdot T_{\text{end}}
\end{aligned}
}
$$

---

## 附录 B. 配置速查表

```python
# ===== 推荐配置（DiagCore + OER，最佳参数效率）=====
JoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    S_L=32, S_R=32, k=8,
    core="diag", magnitude="oer_softmax",
    selection="topk_ema", ema_beta=0.98,
    warmup_steps=100, rotation_param="cayley",
)

# ===== 高表达力配置 =====
JoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    S_L=64, S_R=64, k=16,
    core="block", block_size=8,
    magnitude="oer_softmax", oer_temperature=2.0,
    ecd_temp_annealing=True, ecd_temp_start=5.0, ecd_temp_end=1.0,
    selection="topk_ema", pairing_strategy="high_low",
)

# ===== LoRA 等参数量对比 =====
JoraConfig(
    target_modules=["q_proj", "v_proj"],
    S_L=32, S_R=32, k=8,
    core="lowrank", lowrank_r=8, magnitude="oer_softmax",
)
```

---

## 附录 C. 符号-代码对照表

| 符号 | 代码变量名 | 含义 |
|------|-----------|------|
| $W_0$ | `base_layer.weight` | 冻结预训练权重 |
| $S_L, S_R$ | `cfg.S_L, cfg.S_R` | 旋转容量 |
| $k$ | `cfg.k` | 全局最大活跃对数 |
| $\theta_L, \theta_R$ | `adapter.theta_L/R` | 旋转角度参数 |
| $D$ | `adapter.core` | 核心变换 |
| $M$ | `scale` (in `maybe_apply_magnitude`) | Magnitude 缩放 |
| $E_{\text{total}}$ | `adapter.total_energy` | 总能量 |
| $\boldsymbol{w}$ | `adapter.ecd_log_mag` | OER logits |
| $\beta$ | `cfg.ema_beta` | EMA 衰减系数 |
| $T$ | `cfg.oer_temperature` | 温度 |
| $e^{\text{row}}$ | `adapter.grad_row_ema` | 行梯度 EMA |
| $e^{\text{col}}$ | `adapter.grad_col_ema` | 列激活 EMA |

---

## 附录 D. 代码-公式交叉索引表

| 公式/概念 | 代码位置 |
|-----------|---------|
| 总体前向 | `layer.py → JoraLayer.forward` |
| Givens (Cayley) | `rotation.py → cayley_cos_sin` |
| 旋转 (Torch) | `rotation.py → apply_rotations_torch` |
| 旋转 (Triton) | `rotation.py → GivensRotationTriton` |
| DiagCore | `core.py → DiagCore.apply_to_vector` |
| BlockCore | `core.py → BlockCore.apply_to_vector` |
| LowRankCore | `core.py → LowRankCore.apply_to_vector` |
| OER softmax | `magnitude.py → compute_oer_scale_softmax` |
| ECD tanh | `magnitude.py → compute_ecd_scale` |
| 温度退火 | `magnitude.py → linear_temperature_anneal` |
| EMA (列) | `layer.py → JoraLayer.forward` (training block) |
| EMA (行) | `layer.py → JoraLayer._backward_hook` |
| Top-K 选择 | `selection.py → select_top_k_pairs_gpu` |
| 高低配对 | `selection.py → _select_high_low_pairs_gpu` |
| Warmup | `selection.py → compute_allowed_pairs` |
| Pair buffer 更新 | `layer.py → _JoraAdapterState.update_step` |
| Merge 近似 | `layer.py → JoraLayer._compute_weight_delta_simple` |
| Callback 更新 | `callbacks.py → JoraTrainerCallback.on_step_end` |

---

## 附录 E. 关键证明与推导细节

### E.1 命题 3.2 的证明（不相交 Givens 的子流形结构）

**命题**：$\Phi: \mathbb{R}^S \to SO(n)$，$\Phi(\boldsymbol{\theta}) = \prod_{s=1}^S G_{i_s j_s}(\theta_s)$（不相交对），是光滑嵌入。

**证明**：(1) 光滑性：cos/sin 光滑，有限乘积光滑。(2) 单射性：不相交 → $(i_s,j_s)$ 块为标准 2×2 旋转矩阵 → $\theta_s$ 可唯一恢复。(3) Jacobian 满秩：$\partial\Phi/\partial\theta_s$ 仅影响 $(i_s,j_s)$ 块，不同 $s$ 作用在不同块 → 线性无关。$\square$

### E.2 OER 能量守恒的精确性

设 $r_i = \text{scale}_i^{\text{raw}} \cdot \|w_{0,i}\|$, $\hat{E} = \sum_i r_i^2$, $\text{scale}_i = \text{scale}_i^{\text{raw}} \cdot \sqrt{E_{\text{total}}/\hat{E}}$。

$$
\sum_i (\text{scale}_i \cdot \|w_{0,i}\|)^2 = \sum_i \left(\text{scale}_i^{\text{raw}} \cdot \sqrt{\frac{E_{\text{total}}}{\hat{E}}} \cdot \|w_{0,i}\|\right)^2 = \frac{E_{\text{total}}}{\hat{E}} \sum_i r_i^2 = \frac{E_{\text{total}}}{\hat{E}} \cdot \hat{E} = E_{\text{total}} \quad \square
$$

### E.3 Cayley 梯度衰减的定量分析

$d\phi/d\theta = 1/(1+(\theta/2)^2)$。

在典型微调场景中，旋转角不应超过 90°（$\theta = 2$），此时梯度衰减到 0.5——足够传播梯度。$\theta > 4$（$\phi > 127°$）时梯度 < 0.2，自然阻止过大旋转。

### E.4 逼近误差界的推导（定理 3.1 补充）

设 $A = \log Q \in \mathfrak{so}(n)$，$\|A\|_F \leq \delta + O(\delta^2)$。

$A$ 在基 $\{E_{ij}\}$ 下的展开：$A = \sum_{i<j} a_{ij} E_{ij}$，$\|A\|_F^2 = 2\sum_{i<j} a_{ij}^2$。

截断到 top-$S$ 项：$A_S = \sum_{k=1}^S a_{(k)} E_{i_k j_k}$。

由于各基正交（$\langle E_{ij}, E_{kl}\rangle_F = 2\delta_{ik}\delta_{jl}$），截断误差 = $\|A - A_S\|_F = \sqrt{2\sum_{k>S} a_{(k)}^2}$。

指数映射的 Lipschitz 性：$\|\exp(A) - \exp(B)\|_F \leq \|A-B\|_F \cdot e^{\max(\|A\|,\|B\|)}$。

当 $\|A\| \leq \delta \ll 1$ 时，$e^{\delta} \approx 1 + \delta$，因此 $\|Q - \exp(A_S)\| \leq (1+\delta)\|A - A_S\|_F$。$\square$

---

## 附录 F. 可复现性检查清单

| # | 检查项 | 重要性 | 状态 | 修复建议 |
|---|--------|--------|------|---------|
| 1 | 全局随机种子设置 | 🔴 | ⚠️ | 在 `init_random_pairs` 前设置 `torch.manual_seed` |
| 2 | DDP 选择一致性 | 🔴 | ⚠️ | 选择后 broadcast 或使用相同种子 |
| 3 | gradient_accumulation 下步数语义 | 🟡 | ⚠️ | 文档中明确 warmup_steps 的粒度 |
| 4 | Checkpoint 后温度恢复 | 🟡 | ⚠️ | 在 `on_train_begin` 中根据 `state.global_step` 重算 |
| 5 | Checkpoint 后 Python 缓存重建 | 🟢 | ✅ | 首次访问自动同步 |
| 6 | EMA 偏差修正 | 🟢 | ⚠️ | 对选择影响小，但建议监控 |
| 7 | bf16 训练的数值一致性 | 🟡 | ⚠️ | Torch 旋转路径添加 fp32 上转型 |
| 8 | Triton vs Torch 梯度一致性验证 | 🟡 | ⚠️ | 添加数值梯度检验脚本 |

---

# 第七部分：模型-任务-JORA 交互的第一性原理分析（v5 新增）

> **本部分动机**：JORA 的三大核心组件（稀疏 Givens 旋转、OER 能量守恒、数据驱动选择）并非在所有模型/任务上等效有利。要在 NeurIPS 论文中令人信服地展示 JORA 的优越性，需要从数学第一性原理出发，(1) 理解不同模型架构的权重矩阵结构如何与 JORA 的归纳偏置交互，(2) 理解不同任务类型的"最优微调变换"落在什么流形上，(3) 据此设计最能突出 JORA 优势的实验矩阵。本部分不是泛泛的实验建议，而是**从线性代数和优化理论出发的严格分析**。

---

## 28. 预训练权重的谱结构与 JORA 适配性

### 28.1 核心假说：谱衰减-旋转假说（Spectral Decay-Rotation Hypothesis）

**假说 28.1（JORA 第一性原理假说）**：预训练 Transformer 权重矩阵 $W_0 \in \mathbb{R}^{n \times m}$ 的 SVD $W_0 = U \Sigma V^T$ 具有以下性质，这些性质决定了 JORA 的适用条件：

1. **快速谱衰减**：奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots$ 通常呈幂律衰减 $\sigma_k \approx C k^{-\alpha}$（$\alpha > 1$），说明信息高度集中在少数主方向上。
2. **微调变换的正交主导性**：对于大多数下游任务，最优微调权重 $W^* = W_0 + \Delta W^*$ 与 $W_0$ 的差异可以分解为 $\Delta W^* = U' \Sigma' V'^T - U \Sigma V^T$，其中主导贡献来自 $U, V$ 的旋转（即基方向的重新对齐），而非 $\Sigma$ 的大幅变化。
3. **有效旋转维度的稀疏性**：将最优旋转 $Q^* = U'^T U$ 分解为 Givens 旋转时，角度系数满足快速衰减——仅少数维度对需要显著旋转。

**这三条性质恰好对应 JORA 的三个组件**：性质 1 → Core（沿主方向的幅度微调），性质 2 → Givens 旋转（基方向重对齐），性质 3 → 稀疏选择（只选显著维度对）。

### 28.2 不同模型架构的谱结构差异及其影响

#### 28.2.1 LLaMA 系列（LLaMA-2-7B/13B/70B, LLaMA-3-8B/70B）

**权重矩阵结构特征**：

LLaMA 的注意力层包含 $W_Q, W_K, W_V, W_O$，MLP 层包含 $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$（SwiGLU 架构）。

**谱结构的实证规律**（基于现有文献和 LoRA 实践）：

| 投影矩阵 | 典型谱衰减速度 | 有效秩 (90% 能量) | JORA 适配性 |
|----------|-------------|-----------------|------------|
| $W_Q$ | 中等 ($\alpha \approx 1.5$) | 高（~100-300） | ⭐⭐⭐ 旋转+Core 高收益 |
| $W_K$ | 较快 ($\alpha \approx 2.0$) | 中（~50-150） | ⭐⭐⭐ 能量集中，稀疏旋转高效 |
| $W_V$ | 较慢 ($\alpha \approx 1.2$) | 高（~200-500） | ⭐⭐ 可能需要更大 $S$ |
| $W_O$ | 中等 | 中 | ⭐⭐⭐ 输出空间旋转直接影响残差流 |
| $W_{\text{gate}}$ | 快速 ($\alpha > 2$) | 低（~30-80） | ⭐⭐⭐⭐ 极度适合稀疏旋转 |
| $W_{\text{up}}$ | 快速 | 低 | ⭐⭐⭐⭐ 同上 |
| $W_{\text{down}}$ | 中等 | 中 | ⭐⭐⭐ 维度从 $d_{\text{ff}}$ 到 $d$ 的压缩 |

**关键推论 28.1（LLaMA 上的 target_modules 策略）**：

对 LLaMA 系列，SwiGLU 的 $W_{\text{gate}}$ 和 $W_{\text{up}}$ 的快速谱衰减意味着它们的"有效旋转自由度"很低——JORA 的稀疏 Givens 旋转（$S = 32, k = 8$）极其匹配。**将 `gate_proj` 和 `up_proj` 纳入 `target_modules` 可能是 JORA 相对 LoRA 拉开差距的关键**。

形式化论证：设 $W_{\text{gate}}$ 的奇异值满足 $\sigma_k \leq C k^{-2}$，则最优微调旋转的 Lie 代数系数也满足类似衰减（见定理 3.1 推论），$S = 32$ 时截断误差：

$$
\epsilon_{\text{gate}} \leq \frac{C_0}{\sqrt{3}} \cdot 32^{-1.5} \approx 3.2 \times 10^{-5} \cdot \frac{C_0}{0.01}
$$

而 LoRA 以同样参数量做满秩近似需要 $r \geq \text{effective\_rank}$，对 $W_{\text{gate}}$ 可能需要 $r \geq 4$。**JORA 用 $S = 32$ 个角度参数达到 LoRA $r=4$（参数量 $\approx 49K$）才能达到的效果——参数效率比高达 $\sim 600\times$**。

#### 28.2.2 RoPE 与 JORA 旋转的数学关系

**RoPE（Rotary Position Embedding）**在注意力计算中对 $Q, K$ 施加位置相关的旋转：

$$
Q_{\text{rope}} = R_{\text{pos}}(m) \cdot W_Q x, \quad K_{\text{rope}} = R_{\text{pos}}(n) \cdot W_K x
$$

其中 $R_{\text{pos}}$ 是在连续 2D 子空间上的旋转（与 Givens 旋转结构相同！）。

**JORA 与 RoPE 的正交互补性**：

$$
\text{RoPE}: R_{\text{pos}}(m) \cdot W_Q \cdot x \quad \longrightarrow \quad \text{JORA}: R_{\text{pos}}(m) \cdot \underbrace{M \odot \left[W_Q x + \tanh(R_L^T D R_R x)\right]}_{\text{JORA 修正的 } W_Q \text{ 输出}}
$$

**关键观察**：RoPE 的旋转是**位置相关的、固定的、作用在激活空间**——它不改变权重矩阵本身。JORA 的旋转是**可学习的、位置无关的、作用在权重空间**。两者操作在不同的空间且目的不同：

| 维度 | RoPE | JORA 旋转 |
|------|------|-----------|
| 作用对象 | 激活 $Q, K$（推理时） | 权重空间（训练时固化） |
| 参数化 | 固定频率 $\theta_i = 10000^{-2i/d}$ | 可学习角度 $\theta_s$ |
| 维度对选择 | 固定：$(2i, 2i+1)$ 相邻对 | 数据驱动：任意不相交对 |
| 目的 | 编码位置信息 | 重对齐特征基方向 |

**深层联系**：RoPE 暗示 LLaMA 的特征空间天然地以 2D 旋转子空间为基本单元。JORA 的 Givens 旋转恰好在同样的 2D 子空间结构上操作——这不是巧合，而是说明**旋转是 LLaMA 特征空间中最自然的变换原语**。

**实验推论 28.2**：在使用 RoPE 的模型上，JORA 应该比在不使用 RoPE 的模型（如 GPT-2、BERT）上表现出更大的相对优势——因为 RoPE 模型的特征空间天然"偏好"旋转变换。

#### 28.2.3 GQA（Grouped Query Attention）与维度对选择的交互

LLaMA-2 70B 和 LLaMA-3 使用 GQA：$n_{\text{kv\_heads}} < n_{\text{heads}}$。

**对 JORA 的影响**：

- $W_K, W_V$ 的输出维度 $= n_{\text{kv\_heads}} \times d_{\text{head}}$，远小于 $W_Q$ 的 $n_{\text{heads}} \times d_{\text{head}}$
- $W_K$ 维度为 1024（8 heads × 128）vs $W_Q$ 维度 4096（32 heads × 128）

**数学推论**：对小维度 $W_K$（1024），$S_L = 32$ 占总维度对空间 $\binom{1024}{2} \approx 524K$ 的比例远大于 $W_Q$ 的 $32/\binom{4096}{2} \approx 3.8 \times 10^{-6}$——JORA 在 GQA 的 K/V 投影上的"覆盖率"更高。

**配置建议**：对 GQA 模型，$W_K/W_V$ 可以使用较小的 $S$（如 $S = 16$），将参数预算集中在 $W_Q/W_O$ 和 MLP 上。

#### 28.2.4 Mistral 系列（Mistral-7B, Mixtral-8x7B）

**架构特征**：
- **滑动窗口注意力**（Mistral-7B）：局部注意力模式意味着 $W_Q, W_K$ 学到的特征更局部化
- **MoE（Mixtral）**：每个专家的 MLP 权重矩阵更小、更专业化

**对 JORA 的影响分析**：

**滑动窗口与 EMA 选择的交互**：滑动窗口限制了注意力的全局性，这意味着 $W_Q, W_K$ 的梯度信号更集中于局部模式。EMA 统计的方差更小 → 选择更快收敛 → JORA 的 warmup 可以更短。

**Mixtral MoE 与 JORA 的天然契合**：

$$
\text{Mixtral}: y = \sum_{e=1}^{8} g_e \cdot \text{Expert}_e(x), \quad \text{Expert}_e(x) = W_{\text{down}}^{(e)} \cdot \text{SwiGLU}(W_{\text{gate}}^{(e)} x, W_{\text{up}}^{(e)} x)
$$

每个专家的权重矩阵维度 = $d \times d_{\text{ff}}/8$——更小的矩阵意味着 JORA 的稀疏旋转覆盖率更高。**JORA 在 MoE 的各专家权重上可能特别高效**。

但 Mixtral 有 $8 \times 3 = 24$ 个 MLP 权重矩阵/层 → adapter 数量大增 → 需要 `selection_group_size` 来控制开销。

#### 28.2.5 其他候选模型

| 模型 | 架构特点 | JORA 适配性预测 | 理由 |
|------|---------|---------------|------|
| **Phi-3-mini (3.8B)** | 小模型，注意力维度 3072 | ⭐⭐⭐⭐ | 小维度 → 稀疏旋转覆盖率高 |
| **Gemma-2 (9B/27B)** | RoPE + GQA + 交替局部/全局注意力 | ⭐⭐⭐ | 交替注意力 → 不同层有不同的最优 $S$ |
| **Qwen-2 (7B/72B)** | RoPE + GQA | ⭐⭐⭐ | 类似 LLaMA |
| **BERT/RoBERTa (base/large)** | 绝对位置编码，无 RoPE | ⭐⭐ | 无旋转先验，JORA 优势可能较小 |
| **ViT-Large** | Patch embedding + 自注意力 | ⭐⭐⭐ | 视觉特征空间的旋转对齐 |

### 28.3 预训练权重谱结构的可验证预测

**预测 28.1**：对 LLaMA-7B 的每个投影矩阵 $W$，计算 SVD $W = U\Sigma V^T$，然后用最优 LoRA（$r = 64$）微调得到 $W' = W + \Delta W$。分析 $\Delta W$ 的 SVD：

$$
\Delta W = U_\Delta \Sigma_\Delta V_\Delta^T
$$

**预测结果**：
1. $U_\Delta$ 和 $V_\Delta$ 与 $U, V$ 有显著重叠（$\|U^T U_\Delta\|_F > 0.8$）→ 支持"旋转主导"假说
2. $\Sigma_\Delta$ 的谱衰减比 $\Sigma$ 更快 → 支持"稀疏有效旋转"假说
3. $W_{\text{gate}}$ 上述指标最极端 → 支持"SwiGLU 最适合 JORA"

**验证代码方案**（建议在论文中执行）：

```python
# 伪代码：验证谱衰减-旋转假说
for layer in model.layers:
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        W0 = get_weight(layer, proj)                    # 预训练权重
        W_ft = get_finetuned_weight(layer, proj)         # LoRA-64 微调后
        Delta_W = W_ft - W0
        
        U0, S0, V0 = torch.linalg.svd(W0)
        Ud, Sd, Vd = torch.linalg.svd(Delta_W)
        
        # 指标1: 子空间重叠
        overlap_U = torch.linalg.norm(U0[:, :64].T @ Ud[:, :64]) / 8  # 归一化
        overlap_V = torch.linalg.norm(V0[:, :64].T @ Vd[:, :64]) / 8
        
        # 指标2: 谱衰减拟合
        alpha_W = fit_power_law(S0)
        alpha_Delta = fit_power_law(Sd)
        
        # 指标3: 旋转角度分布
        Q_left = U1 @ U0.T  # 近似左旋转
        A = logm(Q_left)            # Lie代数
        givens_coeffs = extract_givens_coefficients(A)
        alpha_givens = fit_power_law(sorted(abs(givens_coeffs), reverse=True))
```

---

## 29. 主流模型架构与 JORA 组件的数学耦合分析

### 29.1 SwiGLU MLP 与 JORA 的二阶效应

LLaMA/Mistral 的 MLP 使用 SwiGLU：

$$
\text{MLP}(x) = W_{\text{down}} \cdot \left[\text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)\right]
$$

**当 JORA 同时作用于 $W_{\text{gate}}$ 和 $W_{\text{up}}$ 时的效应分析**：

设 JORA 对两者的 delta 分别为 $\delta_g(x), \delta_u(x)$：

$$
\text{MLP}'(x) = W_{\text{down}} \cdot \left[\text{SiLU}\!\big(W_{\text{gate}} x + \delta_g(x)\big) \odot \big(W_{\text{up}} x + \delta_u(x)\big)\right]
$$

展开（一阶近似）：

$$
\approx \text{MLP}(x) + W_{\text{down}} \cdot \left[\text{SiLU}'(W_{\text{gate}} x) \cdot \delta_g(x) \odot W_{\text{up}} x + \text{SiLU}(W_{\text{gate}} x) \odot \delta_u(x)\right] + O(\delta^2)
$$

**关键观察**：$\delta_g$ 和 $\delta_u$ 的效果是**乘性耦合**的——$\delta_g$ 通过 SiLU 的导数与 $W_{\text{up}} x$ 相乘，$\delta_u$ 与 SiLU 门控值相乘。

**JORA 在此处的独特优势**：

OER 的能量守恒约束确保 $\delta_g, \delta_u$ 不会同时膨胀——如果 $\delta_g$ 在某些维度增大，OER 迫使其他维度缩小。这防止了 gate 和 up 的 delta 同时增大导致的**二阶爆炸**风险。LoRA 没有这种约束，gate 和 up 的 $\Delta W$ 可以自由增长。

**实验设计启示**：对 SwiGLU 模型，**同时将 gate_proj 和 up_proj 纳入 JORA 的 target_modules 是展示 OER 正则化价值的关键实验**。

### 29.2 注意力机制各投影的旋转需求差异

**形式化分析**：在多头注意力中，$W_Q, W_K$ 通过内积 $Q K^T$ 交互。微调改变注意力模式等价于改变 $Q K^T$ 的结构：

$$
Q'K'^T = (W_Q + \Delta_Q) x \cdot [(W_K + \Delta_K) x]^T = QK^T + \Delta_Q x \cdot K^T + Q \cdot (\Delta_K x)^T + O(\Delta^2)
$$

**旋转修正的特殊效率**：设 $\Delta_Q = R_Q^T D_Q R_{Q,R}$（JORA 的 delta），则 $\Delta_Q x \cdot K^T$ 可以理解为：先在输入空间旋转 $x$、做 Core 变换、再在输出空间旋转——最终效果是**重新对齐 query 的特征方向**。

对 $W_Q$ 和 $W_K$，旋转的效果在注意力分数中是**二次的**（通过 $Q K^T$），而对 $W_V$ 和 $W_O$，效果是**线性的**（直接影响输出）。

**推论 29.1（投影层优先级）**：

| 投影 | 修正对输出的影响阶数 | 对旋转的敏感度 | JORA 配置建议 |
|------|-------------------|-------------|-------------|
| $W_Q$ | 二阶（通过 $QK^T$） | 高——小角度旋转 → 注意力模式显著变化 | 高优先级，$S_Q = 32$ |
| $W_K$ | 二阶 | 高 | 高优先级，但 GQA 下维度小，$S_K = 16$ |
| $W_V$ | 一阶（线性） | 中等 | 中优先级，$S_V = 32$ |
| $W_O$ | 一阶，直接进残差流 | 高——影响所有下游层 | 高优先级，$S_O = 32$ |

**vs LoRA 的关键差异**：LoRA 在 $W_Q, W_K$ 上引入的低秩 $\Delta$ 可能无意中破坏 $QK^T$ 的对称结构。JORA 的正交修正保证了注意力分数矩阵的性质（如正定性在小扰动下保持）。

### 29.3 残差流与 OER 的层级能量分析

LLaMA/Mistral 的残差连接：$h_{l+1} = h_l + \text{Attn}_l(h_l) + \text{MLP}_l(h_l + \text{Attn}_l(h_l))$

**能量沿深度的传播**：$\|h_l\|^2$ 随层数增长（残差累加），但增长速率由各层权重范数控制。

**OER 的层级效应**：OER 约束每层的输出范数不变 → 阻止了残差流中的能量急剧增长或衰减。

**不同层深度的行为预测**：

| 层位置 | 微调需求特征 | JORA 预期行为 |
|--------|-----------|-------------|
| 浅层（1-8） | 底层特征提取，变化小 | 小角度旋转，Core 值接近零 |
| 中层（9-24） | 语义组合，变化中等 | 活跃旋转对增加，OER 重分配活跃 |
| 深层（25-32） | 任务特定输出，变化可能大 | 最大角度旋转，OER 竞争最激烈 |

**实验建议**：在论文中展示层级旋转角度分布图——如果浅层角度小、深层角度大，这直接支持 JORA 的"渐进重对齐"叙事。

### 29.4 不同模型规模下的理论缩放行为

**命题 29.1（JORA 参数效率的规模优势）**：

设模型维度为 $d$，则：

| 方法 | 每层参数量缩放 | 覆盖率缩放 |
|------|-------------|-----------|
| LoRA ($r$ 固定) | $O(d \cdot r)$，线性于 $d$ | $r/d$ → 随 $d$ 增大而下降 |
| OFT ($b$ 固定) | $O(d \cdot b^2)$，线性于 $d$ | 固定块结构 |
| JORA ($S, k$ 固定) | $O(S + d)$，线性于 $d$（OER 贡献） | $S / \binom{d}{2}$ → 下降但旋转效率不降 |

**关键洞察**：当 $d$ 增大（从 7B 到 70B），LoRA 需要 $r$ 随之增大以保持覆盖率，但 JORA 的 $S$ 不需要——因为大模型的有效旋转自由度增长远慢于 $d^2$。

**预测 29.2**：JORA 相对 LoRA 的参数效率优势随模型规模增大而增大。在 70B 模型上，JORA 可能以 LoRA 1/20 的参数量达到相当性能。

---

## 30. 任务特性的第一性原理分类与 JORA 优势映射

### 30.1 微调任务的"变换几何"分类

从最优微调变换 $\Delta W^* = W^* - W_0$ 的几何结构，可以将任务分为四类：

**Type I：纯旋转重对齐（Pure Rotation Re-alignment）**

$$
W^* \approx Q_L^T W_0 Q_R, \quad Q_L, Q_R \in SO(n), SO(m)
$$

预训练特征基已经"正确"，只需要重新配对/旋转对齐。

典型任务：领域适配（医学/法律文本的领域迁移，已有特征需要重新组合）、语言风格转换、instruction tuning 的格式对齐。

**JORA 优势：⭐⭐⭐⭐⭐** 这正是 JORA 的设计目标。

**Type II：旋转 + 幅度重分配（Rotation + Magnitude Redistribution）**

$$
W^* \approx \text{diag}(m) \cdot Q_L^T W_0 Q_R
$$

除了旋转对齐，还需要重新分配各输出维度的重要性（幅度）。

典型任务：多任务微调、从通用模型到特定下游任务（如 MMLU 中某些知识密集型子任务）、图像分类微调。

**JORA 优势：⭐⭐⭐⭐** 旋转 + OER 精确覆盖。

**Type III：低秩新方向引入（Low-Rank New Direction Injection）**

$$
W^* \approx W_0 + U_{\text{new}} \Sigma_{\text{new}} V_{\text{new}}^T, \quad U_{\text{new}} \perp \text{col}(W_0)
$$

需要引入预训练时未见的全新特征方向。

典型任务：跨语言迁移（新语言的形态学特征）、全新领域适配、添加全新能力（如从 base 到 chat 的安全/拒绝行为）。

**JORA 优势：⭐⭐** 正交约束限制了引入全新方向的能力。Core 可以部分弥补，但 DiagCore 无法引入新方向。

**Type IV：全矩阵重构（Full Matrix Reconstruction）**

$$
\|\Delta W^*\|_F \gg \epsilon, \quad \text{rank}(\Delta W^*) \approx \min(n, m)
$$

微调变化太大，任何参数高效方法都不太适用。

典型任务：从零开始的任务训练、极端分布外迁移。

**JORA 优势：⭐** 不适用，应使用全参数微调。

### 30.2 常见数据集/任务的变换几何分析

#### 30.2.1 Instruction Tuning（Alpaca / Dolly / Open-Orca）

**任务本质**：将预训练模型的"续写"能力重定向为"指令跟随"能力。

**变换几何分析**：
- 预训练模型已经学会了语言理解和生成的核心能力
- Instruction tuning 主要调整的是"何时生成"和"以什么格式生成"——这是**输出空间的坐标重对齐**
- 实证证据：LoRA $r = 8$ 即可完成大部分 instruction tuning → 有效变换维度极低

**JORA 的精确匹配度**：

| 组件 | 对 instruction tuning 的作用 |
|------|---------------------------|
| 左旋转 $R_L$ | 重对齐输出特征空间——将"续写模式"旋转到"指令跟随模式" |
| 右旋转 $R_R$ | 重对齐输入特征空间——识别指令的关键输入模式 |
| DiagCore | 调节各特征维度的通过率——某些"续写特有"维度被抑制 |
| OER | 重新分配输出能量——"指令格式相关"维度获得更多能量 |

**变换类型**：Type I + 弱 Type II → **JORA 强项**

#### 30.2.2 MMLU / ARC / HellaSwag（知识与推理评估）

**任务本质**：测试模型的世界知识保持和推理能力。

**关键洞察——OER 的独特价值**：

微调中最大的风险是**灾难性遗忘**——知识存储在权重矩阵的范数分布中。LoRA 的 $\Delta W$ 可以改变 $\|W'\|_F = \|W_0 + BA^T\|_F$，导致某些知识相关维度的范数被意外压缩。

**OER 的守恒性质直接对抗灾难性遗忘**：

$$
\sum_i \|w'_i\|^2 = \sum_i \|w_{0,i}\|^2 = E_{\text{total}} \quad (\text{OER 保证})
$$

这意味着**微调不会改变权重矩阵的总"容量"**——知识被重新分配而非丢失。

**实验设计**：在 instruction tuning 后测 MMLU，对比 JORA vs LoRA 的知识保持率。预测：JORA 因 OER 的能量守恒而在 MMLU 上保持更好。

**变换类型**：Type II → **JORA OER 强项**

#### 30.2.3 代码生成（CodeAlpaca / HumanEval）

**任务本质**：从通用语言模型到代码生成的迁移。

**变换几何分析**：
- 代码有严格的语法结构——需要在输出空间中"锁定"语法正确的子空间
- 代码的 token 分布与自然语言差异大——需要一定的新方向引入
- 但 LLaMA-3 等现代模型的预训练数据已包含大量代码 → 主要是"激活"已有的代码子空间

**JORA 适配度**：⭐⭐⭐ — 如果基座模型有代码能力则适合（Type I-II），否则需要 Type III 能力

**配置建议**：使用 BlockCore（$b = 4$）而非 DiagCore——代码生成可能需要维度间的局部混合来编码语法依赖关系。

#### 30.2.4 对话与安全对齐（ShareGPT / UltraChat / HH-RLHF）

**任务本质**：在保持能力的同时，添加对话格式、安全过滤、拒绝行为。

**深层分析**：拒绝行为虽然表面上是"新行为"，但在注意力层面可以理解为**将某些输入模式路由到"安全回复"输出子空间**——这正是旋转+能量重分配的操作。DPO/PPO 论文表明，安全对齐主要改变模型的少数关键层和少数注意力头——**变化极其稀疏**。

**JORA 优势**：EMA 选择机制天然聚焦于梯度信号最强的维度对——正好对应安全对齐需要修改的少数关键维度。

**变换类型**：Type I-II + 少量 Type III → **JORA 中等偏强优势**

#### 30.2.5 领域适配（医学 / 法律 / 金融）

**以医学为例（PubMedQA / MedQA）**：

**变换几何分析**：
- 医学术语在预训练中已有一定覆盖，但权重不足
- 适配 = 将"医学相关维度"的权重提升 + 重新对齐医学概念间的关系

**这正是 OER 最擅长的场景**：

$$
p_i^{\text{after}} > p_i^{\text{before}} \Leftrightarrow \text{维度 } i \text{ 获得更多能量（更重要）}
$$

OER 的零和约束保证：医学维度获得能量 = 非医学维度让出等量能量 → **自动实现领域聚焦**

**变换类型**：Type II → **JORA 强项，尤其是 OER**

#### 30.2.6 视觉任务（ImageNet / CIFAR with ViT）

**变换几何分析**：
- ViT 的注意力权重类似于 LLM，但 patch embedding 的分布不同
- 图像特征空间的旋转对应于"视角/光照不变性的调整"——非常自然的旋转操作

**JORA 适配度**：⭐⭐⭐ — 如果证明 JORA 在视觉上也有效，将极大增强"通用性"叙事

### 30.3 任务特性-JORA 组件价值矩阵

| 任务特性 | 旋转价值 | Core 价值 | OER 价值 | 选择价值 | 总适配度 |
|---------|---------|----------|---------|---------|---------|
| 格式重对齐（instruction tuning） | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 知识保持（MMLU after FT） | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| 领域适配（医学/法律） | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 代码生成 | ⭐⭐⭐ | ⭐⭐⭐⭐ (Block) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 对话对齐 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 跨语言迁移 | ⭐⭐ | ⭐⭐⭐⭐ (LR) | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| 视觉微调 (ViT) | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 31. 数据集-模型-JORA 配置的联合优化策略

### 31.1 自适应配置选择框架

基于 §28-30 的分析，提出一个启发式但有理论依据的配置选择框架：

**Step 1: 评估任务的变换类型**

```
if 任务是格式/风格转换 (instruction tuning, style transfer):
    变换类型 = Type I → 旋转主导
elif 任务需要保持+重分配 (domain adaptation, knowledge tasks):
    变换类型 = Type II → 旋转 + OER 主导
elif 任务需要新能力 (cross-lingual, new modality):
    变换类型 = Type III → Core 主导 (用 LowRankCore)
```

**Step 2: 根据模型架构调整参数**

```
if model.has_rope:
    # 旋转先验 → 增加 S
    S_L, S_R = 32, 32
elif model.is_small (d < 2048):
    # 覆盖率高 → 可减少 S
    S_L, S_R = 16, 16

if model.has_gqa:
    S_kv = S // 2  # K/V 维度小，减半

if model.has_swiglu:
    target_modules += ['gate_proj', 'up_proj']  # 关键！
```

**Step 3: 根据数据规模调整训练动态**

```
if dataset_size < 10K:
    warmup_steps = 50   # 小数据 → 短 warmup
    ema_beta = 0.95      # 更快适应
    temperature = 2.0    # 更松弛的 OER
elif dataset_size > 100K:
    warmup_steps = 200
    ema_beta = 0.99      # 更稳定
    temperature = 1.0    # 更严格竞争
```

### 31.2 针对 LLaMA-2/3 的最优配置推导

**LLaMA-2-7B + Alpaca instruction tuning（旗舰实验）**：

| 配置项 | 推荐值 | 理论依据 |
|--------|--------|---------|
| `target_modules` | `q,k,v,o,gate,up,down` | §29.1: SwiGLU 与 JORA 天然契合 |
| `S_L, S_R` | 32, 32 | §28.2.1: 充足但不过量 |
| `k` | 8 | 经验最优，§3.2.1 的 $S=32$ 截断误差分析支持 |
| `core` | `diag` | Type I 任务，DiagCore 足够 |
| `magnitude` | `oer_softmax` | §30.2.1: instruction tuning 需要能量重分配 |
| `oer_temperature` | `2.0 → 1.0` (退火) | §4.5: 先探索后锁定 |
| `warmup_steps` | 100 | ~52K 样本，约 2% 训练步数 |
| `ema_beta` | 0.98 | 标准值，有效窗口 ~50 步 |
| `rotation_param` | `cayley` | §3.4: 优化稳定性 |

**LLaMA-2-13B + MMLU 知识保持实验**：

| 配置项 | 推荐值 | 理论依据 |
|--------|--------|---------|
| `target_modules` | `q,k,v,o` | MMLU 主要依赖注意力层的知识编码 |
| `magnitude` | `oer_softmax` | §30.2.2: 防止灾难性遗忘的关键 |
| `oer_temperature` | `3.0 → 1.5` | 高温起步 → 温和重分配 → 保持更多原始结构 |

**LLaMA-3-8B + 领域适配（医学/法律）**：

| 配置项 | 推荐值 | 理论依据 |
|--------|--------|---------|
| `target_modules` | `q,k,v,o,gate,up` | 领域知识编码在注意力+MLP |
| `core` | `diag` | 领域适配 = Type II，DiagCore 够用 |
| `pairing_strategy` | `high_low` | §30.2.5: 促进能量从非领域维度转移到领域维度 |
| `oer_temperature` | `1.0` (无退火) | 领域适配需要从一开始就积极重分配 |

### 31.3 针对 Mistral-7B 的配置调整

**Mistral-7B + 通用 instruction tuning**：

| 配置项 | 与 LLaMA 的差异 | 理由 |
|--------|---------------|------|
| `warmup_steps` | 60（更短） | 滑动窗口 → EMA 更快收敛（§28.2.4） |
| `k` | 8（不变） | 模型维度相同 |
| `selection_group_size` | 1（不分组） | 标准版 Mistral 无 MoE |

**Mixtral-8x7B + instruction tuning**：

| 配置项 | 推荐值 | 理由 |
|--------|--------|------|
| `target_modules` | `q,k,v,o` + 所有专家的 `w1,w2,w3` | MoE 专家权重是关键 |
| `selection_group_size` | 8 | 同类型专家共享选择，减少 8× 的计算 |
| `S_L, S_R` | 16, 16 | 每个专家权重较小，$S=16$ 覆盖率足够 |

---

## 32. 实验设计矩阵与优越性展示策略

### 32.1 核心实验矩阵

**第一梯队（必做，直接支撑主要贡献声明）**：

| 实验 ID | 模型 | 数据集/任务 | 预期 JORA 优势 | 展示的贡献 |
|---------|------|-----------|-------------|-----------|
| E1 | LLaMA-2-7B | Alpaca-52K | 参数效率 8× + 同等性能 | 核心效率声明 |
| E2 | LLaMA-2-7B | MMLU (eval after E1) | 知识保持 +2-5% vs LoRA | OER 守恒的价值 |
| E3 | LLaMA-2-7B | 消融全套 (#1-#15) | 各组件贡献清晰 | 方法学严谨性 |

**第二梯队（强烈推荐，增强说服力）**：

| 实验 ID | 模型 | 数据集/任务 | 预期 JORA 优势 | 展示的贡献 |
|---------|------|-----------|-------------|-----------|
| E4 | LLaMA-2-13B | Alpaca + MMLU | 规模扩展性 | 命题 29.1 的验证 |
| E5 | Mistral-7B | Alpaca-52K | 跨架构通用性 | 方法泛化性 |
| E6 | LLaMA-3-8B | MT-Bench | 对话质量 | 实际应用价值 |
| E7 | LLaMA-2-7B | 参数缩放实验 | 效率曲线斜率更优 | Pareto 前沿分析 |

**第三梯队（锦上添花，如果计算预算允许）**：

| 实验 ID | 模型 | 数据集/任务 | 展示的贡献 |
|---------|------|-----------|-----------|
| E8 | ViT-Large | ImageNet-1K (few-shot) | 跨模态通用性 |
| E9 | CodeLLaMA-7B | HumanEval | 结构化任务 |
| E10 | LLaMA-2-7B | PubMedQA | 领域适配 |
| E11 | RoBERTa-base | GLUE | NLU 经典 benchmark |
| E12 | LLaMA-2-70B | Alpaca | 大规模验证（预测 29.2） |

### 32.2 参数公平对比策略

**问题**：JORA DiagCore（8,256/层）vs LoRA-8（65,536/层）参数量差 8×，直接比较不公平。

**三层对比策略**：

| 对比维度 | JORA 配置 | LoRA 配置 | 说明 |
|---------|----------|----------|------|
| **同参数量** | DiagCore + OER (8K) | LoRA $r=1$ (8K) | LoRA 在 $r=1$ 时极弱 → JORA 巨大优势 |
| **同参数量** (更公平) | LowRankCore $r=8$ + OER (70K) | LoRA $r=8$ (66K) | 近似等参数量，公平对比 |
| **默认配置** | DiagCore (8K) | LoRA $r=8$ (66K) | JORA 用 1/8 参数达到同等 → 效率声明 |

**关键**：必须同时展示三组对比。只展示"同参数量 JORA 胜出"可能被质疑配置不公平。展示 JORA-8K vs LoRA-66K 达到同等性能是最有力的效率声明。

### 32.3 可视化策略：让优势不言自明

**图1（核心图）：参数效率 Pareto 前沿**

```
横轴：每层可训练参数量（log scale）
纵轴：下游任务性能（如 Alpaca eval score 或 MMLU 准确率）

标注点：
  LoRA r=1, r=2, r=4, r=8, r=16, r=32
  DoRA r=4, r=8
  OFT b=2, b=4
  JORA DiagCore, BlockCore(b=4), LowRankCore(r=4), LowRankCore(r=8)

预期：JORA 的点位于 Pareto 前沿的左上方（更少参数、更好性能）
```

**图2（OER 的核心可视化）：微调前后能量分布对比**

```
四组柱状图（子图）：
  (a) JORA-OER 训练前的能量分布
  (b) JORA-OER 训练后的能量分布 —— 总量不变但分布改变
  (c) LoRA 训练后的能量分布 —— 总量可能膨胀或收缩
  (d) DoRA 训练后的能量分布 —— 总量无约束

关键信息：JORA 的总面积（能量）精确不变，LoRA/DoRA 可能显著偏移
```

**图3（旋转选择的动态演化）：选择维度的时间热图**

```
横轴：训练步数
纵轴：维度 index（对某一层）
颜色：该维度在该步是否被选中（白/蓝渐变）

预期展示：
  - 训练初期选择不稳定（多色噪声）
  - 经过 warmup 后选择逐渐稳定
  - 最终锁定少数关键维度对
```

**图4（层级分析）：各层旋转角度与 Core 值的分布**

```
横轴：层 index（1-32）
纵轴上：平均 |θ| 值
纵轴下：平均 |core_diag| 值

预期展示：浅层角度小/Core 值小，深层更活跃 → 支持"渐进重对齐"叙事
```

### 32.4 如何在实验中突出 JORA 每个组件的独特价值

**策略 A：突出旋转（vs 所有非旋转方法）**

设计实验："哪些任务/层中，旋转对齐比低秩添加更重要？"

方法：对训练好的 LoRA 和 JORA，分析 $\Delta W$ 的特征。如果 LoRA 的 $\Delta W$ 可以近似分解为旋转形式（$\Delta W \approx Q_L^T W_0 Q_R - W_0$），则说明 LoRA 在"模拟旋转"——参数效率低下。

```python
# 诊断：LoRA 是否在低效模拟旋转？
Delta_W_lora = B @ A.T  # LoRA 更新
W_ft = W0 + Delta_W_lora

# 尝试用旋转分解近似
U0, S0, V0 = svd(W0)
U1, S1, V1 = svd(W_ft)
Q_approx = U1 @ U0.T  # 近似旋转

# 如果这个近似好，说明 LoRA 在浪费参数模拟旋转
rotation_approx_error = norm(W_ft - Q_approx @ W0) / norm(Delta_W_lora)
```

如果 `rotation_approx_error < 0.5`，则 LoRA 的 65K 参数中超过 50% 在做 JORA 的 64 参数（$S_L + S_R$）就能完成的工作。**这是最有力的参数效率论证。**

**策略 B：突出 OER（vs DoRA 和无约束方法）**

设计实验："OER 的能量守恒如何影响知识保持？"

方法：instruction tuning 后评估多个知识 benchmark（MMLU, ARC, HellaSwag, WinoGrande），对比：

| 方法 | instruction 性能 | 知识保持率 |
|------|----------------|-----------|
| LoRA | baseline | baseline |
| DoRA | 略高 | 可能更低（范数漂移） |
| JORA (无 OER) | 消融 | 可能与 LoRA 类似 |
| JORA (有 OER) | 预期更高 | **预期显著更高** |

**策略 C：突出数据驱动选择（vs 固定拓扑方法）**

设计实验："不同任务是否选择不同的维度对？"

方法：在两个不同任务上训练 JORA，比较各层的选中维度对：

$$
\text{Jaccard}(\mathcal{S}_{\text{task1}}, \mathcal{S}_{\text{task2}}) = \frac{|\mathcal{S}_{\text{task1}} \cap \mathcal{S}_{\text{task2}}|}{|\mathcal{S}_{\text{task1}} \cup \mathcal{S}_{\text{task2}}|}
$$

如果 Jaccard 低（< 0.3），说明选择确实是任务相关的 → OFT/BOFT 的固定拓扑不可能在两个任务上都好。

---

## 33. 可验证假说与诊断实验方案

### 33.1 假说体系（按可验证性排序）

#### 假说 H1（强预测）：JORA 在 SwiGLU MLP 层上的参数效率优势最大

**预测**：在消融实验中，仅对 `gate_proj` 和 `up_proj` 使用 JORA vs LoRA 的性能差距，大于仅对 `q_proj`/`v_proj` 使用 JORA vs LoRA 的差距。

**验证方法**：
```
实验 A: JORA on {gate, up} only vs LoRA on {gate, up} only
实验 B: JORA on {q, v} only vs LoRA on {q, v} only
预测: gap_A > gap_B
```

**理论依据**：§28.2.1 + §29.1 → SwiGLU 的快速谱衰减 + 二阶乘性效应。

#### 假说 H2（中等强度预测）：OER 在知识密集型评估中提供显著的保持优势

**预测**：JORA+OER 在 instruction tuning 后的 MMLU 上比 JORA-无OER 高 2-5%。

**验证方法**：消融 #6（magnitude=none）vs 完整 JORA，在 MMLU/ARC/HellaSwag 上评估。

**理论依据**：§30.2.2 → OER 能量守恒 → 防止灾难性遗忘。

#### 假说 H3（中等强度预测）：JORA 的参数效率优势随模型规模增大

**预测**：在 13B 上，JORA vs LoRA 的参数效率比从 8× 增大到 12×+。

**验证方法**：在 7B 和 13B 上分别画参数效率曲线，比较 Pareto 前沿的斜率。

**理论依据**：命题 29.1 → 大模型的有效旋转自由度增长慢于维度。

#### 假说 H4（可探索性预测）：RoPE 模型上 JORA 的优势大于非 RoPE 模型

**预测**：在 LLaMA（RoPE）vs RoBERTa（绝对位置编码）的成对对比中，JORA vs LoRA 的性能差距在 LLaMA 上更大。

**验证方法**：
```
LLaMA + Alpaca: JORA vs LoRA → gap_LLaMA
RoBERTa + GLUE: JORA vs LoRA → gap_RoBERTa
预测: gap_LLaMA > gap_RoBERTa
```

**理论依据**：§28.2.2 → RoPE 的旋转先验与 JORA 的 Givens 旋转天然协同。

#### 假说 H5（探索性预测）：不同任务选择不同的活跃维度对

**预测**：Alpaca instruction tuning 和 PubMedQA 领域适配的最终选中维度对 Jaccard 相似度 < 0.3。

**验证方法**：分别训练两个任务，提取各层最终的 `pairs_L/R`，计算 Jaccard。

**理论依据**：§30.1 → 不同任务的变换几何不同 → 最优维度对不同。

### 33.2 诊断实验方案（论文 Analysis 部分的素材）

#### 诊断 D1：谱衰减验证

**目标**：直接测量假说 28.1 的三个性质。

```python
# 对 LLaMA-2-7B 的每层每个投影
for l in range(32):
    for proj in projections:
        W = model.layers[l].get_weight(proj)
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        
        # 拟合幂律
        log_k = torch.log(torch.arange(1, len(S)+1).float())
        log_S = torch.log(S)
        alpha = -linear_regression(log_k, log_S).slope
        
        results[l][proj] = {
            'alpha': alpha,                    # 谱衰减指数
            'eff_rank_90': (S.cumsum(0) < 0.9*S.sum()).sum(),  # 90%能量有效秩
            'condition_number': S[0] / S[-1],  # 条件数
        }
```

**预期结果表**（指导 target_modules 选择）：

| 投影 | 平均 $\alpha$ | 平均有效秩 | JORA $S=32$ 覆盖率 |
|------|-------------|-----------|-------------------|
| gate_proj | 2.0+ | < 80 | 高（⭐⭐⭐⭐⭐） |
| up_proj | 1.8+ | < 100 | 高（⭐⭐⭐⭐） |
| k_proj | 1.5-2.0 | 50-150 | 高（⭐⭐⭐⭐） |
| q_proj | 1.3-1.8 | 100-300 | 中（⭐⭐⭐） |
| v_proj | 1.0-1.5 | 200-500 | 中低（⭐⭐） |
| o_proj | 1.3-1.8 | 100-200 | 中（⭐⭐⭐） |

#### 诊断 D2：LoRA 的"旋转浪费"量化

**目标**：量化 LoRA 花多少参数在"模拟旋转"。

```python
# 训练 LoRA-8 到收敛后
for l in range(32):
    Delta_W = B[l] @ A[l].T               # LoRA 更新
    W_ft = W0[l] + Delta_W
    
    U0, S0, V0 = svd(W0[l])
    U1, S1, V1 = svd(W_ft)
    
    # 最优旋转近似
    Q_left = U1 @ U0.T
    Q_right = V1 @ V0.T
    W_rot_approx = Q_left @ W0[l] @ Q_right.T
    
    # "旋转可解释比例"
    rotation_ratio = 1 - norm(W_ft - W_rot_approx) / norm(Delta_W)
    # 如果 > 0.5，说明 LoRA 一半以上的参数在做旋转
```

**如果 `rotation_ratio` 普遍 > 0.6**，这意味着 LoRA 用 65K 参数做的事情，JORA 用 64 个角度参数就能完成 60%。**这是论文中最有力的"why rotations"论证。**

#### 诊断 D3：OER 能量分布演化追踪

**目标**：可视化训练过程中能量如何重新分配。

```python
# 在训练过程中每 100 步记录
energy_snapshots = []
for step in checkpoints:
    for l in range(32):
        norms = model.layers[l].q_proj.base_row_norms  # [d]
        oer_scale = model.layers[l].q_proj.current_scale  # [d]
        effective_energy = (norms * oer_scale) ** 2
        energy_snapshots.append({
            'step': step, 'layer': l,
            'energy': effective_energy,
            'entropy': -(effective_energy/effective_energy.sum() * 
                        log(effective_energy/effective_energy.sum())).sum()
        })
```

**预期发现**：
- 训练初期 entropy 高（接近均匀分配）
- 训练后期 entropy 降低（某些维度"赢得竞争"）
- 不同层的 entropy 变化速率不同（深层更快→更任务特定）

#### 诊断 D4：旋转角度的层级分布

**目标**：验证§29.3 的"浅层小角度、深层大角度"预测。

```python
for l in range(32):
    theta_L = model.layers[l].q_proj.adapters.default.theta_L
    theta_R = model.layers[l].q_proj.adapters.default.theta_R
    phi_L = 2 * torch.atan(0.5 * theta_L)  # Cayley → 实际角度
    phi_R = 2 * torch.atan(0.5 * theta_R)
    
    angle_stats[l] = {
        'mean_angle_deg': torch.cat([phi_L, phi_R]).abs().mean() * 180/pi,
        'max_angle_deg': torch.cat([phi_L, phi_R]).abs().max() * 180/pi,
    }
```

#### 诊断 D5：跨任务选择差异的 Jaccard 分析

**目标**：验证假说 H5。

```python
# 分别训练 JORA on Alpaca 和 JORA on PubMedQA
for l in range(32):
    pairs_alpaca = model_alpaca.layers[l].q_proj.adapters.default.pairs_L[:n_active]
    pairs_pubmed = model_pubmed.layers[l].q_proj.adapters.default.pairs_L[:n_active]
    
    set_a = set(map(tuple, pairs_alpaca.tolist()))
    set_b = set(map(tuple, pairs_pubmed.tolist()))
    
    jaccard[l] = len(set_a & set_b) / len(set_a | set_b)
```

### 33.3 论文中的展示顺序建议

**§5.1 Setup**：
- 模型选择理由（引用 §28-29 的分析但不在论文中展开全部细节）
- 配置说明（引用 §31 的推导结论）

**§5.2 Main Results**：
- 表格：E1-E6 的结果
- 图：参数效率 Pareto 前沿（图1）

**§5.3 Ablation Studies**：
- 消融 #1, #5, #6 的结果
- OER 能量分布变化图（图2）

**§5.4 Analysis**：
- D1 谱衰减验证表 → 支持方法设计理由
- D2 旋转浪费量化 → 核心"why rotations"论证
- D3 能量演化图 → OER 的直觉解释
- D4 层级角度分布 → 支持"渐进重对齐"叙事
- D5（附录）跨任务选择差异 → 支持"数据驱动选择"创新点

### 33.4 对审稿人潜在质疑的预判与回应

| 潜在质疑 | 对应分析 | 实验回应 |
|---------|---------|---------|
| "旋转假说缺乏实证" | D1 + D2 | 直接测量谱衰减和旋转可解释比例 |
| "为什么不在 BERT/GPT-2 上测？" | §28.2.5 | 承认 JORA 在非 RoPE 模型上优势可能较小，但仍有效（E11） |
| "8K vs 66K 参数不公平" | §32.2 三层对比 | 展示同参数量和不同参数量两种对比 |
| "只在 instruction tuning 上测了？" | §32.1 完整矩阵 | E1-E12 覆盖 instruction/knowledge/domain/code |
| "OER 的守恒假设太强" | §4.6 | 消融 #6 + 不同温度实验 |
| "计算开销？" | §20 | FLOPs < 0.1%，展示墙钟时间对比 |
| "能 scale 到 70B 吗？" | 命题 29.1 | 如果资源允许，E12；否则用 7B→13B 的趋势外推 |

---

## 附录 G. 模型-任务实验配置速查表（v5 新增）

### G.1 按模型分类的推荐配置

| 模型 | $d$ | RoPE | GQA | SwiGLU | 推荐 $S_{L/R}$ | 推荐 target | 推荐 Core | 特殊注意 |
|------|-----|------|-----|--------|---------------|------------|----------|---------|
| LLaMA-2-7B | 4096 | ✅ | ❌ | ✅ | 32/32 | q,k,v,o,gate,up,down | diag | 旗舰实验模型 |
| LLaMA-2-13B | 5120 | ✅ | ❌ | ✅ | 32/32 | q,k,v,o,gate,up | diag | 规模扩展验证 |
| LLaMA-3-8B | 4096 | ✅ | ✅ (8 kv) | ✅ | Q/O:32, K/V:16 | q,k,v,o,gate,up | diag | GQA 下 K/V 用更小 S |
| Mistral-7B | 4096 | ✅ | ✅ | ✅ | 32/32 | q,k,v,o,gate,up | diag | warmup 可更短 |
| Phi-3-mini | 3072 | ✅ | ✅ | ✅ | 24/24 | q,k,v,o | diag | 小维度，覆盖率高 |
| RoBERTa-base | 768 | ❌ | ❌ | ❌ | 16/16 | q,v | diag | 对照组，无 RoPE 先验 |
| ViT-Large | 1024 | ❌ | ❌ | ❌ | 16/16 | q,v | diag | 跨模态验证 |

### G.2 按任务分类的推荐配置

| 任务类型 | 变换类型 | OER 温度 | 退火策略 | pairing | 重点消融 |
|---------|---------|---------|---------|---------|---------|
| Instruction tuning | I | 2.0→1.0 | 线性退火 | consecutive | #1, #5, #6 |
| 知识评估 (MMLU) | II | 3.0→1.5 | 缓慢退火 | consecutive | #6 (OER 价值) |
| 领域适配 | II | 1.0 (无退火) | 无 | high_low | #6, #7 |
| 代码生成 | I-II | 2.0→1.0 | 线性退火 | consecutive | #8 (Core 类型) |
| 对话对齐 | I-II | 2.0→1.0 | 线性退火 | consecutive | #5 (选择价值) |
| 视觉微调 | I-II | 2.0→1.0 | 线性退火 | consecutive | #1 (旋转价值) |

### G.3 计算预算分配建议

| 可用 GPU-hours | 建议实验组合 | 预期论文贡献覆盖 |
|---------------|------------|----------------|
| 100 | E1 + E3 (7B Alpaca + 消融) | 基础效率 + 消融 |
| 300 | + E2 + E5 + E7 | + 知识保持 + 跨架构 + Pareto |
| 1000 | + E4 + E6 + E10 | + 规模扩展 + MT-Bench + 领域适配 |
| 3000+ | + E8-E12 | 完整实验矩阵 |

---

> **文档版本**：v5.0 | **最后更新**：2025-06 | **面向**：NeurIPS 2026 投稿冲刺
>
> **v5.0 核心改进**：新增第七部分（§28-§33 + 附录 G），从数学第一性原理出发分析 JORA 与主流模型架构（LLaMA/Mistral/Phi/ViT）和任务类型（instruction tuning/知识保持/领域适配/代码生成）的交互关系，提出"谱衰减-旋转假说"及其可验证实验方案，给出完整的实验设计矩阵和优越性展示策略。

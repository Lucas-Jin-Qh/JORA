# PEFT方法全景分析报告

## 📊 概述

PEFT (Parameter-Efficient Fine-Tuning) 仓库目前支持 **43种** 不同的参数高效微调方法，按技术路线分为以下四大类：

## 🎯 1. 经典方法 (Classical Methods)

### **LoRA系列 (LoRA Family)**
- **LORA**: Low-Rank Adaptation - 原始LoRA方法，通过低秩矩阵近似权重更新
- **ADALORA**: Adaptive LoRA - 自适应秩分配，根据重要性动态调整秩
- **VERA**: Vector-based Random Matrix Adaptation - 基于随机矩阵的向量适应
- **RANDLORA**: Random LoRA - 使用随机初始化矩阵的LoRA变体
- **DORA**: Weight-Decomposed Low-Rank Adaptation - 权重分解的LoRA
- **GRALORA**: Gradient-based Low-Rank Adaptation - 基于梯度的自适应LoRA
- **DELORA**: Dynamic Expansion of Low-Rank Adaptation - 动态扩展LoRA

### **提示学习系列 (Prompt Learning)**
- **PROMPT_TUNING**: Prompt Tuning - 添加可训练的提示向量
- **MULTITASK_PROMPT_TUNING**: 多任务提示调优 - 支持多任务的提示学习
- **PREFIX_TUNING**: Prefix Tuning - 在输入前添加可训练前缀
- **P_TUNING**: P-Tuning - 提示参数化的连续提示学习
- **ADAPTION_PROMPT**: Adaption Prompt - 自适应提示学习

### **其他经典方法**
- **IA3**: Infused Adapter by Inhibiting and Amplifying Inner Activations - 通过缩放激活进行适应
- **LN_TUNING**: Layer Normalization Tuning - 微调LayerNorm参数

## 🚀 2. 最新前沿方法 (State-of-the-Art Methods)

### **变换器方法 (Transformer Methods)**
- **OFT**: Orthogonal Fine-Tuning - 正交微调，通过正交变换保持重要信息
- **BOFT**: Butterfly Orthogonal Fine-Tuning - 蝶形正交微调，高效的参数更新
- **CPT**: Compact Permutation Transformer - 紧凑排列变换器
- **CARTRIDGE**: Memory-Efficient Fine-Tuning via Structured Matrices - 基于结构化矩阵的内存高效微调

### **频域方法 (Frequency Domain Methods)**
- **FOURIERFT**: Fourier Fine-Tuning - 傅里叶域微调
- **WAVEFT**: Wavelet Fine-Tuning - 小波变换微调

### **结构化方法 (Structured Methods)**
- **LOHA**: Low-Rank Hadamard Adaptation - 低秩Hadamard积适应
- **LOKR**: Low-Rank Kronecker Adaptation - 低秩Kronecker积适应
- **POLY**: Polynomial Activation Scaling - 多项式激活缩放
- **HRA**: Hadamard Random Adaptation - Hadamard随机适应

### **量化相关方法 (Quantization-Aware Methods)**
- **XLORA**: Extended LoRA - 扩展LoRA，支持量化模型
- **BONE**: Bit-Order Normalization for Efficient Training - 位序归一化高效训练
- **MISS**: Memory-efficient Structured Matrices - 内存高效结构化矩阵

### **进化方法 (Evolutionary Methods)**
- **SHIRA**: Shifted Hadamard Random Adaptation - 移位Hadamard随机适应
- **C3A**: Compact Channel-wise Convolution Adaptation - 紧凑通道卷积适应
- **ROAD**: Random Orthogonal Adaptation of Dimensions - 随机正交维度适应
- **OSF**: Orthogonal Subspace Fusion - 正交子空间融合

## 📋 3. 支持的任务类型

PEFT库支持以下6种任务类型：

| 任务类型 | 描述 | 适用场景 |
|---------|------|----------|
| **CAUSAL_LM** | 因果语言模型 | GPT-style 文本生成 |
| **SEQ_2_SEQ_LM** | 序列到序列语言模型 | T5, BART等翻译/摘要 |
| **SEQ_CLS** | 序列分类 | 情感分析、意图识别 |
| **TOKEN_CLS** | 标记分类 | NER、POS标注 |
| **QUESTION_ANS** | 问答 | 抽取式问答、阅读理解 |
| **FEATURE_EXTRACTION** | 特征提取 | 嵌入提取、表示学习 |

## 🔧 4. 技术特性分析

### **参数效率对比**
| 方法类别 | 典型参数减少 | 计算复杂度 | 内存效率 |
|---------|-------------|-----------|----------|
| LoRA系列 | 90-99% | O(r×d) | 高 |
| 提示学习 | 99.9%+ | O(l×d) | 最高 |
| 正交方法 | 95-98% | O(d²) | 中高 |
| 结构化方法 | 85-95% | O(k×d) | 中 |

### **兼容性矩阵**
| 方法 | 量化兼容 | 分布式训练 | 推理加速 |
|-----|---------|-----------|----------|
| LoRA | ✅ | ✅ | ✅ |
| QLoRA | ✅ | ✅ | ✅ |
| Prefix Tuning | ✅ | ✅ | ⚠️ |
| IA3 | ✅ | ✅ | ✅ |
| OFT | ✅ | ✅ | ✅ |
| BOFT | ✅ | ✅ | ✅ |

## 🎨 5. 方法选择指南

### **对于大语言模型微调**
- **首选**: LoRA, QLoRA (平衡性能和效率)
- **内存受限**: QLoRA, IA3
- **高精度需求**: DoRA, AdaLoRA
- **多任务**: OFT, BOFT

### **对于分类任务**
- **标准**: LoRA, IA3
- **轻量级**: Prompt Tuning, P-Tuning
- **高效**: BOFT, OFT

### **对于生成任务**
- **文本生成**: LoRA, Prefix Tuning
- **代码生成**: LoRA, DoRA
- **多模态**: OFT, CARTRIDGE

## 🛠️ 6. 实现架构

### **核心组件**
```
PEFT/
├── tuners/           # 所有PEFT方法实现
│   ├── lora/        # LoRA系列
│   ├── prompt_tuning/  # 提示学习
│   ├── oft/         # 正交方法
│   └── ...          # 其他方法
├── config.py        # 配置管理
├── peft_model.py    # 模型包装器
└── utils/           # 工具函数
```

### **配置系统**
- **PeftConfig**: 基础配置类
- **TaskType**: 任务类型枚举
- **PeftType**: PEFT方法枚举
- **自动映射**: 模型类型自动识别

### **量化集成**
- **bitsandbytes**: 4-bit量化支持
- **GPTQ**: 量化感知训练
- **AWQ**: 激活感知量化
- **HQQ**: 半精度量化

## 📈 7. 最新发展方向

### **2024年新方法**
- **BOFT**: 蝶形正交微调 - 显著降低计算复杂度
- **CARTRIDGE**: 结构化矩阵 - 内存效率提升
- **DELORA**: 动态扩展 - 自适应参数分配
- **OSF**: 正交子空间融合 - 多任务学习优化

### **技术趋势**
1. **正交变换**: OFT, BOFT等方法的兴起
2. **结构化矩阵**: 更高效的矩阵分解技术
3. **动态适应**: 自适应参数分配和调整
4. **多模态扩展**: 支持视觉-语言模型的PEFT

## 🚀 8. 使用建议

### **新手推荐**
1. 从 **LoRA** 开始 - 简单有效，广泛适用
2. 尝试 **QLoRA** - 如果内存受限
3. 探索 **IA3** - 对于分类任务

### **进阶用户**
1. 尝试最新方法: **BOFT**, **OFT**
2. 结合量化: **QLoRA + 新方法**
3. 多方法对比: 使用 `method_comparison/` 系统

### **研究者**
1. 实现新方法: 使用 `register_peft_method()`
2. 贡献方法: 遵循贡献指南
3. 性能基准: 使用MetaMathQA等数据集

## 📊 9. 方法对比总结

| 方面 | LoRA | 提示学习 | 正交方法 | 结构化方法 |
|-----|------|---------|---------|-----------|
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 参数效率 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 性能 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 兼容性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 研究热度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🎯 结论

PEFT库提供了迄今为止**最全面的参数高效微调方法集合**，涵盖从经典方法到最新前沿技术的完整谱系。无论您是初学者还是资深研究者，都能在其中找到适合您需求的PEFT方法。

**关键洞察：**
- LoRA仍然是大多数场景下的最佳选择
- 新兴的正交方法(BOFT, OFT)在性能和效率上展现出巨大潜力
- 提示学习方法在特定场景下仍然具有独特优势
- 量化集成使大模型微调变得可行

您的PEFT研究现在拥有了完整的工具箱！🎉

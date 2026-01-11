# PEFT-JORA 数据集使用指南

## 📁 数据集存储位置
所有数据集已下载到: `/home/jqh/Workshop/JORA/datasets`

## 🔗 下载配置
- **镜像源**: `https://hf-mirror.com` (用于加速下载)
- **缓存目录**: `/home/jqh/Workshop/JORA/datasets`

## 📊 已下载的数据集

### 基础数据集
| 数据集 | 任务类型 | 大小 | 分割 |
|--------|----------|------|------|
| **GLUE SST-2** | 情感分析 | 67K | train/val/test |
| **HellaSwag** | 常识推理 | 40K | train/val/test |
| **GSM8K** | 数学推理 | 7.5K | train/test |
| **ARC-Challenge** | 科学问答 | 1.1K | train/val/test |
| **Alpaca-Cleaned** | 指令微调 | 51.8K | train |

### MMLU 数据集 (多学科知识评估)
| 子集 | 领域 | 测试样本 | 分割 |
|------|------|----------|------|
| college_biology | 大学生物学 | 144 | test/val/dev |
| college_chemistry | 大学化学 | 100 | test/val/dev |
| college_computer_science | 大学计算机科学 | 100 | test/val/dev |
| college_mathematics | 大学数学 | 100 | test/val/dev |
| college_physics | 大学物理 | 102 | test/val/dev |
| electrical_engineering | 电气工程 | 145 | test/val/dev |
| machine_learning | 机器学习 | 112 | test/val/dev |

## 🚀 使用方法

### 方法1: 使用配置文件 (推荐)

```python
from dataset_config import load_peft_dataset, list_available_datasets

# 查看可用数据集
list_available_datasets()

# 加载数据集
ds = load_peft_dataset("glue_sst2", split="train")
print(f"数据集大小: {len(ds)}")
print(f"样例: {ds[0]}")
```

### 方法2: 直接使用 HuggingFace datasets

```python
import os
from datasets import load_dataset

# 设置缓存目录
os.environ['HF_DATASETS_CACHE'] = '/home/jqh/Workshop/JORA/datasets'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 可选，用于加速

# 加载数据集
ds = load_dataset('glue', 'sst2', split='train',
                  cache_dir='/home/jqh/Workshop/JORA/datasets')
```

## 📈 数据集应用场景

### PEFT 方法评估
- **GLUE SST-2**: 基础分类任务，测试PEFT在小数据集上的效果
- **HellaSwag**: 复杂推理任务，评估PEFT在上下文理解上的性能
- **GSM8K**: 数学推理，测试PEFT在逻辑推理上的能力

### 指令微调
- **Alpaca-Cleaned**: 指令跟随微调数据集，包含51.8K个指令-响应对，用于训练对话式AI助手

### 多学科知识评估
- **MMLU系列**: 专业领域知识评估，适合测试PEFT在领域适应性上的表现
- **ARC-Challenge**: 科学推理，评估PEFT在科学问题解决上的能力

## ⚙️ 环境配置

### 激活环境
```bash
conda activate peft-jora
```

### 设置环境变量 (可选，用于加速)
```bash
export HF_DATASETS_CACHE=/home/jqh/Workshop/JORA/datasets
export HF_ENDPOINT=https://hf-mirror.com
```

## 📝 添加新数据集

如需下载其他数据集，可以使用以下命令：

```bash
conda activate peft-jora
export HF_DATASETS_CACHE=/home/jqh/Workshop/JORA/datasets
export HF_ENDPOINT=https://hf-mirror.com

# 下载新数据集
python -c "from datasets import load_dataset; ds = load_dataset('dataset_name', split='train')"
```

## 🔍 故障排除

### 网络问题
如果下载速度慢，可以使用镜像源：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 存储空间
查看数据集占用空间：
```bash
du -sh /home/jqh/Workshop/JORA/datasets/
```

### 清理缓存
如需清理不需要的数据集：
```bash
rm -rf /home/jqh/Workshop/JORA/datasets/dataset_name/
```

## 🎯 下一步建议

1. **模型选择**: 基于这些数据集选择合适的预训练模型
2. **PEFT方法**: 尝试不同的PEFT方法 (LoRA, Prefix Tuning, P-Tuning等)
3. **评估基准**: 使用这些数据集建立性能基准
4. **实验记录**: 使用WandB或TensorBoard记录实验结果

## 📞 技术支持

如遇问题，请检查：
1. conda环境是否正确激活 (`peft-jora`)
2. PyTorch和CUDA版本是否匹配
3. 网络连接是否正常
4. 存储空间是否充足

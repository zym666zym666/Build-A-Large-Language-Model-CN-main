# GPT from Scratch

从零实现GPT模型，基于《LLMs from Scratch》一书。

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（可选，用于加速训练）

## 安装依赖

```bash
pip install torch torchvision tiktoken pandas
```

## 运行

### 方式一：主入口（推荐）

```bash
cd code
python run.py
```

**运行结果**：
- 自动下载示例数据集（the-verdict.txt）
- 初始化GPT-124M模型
- 训练5个epoch，每50步打印训练/验证损失
- 训练完成后生成100个新token的文本

### 方式二：各模块导入使用

大部分章节文件为模块定义，无独立入口，需被导入使用：

```python
# 示例：导入并使用GPT模型
from previous_chapters import GPTModel, GPT_CONFIG_124M

model = GPTModel(GPT_CONFIG_124M)
```

## 项目结构

```
code/
├── run.py                    # 主入口：完整训练流程
├── previous_chapters.py      # 共享模块：整合所有核心类和函数
├── chapter02/
│   └── chapter02.py          # 分词器与数据集（SimpleTokenizer, GPTDatasetV1）
├── chapter03/
│   └── chapter03.py          # 注意力机制（SelfAttention, MultiHeadAttention）
├── chapter04/
│   └── chapter04.py          # GPT模型架构（TransformerBlock, GPTModel）
├── chapter05/
│   └── chapter05.py          # 预训练（训练循环、损失计算、文本生成）
├── chapter06/
│   └── chapter06.py          # 指令微调（InstructionDataset, 指令格式处理）
├── chapter07/
│   └── chapter07.py          # 分布式训练（DDP, 多GPU训练）
├── appendixA/
│   └── appendixA.py          # 学习率调度与梯度裁剪（warmup, 余弦退火）
├── appendixD/
│   └── appendixD.py          # LoRA微调（低秩适应，参数高效微调）
└── appendixE/
    └── appendixE.py          # 分类微调（垃圾邮件分类，GPT-2权重加载）
```

## 模块说明

| 文件 | 核心内容 | 可直接运行 |
|------|---------|-----------|
| `run.py` | 完整训练流程（下载→训练→生成） | ✅ |
| `previous_chapters.py` | 共享依赖模块，整合所有类和函数 | ❌ |
| `chapter02.py` | 分词器、数据集类、数据加载器 | ❌ |
| `chapter03.py` | 自注意力、因果注意力、多头注意力 | ❌ |
| `chapter04.py` | 层归一化、GELU、Transformer块、GPT模型 | ❌ |
| `chapter05.py` | 预训练循环、损失评估、贪婪解码生成 | ❌ |
| `chapter06.py` | 指令微调数据集、自定义批量处理 | ❌ |
| `chapter07.py` | DDP分布式训练、多GPU环境配置 | ❌ |
| `appendixA.py` | 学习率warmup、余弦退火、梯度裁剪 | ❌ |
| `appendixD.py` | LoRA层、参数高效微调 | ❌ |
| `appendixE.py` | GPT-2权重加载、分类微调、垃圾邮件检测 | ❌ |

## 模型配置

默认使用GPT-124M配置：

- vocab_size: 50257
- context_length: 256（训练时）/ 1024（微调时）
- emb_dim: 768
- n_heads: 12
- n_layers: 12
- drop_rate: 0.1

## 注意事项

1. 首次运行会自动下载数据集（约20KB）和GPT-2预训练权重（约500MB）
2. GPT-124M模型训练需要约4GB GPU显存或8GB以上内存
3. 训练数据保存在当前目录下（the-verdict.txt）
4. 预训练权重保存在当前目录下（gpt2/目录）
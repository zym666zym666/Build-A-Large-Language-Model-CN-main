# GPT from Scratch

从零实现GPT模型，基于《LLMs from Scratch》一书。

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（可选，用于加速训练）

## 安装依赖

```bash
pip install torch torchvision tiktoken pandas transformers peft
```

## 运行

### 1. LoRA 微调训练（run.py）

```bash
cd code
python run.py
```

**流程**：
1. 下载《判决》小说数据集（~20KB）
2. 加载 GPT-2 预训练模型（124M 参数）
3. 配置并应用 LoRA（只训练 ~1% 参数）
4. 微调前生成测试
5. LoRA 微调训练（8 epoch）
6. 微调后生成测试
7. 保存模型到 `./gpt2-lora-finetuned/`

**LoRA 配置**：
- r=16, lora_alpha=32
- 目标模块：c_attn, c_proj
- 学习率：3e-4，余弦退火，warmup 10 步

### 2. 交互式文本生成（generate.py）

```bash
cd code
python generate.py
```

**前提**：需先运行 `run.py` 完成微调，生成 `./gpt2-lora-finetuned/` 目录。

**功能**：
- 加载基础模型 + LoRA 适配器
- 4 种生成风格：creative / balanced / conservative / fantasy
- 交互式输入提示词，实时生成文本
- 输入 `/风格名` 切换风格（如 `/creative`）
- 输入 `quit` 退出

**风格参数**：

| 风格 | temperature | top_k | top_p | repetition_penalty |
|------|------------|-------|-------|-------------------|
| creative | 0.95 | 80 | 0.95 | 1.15 |
| balanced | 0.85 | 50 | 0.92 | 1.10 |
| conservative | 0.70 | 30 | 0.90 | 1.05 |
| fantasy | 0.90 | 60 | 0.93 | 1.20 |

### 3. 各模块导入使用

大部分章节文件为模块定义，无独立入口，需被导入使用：

```python
from previous_chapters import GPTModel, GPT_CONFIG_124M
model = GPTModel(GPT_CONFIG_124M)
```

## 项目结构

```
code/
├── run.py                    # LoRA 微调训练主入口
├── generate.py               # 交互式文本生成（需先完成微调）
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
| `run.py` | LoRA 微调训练（下载→训练→生成） | ✅ |
| `generate.py` | 交互式文本生成（加载微调模型） | ✅ |
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

## 注意事项

1. 首次运行 `run.py` 会自动下载 GPT-2 预训练权重（约500MB）和数据集（约20KB）
2. 微调模型需要约 4GB GPU 显存（LoRA 大幅降低显存需求）
3. 微调完成后模型保存在 `./gpt2-lora-finetuned/` 目录
4. 运行 `generate.py` 前需先完成微调

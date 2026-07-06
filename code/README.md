# GPT from Scratch

从零实现GPT模型，基于《LLMs from Scratch》一书。

## 环境要求

- Python 3.8+
- PyTorch 2.0+

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

### 方式二：各章节独立运行

```bash
python chapter02/chapter02.py    # 分词和数据准备
python chapter03/chapter03.py    # 注意力机制
python chapter04/chapter04.py    # GPT模型架构
python chapter05/chapter05.py    # 预训练
python chapter06/chapter06.py    # 分类微调
python chapter07/chapter07.py    # 指令微调
python appendixA/appendixA.py    # PyTorch基础与DDP
python appendixD/appendixD.py    # 学习率调度与梯度裁剪
python appendixE/appendixE.py    # LoRA微调
```

## 项目结构

```
code/
├── run.py                # 主入口
├── previous_chapters.py  # 共享模块
├── chapter02/            # 分词和数据准备
├── chapter03/            # 注意力机制
├── chapter04/            # GPT模型架构
├── chapter05/            # 预训练
├── chapter06/            # 分类微调
├── chapter07/            # 指令微调
├── appendixA/            # PyTorch基础与DDP
├── appendixD/            # 学习率调度与梯度裁剪
└── appendixE/            # LoRA微调
```

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

```bash
cd code
python run.py
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

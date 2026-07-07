"""
LoRA 微调 GPT-2 教程
使用《判决》小说微调 GPT-2，使其学习特定的文学风格

流程：
1. 加载数据
2. 加载 GPT-2 预训练模型
3. 配置并应用 LoRA
4. 微调前测试
5. LoRA 微调训练
6. 微调后测试
7. 保存模型
"""

import torch
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ========== 配置环境 ==========
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ========== 导入依赖 ==========
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from chapter05.chapter05 import download_verdict


# ==================== 1. 数据集类 ====================

class TextDataset(Dataset):
    """
    自定义文本数据集
    将文本转换为模型可接受的格式
    """
    
    def __init__(self, texts, tokenizer, max_length=512):
        """
        初始化数据集
        
        Args:
            texts: 文本片段列表
            tokenizer: GPT-2 分词器
            max_length: 最大序列长度
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Returns:
            dict: {
                'input_ids': tokenized text,
                'attention_mask': attention mask,
                'labels': input_ids (for causal LM)
            }
        """
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


# ==================== 2. 工具函数 ====================

def split_text(text, chunk_size=512, overlap=128):
    """
    将长文本分割成重叠的片段
    
    Args:
        text: 输入文本
        chunk_size: 每个片段的长度
        overlap: 片段之间的重叠长度（让模型学到更多上下文）
    
    Returns:
        list: 文本片段列表
    """
    chunks = []
    step = chunk_size - overlap
    
    # 滑动窗口分割
    for i in range(0, max(1, len(text) - chunk_size + 1), step):
        chunk = text[i:i+chunk_size]
        if len(chunk) > 100:  # 过滤太短的片段
            chunks.append(chunk)
    
    # 确保最后也包含
    if len(text) > chunk_size:
        last_chunk = text[-chunk_size:]
        if len(last_chunk) > 100 and last_chunk not in chunks:
            chunks.append(last_chunk)
            
    return chunks


def print_separator(title=None, char='=', length=60):
    """打印分隔线"""
    if title:
        print(f"\n{char * length}")
        print(f"{title.center(length)}")
        print(f"{char * length}")
    else:
        print(char * length)


def print_config(config_dict):
    """打印配置信息"""
    for key, value in config_dict.items():
        print(f"  {key}: {value}")


# ==================== 3. 训练函数 ====================

def load_data():
    """加载并预处理数据"""
    print_separator("步骤 1: 加载数据")
    
    txt = download_verdict()
    print(f"📄 数据集大小: {len(txt):,} 字符")
    
    # 划分训练集和验证集 (90/10)
    split_idx = int(0.9 * len(txt))
    train_txt = txt[:split_idx]
    val_txt = txt[split_idx:]
    
    # 分割成片段
    train_chunks = split_text(train_txt, chunk_size=512, overlap=128)
    val_chunks = split_text(val_txt, chunk_size=512, overlap=128)
    
    print(f"📊 训练片段: {len(train_chunks)}")
    print(f"📊 验证片段: {len(val_chunks)}")
    print(f"📊 总训练 tokens: ~{len(train_chunks) * 512:,}")
    
    return train_chunks, val_chunks


def load_model():
    """加载 GPT-2 模型和分词器"""
    print_separator("步骤 2: 加载 GPT-2 模型")
    
    model_name = "gpt2"
    
    # 加载分词器
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ 分词器加载完成 (词汇量: {tokenizer.vocab_size})")
    
    # 加载基础模型
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model_size = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型加载完成 (参数量: {model_size/1e6:.1f}M)")
    
    return model, tokenizer


def apply_lora(model):
    """配置并应用 LoRA"""
    print_separator("步骤 3: 配置 LoRA")
    
    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                              # 低秩维度（越大可训练参数越多）
        lora_alpha=32,                     # 缩放因子
        lora_dropout=0.1,                  # Dropout 防止过拟合
        target_modules=["c_attn", "c_proj"],  # 微调的目标模块
        bias="none",
    )
    
    print("🔧 LoRA 配置:")
    print(f"  r={lora_config.r}")
    print(f"  lora_alpha={lora_config.lora_alpha}")
    print(f"  target_modules={lora_config.target_modules}")
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 参数统计:")
    print(f"  总参数: {total_params/1e6:.1f}M")
    print(f"  可训练参数: {trainable_params/1e6:.2f}M")
    print(f"  占比: {100 * trainable_params / total_params:.2f}%")
    
    return model


def test_before_training(model, tokenizer, device):
    """微调前测试"""
    print_separator("步骤 4: 微调前测试")
    
    prompt = "Every effort moves you"
    print(f"📝 提示词: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n📄 生成结果:\n{generated}")
    print("\n⚠️  注意: 这是通用 GPT-2 的风格，还没有学习《判决》")


def setup_trainer(model, train_dataset, val_dataset, tokenizer):
    """配置训练器"""
    print_separator("步骤 5: 配置训练器")
    
    # 训练参数
    training_args = TrainingArguments(
        # 输出设置
        output_dir="./gpt2-lora-finetuned",
        
        # 训练轮数
        num_train_epochs=8,
        
        # 批次大小
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        
        # 学习率
        learning_rate=3e-4,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        
        # 正则化
        weight_decay=0.01,
        
        # 日志与保存
        logging_steps=5,
        save_steps=50,
        eval_steps=20,
        eval_strategy="steps",
        save_total_limit=2,
        
        # 其他
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    print("🔧 训练参数:")
    print_config({
        "学习率": training_args.learning_rate,
        "训练轮数": training_args.num_train_epochs,
        "批次大小": training_args.per_device_train_batch_size,
        "warmup steps": training_args.warmup_steps,
        "weight decay": training_args.weight_decay,
    })
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    return trainer


def test_after_training(model, tokenizer, device):
    """微调后测试"""
    print_separator("步骤 7: 微调后测试")
    
    test_prompts = [
        "Every effort moves you",
        "She turned to him and",
        "The truth was that"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'─'*50}")
        print(f"📝 提示词: {prompt}")
        print(f"{'─'*50}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.85,
                top_k=50,
                top_p=0.92,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n📄 生成结果:\n{generated}")


def print_summary(trainer):
    """打印训练总结"""
    print_separator("🎉 训练完成!")
    
    # 获取训练信息
    try:
        logs = trainer.state.log_history
        final_train_loss = None
        final_eval_loss = None
        
        for log in logs:
            if 'loss' in log and 'eval_loss' not in log:
                final_train_loss = log['loss']
            if 'eval_loss' in log:
                final_eval_loss = log['eval_loss']
        
        print(f"\n📊 训练结果:")
        print(f"  最终训练损失: {final_train_loss:.4f}" if final_train_loss else "  最终训练损失: N/A")
        print(f"  最终验证损失: {final_eval_loss:.4f}" if final_eval_loss else "  最终验证损失: N/A")
    except:
        pass
    
    print(f"\n💾 模型已保存: ./gpt2-lora-finetuned/")
    print(f"\n💡 使用 'generate.py' 加载模型并生成文本")


# ==================== 4. 主函数 ====================

def main():
    """主函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 使用设备: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    start_time = time.time()
    
    try:
        # 1. 加载数据
        train_chunks, val_chunks = load_data()
        
        # 2. 加载模型
        model, tokenizer = load_model()
        model.to(device)
        
        # 3. 应用 LoRA
        model = apply_lora(model)
        
        # 4. 微调前测试
        test_before_training(model, tokenizer, device)
        
        # 5. 准备数据集
        print_separator("准备数据集")
        train_dataset = TextDataset(train_chunks, tokenizer, max_length=512)
        val_dataset = TextDataset(val_chunks, tokenizer, max_length=512)
        
        # 6. 配置训练器并开始训练
        trainer = setup_trainer(model, train_dataset, val_dataset, tokenizer)
        
        print_separator("步骤 6: 开始 LoRA 微调", char='=')
        print(f"⏰ 预计时间: 20-60 秒 (取决于 GPU)")
        print(f"🔄 总步数: {len(train_dataset) // 2 * 8}\n")
        
        trainer.train()
        
        # 7. 保存模型
        model.save_pretrained("./gpt2-lora-finetuned")
        tokenizer.save_pretrained("./gpt2-lora-finetuned")
        
        # 8. 微调后测试
        test_after_training(model, tokenizer, device)
        
        # 9. 打印总结
        elapsed = time.time() - start_time
        print_summary(trainer)
        print(f"\n⏱️  总用时: {elapsed:.1f} 秒")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()
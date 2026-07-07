import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from chapter05.chapter05 import download_verdict
import tiktoken

# ==================== 1. 自定义数据集类 ====================
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # 标签就是输入本身（语言模型任务）
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

# ==================== 2. 主函数 ====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========== 加载数据 ==========
    print("\n" + "="*60)
    print("Step 1: Loading your dataset")
    print("="*60)
    
    txt = download_verdict()
    print(f"Dataset loaded: {len(txt):,} characters")

    # 分割成训练集和验证集
    train_size = int(0.9 * len(txt))
    train_txt = txt[:train_size]
    val_txt = txt[train_size:]
    
    # 分割成文本片段（用于训练）
    def split_text(text, chunk_size=512):
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if len(chunk) > 100:  # 过滤太短的片段
                chunks.append(chunk)
        return chunks
    
    train_chunks = split_text(train_txt)
    val_chunks = split_text(val_txt)
    
    print(f"Training chunks: {len(train_chunks)}")
    print(f"Validation chunks: {len(val_chunks)}")

    # ========== 加载 GPT-2 ==========
    print("\n" + "="*60)
    print("Step 2: Loading GPT-2 from Hugging Face")
    print("="*60)
    
    model_name = "gpt2"  # 124M 参数
    # model_name = "gpt2-medium"  # 355M 参数（需要更多显存）
    
    # 加载 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad token
    
    # 加载模型
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size/1e6:.1f}M parameters")

    # ========== 生成测试（微调前） ==========
    print("\n" + "="*60)
    print("Step 3: Testing before fine-tuning")
    print("="*60)
    
    test_prompts = [
        "Every effort moves you",
        "The future of"
    ]
    
    model.eval()
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")

    # ========== 微调（可选）==========
    print("\n" + "="*60)
    print("Step 4: Fine-tuning on your text")
    print("="*60)
    
    # 准备数据集
    train_dataset = TextDataset(train_chunks, tokenizer)
    val_dataset = TextDataset(val_chunks, tokenizer)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        num_train_epochs=3,  # 只需要 3 轮
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_steps=20,
        save_steps=50,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        learning_rate=5e-5,  # 小学习率
        fp16=torch.cuda.is_available(),  # 如果 GPU 支持，加速训练
        report_to=None  # 不输出到 wandb/tensorboard
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    # 开始训练
    print("\nStarting fine-tuning...")
    trainer.train()

    # 保存微调后的模型
    trainer.save_model("./gpt2-finetuned")
    tokenizer.save_pretrained("./gpt2-finetuned")
    print("✅ Fine-tuned model saved!")

    # ========== 生成测试（微调后）==========
    print("\n" + "="*60)
    print("Step 5: Testing after fine-tuning")
    print("="*60)
    
    model.eval()
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")

    print("\n" + "="*60)
    print("✅ All done! Check the generated text above.")
    print("="*60)

if __name__ == "__main__":
    main()
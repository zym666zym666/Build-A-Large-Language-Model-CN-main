"""
文本生成脚本 - 极简交互
模型只加载一次，快速响应
"""

import torch
import os

# ========== 设置镜像（加速下载） ==========
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
from peft import PeftModel

# ==================== 风格预设 ====================

STYLES = {
    "creative": {
        "temperature": 0.95,
        "top_k": 80,
        "top_p": 0.95,
        "repetition_penalty": 1.15,
    },
    "balanced": {
        "temperature": 0.85,
        "top_k": 50,
        "top_p": 0.92,
        "repetition_penalty": 1.1,
    },
    "conservative": {
        "temperature": 0.7,
        "top_k": 30,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
    },
    "fantasy": {
        "temperature": 0.9,
        "top_k": 60,
        "top_p": 0.93,
        "repetition_penalty": 1.2,
    },
}

# ==================== 加载模型 ====================

print("="*60)
print("⏳ 正在加载模型...")
print("   (首次运行需下载约 500MB 文件，请耐心等待)")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 使用设备: {device}\n")

print("   📥 加载分词器...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print("   ✅ 分词器加载完成\n")

print("   📥 加载基础模型 (124M 参数)...")
base_model = GPT2LMHeadModel.from_pretrained("gpt2")
print("   ✅ 基础模型加载完成\n")

print("   📥 加载 LoRA 适配器...")
model = PeftModel.from_pretrained(base_model, "./gpt2-lora-finetuned")
model.to(device)
model.eval()
print("   ✅ LoRA 适配器加载完成\n")

print("="*60)
print("✅ 模型加载完成！")
print("="*60)

# ==================== 生成函数 ====================

def generate(prompt, style="balanced", max_new_tokens=200):
    """生成文本"""
    params = STYLES.get(style, STYLES["balanced"])
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # ✅ 修复：使用 GenerationConfig
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=params["temperature"],
        top_k=params["top_k"],
        top_p=params["top_p"],
        repetition_penalty=params["repetition_penalty"],
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==================== 交互 ====================

print("\n" + "="*60)
print("📖 《判决》风格文本生成器")
print("="*60)
print("\n🎨 可用风格:")
print("   creative      - 创意模式（文学性强）")
print("   balanced      - 平衡模式（默认）")
print("   conservative  - 保守模式（最稳定）")
print("   fantasy       - 奇幻风格")
print("\n💡 操作说明:")
print("   直接输入提示词 → 生成文本")
print("   /风格名        → 切换风格（如 /creative）")
print("   quit           → 退出程序")
print("="*60)

style = "balanced"

while True:
    prompt = input(f"\n📝 [{style}] 提示词: ").strip()
    
    if prompt.lower() in ["quit", "exit", "q"]:
        print("\n👋 再见!")
        break
    
    if prompt.startswith("/"):
        new_style = prompt[1:].strip()
        if new_style in STYLES:
            style = new_style
            print(f"✅ 切换到: {style}")
        else:
            print(f"❌ 未知风格: {new_style}")
            print(f"   可用: {', '.join(STYLES.keys())}")
        continue
    
    if not prompt:
        continue
    
    print("⏳ 生成中...", end="", flush=True)
    result = generate(prompt, style)
    print(" ✅\n")
    print("─"*60)
    print(result)
    print("─"*60)
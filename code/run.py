import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chapter05.chapter05 import (
    GPT_CONFIG_124M,
    GPTModel,
    download_verdict,
    create_dataloader_v1,
    train_model_simple,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text
)

import tiktoken


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    txt = download_verdict()
    print(f"Dataset loaded: {len(txt):,} characters")

    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(txt))
    print(f"Total tokens: {total_tokens:,}")

    train_size = int(0.9 * len(txt))
    train_txt, val_txt = txt[:train_size], txt[train_size:]

    train_loader = create_dataloader_v1(train_txt, batch_size=2, max_length=256, stride=128)
    val_loader = create_dataloader_v1(val_txt, batch_size=2, max_length=256, stride=128)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = GPTModel(GPT_CONFIG_124M).to(device)
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size/1e6:.1f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    start_context = "Every effort moves you"
    print(f"\nStarting training...")
    train_losses, val_losses, _ = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=5,
        eval_freq=50,
        eval_iter=10,
        start_context=start_context,
        tokenizer=tokenizer
    )

    print(f"\nGenerating text with trained model:")
    model.eval()
    context_size = GPT_CONFIG_124M["context_length"]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=100, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(f"\nGenerated:\n{decoded_text}")


if __name__ == "__main__":
    main()

"""
Mixed-precision (FP16) training pipeline for CodebaseLlama (5.87M parameters).
Trains simultaneously on raw CodeSearchNet Python code AND AST-extracted
`# repo:` context headers so the model natively masters in-context repo grounding.
"""

import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.nano_llama import CodebaseLlama, CodebaseLlamaConfig

TOKENIZER_PATH = ROOT / "data" / "tokenizer_llama_4k.json"
TRAIN_PATH = ROOT / "data" / "processed" / "train.txt"
VAL_PATH = ROOT / "data" / "processed" / "val.txt"
CKPT_DIR = ROOT / "checkpoints"
CKPT_PATH = CKPT_DIR / "codebase_llama_6m.pt"


def build_repo_augmented_corpus(raw_text: str) -> str:
    """
    Parse functions from the training corpus and prepend `# repo: <symbols>`
    headers to 50% of functions so CodebaseLlama learns native AST header attention.
    """
    chunks = raw_text.split("\ndef ")
    augmented = [chunks[0]]

    for idx, chunk in enumerate(chunks[1:]):
        func_code = "def " + chunk
        if idx % 2 == 0:
            # Extract function name and called identifiers to simulate AST repo context
            fn_match = re.match(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", func_code)
            calls = re.findall(r"([A-Za-z_][A-Za-z0-9_]{3,25})\s*\(", func_code)
            symbols = []
            if fn_match:
                symbols.append(fn_match.group(1))
            for c in calls:
                if c not in symbols and c not in {"print", "range", "len", "isinstance", "super"}:
                    symbols.append(c)
            if symbols:
                header = f"# repo: {', '.join(symbols[:4])}\n"
                augmented.append(header + func_code)
                continue
        augmented.append(func_code)

    return "\n".join(augmented)


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    max_idx = len(data) - seq_len - 1
    ix = torch.randint(0, max_idx, (batch_size,), device=device)
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, seq_len, device, eval_iters=25):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, seq_len, device)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()

    print("Building AST-header augmented training corpus...")
    raw_train = TRAIN_PATH.read_text(encoding="utf-8")
    aug_train = build_repo_augmented_corpus(raw_train)
    raw_val = VAL_PATH.read_text(encoding="utf-8")
    aug_val = build_repo_augmented_corpus(raw_val)

    train_ids = tokenizer.encode(aug_train).ids
    val_ids = tokenizer.encode(aug_val).ids

    train_data = torch.tensor(train_ids, dtype=torch.long, device=device)
    val_data = torch.tensor(val_ids, dtype=torch.long, device=device)

    config = CodebaseLlamaConfig(
        vocab_size=vocab_size,
        dim=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=256,
        dropout=0.05,
    )
    model = CodebaseLlama(config).to(device)

    batch_size = 16
    max_steps = 3000
    max_lr = 1e-3
    min_lr = 1e-4
    warmup_steps = 150

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print("=" * 70)
    print(f"  TRAINING CODEBASE-LLAMA (5.87M PARAMS) ON {device.type.upper()}")
    print("=" * 70)
    print(f"Parameters       : {model.count_parameters():,}")
    print(f"Vocab Size       : {vocab_size}")
    print(f"Context Window   : {config.max_seq_len} tokens")
    print(f"Training Tokens  : {len(train_data):,}")
    print(f"Validation Tokens: {len(val_data):,}")
    print("-" * 70)

    start_time = time.time()

    for step in range(1, max_steps + 1):
        # Cosine learning rate schedule with linear warmup
        if step < warmup_steps:
            lr = max_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, batch_size, config.max_seq_len, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 300 == 0 or step == 1:
            metrics = estimate_loss(
                model, train_data, val_data, batch_size, config.max_seq_len, device
            )
            elapsed = time.time() - start_time
            val_ppl = math.exp(metrics["val"]) if metrics["val"] < 20 else float("inf")
            print(
                f"Step {step:4d}/{max_steps} | "
                f"train loss: {metrics['train']:.4f} | "
                f"val loss: {metrics['val']:.4f} | "
                f"val PPL: {val_ppl:6.2f} | "
                f"lr: {lr:.5f} | "
                f"{elapsed:.1f}s"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "vocab_size": vocab_size,
        },
        CKPT_PATH,
    )
    print("-" * 70)
    print(f"Training complete! Saved checkpoint to: {CKPT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
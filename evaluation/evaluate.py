"""
Evaluate the locally trained TinyTransformer code language model.

Metrics:
1. Loss (validation or test)
2. Perplexity
3. Next-token prediction accuracy

Run from the project root:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --split test
"""

import argparse
from pathlib import Path
import sys
import math

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

# Make project modules importable when this script is run directly.
sys.path.insert(0, str(ROOT))

from model.transformer import TinyTransformer

DATA_DIR = ROOT / "data" / "processed"
TOKENIZER_PATH = ROOT / "data" / "tokenizer_codesearchnet.json"
CHECKPOINT_PATH = ROOT / "checkpoints" / "code_model_codesearchnet.pt"


# ============================================================
# 2. EVALUATION SETTINGS
# ============================================================

BATCH_SIZE = 4
EVAL_BATCHES = 50
SEED = 42

torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# 3. LOAD DATASET SPLIT
# ============================================================

def load_split_data(tokenizer, split_path):
    if not split_path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {split_path}\n"
            "Run: python data/prepare_codesearchnet.py"
        )

    text = split_path.read_text(encoding="utf-8")

    encoded = tokenizer.encode(text)
    token_ids = encoded.ids

    if len(token_ids) < 3:
        raise ValueError(
            "Dataset split contains too few tokens."
        )

    return torch.tensor(
        token_ids,
        dtype=torch.long,
        device=DEVICE
    )


# ============================================================
# 4. CREATE EVALUATION BATCHES
# ============================================================

def get_batch(tokens, block_size):
    max_start = len(tokens) - block_size - 1

    if max_start < 0:
        raise ValueError(
            "Evaluation data is too small for the model's "
            f"context length of {block_size}."
        )

    starts = torch.randint(
        0,
        max_start + 1,
        (BATCH_SIZE,),
        device=DEVICE
    )

    x = torch.stack([
        tokens[i.item():i.item() + block_size]
        for i in starts
    ])

    y = torch.stack([
        tokens[i.item() + 1:i.item() + block_size + 1]
        for i in starts
    ])

    return x, y


# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate_model(model, eval_tokens, block_size, vocab_size):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_predictions = 0

    for _ in range(EVAL_BATCHES):

        x, y = get_batch(eval_tokens, block_size)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        predictions = torch.argmax(logits, dim=-1)

        correct = (predictions == y).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_predictions += y.numel()

    average_loss = total_loss / EVAL_BATCHES

    accuracy = (
        total_correct / total_predictions
    ) * 100

    try:
        perplexity = math.exp(average_loss)
    except OverflowError:
        perplexity = float("inf")

    return {
        "loss": average_loss,
        "perplexity": perplexity,
        "accuracy": accuracy
    }


# ============================================================
# 6. MAIN EVALUATION PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate (val or test)",
    )
    args = parser.parse_args()

    split_filename = "val.txt" if args.split == "val" else "test.txt"
    split_label = "Validation" if args.split == "val" else "Test"
    split_path = DATA_DIR / split_filename

    print("\n" + "=" * 55)
    print(f"   TINY CODE LANGUAGE MODEL EVALUATION ({split_label.upper()})")
    print("=" * 55)

    print("\nDevice:", DEVICE)

    required_files = [
        TOKENIZER_PATH,
        CHECKPOINT_PATH,
        split_path
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    print("\nLoading tokenizer...")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    print("Tokenizer loaded successfully.")

    print("\nLoading model checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    vocab_size = checkpoint["vocab_size"]
    block_size = checkpoint["block_size"]
    model_config = checkpoint["model_config"]
    print("Checkpoint loaded successfully.")

    model = TinyTransformer(
        vocab_size=vocab_size,
        embedding_dim=model_config["embedding_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        max_length=model_config["max_length"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_parameters = sum(p.numel() for p in model.parameters())

    print("\nModel information")
    print("-" * 55)
    print("Vocabulary size:", vocab_size)
    print("Context length:", block_size)
    print("Embedding dimension:", model_config["embedding_dim"])
    print("Attention heads:", model_config["num_heads"])
    print("Transformer layers:", model_config["num_layers"])
    print("Total parameters:", f"{total_parameters:,}")

    print(f"\nLoading {split_label.lower()} dataset ({split_filename})...")
    eval_tokens = load_split_data(tokenizer, split_path)
    print(f"{split_label} tokens:", len(eval_tokens))

    print("\nEvaluating model...")
    print("-" * 55)

    results = evaluate_model(
        model=model,
        eval_tokens=eval_tokens,
        block_size=block_size,
        vocab_size=vocab_size
    )

    print("\n" + "=" * 55)
    print(f"            {split_label.upper()} EVALUATION RESULTS")
    print("=" * 55)
    print(f"{split_label} Loss : {results['loss']:.4f}")
    print(f"Perplexity      : {results['perplexity']:.4f}")
    print(f"Token Accuracy  : {results['accuracy']:.2f}%")
    print("=" * 55)
    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Make the project root importable when this file is run directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.transformer import TinyTransformer

# Project paths
DATA_DIR = ROOT / "data" / "processed"
TOKENIZER_PATH = ROOT / "data" / "tokenizer_codesearchnet.json"
CHECKPOINT_DIR = ROOT / "checkpoints"

# Training settings
BATCH_SIZE = 16
BLOCK_SIZE = 64
STEPS = 2000
LEARNING_RATE = 0.001
EVAL_INTERVAL = 200

torch.manual_seed(42)

# Select the best available device.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Training device:", DEVICE)


def main():
    # Check required files.
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}\n"
            "Run: python data/train_tokenizer.py"
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()

    def load_tokens(filename):
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}\n"
                "Run: python data/prepare_codesearchnet.py"
            )

        text = path.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text).ids

        if len(token_ids) < 2:
            raise ValueError(
                f"{filename} contains too few tokens to train."
            )

        return torch.tensor(
            token_ids,
            dtype=torch.long,
            device=DEVICE,
        )

    train_tokens = load_tokens("train.txt")
    val_tokens = load_tokens("val.txt")

    print("Vocabulary size:", vocab_size)
    print("Training tokens:", len(train_tokens))
    print("Validation tokens:", len(val_tokens))

    # Keep the context length small enough for this dataset.
    block_size = min(
        BLOCK_SIZE,
        len(train_tokens) - 1,
        len(val_tokens) - 1,
    )

    if block_size < 2:
        raise ValueError(
            "Dataset is too small. Add more Python examples "
            "to data/python_samples.py, then prepare the dataset again."
        )

    if block_size != BLOCK_SIZE:
        print(
            f"Adjusted block size from {BLOCK_SIZE} to {block_size} "
            "because the dataset is small."
        )

    if len(val_tokens) <= block_size + 1:
        print(
            "Warning: validation set is very small; "
            "validation loss will be unreliable."
        )

    # Initialize a new model from random weights.
    model = TinyTransformer(vocab_size=vocab_size).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    def get_batch(tokens):
        # Each input sequence predicts the next token.
        max_start = len(tokens) - block_size - 1

        if max_start < 0:
            raise ValueError(
                "Not enough tokens to create a training batch."
            )

        starts = torch.randint(
            0,
            max_start + 1,
            (BATCH_SIZE,),
            device=DEVICE,
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

    def evaluate(tokens, batches=5):
        model.eval()
        losses = []

        with torch.no_grad():
            for _ in range(batches):
                x, y = get_batch(tokens)
                logits = model(x)

                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1),
                )
                losses.append(loss.item())

        model.train()
        return sum(losses) / len(losses)

    print("\nStarting training...")
    model.train()

    for step in range(1, STEPS + 1):
        x, y = get_batch(train_tokens)

        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0:
            train_loss = loss.item()
            val_loss = evaluate(val_tokens)

            print(
                f"Step {step}/{STEPS} | "
                f"train loss: {train_loss:.4f} | "
                f"validation loss: {val_loss:.4f}"
            )

    # Save weights and the configuration needed to reload the model.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "code_model_codesearchnet.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "block_size": block_size,
            "tokenizer_path": str(TOKENIZER_PATH),
            "model_config": {
                "embedding_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "max_length": 128,
            },
        },
        checkpoint_path,
    )

    print("\nTraining finished!")
    print("Checkpoint saved to:", checkpoint_path)


if __name__ == "__main__":
    main()
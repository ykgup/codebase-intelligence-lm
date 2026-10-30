
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.transformer import TinyTransformer


ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = ROOT / "data" / "tokenizer.json"
CHECKPOINT_PATH = ROOT / "checkpoints" / "code_model.pt"

DEVICE = torch.device("cpu")


def main():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    config = checkpoint["model_config"]

    model = TinyTransformer(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_length=config["max_length"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = "def add(a, b):\n    "

    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long)

    if input_ids.shape[1] > config["max_length"]:
        raise ValueError("Prompt exceeds the model's maximum context length.")

    generated_ids = input_ids.clone()

    # Generate up to 40 additional tokens.
    with torch.no_grad():
        for _ in range(40):
            context = generated_ids[:, -config["max_length"]:]
            logits = model(context)
            next_token_logits = logits[0, -1, :]

            # Greedy decoding: choose the highest-probability token.
            next_token = torch.argmax(next_token_logits).view(1, 1)
            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=1,
            )

    output = tokenizer.decode(generated_ids[0].tolist())

    print("\nPrompt:")
    print(prompt)
    print("\nModel output:")
    print(output)
    print("\nNote: This is an early experiment, not a reliable code assistant.")


if __name__ == "__main__":
    main()

from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.transformer import TinyTransformer


ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = ROOT / "data" / "tokenizer_codesearchnet.json"
CHECKPOINT_PATH = ROOT / "checkpoints" / "code_model_codesearchnet.pt"

DEVICE = torch.device("cpu")

TEMPERATURE = 0.8
TOP_K = 20
MAX_NEW_TOKENS = 40


def generate_code(model, tokenizer, prompt, max_length):
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=DEVICE)

    if input_ids.shape[1] > max_length:
        raise ValueError("Prompt exceeds the model's maximum context length.")

    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            context = generated_ids[:, -max_length:]
            logits = model(context)
            next_token_logits = logits[0, -1, :] / TEMPERATURE

            # Top-k filtering to prevent low-probability gibberish and greedy loops
            top_k_logits, top_k_indices = torch.topk(next_token_logits, TOP_K)
            probs = F.softmax(top_k_logits, dim=-1)

            sampled_index = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[sampled_index].view(1, 1)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

    return tokenizer.decode(generated_ids[0].tolist())


def main():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    torch.manual_seed(42)

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

    prompts = [
        "def add(a, b):\n    ",
        "def get_user(user_id):\n    ",
    ]

    for prompt in prompts:
        output = generate_code(model, tokenizer, prompt, config["max_length"])
        print("\n" + "=" * 50)
        print("Prompt:")
        print(prompt.strip())
        print("\nModel output:")
        print(output)
    print("=" * 50)


if __name__ == "__main__":
    main()
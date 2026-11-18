"""
Train a 4,096-token Byte-Level BPE tokenizer on the CodeSearchNet corpus
for CodebaseLlama (5.87M parameters).
"""

from pathlib import Path
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CORPUS = ROOT / "data" / "processed" / "train.txt"
OUTPUT_PATH = ROOT / "data" / "tokenizer_llama_4k.json"


def main():
    if not TRAIN_CORPUS.exists():
        raise FileNotFoundError(f"Training corpus not found: {TRAIN_CORPUS}")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=4096,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "# repo:"],
    )

    print(f"Training 4,096-token BPE tokenizer on {TRAIN_CORPUS.name}...")
    tokenizer.train([str(TRAIN_CORPUS)], trainer)
    tokenizer.save(str(OUTPUT_PATH))

    print(f"Saved 4K tokenizer to: {OUTPUT_PATH}")
    print(f"Final Vocabulary Size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


DATA_DIR = Path(__file__).resolve().parent / "processed"
TOKENIZER_PATH = Path(__file__).resolve().parent / "tokenizer_codesearchnet.json"


def main():
    train_path = DATA_DIR / "train.txt"

    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run prepare_dataset.py first."
        )

    # Create a byte-level BPE tokenizer.
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    # IMPORTANT: Restore whitespace correctly during decoding.
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=512,
        min_frequency=1,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train using only the training split.
    tokenizer.train([str(train_path)], trainer)
    tokenizer.save(str(TOKENIZER_PATH))

    print("Tokenizer trained successfully!")
    print("Vocabulary size:", tokenizer.get_vocab_size())
    print("Saved to:", TOKENIZER_PATH)

    # Test Python code round-trip.
    sample = "def add(a, b):\n    return a + b"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)

    print("\nOriginal sample:", repr(sample))
    print("Decoded sample: ", repr(decoded))
    print("Round-trip successful:", decoded == sample)

    if decoded != sample:
        raise ValueError(
            "Tokenizer round-trip failed. "
            "Compare the original and decoded samples above."
        )


if __name__ == "__main__":
    main()
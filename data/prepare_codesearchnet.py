import json
from pathlib import Path

# Project directories
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
OUTPUT_DIR = ROOT / "processed"

# Dataset files
TRAIN_FILE = RAW_DIR / "train.jsonl"
VAL_FILE = RAW_DIR / "validation.jsonl"
TEST_FILE = RAW_DIR / "test.jsonl"


def load_code_samples(file_path):
    """Load valid Python code samples from a JSONL file."""

    samples = []
    seen = set()

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                code = record.get("code", "")

                if not isinstance(code, str):
                    continue

                code = code.strip()

                # Remove empty or very short samples
                if len(code) < 30:
                    continue

                # Remove duplicate code
                if code in seen:
                    continue

                seen.add(code)
                samples.append(code)

            except json.JSONDecodeError:
                continue

    return samples


def save_samples(samples, output_file):
    """Save code samples as a text file."""

    output_file.write_text(
        "\n\n".join(samples),
        encoding="utf-8"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("CODESEARCHNET DATASET PREPARATION")
    print("=" * 55)

    print("\nLoading training data...")
    train_samples = load_code_samples(TRAIN_FILE)

    print("Loading validation data...")
    val_samples = load_code_samples(VAL_FILE)

    print("Loading test data...")
    test_samples = load_code_samples(TEST_FILE)

    # Save each split separately
    save_samples(train_samples, OUTPUT_DIR / "train.txt")
    save_samples(val_samples, OUTPUT_DIR / "val.txt")
    save_samples(test_samples, OUTPUT_DIR / "test.txt")

    print("\nDATASET PREPARATION REPORT")
    print("-" * 55)

    print(f"Training samples:   {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples:       {len(test_samples)}")

    print("\nSaved files:")

    print(OUTPUT_DIR / "train.txt")
    print(OUTPUT_DIR / "val.txt")
    print(OUTPUT_DIR / "test.txt")

    print("\nDataset preparation completed!")


if __name__ == "__main__":
    main()
import random
from pathlib import Path

from python_samples import PYTHON_SAMPLES

SEED = 42
TRAIN_RATIO = 0.8

def main():
    samples = PYTHON_SAMPLES.copy()

    random.Random(SEED).shuffle(samples)

    split_index = int(len(samples) * TRAIN_RATIO)

    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    output_dir = Path(__file__).resolve().parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train.txt").write_text(
        "\n\n".join(train_samples), encoding="utf-8"
    )

    (output_dir / "val.txt").write_text(
        "\n\n".join(val_samples), encoding="utf-8"
    )

    print(f"Total samples: {len(samples)}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    main()
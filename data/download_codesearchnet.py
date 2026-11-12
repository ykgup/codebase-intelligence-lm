import json
import hashlib
import random
import sys
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "Nan-Do/code-search-net-python"

OUTPUT_DIR = Path(__file__).resolve().parent / "raw"

SAMPLE_LIMITS = {
    "train": 10000,
    "validation": 1000,
    "test": 1000,
}

MIN_CODE_LENGTH = 20
MAX_CODE_LENGTH = 12000

SEED = 42
SHUFFLE_BUFFER_SIZE = 10000

# Normalize possible partition names.
PARTITION_MAP = {
    "train": "train",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_partition(value):
    """Convert dataset partition labels to standard names."""

    if value is None:
        return None

    return PARTITION_MAP.get(str(value).strip().lower())


def extract_code(row):
    """Extract code from the dataset record."""

    code = row.get("code") or row.get("original_string") or ""

    if not isinstance(code, str):
        return ""

    return code.strip()


def save_jsonl(records, filename):
    """Save records as JSON Lines."""

    output_file = OUTPUT_DIR / filename

    with output_file.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    return output_file


# ============================================================
# MAIN DATASET DOWNLOAD FUNCTION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CODESearchNet PYTHON DATASET DOWNLOAD")
    print("=" * 60)

    print("\nDataset:", DATASET_NAME)
    print("Output directory:", OUTPUT_DIR)

    print("\nTarget samples:")

    for partition, limit in SAMPLE_LIMITS.items():
        print(f"  {partition}: {limit}")

    # --------------------------------------------------------
    # Load dataset using streaming
    # --------------------------------------------------------

    print("\nConnecting to Hugging Face...")

    try:
        dataset = load_dataset(
            DATASET_NAME,
            split="train",
            streaming=True,
        )

        dataset = dataset.shuffle(
            seed=SEED,
            buffer_size=SHUFFLE_BUFFER_SIZE,
        )

    except Exception as error:
        print("\nERROR: Could not load the dataset.")
        print(error)
        sys.exit(1)

    # --------------------------------------------------------
    # Prepare storage
    # --------------------------------------------------------

    samples = defaultdict(list)

    seen_hashes = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    rejected = 0
    scanned = 0

    print("\nCollecting code samples...\n")

    try:

        for row in dataset:

            scanned += 1

            # Normalize partition label.
            original_partition = row.get("partition")
            partition = normalize_partition(original_partition)

            if partition is None:
                rejected += 1
                continue

            # Skip partitions that already reached their limit.
            if len(samples[partition]) >= SAMPLE_LIMITS[partition]:
                if all(
                    len(samples[name]) >= limit
                    for name, limit in SAMPLE_LIMITS.items()
                ):
                    break

                continue

            # Extract source code.
            code = extract_code(row)

            if len(code) < MIN_CODE_LENGTH:
                rejected += 1
                continue

            if len(code) > MAX_CODE_LENGTH:
                rejected += 1
                continue

            # Remove duplicate code within each partition.
            code_hash = hashlib.sha256(
                code.encode("utf-8")
            ).hexdigest()

            if code_hash in seen_hashes[partition]:
                rejected += 1
                continue

            seen_hashes[partition].add(code_hash)

            # Preserve useful source metadata.
            record = {
                "code": code,
                "func_name": row.get("func_name"),
                "docstring": row.get("docstring"),
                "repo": row.get("repo"),
                "path": row.get("path"),
                "url": row.get("url"),
                "partition": original_partition,
                "source": DATASET_NAME,
            }

            samples[partition].append(record)

            # Progress reporting.
            current_count = len(samples[partition])
            target_count = SAMPLE_LIMITS[partition]

            if current_count % 1000 == 0 or current_count == target_count:
                print(
                    f"{partition}: "
                    f"{current_count}/{target_count}"
                )

            # Stop once all targets are reached.
            if all(
                len(samples[name]) >= limit
                for name, limit in SAMPLE_LIMITS.items()
            ):
                break

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("Collected samples will still be saved.")

    except Exception as error:
        print("\nERROR while reading dataset:")
        print(error)
        print("Saving samples collected so far.")

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SAVING DATASETS")
    print("=" * 60)

    for partition in SAMPLE_LIMITS:

        records = samples[partition]

        filename = f"{partition}.jsonl"

        output_file = save_jsonl(records, filename)

        print(f"\n{partition}: {len(records)} samples")
        print("Saved:", output_file)

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "dataset": DATASET_NAME,
        "seed": SEED,
        "sample_limits": SAMPLE_LIMITS,
        "collected_samples": {
            partition: len(samples[partition])
            for partition in SAMPLE_LIMITS
        },
        "scanned_records": scanned,
        "rejected_records": rejected,
        "minimum_code_length": MIN_CODE_LENGTH,
        "maximum_code_length": MAX_CODE_LENGTH,
    }

    metadata_file = OUTPUT_DIR / "metadata.json"

    with metadata_file.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET COLLECTION REPORT")
    print("=" * 60)

    total_collected = 0

    for partition, target in SAMPLE_LIMITS.items():

        actual = len(samples[partition])
        total_collected += actual

        status = "COMPLETE" if actual == target else "INCOMPLETE"

        print(
            f"{partition:12} "
            f"{actual:6}/{target:<6} "
            f"{status}"
        )

    print("-" * 60)
    print("Total collected:", total_collected)
    print("Records scanned:", scanned)
    print("Records rejected:", rejected)

    print("\nMetadata saved:", metadata_file)

    if all(
        len(samples[name]) == limit
        for name, limit in SAMPLE_LIMITS.items()
    ):
        print("\nSUCCESS: All dataset targets reached!")
    else:
        print("\nWARNING: Some dataset partitions are incomplete.")
        print("Check the partition labels and Hugging Face connection.")

    print("\nDataset preparation finished.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
# Codebase Intelligence LM

An experimental research project investigating whether **AST-extracted structured repository context** improves the repository-level code understanding and symbol prediction of a custom, locally deployable Transformer code language model (`TinyTransformer`).

## Research Question

> *Does structured repository context (AST-extracted module metadata, class definitions, and function signatures) improve the repository-level code understanding of a small locally deployable code model?*

## Model Architecture (`TinyTransformer`)

Built from scratch in PyTorch (`model/transformer.py`):
- **Parameters:** 174,208
- **Vocabulary Size:** 512 (Byte-Level BPE tokenizer trained on CodeSearchNet Python)
- **Context Length (`block_size`):** 64 tokens (`max_length = 128`)
- **Embedding Dimension:** 64
- **Attention Heads:** 4
- **Transformer Layers:** 2

## Project Structure

- `model/transformer.py` — Custom `TinyTransformer` architecture.
- `data/download_codesearchnet.py` — Downloads and deduplicates 12,000 Python functions from `Nan-Do/code-search-net-python` (10,000 train / 1,000 validation / 1,000 test).
- `data/prepare_codesearchnet.py` — Formats raw JSONL splits into processed training and evaluation corpora.
- `data/train_tokenizer.py` — Trains the 512-token Byte-Level BPE tokenizer (`data/tokenizer_codesearchnet.json`).
- `training/train.py` — End-to-end training script supporting CUDA, Apple Silicon MPS, and CPU.
- `evaluation/evaluate.py` — Evaluates validation (`--split val`) and held-out test (`--split test`) loss, perplexity, and next-token accuracy.
- `generate.py` — Autoregressive code generation using temperature and top-k sampling.
- `repo_context/builder.py` — Extracts structured repository context (file tree, imports, classes, methods, and function signatures) using Python's `ast` module.
- `repo_context/benchmark.py` — Controlled benchmark evaluating `TinyTransformer` on unseen repository symbol prediction with vs. without structured AST context headers.

## Experimental Results

### Part 1: Code Language Modeling Baseline (41 Samples vs. 12,000 CodeSearchNet Samples)

| Experiment Run | Dataset Split & Size | Cross-Entropy Loss | Perplexity | Token Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **1. Initial Synthetic Baseline** | Validation (9 samples) | 3.2894 | 26.8257 | 36.33% |
| **2. CodeSearchNet (300 Steps, BS=4)** | Validation (762,709 tokens) | 4.7265 | 112.8980 | 9.88% |
| **3. CodeSearchNet (2,000 Steps, BS=16)** | Validation (762,709 tokens) | 3.5824 | 35.9602 | 28.24% |
| **4. CodeSearchNet Held-Out Test** | Test (669,497 tokens) | 3.5070 | 33.3474 | 19.67% |

### Part 2: Controlled Repository Context Benchmark (`repo_context/benchmark.py`)

We evaluate `TinyTransformer` on 6 unseen repository files from `codebase-intelligence-lm` under two controlled conditions:
1. **No Context (Baseline):** The model receives only the local code prefix (`symbol = `) without file-level metadata.
2. **With Structured Context (Treatment):** The model receives a compact AST-extracted header (`# repo: <SYMBOL>`) prepended to the local prefix.

We evaluate both the **Unadapted CodeSearchNet Model (Zero-Shot)** and the **Context-Adapted Model** (adapted on 738 disjoint CodeSearchNet identifiers with replay regularization, zero evaluation overlap):

| Experimental Stage & Condition | Avg Loss | Avg Perplexity | Avg Token Accuracy |
| :--- | :--- | :--- | :--- |
| **Exp 1: Unadapted Model (No Context)** | 4.0413 | 56.90 | 7.03% |
| **Exp 1: Unadapted Model (With AST Context)** | 3.9665 | 52.80 | 10.63% |
| **Exp 2: Context-Adapted (No Context)** | 3.9547 | 52.18 | 10.33% |
| **Exp 2: Context-Adapted (With AST Context)** | **2.6690** | **14.43** | **36.39%** |

#### Per-File Breakdown (Experiment 2: Context-Adapted `TinyTransformer`)

| Task ID | Source File | Target Symbol | No-Ctx PPL | With-Ctx PPL | No-Ctx Acc | With-Ctx Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `T1_Transformer` | `model/transformer.py` | `TinyTransformer` | 185.23 | **131.61** | 0.00% | 0.00% |
| `T2_Downloader` | `data/download_codesearchnet.py` | `normalize_partition` | 18.75 | **5.20** | 22.22% | **55.56%** |
| `T3_PrepDataset` | `data/prepare_codesearchnet.py` | `load_code_samples` | 24.44 | **8.95** | 27.27% | **54.55%** |
| `T4_Evaluator` | `evaluation/evaluate.py` | `load_split_data` | 47.12 | **8.88** | 12.50% | **50.00%** |
| `T5_Generator` | `generate.py` | `generate_code` | 46.28 | **3.61** | 0.00% | **42.86%** |
| `T6_ASTBuilder` | `repo_context/builder.py` | `analyze_python_file` | 109.06 | **45.89** | 0.00% | **15.38%** |
| **AVERAGE** | **6 Repository Files** | **6 Unseen Symbols** | 52.18 | **14.43** | 10.33% | **36.39%** |

## Setup & Reproduction

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 2. Download and prepare CodeSearchNet dataset (12,000 samples)
python data/download_codesearchnet.py
python data/prepare_codesearchnet.py

# 3. Train and evaluate the baseline TinyTransformer
python training/train.py
python evaluation/evaluate.py --split val
python evaluation/evaluate.py --split test
python generate.py

# 4. Generate AST repository context and run the controlled benchmark
python repo_context/builder.py
python repo_context/benchmark.py
```
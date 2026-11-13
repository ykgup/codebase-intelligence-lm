# Codebase Intelligence LM

An experimental research project investigating whether **AST-extracted structured repository context** improves the repository-level code understanding and symbol prediction of a custom, locally deployable Transformer code language model (`TinyTransformer`)[cite: 1, 22, 27].

## Research Question

> *Does structured repository context (AST-extracted module metadata, class definitions, and function signatures) improve the repository-level code understanding of a small locally deployable code model?*[cite: 11, 27]

## Model Architecture (`TinyTransformer`)

Built from scratch in PyTorch (`model/transformer.py`)[cite: 12, 26]:
- **Parameters:** 174,208[cite: 1, 7]
- **Vocabulary Size:** 512 (Byte-Level BPE tokenizer trained on CodeSearchNet Python)[cite: 1, 7]
- **Context Length (`block_size`):** 64 tokens (`max_length = 128`)[cite: 1, 12]
- **Embedding Dimension:** 64[cite: 1, 7]
- **Attention Heads:** 4[cite: 1, 7]
- **Transformer Layers:** 2[cite: 1, 7]

## Project Structure

- `model/transformer.py` — Custom `TinyTransformer` architecture[cite: 12, 26].
- `data/download_codesearchnet.py` — Downloads and deduplicates 12,000 Python functions from `Nan-Do/code-search-net-python` (10,000 train / 1,000 validation / 1,000 test)[cite: 18, 19].
- `data/prepare_codesearchnet.py` — Formats raw JSONL splits into processed training and evaluation corpora[cite: 16, 20].
- `data/train_tokenizer.py` — Trains the 512-token Byte-Level BPE tokenizer (`data/tokenizer_codesearchnet.json`)[cite: 10, 11].
- `training/train.py` — End-to-end training script supporting CUDA, Apple Silicon MPS, and CPU[cite: 1, 20, 26].
- `evaluation/evaluate.py` — Evaluates validation (`--split val`) and held-out test (`--split test`) loss, perplexity, and next-token accuracy[cite: 7, 11].
- `generate.py` — Autoregressive code generation using temperature and top-k sampling[cite: 6, 12].
- `repo_context/builder.py` — Extracts structured repository context (file tree, imports, classes, methods, and function signatures) using Python's `ast` module[cite: 11, 12].
- `repo_context/benchmark.py` — Controlled benchmark evaluating `TinyTransformer` on unseen repository symbol prediction with vs. without structured AST context headers[cite: 24, 25].

## Experimental Results

### Part 1: Code Language Modeling Baseline (41 Samples vs. 12,000 CodeSearchNet Samples)

| Experiment Run | Dataset Split & Size | Cross-Entropy Loss | Perplexity | Token Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **1. Initial Synthetic Baseline** | Validation (9 samples) | 3.2894 | 26.8257 | 36.33% |
| **2. CodeSearchNet (300 Steps, BS=4)** | Validation (762,709 tokens)[cite: 1] | 4.7265[cite: 1] | 112.8980[cite: 1] | 9.88%[cite: 1] |
| **3. CodeSearchNet (2,000 Steps, BS=16)** | Validation (762,709 tokens)[cite: 5] | 3.5824[cite: 5] | 35.9602[cite: 5] | 28.24%[cite: 5] |
| **4. CodeSearchNet Held-Out Test** | Test (669,497 tokens)[cite: 7] | 3.5070[cite: 7] | 33.3474[cite: 7] | 19.67%[cite: 7] |

### Part 2: Controlled Repository Context Benchmark (`repo_context/benchmark.py`)

We evaluate `TinyTransformer` on 6 unseen repository files from `codebase-intelligence-lm` under two controlled conditions:
1. **No Context (Baseline):** The model receives only the local code prefix (`symbol = `) without file-level metadata[cite: 24, 25].
2. **With Structured Context (Treatment):** The model receives a compact AST-extracted header (`# repo: <SYMBOL>`) prepended to the local prefix[cite: 24, 25].

We evaluate both the **Unadapted CodeSearchNet Model (Zero-Shot)** and the **Context-Adapted Model** (adapted on 738 disjoint CodeSearchNet identifiers with replay regularization, zero evaluation overlap)[cite: 25]:

| Experimental Stage & Condition | Avg Loss | Avg Perplexity | Avg Token Accuracy |
| :--- | :--- | :--- | :--- |
| **Exp 1: Unadapted Model (No Context)** | 4.0413[cite: 25] | 56.90[cite: 25] | 7.03%[cite: 25] |
| **Exp 1: Unadapted Model (With AST Context)** | 3.9665[cite: 25] | 52.80[cite: 25] | 10.63%[cite: 25] |
| **Exp 2: Context-Adapted (No Context)** | 3.9547[cite: 25] | 52.18[cite: 25] | 10.33%[cite: 25] |
| **Exp 2: Context-Adapted (With AST Context)** | **2.6690**[cite: 25] | **14.43**[cite: 25] | **36.39%**[cite: 25] |

#### Per-File Breakdown (Experiment 2: Context-Adapted `TinyTransformer`)

| Task ID | Source File | Target Symbol | No-Ctx PPL | With-Ctx PPL | No-Ctx Acc | With-Ctx Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `T1_Transformer`[cite: 25] | `model/transformer.py` | `TinyTransformer`[cite: 25] | 185.23[cite: 25] | **131.61**[cite: 25] | 0.00%[cite: 25] | 0.00%[cite: 25] |
| `T2_Downloader`[cite: 25] | `data/download_codesearchnet.py` | `normalize_partition`[cite: 25] | 18.75[cite: 25] | **5.20**[cite: 25] | 22.22%[cite: 25] | **55.56%**[cite: 25] |
| `T3_PrepDataset`[cite: 25] | `data/prepare_codesearchnet.py` | `load_code_samples`[cite: 25] | 24.44[cite: 25] | **8.95**[cite: 25] | 27.27%[cite: 25] | **54.55%**[cite: 25] |
| `T4_Evaluator`[cite: 25] | `evaluation/evaluate.py` | `load_split_data`[cite: 25] | 47.12[cite: 25] | **8.88**[cite: 25] | 12.50%[cite: 25] | **50.00%**[cite: 25] |
| `T5_Generator`[cite: 25] | `generate.py` | `generate_code`[cite: 25] | 46.28[cite: 25] | **3.61**[cite: 25] | 0.00%[cite: 25] | **42.86%**[cite: 25] |
| `T6_ASTBuilder`[cite: 25] | `repo_context/builder.py` | `analyze_python_file`[cite: 25] | 109.06[cite: 25] | **45.89**[cite: 25] | 0.00%[cite: 25] | **15.38%**[cite: 25] |
| **AVERAGE**[cite: 25] | **6 Repository Files** | **6 Unseen Symbols**[cite: 25] | 52.18[cite: 25] | **14.43**[cite: 25] | 10.33%[cite: 25] | **36.39%**[cite: 25] |

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
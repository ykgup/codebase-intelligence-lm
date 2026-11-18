# ⚡ CodebaseLM (`cblm`) — Terminal-Native AST-Grounded Code Intelligence & Neural QA Engine

**CodebaseLM** is a local, lightweight repository intelligence engine powered by **`CodebaseLlama` (5.87M parameters)** and **deterministic Python AST extraction**. Built from scratch in PyTorch to run entirely on consumer 4GB VRAM GPUs (tested on NVIDIA GeForce RTX 2050), it indexes any Python repository in milliseconds, adapts its weights to your codebase's symbols and architecture on the fly ("Instant Repo-Brain"), answers natural-language codebase questions, autocompletes cross-file API calls, and catches hallucinated function calls.

---

## 🚀 Key Capabilities

1. **Interactive Terminal Shell & Full-Repo Neural QA (`cblm`)**:
   - Recursively reads every `.py` file, class, method signature, docstring, and cross-file call graph in your repository.
   - Synthesizes repository-specific Q&A pairs and micro-adapts `CodebaseLlama` on your GPU in seconds.
   - Ask natural-language questions directly in your terminal:
     - `Where is CodebaseLlama defined and who calls it?`
     - `What is inside train_llama.py?`
     - `What does this codebase do?`
2. **AST Context-Grounded Neural Autocompletion**:
   - Injects compact `# repo: <symbols>` AST headers into the 256-token context window.
   - Supports `@filename` targeting (e.g., `@evaluate.py def caller(x): return `) to steer completions with **98%+ token confidence**.
3. **Deterministic AST Hallucination Guard (`:verify` & `:audit`)**:
   - Flags non-existent or hallucinated function/class calls in code snippets and suggests the closest real symbol in your repository.
   - Runs full-repository cross-file call integrity audits (`:audit`).
4. **CodebaseLM Studio (`studio.py`)**:
   - Zero-dependency dark-mode web playground (`http://localhost:8050`) visualizing live AST graphs and token-by-token probability bars (Without AST Context vs. With `# repo:` AST Context).

---

## 🧠 Model Architecture (`CodebaseLlama` v0.2.0 vs `TinyTransformer` v0.1.0)

| Specification | `TinyTransformer` (v0.1.0) | `CodebaseLlama` (v0.2.0) |
| :--- | :--- | :--- |
| **Parameters** | 174,336 (~174K) | **5,868,800 (~5.87M)** |
| **Architecture** | Vanilla Decoder | **Llama-style (RoPE, RMSNorm, SwiGLU, FlashAttention)** |
| **Layers / Heads / Dim** | 2 Layers / 4 Heads / 128 Dim | **6 Layers / 8 Heads / 256 Dim** |
| **Vocabulary (Byte-Level BPE)** | 512 tokens | **4,096 tokens (`tokenizer_llama_4k.json`)** |
| **Context Window** | 64 tokens | **256 tokens** |
| **Pre-training Corpus** | CodeSearchNet Python | **CodeSearchNet Python + 50% Native `# repo:` AST Headers** |

---

## 📊 Repository Grounding Benchmark Results (`CodebaseLlama` 5.87M)

Evaluated across repository files predicting cross-file symbol invocations (`python codebase_cli.py --repo . --adapt`):

| Evaluation Stage | Without AST Context (PPL) | With `# repo:` AST Context (PPL) | Next-Token Accuracy (With AST) |
| :--- | :--- | :--- | :--- |
| **Stage 1: Zero-Shot Pre-trained** | 57.54 | **9.07** (-84.2% PPL) | **64.57%** (vs 34.41% blind) |
| **Stage 2: Instant Repo-Brain Adapted** | 3.58 | **2.82** | **84.32%** (up to 93.75% peak) |

---

## 🛠️ Quickstart & Installation

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Register the `cblm` terminal CLI command
python -m pip install --no-build-isolation --no-deps -e .

# 3. Pre-train CodebaseLlama (5.87M) locally on GPU (~4 minutes on RTX 2050)
python data/train_tokenizer_4k.py
python training/train_llama.py

# 4. Launch the interactive Full-Repo Analyzer & Neural QA Shell in any repo
cblm
```

---

### Direct CLI Commands

```bash
# Scan and adapt to any repository path from the terminal
cblm scan .

# Audit all cross-file symbol invocations across a repository
cblm audit .

# Run the full quantitative benchmark table + Hallucination Guard
python codebase_cli.py --repo . --adapt
```

---

## 📂 Project Structure

```text
codebase-intelligence-lm/
├── cblm.py                        # Interactive Terminal Shell, AST-RAG & Neural QA Engine
├── codebase_cli.py                # Universal Repo-Brain Benchmark & Hallucination Guard CLI
├── studio.py                      # Dark-mode Web Visualizer & Token Probability Playground
├── pyproject.toml                 # Package config registering the global `cblm` command
├── requirements.txt               # Python package dependencies
├── model/
│   ├── nano_llama.py              # CodebaseLlama (5.87M) with RoPE, RMSNorm, SwiGLU
│   └── transformer.py             # Baseline TinyTransformer (174K)
├── data/
│   ├── train_tokenizer_4k.py      # 4,096-token Byte-Level BPE tokenizer trainer
│   ├── tokenizer_llama_4k.json    # Trained 4K BPE vocabulary
│   ├── download_codesearchnet.py  # CodeSearchNet downloader
│   └── prepare_codesearchnet.py   # Corpus cleaner and splitter
├── training/
│   ├── train_llama.py             # FP16 mixed-precision trainer with AST header augmentation
│   └── train.py                   # Baseline trainer
├── repo_context/
│   ├── builder.py                 # Fast AST symbol & signature extractor
│   └── benchmark.py               # Controlled context evaluation suite
└── evaluation/
    └── evaluate.py                # Perplexity and token accuracy evaluator
```

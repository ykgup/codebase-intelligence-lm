# Codebase Intelligence LM

An experimental research project exploring whether a small,
locally deployable Transformer model can learn useful code patterns.

## Current status

- Tiny Transformer architecture implemented.
- Training pipeline runs using PyTorch.
- Initial experiments use a small, repetitive demonstration dataset.
- General code understanding has not yet been established.

## Project structure

- `model/transformer.py` — Transformer architecture.
- `training/train.py` — training script.
- `data/` — dataset files.
- `checkpoints/` — locally saved model checkpoints.

## Setup

Create and activate a Python virtual environment, then install
the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run training:

```bash
python training/train.py
```

## Research goal

Evaluate whether structured repository context improves the
repository-level code understanding of a small local code model.
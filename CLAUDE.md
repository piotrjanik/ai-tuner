# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A generic fine-tuning pipeline that trains LLMs (via LoRA/QLoRA) to serve as domain-specific assistants. Training data comes from Go source code repositories, Markdown specification files, and public HuggingFace datasets. The domain, system prompt, sources, and model are all configured in `config.yaml`. Fine-tuning runs on Apple Silicon (MLX) or NVIDIA GPUs (PyTorch + PEFT). Models are exported as GGUF for Ollama and LM Studio.

## Commands

### CLI (primary)
```bash
python src/cli.py --help
python src/cli.py data                             # extract training data
python src/cli.py train --nice 10 --batch-size 2   # fine-tune (throttled, Apple Silicon)
python src/cli.py train --unsloth                  # fine-tune (Unsloth on NVIDIA GPU / Colab)
python src/cli.py export                           # fuse adapters → output/merged/
python src/cli.py export-gguf --quant Q4_K_M       # GGUF for Ollama/LM Studio
python src/cli.py push --gguf                      # push GGUF to HuggingFace
python src/cli.py infer -q "How does X work?"      # single question
python src/cli.py infer                            # interactive mode
```

### Taskfile
```bash
task venv           # Create .venv and install dependencies
task data           # Clone repos + extract training data
task train          # Fine-tune (pass -- --resume to continue)
task export         # Fuse LoRA adapters into output/merged/
task export:gguf    # Convert to GGUF via llama.cpp
task push:hf        # Push adapters to HuggingFace
task push:hf:gguf   # Push GGUF to HuggingFace
task infer          # Interactive inference
task clean          # Remove data/ and output/
task clean:all      # Remove everything including .venv and repos/
```

**Platform**: Data prep runs anywhere. Training has two backends:
- **MLX** (default) — local Apple Silicon (M1–M4)
- **Unsloth** (`--unsloth` / `--cuda`) — NVIDIA GPUs, Google Colab, cloud VMs

Export auto-detects adapter format (MLX vs PEFT). GGUF conversion and inference work on both.

## Architecture

### Pipeline
```
config.yaml (sources, system_prompt, model, hyperparams)
    ↓
src/data/prepare_data.py  →  data/{train,val,units}.jsonl
    ↓
src/train/train.py        →  output/adapters/  (LoRA, Apple Silicon)
src/train/train_cuda.py   →  output/adapters/  (QLoRA, NVIDIA GPU)
    ↓
src/export/export_model.py →  output/merged/  →  output/gguf/
    ↓
src/inference/inference.py  or  ollama run <model-name>
```

### Key files
- `config.yaml` — all pipeline parameters: sources, system prompt, model, training hyperparams, export settings
- `src/prompt.py` — loads system prompt from config.yaml (used by data gen + inference)
- `src/cli.py` — unified CLI entry point (typer)

### Data Preparation (`src/data/prepare_data.py`)

Three source types dispatched by `type` field in `config.yaml`:

**`go_code`** — `GoParser` extracts documented functions, methods, structs, interfaces. Only units with doc comments generate explanation/documentation examples (undocumented code is skipped to avoid low-quality auto-generated answers). Generates: code_explanation, code_completion, type_documentation, package_overview.

**`spec`** — `SpecParser` splits Markdown by h2/h3 headings into Q&A examples.

**`golang_datasets`** — HuggingFace datasets. Supports formats: `messages`, `qa`, `instruction`, `code_search_net`. External system prompts are replaced with the project's shared prompt from config.yaml.

### Training

Two backends, same config.yaml and data format:

- **`src/train/train.py`** — Apple Silicon via mlx-lm. Default for `python src/cli.py train`. Uses `training.base_model` from config.
- **`src/train/train_cuda.py`** — NVIDIA GPUs via Unsloth + QLoRA. Use `python src/cli.py train --unsloth` or the Colab notebook at `notebooks/train_colab.ipynb`. Uses `training.cuda_model` from config.

Both auto-tune batch_size, max_seq_length, and gradient checkpointing based on available GPU memory. Override with `--batch-size` / `--max-seq-length` or disable with `--no-auto-tune`.

### Configuration

Everything domain-specific lives in `config.yaml`:
- `system_prompt` — injected into every training example and inference session
- `sources` — Go repos, spec repos to clone and parse
- `golang_datasets` — external HuggingFace datasets
- `training.base_model` — MLX model for local training (Apple Silicon)
- `training.cuda_model` — Unsloth model for cloud training (NVIDIA GPUs)
- `export.ollama_model_name` / `export.hf_repo` — deployment targets

## Dependencies

- **Local (MLX)**: `src/requirements.txt` — install with `task venv` (uses **uv**)
- **Cloud (Unsloth)**: `src/requirements-cuda.txt` — installed by notebook / cloud scripts

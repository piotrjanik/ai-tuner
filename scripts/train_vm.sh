#!/usr/bin/env bash
# Train locally on a VM with an NVIDIA GPU.
#
# Run this directly on the VM after cloning the repo.
# Waits for NVIDIA drivers, installs dependencies, prepares data, and trains.
#
# Usage:
#   task train:vm                  # install deps + prepare data + train
#   task train:vm -- --push        # also push adapters to HF Hub
#   task train:vm -- --resume      # resume from last checkpoint
#   task train:vm -- --skip-deps   # skip dependency installation

set -euo pipefail

# ── Parse args ───────────────────────────────────────────────────────────────
PUSH_HF=false
RESUME=""
SKIP_DEPS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)      PUSH_HF=true; shift ;;
    --resume)    RESUME="--resume"; shift ;;
    --skip-deps) SKIP_DEPS=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "══════════════════════════════════════════════════════════════"
echo "  VM Training (local)"
echo "══════════════════════════════════════════════════════════════"

# ── Wait for NVIDIA drivers ─────────────────────────────────────────────────
echo "── Checking NVIDIA GPU ──"
for i in $(seq 1 30); do
  if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    break
  fi
  if [ "$i" -eq 1 ]; then
    echo "  Drivers not ready yet (normal on first boot, waiting up to 5 min)..."
  fi
  [ "$((i % 6))" -eq 0 ] && echo "  Still waiting... ($((i*10))s)"
  sleep 10
done

if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
  echo "✗ NVIDIA drivers not found after 5 min."
  echo "  Install drivers:  sudo apt install -y nvidia-driver-560"
  echo "  Or use a Deep Learning VM image with pre-installed drivers."
  exit 1
fi

# ── Create / activate venv ───────────────────────────────────────────────────
VENV_DIR="${HOME}/.venv-ai-tuner"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "── Creating Python venv at $VENV_DIR ──"
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-full 2>/dev/null || true
  python3 -m venv "$VENV_DIR" --system-site-packages
fi
source "$VENV_DIR/bin/activate"
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python3"
echo "Using Python: $($PYTHON --version) from $VENV_DIR"

# ── Install dependencies ────────────────────────────────────────────────────
if [[ "$SKIP_DEPS" == "false" ]]; then
  echo "── Installing dependencies ──"

  # Install unsloth + trl with --no-deps to preserve CUDA torch
  $PIP install -q --no-deps \
      "unsloth @ git+https://github.com/unslothai/unsloth" \
      "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo"
  $PIP install -q --no-deps trl==0.22.2

  # Install remaining deps (these don't replace torch)
  $PIP install -q transformers==4.56.2 tokenizers bitsandbytes triton \
      "peft>=0.13.0" "accelerate>=1.0.0" "typer>=0.12.0" \
      pyyaml tqdm "huggingface_hub[cli]" "datasets>=2.14.0" xformers

  # Flash Attention (optional, Ampere+ only)
  timeout 120 $PIP install -q flash-attn --no-build-isolation 2>/dev/null \
      || echo "⚠ flash-attn not available — using eager attention"
else
  echo "── Skipping dependency installation (--skip-deps) ──"
fi

# ── Verify CUDA ─────────────────────────────────────────────────────────────
$PYTHON -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check NVIDIA drivers'
props = torch.cuda.get_device_properties(0)
print(f'✓ torch {torch.__version__}, GPU: {props.name} ({props.total_mem / 1e9:.0f} GB)')
"

# ── Prepare data ────────────────────────────────────────────────────────────
echo "── Preparing training data ──"
$PYTHON src/data/prepare_data.py --config config.yaml

# ── Train ────────────────────────────────────────────────────────────────────
echo "── Starting training ──"
$PYTHON src/train/train_cuda.py --config config.yaml $RESUME

echo "── Training complete! ──"
ls -lh output/adapters/

# ── Push to HuggingFace (optional) ──────────────────────────────────────────
if [[ "$PUSH_HF" == "true" ]]; then
  echo "── Pushing adapters to HuggingFace ──"
  $PYTHON src/cli.py push
fi

echo ""
echo "Done! Adapters are in output/adapters/"

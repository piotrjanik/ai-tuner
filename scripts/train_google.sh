#!/usr/bin/env bash
# Train on Google infrastructure.
#
# Default mode (Colab): pushes latest code to GitHub, opens the Colab notebook
# in the browser. You click "Runtime > Run all" and adapters get pushed to HF.
#
# GCE mode (--gce): fully automated — creates a GPU VM, runs training, pulls
# adapters back, deletes the VM. Requires gcloud CLI + GCP project with GPU quota.
#
# Usage:
#   ./scripts/train_google.sh                          # Open Colab notebook
#   ./scripts/train_google.sh --gce                    # GCE with L4 GPU (default)
#   ./scripts/train_google.sh --gce --gpu a100         # GCE with A100 40GB
#   ./scripts/train_google.sh --gce --gpu t4           # GCE with T4 (cheapest)
#   ./scripts/train_google.sh --gce --gpu a100-80gb    # GCE with A100 80GB
#   ./scripts/train_google.sh --gce --keep             # Don't delete VM when done
#   ./scripts/train_google.sh --gce --push             # Also push adapters to HF Hub

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODE="colab"
GPU_PRESET="l4"
ZONE=""
KEEP=false
PUSH_HF=false
INSTANCE_NAME="ai-tuner-train"
MACHINE_TYPE=""
BOOT_DISK_SIZE="200GB"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gce)       MODE="gce"; shift ;;
    --gpu)       GPU_PRESET="$2"; shift 2 ;;
    --zone)      ZONE="$2"; shift 2 ;;
    --keep)      KEEP=true; shift ;;
    --push)      PUSH_HF=true; shift ;;
    --name)      INSTANCE_NAME="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Resolve repo info ───────────────────────────────────────────────────────
REPO_URL=$(git remote get-url origin 2>/dev/null || true)
BRANCH=$(git branch --show-current)

if [[ -z "$REPO_URL" ]]; then
  echo "Error: no git remote 'origin'. Run: git remote add origin <url> && git push"
  exit 1
fi

# Warn about uncommitted changes
if ! git diff --quiet HEAD 2>/dev/null; then
  echo "Warning: you have uncommitted changes that won't be available remotely."
  read -rp "Continue anyway? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
# COLAB MODE
# ═════════════════════════════════════════════════════════════════════════════
if [[ "$MODE" == "colab" ]]; then
  # Push latest code so Colab can clone it
  echo "Pushing latest code to origin/$BRANCH..."
  git push origin "$BRANCH"

  # Build the Colab URL — GitHub-hosted notebooks open directly in Colab
  # Convert git@github.com:user/repo.git or https://github.com/user/repo.git → user/repo
  REPO_PATH=$(echo "$REPO_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
  COLAB_URL="https://colab.research.google.com/github/${REPO_PATH}/blob/${BRANCH}/notebooks/train_colab.ipynb"

  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  Opening Colab notebook"
  echo "══════════════════════════════════════════════════════════════"
  echo "  Repo:   $REPO_PATH (branch: $BRANCH)"
  echo "  URL:    $COLAB_URL"
  echo ""
  echo "  Steps in Colab:"
  echo "    1. Runtime > Change runtime type > GPU (A100 recommended)"
  echo "    2. Runtime > Run all"
  echo "    3. Adapters are pushed to HF Hub automatically (cell 8)"
  echo "══════════════════════════════════════════════════════════════"
  echo ""

  # Open in default browser (macOS: open, Linux: xdg-open)
  if command -v open &>/dev/null; then
    open "$COLAB_URL"
  elif command -v xdg-open &>/dev/null; then
    xdg-open "$COLAB_URL"
  else
    echo "Open this URL in your browser:"
    echo "  $COLAB_URL"
  fi
  exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
# GCE MODE
# ═════════════════════════════════════════════════════════════════════════════

# ── GPU presets ──────────────────────────────────────────────────────────────
case "$GPU_PRESET" in
  t4)
    ACCELERATOR="type=nvidia-tesla-t4,count=1"
    MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"
    DEFAULT_ZONE="us-central1-a"
    ;;
  l4)
    ACCELERATOR="type=nvidia-l4,count=1"
    MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
    DEFAULT_ZONE="us-central1-a"
    ;;
  a100)
    ACCELERATOR="type=nvidia-tesla-a100,count=1"
    MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-1g}"
    DEFAULT_ZONE="us-central1-a"
    ;;
  a100-80gb)
    ACCELERATOR="type=nvidia-a100-80gb,count=1"
    MACHINE_TYPE="${MACHINE_TYPE:-a2-ultragpu-1g}"
    DEFAULT_ZONE="us-central1-c"
    ;;
  *)
    echo "Unknown GPU preset: $GPU_PRESET (use: t4, l4, a100, a100-80gb)"
    exit 1
    ;;
esac

ZONE="${ZONE:-$DEFAULT_ZONE}"

PROJECT=$(gcloud config get-value project 2>/dev/null)
if [[ -z "$PROJECT" ]]; then
  echo "Error: no GCP project set. Run: gcloud config set project YOUR_PROJECT"
  exit 1
fi

echo "══════════════════════════════════════════════════════════════"
echo "  Google Cloud Training (GCE)"
echo "══════════════════════════════════════════════════════════════"
echo "  Project:  $PROJECT"
echo "  Zone:     $ZONE"
echo "  GPU:      $GPU_PRESET ($ACCELERATOR)"
echo "  Machine:  $MACHINE_TYPE"
echo "  Repo:     $REPO_URL (branch: $BRANCH)"
echo "  Instance: $INSTANCE_NAME"
echo "  Push HF:  $PUSH_HF"
echo "  Keep VM:  $KEEP"
echo "══════════════════════════════════════════════════════════════"
echo ""
read -rp "Create VM and start training? [y/N] " confirm
[[ "$confirm" =~ ^[Yy]$ ]] || exit 0

# Push latest code
echo "Pushing latest code to origin/$BRANCH..."
git push origin "$BRANCH"

# ── Create VM ────────────────────────────────────────────────────────────────
echo ""
echo "Creating VM '$INSTANCE_NAME'..."

gcloud compute instances create "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="$ACCELERATOR" \
  --maintenance-policy=TERMINATE \
  --boot-disk-size="$BOOT_DISK_SIZE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --scopes=cloud-platform \
  --metadata="install-nvidia-driver=True" \
  --quiet

echo "VM created. Waiting for SSH to become available..."
sleep 30

# Retry SSH until ready
for i in $(seq 1 12); do
  if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="echo ready" 2>/dev/null; then
    break
  fi
  echo "  SSH not ready, retrying in 10s... ($i/12)"
  sleep 10
done

# ── Run training on VM ──────────────────────────────────────────────────────
echo ""
echo "Setting up and running training on VM..."

PUSH_FLAG=""
if [[ "$PUSH_HF" == "true" ]]; then
  PUSH_FLAG="--push"
fi

gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" -- bash -s <<REMOTE_SCRIPT
set -euo pipefail

echo "── Cloning repo ──"
git clone --depth 1 --branch "$BRANCH" "$REPO_URL" ~/ai-tuner
cd ~/ai-tuner

echo "── Checking GPU ──"
nvidia-smi

echo "── Installing dependencies ──"
pip install -q --upgrade uv
uv pip install -q --system \
    unsloth \
    torch \
    transformers==4.56.2 \
    peft>=0.13.0 \
    bitsandbytes>=0.44.0 \
    datasets>=2.14.0 \
    accelerate>=1.0.0 \
    xformers \
    pyyaml "typer[all]" tqdm "huggingface_hub[cli]"
uv pip install -q --system --no-deps trl==0.22.2
pip install -q flash-attn --no-build-isolation 2>/dev/null \
    || echo "⚠ flash-attn not available — using eager attention"

echo "── Preparing training data ──"
python src/data/prepare_data.py --config config.yaml

echo "── Starting training ──"
python src/train/train_cuda.py --config config.yaml

echo "── Training complete! ──"
ls -lh output/adapters/

if [[ "$PUSH_FLAG" == "--push" ]]; then
  echo "── Pushing adapters to HuggingFace ──"
  python src/cli.py push
fi

echo "── Packing adapters for download ──"
tar czf /tmp/adapters.tar.gz -C output adapters/
echo "TRAINING_DONE"
REMOTE_SCRIPT

# ── Download adapters ────────────────────────────────────────────────────────
echo ""
echo "Downloading adapters..."
mkdir -p output
gcloud compute scp "$INSTANCE_NAME":/tmp/adapters.tar.gz /tmp/adapters.tar.gz --zone="$ZONE"
tar xzf /tmp/adapters.tar.gz -C output/
rm /tmp/adapters.tar.gz
echo "Adapters saved to output/adapters/"

# ── Cleanup ──────────────────────────────────────────────────────────────────
if [[ "$KEEP" == "false" ]]; then
  echo ""
  echo "Deleting VM '$INSTANCE_NAME'..."
  gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
  echo "VM deleted."
else
  echo ""
  echo "VM '$INSTANCE_NAME' is still running (--keep). Delete manually with:"
  echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
fi

echo ""
echo "Done! Adapters are in output/adapters/"
echo "Next steps:"
echo "  task export        # fuse adapters into output/merged/"
echo "  task export:gguf   # convert to GGUF"
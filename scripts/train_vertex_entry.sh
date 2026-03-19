#!/bin/bash
# Entry point for Vertex AI Custom Training container.
# Runs inside Google's pre-built PyTorch GPU container.
#
# Usage (called by train_vertex.sh, not directly):
#   bash train_vertex_entry.sh gs://bucket/jobs/job-id
set -euo pipefail

GCS_BASE="${1:?Usage: train_vertex_entry.sh GCS_BASE}"

echo "── Environment ──"
nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

echo "── Copying input from GCS ──"
mkdir -p /workspace/project
gsutil -m rsync -r "${GCS_BASE}/input/" /workspace/project/
cd /workspace/project

echo "── Installing dependencies ──"
pip install -q --upgrade uv
uv pip install -q --system \
    unsloth \
    transformers==4.56.2 \
    peft">=0.13.0" \
    bitsandbytes">=0.44.0" \
    datasets">=2.14.0" \
    accelerate">=1.0.0" \
    xformers \
    pyyaml "typer[all]" tqdm "huggingface_hub[cli]"
uv pip install -q --system --no-deps trl==0.22.2
pip install -q flash-attn --no-build-isolation 2>/dev/null \
    || echo "⚠ flash-attn not available — using eager attention"

echo "── Starting training ──"
python src/train/train_cuda.py --config config.yaml

echo "── Uploading adapters to GCS ──"
gsutil -m rsync -r output/adapters/ "${GCS_BASE}/output/adapters/"

echo "VERTEX_TRAINING_DONE"

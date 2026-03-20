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
    "torch>=2.8.0" "triton>=3.4.0" torchvision bitsandbytes \
    "transformers==4.56.2" \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install -q --system --upgrade --no-deps \
    transformers==4.56.2 tokenizers trl==0.22.2 unsloth unsloth_zoo
uv pip install -q --system \
    "peft>=0.13.0" "accelerate>=1.0.0" "datasets>=2.14.0" \
    pyyaml "typer[all]" tqdm "huggingface_hub[cli]"
pip install -q flash-attn --no-build-isolation 2>/dev/null \
    || echo "⚠ flash-attn not available — using eager attention"

echo "── Starting training ──"
python src/train/train_cuda.py --config config.yaml

echo "── Uploading adapters to GCS ──"
gsutil -m rsync -r output/adapters/ "${GCS_BASE}/output/adapters/"

echo "VERTEX_TRAINING_DONE"

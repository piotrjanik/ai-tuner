#!/usr/bin/env bash
# Train on Vertex AI Custom Training.
#
# Stages code + data to GCS, submits a custom training job with a pre-built
# PyTorch container, streams logs, and downloads adapters when done.
#
# Usage:
#   ./scripts/train_vertex.sh                          # L4 GPU (default)
#   ./scripts/train_vertex.sh --gpu a100               # A100 40GB
#   ./scripts/train_vertex.sh --gpu a100-80gb          # A100 80GB
#   ./scripts/train_vertex.sh --bucket my-bucket       # Custom GCS bucket
#   ./scripts/train_vertex.sh --region europe-west4    # Custom region
#   ./scripts/train_vertex.sh --push                   # Push adapters to HF Hub after
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
GPU_PRESET="l4"
REGION=""
BUCKET=""
PUSH_HF=false
CONTAINER_IMAGE="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-5:latest"

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)      GPU_PRESET="$2"; shift 2 ;;
    --region)   REGION="$2"; shift 2 ;;
    --bucket)   BUCKET="$2"; shift 2 ;;
    --push)     PUSH_HF=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Resolve project ──────────────────────────────────────────────────────────
PROJECT=$(gcloud config get-value project 2>/dev/null)
if [[ -z "$PROJECT" ]]; then
  echo "Error: no GCP project set. Run: gcloud config set project YOUR_PROJECT"
  exit 1
fi

# ── GPU presets (Vertex AI accelerator types) ────────────────────────────────
case "$GPU_PRESET" in
  l4)
    ACCEL_TYPE="NVIDIA_L4"
    MACHINE_TYPE="g2-standard-12"
    DEFAULT_REGION="us-central1"
    ;;
  a100)
    ACCEL_TYPE="NVIDIA_TESLA_A100"
    MACHINE_TYPE="a2-highgpu-1g"
    DEFAULT_REGION="us-central1"
    ;;
  a100-80gb)
    ACCEL_TYPE="NVIDIA_A100_80GB"
    MACHINE_TYPE="a2-ultragpu-1g"
    DEFAULT_REGION="us-central1"
    ;;
  *)
    echo "Unknown GPU preset: $GPU_PRESET (use: l4, a100, a100-80gb)"
    exit 1
    ;;
esac

REGION="${REGION:-$DEFAULT_REGION}"
BUCKET="${BUCKET:-${PROJECT}-ai-training}"
JOB_ID="ocm-train-$(date +%Y%m%d-%H%M%S)"
GCS_BASE="gs://${BUCKET}/jobs/${JOB_ID}"

# ── Verify data exists ──────────────────────────────────────────────────────
if [[ ! -f data/train.jsonl ]]; then
  echo "Error: data/train.jsonl not found. Run 'task data' first."
  exit 1
fi

echo "══════════════════════════════════════════════════════════════"
echo "  Vertex AI Custom Training"
echo "══════════════════════════════════════════════════════════════"
echo "  Project:    $PROJECT"
echo "  Region:     $REGION"
echo "  GPU:        $GPU_PRESET ($ACCEL_TYPE)"
echo "  Machine:    $MACHINE_TYPE"
echo "  Container:  $CONTAINER_IMAGE"
echo "  GCS:        $GCS_BASE"
echo "  Job ID:     $JOB_ID"
echo "  Push HF:    $PUSH_HF"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Ensure GCS bucket exists ────────────────────────────────────────────────
if ! gsutil ls "gs://${BUCKET}" &>/dev/null; then
  echo "Creating GCS bucket gs://${BUCKET} in ${REGION}..."
  gsutil mb -l "$REGION" "gs://${BUCKET}"
fi

# ── Stage code + data to GCS ────────────────────────────────────────────────
echo "Staging code and data to ${GCS_BASE}/input/..."
gsutil -m rsync -r src/ "${GCS_BASE}/input/src/"
gsutil cp config.yaml "${GCS_BASE}/input/"
gsutil -m cp data/train.jsonl data/val.jsonl "${GCS_BASE}/input/data/"
gsutil cp scripts/train_vertex_entry.sh "${GCS_BASE}/input/"
echo "Staging complete."

# ── Submit Vertex AI custom job ─────────────────────────────────────────────
echo ""
echo "Submitting Vertex AI custom training job..."

gcloud ai custom-jobs create \
  --region="$REGION" \
  --display-name="$JOB_ID" \
  --worker-pool-spec="\
machine-type=${MACHINE_TYPE},\
accelerator-type=${ACCEL_TYPE},\
accelerator-count=1,\
replica-count=1,\
container-image-uri=${CONTAINER_IMAGE}" \
  --command="bash" \
  --args="${GCS_BASE}/input/train_vertex_entry.sh,${GCS_BASE}"

# ── Get job resource name ───────────────────────────────────────────────────
JOB_NAME=$(gcloud ai custom-jobs list \
  --region="$REGION" \
  --filter="displayName=$JOB_ID" \
  --format="value(name)" \
  --limit=1)

if [[ -z "$JOB_NAME" ]]; then
  echo "Error: could not find submitted job. Check the Vertex AI console."
  exit 1
fi

echo "Job submitted: $JOB_NAME"
echo ""

# ── Stream logs ─────────────────────────────────────────────────────────────
echo "Streaming logs (Ctrl+C to stop streaming — job continues in the cloud)..."
echo ""

gcloud ai custom-jobs stream-logs "$JOB_NAME" --region="$REGION" || true

# ── Wait for completion ─────────────────────────────────────────────────────
echo ""
echo "Checking final job state..."
while true; do
  STATE=$(gcloud ai custom-jobs describe "$JOB_NAME" \
    --region="$REGION" \
    --format="value(state)")
  case "$STATE" in
    JOB_STATE_SUCCEEDED)
      echo "Job succeeded!"
      break
      ;;
    JOB_STATE_FAILED|JOB_STATE_CANCELLED)
      echo "Job $STATE. Check logs:"
      echo "  gcloud ai custom-jobs describe $JOB_NAME --region=$REGION"
      exit 1
      ;;
    *)
      echo "  State: $STATE — waiting 30s..."
      sleep 30
      ;;
  esac
done

# ── Download adapters ───────────────────────────────────────────────────────
echo ""
echo "Downloading adapters..."
mkdir -p output/adapters
gsutil -m rsync -r "${GCS_BASE}/output/adapters/" output/adapters/
echo "Adapters saved to output/adapters/"

# ── Push to HuggingFace ─────────────────────────────────────────────────────
if [[ "$PUSH_HF" == "true" ]]; then
  echo ""
  echo "Pushing adapters to HuggingFace..."
  python src/cli.py push
fi

echo ""
echo "Done! Adapters are in output/adapters/"
echo "Next steps:"
echo "  task export        # fuse adapters into output/merged/"
echo "  task export:gguf   # convert to GGUF"

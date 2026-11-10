#!/usr/bin/env bash
# Build (optionally) and run the MLX DeepSeek-V4.1 container on one DGX Spark.
#
# The host needs Docker and the NVIDIA container runtime only. CUDA, Zig, and
# TensorFold live in the image; nothing is installed on the host.
#
# Usage:
#   scripts/mlx/run_container.sh --help
#   BUILD=1 scripts/mlx/run_container.sh tensorfold-dsv41 --help
#   scripts/mlx/run_container.sh bash
#   MODEL_DIR=/models/mlx scripts/mlx/run_container.sh tensorfold-dsv41 serve /model
#
# Environment:
#   IMAGE        image tag (default tf-dsv41-mlx:dev)
#   PLATFORM     docker platform (default linux/arm64)
#   BASE_IMAGE   CUDA/PyTorch base (default nvcr.io/nvidia/pytorch:26.07-py3)
#   TF_REPO      TensorFold repository (default upstream)
#   TF_REF       TensorFold commit (default pinned upstream commit)
#   MODEL_DIR    checkpoint directory (mounted at /model, read-only)
#   ENGRAM_DIR   Engram directory (mounted at /engram, read-only; defaults to MODEL_DIR)
#   HF_CACHE     Hugging Face cache (mounted at /hf)
#   OUT_DIR      output directory (mounted at /out)
#   ASSETS_HOST  optional host assets directory (mounted at /assets, read-only)
#   ASSETS_DIR   engine assets path inside the container (default /assets if ASSETS_HOST is set, else /model)
#   BUILD=1      build the image before running
#   DRY_RUN=1    print the docker command instead of running it
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE=${IMAGE:-tf-dsv41-mlx:dev}
PLATFORM=${PLATFORM:-linux/arm64}
BASE_IMAGE=${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}
TF_REPO=${TF_REPO:-https://github.com/ashhart/TensorFold.git}
TF_REF=${TF_REF:-26e1b074b1512d645698eba269ccfb8ebc974c10}
MODEL_DIR=${MODEL_DIR:-$HOME/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-MLX}
ENGRAM_DIR=${ENGRAM_DIR:-$MODEL_DIR}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
OUT_DIR=${OUT_DIR:-$ROOT/out}
ASSETS_HOST=${ASSETS_HOST:-}
ASSETS_DIR=${ASSETS_DIR:-}
SHM_SIZE=${SHM_SIZE:-16g}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

if [[ "${BUILD:-0}" == "1" ]]; then
  docker build -f "$ROOT/docker/Dockerfile.mlx" \
    --platform "$PLATFORM" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_REPO="$TF_REPO" \
    --build-arg TF_REF="$TF_REF" \
    -t "$IMAGE" "$ROOT"
fi

mkdir -p "$OUT_DIR"
args=(--rm --gpus all --ipc=host --shm-size="$SHM_SIZE"
  --ulimit memlock=-1 --ulimit stack=67108864
  -v "$MODEL_DIR:/model:ro"
  -v "$HF_CACHE:/hf"
  -v "$OUT_DIR:/out"
  -e HF_HOME=/hf)

if [[ -d "$ENGRAM_DIR" ]]; then
  args+=(-v "$ENGRAM_DIR:/engram:ro")
fi

if [[ -n "$ASSETS_HOST" ]]; then
  args+=(-v "$ASSETS_HOST:/assets:ro")
  ASSETS_DIR=${ASSETS_DIR:-/assets}
fi
ASSETS_DIR=${ASSETS_DIR:-/model}
args+=(-e TF_DSV41_ASSETS="$ASSETS_DIR")

# Pass the engine's own environment through to the container (layers, drafts, world, ...).
while IFS= read -r v; do
  case "$v" in
    TF_DSV41_*|TF_TP_*|TF_DS_*|TENSORFOLD_*) args+=(-e "$v=${!v}") ;;
  esac
done < <(compgen -e)

cmd=(docker run --platform "$PLATFORM" "${args[@]}" "$IMAGE")

if [[ $# -eq 0 ]]; then
  cmd+=(tensorfold-dsv41 --help)
else
  cmd+=("$@")
fi

printf 'running:' >&2
printf ' %q' "${cmd[@]}" >&2
printf '\n' >&2

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${cmd[@]}"

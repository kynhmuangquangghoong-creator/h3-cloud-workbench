#!/usr/bin/env bash
# Ephemeral one-shot ComfyUI Cloud Workbench worker for 5090 (0 persistent cache, auto-clean via --rm).
# Usage:
#   WORKER_TOKEN="xxx" HF_TOKEN="hf_xxx" bash ~/run_comfy_once.sh
# Optional env:
#   IMAGE="ghcr.io/kynhmuangquangghoong-creator/h3-cloud-workbench:latest"
#   GPU_SPEC="all"   (or "device=0")
#   IDLE_SHUTDOWN_SEC="180"
set -euo pipefail

: "${WORKER_TOKEN:?WORKER_TOKEN is required}"
IMAGE="${IMAGE:-ghcr.io/kynhmuangquangghoong-creator/h3-cloud-workbench:latest}"
GPU_SPEC="${GPU_SPEC:-all}"
IDLE_SHUTDOWN_SEC="${IDLE_SHUTDOWN_SEC:-180}"
CONTROL_PLANE_URL="${CONTROL_PLANE_URL:-https://gpu-workbench.mamegoya6900.workers.dev}"

echo "[5090-ONCE] Pulling/starting ephemeral container (${IMAGE}, gpus=${GPU_SPEC})..."
echo "[5090-ONCE] Will auto-exit & wipe all temporary weights after ${IDLE_SHUTDOWN_SEC}s idle."

exec docker run --rm --gpus "${GPU_SPEC}" \
  -e WORKER_PLATFORM=local5090 \
  -e CONTROL_PLANE_URL="${CONTROL_PLANE_URL}" \
  -e WORKER_TOKEN="${WORKER_TOKEN}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e EXIT_ON_IDLE=1 \
  -e IDLE_SHUTDOWN_SEC="${IDLE_SHUTDOWN_SEC}" \
  "${IMAGE}"

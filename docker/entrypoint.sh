#!/usr/bin/env bash
set -euo pipefail

WORKBENCH_DIR="${WORKBENCH_DIR:-/opt/workbench}"
COMFY_DIR="${COMFY_DIR:-/opt/ComfyUI}"
PLATFORM="${WORKER_PLATFORM:-runpod}"

echo "[BOOT] Starting H3 Cloud Workbench (platform=${PLATFORM}, comfy=${COMFY_DIR})"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Pull latest universal_worker.py from Cloudflare control plane if available (single source of truth)
if [ -n "${CONTROL_PLANE_URL:-}" ] && [ -n "${WORKER_TOKEN:-}" ]; then
  echo "[BOOT] Syncing latest universal_worker.py from control plane..."
  curl -fsSL -H "Authorization: Bearer ${WORKER_TOKEN}" \
    "${CONTROL_PLANE_URL%/}/api/assets/runner/universal_worker.py" \
    -o "${WORKBENCH_DIR}/universal_worker.py.new" && \
    mv "${WORKBENCH_DIR}/universal_worker.py.new" "${WORKBENCH_DIR}/universal_worker.py" || \
    echo "[BOOT] Using bundled universal_worker.py fallback."
fi

exec python3 -u "${WORKBENCH_DIR}/universal_worker.py" \
  --platform "${PLATFORM}" \
  --comfy-dir "${COMFY_DIR}" \
  --work-dir "/tmp/h3_work"

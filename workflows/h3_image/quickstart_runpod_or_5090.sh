#!/usr/bin/env bash
# Quickstart bootstrap for MiniMax-H3-Image (sglang[diffusion], ~140 GB public weights)
# Works on any Linux GPU host (5090 / Runpod Pod with >=200 GB container disk).
set -euo pipefail

ROOT="${H3IMAGE_ROOT:-/workspace/h3image}"
mkdir -p "$ROOT/logs" "$ROOT/models" "$ROOT/outputs"
cp -f "$(dirname "$0")"/remote_scripts/* "$ROOT/"
cd "$ROOT"

echo "[1/3] Installing sglang[diffusion]..."
pip install -q --upgrade pip uv
uv pip install --system "sglang[diffusion]"

echo "[2/3] Downloading public MiniMax-H3-Image weights (~140 GB, SHA256 verified)..."
python3 "$ROOT/h3_download.py"

echo "[3/3] Launching MiniMax-H3-Image on 127.0.0.1:30083 (single GPU, peak ~12.5 GB VRAM)..."
exec sglang serve \
  --model-path "$ROOT/models/MiniMax-H3-Image" \
  --host 127.0.0.1 --port 30083 \
  --num-gpus 1 \
  --layerwise-offload-components dit text_encoder vae \
  --minimax-h3-adaln-online true \
  --dit-offload-prefetch-size 1

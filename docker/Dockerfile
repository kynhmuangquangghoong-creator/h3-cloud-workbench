# Lightweight zero-volume Runpod image for MiniMax H3 Cloud Workbench
# Pre-installs ComfyUI + 4 open-source H3 custom nodes + Python deps + HF transfer tools.
# Weights (~44 GB) are fetched at pod boot directly from HuggingFace to local NVMe container disk.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFY_DIR=/opt/ComfyUI \
    WORKBENCH_DIR=/opt/workbench \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg aria2 curl unzip rclone libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 1. Clone ComfyUI & install core dependencies
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ${COMFY_DIR} \
    && pip install --upgrade pip \
    && pip install -r ${COMFY_DIR}/requirements.txt \
    && pip install av huggingface_hub hf_transfer websocket-client requests

# 2. Pre-install the 4 open-source H3 nodes + 3 audio nodes (verified on 5090/Colab)
WORKDIR ${COMFY_DIR}/custom_nodes
RUN git clone --depth 1 https://github.com/facok/comfyui-SelfLift.git comfyui-SelfLift \
    && git clone --depth 1 https://github.com/goohai/Goohaitools-comfyui.git Goohaitools-comfyui \
    && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git ComfyUI-KJNodes \
    && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git ComfyUI-VideoHelperSuite \
    && git clone --depth 1 https://github.com/smthemex/ComfyUI_Demucs.git Demucs \
    && git clone --depth 1 https://github.com/shinrali/ComfyUI-Qwen3-TTS.git Qwen3-TTS \
    && git clone --depth 1 https://github.com/1038lab/ComfyUI-QwenASR.git QwenASR \
    && for d in comfyui-SelfLift Goohaitools-comfyui ComfyUI-KJNodes ComfyUI-VideoHelperSuite Demucs Qwen3-TTS QwenASR; do \
         if [ -f "$d/requirements.txt" ]; then pip install -r "$d/requirements.txt" || true; fi; \
       done

# 3. Stage bootstrap entrypoint & fallback worker
RUN mkdir -p ${WORKBENCH_DIR}
COPY docker/entrypoint.sh ${WORKBENCH_DIR}/entrypoint.sh
COPY runner/universal_worker.py ${WORKBENCH_DIR}/universal_worker.py
RUN chmod +x ${WORKBENCH_DIR}/entrypoint.sh

WORKDIR ${WORKBENCH_DIR}
ENTRYPOINT ["/opt/workbench/entrypoint.sh"]

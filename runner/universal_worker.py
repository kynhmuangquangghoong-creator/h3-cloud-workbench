#!/usr/bin/env python3
"""Universal GPU Worker for H3 Cloud Workbench (Runpod / Colab / Local 5090).

Single codebase across all compute tiers:
- Syncs models_manifest.json & h3_relay_bundle.zip from Cloudflare control plane (R2).
- Downloads missing weights directly from HuggingFace to local NVMe (or links Drive on Colab).
- Starts ComfyUI headless, claims queued jobs, reports live heartbeats (phase, stage, VRAM, logs).
- Stitches final video, transcodes lightweight preview MP4 back to R2, optionally archives to Drive.
- On Runpod: requests self-termination via control plane after idle timeout to stop billing.
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

COMFY_URL = "http://127.0.0.1:8188"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
AUD_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def get_gpu_telemetry() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_name": parts[0],
            "vram_used_mb": int(float(parts[1])),
            "vram_total_mb": int(float(parts[2])),
            "gpu_util_pct": int(float(parts[3])),
        }
    except Exception:
        return {"gpu_name": "unknown", "vram_used_mb": 0, "vram_total_mb": 0, "gpu_util_pct": 0}


class ControlPlaneClient:
    def __init__(self, base_url: str, token: str, worker_id: str, platform: str, pod_id: str = ""):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.worker_id = worker_id
        self.platform = platform
        self.pod_id = pod_id

    def _req(self, method: str, path: str, body=None, content_type="application/json", timeout=60):
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": "H3UniversalWorker/1.0"}
        if body is not None:
            if isinstance(body, (dict, list)):
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                headers["Content-Type"] = "application/json"
            elif isinstance(body, (bytes, bytearray)):
                data = bytes(body)
                headers["Content-Type"] = content_type
            else:
                data = str(body).encode("utf-8")
                headers["Content-Type"] = content_type
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def get_json(self, path: str) -> dict:
        return json.loads(self._req("GET", path).decode("utf-8"))

    def post_json(self, path: str, payload: dict) -> dict:
        raw = self._req("POST", path, payload)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def download_asset(self, r2_key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/api/assets/{r2_key.lstrip('/')}"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.token}", "User-Agent": "H3UniversalWorker/1.0"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)

    def upload_job_file(self, job_id: str, filename: str, file_path: Path, content_type="application/octet-stream") -> dict:
        url = f"{self.base_url}/api/jobs/{job_id}/upload/{filename}"
        data = file_path.read_bytes()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": content_type,
                "User-Agent": "H3UniversalWorker/1.0",
            },
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def worker_heartbeat(self, state: str, detail: str = "", job_id: str = "") -> None:
        try:
            self.post_json(
                "/api/workers/heartbeat",
                {
                    "worker_id": self.worker_id,
                    "platform": self.platform,
                    "pod_id": self.pod_id,
                    "state": state,
                    "detail": detail[:400],
                    "current_job_id": job_id,
                    "telemetry": get_gpu_telemetry(),
                },
            )
        except Exception as exc:
            print(f"[WARN] worker_heartbeat failed: {exc}")

    def job_heartbeat(self, job_id: str, phase: str, progress: float, detail: str = "", stage: str = "") -> None:
        try:
            self.post_json(
                f"/api/jobs/{job_id}/heartbeat",
                {
                    "worker_id": self.worker_id,
                    "phase": phase,
                    "stage": stage,
                    "progress": round(progress, 3),
                    "detail": detail[:500],
                    "telemetry": get_gpu_telemetry(),
                },
            )
        except Exception as exc:
            print(f"[WARN] job_heartbeat failed: {exc}")

    def claim_job(self) -> dict | None:
        res = self.post_json(
            "/api/jobs/claim",
            {"worker_id": self.worker_id, "platform": self.platform, "pod_id": self.pod_id},
        )
        return res.get("job")

    def complete_job(self, job_id: str, result_payload: dict) -> None:
        self.post_json(f"/api/jobs/{job_id}/complete", result_payload)

    def fail_job(self, job_id: str, error_msg: str) -> None:
        self.post_json(f"/api/jobs/{job_id}/fail", {"worker_id": self.worker_id, "error": error_msg[:4000]})

    def request_self_terminate(self, reason: str = "idle_timeout") -> None:
        if not self.pod_id:
            return
        print(f"[SHUTDOWN] Requesting control plane to terminate Runpod pod {self.pod_id} ({reason})...")
        try:
            self.post_json(
                "/api/pods/self-terminate",
                {"pod_id": self.pod_id, "worker_id": self.worker_id, "reason": reason},
            )
        except Exception as exc:
            print(f"[WARN] self-terminate request failed: {exc}")


def sync_workflow_bundle(cp: ControlPlaneClient, comfy_dir: Path, work_dir: Path) -> Path:
    wf_dir = work_dir / "workflows" / "h3_relay"
    wf_dir.mkdir(parents=True, exist_ok=True)
    bundle_zip = work_dir / "h3_relay_bundle.zip"
    cp.worker_heartbeat("bootstrapping", "Downloading h3_relay_bundle.zip from R2")
    cp.download_asset("workflows/h3_relay/h3_relay_bundle.zip", bundle_zip)
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        zf.extractall(wf_dir)

    plugin_zip = wf_dir / "h3_relay_plugin.zip"
    if plugin_zip.is_file():
        plugin_dst = comfy_dir / "custom_nodes" / "ComfyUI-H3-Latent-Relay"
        plugin_dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(plugin_zip, "r") as zf:
            zf.extractall(plugin_dst)
        print(f"[OK] Installed ComfyUI-H3-Latent-Relay -> {plugin_dst}")
    return wf_dir


def _download_with_aria2(url: str, dest: Path, hf_token: str | None) -> bool:
    if not shutil.which("aria2c"):
        return False
    tmp_dir = dest.parent / "_aria_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_name = f"{dest.name}.part"
    cmd = [
        "aria2c",
        "--console-log-level=warn",
        "--summary-interval=0",
        "--allow-overwrite=true",
        "--file-allocation=none",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=4M",
        "--max-tries=5",
        "--retry-wait=2",
        "-d", str(tmp_dir),
        "-o", tmp_name,
    ]
    if hf_token:
        cmd += ["--header", f"Authorization: Bearer {hf_token}"]
    cmd.append(url)
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    part_path = tmp_dir / tmp_name
    if res.returncode == 0 and part_path.is_file() and part_path.stat().st_size > 1_000_000:
        shutil.move(str(part_path), str(dest))
        return True
    part_path.unlink(missing_ok=True)
    return False


def ensure_models(
    cp: ControlPlaneClient,
    comfy_dir: Path,
    manifest: dict,
    models_override_dir: str = "",
    workflow: str = "h3_relay",
    allowed_tiers: set[str] | None = None,
) -> None:
    models = []
    for m in manifest.get("models", []):
        if workflow not in m.get("workflows", []):
            continue
        tier = m.get("tier", "core")
        if allowed_tiers is not None and tier not in allowed_tiers:
            continue
        models.append(m)

    if not models:
        return

    hf_token = os.environ.get("HF_TOKEN") or None
    from huggingface_hub import hf_hub_download

    total = len(models)
    done = 0
    lock = threading.Lock()
    t0 = time.time()

    def fetch_one(m: dict):
        nonlocal done
        target_dir = comfy_dir / m["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        dst_file = target_dir / m["local_name"]

        # Optional symlink reuse (e.g. Colab Google Drive or Local 5090 model dir)
        if models_override_dir:
            sub = m["target_dir"].replace("models/", "", 1)
            candidate = Path(models_override_dir) / sub / m["local_name"]
            if candidate.is_file():
                if not dst_file.exists():
                    os.symlink(candidate, dst_file)
                with lock:
                    done += 1
                    cp.worker_heartbeat("downloading_models", f"[{done}/{total}] Linked {m['local_name']}")
                return

        if dst_file.is_file() and dst_file.stat().st_size > 1_000_000:
            with lock:
                done += 1
                cp.worker_heartbeat("downloading_models", f"[{done}/{total}] Cached {m['local_name']}")
            return

        print(f"[HF] Downloading {m['repo_id']} :: {m['filename']} (~{m.get('approx_gb', '?')} GB)...")
        cp.worker_heartbeat("downloading_models", f"[{done}/{total}] Fetching {m['local_name']} ({m.get('approx_gb', '?')} GB)")

        # Fast path: 16-connection aria2c chunked transfer; fallback to hf_hub_download (Xet / hf_transfer)
        direct_url = f"https://huggingface.co/{m['repo_id']}/resolve/main/{m['filename']}"
        ok = _download_with_aria2(direct_url, dst_file, hf_token)
        if not ok:
            cached_path = hf_hub_download(
                repo_id=m["repo_id"],
                filename=m["filename"],
                token=hf_token,
                local_dir=str(target_dir / "_hf_tmp" / m["id"]),
            )
            shutil.move(cached_path, dst_file)
            shutil.rmtree(target_dir / "_hf_tmp" / m["id"], ignore_errors=True)

        with lock:
            done += 1
            elapsed = time.time() - t0
            print(f"[HF] Done [{done}/{total}] {m['local_name']} ({'aria2c-16x' if ok else 'hf_hub'}, {elapsed:.1f}s)")
            cp.worker_heartbeat("downloading_models", f"[{done}/{total}] Ready {m['local_name']} ({elapsed:.0f}s)")

    # 4 parallel workers x 16 connections saturates datacenter 1-2 GB/s NIC
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(fetch_one, m) for m in models]
        for fut in futs:
            fut.result()


def launch_comfyui_async(comfy_dir: Path, log_path: Path) -> subprocess.Popen:
    log_f = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-u", str(comfy_dir / "main.py"), "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"],
        cwd=str(comfy_dir),
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )


def wait_comfyui_ready(proc: subprocess.Popen, log_path: Path, timeout_sec: int = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/object_info/MiniMaxH3AddGuide", timeout=2) as r:
                data = json.loads(r.read().decode("utf-8"))
                if "MiniMaxH3AddGuide" in data:
                    print(f"[OK] ComfyUI ready ({time.time() - t0:.1f}s wait after model sync)")
                    return
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early with code {proc.returncode}. Check {log_path}")
        time.sleep(1.5)
    raise TimeoutError("Timed out waiting for ComfyUI on 127.0.0.1:8188")


def make_silent_wav(path: Path, seconds: float = 2.0) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(seconds), str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def adapt_graph(graph: dict, comfy_dir: Path) -> dict:
    models_root = comfy_dir / "models"
    avail_clips = sorted(p.name for p in (models_root / "text_encoders").glob("*.safetensors")) if (models_root / "text_encoders").exists() else []
    avail_ups = sorted(p.name for p in (models_root / "latent_upscale_models").glob("*.safetensors")) if (models_root / "latent_upscale_models").exists() else []
    avail_loras = {p.name for p in (models_root / "loras").glob("*.safetensors")} if (models_root / "loras").exists() else set()

    removed = {}
    for nid, node in list(graph.items()):
        if node.get("class_type") == "LoraLoaderModelOnly":
            lname = node.get("inputs", {}).get("lora_name", "")
            if lname and lname not in avail_loras:
                removed[nid] = node.get("inputs", {}).get("model")
                del graph[nid]
    if removed:
        def resolve(ref):
            while isinstance(ref, list) and len(ref) == 2 and ref[0] in removed:
                ref = removed[ref[0]]
            return ref
        for node in graph.values():
            for k, v in list(node.get("inputs", {}).items()):
                if isinstance(v, list) and len(v) == 2 and v[0] in removed:
                    node["inputs"][k] = resolve(v)

    for nid, node in graph.items():
        ct = node.get("class_type")
        ins = node.get("inputs", {})
        if ct == "CLIPLoader" and avail_clips and ins.get("clip_name") not in avail_clips:
            ins["clip_name"] = avail_clips[0]
        elif ct == "SelfLiftH3Sampler" and avail_ups and ins.get("upscaler_model") not in avail_ups:
            ins["upscaler_model"] = avail_ups[0]
    return graph


def queue_and_wait(
    graph: dict,
    comfy_dir: Path,
    cp: ControlPlaneClient,
    job_id: str,
    stage_idx: int,
    total_stages: int,
) -> dict:
    client_id = uuid.uuid4().hex
    graph = adapt_graph(graph, comfy_dir)
    payload = json.dumps({"prompt": graph, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    prompt_id = data["prompt_id"]

    t0 = time.time()
    last_hb = 0.0
    while True:
        with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}", timeout=15) as resp:
            hist = json.loads(resp.read().decode("utf-8"))
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI error: {json.dumps(status, ensure_ascii=False)}")
            if status.get("completed", False) or entry.get("outputs"):
                return entry
        now = time.time()
        if now - last_hb >= 6.0:
            elapsed = int(now - t0)
            base_prog = stage_idx / max(1, total_stages)
            cp.job_heartbeat(
                job_id,
                phase="generating",
                progress=min(0.95, base_prog + 0.5 / max(1, total_stages)),
                stage=f"Stage {stage_idx + 1}/{total_stages}",
                detail=f"Running ComfyUI prompt {prompt_id[:8]} ({elapsed}s elapsed)",
            )
            last_hb = now
        time.sleep(3.0)


def stitch_and_preview(segment_mp4s: list[Path], final_mp4: Path, preview_mp4: Path) -> None:
    if len(segment_mp4s) == 1:
        shutil.copy2(segment_mp4s[0], final_mp4)
    else:
        concat_txt = final_mp4.with_suffix(".concat.txt")
        concat_txt.write_text("\n".join(f"file '{p.as_posix()}'" for p in segment_mp4s), encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(final_mp4)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        concat_txt.unlink(missing_ok=True)

    # Lightweight preview MP4 (<= 20 MB for R2 free-tier web playback)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(final_mp4),
            "-vf", "scale='min(1280,iw)':-2",
            "-c:v", "libx264", "-preset", "fast", "-crf", "26",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(preview_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def execute_h3_job(
    job: dict,
    wf_dir: Path,
    comfy_dir: Path,
    work_dir: Path,
    cp: ControlPlaneClient,
    manifest: dict,
    models_override_dir: str = "",
) -> dict:
    job_id = job["id"]
    t_start = time.time()
    params_in = job.get("params", {})
    resolution = params_in.get("resolution", "1344x768")

    # Lazy model loading: only fetch 2K upscaler if the job actually requests 2K
    needed_tiers = {"core", "lora"}
    if resolution == "2048x1152":
        needed_tiers.add("upscale_2k")
    ensure_models(cp, comfy_dir, manifest, models_override_dir, workflow=job.get("workflow", "h3_relay"), allowed_tiers=needed_tiers)

    if str(wf_dir) not in sys.path:
        sys.path.insert(0, str(wf_dir))
    import h3_workflow
    import relay_mode

    comfy_input = comfy_dir / "input"
    comfy_output = comfy_dir / "output"
    comfy_input.mkdir(parents=True, exist_ok=True)

    job_dir = work_dir / "jobs" / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    cp.job_heartbeat(job_id, phase="downloading_inputs", progress=0.05, detail="Downloading reference assets from R2")
    input_keys = job.get("input_keys", [])
    imgs, auds = [], []
    for key in input_keys:
        fname = Path(key).name
        local_f = input_dir / fname
        cp.download_asset(key, local_f)
        ext = local_f.suffix.lower()
        if ext in IMG_EXTS:
            imgs.append(local_f)
        elif ext in AUD_EXTS:
            auds.append(local_f)

    if not imgs:
        raise ValueError("任务缺少参考图片（请至少上传 1 张 png/jpg/webp）")

    run_id = f"job_{job_id[:10]}"
    references = {}
    keyframes = []
    for idx, src in enumerate(imgs[:9]):
        safe_name = f"{run_id}_img_{idx}{src.suffix.lower()}"
        shutil.copy2(src, comfy_input / safe_name)
        role = f"ref_image_{idx}"
        references[role] = {"filename": safe_name}
        frame_idx = relay_mode.FRAMES[idx] if idx < len(relay_mode.FRAMES) else idx * 24
        keyframes.append({"role": role, "frame_idx": frame_idx})

    for idx in range(3):
        role = f"ref_audio_{idx}"
        if idx < len(auds):
            src = auds[idx]
            safe_name = f"{run_id}_aud_{idx}{src.suffix.lower()}"
            shutil.copy2(src, comfy_input / safe_name)
        else:
            safe_name = f"{run_id}_aud_{idx}_silent.wav"
            make_silent_wav(comfy_input / safe_name)
        references[role] = {"filename": safe_name}

    w, h = (int(x) for x in resolution.split("x"))
    prompts = params_in.get("prompts", [])
    if isinstance(prompts, str):
        prompts = [p.strip() for p in prompts.splitlines() if p.strip()]
    if len(prompts) < 2:
        prompts = [prompts[0], prompts[0]] if prompts else ["cinematic scene", "cinematic continuation"]

    raw_params = {
        "preset": "goldfish_relay",
        "mode": "reference",
        "resolution": "768p",
        "goldfish_resolution": resolution,
        "width": w,
        "height": h,
        "audio": bool(params_in.get("audio", True)),
        "seed": int(params_in.get("seed", -1)),
        "relay_segments": len(prompts),
        "keyframes": keyframes,
    }
    params = h3_workflow.parameters(raw_params)
    graphs = relay_mode.graphs(params, prompts, references, run_id)

    segment_mp4s: list[Path] = []
    for stage, g in enumerate(graphs):
        cp.job_heartbeat(
            job_id,
            phase="generating",
            progress=round(0.1 + 0.75 * (stage / len(graphs)), 3),
            stage=f"Stage {stage + 1}/{len(graphs)}",
            detail=f"Submitting segment {stage + 1}/{len(graphs)} to ComfyUI",
        )
        queue_and_wait(g, comfy_dir, cp, job_id, stage, len(graphs))
        stage_dir = comfy_output / "segments" / f"stage_{stage:02d}"
        mp4s = sorted(stage_dir.glob("goldfish*-audio.mp4")) or sorted(stage_dir.glob("*.mp4"))
        if mp4s:
            segment_mp4s.append(mp4s[-1])

    if not segment_mp4s:
        raise RuntimeError("ComfyUI finished without producing any segment MP4 files.")

    cp.job_heartbeat(job_id, phase="uploading_outputs", progress=0.90, detail="Stitching MP4 & uploading preview to R2")
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = out_dir / f"{run_id}_final.mp4"
    preview_mp4 = out_dir / f"{run_id}_preview.mp4"
    stitch_and_preview(segment_mp4s, final_mp4, preview_mp4)

    preview_res = cp.upload_job_file(job_id, "preview.mp4", preview_mp4, "video/mp4")

    # Optional rclone archive of full-res master to Google Drive 20TB
    gdrive_dest = os.environ.get("RCLONE_GDRIVE_DEST", "").strip()
    gdrive_path = ""
    if gdrive_dest:
        try:
            subprocess.run(["rclone", "copy", str(final_mp4), f"{gdrive_dest.rstrip('/')}/{job_id}/"], check=True)
            gdrive_path = f"{gdrive_dest.rstrip('/')}/{job_id}/{final_mp4.name}"
        except Exception as exc:
            print(f"[WARN] rclone archive skipped/failed: {exc}")

    elapsed = round(time.time() - t_start, 2)
    receipt = {
        "job_id": job_id,
        "workflow": job.get("workflow", "h3_relay"),
        "workflow_version": job.get("workflow_version", "v1"),
        "platform": cp.platform,
        "worker_id": cp.worker_id,
        "pod_id": cp.pod_id,
        "elapsed_sec": elapsed,
        "resolution": resolution,
        "segments": len(graphs),
        "seed": params.get("seed"),
        "preview_key": preview_res.get("key", f"jobs/{job_id}/output/preview.mp4"),
        "gdrive_master_path": gdrive_path,
        "models_schema_version": manifest.get("schema_version"),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_file = out_dir / "manifest.json"
    manifest_file.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    cp.upload_job_file(job_id, "manifest.json", manifest_file, "application/json")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Universal GPU Worker for H3 Cloud Workbench")
    parser.add_argument("--platform", default=os.environ.get("WORKER_PLATFORM", "runpod"), choices=["runpod", "colab", "local5090"])
    parser.add_argument("--comfy-dir", default=os.environ.get("COMFY_DIR", "/opt/ComfyUI"))
    parser.add_argument("--work-dir", default="/tmp/h3_work")
    parser.add_argument("--models-dir", default=os.environ.get("MODELS_OVERRIDE_DIR", ""), help="Optional pre-existing models dir (e.g. Drive on Colab)")
    parser.add_argument("--idle-shutdown-sec", type=int, default=int(os.environ.get("IDLE_SHUTDOWN_SEC", "180")))
    args = parser.parse_args()

    cp_url = os.environ.get("CONTROL_PLANE_URL", "").strip()
    token = os.environ.get("WORKER_TOKEN", "").strip()
    if not cp_url or not token:
        print("[ERROR] CONTROL_PLANE_URL and WORKER_TOKEN must be set.", file=sys.stderr)
        return 2

    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    worker_id = f"{args.platform}-{pod_id or socket.gethostname()}-{uuid.uuid4().hex[:6]}"
    cp = ControlPlaneClient(cp_url, token, worker_id, args.platform, pod_id)

    comfy_dir = Path(args.comfy_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        wf_dir = sync_workflow_bundle(cp, comfy_dir, work_dir)
        manifest = cp.get_json("/api/assets/registry/models_manifest.json")
        # Overlap ComfyUI boot with 4x16-conn model download (saves ~20s)
        cp.worker_heartbeat("bootstrapping", "Launching ComfyUI in parallel with HF model download")
        comfy_log = work_dir / "comfyui.log"
        comfy_proc = launch_comfyui_async(comfy_dir, comfy_log)
        ensure_models(cp, comfy_dir, manifest, args.models_dir, workflow="h3_relay", allowed_tiers={"core", "lora"})
        wait_comfyui_ready(comfy_proc, comfy_log)
    except Exception as exc:
        cp.worker_heartbeat("error", f"Boot failed: {exc}")
        traceback.print_exc()
        if args.platform == "runpod":
            cp.request_self_terminate("boot_failed")
        return 1

    cp.worker_heartbeat("idle", "Ready and polling queue")
    idle_since = time.time()

    while True:
        try:
            job = cp.claim_job()
        except Exception as exc:
            print(f"[WARN] claim_job error: {exc}")
            time.sleep(8)
            continue

        if not job:
            idle_sec = time.time() - idle_since
            cp.worker_heartbeat("idle", f"Queue empty ({int(idle_sec)}s / {args.idle_shutdown_sec}s)")
            if args.platform == "runpod" and idle_sec >= args.idle_shutdown_sec:
                cp.request_self_terminate("idle_timeout")
                return 0
            time.sleep(6)
            continue

        idle_since = time.time()
        job_id = job["id"]
        print(f"[JOB] Claimed {job_id} ({job.get('workflow')})")
        cp.worker_heartbeat("busy", f"Running job {job_id}", job_id=job_id)
        try:
            receipt = execute_h3_job(job, wf_dir, comfy_dir, work_dir, cp, manifest, args.models_dir)
            cp.complete_job(job_id, receipt)
            print(f"[JOB] Completed {job_id} in {receipt['elapsed_sec']}s")
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[JOB] Failed {job_id}: {exc}\n{tb}")
            cp.fail_job(job_id, f"{exc}\n{tb[-1500:]}")
        finally:
            idle_since = time.time()
            cp.worker_heartbeat("idle", "Job finished, checking next")


if __name__ == "__main__":
    raise SystemExit(main())

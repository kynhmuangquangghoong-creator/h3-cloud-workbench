#!/usr/bin/env python3
"""Universal GPU Worker for H3 Cloud Workbench (Runpod / Colab / Local 5090).

Upgraded for:
- True Parallel Cold-Start: claim job immediately -> launch ComfyUI, 4x16 HF weight sync,
  and R2 input download concurrently (hiding ComfyUI's ~30s boot inside weight download).
- Real-Time Intermediate Node Telemetry: pushes structured `telemetry.stages` + live ComfyUI
  WebSocket node/step progress + nvidia-smi VRAM/GPU utilization to the Cloud Workbench UI.
- Second-Level Shutdown: 10s idle auto-terminate on Runpod with 3x retry.
- Lossless Master Return: uploads original stitched `relay_final.mp4` when <= 19 MB.
"""
from __future__ import annotations

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
COMFY_WS_URL = "ws://127.0.0.1:8188/ws"
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


class StageTracker:
    """Thread-safe tracker for intermediate execution nodes shown live in the Workbench UI."""

    def __init__(self, stage_defs: list[tuple[str, str]]) -> None:
        self._lock = threading.Lock()
        self.t_boot = time.time()
        self.peak_vram_mb = 0
        self.extra: dict = {}
        self.stages: list[dict] = [
            {
                "id": sid,
                "label": label,
                "status": "pending",
                "started_at": None,
                "ended_at": None,
                "elapsed_sec": None,
                "detail": "",
                "meta": "",
            }
            for sid, label in stage_defs
        ]

    def _find(self, sid: str) -> dict | None:
        for s in self.stages:
            if s["id"] == sid:
                return s
        return None

    def start(self, sid: str, detail: str = "", meta: str = "") -> None:
        with self._lock:
            s = self._find(sid)
            if s:
                now = time.time()
                if s["started_at"] is None:
                    s["started_at"] = round(now, 2)
                s["status"] = "running"
                if detail:
                    s["detail"] = detail
                if meta:
                    s["meta"] = meta

    def update(self, sid: str, detail: str = "", meta: str = "") -> None:
        with self._lock:
            s = self._find(sid)
            if s:
                if s["started_at"] is None:
                    s["started_at"] = round(time.time(), 2)
                    s["status"] = "running"
                if detail:
                    s["detail"] = detail
                if meta:
                    s["meta"] = meta
                s["elapsed_sec"] = round(time.time() - s["started_at"], 1)

    def finish(self, sid: str, detail: str = "", meta: str = "") -> None:
        with self._lock:
            s = self._find(sid)
            if s:
                now = time.time()
                if s["started_at"] is None:
                    s["started_at"] = round(now, 2)
                s["ended_at"] = round(now, 2)
                s["elapsed_sec"] = round(now - s["started_at"], 1)
                s["status"] = "done"
                if detail:
                    s["detail"] = detail
                if meta:
                    s["meta"] = meta

    def fail(self, sid: str, detail: str = "") -> None:
        with self._lock:
            s = self._find(sid)
            if s:
                now = time.time()
                if s["started_at"] is None:
                    s["started_at"] = round(now, 2)
                s["ended_at"] = round(now, 2)
                s["elapsed_sec"] = round(now - s["started_at"], 1)
                s["status"] = "error"
                if detail:
                    s["detail"] = detail[:240]

    def set_extra(self, **kwargs) -> None:
        with self._lock:
            self.extra.update(kwargs)

    def snapshot(self) -> dict:
        gpu = get_gpu_telemetry()
        with self._lock:
            if gpu.get("vram_used_mb", 0) > self.peak_vram_mb:
                self.peak_vram_mb = gpu["vram_used_mb"]
            now = time.time()
            stages_copy = []
            for s in self.stages:
                sc = dict(s)
                if sc["status"] == "running" and sc["started_at"] is not None:
                    sc["elapsed_sec"] = round(now - sc["started_at"], 1)
                stages_copy.append(sc)
            out = dict(gpu)
            out["peak_vram_mb"] = self.peak_vram_mb
            out["total_elapsed_sec"] = round(now - self.t_boot, 1)
            out.update(self.extra)
            out["stages"] = stages_copy
            return out


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
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": "H3UniversalWorker/2.0"}
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
            url, headers={"Authorization": f"Bearer {self.token}", "User-Agent": "H3UniversalWorker/2.0"}
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
                "User-Agent": "H3UniversalWorker/2.0",
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

    def job_heartbeat(
        self,
        job_id: str,
        phase: str,
        progress: float,
        detail: str = "",
        stage: str = "",
        tracker: StageTracker | None = None,
    ) -> None:
        try:
            tel = tracker.snapshot() if tracker is not None else get_gpu_telemetry()
            self.post_json(
                f"/api/jobs/{job_id}/heartbeat",
                {
                    "worker_id": self.worker_id,
                    "phase": phase,
                    "stage": stage,
                    "progress": round(progress, 3),
                    "detail": detail[:500],
                    "telemetry": tel,
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
        for attempt in range(1, 4):
            try:
                res = self.post_json(
                    "/api/pods/self-terminate",
                    {"pod_id": self.pod_id, "worker_id": self.worker_id, "reason": reason},
                )
                print(f"[SHUTDOWN] Self-terminate ack (attempt {attempt}): {res}")
                if res.get("ok"):
                    return
            except Exception as exc:
                print(f"[WARN] self-terminate attempt {attempt}/3 failed: {exc}")
            time.sleep(3)


def sync_workflow_bundle(cp: ControlPlaneClient, comfy_dir: Path, work_dir: Path) -> Path:
    wf_dir = work_dir / "workflows" / "h3_relay"
    if (wf_dir / "h3_workflow.py").is_file() and (comfy_dir / "custom_nodes" / "ComfyUI-H3-Latent-Relay").is_dir():
        return wf_dir
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
    job_id: str = "",
    tracker: StageTracker | None = None,
) -> dict:
    models = []
    for m in manifest.get("models", []):
        if workflow not in m.get("workflows", []):
            continue
        tier = m.get("tier", "core")
        if allowed_tiers is not None and tier not in allowed_tiers:
            continue
        models.append(m)

    if not models:
        return {"count": 0, "downloaded_bytes": 0, "elapsed_sec": 0.0, "speed_mbps": 0.0}

    hf_token = os.environ.get("HF_TOKEN") or None
    from huggingface_hub import hf_hub_download

    total = len(models)
    done = 0
    total_bytes = 0
    lock = threading.Lock()
    t0 = time.time()

    def report_progress(msg: str) -> None:
        elapsed = max(0.1, time.time() - t0)
        gb = total_bytes / (1024 ** 3)
        mbps = (total_bytes / (1024 ** 2)) / elapsed
        meta = f"{done}/{total} 权重 · {gb:.1f} GB · {mbps:.0f} MB/s"
        if tracker is not None:
            tracker.update("parallel_init", detail=msg, meta=meta)
            tracker.set_extra(dl_gb=round(gb, 2), dl_speed_mbps=round(mbps, 1))
        if job_id:
            prog = 0.03 + 0.12 * (done / max(1, total))
            cp.job_heartbeat(job_id, phase="parallel_init", progress=prog, stage="并行冷启", detail=msg, tracker=tracker)
        else:
            cp.worker_heartbeat("downloading_models", msg)

    def fetch_one(m: dict):
        nonlocal done, total_bytes
        target_dir = comfy_dir / m["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        dst_file = target_dir / m["local_name"]

        if models_override_dir:
            sub = m["target_dir"].replace("models/", "", 1)
            candidate = Path(models_override_dir) / sub / m["local_name"]
            if candidate.is_file():
                if not dst_file.exists():
                    os.symlink(candidate, dst_file)
                with lock:
                    done += 1
                    report_progress(f"软链复用 {m['local_name']}")
                return

        if m.get("download_type") == "snapshot":
            from huggingface_hub import snapshot_download
            if target_dir.is_dir() and any(target_dir.iterdir()):
                with lock:
                    done += 1
                    report_progress(f"已缓存 {m['local_name']}")
                return
            print(f"[HF] Snapshot {m['repo_id']} -> {target_dir}...")
            snapshot_download(repo_id=m["repo_id"], local_dir=str(target_dir), token=hf_token)
            sz = sum(p.stat().st_size for p in target_dir.rglob("*") if p.is_file())
            with lock:
                done += 1
                total_bytes += sz
                report_progress(f"完成 {m['local_name']}")
            return

        if dst_file.is_file() and dst_file.stat().st_size > 1_000_000:
            with lock:
                done += 1
                report_progress(f"已缓存 {m['local_name']}")
            return

        print(f"[HF] Downloading {m['repo_id']} :: {m['filename']} (~{m.get('approx_gb', '?')} GB)...")
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

        sz = dst_file.stat().st_size if dst_file.is_file() else 0
        with lock:
            done += 1
            total_bytes += sz
            elapsed = time.time() - t0
            print(f"[HF] Done [{done}/{total}] {m['local_name']} ({'aria2c-16x' if ok else 'hf_hub'}, {elapsed:.1f}s)")
            report_progress(f"就绪 {m['local_name']} ({elapsed:.0f}s)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(fetch_one, m) for m in models]
        for fut in futs:
            fut.result()

    elapsed = round(time.time() - t0, 2)
    speed = round((total_bytes / (1024 ** 2)) / max(0.1, elapsed), 1)
    return {"count": total, "downloaded_bytes": total_bytes, "elapsed_sec": elapsed, "speed_mbps": speed}


def is_comfyui_up() -> bool:
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/object_info/MiniMaxH3AddGuide", timeout=2) as r:
            data = json.loads(r.read().decode("utf-8"))
            return "MiniMaxH3AddGuide" in data
    except Exception:
        return False


def launch_comfyui_async(comfy_dir: Path, log_path: Path) -> subprocess.Popen:
    log_f = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-u", str(comfy_dir / "main.py"), "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"],
        cwd=str(comfy_dir),
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )


def wait_comfyui_ready(proc: subprocess.Popen, log_path: Path, timeout_sec: int = 180) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if is_comfyui_up():
            elapsed = round(time.time() - t0, 1)
            print(f"[OK] ComfyUI ready ({elapsed}s)")
            return elapsed
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early with code {proc.returncode}. Check {log_path}")
        time.sleep(1.0)
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
    stage_id: str = "infer",
    tracker: StageTracker | None = None,
) -> dict:
    client_id = uuid.uuid4().hex
    graph = adapt_graph(graph, comfy_dir)

    # Start lightweight ComfyUI WebSocket listener to capture real-time node name & step progress
    ws_state = {"node": "", "step": 0, "max_steps": 0, "stop": False}

    def ws_listener():
        try:
            import websocket  # type: ignore
            ws = websocket.create_connection(f"{COMFY_WS_URL}?clientId={client_id}", timeout=5)
            ws.settimeout(2.0)
            while not ws_state["stop"]:
                try:
                    raw = ws.recv()
                    if not isinstance(raw, str):
                        continue
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    d = msg.get("data") or {}
                    if mtype == "executing":
                        nid = str(d.get("node") or "")
                        if nid and nid in graph:
                            ws_state["node"] = graph[nid].get("class_type", nid)
                    elif mtype == "progress":
                        ws_state["step"] = int(d.get("value") or 0)
                        ws_state["max_steps"] = int(d.get("max") or 0)
                except Exception:
                    continue
            ws.close()
        except Exception:
            pass

    ws_thread = threading.Thread(target=ws_listener, daemon=True)
    ws_thread.start()
    time.sleep(0.2)

    payload = json.dumps({"prompt": graph, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    prompt_id = data["prompt_id"]

    t0 = time.time()
    last_hb = 0.0
    try:
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
            if now - last_hb >= 4.0:
                elapsed = int(now - t0)
                node_name = ws_state["node"] or "ComfyUI"
                step, max_s = ws_state["step"], ws_state["max_steps"]
                step_txt = f"Step {step}/{max_s}" if max_s > 0 else f"{elapsed}s"
                sub_frac = (step / max_s) if max_s > 0 else 0.35
                base_prog = 0.16 + 0.72 * ((stage_idx + sub_frac) / max(1, total_stages))
                detail_txt = f"{node_name} · {step_txt}"
                if tracker is not None:
                    tracker.update(stage_id, detail=detail_txt, meta=step_txt)
                    tracker.set_extra(current_node=node_name, step=step, total_steps=max_s)
                cp.job_heartbeat(
                    job_id,
                    phase="generating",
                    progress=min(0.89, base_prog),
                    stage=f"第 {stage_idx + 1}/{total_stages} 段 · {step_txt}",
                    detail=detail_txt,
                    tracker=tracker,
                )
                last_hb = now
            time.sleep(2.0)
    finally:
        ws_state["stop"] = True


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
    comfy_proc_holder: dict,
    models_override_dir: str = "",
) -> dict:
    job_id = job["id"]
    t_start = time.time()
    params_in = job.get("params", {})
    resolution = params_in.get("resolution", "1344x768")

    prompts = params_in.get("prompts", [])
    if isinstance(prompts, str):
        prompts = [p.strip() for p in prompts.splitlines() if p.strip()]
    if len(prompts) < 2:
        prompts = [prompts[0], prompts[0]] if prompts else ["cinematic scene", "cinematic continuation"]
    n_segs = len(prompts)

    stage_defs = [
        ("pod_boot", "建卡与容器就绪"),
        ("parallel_init", "并行冷启 (ComfyUI+权重+素材)"),
    ] + [(f"seg_{i + 1}", f"第 {i + 1}/{n_segs} 段推理") for i in range(n_segs)] + [
        ("stitch_upload", "拼接与回传中转"),
        ("shutdown", "本地拉回与秒级停表"),
    ]
    tracker = StageTracker(stage_defs)
    tracker.finish("pod_boot", detail="容器就绪并已接单", meta=cp.platform)
    tracker.start("parallel_init", detail="并行启动 ComfyUI + 4×16 拉权重 + 拉素材")
    cp.job_heartbeat(job_id, phase="parallel_init", progress=0.03, stage="并行冷启", detail="并行启动 ComfyUI 与下载权重", tracker=tracker)

    needed_tiers = {"core", "lora"}
    if resolution == "2048x1152":
        needed_tiers.add("upscale_2k")

    job_dir = work_dir / "jobs" / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_keys = job.get("input_keys", [])

    def _boot_comfy() -> float:
        if is_comfyui_up():
            return 0.0
        comfy_log = work_dir / "comfyui.log"
        if comfy_proc_holder.get("proc") is None or comfy_proc_holder["proc"].poll() is not None:
            comfy_proc_holder["proc"] = launch_comfyui_async(comfy_dir, comfy_log)
        return wait_comfyui_ready(comfy_proc_holder["proc"], comfy_log)

    def _download_inputs() -> tuple[list[Path], list[Path]]:
        imgs_l, auds_l = [], []
        for key in input_keys:
            fname = Path(key).name
            local_f = input_dir / fname
            cp.download_asset(key, local_f)
            ext = local_f.suffix.lower()
            if ext in IMG_EXTS:
                imgs_l.append(local_f)
            elif ext in AUD_EXTS:
                auds_l.append(local_f)
        return imgs_l, auds_l

    # TRUE PARALLEL COLD-START: ComfyUI boot + HF 4x16 model sync + R2 inputs simultaneously!
    t_init0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_comfy = pool.submit(_boot_comfy)
        f_models = pool.submit(
            ensure_models, cp, comfy_dir, manifest, models_override_dir,
            job.get("workflow", "h3_relay"), needed_tiers, job_id, tracker
        )
        f_inputs = pool.submit(_download_inputs)
        comfy_boot_sec = f_comfy.result()
        dl_stats = f_models.result()
        imgs, auds = f_inputs.result()
    parallel_init_sec = round(time.time() - t_init0, 2)

    dl_gb = round(dl_stats["downloaded_bytes"] / (1024 ** 3), 2)
    init_meta = f"{dl_gb}GB · {dl_stats['speed_mbps']}MB/s · ComfyUI {comfy_boot_sec}s"
    tracker.finish("parallel_init", detail=f"并行冷启完成 ({parallel_init_sec}s)", meta=init_meta)
    cp.job_heartbeat(job_id, phase="generating", progress=0.16, stage="冷启完成", detail=init_meta, tracker=tracker)

    if not imgs:
        raise ValueError("任务缺少参考图片（请至少上传 1 张 png/jpg/webp）")

    if str(wf_dir) not in sys.path:
        sys.path.insert(0, str(wf_dir))
    import h3_workflow
    import relay_mode

    comfy_input = comfy_dir / "input"
    comfy_output = comfy_dir / "output"
    comfy_input.mkdir(parents=True, exist_ok=True)

    run_id = f"job_{job_id[:10]}"
    references = {}
    keyframes = []
    # Sort input files deterministically by name so ref_image_0..8 order matches upload naming
    imgs = sorted(imgs, key=lambda p: p.name)
    auds = sorted(auds, key=lambda p: p.name)

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
    raw_params = {
        "preset": "goldfish_relay",
        "mode": "reference",
        "resolution": "768p",
        "goldfish_resolution": resolution,
        "width": w,
        "height": h,
        "audio": bool(params_in.get("audio", True)),
        "seed": int(params_in.get("seed", -1)),
        "steps": int(params_in.get("steps", 12)),
        "crf": int(params_in.get("crf", 23)),
        "relay_segments": n_segs,
        "keyframes": keyframes,
    }
    params = h3_workflow.parameters(raw_params)
    graphs = relay_mode.graphs(params, prompts, references, run_id)

    segment_mp4s: list[Path] = []
    segment_timings: list[dict] = []
    for stage, g in enumerate(graphs):
        sid = f"seg_{stage + 1}"
        t_seg0 = time.time()
        tracker.start(sid, detail=f"提交第 {stage + 1}/{n_segs} 段至 ComfyUI", meta="启动中")
        cp.job_heartbeat(
            job_id,
            phase="generating",
            progress=round(0.16 + 0.72 * (stage / len(graphs)), 3),
            stage=f"第 {stage + 1}/{len(graphs)} 段",
            detail=f"开始第 {stage + 1}/{len(graphs)} 段采样",
            tracker=tracker,
        )
        queue_and_wait(g, comfy_dir, cp, job_id, stage, len(graphs), stage_id=sid, tracker=tracker)
        seg_sec = round(time.time() - t_seg0, 2)
        stage_dir = comfy_output / "segments" / f"stage_{stage:02d}"
        mp4s = sorted(stage_dir.glob("goldfish*-audio.mp4")) or sorted(stage_dir.glob("*.mp4"))
        if mp4s:
            segment_mp4s.append(mp4s[-1])
        segment_timings.append({"segment": stage + 1, "elapsed_sec": seg_sec})
        tracker.finish(sid, detail=f"第 {stage + 1} 段完成", meta=f"{seg_sec}s")

    if not segment_mp4s:
        raise RuntimeError("ComfyUI finished without producing any segment MP4 files.")

    tracker.start("stitch_upload", detail="ffmpeg 无损拼接与回传 R2")
    cp.job_heartbeat(job_id, phase="uploading_outputs", progress=0.90, stage="拼接回传", detail="正在拼接视频并上传 R2", tracker=tracker)
    t_up0 = time.time()
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = out_dir / f"{run_id}_final.mp4"
    preview_mp4 = out_dir / f"{run_id}_preview.mp4"
    stitch_and_preview(segment_mp4s, final_mp4, preview_mp4)

    # Prefer uploading the lossless stitched final MP4 directly if <= 19 MB (for 1:1 comparison!)
    if final_mp4.stat().st_size <= 19 * 1024 * 1024:
        up_res = cp.upload_job_file(job_id, "relay_final.mp4", final_mp4, "video/mp4")
        preview_key = up_res.get("key", f"jobs/{job_id}/output/relay_final.mp4")
        out_size = final_mp4.stat().st_size
    else:
        up_res = cp.upload_job_file(job_id, "preview.mp4", preview_mp4, "video/mp4")
        preview_key = up_res.get("key", f"jobs/{job_id}/output/preview.mp4")
        out_size = preview_mp4.stat().st_size

    upload_sec = round(time.time() - t_up0, 2)
    tracker.finish("stitch_upload", detail=f"回传完成 ({out_size / (1024*1024):.2f} MB)", meta=f"{upload_sec}s")
    tracker.start("shutdown", detail="等待本地自动拉回，10s 内自动停表", meta="待停表")

    elapsed = round(time.time() - t_start, 2)
    snap = tracker.snapshot()
    receipt = {
        "job_id": job_id,
        "workflow": job.get("workflow", "h3_relay"),
        "workflow_version": job.get("workflow_version", "v1"),
        "platform": cp.platform,
        "worker_id": cp.worker_id,
        "pod_id": cp.pod_id,
        "gpu_name": snap.get("gpu_name"),
        "peak_vram_mb": snap.get("peak_vram_mb"),
        "elapsed_sec": elapsed,
        "parallel_init_sec": parallel_init_sec,
        "comfy_boot_sec": comfy_boot_sec,
        "model_download_stats": dl_stats,
        "segment_timings": segment_timings,
        "upload_sec": upload_sec,
        "output_bytes": out_size,
        "resolution": resolution,
        "segments": len(graphs),
        "seed": params.get("seed"),
        "preview_key": preview_key,
        "models_schema_version": manifest.get("schema_version"),
        "stages": snap.get("stages", []),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cp.job_heartbeat(job_id, phase="completed", progress=1.0, stage="已完成", detail=f"总耗时 {elapsed}s", tracker=tracker)
    manifest_file = out_dir / "manifest.json"
    manifest_file.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    cp.upload_job_file(job_id, "manifest.json", manifest_file, "application/json")
    return receipt


AUDIO_TEMPLATES = {
    "audio_ace_sft": ("ace15_xl_sft.api.json", {"prompt": ("5", "tags"), "lyrics": ("5", "lyrics"), "duration": ("5", "duration"), "seed": ("5", "seed")}),
    "audio_ace_turbo": ("ace15_xl_turbo.api.json", {"prompt": ("5", "tags"), "lyrics": ("5", "lyrics"), "duration": ("5", "duration"), "seed": ("5", "seed")}),
    "audio_yue2_music": ("yue2_text_to_music.api.json", {"prompt": ("2", "style"), "lyrics": ("2", "lyrics"), "duration": ("2", "duration"), "seed": ("2", "seed")}),
    "audio_yue2_cover": ("yue2_cover.api.json", {"prompt": ("5", "style"), "lyrics": ("5", "lyrics"), "duration": ("5", "duration"), "seed": ("5", "seed")}),
    "audio_sfx": ("stable_audio3_sfx.api.json", {"prompt": ("3", "text"), "duration": ("4", "seconds"), "seed": ("5", "seed")}),
    "audio_tts_design": ("tts_voice_design.api.json", {"prompt": ("1", "text"), "instruction": ("1", "instruct"), "seed": ("1", "seed")}),
    "audio_tts_clone": ("tts_voice_clone.api.json", {"prompt": ("2", "text"), "seed": ("2", "seed")}),
    "audio_asr": ("asr_from_tts.api.json", {}),
    "audio_stems": ("demucs_stems.api.json", {}),
}


def execute_audio_job(
    job: dict,
    comfy_dir: Path,
    work_dir: Path,
    cp: ControlPlaneClient,
    manifest: dict,
    comfy_proc_holder: dict,
    models_override_dir: str = "",
) -> dict:
    job_id = job["id"]
    wf_id = job.get("workflow", "audio_sfx")
    t_start = time.time()

    tracker = StageTracker([
        ("pod_boot", "建卡与容器就绪"),
        ("parallel_init", "并行冷启 (ComfyUI+音频权重)"),
        ("infer", f"音频推理 ({wf_id})"),
        ("stitch_upload", "回传音频至中转"),
        ("shutdown", "本地拉回与秒级停表"),
    ])
    tracker.finish("pod_boot", detail="容器就绪并已接单", meta=cp.platform)
    tracker.start("parallel_init", detail=f"并行启动 ComfyUI + 拉取 {wf_id} 权重")

    def _boot_comfy() -> float:
        if is_comfyui_up():
            return 0.0
        comfy_log = work_dir / "comfyui.log"
        if comfy_proc_holder.get("proc") is None or comfy_proc_holder["proc"].poll() is not None:
            comfy_proc_holder["proc"] = launch_comfyui_async(comfy_dir, comfy_log)
        return wait_comfyui_ready(comfy_proc_holder["proc"], comfy_log)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_comfy = pool.submit(_boot_comfy)
        f_models = pool.submit(ensure_models, cp, comfy_dir, manifest, models_override_dir, wf_id, None, job_id, tracker)
        comfy_boot_sec = f_comfy.result()
        dl_stats = f_models.result()

    tracker.finish("parallel_init", detail="并行冷启完成", meta=f"ComfyUI {comfy_boot_sec}s")

    audio_wf_dir = work_dir / "workflows" / "audio_suite"
    if not (audio_wf_dir / "workflows").is_dir():
        audio_wf_dir.mkdir(parents=True, exist_ok=True)
        bzip = work_dir / "audio_suite_bundle.zip"
        cp.download_asset("workflows/audio_suite/audio_suite_bundle.zip", bzip)
        with zipfile.ZipFile(bzip, "r") as zf:
            zf.extractall(audio_wf_dir)

    tpl_name, field_map = AUDIO_TEMPLATES.get(wf_id, ("stable_audio3_sfx.api.json", {}))
    graph = json.loads((audio_wf_dir / "workflows" / tpl_name).read_text(encoding="utf-8"))

    comfy_input = comfy_dir / "input"
    comfy_input.mkdir(parents=True, exist_ok=True)
    staged_audio = ""
    for key in job.get("input_keys", []):
        fname = f"{job_id[:8]}_{Path(key).name}"
        dest = comfy_input / fname
        cp.download_asset(key, dest)
        if dest.suffix.lower() in AUD_EXTS and not staged_audio:
            staged_audio = fname

    params = job.get("params", {})
    if staged_audio:
        for node in graph.values():
            if node.get("class_type") == "LoadAudio" and "audio" in node.get("inputs", {}):
                node["inputs"]["audio"] = staged_audio

    for pkey, (nid, field) in field_map.items():
        val = params.get(pkey)
        if val not in (None, "", -1) and nid in graph and field in graph[nid].get("inputs", {}):
            graph[nid]["inputs"][field] = val

    tracker.start("infer", detail=f"运行 {tpl_name}")
    cp.job_heartbeat(job_id, phase="generating", progress=0.35, stage=wf_id, detail=f"Running {tpl_name} in ComfyUI", tracker=tracker)
    entry = queue_and_wait(graph, comfy_dir, cp, job_id, 0, 1, stage_id="infer", tracker=tracker)
    tracker.finish("infer", detail="音频生成完成")

    tracker.start("stitch_upload", detail="回传音频至 R2")
    comfy_output = comfy_dir / "output"
    uploaded_key = ""
    outputs = entry.get("outputs", {})
    for node_out in outputs.values():
        for bucket in ("audio", "gifs", "videos", "images"):
            for item in node_out.get(bucket, []) or []:
                sub = item.get("subfolder", "")
                fn = item.get("filename", "")
                fp = (comfy_output / sub / fn) if sub else (comfy_output / fn)
                if fp.is_file():
                    ct = "audio/flac" if fp.suffix == ".flac" else ("audio/wav" if fp.suffix == ".wav" else "application/octet-stream")
                    res = cp.upload_job_file(job_id, fp.name, fp, ct)
                    if not uploaded_key:
                        uploaded_key = res.get("key", f"jobs/{job_id}/output/{fp.name}")
    tracker.finish("stitch_upload", detail="上传完成")
    tracker.start("shutdown", detail="等待本地拉回，10s 内自动停表")

    elapsed = round(time.time() - t_start, 2)
    snap = tracker.snapshot()
    receipt = {
        "job_id": job_id,
        "workflow": wf_id,
        "platform": cp.platform,
        "elapsed_sec": elapsed,
        "preview_key": uploaded_key,
        "stages": snap.get("stages", []),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cp.job_heartbeat(job_id, phase="completed", progress=1.0, stage="已完成", detail=f"总耗时 {elapsed}s", tracker=tracker)
    mf_path = work_dir / f"{job_id}_manifest.json"
    mf_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    cp.upload_job_file(job_id, "manifest.json", mf_path, "application/json")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Universal GPU Worker for H3 Cloud Workbench")
    parser.add_argument("--platform", default=os.environ.get("WORKER_PLATFORM", "runpod"), choices=["runpod", "colab", "local5090"])
    parser.add_argument("--comfy-dir", default=os.environ.get("COMFY_DIR", "/opt/ComfyUI"))
    parser.add_argument("--work-dir", default="/tmp/h3_work")
    parser.add_argument("--models-dir", default=os.environ.get("MODELS_OVERRIDE_DIR", ""), help="Optional pre-existing models dir")
    parser.add_argument("--idle-shutdown-sec", type=int, default=int(os.environ.get("IDLE_SHUTDOWN_SEC", "10")))
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
    comfy_proc_holder: dict = {"proc": None}

    try:
        wf_dir = sync_workflow_bundle(cp, comfy_dir, work_dir)
        manifest = cp.get_json("/api/assets/registry/models_manifest.json")
    except Exception as exc:
        cp.worker_heartbeat("error", f"Boot sync failed: {exc}")
        traceback.print_exc()
        if args.platform == "runpod":
            cp.request_self_terminate("boot_failed")
        return 1

    # Claim job FIRST so ComfyUI boot + model download run in parallel immediately!
    cp.worker_heartbeat("idle", "Ready and claiming queued job")
    idle_since = time.time()
    exit_on_idle = os.environ.get("EXIT_ON_IDLE", "1") == "1"

    while True:
        try:
            job = cp.claim_job()
        except Exception as exc:
            print(f"[WARN] claim_job error: {exc}")
            time.sleep(3)
            continue

        if not job:
            idle_sec = time.time() - idle_since
            cp.worker_heartbeat("idle", f"Queue empty ({int(idle_sec)}s / {args.idle_shutdown_sec}s)")
            if (args.platform == "runpod" or exit_on_idle) and idle_sec >= args.idle_shutdown_sec:
                if args.platform == "runpod":
                    cp.request_self_terminate("idle_timeout")
                return 0
            time.sleep(3)
            continue

        idle_since = time.time()
        job_id = job["id"]
        wf_id = job.get("workflow", "h3_relay")
        print(f"[JOB] Claimed {job_id} ({wf_id})")
        cp.worker_heartbeat("busy", f"Running {wf_id} ({job_id})", job_id=job_id)
        try:
            if wf_id.startswith("audio_"):
                receipt = execute_audio_job(job, comfy_dir, work_dir, cp, manifest, comfy_proc_holder, args.models_dir)
            else:
                receipt = execute_h3_job(job, wf_dir, comfy_dir, work_dir, cp, manifest, comfy_proc_holder, args.models_dir)
            cp.complete_job(job_id, receipt)
            print(f"[JOB] Completed {job_id} in {receipt['elapsed_sec']}s")
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[JOB] Failed {job_id}: {exc}\n{tb}")
            cp.fail_job(job_id, f"{exc}\n{tb[-1500:]}")
        finally:
            idle_since = time.time()
            cp.worker_heartbeat("idle", "Job finished, checking next (fast shutdown armed)")


if __name__ == "__main__":
    raise SystemExit(main())

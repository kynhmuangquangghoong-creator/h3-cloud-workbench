#!/usr/bin/env python3
"""
Auto-managed 5090 WireProxy Tunnel + Ephemeral Docker Runner.

- If 127.0.0.1:55090 is not open, automatically decrypts the DPAPI tunnel config
  via E:/Apps/GPU-workbench-secondary/migration.py, starts wireproxy.exe, runs the
  requested remote command/script or ephemeral ComfyUI container, and cleanly tears
  down wireproxy.exe + deletes the temporary config file on exit.
- Zero 24/7 daemons, zero manual GPU工作台.exe clicks, zero secret duplication.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

WORKBENCH_DIR = Path(r"E:\Apps\GPU-workbench-secondary")
REMOTE_EXEC = Path(r"E:\Apps\audio-workspace\remote_exec.py")
MIGRATION_PY = Path(r"E:\CodexData\tools\migration-runtime\python\python.exe")


def port_open(host: str = "127.0.0.1", port: int = 55090, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class Auto5090Tunnel:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.started_by_us = False
        self.mig = None

    def __enter__(self) -> "Auto5090Tunnel":
        if port_open():
            print("[5090-TUNNEL] Reusing existing tunnel on 127.0.0.1:55090", flush=True)
            return self

        if str(WORKBENCH_DIR) not in sys.path:
            sys.path.insert(0, str(WORKBENCH_DIR))
        import migration  # type: ignore
        from portable_paths import WIREPROXY  # type: ignore

        self.mig = migration
        cfg_path = migration.tunnel_config()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            [str(WIREPROXY), "-s", "-c", str(cfg_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self.started_by_us = True
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if port_open():
                print(f"[5090-TUNNEL] Auto-started wireproxy (PID {self.proc.pid}) -> 127.0.0.1:55090", flush=True)
                return self
            if self.proc.poll() is not None:
                raise RuntimeError(f"wireproxy exited early with code {self.proc.returncode}")
            time.sleep(0.25)
        raise TimeoutError("Timed out waiting for wireproxy to bind 127.0.0.1:55090")

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.started_by_us:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                print(f"[5090-TUNNEL] Closed auto-started wireproxy (PID {self.proc.pid})", flush=True)
            if self.mig is not None:
                try:
                    self.mig.cleanup_tunnel_config()
                except Exception:
                    pass


def run_remote(command: str = "", script: Path | None = None, timeout: int = 7200) -> int:
    py = str(MIGRATION_PY) if MIGRATION_PY.is_file() else sys.executable
    cmd = [py, "-B", str(REMOTE_EXEC), "--timeout", str(timeout)]
    if script is not None:
        cmd += ["--script", str(script)]
    elif command:
        cmd.append(command)
    else:
        raise ValueError("Provide either --command or --script")
    with Auto5090Tunnel():
        res = subprocess.run(cmd)
        return res.returncode


def report_gpus_to_cloud() -> list[dict]:
    """Query all 4x RTX 5090 cards via auto-tunnel and push live status to gpu-workbench."""
    import json
    import urllib.request
    import winreg

    py = str(MIGRATION_PY) if MIGRATION_PY.is_file() else sys.executable
    q_cmd = "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits"
    with Auto5090Tunnel():
        out = subprocess.check_output(
            [py, "-B", str(REMOTE_EXEC), "--timeout", "20", q_cmd],
            text=True,
        )
    gpus = []
    for ln in out.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 6 and parts[0].isdigit():
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "vram_used_mb": int(float(parts[2])),
                "vram_total_mb": int(float(parts[3])),
                "util_pct": int(float(parts[4])),
                "temp_c": int(float(parts[5])),
            })
    tok = os.environ.get("GPU_WORKBENCH_TOKEN", "").strip()
    if not tok and os.name == "nt":
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
                tok, _ = winreg.QueryValueEx(k, "GPU_WORKBENCH_TOKEN")
        except OSError:
            pass
    if tok and gpus:
        req = urllib.request.Request(
            "https://gpu-workbench.mamegoya6900.workers.dev/api/local5090/gpus",
            data=json.dumps({"gpus": gpus, "host": "5090-4x"}).encode("utf-8"),
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json", "User-Agent": "run-5090-auto/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            print("[5090-TELEMETRY] Reported 4-GPU status:", r.read().decode("utf-8"), flush=True)
    return gpus


def run_local5090_worker(gpu_index: int = 0, idle_sec: int = 15) -> int:
    """Launch persistent-cache h3-local:latest on 5090 (mounts /home/xuliangliang/h3_cache/models, uses hf-mirror)."""
    import winreg

    tok = os.environ.get("GPU_WORKBENCH_TOKEN", "").strip()
    if not tok and os.name == "nt":
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
                tok, _ = winreg.QueryValueEx(k, "GPU_WORKBENCH_TOKEN")
        except OSError:
            pass
    if not tok:
        raise RuntimeError("Missing GPU_WORKBENCH_TOKEN")

    tmp_sh = Path(os.environ.get("TEMP", ".")) / f"run_h3_local_gpu{gpu_index}.sh"
    script = f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p /home/xuliangliang/h3_cache/models
exec docker run --rm --gpus device={gpu_index} --ipc=host --network host \\
  -v /home/xuliangliang/h3_cache/models:/opt/ComfyUI/models \\
  -e http_proxy=http://127.0.0.1:7897 \\
  -e https_proxy=http://127.0.0.1:7897 \\
  -e HTTP_PROXY=http://127.0.0.1:7897 \\
  -e HTTPS_PROXY=http://127.0.0.1:7897 \\
  -e NO_PROXY=127.0.0.1,localhost,hf-mirror.com,.hf-mirror.com,cdn-lfs.hf-mirror.com,cdn-lfs-us-1.hf-mirror.com \\
  -e no_proxy=127.0.0.1,localhost,hf-mirror.com,.hf-mirror.com,cdn-lfs.hf-mirror.com,cdn-lfs-us-1.hf-mirror.com \\
  -e HF_ENDPOINT=https://hf-mirror.com \\
  -e WORKER_PLATFORM=local5090 \\
  -e CONTROL_PLANE_URL=https://gpu-workbench.mamegoya6900.workers.dev \\
  -e WORKER_TOKEN='{tok}' \\
  -e EXIT_ON_IDLE=1 \\
  -e IDLE_SHUTDOWN_SEC={idle_sec} \\
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  h3-local:latest \\
  bash -lc 'curl -fsSL -H "Authorization: Bearer $WORKER_TOKEN" "$CONTROL_PLANE_URL/api/assets/runner/universal_worker.py" -o /opt/workbench/universal_worker.py && exec python3 -u /opt/workbench/universal_worker.py --platform local5090 --comfy-dir /opt/ComfyUI --work-dir /tmp/h3_work --idle-shutdown-sec {idle_sec}'
"""
    tmp_sh.write_bytes(script.encode("utf-8"))
    try:
        return run_remote(script=tmp_sh, timeout=7200)
    finally:
        tmp_sh.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-start 5090 tunnel, run command/script, auto-close tunnel.")
    parser.add_argument("--command", default="", help="Remote bash command to execute on 5090")
    parser.add_argument("--script", default="", help="Local .sh script to execute on 5090")
    parser.add_argument("--report-gpus", action="store_true", help="Report 4x RTX 5090 live status to gpu-workbench")
    parser.add_argument("--run-worker", action="store_true", help="Launch h3-local:latest worker on 5090 with persistent model cache")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index on 5090 (0-3) for --run-worker")
    parser.add_argument("--idle-sec", type=int, default=15, help="Idle shutdown seconds for --run-worker")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.report_gpus:
        report_gpus_to_cloud()
        return 0
    if args.run_worker:
        return run_local5090_worker(gpu_index=args.gpu_index, idle_sec=args.idle_sec)
    script_path = Path(args.script).resolve() if args.script else None
    return run_remote(command=args.command, script=script_path, timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())

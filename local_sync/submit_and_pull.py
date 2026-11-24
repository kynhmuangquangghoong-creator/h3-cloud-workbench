#!/usr/bin/env python3
"""
Unified Pure-Automation, Per-Job Triggered Cloud GPU Runner.

Lifecycle (Zero 24/7 Daemon, Zero Cloud Trace):
  1. Create job on gpu-workbench
  2. Upload local --input files (if any)
  3. Enqueue job (auto-launches Runpod pod if platform=runpod)
  4. Either wait inline OR spawn a one-shot hidden waiter (--detach-wait)
     strictly scoped to THIS job_id
  5. Download all outputs to local disk, verify exact byte sizes, save job_meta.json
  6. Immediately DELETE /api/jobs/<id>/purge (wipes R2 input+output and D1 record)
  7. Exit cleanly (0 lingering background processes)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from pull_and_purge import (
    DEFAULT_BASE_URL,
    DEFAULT_OUT_DIR,
    WorkbenchClient,
    log_line,
    resolve_token,
    wait_for_job,
)


def upload_input_file(client: WorkbenchClient, job_id: str, local_path: Path) -> str:
    fname = local_path.name
    ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
    url = f"{client.base_url}/api/jobs/{urllib.parse.quote(job_id, safe='')}/input/{urllib.parse.quote(fname, safe='')}"
    headers = client._headers()
    headers["Content-Type"] = ctype
    data = local_path.read_bytes()
    req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    with urllib.request.urlopen(req, timeout=300) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return str(res.get("key") or "")


def spawn_detached_waiter(job_id: str, base_url: str, out_dir: Path, interval: int, timeout_sec: int) -> int:
    pull_script = Path(__file__).resolve().with_name("pull_and_purge.py")
    py_exe = shutil.which("pythonw.exe") or sys.executable
    cmd = [
        py_exe,
        str(pull_script),
        "--base-url",
        base_url,
        "--out-dir",
        str(out_dir),
        "--wait-job",
        job_id,
        "--interval",
        str(interval),
        "--timeout",
        str(timeout_sec),
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )
    return proc.pid


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit a cloud GPU job and automatically pull & purge ONLY this job upon completion."
    )
    parser.add_argument("--workflow", required=True, help="Workflow id, e.g. audio_sfx, audio_ace_step, h3_relay")
    parser.add_argument("--title", default="", help="Human-readable local folder label")
    parser.add_argument("--platform", default="runpod", choices=["runpod", "colab", "local5090"])
    parser.add_argument("--gpu", default="auto", help="auto | rtx5090 | a100 | pro6000")
    parser.add_argument("--params-json", default="{}", help="JSON string of workflow params")
    parser.add_argument("--params-file", default="", help="Path to JSON file of workflow params")
    parser.add_argument("--input", action="append", default=[], help="Local input file to upload (repeatable)")
    parser.add_argument("--base-url", default=os.environ.get("GPU_WORKBENCH_URL", DEFAULT_BASE_URL))
    parser.add_argument("--token", default="")
    parser.add_argument("--out-dir", default=os.environ.get("GPU_LOCAL_OUT_DIR", DEFAULT_OUT_DIR))
    parser.add_argument("--interval", type=int, default=10, help="Wait check interval in seconds")
    parser.add_argument("--timeout", type=int, default=7200, help="Max wait seconds before giving up")
    parser.add_argument(
        "--detach-wait",
        action="store_true",
        help="Return immediately after enqueue while a one-shot hidden waiter pulls & purges ONLY this job and exits.",
    )
    args = parser.parse_args()

    if args.params_file:
        params = json.loads(Path(args.params_file).read_text(encoding="utf-8"))
    else:
        params = json.loads(args.params_json)

    token = resolve_token(args.token)
    out_dir = Path(args.out_dir).expanduser().resolve()
    log_file = out_dir / "_pull_sync.log"
    client = WorkbenchClient(args.base_url, token)

    # 1. Create job
    created = client.request_json(
        "POST",
        "/api/jobs",
        {
            "title": args.title or args.workflow,
            "platform": args.platform,
            "gpu_preference": args.gpu,
            "workflow": args.workflow,
            "params": params,
        },
    )
    job_id = str(created["id"])
    log_line(f"Submitted job {job_id} (workflow={args.workflow}, platform={args.platform}, gpu={args.gpu})", log_file)

    # 2. Upload local inputs in parallel (4 workers cuts 12-file upload from ~3 min to ~45s)
    if args.input:
        paths = [Path(inp).expanduser().resolve() for inp in args.input]
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(f"Input file not found: {p}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(upload_input_file, client, job_id, p): p for p in paths}
            for fut in concurrent.futures.as_completed(futs):
                p = futs[fut]
                key = fut.result()
                log_line(f"  -> Uploaded input {p.name} ({p.stat().st_size:,} bytes) -> {key}", log_file)

    # 3. Enqueue (auto-launches Runpod if platform == runpod)
    enq = client.request_json("POST", f"/api/jobs/{urllib.parse.quote(job_id, safe='')}/enqueue")
    log_line(f"  -> Enqueued {job_id}: {json.dumps(enq, ensure_ascii=False)}", log_file)

    # 4. Per-job auto-pull & zero-trace purge
    if args.detach_wait:
        pid = spawn_detached_waiter(job_id, args.base_url, out_dir, args.interval, args.timeout)
        log_line(
            f"  -> Spawned one-shot waiter (PID {pid}) for {job_id}. Will auto-pull to {out_dir}, purge cloud, and exit.",
            log_file,
        )
        print(json.dumps({"job_id": job_id, "waiter_pid": pid, "out_dir": str(out_dir)}, ensure_ascii=False))
        return 0

    return wait_for_job(
        client,
        job_id,
        out_dir,
        interval=args.interval,
        timeout_sec=args.timeout,
        log_file=log_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())

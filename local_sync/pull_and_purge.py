#!/usr/bin/env python3
"""
Zero-Trace Local Auto-Pull & Cloud Purge for ComfyUI Cloud Workbench.

Polls https://gpu-workbench.mamegoya6900.workers.dev for completed jobs,
downloads every output file to local disk, verifies exact byte sizes,
saves local job_meta.json, and immediately calls DELETE /api/jobs/<id>/purge
to wipe all R2 input/output objects and the D1 gpu_jobs record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://gpu-workbench.mamegoya6900.workers.dev"
DEFAULT_OUT_DIR = r"E:\GPU_Outputs"


def _read_windows_user_env(name: str) -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg  # type: ignore

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            val, _ = winreg.QueryValueEx(key, name)
            return str(val or "").strip()
    except OSError:
        return ""


def resolve_token(cli_token: str) -> str:
    if cli_token:
        return cli_token.strip()
    for var in ("GPU_WORKBENCH_TOKEN", "WORKER_TOKEN", "CLOUD_CONSOLE_TOKEN"):
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
        val = _read_windows_user_env(var)
        if val:
            return val
    raise RuntimeError(
        "Missing token. Pass --token or set GPU_WORKBENCH_TOKEN / CLOUD_CONSOLE_TOKEN in environment."
    )


def safe_name(text: str, max_len: int = 48) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (text or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._")
    return cleaned[:max_len] if cleaned else "job"


class WorkbenchClient:
    def __init__(self, base_url: str, token: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "gpu-workbench-local-puller/1.0",
        }

    def request_json(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        data = None
        headers = self._headers()
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    def download_asset(self, r2_key: str, dest: Path, expected_size: int | None = None) -> int:
        encoded = urllib.parse.quote(r2_key, safe="/")
        url = f"{self.base_url}/api/assets/{encoded}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        total = 0
        with urllib.request.urlopen(req, timeout=max(self.timeout, 300)) as resp, open(part, "wb") as f:
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        if expected_size is not None and total != expected_size:
            part.unlink(missing_ok=True)
            raise IOError(
                f"Size mismatch for {r2_key}: expected {expected_size} bytes, got {total} bytes"
            )
        if dest.exists():
            dest.unlink()
        part.replace(dest)
        return total


def log_line(msg: str, log_file: Path | None = None) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def process_job(
    client: WorkbenchClient,
    job: dict,
    out_dir: Path,
    dry_run: bool = False,
    log_file: Path | None = None,
) -> bool:
    job_id = str(job["id"])
    status = str(job.get("status") or "")
    title = str(job.get("title") or job_id)
    created = str(job.get("created_at") or "")
    date_part = created[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", created) else dt.date.today().isoformat()

    folder_name = f"{job_id}_{safe_name(title)}" if title != job_id else job_id
    job_dir = out_dir / date_part / folder_name

    listing = client.request_json("GET", f"/api/jobs/{urllib.parse.quote(job_id, safe='')}/files?scope=output")
    files = listing.get("files") or []

    log_line(
        f"Job {job_id} ({status}, workflow={job.get('workflow')}): {len(files)} output file(s) found.",
        log_file,
    )

    if dry_run:
        log_line(f"[DRY-RUN] Would save to {job_dir} and purge cloud job {job_id}.", log_file)
        return True

    job_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for fobj in files:
        rel_name = str(fobj.get("name") or "").lstrip("/")
        if not rel_name or ".." in Path(rel_name).parts:
            rel_name = safe_name(Path(str(fobj.get("key") or "output.bin")).name)
        dest = job_dir / rel_name
        expected_size = fobj.get("size")
        expected_int = int(expected_size) if expected_size is not None else None
        actual = client.download_asset(str(fobj["key"]), dest, expected_size=expected_int)
        downloaded.append({"name": rel_name, "bytes": actual, "r2_key": fobj["key"]})
        log_line(f"  -> Saved {dest} ({actual:,} bytes)", log_file)

    # Save full local metadata before cloud purge wipes D1 row
    meta = {
        "pulled_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local_dir": str(job_dir),
        "downloaded_files": downloaded,
        "cloud_job": job,
    }
    meta_path = job_dir / "job_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Purge cloud R2 (input + output) and D1 record
    purge_res = client.request_json("DELETE", f"/api/jobs/{urllib.parse.quote(job_id, safe='')}/purge")
    log_line(
        f"  -> Cloud purged for {job_id}: deleted {purge_res.get('deleted_objects', 0)} R2 object(s) & D1 record.",
        log_file,
    )

    # Double-safety second-level shutdown: if this job had a Runpod pod and no other jobs are queued/running, terminate pod now!
    pod_id = str(job.get("pod_id") or "").strip()
    if pod_id and str(job.get("platform") or "") == "runpod":
        try:
            ov = client.request_json("GET", "/api/overview")
            remaining_active = [
                j for j in (ov.get("jobs") or [])
                if str(j.get("status") or "") in ("uploading", "queued", "running")
            ]
            if not remaining_active:
                kill_res = client.request_json("DELETE", f"/api/pods/{urllib.parse.quote(pod_id, safe='')}")
                log_line(f"  -> Local double-safety pod terminate ({pod_id}): {kill_res}", log_file)
        except Exception as exc:
            log_line(f"  -> [WARN] Local pod terminate check error for {pod_id}: {exc}", log_file)
    return True


def run_cycle(
    client: WorkbenchClient,
    out_dir: Path,
    include_failed: bool = False,
    dry_run: bool = False,
    log_file: Path | None = None,
) -> int:
    overview = client.request_json("GET", "/api/overview")
    jobs = overview.get("jobs") or []
    target_statuses = {"completed", "failed"} if include_failed else {"completed"}
    ready = [j for j in jobs if str(j.get("status") or "") in target_statuses]
    processed = 0
    for job in ready:
        try:
            if process_job(client, job, out_dir, dry_run=dry_run, log_file=log_file):
                processed += 1
        except Exception as exc:
            log_line(f"[ERROR] Failed processing job {job.get('id')}: {exc}", log_file)
    return processed


def wait_for_job(
    client: WorkbenchClient,
    job_id: str,
    out_dir: Path,
    interval: int = 10,
    timeout_sec: int = 7200,
    dry_run: bool = False,
    log_file: Path | None = None,
) -> int:
    """Wait for a single specific job to finish, pull outputs, purge cloud, and exit."""
    t0 = time.time()
    last_state = ""
    log_line(
        f"Waiting for job {job_id} (interval={interval}s, timeout={timeout_sec}s) -> {out_dir}",
        log_file,
    )
    while True:
        try:
            overview = client.request_json("GET", "/api/overview")
            jobs = overview.get("jobs") or []
            job = next((j for j in jobs if str(j.get("id")) == job_id), None)
            if job is None:
                log_line(f"Job {job_id} not found in cloud (already purged or invalid). Exiting.", log_file)
                return 0

            status = str(job.get("status") or "")
            phase = str(job.get("phase") or "")
            detail = str(job.get("detail") or "")
            prog = job.get("progress")
            prog_str = f"{float(prog) * 100:.0f}%" if prog is not None else ""
            state_sig = f"{status}|{phase}|{prog_str}|{detail}"
            if state_sig != last_state:
                log_line(f"[{job_id}] {status} ({phase} {prog_str}) {detail}".strip(), log_file)
                last_state = state_sig

            if status in ("completed", "failed"):
                process_job(client, job, out_dir, dry_run=dry_run, log_file=log_file)
                log_line(f"Per-job sync finished for {job_id} ({status}). Exiting cleanly.", log_file)
                return 0 if status == "completed" else 2
        except urllib.error.URLError as exc:
            log_line(f"[WARN] Network error while waiting for {job_id}: {exc}", log_file)
        except Exception as exc:
            log_line(f"[WARN] Error while waiting for {job_id}: {exc}", log_file)

        if time.time() - t0 > timeout_sec:
            log_line(f"[TIMEOUT] Gave up waiting for {job_id} after {timeout_sec}s.", log_file)
            return 3
        time.sleep(max(3, interval))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-job zero-trace pull & cloud purge for ComfyUI Cloud Workbench."
    )
    parser.add_argument("--base-url", default=os.environ.get("GPU_WORKBENCH_URL", DEFAULT_BASE_URL))
    parser.add_argument("--token", default="")
    parser.add_argument("--out-dir", default=os.environ.get("GPU_LOCAL_OUT_DIR", DEFAULT_OUT_DIR))
    parser.add_argument(
        "--wait-job",
        default="",
        help="Wait ONLY for this specific job_id, pull & purge when done, then exit immediately.",
    )
    parser.add_argument("--once", action="store_true", help="Pull & purge any already-finished jobs once and exit.")
    parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds while waiting.")
    parser.add_argument("--timeout", type=int, default=7200, help="Max wait seconds for --wait-job (default 2h).")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also save error metadata locally and purge failed jobs from cloud (for --once / watch).",
    )
    parser.add_argument("--dry-run", action="store_true", help="List eligible jobs without downloading/purging.")
    parser.add_argument("--no-log-file", action="store_true", help="Disable writing _pull_sync.log.")
    args = parser.parse_args()

    token = resolve_token(args.token)
    out_dir = Path(args.out_dir).expanduser().resolve()
    log_file = None if args.no_log_file else (out_dir / "_pull_sync.log")
    client = WorkbenchClient(args.base_url, token)

    if args.wait_job:
        return wait_for_job(
            client,
            args.wait_job.strip(),
            out_dir,
            interval=args.interval,
            timeout_sec=args.timeout,
            dry_run=args.dry_run,
            log_file=log_file,
        )

    if args.once:
        count = run_cycle(
            client,
            out_dir,
            include_failed=args.include_failed,
            dry_run=args.dry_run,
            log_file=log_file,
        )
        log_line(f"Cycle complete. Pulled & purged {count} job(s).", log_file)
        return 0

    log_line(
        f"Watching {args.base_url} every {args.interval}s -> local dir {out_dir} (zero-trace purge ON)",
        log_file,
    )
    while True:
        try:
            run_cycle(
                client,
                out_dir,
                include_failed=args.include_failed,
                dry_run=args.dry_run,
                log_file=log_file,
            )
        except urllib.error.URLError as exc:
            log_line(f"[WARN] Network error: {exc}", log_file)
        except Exception as exc:
            log_line(f"[WARN] Unexpected error: {exc}", log_file)
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())

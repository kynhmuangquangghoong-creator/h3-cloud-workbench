-- GPU Cloud Workbench tables in console-db (metadata & queue only, zero binaries)

CREATE TABLE IF NOT EXISTS gpu_jobs (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  platform TEXT NOT NULL DEFAULT 'runpod',
  gpu_preference TEXT NOT NULL DEFAULT 'NVIDIA GeForce RTX 5090',
  workflow TEXT NOT NULL DEFAULT 'h3_relay',
  workflow_version TEXT NOT NULL DEFAULT 'v1',
  status TEXT NOT NULL DEFAULT 'queued',
  phase TEXT NOT NULL DEFAULT 'queued',
  stage TEXT NOT NULL DEFAULT '',
  progress REAL NOT NULL DEFAULT 0,
  detail TEXT NOT NULL DEFAULT '',
  params_json TEXT NOT NULL DEFAULT '{}',
  input_keys_json TEXT NOT NULL DEFAULT '[]',
  preview_key TEXT NOT NULL DEFAULT '',
  manifest_key TEXT NOT NULL DEFAULT '',
  worker_id TEXT NOT NULL DEFAULT '',
  pod_id TEXT NOT NULL DEFAULT '',
  telemetry_json TEXT NOT NULL DEFAULT '{}',
  error_msg TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_gpu_jobs_status_platform ON gpu_jobs(status, platform, created_at);

CREATE TABLE IF NOT EXISTS gpu_workers (
  worker_id TEXT PRIMARY KEY,
  platform TEXT NOT NULL,
  pod_id TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT 'idle',
  detail TEXT NOT NULL DEFAULT '',
  current_job_id TEXT NOT NULL DEFAULT '',
  telemetry_json TEXT NOT NULL DEFAULT '{}',
  last_heartbeat_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gpu_pods (
  pod_id TEXT PRIMARY KEY,
  gpu_type TEXT NOT NULL,
  cloud_type TEXT NOT NULL DEFAULT 'COMMUNITY',
  status TEXT NOT NULL DEFAULT 'starting',
  hourly_price REAL NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  last_heartbeat_at TEXT NOT NULL,
  terminated_at TEXT NOT NULL DEFAULT '',
  terminate_reason TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_gpu_pods_status ON gpu_pods(status);

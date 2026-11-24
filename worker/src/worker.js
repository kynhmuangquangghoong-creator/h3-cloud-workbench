import UI_HTML from "./ui.html";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type,Authorization",
};

function jsonResp(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS },
  });
}

function nowIso() {
  return new Date().toISOString();
}

function checkAuth(request, env) {
  const expected = env.WORKER_TOKEN;
  if (!expected) return true;
  const auth = request.headers.get("Authorization") || "";
  if (auth === `Bearer ${expected}`) return true;
  const url = new URL(request.url);
  if (url.searchParams.get("token") === expected) return true;
  return false;
}

const GPU_PRIORITY_MAP = {
  "video_auto": [
    { id: "NVIDIA GeForce RTX 5090", cloud: "COMMUNITY", price: 0.69 },
    { id: "NVIDIA A100 80GB PCIe", cloud: "COMMUNITY", price: 1.19 },
    { id: "NVIDIA A100-SXM4-80GB", cloud: "COMMUNITY", price: 1.39 },
    { id: "NVIDIA RTX PRO 6000 Blackwell Server Edition", cloud: "COMMUNITY", price: 1.69 },
    { id: "NVIDIA RTX PRO 4500", cloud: "COMMUNITY", price: 0.34 },
    { id: "NVIDIA GeForce RTX 5090", cloud: "SECURE", price: 1.39 },
    { id: "NVIDIA A100-SXM4-80GB", cloud: "SECURE", price: 2.19 },
  ],
  "audio_auto": [
    { id: "NVIDIA RTX PRO 4500", cloud: "COMMUNITY", price: 0.34 },
    { id: "NVIDIA GeForce RTX 4090", cloud: "COMMUNITY", price: 0.34 },
    { id: "NVIDIA RTX PRO 4000", cloud: "COMMUNITY", price: 0.50 },
    { id: "NVIDIA GeForce RTX 5090", cloud: "COMMUNITY", price: 0.69 },
    { id: "NVIDIA A100 80GB PCIe", cloud: "COMMUNITY", price: 1.19 },
  ],
  "rtx5090": [
    { id: "NVIDIA GeForce RTX 5090", cloud: "COMMUNITY", price: 0.69 },
  ],
  "rtx5090_any": [
    { id: "NVIDIA GeForce RTX 5090", cloud: "COMMUNITY", price: 0.69 },
    { id: "NVIDIA GeForce RTX 5090", cloud: "SECURE", price: 1.39 },
  ],
  "a100": [
    { id: "NVIDIA A100 80GB PCIe", cloud: "COMMUNITY", price: 1.19 },
    { id: "NVIDIA A100-SXM4-80GB", cloud: "COMMUNITY", price: 1.39 },
    { id: "NVIDIA A100-SXM4-80GB", cloud: "SECURE", price: 2.19 },
  ],
  "pro6000": [
    { id: "NVIDIA RTX PRO 6000 Blackwell Server Edition", cloud: "COMMUNITY", price: 1.69 },
    { id: "NVIDIA RTX PRO 6000 Blackwell Server Edition", cloud: "SECURE", price: 2.69 },
  ],
};

function buildInlineBootstrapCmd(origin) {
  // Fallback when using stock runpod/pytorch image before custom GHCR image is pushed.
  return [
    "bash",
    "-lc",
    [
      "set -euo pipefail",
      "export DEBIAN_FRONTEND=noninteractive HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1",
      "apt-get update -qq && apt-get install -y -qq git ffmpeg aria2 curl unzip rclone >/dev/null",
      "if [ ! -d /opt/ComfyUI ]; then git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI; fi",
      "pip install -q -r /opt/ComfyUI/requirements.txt av huggingface_hub hf_transfer websocket-client requests",
      "cd /opt/ComfyUI/custom_nodes",
      "for repo in 'facok/comfyui-SelfLift' 'goohai/Goohaitools-comfyui' 'kijai/ComfyUI-KJNodes' 'Kosinkadink/ComfyUI-VideoHelperSuite' 'smthemex/ComfyUI_Demucs' 'shinrali/ComfyUI-Qwen3-TTS' '1038lab/ComfyUI-QwenASR'; do " +
        "name=${repo#*/}; [ -d \"$name\" ] || git clone --depth 1 \"https://github.com/${repo}.git\" \"$name\"; " +
        "[ -f \"$name/requirements.txt\" ] && pip install -q -r \"$name/requirements.txt\" || true; done",
      "mkdir -p /opt/workbench",
      `curl -fsSL -H "Authorization: Bearer $WORKER_TOKEN" "${origin}/api/assets/runner/universal_worker.py" -o /opt/workbench/universal_worker.py`,
      "exec python3 -u /opt/workbench/universal_worker.py --platform runpod --comfy-dir /opt/ComfyUI --work-dir /tmp/h3_work",
    ].join(" && "),
  ];
}

async function runpodMcpCall(env, toolName, argsObj) {
  const resp = await fetch("https://mcp.runpod.io/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream",
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: Date.now() % 100000,
      method: "tools/call",
      params: { name: toolName, arguments: argsObj },
    }),
    signal: AbortSignal.timeout(20000),
  });
  const rawText = await resp.text();
  let parsed = null;
  try {
    parsed = JSON.parse(rawText);
  } catch {
    // Handle SSE response lines if mcp.runpod.io streams data: {...}
    for (const ln of rawText.split("\n")) {
      if (ln.startsWith("data:")) {
        try {
          parsed = JSON.parse(ln.slice(5).trim());
        } catch {}
      }
    }
  }
  const contentText = parsed?.result?.content?.[0]?.text || "";
  const isErr = Boolean(parsed?.error || parsed?.result?.isError || !resp.ok);
  let inner = null;
  if (contentText) {
    try {
      inner = JSON.parse(contentText);
    } catch {
      inner = { text: contentText };
    }
  }
  return { ok: !isErr, status: resp.status, data: inner, raw: contentText || rawText.slice(0, 300) };
}

async function launchRunpodPod(env, origin, gpuPrefKey = "auto", workflow = "h3_relay") {
  if (!env.RUNPOD_API_KEY) {
    return { launched: false, reason: "RUNPOD_API_KEY secret not configured on Worker" };
  }

  // Check if there is already an active Runpod pod starting/running
  const existing = await env.DB.prepare(
    "SELECT pod_id, gpu_type, status FROM gpu_pods WHERE status IN ('starting','running') ORDER BY created_at DESC LIMIT 1"
  ).first();
  if (existing) {
    return { launched: false, reused: true, pod_id: existing.pod_id, gpu_type: existing.gpu_type };
  }

  const imageName = (env.RUNPOD_IMAGE_NAME || "").trim() || "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404";
  const isCustomImage = imageName.includes("h3-cloud-workbench");
  const resolvedKey = gpuPrefKey === "auto" ? (String(workflow || "").startsWith("audio_") ? "audio_auto" : "video_auto") : gpuPrefKey;
  const candidates = GPU_PRIORITY_MAP[resolvedKey] || GPU_PRIORITY_MAP.video_auto;
  const errors = [];

  for (const cand of candidates) {
    const body = {
      name: `h3-worker-${Date.now().toString(36)}`,
      image: imageName,
      cloud: cand.cloud,
      disk: 90,
      gpu: {
        id: cand.id,
        count: 1,
        minCudaVersion: "12.8",
      },
      env: {
        WORKER_PLATFORM: "runpod",
        CONTROL_PLANE_URL: origin,
        WORKER_TOKEN: env.WORKER_TOKEN || "",
        HF_TOKEN: "{{ RUNPOD_SECRET_HF_TOKEN }}",
        IDLE_SHUTDOWN_SEC: "10",
        EXIT_ON_IDLE: "1",
      },
    };
    if (!isCustomImage) {
      body.entrypoint = buildInlineBootstrapCmd(origin);
    }

    const res = await runpodMcpCall(env, "create-pod", { body });
    const podId = res.data?.id || res.data?.pod?.id;
    if (res.ok && podId) {
      const ts = nowIso();
      await env.DB.prepare(
        "INSERT OR REPLACE INTO gpu_pods (pod_id, gpu_type, cloud_type, status, hourly_price, created_at, last_heartbeat_at) VALUES (?, ?, ?, 'starting', ?, ?, ?)"
      )
        .bind(podId, cand.id, cand.cloud, cand.price, ts, ts)
        .run();
      return { launched: true, pod_id: podId, gpu_type: cand.id, cloud_type: cand.cloud, hourly_price: cand.price };
    }
    errors.push(`${cand.id} (${cand.cloud}): HTTP ${res.status} ${String(res.raw || "").slice(0, 180)}`);
  }

  return { launched: false, reason: "No GPU stock matched", errors };
}

async function terminateRunpodPod(env, podId, reason = "manual") {
  if (!env.RUNPOD_API_KEY) {
    return { ok: false, error: "RUNPOD_API_KEY not configured" };
  }
  const res = await runpodMcpCall(env, "delete-pod", { id: podId });
  const notFound = String(res.raw || "").toLowerCase().includes("not found") || res.status === 404;
  const ok = res.ok || notFound;
  if (ok) {
    await env.DB.prepare(
      "UPDATE gpu_pods SET status = 'terminated', terminated_at = ?, terminate_reason = ? WHERE pod_id = ?"
    )
      .bind(nowIso(), reason, podId)
      .run();
  }
  return { ok, status: res.status, detail: res.raw };
}

async function purgeJobData(env, jobId) {
  const prefix = `jobs/${jobId}/`;
  let cursor = undefined;
  let deletedObjects = 0;
  while (true) {
    const listed = await env.GPU_JOBS.list({ prefix, cursor, limit: 1000 });
    const keys = (listed.objects || []).map((o) => o.key);
    if (keys.length > 0) {
      await env.GPU_JOBS.delete(keys);
      deletedObjects += keys.length;
    }
    if (!listed.truncated || !listed.cursor) break;
    cursor = listed.cursor;
  }
  await env.DB.prepare("DELETE FROM gpu_jobs WHERE id = ?").bind(jobId).run();
  return { ok: true, job_id: jobId, deleted_objects: deletedObjects };
}

async function runWatchdog(env) {
  const maxRunMin = Number(env.POD_MAX_RUNTIME_MIN || "90");
  const staleHbMin = Number(env.POD_STALE_HEARTBEAT_MIN || "15");
  const retentionHours = Number(env.JOB_RETENTION_HOURS || "24");
  const active = await env.DB.prepare(
    "SELECT pod_id, created_at, last_heartbeat_at FROM gpu_pods WHERE status IN ('starting','running')"
  ).all();

  const now = Date.now();
  const killed = [];
  for (const row of active.results || []) {
    const ageMin = (now - Date.parse(row.created_at)) / 60000;
    const hbAgeMin = (now - Date.parse(row.last_heartbeat_at || row.created_at)) / 60000;
    if (ageMin > maxRunMin) {
      await terminateRunpodPod(env, row.pod_id, `watchdog_max_runtime_${Math.round(ageMin)}m`);
      killed.push({ pod_id: row.pod_id, reason: "max_runtime" });
    } else if (hbAgeMin > staleHbMin) {
      await terminateRunpodPod(env, row.pod_id, `watchdog_stale_heartbeat_${Math.round(hbAgeMin)}m`);
      killed.push({ pod_id: row.pod_id, reason: "stale_heartbeat" });
    }
  }

  // Privacy safety net: auto-purge finished jobs & R2 files older than retentionHours
  const finished = await env.DB.prepare(
    "SELECT id, completed_at, updated_at FROM gpu_jobs WHERE status IN ('completed','failed')"
  ).all();
  const purged = [];
  for (const row of finished.results || []) {
    const refTs = Date.parse(row.completed_at || row.updated_at || "");
    if (!Number.isNaN(refTs) && (now - refTs) / 3600000 > retentionHours) {
      await purgeJobData(env, row.id);
      purged.push(row.id);
    }
  }
  return { killed, purged };
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS });
    }
    const url = new URL(request.url);
    const path = url.pathname;

    // Public UI shell (API calls require Bearer token)
    if (request.method === "GET" && (path === "/" || path === "/index.html")) {
      return new Response(UI_HTML, {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }

    if (!checkAuth(request, env)) {
      return jsonResp({ error: "Unauthorized: valid Bearer token required" }, 401);
    }

    // 1. Read R2 single-source assets or job outputs (streaming)
    if (request.method === "GET" && path.startsWith("/api/assets/")) {
      const key = decodeURIComponent(path.slice("/api/assets/".length));
      const obj = await env.GPU_JOBS.get(key);
      if (!obj) return jsonResp({ error: `Asset not found: ${key}` }, 404);
      const headers = new Headers(CORS);
      obj.writeHttpMetadata(headers);
      if (!headers.has("Content-Type")) {
        if (key.endsWith(".json")) headers.set("Content-Type", "application/json");
        else if (key.endsWith(".mp4")) headers.set("Content-Type", "video/mp4");
        else if (key.endsWith(".py")) headers.set("Content-Type", "text/x-python");
      }
      return new Response(obj.body, { headers });
    }

    // 2. Dashboard overview (jobs, workers, pods)
    if (request.method === "GET" && path === "/api/overview") {
      const [jobs, workers, pods] = await Promise.all([
        env.DB.prepare("SELECT * FROM gpu_jobs ORDER BY created_at DESC LIMIT 30").all(),
        env.DB.prepare("SELECT * FROM gpu_workers ORDER BY last_heartbeat_at DESC LIMIT 15").all(),
        env.DB.prepare("SELECT * FROM gpu_pods ORDER BY created_at DESC LIMIT 15").all(),
      ]);
      return jsonResp({
        jobs: jobs.results || [],
        workers: workers.results || [],
        pods: pods.results || [],
        image_name: env.RUNPOD_IMAGE_NAME || "",
        runpod_ready: Boolean(env.RUNPOD_API_KEY),
      });
    }

    // 2.5 Live Runpod GPU availability & pricing catalog (for compute fleet cockpit)
    if (request.method === "GET" && path === "/api/runpod/catalog") {
      if (!env.RUNPOD_API_KEY) return jsonResp({ gpus: [], balance: null, error: "RUNPOD_API_KEY not set" });
      try {
        const mcpCall = async (id, name, argsObj) => {
          const r = await fetch("https://mcp.runpod.io/", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json, text/event-stream",
              Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
            },
            body: JSON.stringify({ jsonrpc: "2.0", id, method: "tools/call", params: { name, arguments: argsObj } }),
            signal: AbortSignal.timeout(10000),
          });
          if (!r.ok) return null;
          const j = await r.json().catch(() => null);
          const txt = j?.result?.content?.[0]?.text;
          return txt ? JSON.parse(txt) : null;
        };
        const [gData, bData] = await Promise.all([
          mcpCall(1, "list-gpu-types", { include: ["AVAILABILITY"], product: ["POD"], limit: 50 }),
          mcpCall(2, "get-balance", {}),
        ]);
        const focus = new Set([
          "RTX PRO 4500", "RTX 4090", "RTX PRO 4000", "RTX 5090",
          "A100 PCIe", "A100 SXM", "RTX PRO 6000", "H100 SXM",
        ]);
        const list = (gData?.gpus || [])
          .filter((g) => focus.has(g.name) || (g.availability && g.availability !== "NONE" && g.memory >= 24))
          .map((g) => ({
            id: g.id,
            name: g.name,
            memory: g.memory,
            community: g.price?.community ?? null,
            secure: g.price?.secure ?? null,
            availability: g.availability || "NONE",
            dcs: (g.dataCenters || []).filter((d) => d.availability !== "NONE").map((d) => d.id),
          }))
          .sort((a, b) => {
            const avA = a.availability !== "NONE" ? 0 : 1;
            const avB = b.availability !== "NONE" ? 0 : 1;
            if (avA !== avB) return avA - avB;
            return (a.community || 99) - (b.community || 99);
          });
        return jsonResp({ gpus: list, balance: bData });
      } catch (e) {
        return jsonResp({ gpus: [], balance: null, error: String(e.message || e) });
      }
    }

    // 3. Create a new job
    if (request.method === "POST" && path === "/api/jobs") {
      const body = await request.json();
      const id = body.id || `h3_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
      const platform = body.platform || "runpod";
      const gpuPref = body.gpu_preference || "auto";
      const ts = nowIso();
      await env.DB.prepare(
        `INSERT INTO gpu_jobs
          (id, title, platform, gpu_preference, workflow, workflow_version, status, phase, detail, params_json, input_keys_json, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, 'uploading', 'uploading', 'Waiting for input assets', ?, '[]', ?, ?)`
      )
        .bind(
          id,
          body.title || id,
          platform,
          gpuPref,
          body.workflow || "h3_relay",
          body.workflow_version || "v1",
          JSON.stringify(body.params || {}),
          ts,
          ts
        )
        .run();
      return jsonResp({ id, status: "uploading" }, 201);
    }

    // 4. Upload input asset directly to R2 for a job
    const inMatch = path.match(/^\/api\/jobs\/([^/]+)\/input\/(.+)$/);
    if (request.method === "PUT" && inMatch) {
      const jobId = decodeURIComponent(inMatch[1]);
      const fname = decodeURIComponent(inMatch[2]).replace(/[^a-zA-Z0-9._-]/g, "_");
      const key = `jobs/${jobId}/input/${fname}`;
      await env.GPU_JOBS.put(key, request.body, {
        httpMetadata: { contentType: request.headers.get("Content-Type") || "application/octet-stream" },
      });
      const row = await env.DB.prepare("SELECT input_keys_json FROM gpu_jobs WHERE id = ?").bind(jobId).first();
      const keys = row ? JSON.parse(row.input_keys_json || "[]") : [];
      if (!keys.includes(key)) keys.push(key);
      await env.DB.prepare("UPDATE gpu_jobs SET input_keys_json = ?, updated_at = ? WHERE id = ?")
        .bind(JSON.stringify(keys), nowIso(), jobId)
        .run();
      return jsonResp({ ok: true, key });
    }

    // 5. Enqueue job & auto-launch Runpod zero-volume pod if platform == runpod
    const enqueueMatch = path.match(/^\/api\/jobs\/([^/]+)\/enqueue$/);
    if (request.method === "POST" && enqueueMatch) {
      const jobId = decodeURIComponent(enqueueMatch[1]);
      const job = await env.DB.prepare("SELECT * FROM gpu_jobs WHERE id = ?").bind(jobId).first();
      if (!job) return jsonResp({ error: "Job not found" }, 404);
      const ts = nowIso();
      const nowSec = Math.round(Date.now() / 1000);
      const initStages = [
        { id: "pod_boot", label: "建卡与容器就绪", status: "running", started_at: nowSec, ended_at: null, elapsed_sec: 0, detail: "已入队，正在拉起云端 GPU 实例", meta: job.platform || "runpod" },
        { id: "parallel_init", label: "并行冷启 (ComfyUI+权重+素材)", status: "pending", started_at: null, ended_at: null, elapsed_sec: null, detail: "等待容器就绪", meta: "" },
        { id: "seg_1", label: "分段采样推理", status: "pending", started_at: null, ended_at: null, elapsed_sec: null, detail: "待推理", meta: "" },
        { id: "stitch_upload", label: "拼接与回传中转", status: "pending", started_at: null, ended_at: null, elapsed_sec: null, detail: "待回传", meta: "" },
        { id: "shutdown", label: "本地拉回与秒级停表", status: "pending", started_at: null, ended_at: null, elapsed_sec: null, detail: "10s 内自动停表", meta: "" },
      ];
      await env.DB.prepare(
        "UPDATE gpu_jobs SET status = 'queued', phase = 'pod_boot', stage = '建卡拉起中', detail = 'Queued & launching GPU pod', telemetry_json = ?, updated_at = ? WHERE id = ?"
      )
        .bind(JSON.stringify({ stages: initStages, enqueued_at: ts }), ts, jobId)
        .run();

      let podLaunch = null;
      if (job.platform === "runpod") {
        podLaunch = await launchRunpodPod(env, url.origin, job.gpu_preference || "auto", job.workflow || "h3_relay");
        if (podLaunch.pod_id) {
          await env.DB.prepare("UPDATE gpu_jobs SET pod_id = ?, detail = ?, updated_at = ? WHERE id = ?")
            .bind(
              podLaunch.pod_id,
              podLaunch.launched
                ? `Runpod pod ${podLaunch.pod_id} (${podLaunch.gpu_type} ${podLaunch.cloud_type} $${podLaunch.hourly_price}/hr) starting...`
                : `Reusing active Runpod pod ${podLaunch.pod_id}`,
              nowIso(),
              jobId
            )
            .run();
        }
      }
      return jsonResp({ ok: true, job_id: jobId, pod_launch: podLaunch });
    }

    // 6. Worker claim next queued job for its platform
    if (request.method === "POST" && path === "/api/jobs/claim") {
      const body = await request.json();
      const platform = body.platform || "runpod";
      const workerId = body.worker_id || "unknown";
      const podId = body.pod_id || "";
      const row = await env.DB.prepare(
        "SELECT * FROM gpu_jobs WHERE status = 'queued' AND platform = ? ORDER BY created_at ASC LIMIT 1"
      )
        .bind(platform)
        .first();
      if (!row) return jsonResp({ job: null });

      const ts = nowIso();
      const res = await env.DB.prepare(
        "UPDATE gpu_jobs SET status = 'running', phase = 'claimed', worker_id = ?, pod_id = ?, updated_at = ? WHERE id = ? AND status = 'queued'"
      )
        .bind(workerId, podId, ts, row.id)
        .run();
      if (!res.meta.changes) return jsonResp({ job: null });

      return jsonResp({
        job: {
          id: row.id,
          title: row.title,
          platform: row.platform,
          workflow: row.workflow,
          workflow_version: row.workflow_version,
          params: JSON.parse(row.params_json || "{}"),
          input_keys: JSON.parse(row.input_keys_json || "[]"),
        },
      });
    }

    // 7. Worker heartbeat & pod liveness update
    if (request.method === "POST" && path === "/api/workers/heartbeat") {
      const b = await request.json();
      const ts = nowIso();
      await env.DB.prepare(
        `INSERT OR REPLACE INTO gpu_workers
          (worker_id, platform, pod_id, state, detail, current_job_id, telemetry_json, last_heartbeat_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      )
        .bind(
          b.worker_id,
          b.platform || "runpod",
          b.pod_id || "",
          b.state || "idle",
          b.detail || "",
          b.current_job_id || "",
          JSON.stringify(b.telemetry || {}),
          ts
        )
        .run();
      if (b.pod_id) {
        await env.DB.prepare(
          "UPDATE gpu_pods SET status = 'running', last_heartbeat_at = ? WHERE pod_id = ? AND status != 'terminated'"
        )
          .bind(ts, b.pod_id)
          .run();
      }
      return jsonResp({ ok: true });
    }

    // 8. Job progress heartbeat
    const hbMatch = path.match(/^\/api\/jobs\/([^/]+)\/heartbeat$/);
    if (request.method === "POST" && hbMatch) {
      const jobId = decodeURIComponent(hbMatch[1]);
      const b = await request.json();
      await env.DB.prepare(
        "UPDATE gpu_jobs SET phase = ?, stage = ?, progress = ?, detail = ?, telemetry_json = ?, updated_at = ? WHERE id = ?"
      )
        .bind(
          b.phase || "running",
          b.stage || "",
          Number(b.progress || 0),
          b.detail || "",
          JSON.stringify(b.telemetry || {}),
          nowIso(),
          jobId
        )
        .run();
      return jsonResp({ ok: true });
    }

    // 9. Upload job output file (preview.mp4 / manifest.json)
    const upMatch = path.match(/^\/api\/jobs\/([^/]+)\/upload\/(.+)$/);
    if (request.method === "PUT" && upMatch) {
      const jobId = decodeURIComponent(upMatch[1]);
      const fname = decodeURIComponent(upMatch[2]).replace(/[^a-zA-Z0-9._-]/g, "_");
      const key = `jobs/${jobId}/output/${fname}`;
      await env.GPU_JOBS.put(key, request.body, {
        httpMetadata: { contentType: request.headers.get("Content-Type") || "application/octet-stream" },
      });
      return jsonResp({ ok: true, key });
    }

    // 10. Mark job complete / fail
    const doneMatch = path.match(/^\/api\/jobs\/([^/]+)\/(complete|fail)$/);
    if (request.method === "POST" && doneMatch) {
      const jobId = decodeURIComponent(doneMatch[1]);
      const action = doneMatch[2];
      const b = await request.json();
      const ts = nowIso();
      if (action === "complete") {
        await env.DB.prepare(
          "UPDATE gpu_jobs SET status = 'completed', phase = 'completed', progress = 1.0, detail = ?, preview_key = ?, manifest_key = ?, updated_at = ?, completed_at = ? WHERE id = ?"
        )
          .bind(
            `Completed in ${b.elapsed_sec || "?"}s`,
            b.preview_key || `jobs/${jobId}/output/preview.mp4`,
            `jobs/${jobId}/output/manifest.json`,
            ts,
            ts,
            jobId
          )
          .run();
      } else {
        await env.DB.prepare(
          "UPDATE gpu_jobs SET status = 'failed', phase = 'failed', error_msg = ?, detail = 'Execution failed', updated_at = ? WHERE id = ?"
        )
          .bind(b.error || "Unknown error", ts, jobId)
          .run();
      }
      return jsonResp({ ok: true });
    }

    // 10.5 List R2 files for a job (used by local zero-trace auto-puller)
    const filesMatch = path.match(/^\/api\/jobs\/([^/]+)\/files$/);
    if (request.method === "GET" && filesMatch) {
      const jobId = decodeURIComponent(filesMatch[1]);
      const scope = url.searchParams.get("scope") || "output"; // output | all
      const prefix = scope === "all" ? `jobs/${jobId}/` : `jobs/${jobId}/output/`;
      const listed = await env.GPU_JOBS.list({ prefix, limit: 1000 });
      const files = (listed.objects || []).map((o) => ({
        key: o.key,
        name: o.key.slice(prefix.length),
        size: o.size,
        uploaded: o.uploaded,
      }));
      return jsonResp({ job_id: jobId, prefix, files });
    }

    // 10.6 Zero-trace purge: delete all R2 input/output objects and D1 row for a job
    const purgeMatch = path.match(/^\/api\/jobs\/([^/]+)\/purge$/);
    if ((request.method === "DELETE" || request.method === "POST") && purgeMatch) {
      const jobId = decodeURIComponent(purgeMatch[1]);
      const res = await purgeJobData(env, jobId);
      return jsonResp(res);
    }

    // 11. Self-terminate or manual kill pod
    if (request.method === "POST" && path === "/api/pods/self-terminate") {
      const b = await request.json();
      if (!b.pod_id) return jsonResp({ error: "pod_id required" }, 400);
      const res = await terminateRunpodPod(env, b.pod_id, b.reason || "worker_idle");
      return jsonResp(res);
    }

    const killMatch = path.match(/^\/api\/pods\/([^/]+)$/);
    if (request.method === "DELETE" && killMatch) {
      const podId = decodeURIComponent(killMatch[1]);
      const res = await terminateRunpodPod(env, podId, "manual_kill");
      return jsonResp(res);
    }

    return jsonResp({ error: "Not found" }, 404);
  },

  async scheduled(_event, env, ctx) {
    ctx.waitUntil(runWatchdog(env));
  },
};

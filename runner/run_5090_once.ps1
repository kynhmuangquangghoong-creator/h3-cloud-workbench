# One-click trigger to borrow 5090 temporarily via ephemeral Docker (--rm, zero residue).
param(
  [string]$GpuSpec = "all",
  [int]$IdleSec = 180,
  [string]$HfToken = $env:HF_TOKEN
)

$bridge = Get-Content "C:\Users\Administrator.LAPTOP-GL2SIF73\.gemini\antigravity\mcp\cloud-console-bridge.py" -Raw
$tok = [regex]::Match($bridge, 'CLOUD_CONSOLE_TOKEN",\s*"([^"]+)"').Groups[1].Value
if (-not $tok) { Write-Error "Cannot read WORKER_TOKEN from cloud-console-bridge.py"; exit 1 }

$cmd = "WORKER_TOKEN='$tok' HF_TOKEN='$HfToken' GPU_SPEC='$GpuSpec' IDLE_SHUTDOWN_SEC='$IdleSec' bash /home/xuliangliang/run_comfy_once.sh"
Write-Host "[5090] Starting ephemeral ComfyUI worker (gpus=$GpuSpec, auto-clean after ${IdleSec}s idle)..."
& 'E:\CodexData\tools\migration-runtime\python\python.exe' -B 'E:\Apps\audio-workspace\remote_exec.py' --timeout 7200 $cmd

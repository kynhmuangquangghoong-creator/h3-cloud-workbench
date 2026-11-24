#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/home/xuliangliang/apps/h3image
UV=/home/xuliangliang/.local/bin/uv
export UV_CACHE_DIR="$ROOT/cache/uv"
export XDG_CACHE_HOME="$ROOT/cache"
export TMPDIR="$ROOT/tmp"
export HF_HOME="$ROOT/cache/hf"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export UV_PYTHON_DOWNLOADS=never
export UV_NO_PROGRESS=1
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_BUILD_RUST_EXTS=none
mkdir -p "$ROOT/tmp" "$ROOT/cache" "$ROOT/logs"
trap 'code=$?; printf "{\"state\":\"failed\",\"exit\":%s}\n" "$code" > "$ROOT/logs/install_status.json"' ERR
printf '{"state":"source_download"}\n' > "$ROOT/logs/install_status.json"
python3 - <<'H3_PY'
import hashlib,json,pathlib,urllib.request,shutil
root=pathlib.Path('/home/xuliangliang/apps/h3image')
url='https://api.github.com/repos/sgl-project/sglang/commits/main'
record=root/'logs/source_revision.json'
if record.exists():commit=json.loads(record.read_text())['commit']
else:
    with urllib.request.urlopen(url,timeout=30) as f:commit=json.load(f)['sha']
archive=root/'sglang-source.tar.gz'
if not archive.exists():
    with urllib.request.urlopen('https://codeload.github.com/sgl-project/sglang/tar.gz/'+commit,timeout=45) as f, archive.with_suffix('.part').open('wb') as out:
        shutil.copyfileobj(f,out,1024*1024)
    archive.with_suffix('.part').replace(archive)
(root/'logs/source_revision.json').write_text(json.dumps({'commit':commit,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size}))
print('SOURCE',commit,archive.stat().st_size,flush=True)
H3_PY
if [ ! -d "$ROOT/sglang" ]; then
    mkdir "$ROOT/sglang"
    tar -xzf "$ROOT/sglang-source.tar.gz" -C "$ROOT/sglang" --strip-components=1
fi
if [ ! -x "$ROOT/.venv/bin/python" ]; then
    "$UV" --no-config venv --python /home/xuliangliang/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 --seed "$ROOT/.venv"
fi
printf '{"state":"installing"}\n' > "$ROOT/logs/install_status.json"
"$UV" --no-config pip install --python "$ROOT/.venv/bin/python" --index-url https://pypi.tuna.tsinghua.edu.cn/simple --torch-backend=cu130 -e "$ROOT/sglang/python[diffusion]"
/bin/bash "$ROOT/h3_cuda_layout_fix.sh"
export CUDA_HOME="$ROOT/cuda-toolkit"
export PATH="$ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
"$UV" --no-config pip freeze --python "$ROOT/.venv/bin/python" > "$ROOT/logs/dependencies.txt"
"$ROOT/.venv/bin/python" - <<'H3_PY'
import json,torch,sglang,transformers
print(json.dumps({'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu_count':torch.cuda.device_count(),'transformers':transformers.__version__,'sglang_file':sglang.__file__}))
H3_PY
"$ROOT/.venv/bin/sglang" serve --help > "$ROOT/logs/serve_help.txt"
printf '{"state":"complete"}\n' > "$ROOT/logs/install_status.json"

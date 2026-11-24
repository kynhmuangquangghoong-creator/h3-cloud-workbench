#!/usr/bin/env bash
set -eu
python3 - <<'H3_PY'
from pathlib import Path
import json
r=Path('/home/xuliangliang/apps/h3image')
assert (r/'H3IMAGE_TASK_OWNED').is_file()
vendor=r/'.venv/lib/python3.12/site-packages/nvidia/cu13'
facade=r/'cuda-toolkit'
facade.mkdir(exist_ok=True)
def link(target,dest):
 target.resolve().relative_to(r.resolve())
 if dest.exists() or dest.is_symlink():
  if not dest.is_symlink() or dest.resolve()!=target.resolve():raise RuntimeError('CUDA_FACADE_PATH_CONFLICT')
 else:dest.symlink_to(target,target_is_directory=target.is_dir())
for name in ('bin','include','nvvm','cccl','lib'):
 if (vendor/name).exists():link(vendor/name,facade/name)
lib64=facade/'lib64'
lib64.mkdir(exist_ok=True)
if lib64.is_symlink():raise RuntimeError('CUDA_FACADE_LIB64_MUST_BE_OWN_DIRECTORY')
for p in (vendor/'lib').iterdir():
 if p.is_file():link(p,lib64/p.name)
runtime=vendor/'lib/libcudart.so.13'
assert runtime.is_file()
link(runtime,lib64/'libcudart.so')
print(json.dumps({'cuda_home':str(facade),'runtime_link':str((lib64/'libcudart.so').resolve()),'binary':str((facade/'bin/nvcc').resolve()),'copied_weight_bytes':0}))
H3_PY

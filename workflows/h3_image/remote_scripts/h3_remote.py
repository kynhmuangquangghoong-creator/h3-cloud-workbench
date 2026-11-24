"""Task-owned H3 service supervisor and background image requests (remote only)."""
import argparse
import base64
import fcntl
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parent
LOG=ROOT/'logs'
PORT=30083
BASE=f'http://127.0.0.1:{PORT}'

def save(path,data):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2))
    tmp.replace(path)

def load(path,default=None):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return default

def identity(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError):return None

def healthy():
    try:
        with urllib.request.urlopen(BASE+'/health',timeout=3) as r:return r.status==200
    except Exception:return False

def status():
    state=load(LOG/'service.json',{})
    pid=state.get('supervisor_pid')
    state['supervisor_alive']=bool(pid and state.get('supervisor_identity') and identity(pid)==state.get('supervisor_identity'))
    state['healthy']=healthy() if state['supervisor_alive'] else False
    state['manual_hold']=state['supervisor_alive'] and held(state)
    state['active']=(ROOT/'active.flag').exists()
    return state

def held(state):
    try:
        hold=load(LOG/'manual_hold.json',{})
        return bool(state.get('supervisor_identity') and all(hold.get(k)==state.get(k) for k in ('supervisor_pid','supervisor_identity')))
    except (OSError,ValueError,TypeError,AttributeError):return False

def clear_hold(state):
    if held(state):(LOG/'manual_hold.json').unlink(missing_ok=True)

def keep(state):
    if not state.get('supervisor_identity'):raise RuntimeError('SUPERVISOR_IDENTITY_UNAVAILABLE')
    save(LOG/'manual_hold.json',{k:state[k] for k in ('supervisor_pid','supervisor_identity')})
    state['manual_hold']=True

def stop(release=False):
    if not release and (ROOT/'active.flag').exists():raise RuntimeError('GENERATION_ACTIVE')
    state=status()
    if state['supervisor_alive']:
        if not release:(ROOT/'stop.flag').touch()
        clear_hold(state)
    return status() if release else {'state':'stop_requested'}

def gpu_inventory():
    """Fail closed if either device memory or compute-process query fails."""
    devices=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],
                           capture_output=True,text=True,check=True,timeout=15).stdout
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
                             capture_output=True,text=True,check=True,timeout=15).stdout
    occupied=set()
    for line in processes.splitlines():
        if not line.strip():continue
        device,pid=[x.strip() for x in line.split(',')]
        int(pid)  # Unknown process data must not be treated as an idle device.
        occupied.add(device)
    inventory=[]
    for line in devices.splitlines():
        if not line.strip():continue
        index,device,used=[x.strip() for x in line.split(',')]
        inventory.append({'index':int(index),'uuid':device,'used_mib':float(used),
                          'has_compute_process':device in occupied})
    if not inventory:raise RuntimeError('GPU_INVENTORY_UNAVAILABLE')
    return inventory

def select_idle_gpu(inventory):
    for device in sorted(inventory,key=lambda item:item['index']):
        if 0<=device['used_mib']<128 and not device['has_compute_process']:
            return str(device['index'])
    raise RuntimeError('NO_IDLE_GPU_BELOW_128MIB_WITHOUT_COMPUTE_PROCESSES')

def start(hold=False):
    with (ROOT/'service.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=status()
        if not state['supervisor_alive']:(LOG/'manual_hold.json').unlink(missing_ok=True)
        if state['supervisor_alive']:
            (ROOT/'activity').touch()
            if hold:keep(state)
            return state
        if load(LOG/'download_status.json',{}).get('state')!='complete':raise RuntimeError('MODEL_DOWNLOAD_NOT_COMPLETE')
        if load(LOG/'install_status.json',{}).get('state')!='complete':raise RuntimeError('INSTALL_NOT_COMPLETE')
        config=load(ROOT/'service_config.json')
        if not config:raise RuntimeError('SERVICE_CONFIG_NOT_VALIDATED')
        selected=select_idle_gpu(gpu_inventory())
        with socket.socket() as sock:
            try:sock.bind(('127.0.0.1',PORT))
            except OSError:raise RuntimeError('PORT_IN_USE_DO_NOT_TERMINATE') from None
        config['devices']=selected
        save(ROOT/'service_config.json',config)
        (ROOT/'stop.flag').unlink(missing_ok=True)
        (ROOT/'activity').touch()
        with (LOG/'supervisor.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'supervise'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
        # The supervisor also writes its identity, allowing status after this call returns.
        state={'state':'starting','supervisor_pid':proc.pid,'devices':selected}
        if hold:
            state['supervisor_identity']=identity(proc.pid)
            keep(state)
        return state

def supervise():
    try:_supervise()
    finally:clear_hold({'supervisor_pid':os.getpid(),'supervisor_identity':identity(os.getpid())})

def _supervise():
    config=load(ROOT/'service_config.json')
    selected=config.get('devices','')
    # Recheck immediately before launching; never terminate an existing process.
    inventory=gpu_inventory()
    device=next((item for item in inventory if str(item['index'])==selected),None)
    if device is None or not (0<=device['used_mib']<128 and not device['has_compute_process']):
        raise RuntimeError('SELECTED_GPU_NO_LONGER_IDLE_DO_NOT_TERMINATE')
    env=os.environ.copy()
    cuda_home=(ROOT/'.venv').resolve().parent/'cuda-toolkit'
    env['PATH']=os.pathsep.join([str(ROOT/'.venv/bin'),str(cuda_home/'bin'),env.get('PATH','')])
    if (cuda_home/'bin/nvcc').is_file():env['CUDA_HOME']=str(cuda_home)
    env.update({'CUDA_VISIBLE_DEVICES':config.get('devices','0'),'HF_HOME':str(ROOT/'cache/hf'),
                'XDG_CACHE_HOME':str(ROOT/'cache'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                'HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','TMPDIR':str(ROOT/'tmp'),'PYTHONDONTWRITEBYTECODE':'1',
                'PYTHONNOUSERSITE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
                'TORCH_HOME':str(ROOT/'cache/torch'),'TORCH_EXTENSIONS_DIR':str(ROOT/'cache/torch_extensions'),
                'TRITON_CACHE_DIR':str(ROOT/'cache/triton'),'TORCHINDUCTOR_CACHE_DIR':str(ROOT/'cache/inductor'),
                'CUDA_CACHE_PATH':str(ROOT/'cache/cuda'),'CUTE_DSL_CACHE_DIR':str(ROOT/'cache/cute'),
                'SGLANG_CACHE_DIR':str(ROOT/'cache/sglang'),'SGLANG_JIT_CACHE_DIR':str(ROOT/'cache/sglang/jit'),
                'SGLANG_CUTE_AOT_CACHE_DIR':str(ROOT/'cache/sglang/cute_aot'),
                'FLASHINFER_WORKSPACE_BASE':str(ROOT/'cache/flashinfer')})
    if config.get('ffmpeg_path'):
        env['PATH']=str(Path(config['ffmpeg_path']).parent)+os.pathsep+env.get('PATH','')
        env['IMAGEIO_FFMPEG_EXE']=config['ffmpeg_path']
    command=[str(ROOT/'.venv/bin/sglang'),'serve','--model-path',config.get('model_path',str(ROOT/'models/MiniMax-H3-Image')),
             '--text-encoder-path',config.get('text_encoder_path',str(ROOT/'models/MiniMax-H3/text_encoder')),
             '--host','127.0.0.1','--port',str(PORT),'--warmup-mode','off']+config.get('extra_args',[])
    with (LOG/'service.log').open('ab') as log:
        proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=ROOT)
    state={'state':'starting','supervisor_pid':os.getpid(),'supervisor_identity':identity(os.getpid()),
           'service_pid':proc.pid,'service_identity':identity(proc.pid),'started_at':time.time(),
           'command':command,'devices':config.get('devices','0'),'idle_seconds':600}
    save(LOG/'service.json',state)
    became_healthy=False
    while proc.poll() is None:
        if not became_healthy and healthy():
            became_healthy=True
            (ROOT/'activity').touch()
            state.update(state='ready',startup_seconds=round(time.time()-state['started_at'],2))
            save(LOG/'service.json',state)
        idle=time.time()-(ROOT/'activity').stat().st_mtime
        stopping=(ROOT/'stop.flag').exists() or (became_healthy and idle>600 and not (ROOT/'active.flag').exists() and not held(state))
        if stopping:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait(timeout=15)
            state['state']='stopped'
            break
        if not became_healthy and time.time()-state['started_at']>1800:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
            state['state']='startup_timeout'
            break
        time.sleep(5)
    else:state['state']='failed'
    state.update(exit=proc.poll(),ended_at=time.time())
    save(LOG/'service.json',state)

def multipart(fields,images):
    boundary='h3image-'+uuid.uuid4().hex
    parts=[]
    for name,value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for index,path in enumerate(images):
        mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        parts.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="reference{index}{path.suffix}"\r\nContent-Type: {mime}\r\n\r\n'.encode(),path.read_bytes(),b'\r\n'])
    parts.append(f'--{boundary}--\r\n'.encode())
    return b''.join(parts),'multipart/form-data; boundary='+boundary

def run_job(directory):
    run=ROOT/'runs'/directory
    request=load(run/'request.json')
    initial=time.monotonic()
    state={'state':'running','request':{k:v for k,v in request.items() if k!='prompt'},'outputs':[]}
    stop=threading.Event()
    peaks={}
    def monitor():
        while not stop.is_set():
            try:
                out=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout
                for line in out.splitlines():
                    index,used=[int(x.strip()) for x in line.split(',')]
                    peaks[str(index)]=max(peaks.get(str(index),0),used)
            except Exception:pass
            stop.wait(1)
    with (ROOT/'request.lock').open('w') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            save(run/'status.json',{'state':'failed','error':'ANOTHER_GENERATION_ACTIVE'})
            return
        (ROOT/'active.flag').touch()
        thread=threading.Thread(target=monitor,daemon=True)
        thread.start()
        try:
            if not healthy():raise RuntimeError('SERVICE_NOT_HEALTHY')
            save(run/'status.json',state)
            images=[run/name for name in request.get('images',[])]
            for i in range(request['n']):
                fields={'prompt':request['prompt'],'size':request['size'],'seed':request['seed']+i,
                        'generator_device':'cpu','response_format':'b64_json','output_format':'png',
                        'min_generation_area':request['min_generation_area']}
                if images:
                    body,content_type=multipart(fields,images)
                    endpoint='/v1/images/edits'
                else:
                    body=json.dumps(fields).encode()
                    content_type='application/json'
                    endpoint='/v1/images/generations'
                began=time.monotonic()
                req=urllib.request.Request(BASE+endpoint,data=body,headers={'Content-Type':content_type})
                try:
                    with urllib.request.urlopen(req,timeout=3600) as response:payload=json.load(response)
                except urllib.error.HTTPError as e:
                    detail=e.read(2000).decode(errors='replace')
                    detail=re.sub(r'https?://\S+','[URL]',detail)
                    raise RuntimeError('HTTP_'+str(e.code)+' '+detail) from None
                http_seconds=round(time.monotonic()-began,2)
                postprocess_start=time.monotonic()
                raw=base64.b64decode(payload['data'][0]['b64_json'],validate=True)
                if raw[:8]!=b'\x89PNG\r\n\x1a\n':raise RuntimeError('INVALID_PNG_RESPONSE')
                import io
                from PIL import Image
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    if image.size!=tuple(map(int,request['size'].split('x'))):raise RuntimeError('OUTPUT_SIZE_MISMATCH')
                    if request.get('crop_uhd'):
                        image=image.crop((0,8,3840,2168))
                    # The API may return an uncompressed PNG; preserve every pixel
                    # while reducing traffic over the slow host bridge.
                    buffer=io.BytesIO()
                    image.save(buffer,format='PNG',compress_level=6)
                    raw=buffer.getvalue()
                name=f'image_{i+1:03}.png'
                (run/name).write_bytes(raw)
                state['outputs'].append({'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                                         'seconds':http_seconds,'postprocess_seconds':round(time.monotonic()-postprocess_start,2),'seed':fields['seed']})
                (ROOT/'activity').touch()
                save(run/'status.json',state)
            state.update(state='complete',elapsed_seconds=round(time.monotonic()-initial,2))
        except Exception as e:
            message=str(e) if isinstance(e,RuntimeError) else type(e).__name__
            state.update(state='failed',error=message,elapsed_seconds=round(time.monotonic()-initial,2))
        finally:
            stop.set()
            thread.join(timeout=12)
            state['peak_gpu_mib']=peaks
            save(run/'status.json',state)
            (ROOT/'active.flag').unlink(missing_ok=True)
            (ROOT/'activity').touch()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['start','status','stop','release-hold','supervise','run'])
    p.add_argument('--run')
    p.add_argument('--hold',action='store_true')
    a=p.parse_args()
    if a.action=='supervise':supervise()
    elif a.action=='run':
        if not a.run or not re.fullmatch(r'[a-f0-9]{32}',a.run):p.error('invalid run')
        run_job(a.run)
    elif a.action=='start':print(json.dumps(start(hold=a.hold)))
    elif a.action=='status':print(json.dumps(status()))
    elif a.action in ('stop','release-hold'):print(json.dumps(stop(release=a.action=='release-hold')))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if len(sys.argv)>1 and sys.argv[1] in ('supervise','run'):raise
        print(json.dumps({'error':str(error) if isinstance(error,RuntimeError) and re.fullmatch('[A-Z0-9_]+',str(error)) else 'OPERATION_FAILED'}))
        sys.exit(1)

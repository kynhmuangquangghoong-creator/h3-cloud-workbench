"""Anonymous, allowlisted ModelScope downloads; run on the 5090 host only."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
STOP = threading.Event()
LOCK = threading.Lock()
START = time.time()
DONE = []

def save(name, data):
    target=ROOT/'logs'/name
    with LOCK:
        tmp=target.with_suffix('.tmp')
        tmp.write_text(json.dumps(data,indent=2))
        tmp.replace(target)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def fetch_json(url):
    with urllib.request.urlopen(url,timeout=30) as f:return json.load(f)

def plan():
    result=[]
    for repo in ('MiniMax-H3-Image','MiniMax-H3'):
        j=fetch_json('https://modelscope.cn/api/v1/models/MiniMax/'+repo+'/repo/files?Revision=master&Recursive=true')
        if j.get('Code')!=200:raise RuntimeError('MODEL_LIST_FAILED')
        for entry in j['Data']['Files']:
            name=entry['Path']
            if entry['Type']!='blob':continue
            allowed=(name.startswith('text_encoder/') if repo=='MiniMax-H3' else
                     name=='model_index.json' or name.startswith(('pdd_','processor/','tokenizer/','transformer/','vae/')))
            if not allowed:continue
            if '..' in Path(name).parts or Path(name).is_absolute():raise RuntimeError('UNSAFE_PATH')
            sha=entry.get('Sha256','')
            if len(sha)!=64:raise RuntimeError('MISSING_SHA256')
            result.append({'repo':repo,'path':name,'size':entry['Size'],'sha256':sha})
    total=sum(x['size'] for x in result)
    if not 130_000_000_000 < total < 145_000_000_000:raise RuntimeError('UNEXPECTED_DOWNLOAD_SIZE')
    if shutil.disk_usage(ROOT).free < max(200_000_000_000,total+50_000_000_000):raise RuntimeError('DISK_BELOW_200GB')
    return result

def download(item):
    dest=ROOT/'models'/item['repo']/item['path']
    dest.parent.mkdir(parents=True,exist_ok=True)
    partial=dest.with_name(dest.name+'.part')
    if dest.exists():
        if dest.stat().st_size==item['size'] and digest(dest)==item['sha256']:return complete(item)
        raise RuntimeError('EXISTING_FILE_MISMATCH')
    url='https://modelscope.cn/api/v1/models/MiniMax/'+item['repo']+'/repo?'+urllib.parse.urlencode({'Revision':'master','FilePath':item['path']})
    for attempt in range(5):
        if STOP.is_set():raise RuntimeError('STOPPED')
        offset=partial.stat().st_size if partial.exists() else 0
        if offset==item['size']:break
        if offset>item['size']:raise RuntimeError('PART_TOO_LARGE')
        try:
            headers={'Range':f'bytes={offset}-'} if offset else {}
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=30) as response:
                if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                    raise RuntimeError('RANGE_NOT_HONORED')
                with partial.open('ab' if offset else 'wb') as f:
                    while not STOP.is_set():
                        block=response.read(4*1024*1024)
                        if not block:break
                        f.write(block)
                    if STOP.is_set():raise RuntimeError('STOPPED')
            if partial.stat().st_size==item['size']:break
        except urllib.error.HTTPError as e:
            if e.code in (401,403):
                STOP.set()
                raise RuntimeError('AUTH_OR_PERMISSION_REQUIRED_HTTP_'+str(e.code)) from None
            print('RETRY',item['path'],'HTTP',e.code,flush=True)
        except (urllib.error.URLError,TimeoutError,OSError):
            print('RETRY',item['path'],'NETWORK',flush=True)
        if attempt<4:time.sleep(2)
    if partial.stat().st_size!=item['size']:raise RuntimeError('INCOMPLETE_DOWNLOAD')
    if digest(partial)!=item['sha256']:raise RuntimeError('SHA256_MISMATCH')
    partial.replace(dest)
    return complete(item)

def complete(item):
    with LOCK:DONE.append(item['path'])
    save('download_status.json',{'state':'downloading','verified_files':len(DONE),'verified_bytes':sum(x['size'] for x in ITEMS if x['path'] in DONE),'elapsed_seconds':round(time.time()-START,2)})
    print('VERIFIED',item['repo'],item['path'],item['size'],flush=True)

if __name__=='__main__':
    ITEMS=[]
    try:
        manifest=ROOT/'logs/download_manifest.json'
        ITEMS=json.loads(manifest.read_text()) if manifest.exists() else plan()
        if not manifest.exists():manifest.write_text(json.dumps(ITEMS,indent=2))
        save('download_status.json',{'state':'downloading','files':len(ITEMS),'bytes':sum(x['size'] for x in ITEMS)})
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            futures=[pool.submit(download,item) for item in ITEMS]
            for future in concurrent.futures.as_completed(futures):future.result()
        save('download_status.json',{'state':'complete','files':len(ITEMS),'bytes':sum(x['size'] for x in ITEMS),'elapsed_seconds':round(time.time()-START,2),'sha256_verified':True})
    except Exception as e:
        STOP.set()
        message=str(e) if isinstance(e,RuntimeError) else type(e).__name__
        save('download_status.json',{'state':'failed','error':message,'verified_files':len(DONE),'elapsed_seconds':round(time.time()-START,2)})
        print('FAILED',message,flush=True)
        raise SystemExit(1)

"""Launch balanced policy windows, then independent diagnostics, inside an allocated job."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['SLURM_SUBMIT_DIR'])
OUT=Path(os.environ['RESULT_DIR'])
BUILD=Path(os.environ.get('PHASE3_BUILD_DIR', str(ROOT/'outputs/a800/phase3/extension-build'))).resolve()
SCRIPTS=OUT/'scripts'
config_path=SCRIPTS/os.environ.get('PHASE3_CONFIG','extension.json')
config=json.loads(config_path.read_text())
world=int(os.environ['PHASE3_TP']);assert world in config['world_sizes']
manifest=json.loads((BUILD/'manifest.json').read_text())
if 'config_sha256' in manifest:
 assert hashlib.sha256(config_path.read_bytes()).hexdigest()==manifest['config_sha256']
# Fail before timing if any frozen library, binding or Python module changed.
for name,expected in manifest['files'].items():
 assert hashlib.sha256((BUILD/name).read_bytes()).hexdigest()==expected,name
shutil.copy2(BUILD/'manifest.json',OUT/'manifest.json')
for name in ['manifest.json','commands.json']:shutil.copy2(ROOT/'outputs/a800/phase3/build'/name,OUT/('parent-'+name))
(OUT/'experiment.json').write_text(json.dumps(dict(config,world_size=world,build=str(BUILD)),indent=2)+'\n')
commands=[]
def run(cmd,log,env=None):
 commands.append({'args':list(map(str,cmd)),'env':env or {}})
 (OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 with log.open('w') as f:subprocess.run(cmd,env=dict(os.environ,**(env or {})),stdout=f,stderr=subprocess.STDOUT,check=True)
if os.environ.get('PHASE3_IPC_CHECK') == '1':
 # Probe the unchanged library on the same allocation, then require corrected IPC reads/writes.
 old=ROOT/'outputs/a800/phase3/extension-build/original'
 old_env=dict(PYTHONPATH=str(old/'python'),LD_LIBRARY_PATH=f'{old}:/data/apps/cuda/12.8/lib64',
              TORCH_SHOW_CPP_STACKTRACES='1',TORCH_DISABLE_ADDR2LINE='1')
 probe=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
        str(SCRIPTS/'ipc_check.py'),'--probe']
 commands.append({'args':probe,'env':old_env})
 (OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 with (OUT/'ipc-original-probe.log').open('w') as f:
  result=subprocess.run(probe,env=dict(os.environ,**old_env),stdout=f,stderr=subprocess.STDOUT)
 original_log=(OUT/'ipc-original-probe.log').read_text()
 reproduced=result.returncode != 0 and 'does not match device of data' in original_log
 assert result.returncode == 0 or reproduced, 'Original probe failed for an unexpected reason'
 (OUT/'ipc-original-probe.json').write_text(json.dumps(dict(returncode=result.returncode,
    device_mismatch_reproduced=reproduced),indent=2)+'\n')
 fixed=BUILD/'original'
 ipc_env=dict(PYTHONPATH=str(fixed/'python'),LD_LIBRARY_PATH=f'{fixed}:/data/apps/cuda/12.8/lib64')
 for count,reverse in [(2,False),(world,False),(world,True)]:
  cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={count}',
       str(SCRIPTS/'ipc_check.py')]+(['--reverse'] if reverse else [])
  run(cmd,OUT/f'ipc-fixed-tp{count}-reverse{int(reverse)}.log',ipc_env)
 print('IPC PREFLIGHT PASS; starting the unchanged performance protocol',flush=True)
for policy in config['policies']:
 shutil.copy2(BUILD/policy/'swizzle.patch',OUT/f'{policy}.patch')
 for m,n,k in config['shapes']:
  case=OUT/f'tp{world}-m{m}-n{n}-k{k}';case.mkdir(exist_ok=True)
  with (case/f'mapping-{policy}.csv').open('w') as f,(case/f'mapping-{policy}.log').open('w') as err:
   policy_id={'original':0,'remote_first':1,'interleaved':2}[policy]
   cmd=[str(BUILD/policy/'check_mapping'),str(m),str(n),str(k),str(world),str(policy_id)]
   commands.append({'args':cmd});subprocess.run(cmd,stdout=f,stderr=err,check=True)
window=0
for diagnostic,orders in [(False,config['orders']),(True,[config['policies']])]:
 for block,order in enumerate(orders):
  for policy in order:
   env=dict(PHASE3_POLICY=policy,PHASE3_BLOCK=str(block),PHASE3_WINDOW=str(window),
     PHASE3_LIBRARY=str(BUILD/policy/'libflux_cuda.so'),PYTHONPATH=str(BUILD/policy/'python'),
     LD_LIBRARY_PATH=f'{BUILD/policy}:/data/apps/cuda/12.8/lib64',
     PHASE3_DIAGNOSTIC=str(int(diagnostic)),PHASE3_DIAGNOSTIC_ONLY=str(int(diagnostic)))
   print('START',world,window,policy,'diagnostic',diagnostic,flush=True)
   run([sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
        str(SCRIPTS/'extension_window.py')],OUT/f'window-{window:02d}-{policy}.log',env)
   window+=1
run([sys.executable,str(SCRIPTS/'summarize_extension.py'),str(OUT)],OUT/'analysis.log')
print('EXTENSION COMPLETE',OUT,flush=True)

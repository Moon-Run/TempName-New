"""Calibration and independent, balanced five-policy operator measurements."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['SLURM_SUBMIT_DIR']);OUT=Path(os.environ['RESULT_DIR']);S=OUT/'scripts'
CFG=json.loads((S/'config.json').read_text());STAGE=os.environ['ARRIVAL_STAGE']
ARTIFACT_ROOT=Path(os.environ.get('ARRIVAL_OUTPUT_ROOT',str(ROOT/'outputs/a800/arrival8')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(build):
 manifest=json.loads((build/'manifest.json').read_text())
 for name,digest in manifest['files'].items():assert sha(build/name)==digest,name
 shutil.copy2(build/'manifest.json',OUT/(build.name+'-manifest.json'))
commands=[]
(OUT/'script-hashes.json').write_text(json.dumps({p.name:sha(p) for p in S.iterdir() if p.is_file()},indent=2)+'\n')
def run(cmd,name,env=None):
 commands.append(dict(command=cmd,name=name,environment=env or {}))
 (OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 with (OUT/(name+'.log')).open('w') as f:
  r=subprocess.run(cmd,env=dict(os.environ,**(env or {})),stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:
  print((OUT/(name+'.log')).read_text()[-12000:],flush=True)
  raise RuntimeError((name,r.returncode))
 print('DONE',name,flush=True)
if STAGE=='calibrate':
 source=ARTIFACT_ROOT/'calibration-scripts'
 shutil.copytree(source,OUT/'calibration-scripts')
 build=ROOT/'outputs/a800/arrival8/mechanism-build';verify(build)
 base=ROOT/'outputs/a800/phase3/three-way-build';verify(base)
 for p in ['remote_first','interleaved']:
  for m,n,k in CFG['shapes']:
   with (OUT/f'mapping-{p}-{m}-{n}-{k}.csv').open('w') as f,(OUT/f'mapping-{p}-{m}-{n}-{k}.log').open('w') as e:
    subprocess.run([str(ROOT/'outputs/a800/megatron-e2e/scale-mapping'/f'check_mapping-{p}'),str(m),str(n),str(k),'8',str(1 if p=='remote_first' else 2)],stdout=f,stderr=e,check=True)
  path=build/p
  env=dict(MECH_POLICY=p,MECH_KIND='instrumented',MECH_BLOCK='0',MECH_LIBRARY=str(path/'libflux_cuda.so'),PYTHONPATH=str(path/'python'),LD_LIBRARY_PATH=f'{path}:/data/apps/cuda/12.8/lib64')
  run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(OUT/'calibration-scripts/worker.py')],p,env)
 (OUT/'calibration-complete.json').write_text(json.dumps(dict(completed=True,config=CFG,scope='Exploratory arrival labels; independent uninstrumented performance is required.'),indent=2)+'\n')
elif STAGE in ('operator','operator-preflight'):
 build=ARTIFACT_ROOT/'build';verify(build)
 # Timing is admitted only after all-model-policy and exact-shape preflight succeeds.
 if STAGE=='operator':
  preflight=Path(os.environ['E2E_SCALE_PREFLIGHT'])
  checked=json.loads((preflight/'preflight.json').read_text());assert checked['completed']
  assert checked['flux_manifest_sha256']==sha(build/'manifest.json')
  shutil.copy2(preflight/'preflight.json',OUT/'preflight.json')
 orders=CFG['orders'] if STAGE=='operator' else []
 if os.environ.get('ARRIVAL_REVERSE')=='1':orders=[list(reversed(o)) for o in reversed(orders)]
 CFG['orders']=orders
 (OUT/'experiment.json').write_text(json.dumps(CFG,indent=2)+'\n')
 for p in CFG['policies']:
  for m,n,k in CFG['shapes']:
   case=OUT/f'tp8-m{m}-n{n}-k{k}';case.mkdir(exist_ok=True)
   with (case/f'mapping-{p}.csv').open('w') as f,(case/f'mapping-{p}.log').open('w') as e:
    subprocess.run([str(build/'mapping'/f'check_mapping-{p}'),str(m),str(n),str(k),'8',str(CFG['policies'].index(p))],stdout=f,stderr=e,check=True)
 window=0
 for diagnostic,blocks in [(False,orders),(True,[CFG['policies']])]:
  for block,order in enumerate(blocks):
   for p in order:
    path=build/p
    env=dict(PHASE3_CONFIG='config.json',PHASE3_POLICY=p,PHASE3_BLOCK=str(block),PHASE3_WINDOW=str(window),PHASE3_LIBRARY=str(path/'libflux_cuda.so'),PYTHONPATH=str(path/'python'),LD_LIBRARY_PATH=f'{path}:/data/apps/cuda/12.8/lib64',PHASE3_DIAGNOSTIC=str(int(diagnostic)),PHASE3_DIAGNOSTIC_ONLY=str(int(diagnostic)))
    run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(S/'operator_worker.py')],f'window-{window:02d}-{p}',env)
    window+=1
 run([sys.executable,str(S/'summarize_operator.py'),str(OUT)],'summary')
else:raise ValueError(STAGE)

"""Measure all six BF16 policies inside one existing TP8 allocation, without a speed gate."""
import datetime,hashlib,json,os,shutil,signal,subprocess,sys,traceback
from pathlib import Path
from report import report
root=Path(sys.argv[1]).resolve();info=json.loads((root/'submission.json').read_text());repo=Path(info['repo'])
assert os.environ.get('SLURM_JOB_ID')==str(info['job_id'])
assert not (root/'state.json').exists(),'Use a fresh experiment directory'
now=lambda:datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
state=dict(status='running',job_id=info['job_id'],step_id=os.environ.get('SLURM_STEP_ID'),started_at=now(),stages=[])
def save():
 state['updated_at']=now();p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')
def verify():
 for name,h in info['files'].items():assert sha(root/name)==h,name
 for name,h in info['protected_sources'].items():assert sha(name)==h,name
 assert sha(Path(info['build'])/'manifest.json')==info['build_manifest_sha256']
def stopped(signum,frame):raise RuntimeError(f'Test controller received signal {signum}')
signal.signal(signal.SIGTERM,stopped)
try:
 save();verify()
 snapshot=subprocess.check_output(['squeue','--steps','-h','-j',str(info['job_id']),'-o','%i'],text=True)
 allowed={f"{info['job_id']}.batch",f"{info['job_id']}.extern",f"{info['job_id']}.{state['step_id']}"}
 assert set(snapshot.split())<=allowed,snapshot
 (root/'initial-slurm-steps.txt').write_text(snapshot)
 for name,mode,reverse in [('preflight','preflight',0),('model-1','timing',0),('model-2','timing',1)]:
  verify();directory=root/name;directory.mkdir();shutil.copytree(root/'scripts',directory/'scripts')
  python=Path(info['python']);cuda='/data/apps/cuda/12.4'
  env=dict(os.environ,SLURM_SUBMIT_DIR=str(repo),ARRIVAL_OUTPUT_ROOT=str(Path(info['build']).parent),CUDA_HOME=cuda,CUDA_PATH='/data/apps/cuda/12.8',PATH=f'{python.parent}:{cuda}/bin:'+os.environ['PATH'],CUDA_DEVICE_MAX_CONNECTIONS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='0',NCCL_P2P_LEVEL='NVL',FLUX_FORCE_NVLINK='1',RESULT_DIR=str(directory),E2E_CASE=info['case'],E2E_SCALE_STAGE=mode,E2E_SCALE_PREFLIGHT=str(root/'preflight'),ARRIVAL_REVERSE=str(reverse))
  entry=dict(name=name,started_at=now());state['stages'].append(entry);state['stage']=name;save()
  print(now(),'START',name,flush=True)
  with (directory/'stage.log').open('w') as log:
   process=subprocess.Popen([str(python),str(directory/'scripts/launch.py')],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=process.wait()
   finally:
    if process.poll() is None:
     os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
  entry.update(exit_code=rc,finished_at=now());save();assert rc==0,f'{name} failed; see stage.log'
  artifact='preflight.json' if mode=='preflight' else 'analysis.json';assert json.loads((directory/artifact).read_text())['completed']
  print(now(),'DONE',name,flush=True)
 verify();report(root)
 state.update(status='completed',stage='completed',finished_at=now())
except BaseException:
 state.update(status='failed',error=traceback.format_exc(),finished_at=now());raise
finally:save()

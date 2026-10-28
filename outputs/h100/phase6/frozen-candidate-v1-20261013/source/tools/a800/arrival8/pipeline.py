"""Resume a finite, fail-closed experiment after the ordinary queue supplies GPUs.

No changes to unrelated jobs. Builds are serial on the launch host. Performance
jobs may be prequeued with afterok dependencies; execution requires preflight success. Each job is
awaited before the next, so this experiment cannot compete with itself.
"""
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
LOG=ROOT/'logs/a800/arrival8';LOG.mkdir(exist_ok=True)
STATE=LOG/'pipeline.json'

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(state):
 state['updated_at']=now();tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
def command(args,log):
 with (LOG/log).open('w') as f:subprocess.run(list(map(str,args)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def wait(job,state,label):
 state['stage']=label;state['waiting_job']=job;write(state)
 last=None
 while True:
  r=subprocess.run(['sacct','-j',str(job),'--format=JobIDRaw,State,ExitCode','-n','-P'],capture_output=True,text=True)
  rows=[line.split('|') for line in r.stdout.splitlines() if line.split('|')[0]==str(job)]
  if rows:
   status,exit_code=rows[0][1:3]
   state['scheduler_state']=status
   if status!=last:write(state);print(now(),label,job,status,flush=True);last=status
   if status=='COMPLETED':
    out=LOG/str(job)
    if (out/'exit-code.txt').exists():
     assert exit_code=='0:0' and (out/'exit-code.txt').read_text().strip()=='0'
     return out
   elif status.split()[0].rstrip('+') in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}:
    raise RuntimeError(f'{label}: job {job} {status} {exit_code}; see {LOG/str(job)}')
  time.sleep(30)
def submit(stage,state,extra=None,dependency=None):
 exports=dict(ARRIVAL_STAGE=stage,**(extra or {}))
 arg='ALL,'+','.join(f'{k}={v}' for k,v in exports.items())
 cmd=['sbatch','--parsable','--qos=normal','--job-name=arrival8-'+stage,'--export='+arg]
 if dependency:cmd.append(f'--dependency=afterok:{dependency}')
 if stage=='preflight':cmd.append('--time=00:20:00')
 cmd.append(str(HERE/'run.sbatch'))
 job=int(subprocess.check_output(cmd,cwd=ROOT,text=True).strip().split(';')[0])
 state['jobs'].append(dict(stage=stage,job_id=job,export=exports,submitted_at=now(),dependency=f'afterok:{dependency}' if dependency else None,qos='normal'))
 write(state);return job

def main(calibration_job):
 lock=(LOG/'pipeline.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if STATE.exists():
  state=json.loads(STATE.read_text())
  assert state['calibration_job']==calibration_job
 else:
  state=dict(created_at=now(),calibration_job=calibration_job,status='running',jobs=[],
      scripts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*') if p.is_file()})
 for name,digest in state['scripts'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, 'Scripts changed; review before resuming: '+name
 state['status']='running';state.pop('error',None);write(state)
 try:
  calibration=wait(calibration_job,state,'calibration')
  plan=ROOT/'outputs/a800/arrival8/plan.json'
  state['stage']='fit_and_build';write(state)
  if not plan.exists():command([sys.executable,HERE/'plan.py',calibration,plan],'fit.log')
  build=ROOT/'outputs/a800/arrival8/build'
  if not (build/'manifest.json').exists():
   # Partial failed builds are retained for diagnosis; do not overwrite them.
   assert not build.exists(), 'Partial build exists; inspect build.log before retrying'
   command([sys.executable,HERE/'build.py',plan],'build.log')
  def job_for(stage,index=0,extra=None):
   prior=[j for j in state['jobs'] if j['stage']==stage]
   return prior[index]['job_id'] if len(prior)>index else submit(stage,state,extra)
  preflight_job=job_for('preflight')
  previous=preflight_job
  for iteration in range(2):
   extra=dict(E2E_SCALE_PREFLIGHT=str(LOG/str(preflight_job)),ARRIVAL_REVERSE=str(iteration))
   for stage in ('operator','timing'):
    prior=[j for j in state['jobs'] if j['stage']==stage]
    if len(prior)>iteration:previous=prior[iteration]['job_id']
    else:previous=submit(stage,state,extra,dependency=previous)
  preflight=wait(preflight_job,state,'preflight')
  assert json.loads((preflight/'preflight.json').read_text())['completed']
  assert json.loads((preflight/'operator-checks/analysis.json').read_text())['completed']
  for iteration in range(2):
   extra=dict(E2E_SCALE_PREFLIGHT=str(preflight),ARRIVAL_REVERSE=str(iteration))
   op=wait(job_for('operator',iteration,extra),state,f'operator_{iteration+1}')
   assert json.loads((op/'analysis.json').read_text())['completed']
   model=wait(job_for('timing',iteration,extra),state,f'e2e_{iteration+1}')
   assert json.loads((model/'analysis.json').read_text())['completed']
  state['stage']='report';write(state)
  command([sys.executable,HERE/'report.py',STATE],'report.log')
  state['status']='completed';state['stage']='completed';write(state)
 except Exception:
  state['status']='failed';state['error']=traceback.format_exc();write(state);raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('calibration_job',type=int);v=a.parse_args();main(v.calibration_job)

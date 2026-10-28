"""Serial exploratory scenarios, followed by independent balanced confirmation."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import shutil

root=Path(__file__).resolve().parent
info=json.loads((root/'suite.json').read_text());repo=Path(info['repo'])
assert os.environ.get('SLURM_JOB_ID')=='179147'
state=dict(status='running',stage='starting',results=[],job_id=179147,step_id=os.environ.get('SLURM_STEP_ID'))
def save():
    state['updated_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    p=root/'suite-state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'suite-state.json')
def run_campaign(path):
    state['stage']=path.name;save()
    with (path/'driver.log').open('w') as log:
        subprocess.run([sys.executable,str(path/'campaign.py')],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (path/'verification.log').open('w') as log:
        subprocess.run([sys.executable,str(repo/'tools/a800/phase6/verify_run.py'),str(path)],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
try:
    state['stage']='operator-sweep';save()
    operator=root/'operator-sweep';operator.mkdir();scripts=operator/'scripts';scripts.mkdir()
    model_root=repo/'logs/a800/phase6/communication-pilot-h512-tp4-20261003'
    model_info=json.loads((model_root/'submission.json').read_text())
    config=json.loads((model_root/'scripts/config.json').read_text())
    shutil.copy2(repo/'tools/a800/phase6/scenario_operator.py',scripts/'worker.py')
    for name in ('reference.py','routing.py','taco_support.py'):shutil.copy2(model_root/'scripts'/name,scripts/name)
    for policy in [p for p in config['policies'] if p not in ('native','native_taco')]:
        lib=Path(model_info['artifact_root'])/'build'/policy
        env=dict(os.environ,CUDA_HOME='/data/apps/cuda/12.4',CUDA_DEVICE_MAX_CONNECTIONS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 PYTHONNOUSERSITE='1',NCCL_DEBUG='WARN',NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='0',NCCL_P2P_LEVEL='NVL',FLUX_FORCE_NVLINK='1',
                 PYTHONPATH=str(lib/'python'),LD_LIBRARY_PATH=f'{lib}:/data/apps/cuda/12.4/lib64')
        with (operator/f'{policy}.log').open('w') as log:
            subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(scripts/'worker.py'),
                            str(operator/policy),'--policy',policy,'--plan',info['plan']],env=env,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    for hidden in (512,1024,2048):
        path=repo/f'logs/a800/phase6/communication-pilot-h{hidden}-tp4-20261003'
        run_campaign(path)
        rounds=[json.loads((path/f'results/model-{i}/analysis.json').read_text()) for i in (1,2)]
        policies=[r['policy'] for r in rounds[0]['summary'] if r['policy'].endswith('_selective')]
        scores={p:min(next(r for r in a['summary'] if r['policy']==p)[f'latency_reduction_vs_{b}_percent']
                      for a in rounds for b in ('native_taco','taco_fused')) for p in policies}
        state['results'].append(dict(hidden=hidden,path=str(path),minimum_repeated_required_gains=scores,
                                     best_policy=max(scores,key=scores.get),best_score=max(scores.values())))
        save()
    best=max(state['results'],key=lambda r:r['best_score'])
    state['selected_for_confirmation']=best;save()
    out=repo/f"logs/a800/phase6/communication-confirm-h{best['hidden']}-tp4-20261003"
    subprocess.run([sys.executable,str(repo/'tools/a800/phase6/prepare_scenario.py'),str(out),
                    '--build',info['build'],'--plan',info['plan'],'--hidden',str(best['hidden'])],cwd=repo,check=True)
    run_campaign(out)
    with (out/'assessment.log').open('w') as log:
        subprocess.run([sys.executable,str(repo/'tools/a800/phase6/assess.py'),str(out)],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    state.update(status='completed',stage='completed',confirmation=str(out));save()
except BaseException:
    state.update(status='failed',error=traceback.format_exc());save();raise

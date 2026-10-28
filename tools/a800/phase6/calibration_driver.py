"""Run only inside the existing allocation, after prior GPU experiments finish."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
info=json.loads((root/'plan.json').read_text());repo=Path(info['repo'])
assert os.environ.get('SLURM_JOB_ID')==str(info['job_id'])
world=info['world_size']
assert not (root/'execution.json').exists(), 'Use a new calibration directory'
for name,h in json.loads((root/'manifest.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
cuda='/data/apps/cuda/12.4'
base=dict(os.environ,RESULT_DIR=str(root),SLURM_SUBMIT_DIR=str(repo),CUDA_HOME=cuda,
          CUDA_DEVICE_MAX_CONNECTIONS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1',
          NCCL_DEBUG='WARN',NCCL_IB_DISABLE='1',NCCL_P2P_DISABLE='0',NCCL_P2P_LEVEL='NVL',FLUX_FORCE_NVLINK='1')
base.pop('MECH_SMOKE',None)
base.pop('PYTHONOPTIMIZE',None)
(root/'execution.json').write_text(json.dumps(dict(status='started',job_id=info['job_id']))+'\n')
for policy in info['policies']:
    library=Path(info['group_sampler']) if policy=='interleaved_remote_group' else Path(info['sampler']) if policy=='interleaved_remote' else Path(info['sampler_root'])/policy
    checker=library/'check_mapping' if policy.startswith('interleaved_remote') else repo/'outputs/a800/megatron-e2e/scale-mapping'/f'check_mapping-{policy}'
    for m,n,k in info['shapes']:
        with (root/f'mapping-{policy}-{m}-{n}-{k}.csv').open('w') as out:
            subprocess.run([str(checker),str(m),str(n),str(k),str(world),'1' if policy=='remote_first' else '2'],stdout=out,check=True)
    env=dict(base,MECH_POLICY=policy,MECH_KIND='instrumented',MECH_BLOCK='0',MECH_LIBRARY=str(library/'libflux_cuda.so'),
             PYTHONPATH=str(library/'python'),LD_LIBRARY_PATH=f'{library}:{cuda}/lib64')
    with (root/f'{policy}.log').open('w') as log:
        subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(root/'calibration-scripts/worker.py')],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('CALIBRATION COMPLETE',policy,flush=True)
(root/'calibration-complete.json').write_text(json.dumps(dict(completed=True,config=dict(shapes=info['shapes'],world_size=world),
    scope='Fresh per-base BF16 arrival calibration, training passes 0/1 and held-out pass 2; measure perturbation explicitly.'),indent=2)+'\n')
(root/'execution.json').write_text(json.dumps(dict(status='completed',job_id=info['job_id']))+'\n')

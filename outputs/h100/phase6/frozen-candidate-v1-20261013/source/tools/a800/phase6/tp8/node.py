"""One short-lived launcher inside each independently held Slurm allocation."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from protocol import sha, write_json

root, request_path, node = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
request = json.loads(request_path.read_text())
spec = json.loads((root/'submission.json').read_text())
assert int(os.environ['SLURM_JOB_ID']) == spec['jobs'][node]
out = Path(request['out'])
repo = Path(spec['repo'])
build = Path(spec['build'])
policy = request['policy']
cuda = '/data/apps/cuda/12.4'
env = dict(os.environ)
# Remove inherited single-node network constraints. Select observed, active IB HCAs.
for key in ('NCCL_IB_DISABLE', 'NCCL_P2P_LEVEL', 'NCCL_P2P_DISABLE',
            'FLUX_FORCE_NVLINK', 'NCCL_NET', 'NCCL_DEBUG_FILE'):
    env.pop(key, None)
env.update(CUDA_HOME=cuda, PATH=f'{Path(sys.executable).parent}:{cuda}/bin:'+env['PATH'],
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
           PYTHONNOUSERSITE='1', CUDA_DEVICE_MAX_CONNECTIONS='1',
           TORCH_NCCL_ASYNC_ERROR_HANDLING='1', NCCL_DEBUG='INFO',
           NCCL_DEBUG_SUBSYS='INIT,NET', NCCL_SOCKET_IFNAME='=bond0',
           NCCL_IB_HCA='=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1',
           NCCL_IB_DISABLE='0',
           NCCL_DEBUG_FILE=str(out/f'nccl-node{node}-%h-%p.log'),
           E2E_PROCESS_OUT=str(out), E2E_BUILD=str(build), E2E_POLICY=policy,
           E2E_MODE=request['mode'], E2E_CASE=request.get('case', ''),
           E2E_BLOCK=str(request.get('block', -1)), E2E_WINDOW=str(request.get('window', -1)),
           E2E_DIST_OUT=str(Path(request['config']).parent) if request.get('config') else str(root))
env['PYTHONPATH'] = str(repo.parent/'Megatron-LM')
env['LD_LIBRARY_PATH'] = cuda+'/lib64'
if policy not in ('native', 'native_taco', 'probe', 'mapping'):
    env['PYTHONPATH'] += ':'+str(build/policy/'python')
    env['LD_LIBRARY_PATH'] = str(build/policy)+':'+env['LD_LIBRARY_PATH']
meta = dict(host=socket.gethostname(), node_rank=node, job_id=spec['jobs'][node],
            step_id=os.environ.get('SLURM_STEP_ID'), request_sha256=sha(request_path),
            environment={k: v for k, v in env.items() if k.startswith(('NCCL_', 'CUDA_', 'FLUX_'))})
for label, command in [('gpus', ['nvidia-smi', '-L']), ('topology', ['nvidia-smi', 'topo', '-m'])]:
    r = subprocess.run(command, capture_output=True, text=True)
    meta[label] = dict(code=r.returncode, stdout=r.stdout, stderr=r.stderr)
write_json(out/f'node{node}.json', meta)
if policy == 'mapping':
    for mapping in ('original','remote_first','interleaved'):
        for k_global in (1024,4096):
            index = 0 if k_global==1024 else ('original','remote_first','interleaved').index(mapping)
            cmd=[str(build/'mapping'/('check_mapping-'+mapping)), '4096','2048',str(k_global),'4',str(index)]
            with (out/f'mapping-{mapping}-k{k_global}-node{node}.csv').open('w') as output, (out/f'mapping-{mapping}-k{k_global}-node{node}.log').open('w') as errors:
                subprocess.run(cmd,stdout=output,stderr=errors,env=env,check=True)
    (out/f'node{node}-exit-code.txt').write_text('0\n')
    sys.exit(0)
command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=2', '--nproc_per_node=4',
           f'--node_rank={node}', '--master_addr='+spec['master_addr'],
           '--master_port='+str(request['port']), '--max_restarts=0',
           str(root/'scripts'/('probe.py' if policy == 'probe' else 'worker.py')),
           *request.get('args', [])]
with (out/f'node{node}.log').open('w') as stream:
    r = subprocess.run(command, env=env, cwd=repo.parent/'Megatron-LM',
                       stdout=stream, stderr=subprocess.STDOUT)
(out/f'node{node}-exit-code.txt').write_text(str(r.returncode)+'\n')
if r.returncode:
    print((out/f'node{node}.log').read_text()[-12000:], flush=True)
sys.exit(r.returncode)

"""One Slurm task per node; torchrun owns that node's GPU workers."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

out = Path(os.environ['E2E_PROCESS_OUT'])
c = json.loads(Path(os.environ['E2E_DIST_OUT'], 'config.json').read_text())
node = int(os.environ['SLURM_NODEID'])
meta = dict(host=socket.gethostname(), node_rank=node, environment={k:v for k,v in os.environ.items()
            if k.startswith(('NCCL_', 'SLURM_', 'CUDA_VISIBLE', 'MASTER_'))})
for name, cmd in [('topology', ['nvidia-smi','topo','-m']), ('gpus', ['nvidia-smi','-L']),
                  ('network', ['ip','-brief','address']), ('ib', ['ibstat'])]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        meta[name] = dict(code=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except FileNotFoundError:
        meta[name] = 'command unavailable'
(out/f'node{node}.json').write_text(json.dumps(meta, indent=2)+'\n')
args = json.loads(os.environ['E2E_MEGATRON_ARGS'])
cmd = [sys.executable, '-m', 'torch.distributed.run', f'--nnodes={c["nodes"]}',
       f'--nproc_per_node={c["gpus_per_node"]}', f'--node_rank={node}',
       '--master_addr='+os.environ['MASTER_ADDR'], '--master_port='+os.environ['MASTER_PORT'],
       '--max_restarts=0', str(Path(__file__).with_name('worker.py')), *args]
with (out/f'node{node}.log').open('w') as log:
    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
(out/f'node{node}-exit-code.txt').write_text(str(result.returncode)+'\n')
if result.returncode:
    print((out/f'node{node}.log').read_text()[-12000:], flush=True)
sys.exit(result.returncode)

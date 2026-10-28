#!/usr/bin/env bash
#SBATCH --job-name=flux-a800-probe
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=/data/run01/scyb672/hjr/TempName/outputs/a800/gemm_rs_validation/logs/probe-%j.log
set -euo pipefail
cd /data/run01/scyb672/hjr/TempName
hostname
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv
nvidia-smi topo -m
outputs/a800/venv/bin/python - <<'PY'
import torch,json,os
assert torch.cuda.device_count()==2
info={'torch':torch.__version__,'torch_cuda':torch.version.cuda,'nccl':torch.cuda.nccl.version(),'visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpus':[]}
for i in range(2):
    p=torch.cuda.get_device_properties(i)
    info['gpus'].append({'name':p.name,'sm':p.multi_processor_count,'capability':torch.cuda.get_device_capability(i),'memory':p.total_memory})
    assert 'A800' in p.name
    with torch.cuda.device(i):
        x=torch.ones((128,128),device='cuda',dtype=torch.bfloat16)
        assert (x@x == 128).all().item()
info['p2p']=torch.cuda.can_device_access_peer(0,1)
assert info['p2p']
print(json.dumps(info,indent=2))
with open('outputs/a800/gemm_rs_validation/probe.json','w') as f: json.dump(info,f,indent=2)
PY

"""Eight-rank network and subgroup correctness probe, outside training timings."""
import datetime
import json
import os
from pathlib import Path
import socket
import time

rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
allowed = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {allowed[local*len(allowed)//4]})
import torch
import torch.distributed as dist

torch.cuda.set_device(local)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=120))
assert dist.get_world_size() == 8
for ranks in ([0,1,2,3], [4,5,6,7], [0,4], [1,5], [2,6], [3,7]):
    group = dist.new_group(ranks)
    if rank in ranks:
        tensor = torch.full((1024*1024,), rank+1., device='cuda')
        dist.all_reduce(tensor, group=group)
        assert bool((tensor == sum(r+1 for r in ranks)).all())
payload = torch.ones(16*1024*1024, device='cuda')
dist.barrier(); torch.cuda.synchronize()
start = time.perf_counter()
dist.all_reduce(payload)
torch.cuda.synchronize()
elapsed = time.perf_counter()-start
assert bool((payload == 8).all())
out = Path(os.environ['E2E_PROCESS_OUT'])
(out/f'rank{rank}.json').write_text(json.dumps(dict(passed=True, rank=rank, local_rank=local,
    host=socket.gethostname(), gpu_uuid=str(torch.cuda.get_device_properties(local).uuid),
    allreduce_64mib_seconds=elapsed), indent=2)+'\n')
dist.barrier()
dist.destroy_process_group()

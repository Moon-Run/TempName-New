"""Exploratory complete-operator sweep; never used as full-step acceptance."""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'phase5/e2e'),str(HERE.parent/'phase4')]
import torch
import torch.distributed as dist
from flux.gemm_rs_taco import GemmRSTaco
from taco_support import check_taco_output


def main(args):
    rank=int(os.environ['RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=5))
    group=dist.new_group(list(range(4)))
    torch.manual_seed(817+rank)
    a=(torch.randn(2048,2048,device='cuda')*.02).to(torch.bfloat16)
    w=(torch.randn(2048,2048,device='cuda')*.02).to(torch.bfloat16)
    partial=a@w.t()
    results=[]
    for graph in (False,True):
        for path in args.plans:
            plan=json.loads(path.read_text())
            mask=plan['policies'][args.policy]['mask']
            op=GemmRSTaco(group,2048,2048,2048,placement='fused',selected=mask,graph=graph)
            y=op.forward(a,w)
            error=check_taco_output(partial,y,group,op.selection_cpu)
            saved=y.clone()
            for _ in range(10):op.forward(a,w)
            assert torch.equal(y,saved)
            times=[]
            for _ in range(7):
                dist.barrier();torch.cuda.synchronize()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start=time.perf_counter();begin.record()
                for _ in range(50):op.forward(a,w)
                end.record();end.synchronize()
                times.append(dict(wall_us=(time.perf_counter()-start)*1e6/50,gpu_us=begin.elapsed_time(end)*1000/50))
            results.append(dict(graph=graph,plan=str(path.resolve()),fraction=plan['budget_fraction'],
                                selected_per_source=[sum(r) for r in mask],error=error,times=times,
                                median_wall_us=statistics.median(t['wall_us'] for t in times),
                                median_gpu_us=statistics.median(t['gpu_us'] for t in times)))
            if rank==0: print(args.policy,graph,plan['budget_fraction'],results[-1]['median_wall_us'],flush=True)
            dist.barrier();torch.cuda.synchronize()
            del op
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/f'rank{rank}.json').write_text(json.dumps(dict(completed=True,rank=rank,policy=args.policy,results=results),indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--policy',choices=['remote_first','interleaved'],required=True)
    p.add_argument('--plans',type=Path,nargs='+',required=True)
    main(p.parse_args())

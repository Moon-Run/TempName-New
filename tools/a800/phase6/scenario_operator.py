"""Diagnostic shape sweep; full-model paired rounds remain the acceptance test."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist
import flux
from routing import base_policy
from taco_support import check_taco_output


@torch.no_grad()
def main(args):
    rank=int(os.environ['RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[rank*len(cpus)//4]})
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=5))
    pg=dist.new_group(list(range(4)));flux.init_flux_shm(pg)
    plan=json.loads(args.plan.read_text());rows=[]
    args.out.mkdir(parents=True,exist_ok=True)
    for m,n,k in plan['shape_catalog']:
        torch.manual_seed(732+rank)
        a=(torch.randn(m,k//4,device='cuda')*k**-.25).to(torch.bfloat16)
        w=(torch.randn(n,k//4,device='cuda')*k**-.25).to(torch.bfloat16)
        opt=flux.ReduceScatterOption()
        for key,v in dict(use_1d_ring=True,use_p2p_read=True,use_cudaMemcpyAsync=False,use_gemmk=False,
                          per_tile_flags=False,use_barrier_queue=False,num_blocks=6,n_split=1).items():setattr(opt,key,v)
        opt.ring_mode=flux.RingMode.Ring1D
        mask=None
        if args.policy=='original':
            op=flux.GemmRS(pg,1,m,n,torch.bfloat16,torch.bfloat16,transpose_weight=False,fuse_reduction=False,ring_reduction=True)
            forward=lambda:op.forward(a,w,reduce_scatter_option=opt)
        elif args.policy=='taco_fused':
            from flux.gemm_rs_taco import GemmRSTaco
            op=GemmRSTaco(pg,m,n,k//4,placement='fused');forward=lambda:op.forward(a,w)
        else:
            from flux.gemm_rs_taco import GemmRSTacoDoubleBuffered
            mask=next(s for s in plan['policies'][base_policy(args.policy)] if s['shape']==[m,n,k])['mask']
            op=GemmRSTacoDoubleBuffered(pg,m,n,k//4,placement='fused',selected=mask);forward=lambda:op.forward(a,w)
        partial=a@w.t();actual=forward().reshape(m//4,n)
        if args.policy=='original':
            parts=[torch.empty_like(partial) for _ in range(4)];dist.all_gather(parts,partial,group=pg)
            ref=torch.zeros_like(actual)
            for i in range(1,5):ref=(ref+parts[(rank+i)%4][rank*(m//4):(rank+1)*(m//4)]).to(torch.bfloat16)
            torch.testing.assert_close(actual,ref,atol=.02,rtol=.02)
            error=dict(relative_l2=((actual.float()-ref.float()).norm()/ref.float().norm()).item(),passed=True)
            del parts,ref
        else:error=check_taco_output(partial,actual,pg,None if mask is None else op.selection_cpu)
        for _ in range(30):forward()
        samples=[]
        for _ in range(7):
            dist.barrier();torch.cuda.synchronize()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            t=time.perf_counter();start.record()
            for _ in range(50):forward()
            end.record();end.synchronize()
            samples.append(dict(wall_us=(time.perf_counter()-t)*1e6/50,gpu_us=start.elapsed_time(end)*1000/50))
        rows.append(dict(shape=[m,n,k],local_k=k//4,remote_bf16_bytes=2*m*n*3//4,
                         selected_per_source=[sum(r) for r in mask] if mask is not None else None,
                         error=error,samples=samples,median_wall_us=statistics.median(r['wall_us'] for r in samples)))
        (args.out/f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,policy=args.policy,completed=False,rows=rows),indent=2)+'\n')
        if rank==0:print(args.policy,[m,n,k],rows[-1]['median_wall_us'],flush=True)
        dist.barrier();torch.cuda.synchronize();del op
    (args.out/f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,policy=args.policy,completed=True,rows=rows),indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('out',type=Path);p.add_argument('--policy',required=True);p.add_argument('--plan',type=Path,required=True)
    main(p.parse_args())

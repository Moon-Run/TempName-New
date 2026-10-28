"""GPU preflight only; launch explicitly with torchrun in a future allocation.

No Slurm submissions and no performance claims. Requires the isolated TACO build
on PYTHONPATH/LD_LIBRARY_PATH. Saves one report per rank and a separate trace.
"""
import argparse
import datetime
import json
import os
import traceback
from pathlib import Path
import torch
import torch.distributed as dist
import flux
from flux.gemm_rs_taco import GemmRSTaco
from reference import roundtrip


def main(out, model_shapes=False, placement=None):
    world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    assert os.environ.get('SLURM_JOB_ID'), 'Run GPU validation only within an allocated job'
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=5))
    pg=dist.new_group(list(range(world)))
    out.mkdir(parents=True,exist_ok=True)
    report=dict(rank=rank,world_size=world,placement=placement,checks=[],scope='GPU preflight, no performance measurement',
                budgets=dict(relative_l2_vs_bf16=.05,relative_l2_vs_codec_reference=.01),completed=False)
    def save():
        (out/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        shapes=[(128*world,n,256) for n in (128,136,256)]
        if model_shapes:
            assert world==4, 'Frozen model-shape preflight is TP4 only'
            shapes += [(2048,2048,512),(2048,2048,2048)]
        for m,n,k in shapes:
            op=GemmRSTaco(pg,m,n,k,**({'placement':placement} if placement else {}))
            saved=[]
            for case in ('zero','random','spiky','repeat'):
                torch.manual_seed(73+rank+(case=='repeat'))
                a=(torch.randn(m,k,device='cuda')*k**-.25).to(torch.bfloat16)
                b=(torch.randn(n,k,device='cuda')*k**-.25).to(torch.bfloat16)
                if case=='zero':a.zero_()
                if case=='spiky':a[:,0]*=10
                partial=a@b.t()
                contributions=[torch.empty_like(partial) for _ in range(world)]
                dist.all_gather(contributions,partial,group=pg)
                part=m//world
                ref=torch.zeros((part,n),device='cuda',dtype=torch.bfloat16)
                codec_ref=torch.zeros_like(ref)
                saturation=[]
                for i in range(1,world+1):
                    src=(rank+i)%world
                    piece=contributions[src][rank*part:(rank+1)*part]
                    ref=(ref+piece).to(torch.bfloat16)
                    if src!=rank:piece,stats=roundtrip(piece);saturation.append(stats)
                    codec_ref=(codec_ref+piece).to(torch.bfloat16)
                y=op.forward(a,b)
                torch.cuda.synchronize()
                denom=max(ref.float().norm().item(),1e-20)
                delta=(y.float()-ref.float())
                protocol=(y.float()-codec_ref.float()).norm().item()/denom
                relative=delta.norm().item()/denom
                passed=bool(torch.isfinite(y).all()) and relative<=.05 and protocol<=.01
                # Prior outputs must survive packet/workspace reuse.
                for previous,copy in saved:passed &= torch.equal(previous,copy)
                saved.append((y,y.clone()))
                report['checks'].append(dict(M=m,N=n,K_local=k,case=case,passed=passed,
                    max_abs=delta.abs().max().item(),relative_l2=relative,protocol_relative_l2=protocol,
                    wire_bytes=op.wire_bytes_per_rank,bf16_wire_bytes=op.bf16_wire_bytes_per_rank,
                    reference_saturation=saturation))
                save()
            if n==128 or model_shapes and m==2048:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    op.forward(a,b);torch.cuda.synchronize()
                trace=out/f'rank{rank}-m{m}-n{n}-k{k}-profile.json';prof.export_chrome_trace(str(trace))
                kernels=[e['name'] for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel']
                report['kernels']=kernels
                assert sum('taco_decode_ring_kernel' in name for name in kernels)==1
                assert sum('flux_bf16' in name for name in kernels)==1
                if placement is not None:
                    assert sum('taco_encode_scatter_kernel' in name for name in kernels)==int(placement=='separate')
                report.setdefault('kernel_checks',[]).append(dict(M=m,N=n,K_local=k,trace=str(trace),kernels=kernels))
                save()
            dist.barrier(pg);torch.cuda.synchronize()
            del op
        report['completed']=True;report['passed']=all(c['passed'] for c in report['checks'])
        save()
        ok=torch.tensor(int(report['passed']),device='cuda');dist.all_reduce(ok,op=dist.ReduceOp.MIN)
        assert ok.item(), 'TACO preflight failed: retain reports, do not time performance'
    except BaseException:
        report.update(completed=False,error=traceback.format_exc())
        save()
        raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--model-shapes',action='store_true');p.add_argument('--placement',choices=['fused','separate']);a=p.parse_args();main(a.out,a.model_shapes,a.placement)

"""Check the native-Megatron tensor codec, NCCL exchange and STE gather gradient."""
import argparse
import datetime
import json
import os
from pathlib import Path
import traceback
import torch
import torch.distributed as dist
from megatron_taco import TensorTacoRS, QuantizedRS
from taco_support import check_taco_output


def main(out):
    rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==4 and os.environ.get('SLURM_JOB_ID')
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=5))
    group=dist.new_group(list(range(4)))
    report=dict(rank=rank,checks=[],completed=False)
    def save(): (out/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for m,n,k in [(512,128,256),(512,136,256),(512,256,256),(2048,2048,512),(2048,2048,2048)]:
            codec=TensorTacoRS(group,m,n);saved=[]
            for case in ('zero','random','spiky','repeat'):
                torch.manual_seed(73+rank+(case=='repeat'))
                a=(torch.randn(m,k,device='cuda')*k**-.25).to(torch.bfloat16)
                b=(torch.randn(n,k,device='cuda')*k**-.25).to(torch.bfloat16)
                if case=='zero':a.zero_()
                if case=='spiky':a[:,0]*=10
                partial=a@b.t()
                y=codec.reduce(partial)
                error=check_taco_output(partial,y,group)
                for old,copy in saved:assert torch.equal(old,copy)
                saved.append((y,y.clone()))
                report['checks'].append(dict(M=m,N=n,K_local=k,case=case,**error));save()
        partial=partial.detach().requires_grad_(True)
        y=QuantizedRS.apply(partial,codec)
        grad=torch.randn_like(y)
        y.backward(grad)
        gradients=[torch.empty_like(grad) for _ in range(4)]
        dist.all_gather(gradients,grad,group=group)
        assert torch.equal(partial.grad,torch.cat(gradients,dim=0))
        report.update(completed=True,passed=True,ste_gradient_gather_exact=True)
        save()
    except BaseException:
        report.update(passed=False,error=traceback.format_exc());save();raise
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);main(p.parse_args().out)

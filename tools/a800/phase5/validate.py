"""GPU preflight for mixed BF16/FP8 source slots, padding, ordering and lifetimes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import traceback
import torch
import torch.distributed as dist
from flux.gemm_rs_taco import GemmRSTaco
from taco_support import check_taco_output


def main(out, policy, plan_path, graph=False, double_buffered=False, model_shape=None,
         decode_variant=None, compact_boundaries=False):
    assert not (graph and compact_boundaries), 'Boundary A/B requires eager decoder switching'
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert world == 4 and os.environ.get('SLURM_JOB_ID')
    torch.cuda.set_device(rank)
    if decode_variant is not None:
        import ctypes
        import flux
        library = ctypes.CDLL(str((Path(flux.__file__).parent/'lib/libflux_cuda.so').resolve()))
        library.taco_set_decode_variant.argtypes = [ctypes.c_int]
        library.taco_set_decode_variant.restype = ctypes.c_int
        assert library.taco_set_decode_variant(decode_variant) == 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=5))
    pg=dist.new_group(list(range(world)))
    report=dict(rank=rank,world_size=world,policy=policy,checks=[],completed=False)
    if decode_variant is not None:
        report['decoder_variant'] = decode_variant
    out.mkdir(parents=True,exist_ok=True)
    def save(): (out/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    plan=json.loads(plan_path.read_text())
    Codec=GemmRSTaco
    if double_buffered:
        from flux.gemm_rs_taco import GemmRSTacoDoubleBuffered
        Codec=GemmRSTacoDoubleBuffered
    report['skewed_bursts']=[]
    model_shape=tuple(model_shape or (2048,2048,2048))
    try:
        for m,n,k in [(512,128,256),(512,136,256),(512,256,256),(model_shape[0],model_shape[1],model_shape[1]//world),model_shape]:
            tiles=(m//128)*((n+127)//128)
            modes=['all','none','checkerboard']+(['calibrated'] if (m,n,k)==model_shape else [])
            if compact_boundaries and (m,n,k)==model_shape and (m,n)==(8192,2048):
                modes += ['sparse32','sparse33']
            for mode in modes:
                mask=None
                if mode!='all':
                    mask=torch.zeros((world,tiles),dtype=torch.uint8)
                    if mode=='calibrated':
                        entry=next(s for s in plan['policies'][policy] if s['shape']==[m,n,k*world]) if plan.get('schema')==2 else plan['policies'][policy]
                        mask=torch.tensor(entry['mask'],dtype=torch.uint8)
                    if mode=='checkerboard':
                        for src in range(world):
                            for tile in range(tiles):
                                mask[src,tile]=int(tile//(tiles//world)!=src and (tile+src)%2==0)
                    if mode.startswith('sparse'):
                        count=int(mode.removeprefix('sparse'))
                        for dst in range(world):
                            for tile in range(dst*(tiles//world),dst*(tiles//world)+count):
                                mask[(dst+1+tile%3)%world,tile]=1
                                if tile%4==0:mask[(dst+2)%world,tile]=1
                        assert all(sum(bool(mask[:,t].any()) for t in range(dst*(tiles//world),(dst+1)*(tiles//world)))==count for dst in range(world))
                op=Codec(pg,m,n,k,placement='fused',**({'selected':mask} if mask is not None else {}),
                              **({'graph':True} if graph else {}))
                graph_weight=torch.empty((n,k),device='cuda',dtype=torch.bfloat16) if graph else None
                saved=[]
                for case in ('zero','random','spiky','repeat'):
                    torch.manual_seed(73+rank+(case=='repeat'))
                    a=(torch.randn(m,k,device='cuda')*k**-.25).to(torch.bfloat16)
                    b=(torch.randn(n,k,device='cuda')*k**-.25).to(torch.bfloat16)
                    if graph:
                        graph_weight.copy_(b)
                        b=graph_weight
                    if case=='zero':a.zero_()
                    if case=='spiky':a[:,0]*=10
                    partial=a@b.t()
                    legacy=None
                    if mode.startswith('sparse'):
                        import ctypes
                        lib=op.slots[0].lib if double_buffered else op.lib
                        lib.taco_set_decode_variant.argtypes=[ctypes.c_int]
                        lib.taco_set_decode_variant.restype=ctypes.c_int
                        assert lib.taco_set_decode_variant(3)==0
                        legacy=op.forward(a,b)
                        assert lib.taco_set_decode_variant(-1 if decode_variant is None else decode_variant)==0
                    y=op.forward(a,b)
                    torch.cuda.synchronize()
                    if legacy is not None:
                        assert torch.equal(y.view(torch.int16),legacy.view(torch.int16)), 'Compact threshold changed output bits'
                    error=check_taco_output(partial,y,pg,mask)
                    for previous,copy in saved:assert torch.equal(previous,copy)
                    saved.append((y,y.clone()))
                    report['checks'].append(dict(M=m,N=n,K_local=k,selection=mode,case=case,
                        wire_bytes=op.wire_bytes_per_rank,bf16_wire_bytes=op.bf16_wire_bytes_per_rank,
                        **({'legacy_bits_exact':True} if legacy is not None else {}),**error))
                    if mode=='none':assert op.wire_bytes_per_rank==op.bf16_wire_bytes_per_rank
                    save()
                if double_buffered and (mode=='calibrated' or mode.startswith('sparse') or (n==136 and mode=='checkerboard')):
                    # No cross-rank synchronization between these submissions.
                    # Alternate data and stagger ranks across multiple A/B reuses.
                    inputs=[a,(a*.75).to(torch.bfloat16)]
                    expected=[]
                    for x in inputs:
                        expected.append(op.forward(x,b).clone())
                        torch.cuda.synchronize();dist.barrier()
                    burst=[]
                    for i in range(12):
                        if (i+rank)%3==0:torch.cuda._sleep(300000)
                        burst.append(op.forward(inputs[i%2],b))
                    torch.cuda.synchronize()
                    assert all(torch.equal(y,expected[i%2]) for i,y in enumerate(burst)), 'Workspace reuse raced under skew'
                    report['skewed_bursts'].append(dict(M=m,N=n,K_local=k,calls=12,exact=True))
                    save()
                dist.barrier(pg);torch.cuda.synchronize()
                del op
        report.update(completed=True,graph_replay=graph,double_buffered=double_buffered,passed=all(c['passed'] for c in report['checks']))
        save()
    except BaseException:
        report.update(passed=False,error=traceback.format_exc())
        save()
        raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    p.add_argument('--policy',choices=['remote_first','interleaved','interleaved_remote','interleaved_remote_group'],required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--graph',action='store_true')
    p.add_argument('--double-buffered',action='store_true')
    p.add_argument('--model-shape',type=int,nargs=3)
    p.add_argument('--decode-variant',type=int)
    p.add_argument('--compact-boundaries',action='store_true')
    a=p.parse_args();main(a.out,a.policy,a.plan,a.graph,a.double_buffered,a.model_shape,a.decode_variant,a.compact_boundaries)

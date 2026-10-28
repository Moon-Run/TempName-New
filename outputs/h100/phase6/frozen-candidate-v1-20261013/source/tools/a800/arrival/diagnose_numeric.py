"""Read-only numerical diagnosis of existing libraries; never admits performance jobs."""
import json,os,subprocess,sys
from pathlib import Path
if '--launch' in sys.argv:
 root=Path(os.environ['SLURM_SUBMIT_DIR']);out=Path(os.environ['RESULT_DIR'])
 for kind,build in [('baseline','three-way-build'),('instrumented_off','mechanism-build')]:
  for policy in ['original','remote_first','interleaved']:
   path=root/'outputs/a800/phase3'/build/policy
   env=dict(os.environ,DIAG_KIND=kind,DIAG_POLICY=policy,PYTHONPATH=str(path/'python'),LD_LIBRARY_PATH=f'{path}:/data/apps/cuda/12.8/lib64')
   with (out/f'{kind}-{policy}.log').open('w') as f:
    subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',__file__],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 rows=[json.loads(p.read_text()) for p in sorted(out.glob('*-rank*.json'))]
 assert len(rows)==24
 (out/'diagnostic-summary.json').write_text(json.dumps(dict(completed=True,scope='Existing-library numerical diagnosis only, not performance admission',records=rows),indent=2)+'\n')
 print('NUMERICAL DIAGNOSIS COMPLETE',flush=True)
 sys.exit()
import datetime,hashlib
import torch
import torch.distributed as dist
import flux
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
assert int(os.environ['WORLD_SIZE'])==4 and os.environ.get('SLURM_JOB_ID')
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=3));pg=dist.new_group(list(range(4)),backend='nccl');flux.init_flux_shm(pg)
opt=flux.ReduceScatterOption()
for key,value in dict(use_1d_ring=True,use_p2p_read=True,use_cudaMemcpyAsync=False,use_gemmk=False,per_tile_flags=False,use_barrier_queue=False,num_blocks=6,n_split=1).items():setattr(opt,key,value)
opt.ring_mode=flux.RingMode.Ring1D
m,n,k=8192,4096,2048
torch.manual_seed(1700+rank)
a=(torch.randn((m,k//4),device='cuda')*k**-.25).to(torch.bfloat16)
b=(torch.randn((n,k//4),device='cuda')*k**-.25).to(torch.bfloat16)
ref=torch.empty((m//4,n),dtype=torch.bfloat16,device='cuda');fp=torch.empty_like(ref,dtype=torch.float32)
dist.reduce_scatter_tensor(ref,a@b.t(),group=pg);dist.reduce_scatter_tensor(fp,a.float()@b.float().t(),group=pg)
def error(y,r):
 d=(y.float()-r.float()).abs();tol=.02+.02*r.float().abs();mask=d>tol;count=int(mask.sum().item())
 flat=(d/tol).flatten();idx=int(flat.argmax().item())
 return dict(passed=count==0,mismatches=count,max_abs=float(d.max().item()),relative_l2=float((d.norm()/r.float().norm()).item()),max_tolerance_ratio=float(flat[idx].item()),worst_flat_index=idx,actual=float(y.flatten()[idx].item()),reference=float(r.flatten()[idx].item()))
rows=[]
for ring in [False,True]:
 op=flux.GemmRS(pg,1,m,n,torch.bfloat16,torch.bfloat16,transpose_weight=False,fuse_reduction=False,ring_reduction=ring)
 for repeat in range(3):
  y=op.forward(a,b,reduce_scatter_option=opt).reshape_as(ref).clone();torch.cuda.synchronize()
  rows.append(dict(ring_reduction=ring,repeat=repeat,vs_bf16=error(y,ref),vs_fp32=error(y,fp),output_sha256=hashlib.sha256(y.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()))
  op.forward_barrier(a,b);torch.cuda.synchronize();dist.barrier(pg)
 del op
 torch.cuda.synchronize();dist.barrier(pg)
result=dict(rank=rank,kind=os.environ['DIAG_KIND'],policy=os.environ['DIAG_POLICY'],M=m,N=n,K_global=k,seed=1700+rank,atol=.02,rtol=.02,records=rows,completed=True)
(Path(os.environ['RESULT_DIR'])/f"{result['kind']}-{result['policy']}-rank{rank}.json").write_text(json.dumps(result,indent=2)+'\n')
dist.destroy_process_group()

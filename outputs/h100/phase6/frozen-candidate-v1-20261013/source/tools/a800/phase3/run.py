"""One uninstrumented static-policy window. The sbatch balances six policy orders."""
import datetime,json,os,statistics
from pathlib import Path
allowed=sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0,{allowed[int(os.environ['LOCAL_RANK'])*len(allowed)//2]})
import torch
import torch.distributed as dist
import flux
assert os.environ.get('SLURM_JOB_ID')
rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
assert torch.cuda.device_count()==2 and 'A800' in torch.cuda.get_device_name()
assert torch.cuda.get_device_capability()==(8,0) and torch.cuda.can_device_access_peer(0,1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=3))
pg=dist.new_group([0,1],backend='nccl');flux.init_flux_shm(pg)
policy=os.environ['PHASE3_POLICY'];block=int(os.environ['PHASE3_BLOCK']);window=int(os.environ['PHASE3_WINDOW'])
out=Path(os.environ['RESULT_DIR']);prefix=f'window-{window:02d}-{policy}-rank{rank}'
expected=Path(os.environ['PHASE3_LIBRARY']).resolve()
loaded={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if line.endswith('/libflux_cuda.so')}
assert loaded and all(Path(p).resolve()==expected for p in loaded),(loaded,expected)
opt=flux.ReduceScatterOption()
for key,value in dict(use_1d_ring=True,use_p2p_read=True,use_cudaMemcpyAsync=False,use_gemmk=False,
                      per_tile_flags=False,use_barrier_queue=False,num_blocks=6,n_split=1).items():setattr(opt,key,value)
opt.ring_mode=flux.RingMode.Ring1D
m=n=4096;k=8192
op=flux.GemmRS(pg,1,m,n,torch.bfloat16,torch.bfloat16,transpose_weight=False,fuse_reduction=False,ring_reduction=False)
report=dict(job_id=os.environ['SLURM_JOB_ID'],policy=policy,block=block,window=window,rank=rank,
 M=m,N=n,K_global=k,K_local=k//2,world_size=2,flux_module=flux.__file__,loaded_libraries=sorted(loaded),
 torch=torch.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),checks=[],trials_us=[],
 warmup_initial=500,warmup_per_trial=30,iters=100,trial_count=12,sampling=False)
@torch.no_grad()
def run():
 global a,b
 for seed in [17,29,43]:
  torch.manual_seed(seed+1000*rank)
  a=(torch.randn((m,k//2),device='cuda')*k**-0.25).to(torch.bfloat16)
  b=(torch.randn((n,k//2),device='cuda')*k**-0.25).to(torch.bfloat16)
  ref=torch.empty((m//2,n),dtype=torch.bfloat16,device='cuda')
  dist.reduce_scatter_tensor(ref,a@b.t(),group=pg)
  ref32=torch.empty_like(ref,dtype=torch.float32)
  dist.reduce_scatter_tensor(ref32,a.float()@b.float().t(),group=pg)
  actual=op.forward(a,b,reduce_scatter_option=opt).reshape(m//2,n).clone();torch.cuda.synchronize()
  torch.testing.assert_close(actual,ref,atol=.02,rtol=.02)
  torch.testing.assert_close(actual.float(),ref32,atol=.02,rtol=.02)
  delta=actual.float()-ref32
  report['checks'].append(dict(seed=seed,passed=True,max_abs_vs_bf16=(actual.float()-ref.float()).abs().max().item(),
    max_abs_vs_fp32=delta.abs().max().item(),relative_l2_vs_fp32=(delta.norm()/ref32.norm()).item()))
 def fused():return op.forward(a,b,reduce_scatter_option=opt)
 for _ in range(500):fused()
 torch.cuda.synchronize();dist.barrier(pg)
 for trial in range(12):
  for _ in range(30):fused()
  torch.cuda.synchronize();dist.barrier(pg)
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(100):fused()
  end.record();end.synchronize()
  report['trials_us'].append(start.elapsed_time(end)*10)
 # Diagnostics are separate and run after every official timing window is already complete.
 if os.environ.get('PHASE3_DIAGNOSTIC')=='1':
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   dist.barrier(pg);torch.cuda.synchronize()
   for _ in range(10):fused()
   torch.cuda.synchronize()
  prof.export_chrome_trace(str(out/f'{prefix}-profile.json'))
 report['completed']=True
 (out/f'{prefix}.json').write_text(json.dumps(report,indent=2)+'\n')
 if rank==0:print('WINDOW COMPLETE',window,policy,'rank0 median us',statistics.median(report['trials_us']),flush=True)
 dist.barrier(pg)
run()
dist.destroy_process_group()

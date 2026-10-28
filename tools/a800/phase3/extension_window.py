"""Multi-shape extension; each policy process uses the same predeclared inputs and order."""
import datetime,gc,json,os,statistics
from pathlib import Path
world=int(os.environ['WORLD_SIZE'])
allowed=sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0,{allowed[int(os.environ['LOCAL_RANK'])*len(allowed)//world]})
import torch
import torch.distributed as dist
import flux
assert os.environ.get('SLURM_JOB_ID')
rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
assert world in (2,4) and torch.cuda.device_count()==world
assert all('A800' in torch.cuda.get_device_name(i) for i in range(world))
assert torch.cuda.get_device_capability()==(8,0)
assert all(torch.cuda.can_device_access_peer(i,j) for i in range(world) for j in range(world) if i!=j)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=3))
pg=dist.new_group(list(range(world)),backend='nccl');flux.init_flux_shm(pg)
policy=os.environ['PHASE3_POLICY'];block=int(os.environ['PHASE3_BLOCK']);window=int(os.environ['PHASE3_WINDOW'])
root_out=Path(os.environ['RESULT_DIR'])
config=json.loads(Path(__file__).with_name(os.environ.get('PHASE3_CONFIG','extension.json')).read_text())
expected=Path(os.environ['PHASE3_LIBRARY']).resolve()
loaded={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if line.endswith('/libflux_cuda.so')}
assert loaded and all(Path(p).resolve()==expected for p in loaded),(loaded,expected)
expected_ths=expected.with_name('libflux_cuda_ths_op.so')
loaded_ths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if line.endswith('/libflux_cuda_ths_op.so')}
assert loaded_ths and all(Path(p).resolve()==expected_ths for p in loaded_ths),(loaded_ths,expected_ths)
opt=flux.ReduceScatterOption()
for key,value in dict(use_1d_ring=True,use_p2p_read=True,use_cudaMemcpyAsync=False,use_gemmk=False,
                      per_tile_flags=False,use_barrier_queue=False,num_blocks=6,n_split=1).items():setattr(opt,key,value)
opt.ring_mode=flux.RingMode.Ring1D
def run_shape(m,n,k):
 assert m % (128*world)==0 and n % 8==0 and k % (world*32)==0
 out=root_out/f'tp{world}-m{m}-n{n}-k{k}'
 out.mkdir(exist_ok=True)
 prefix=f'window-{window:02d}-{policy}-rank{rank}'
 op=flux.GemmRS(pg,1,m,n,torch.bfloat16,torch.bfloat16,transpose_weight=False,fuse_reduction=False,ring_reduction=False)
 report=dict(job_id=os.environ['SLURM_JOB_ID'],policy=policy,block=block,window=window,rank=rank,
  M=m,N=n,K_global=k,K_local=k//world,world_size=world,flux_module=flux.__file__,loaded_libraries=sorted(loaded | loaded_ths),
  torch=torch.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),checks=[],trials_us=[],
  warmup_initial=config['warmup_initial'],warmup_per_trial=config['warmup_per_trial'],
  iters=config['iters'],trial_count=config['trial_count'],sampling=False,
  timing_protocol="cuda-events-preinitialized-v2",
  diagnostic_only=os.environ.get('PHASE3_DIAGNOSTIC_ONLY')=='1',
  gpu_uuid=str(torch.cuda.get_device_properties(rank).uuid),
  cuda=torch.version.cuda,nccl=torch.cuda.nccl.version())
 @torch.no_grad()
 def run():

  for seed in config['seeds']:
   torch.manual_seed(seed+1000*rank)
   a=(torch.randn((m,k//world),device='cuda')*k**-0.25).to(torch.bfloat16)
   b=(torch.randn((n,k//world),device='cuda')*k**-0.25).to(torch.bfloat16)
   ref=torch.empty((m//world,n),dtype=torch.bfloat16,device='cuda')
   dist.reduce_scatter_tensor(ref,a@b.t(),group=pg)
   ref32=torch.empty_like(ref,dtype=torch.float32)
   dist.reduce_scatter_tensor(ref32,a.float()@b.float().t(),group=pg)
   actual=op.forward(a,b,reduce_scatter_option=opt).reshape(m//world,n).clone();torch.cuda.synchronize()
   torch.testing.assert_close(actual,ref,atol=.02,rtol=.02)
   torch.testing.assert_close(actual.float(),ref32,atol=.02,rtol=.02)
   delta=actual.float()-ref32
   report['checks'].append(dict(seed=seed,passed=True,max_abs_vs_bf16=(actual.float()-ref.float()).abs().max().item(),
     max_abs_vs_fp32=delta.abs().max().item(),relative_l2_vs_fp32=(delta.norm()/ref32.norm()).item()))
  def fused():return op.forward(a,b,reduce_scatter_option=opt)
  for _ in range(config['warmup_initial']):fused()
  torch.cuda.synchronize();dist.barrier(pg)
  if not report['diagnostic_only']:
   # CUDA events are lazy: materialize both outside every measured interval.
   start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   start.record();end.record();end.synchronize()
   for trial in range(config['trial_count']):
    for _ in range(config['warmup_per_trial']):fused()
    torch.cuda.synchronize();dist.barrier(pg);torch.cuda.synchronize()
    start.record()
    for _ in range(config['iters']):fused()
    end.record();end.synchronize()
    report['trials_us'].append(start.elapsed_time(end)*1000/config['iters'])
  # Diagnostics are separate and run after every official timing window is already complete.
  if os.environ.get('PHASE3_DIAGNOSTIC')=='1':
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    dist.barrier(pg);torch.cuda.synchronize()
    for _ in range(10):fused()
    torch.cuda.synchronize()
   prof.export_chrome_trace(str(out/f'{prefix}-profile.json'))
  report['completed']=True
  (out/f'{prefix}.json').write_text(json.dumps(report,indent=2)+'\n')
  if rank==0:print('WINDOW COMPLETE',window,policy,'rank0 median us',statistics.median(report['trials_us']) if report['trials_us'] else 'diagnostic',flush=True)
  dist.barrier(pg)
 run()


shapes=config['shapes']
shift=block % len(shapes)
for shape in shapes[shift:]+shapes[:shift]:
 run_shape(*shape)
 gc.collect()
 torch.cuda.empty_cache()
 dist.barrier(pg)
dist.destroy_process_group()

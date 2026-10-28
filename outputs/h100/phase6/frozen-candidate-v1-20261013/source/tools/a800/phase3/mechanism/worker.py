"""Four-source fragment observations. All event differences use one GPU's clock."""
import ctypes as C
import datetime,json,os,random,statistics
from pathlib import Path
world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[local*len(cpus)//world]})
import torch
import torch.distributed as dist
import flux
assert os.environ.get('SLURM_JOB_ID') and world==4
assert torch.cuda.device_count()==world
torch.cuda.set_device(local)
assert all('A800' in torch.cuda.get_device_name(i) for i in range(world))
assert all(torch.cuda.can_device_access_peer(i,j) for i in range(world) for j in range(world) if i!=j)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=3))
pg=dist.new_group(list(range(world)),backend='nccl');flux.init_flux_shm(pg)
config=json.loads(Path(__file__).with_name('config.json').read_text())
smoke=os.environ.get('MECH_SMOKE')=='1';repeats=2 if smoke else config['repeats']
sample_warmup=config.get('sample_warmup',10)
policy=os.environ['MECH_POLICY'];kind=os.environ['MECH_KIND'];block=int(os.environ['MECH_BLOCK'])
instrumented=kind=='instrumented';out=Path(os.environ['RESULT_DIR']);prefix=f'b{block}-{policy}-{kind}'
library=Path(os.environ['MECH_LIBRARY']).resolve()
for name in ['libflux_cuda.so','libflux_cuda_ths_op.so']:
 loaded={Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines() if line.endswith('/'+name)}
 assert loaded=={library.with_name(name)},(loaded,library)
report=dict(job_id=os.environ['SLURM_JOB_ID'],policy=policy,kind=kind,block=block,rank=rank,world_size=world,
            gpu_uuid=str(torch.cuda.get_device_properties(local).uuid),library=str(library),smoke=smoke,sample_warmup=sample_warmup,shapes=[])
ROWS,FRAGS,FIELDS=2048,16,8;SLOTS=ROWS*FRAGS
opt=flux.ReduceScatterOption()
for k,v in dict(use_1d_ring=True,use_p2p_read=True,use_cudaMemcpyAsync=False,use_gemmk=False,
               per_tile_flags=False,use_barrier_queue=False,num_blocks=6,n_split=1).items():setattr(opt,k,v)
opt.ring_mode=flux.RingMode.Ring1D

def check(code):assert code==0,code

def sync():
 torch.cuda.synchronize();dist.barrier(pg);torch.cuda.synchronize()

def save(): (out/f'{prefix}-rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')

if instrumented:
 lib=C.CDLL(str(library))
 lib.mech_configure.argtypes=[C.c_void_p,C.POINTER(C.c_void_p)]+[C.c_int]*5
 lib.mech_configure.restype=C.c_int
 lib.mech_observe.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
 lib.mech_timer_check.argtypes=[C.c_void_p,C.c_void_p]
 lib.mech_mark.argtypes=[C.c_void_p,C.c_int,C.c_void_p]
 producer=torch.zeros((ROWS,FRAGS,FIELDS),dtype=torch.int64,device='cuda')
 peers=flux.create_tensor_list([world,SLOTS],torch.int32,pg,False,True)
 flags=peers[rank];flag_ptrs=(C.c_void_p*world)(*[p.data_ptr() for p in peers])
 seen=torch.zeros((world,SLOTS),dtype=torch.int64,device='cuda')
 status=torch.zeros(132,dtype=torch.int64,device='cuda');markers=torch.zeros(2,dtype=torch.int64,device='cuda')
 observer=torch.cuda.Stream(priority=0)
 def configure(mode=0,stride=0,offset=0,epoch=1):
  check(lib.mech_configure(producer.data_ptr(),flag_ptrs,world,stride,offset,epoch,mode))
 configure()
 timer=torch.zeros(2,dtype=torch.int64,device='cuda')
 for _ in range(5):check(lib.mech_timer_check(timer.data_ptr(),torch.cuda.current_stream().cuda_stream))
 torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 a.record();check(lib.mech_timer_check(timer.data_ptr(),torch.cuda.current_stream().cuda_stream));b.record();b.synchronize()
 stamps=timer.cpu().tolist();ratio=(stamps[1]-stamps[0])/(a.elapsed_time(b)*1e6)
 assert .8<ratio<1.05,ratio
 report['clock_ratio']=ratio
 # A first marker launch can lazily load its kernel and synchronize the context.
 # Load it before any observer can wait for work on the producer stream.
 for index in (0,1):check(lib.mech_mark(markers.data_ptr(),index,torch.cuda.current_stream().cuda_stream))
 torch.cuda.synchronize()
 report['marker_prewarmed']=True

@torch.no_grad()
def run_case(m,n,k):
 torch.manual_seed(1700+rank)
 a=(torch.randn((m,k//world),device='cuda')*k**-.25).to(torch.bfloat16)
 b=(torch.randn((n,k//world),device='cuda')*k**-.25).to(torch.bfloat16)
 ref=torch.empty((m//world,n),dtype=torch.bfloat16,device='cuda')
 ref32=torch.empty_like(ref,dtype=torch.float32)
 dist.reduce_scatter_tensor(ref,a@b.t(),group=pg)
 dist.reduce_scatter_tensor(ref32,a.float()@b.float().t(),group=pg)
 op=flux.GemmRS(pg,1,m,n,torch.bfloat16,torch.bfloat16,transpose_weight=False,fuse_reduction=False,ring_reduction=False)
 def forward():return op.forward(a,b,reduce_scatter_option=opt).reshape(m//world,n)
 def correct(y):
  torch.testing.assert_close(y,ref,rtol=.02,atol=.02)
  torch.testing.assert_close(y.float(),ref32,rtol=.02,atol=.02)
 if instrumented:configure()
 correct(forward())
 for _ in range(50 if smoke else config['warmup']):forward()
 row=dict(M=m,N=n,K_global=k,correctness=True,batch_us=[],samples=[])
 for _ in range(2 if smoke else config['trials']):
  for _ in range(30):forward()
  sync();start,stop=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(config['iters']):forward()
  stop.record();stop.synchronize();row['batch_us'].append(start.elapsed_time(stop)*1000/config['iters'])
 if instrumented:
  tm,tn=m//128,(n+127)//128;tiles=tm*tn;per=tiles//world
  assert tiles<=ROWS and per%4==0
  producer.zero_();configure(1,per,0);forward();torch.cuda.synchronize()
  ids=list(range(0,tiles,per));raw=producer[ids].cpu();fragments=int(raw[0,0,4])
  assert fragments==8 and torch.count_nonzero(producer[:,:,7]).item()==len(ids)*fragments
  assert all(raw[i,f,2:5].tolist()==[128,128,fragments] and raw[i,f,7].item()==1 for i in range(len(ids)) for f in range(fragments))
  row['fragment_count']=fragments;row['tiles']=tiles;row['partition_tiles']=per
  configure()
  modes=[('off',0,False),('producer_sparse',1,False),('publication_sparse',2,False),('receiver_sparse',2,True),('receiver_dense',2,True)]
  # Identical physical output coordinates across all policies. The final offset includes the N edge.
  offsets=[0,per//4,per//2,3*per//4,per-1]
  # Materialize timing events before launching a polling kernel. Reuse them only
  # after done.synchronize(), so no event allocation is needed in the overlap.
  start,stop,done=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
  for event in (start,stop,done):event.record()
  done.synchronize()
  for repeat in range(repeats):
   order=list(modes);random.Random(100+block*1000+repeat).shuffle(order)
   for label,mode,observe in order:
    # Correctness checks touch large reference tensors. Restore the same warmed
    # operator state before every condition instead of timing that cold aftermath.
    configure()
    for _ in range(sample_warmup):forward()
    stride=0 if not mode else per//4 if label=='receiver_dense' else per
    offset=offsets[repeat%len(offsets)]%stride if stride else 0
    selected=list(range(offset,tiles,stride)) if stride else []
    owned=[t for t in selected if t//per==rank]
    slots=[src*SLOTS+t*FRAGS+f for src in range(world) for t in owned for f in range(fragments)]
    indices=torch.tensor(slots,dtype=torch.int64,device='cuda')
    producer.zero_();flags.zero_();seen.zero_();status.zero_();markers.zero_()
    epoch=repeat+2;configure(mode,stride,offset,epoch);sync()
    if observe:
     assert slots
     observer.wait_stream(torch.cuda.current_stream())
     check(lib.mech_observe(flags.data_ptr(),indices.data_ptr(),len(slots),seen.data_ptr(),status.data_ptr(),epoch,observer.cuda_stream))
    stream=torch.cuda.current_stream().cuda_stream
    check(lib.mech_mark(markers.data_ptr(),0,stream))
    start.record();y=forward();check(lib.mech_mark(markers.data_ptr(),1,stream));stop.record()
    if observe:torch.cuda.current_stream().wait_stream(observer)
    done.record();done.synchronize();correct(y)
    sample=dict(repeat=repeat,mode=label,stride=stride,offset=offset,selected_tiles=selected,
                forward_us=start.elapsed_time(stop)*1000,envelope_us=start.elapsed_time(done)*1000,
                local_markers=markers.cpu().tolist(),correctness=True)
    if mode:
     raw=producer[selected].cpu()
     assert torch.count_nonzero(producer[:,:,7]).item()==len(selected)*fragments,(label,'missing/extra records')
     assert all(raw[i,f,7].item()==1 for i in range(len(selected)) for f in range(fragments)),(label,'duplicate fragments')
     assert all(raw[i,f,5].item()==tile//per for i,tile in enumerate(selected) for f in range(fragments)),(label,'destination mismatch')
     sample['producer']=[dict(tile=t,fragment=f,values=raw[i,f].tolist()) for i,t in enumerate(selected) for f in range(fragments)]
    if observe:
     st=status.cpu().tolist()
     if st[2]!=0 or st[3]!=len(slots):
      (out/f'{prefix}-rank{rank}-observer-failure.json').write_text(json.dumps(dict(
       M=m,N=n,K_global=k,repeat=repeat,epoch=epoch,mode=label,selected_tiles=selected,
       observer_status=st,expected=len(slots),local_markers=sample['local_markers'],
       forward_us=sample['forward_us'],envelope_us=sample['envelope_us']),indent=2)+'\n')
     assert st[2]==0 and st[3]==len(slots),(label,st[:4],len(slots))
     values=seen.reshape(-1)[indices].cpu().tolist();assert all(v!=0 for v in values)
     sample.update(observer_status=st[:4],max_poll_cycle_ns=max(st[4:]),
                   receiver=[dict(source=s//SLOTS,tile=s%SLOTS//FRAGS,fragment=s%FRAGS,observed_ns=v) for s,v in zip(slots,values)])
    row['samples'].append(sample)
   if repeat%5==0 and rank==0:print('SAMPLE',policy,block,m,n,k,repeat,flush=True)
  configure()
 # Actual hparams/grid/resources, separate from timing and sampled observations.
 sync()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  forward();torch.cuda.synchronize()
 trace=out/f'{prefix}-m{m}-n{n}-k{k}-rank{rank}-profile.json';prof.export_chrome_trace(str(trace))
 kernels=[e for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel' and 'flux_bf16' in e['name']]
 assert len(kernels)==1
 row['kernel']=dict(name=kernels[0]['name'],duration_us=kernels[0]['dur'],args=kernels[0]['args'])
 report['shapes'].append(row);save();sync()
 if rank==0:print('CASE COMPLETE',prefix,m,n,k,flush=True)

try:
 for shape in config['shapes']:run_case(*shape)
 report['completed']=True;save();sync()
finally:dist.destroy_process_group()

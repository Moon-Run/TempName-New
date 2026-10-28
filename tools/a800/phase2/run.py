"""Phase2 diagnostic sampler, fixed BF16/SM80/NVLink; launched by Slurm only."""
import ctypes as C
import datetime,json,os,random,statistics
from pathlib import Path
# Pin each launcher to its half of the allocated CPU set; keep GPU visibility unchanged.
_allowed_cpus=sorted(os.sched_getaffinity(0))
_cpu=_allowed_cpus[int(os.environ['LOCAL_RANK'])*len(_allowed_cpus)//2]
os.sched_setaffinity(0,{_cpu})
import torch
import torch.distributed as dist
import flux

assert os.environ.get('SLURM_JOB_ID')
rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
assert torch.cuda.device_count()==2 and 'A800' in torch.cuda.get_device_name()
assert torch.cuda.can_device_access_peer(0,1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=3))
pg=dist.new_group([0,1],backend='nccl');flux.init_flux_shm(pg)
kind=os.environ['PHASE2_KIND'];instrumented=kind=='instrumented'
out=Path(os.environ['RESULT_DIR']);ROWS,SLOTS,FIELDS=2048,32768,8
opt=flux.ReduceScatterOption()
for k,v in dict(use_1d_ring=True,use_p2p_read=True,use_cudaMemcpyAsync=False,use_gemmk=False,
               per_tile_flags=False,use_barrier_queue=False,num_blocks=6,n_split=1).items():setattr(opt,k,v)
opt.ring_mode=flux.RingMode.Ring1D
report=dict(kind=kind,rank=rank,job_id=os.environ['SLURM_JOB_ID'],flux_module=flux.__file__,
            torch=torch.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),protocol='epilogue-callback publication v2',shapes=[])
def gather(x):
    values=[None,None];dist.all_gather_object(values,x,group=pg);return values

def save():
    (out/f'{kind}-rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')

def chk(code):
    assert code==0,f'CUDA launcher error {code}'

if instrumented:
    lib=C.CDLL(os.environ['PHASE2_LIBRARY'])
    lib.phase2_configure.argtypes=[C.c_void_p]*3+[C.c_int]*4
    lib.phase2_configure.restype=None
    lib.phase2_observe.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
    lib.phase2_timer_check.argtypes=[C.c_void_p,C.c_void_p]
    lib.phase2_mark.argtypes=[C.c_void_p,C.c_int,C.c_void_p]
    producer=torch.zeros((ROWS,16,FIELDS),dtype=torch.int64,device='cuda')
    peer_flags=flux.create_tensor_list([2,SLOTS],torch.int32,pg,False,True)
    flags=peer_flags[rank]
    seen=torch.zeros((2,SLOTS),dtype=torch.int64,device='cuda')
    status=torch.zeros(132,dtype=torch.int64,device='cuda')
    markers=torch.zeros(2,dtype=torch.int64,device='cuda')
    observer=torch.cuda.Stream(priority=0)
    clock=torch.zeros(2,dtype=torch.int64,device='cuda')
    # Warm lazy CUDA module loading before comparing device clocks with CUDA events.
    for _ in range(5):chk(lib.phase2_timer_check(clock.data_ptr(),torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    e0.record();chk(lib.phase2_timer_check(clock.data_ptr(),torch.cuda.current_stream().cuda_stream));e1.record();e1.synchronize()
    host=clock.cpu().tolist();ns=host[1]-host[0];elapsed=e0.elapsed_time(e1)*1e6
    assert 0.8 < ns/elapsed < 1.05,(ns,elapsed)
    report['clock_check']=dict(globaltimer_delta_ns=ns,cuda_event_ns=elapsed,ratio=ns/elapsed)
    def configure(mode=0,stride=0,offset=0,epoch=1):
        lib.phase2_configure(producer.data_ptr(),peer_flags[0].data_ptr(),peer_flags[1].data_ptr(),stride,offset,epoch,mode)
    configure()
    dist.barrier(pg)

@torch.no_grad()
def batch_time(fn):
    for _ in range(30):fn()
    torch.cuda.synchronize();dist.barrier(pg)
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(100):fn()
    b.record();b.synchronize()
    return a.elapsed_time(b)*10  # us per invocation

@torch.no_grad()
def case(k):
    m=n=4096
    torch.manual_seed(1700+rank)
    a=(torch.randn((m,k//2),device='cuda')*k**-0.25).to(torch.bfloat16)
    b=(torch.randn((n,k//2),device='cuda')*k**-0.25).to(torch.bfloat16)
    ref=torch.empty((m//2,n),device='cuda',dtype=torch.bfloat16)
    dist.reduce_scatter_tensor(ref,a@b.t(),group=pg)
    ref32=torch.empty((m//2,n),device='cuda',dtype=torch.float32)
    dist.reduce_scatter_tensor(ref32,a.float()@b.float().t(),group=pg)
    op=flux.GemmRS(pg,1,m,n,torch.bfloat16,torch.bfloat16,transpose_weight=False,fuse_reduction=False,ring_reduction=False)
    def fused():return op.forward(a,b,reduce_scatter_option=opt).reshape(m//2,n)
    if instrumented:configure()
    y=fused().clone();torch.testing.assert_close(y,ref,rtol=.02,atol=.02)
    torch.testing.assert_close(y.float(),ref32,rtol=.02,atol=.02)
    row=dict(M=m,N=n,K_global=k,correctness=True,batch_us=[batch_time(fused) for _ in range(10)])
    # Record actual kernel identity/hparams, separately from all headline timings.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        dist.barrier(pg);torch.cuda.synchronize();fused();torch.cuda.synchronize()
    trace=out/f'{kind}-k{k}-rank{rank}-kernel.json';prof.export_chrome_trace(str(trace))
    row['kernels']=[e['name'] for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel' and 'flux_bf16' in e['name']]
    assert len(row['kernels'])==1
    if instrumented:
        producer.zero_();configure(1,256,0,1);fused();torch.cuda.synchronize()
        raw=producer.cpu();locations=(raw[:,:,7]>0).nonzero()
        assert len(locations)>0,'No sampling records'
        records=[raw[t,f].tolist() for t,f in locations.tolist()]
        km,kn,fragments=records[0][2:5]
        assert 0<fragments<=16 and all(r[2:5]==[km,kn,fragments] and r[7]==1 for r in records)
        total_tiles=(m//km)*(n//kn);assert total_tiles<=ROWS
        selected=list(range(0,total_tiles,256))
        assert {(t,f) for t,f in locations.tolist()}=={(t,f) for t in selected for f in range(fragments)}
        row['tile_shape']=[km,kn];row['fragment_count']=fragments
        configure()
        for _ in range(60):fused()
        row['samples']=[]
        configs=[('off',0,0,False),('producer_sparse',1,256,False),
                 ('publication_sparse',2,256,False),('receiver_sparse',2,256,True),
                 ('receiver_dense',2,64,True)]
        for repeat in range(20):
            choices=list(configs);random.Random(100+repeat).shuffle(choices)
            for label,mode,stride,observe in choices:
                offset=(repeat%4)*(stride//4) if stride else 0
                selected=[t for t in range(total_tiles) if stride and t%stride==offset]
                owned=[t for t in selected if (t//(n//kn)*km)//(m//2)==rank]
                slots=[src*SLOTS+t*16+f for src in range(2) for t in owned for f in range(fragments)]
                indices=torch.tensor(slots,dtype=torch.int64,device='cuda')
                producer.zero_();flags.zero_();seen.zero_();status.zero_();markers.zero_()
                configure(mode,stride,offset,repeat+2)
                torch.cuda.synchronize();dist.barrier(pg);torch.cuda.synchronize()
                if observe:
                    assert slots
                    observer.wait_stream(torch.cuda.current_stream())
                    chk(lib.phase2_observe(flags.data_ptr(),indices.data_ptr(),len(slots),seen.data_ptr(),status.data_ptr(),repeat+2,observer.cuda_stream))
                chk(lib.phase2_mark(markers.data_ptr(),0,torch.cuda.current_stream().cuda_stream))
                start,stop,all_done=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
                start.record();y=fused();stop.record()
                if observe:torch.cuda.current_stream().wait_stream(observer)
                all_done.record();all_done.synchronize()
                forward_us=start.elapsed_time(stop)*1000;envelope_us=start.elapsed_time(all_done)*1000
                chk(lib.phase2_mark(markers.data_ptr(),1,torch.cuda.current_stream().cuda_stream));torch.cuda.synchronize()
                torch.testing.assert_close(y,ref,rtol=.02,atol=.02)
                torch.testing.assert_close(y.float(),ref32,rtol=.02,atol=.02)
                sample=dict(repeat=repeat,mode=label,stride=stride,offset=offset,forward_us=forward_us,
                            envelope_us=envelope_us,correctness=True,local_markers=markers.cpu().tolist())
                if mode:
                    raw=producer.cpu();locations=(raw[:,:,7]>0).nonzero().tolist()
                    expected={(t,f) for t in selected for f in range(fragments)}
                    assert set(map(tuple,locations))==expected,(label,'missing/extra fragments')
                    assert all(raw[t,f,7].item()==1 for t,f in locations),'Duplicate fragment publication'
                    sample['producer']=[dict(tile=t,fragment=f,values=raw[t,f].tolist()) for t,f in locations]
                if observe:
                    st=status.cpu().tolist();assert st[2]==0 and st[3]==len(slots),(label,st[:4],len(slots))
                    recv=seen.cpu().reshape(-1)
                    assert all(recv[s].item()!=0 for s in slots)
                    sample['receiver']=[dict(source=s//SLOTS,tile=(s%SLOTS)//16,fragment=s%16,observed_ns=recv[s].item()) for s in slots]
                    sample['observer_status']=st[:4];sample['max_poll_cycle_ns']=max(st[4:])
                row['samples'].append(sample)
            if repeat%5==0 and rank==0:print('PROGRESS',k,repeat,flush=True)
        configure()
    report['shapes'].append(row);save()
    dist.barrier(pg);del op;torch.cuda.synchronize()
    if rank==0:print('CASE COMPLETE',kind,k,flush=True)

for k in [2048,8192]:case(k)
report['completed']=True;save()
dist.barrier(pg);dist.destroy_process_group()

"""Baseline repetition and separate CUDA profiling; run via phase1.sbatch."""
import datetime
import json
import os
from pathlib import Path
import random
import socket
import statistics
import torch
import torch.distributed as dist
import flux

assert os.environ.get('SLURM_JOB_ID'), 'Use a Slurm GPU job'
rank = int(os.environ['RANK'])
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
assert torch.cuda.device_count() == 2
assert 'A800' in torch.cuda.get_device_name()
assert torch.cuda.get_device_capability() == (8, 0)
assert torch.cuda.can_device_access_peer(0, 1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=5))
assert dist.get_world_size() == 2
pg = dist.new_group([0, 1], backend='nccl')
flux.init_flux_shm(pg)
opt = flux.ReduceScatterOption()
for name, value in dict(use_1d_ring=True, use_p2p_read=True, use_cudaMemcpyAsync=False,
                        use_gemmk=False, per_tile_flags=False, use_barrier_queue=False,
                        num_blocks=6, n_split=1).items():
    setattr(opt, name, value)
opt.ring_mode = flux.RingMode.Ring1D
out = Path(os.environ['RESULT_DIR'])
report = dict(job_id=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
              gpu=torch.cuda.get_device_name(), world_size=2, torch=torch.__version__,
              torch_cuda=torch.version.cuda, torch_nccl=torch.cuda.nccl.version(),
              flux_module=flux.__file__, path='test1 BF16/RCR/SM80/NVLink/default swizzle',
              warmup=20, iters=100, trials=12, seeds=[17,29,43], atol=0.02, rtol=0.02,
              statistic='per trial max(rank mean CUDA-event latency); median over trials',
              shapes=[], profiles=[])

def gather(value):
    values = [None, None]
    dist.all_gather_object(values, value, group=pg)
    return values

def save():
    if rank == 0:
        (out/'baseline.json').write_text(json.dumps(report, indent=2)+'\n')

def stats(values):
    q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
    return dict(median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values),
                mean_ms=statistics.mean(values), iqr_ms=q3-q1,
                cv_percent=100*statistics.stdev(values)/statistics.mean(values))

@torch.no_grad()
def timed(fn, iters=100, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    dist.barrier(pg)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn()
    end.record(); end.synchronize()
    return gather(start.elapsed_time(end)/iters)

@torch.no_grad()
def make_case(m,n,k):
    op = flux.GemmRS(pg,1,m,n,torch.bfloat16,torch.bfloat16,
                     transpose_weight=False,fuse_reduction=False,ring_reduction=False)
    full = torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
    reduced = torch.empty((m//2,n),device='cuda',dtype=torch.bfloat16)
    a = torch.empty((m,k//2),device='cuda',dtype=torch.bfloat16)
    b = torch.empty((n,k//2),device='cuda',dtype=torch.bfloat16)
    def baseline():
        torch.mm(a,b.t(),out=full)
        dist.reduce_scatter_tensor(reduced,full,group=pg)
        return reduced
    def fused(): return op.forward(a,b,reduce_scatter_option=opt).reshape(m//2,n)
    checks=[]
    for seed in report['seeds']:
        torch.manual_seed(seed+1000*rank)
        a.copy_(torch.randn(a.shape,device='cuda')*k**-0.25)
        b.copy_(torch.randn(b.shape,device='cuda')*k**-0.25)
        ref_bf16=baseline().clone()
        partial=a.float()@b.float().t()
        ref_fp32=torch.empty((m//2,n),device='cuda',dtype=torch.float32)
        dist.reduce_scatter_tensor(ref_fp32,partial,group=pg)
        actual=fused().clone(); torch.cuda.synchronize()
        torch.testing.assert_close(actual,ref_bf16,rtol=0.02,atol=0.02)
        torch.testing.assert_close(actual.float(),ref_fp32,rtol=0.02,atol=0.02)
        delta=actual.float()-ref_fp32
        checks.append(dict(seed=seed,ranks=gather(dict(rank=rank,passed=True,
            max_abs_vs_bf16=(actual.float()-ref_bf16.float()).abs().max().item(),
            max_abs_vs_fp32=delta.abs().max().item(),
            relative_l2_vs_fp32=(delta.norm()/ref_fp32.norm()).item()))))
    return op,baseline,fused,checks

@torch.no_grad()
def baseline_case(m,n,k):
    op,baseline,fused,checks=make_case(m,n,k)
    samples=[]
    for trial in range(report['trials']):
        methods=[('flux',fused),('torch',baseline)]
        if trial%2: methods.reverse()
        row=dict(trial=trial,order=[name for name,_ in methods])
        for name,fn in methods: row[name]=timed(fn)
        samples.append(row)
    fs=stats([max(r['flux']) for r in samples]); ts=stats([max(r['torch']) for r in samples])
    record=dict(M=m,N=n,K_global=k,K_local=k//2,checks=checks,trials=samples,
                flux=fs,torch=ts,speedup=ts['median_ms']/fs['median_ms'])
    report['shapes'].append(record); save()
    if rank==0: print('BASELINE',json.dumps({k:v for k,v in record.items() if k not in ['checks','trials']}),flush=True)
    dist.barrier(pg); del op,fused,baseline; torch.cuda.synchronize()

@torch.no_grad()
def profile_case(m,n,k):
    op,baseline,fused,_=make_case(m,n,k)
    # Same iteration count before/with/after profiler; only diagnostics, never the headline latency.
    before=timed(fused,iters=20)
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        dist.barrier(pg); torch.cuda.synchronize()
        start.record()
        for _ in range(20): fused()
        end.record(); end.synchronize()
    instrumented=gather(start.elapsed_time(end)/20)
    trace=f'profile-m{m}-n{n}-k{k}-rank{rank}.json'
    prof.export_chrome_trace(str(out/trace))
    after=timed(fused,iters=20)
    report['profiles'].append(dict(M=m,N=n,K_global=k,iters=20,
        unprofiled_before_ms=before,profiled_ms=instrumented,unprofiled_after_ms=after,
        trace_pattern=f'profile-m{m}-n{n}-k{k}-rank*.json'))
    save()
    dist.barrier(pg); del op,fused,baseline; torch.cuda.synchronize()

shapes=[(m,4096,k) for m in [1024,4096,8192] for k in [2048,8192]]
random.Random(20261001).shuffle(shapes)
report['shape_order']=shapes
for shape in shapes: baseline_case(*shape)
for k in [2048,8192]: profile_case(4096,4096,k)
report['completed']=True; save()
if rank==0: print('PASS: phase1 baseline correctness, repetitions and CUDA traces complete',flush=True)
dist.barrier(pg); dist.destroy_process_group()

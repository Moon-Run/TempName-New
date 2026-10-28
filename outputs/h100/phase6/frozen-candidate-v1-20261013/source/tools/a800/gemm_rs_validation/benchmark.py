"""Fixed A800/TP2/BF16/RCR/NVLink GEMM-RS validation; run only under Slurm."""
import datetime
import json
import os
from pathlib import Path
import statistics
import socket
import torch
import torch.distributed as dist
import flux

assert os.environ.get('SLURM_JOB_ID'), 'Run on a Slurm GPU allocation'
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
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
opt.use_1d_ring = True
opt.use_p2p_read = True
opt.use_cudaMemcpyAsync = False
opt.use_gemmk = False
opt.per_tile_flags = False
opt.use_barrier_queue = False
opt.num_blocks = 6
opt.n_split = 1
opt.ring_mode = flux.RingMode.Ring1D
outdir = Path(os.environ['RESULT_DIR'])
results = []

def gather(obj):
    values = [None, None]
    dist.all_gather_object(values, obj, group=pg)
    return values

@torch.no_grad()
def timed(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(pg)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return gather(start.elapsed_time(end) / iters)

@torch.no_grad()
def run_shape(m, n, k):
    local_k = k // 2
    op = flux.GemmRS(pg, 1, m, n, torch.bfloat16, torch.bfloat16,
                     transpose_weight=False, fuse_reduction=False, ring_reduction=False)
    full = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    baseline_out = torch.empty((m // 2, n), device='cuda', dtype=torch.bfloat16)
    checks = []
    for seed in [17, 29, 43]:
        torch.manual_seed(seed + rank * 1000)
        a = (torch.randn((m, local_k), device='cuda') * k ** -0.25).to(torch.bfloat16)
        b = (torch.randn((n, local_k), device='cuda') * k ** -0.25).to(torch.bfloat16)
        def baseline():
            torch.mm(a, b.t(), out=full)
            dist.reduce_scatter_tensor(baseline_out, full, group=pg)
            return baseline_out
        def fused():
            return op.forward(a, b, reduce_scatter_option=opt).reshape(m // 2, n)
        reference_bf16 = baseline().clone()
        partial_fp32 = a.float() @ b.float().t()
        reference_fp32 = torch.empty((m // 2, n), device='cuda', dtype=torch.float32)
        dist.reduce_scatter_tensor(reference_fp32, partial_fp32, group=pg)
        actual = fused().clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, reference_bf16, rtol=0.02, atol=0.02)
        torch.testing.assert_close(actual.float(), reference_fp32, rtol=0.02, atol=0.02)
        delta = actual.float() - reference_fp32
        checks.append({'seed':seed, 'ranks':gather({
            'rank':rank, 'passed':True,
            'max_abs_vs_bf16':(actual.float()-reference_bf16.float()).abs().max().item(),
            'max_abs_vs_fp32':delta.abs().max().item(),
            'relative_l2_vs_fp32':(delta.norm()/reference_fp32.norm()).item()})})
        del partial_fp32, reference_fp32, reference_bf16, actual, delta
    trials = {'flux_ms':[], 'torch_ms':[]}
    for trial in range(3):
        order = [('torch_ms', baseline), ('flux_ms', fused)]
        if trial % 2: order.reverse()
        for name, fn in order:
            trials[name].append(timed(fn))
    flux_ms = statistics.median(max(ranks) for ranks in trials['flux_ms'])
    torch_ms = statistics.median(max(ranks) for ranks in trials['torch_ms'])
    row = {'M':m,'N':n,'K_global':k,'K_local':local_k,'checks':checks,
           'flux_ms':flux_ms,'torch_ms':torch_ms,'speedup':torch_ms/flux_ms,
           'flux_tflops_per_gpu':2*m*n*local_k/(flux_ms*1e9),'trials':trials}
    if rank == 0: print(json.dumps(row), flush=True)
    dist.barrier(pg)
    del op
    torch.cuda.synchronize()
    return row

with torch.no_grad():
    for m in [1024, 4096, 8192]:
        results.append(run_shape(m, 4096, 8192))
        if rank == 0:
            report={'job_id':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),
                    'gpu':torch.cuda.get_device_name(),'world_size':2,
                    'torch':torch.__version__,'torch_cuda':torch.version.cuda,
                    'torch_nccl':torch.cuda.nccl.version(),'flux_module':flux.__file__,
                    'path':'CUTLASS V2 / SM80 / BF16 / RCR / no bias / intra-node NVLink / fuse_reduction=False',
                    'method':'median of 3 trials, each max of 2 rank means; CUDA events; 10 warmups + 50 iterations; alternating order',
                    'sharding':'A[M,K/2] and B[N,K/2] per rank; sum products then scatter rows to C[M/2,N]',
                    'reduce_scatter_options':{'use_1d_ring':True,'use_p2p_read':True,'use_cudaMemcpyAsync':False,'use_gemmk':False,'per_tile_flags':False,'use_barrier_queue':False,'num_blocks':6,'n_split':1,'ring_mode':'Ring1D'},
                    'atol':0.02,'rtol':0.02,'shapes':results}
            (outdir/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if rank == 0: print('PASS: all shapes and seeds passed on both ranks',flush=True)
dist.barrier(pg)
dist.destroy_process_group()

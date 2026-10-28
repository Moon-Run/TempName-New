"""Paired operator sweep of selective MLP pipeline depth and Stream-K SM budget.

This is candidate screening, not full-step acceptance. All settings retain the
same physical tile map, selection mask, decoder and wire protocol.
"""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import random
import statistics
import time

# Match model-worker initialization: driver threads inherit the rank's CPU.
_rank = int(os.environ['LOCAL_RANK'])
_cpus = sorted(os.sched_getaffinity(0))
assert int(os.environ['WORLD_SIZE']) == 8
os.sched_setaffinity(0, {_cpus[_rank * len(_cpus) // 8]})

import torch
import torch.distributed as dist
from flux.gemm_rs_taco import GemmRSTacoDoubleBuffered
from taco_support import check_taco_output


@torch.no_grad()
def main(args):
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpus[rank * len(cpus) // 8]})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=5))
    group = dist.new_group(list(range(8)))
    plan = json.loads(args.plan.read_text())
    entry = next(s for s in plan['policies'][args.policy] if s['shape'] == [2048,2048,8192])
    m, n, k = entry['M'], entry['N'], entry['K_local']
    torch.manual_seed(817 + rank)
    a = (torch.randn(m, k, device='cuda') * .02).bfloat16()
    w = (torch.randn(n, k, device='cuda') * .02).bfloat16()
    op = GemmRSTacoDoubleBuffered(group, m, n, k, placement='fused', selected=entry['mask'])
    lib = op.slots[0].lib
    lib.taco_set_gemm_tuning.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.taco_set_gemm_tuning.restype = ctypes.c_int
    conditions = [(stage, sms) for stage in args.stages for sms in args.sms]
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(rank=rank, completed=False, policy=args.policy, shape=[m,n,k],
                  gpu_uuid=str(torch.cuda.get_device_properties(rank).uuid), conditions={})
    partial = a @ w.t()
    first = None
    for stages, sms in conditions:
        assert lib.taco_set_gemm_tuning(stages, sms) == 0
        actual = op.forward(a, w)
        torch.cuda.synchronize()
        error = check_taco_output(partial, actual, group, op.selection_cpu)
        if first is None:
            first = actual.clone()
        delta = actual.float() - first.float()
        report['conditions'][f's{stages}-sm{sms}'] = dict(stages=stages, avail_sms=sms,
            error=error, baseline_relative_l2=(delta.norm()/first.float().norm()).item(), samples=[])
    for block in range(args.blocks):
        order = conditions.copy()
        random.Random(1903 + block//2).shuffle(order)
        if block % 2:
            order.reverse()
        for stages, sms in order:
            assert lib.taco_set_gemm_tuning(stages, sms) == 0
            for _ in range(15):
                op.forward(a, w)
            dist.barrier()
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            begin.record()
            for _ in range(100):
                op.forward(a, w)
            end.record()
            end.synchronize()
            report['conditions'][f's{stages}-sm{sms}']['samples'].append(dict(
                block=block, wall_us=(time.perf_counter()-start)*1e4,
                gpu_us=begin.elapsed_time(end)*10))
        if rank == 0:
            print('BLOCK', block, flush=True)
    for key, row in report['conditions'].items():
        row['median_gpu_us'] = statistics.median(s['gpu_us'] for s in row['samples'])
        row['median_wall_us'] = statistics.median(s['wall_us'] for s in row['samples'])
    report['completed'] = True
    (args.out/f'rank{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    if rank == 0:
        print({k:round(v['median_gpu_us'],2) for k,v in report['conditions'].items()}, flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--policy', choices=['remote_first','interleaved'], required=True)
    parser.add_argument('--stages', type=int, nargs='+', default=[3,4])
    parser.add_argument('--sms', type=int, nargs='+', default=[0,104,100,96,92,88,84,80,72,64])
    parser.add_argument('--blocks', type=int, default=8)
    main(parser.parse_args())

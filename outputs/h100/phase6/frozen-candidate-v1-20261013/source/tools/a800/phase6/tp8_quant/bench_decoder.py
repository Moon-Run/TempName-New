"""Paired whole-operator diagnostics for decoder geometry; never step acceptance."""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
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
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(rank=rank, completed=False, policy=args.policy, rows=[])
    for entry in plan['policies'][args.policy]:
        m, n, k = entry['M'], entry['N'], entry['K_local']
        torch.manual_seed(817 + rank)
        a = (torch.randn(m, k, device='cuda') * .02).bfloat16()
        w = (torch.randn(n, k, device='cuda') * .02).bfloat16()
        op = GemmRSTacoDoubleBuffered(group, m, n, k, placement='fused', selected=entry['mask'])
        lib = op.slots[0].lib
        lib.taco_set_decode_variant.argtypes = [ctypes.c_int]
        lib.taco_set_decode_variant.restype = ctypes.c_int
        partial = a @ w.t()
        expected = None
        row = dict(shape=[m, n, k], variants={})
        for variant in args.variants:
            assert lib.taco_set_decode_variant(variant) == 0
            actual = op.forward(a, w)
            torch.cuda.synchronize()
            error = check_taco_output(partial, actual, group, op.selection_cpu)
            if expected is None:
                expected = actual.clone()
            else:
                assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), 'Decoder geometry changed output bits'
            row['variants'][variant] = dict(error=error, samples=[], output_exact=True)
        for block in range(args.blocks):
            order = args.variants if block % 2 == 0 else list(reversed(args.variants))
            for variant in order:
                assert lib.taco_set_decode_variant(variant) == 0
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
                row['variants'][variant]['samples'].append(dict(
                    block=block, wall_us=(time.perf_counter()-start)*1e4,
                    gpu_us=begin.elapsed_time(end)*10))
        for variant in args.variants:
            values = row['variants'][variant]
            values['median_gpu_us'] = statistics.median(s['gpu_us'] for s in values['samples'])
            values['median_wall_us'] = statistics.median(s['wall_us'] for s in values['samples'])
            assert lib.taco_set_decode_variant(variant) == 0
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    op.forward(a, w)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.out/f'm{m}-n{n}-k{k}-v{variant}-rank{rank}.json'))
        report['rows'].append(row)
        if rank == 0:
            print(row['shape'], {v:r['median_gpu_us'] for v,r in row['variants'].items()}, flush=True)
        (args.out/f'rank{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
        dist.barrier()
        torch.cuda.synchronize()
        del op
    report['completed'] = True
    (args.out/f'rank{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--policy', choices=['remote_first', 'interleaved'], required=True)
    parser.add_argument('--variants',type=int,nargs='+',default=[0,14,15,16,17])
    parser.add_argument('--blocks',type=int,default=8)
    main(parser.parse_args())

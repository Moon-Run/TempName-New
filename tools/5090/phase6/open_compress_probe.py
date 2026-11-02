#!/usr/bin/env python3
"""Eight-rank codec/oracle, wire, mapping, lifetime and operator measurements."""
import argparse
import ctypes
from datetime import timedelta
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

from rtx5090_loader import HERE
import open_compress_protocol as p

ROLES = {'attn': 512, 'mlp': 2048}
CASES = ('zero', 'random', 'outlier')
CODEC_CASES = ('zero', 'signed_halfway', 'random', 'outlier', 'tiny')


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_codec_cases(record):
    p.require(record['completed'] is True and set(record['cases']) == set(CODEC_CASES),
              'Missing standalone codec arithmetic cases')
    for case, item in record['cases'].items():
        p.require(all(item[key] is True for key in ('finite_input', 'encode_bitwise', 'decode_bitwise',
                  'reduce_bitwise', 'finite_output')) and is_digest(item['packet_sha256']),
                  'Standalone codec arithmetic/reference failed: '+case)
    p.require(record['cases']['signed_halfway']['expected_signed_bytes'] ==
              [127, -127, 0, 2, 2, 4, 0, -2, -2, -4], 'Nearest-even signed rounding failed')
    p.require(record['nonfinite'] == dict(nan_inf_negative_inf=True, affected_remote_groups=7,
              nan_scale_propagated=True, decode_nan_propagated=True,
              reduction_nan_propagated=True, unaffected_values_zero=True),
              'NaN/Inf silently disappeared in the codec')


def codec_cases(rank):
    """Isolated arithmetic boundaries; these do not exercise network collectives."""
    import torch
    import open_compress_codec as codec
    cases = {}
    for case in CODEC_CASES:
        if case in ('random', 'outlier'):
            generator = torch.Generator(device='cuda').manual_seed(44007+rank)
            partial = torch.randn((2048, 4096), dtype=torch.bfloat16, device='cuda', generator=generator)
            if case == 'outlier':
                partial.view(-1, 128)[:, 17] = 32
        else:
            partial = torch.zeros((2048, 4096), dtype=torch.bfloat16, device='cuda')
            if case == 'signed_halfway':
                values = torch.tensor([127, -127, .5, 1.5, 2.5, 3.5, -.5, -1.5, -2.5, -3.5],
                                      dtype=torch.bfloat16, device='cuda')
                partial.view(-1, 128)[:, :10] = values
                del values
            elif case == 'tiny':
                partial.view(-1, 128)[:, 0] = 2.0**-100
                partial.view(-1, 128)[:, 1] = -2.0**-100
        wire = codec.encode(partial, rank=rank)
        decoded = codec.decode(wire)
        reduced = codec.decode_reduce(wire, partial, rank=rank)
        item = dict(finite_input=bool(torch.isfinite(partial).all()),
            encode_bitwise=torch.equal(wire, codec.reference_packets(partial, rank)),
            decode_bitwise=torch.equal(decoded, codec.reference_decode(wire)),
            reduce_bitwise=torch.equal(reduced, codec.reference_reduce(wire, partial, rank)),
            finite_output=bool(torch.isfinite(decoded).all() & torch.isfinite(reduced).all()),
            packet_sha256=tensor_sha(wire))
        if case == 'signed_halfway':
            item['expected_signed_bytes'] = wire[:10].view(torch.int8).cpu().tolist()
        cases[case] = item
        del partial, wire, decoded, reduced
    partial = torch.zeros((2048, 4096), dtype=torch.bfloat16, device='cuda')
    # The first block in every peer contains all three invalid classes.
    partial.view(8, -1)[:, 0] = float('nan')
    partial.view(8, -1)[:, 1] = float('inf')
    partial.view(8, -1)[:, 2] = float('-inf')
    wire = codec.encode(partial, rank=rank)
    decoded = codec.decode(wire).view(7, -1)
    local = torch.zeros((256, 4096), dtype=torch.bfloat16, device='cuda')
    reduced = codec.decode_reduce(wire, local, rank=rank).view(-1)
    scale_nan = all(bool(torch.isnan(packet[codec.ELEMENTS:].view(torch.float32)[0]))
                    for packet in wire.view(7, codec.PEER_BYTES))
    nonfinite = dict(nan_inf_negative_inf=True, affected_remote_groups=7,
        nan_scale_propagated=scale_nan, decode_nan_propagated=bool(torch.isnan(decoded[:, :128]).all()),
        reduction_nan_propagated=bool(torch.isnan(reduced[:128]).all()),
        unaffected_values_zero=bool((decoded[:, 128:] == 0).all() & (reduced[128:] == 0).all()))
    result = dict(completed=True, cases=cases, nonfinite=nonfinite)
    validate_codec_cases(result)
    return result


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def measure(fn, iterations=10, samples=5):
    import torch
    import torch.distributed as dist
    # Reference tensors are released by the caller. Drop cached allocations
    # before warming this component so oracle scratch cannot inflate reserved.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    records = []
    for _ in range(samples):
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        free, total = torch.cuda.mem_get_info()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        elapsed = time.perf_counter()-wall
        records.append(dict(cuda_ms=start.elapsed_time(end)/iterations, wall_ms=elapsed*1000/iterations,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            incremental_allocated_bytes=torch.cuda.max_memory_allocated()-allocated,
            baseline_allocated_bytes=allocated, device_free_before_bytes=free, device_total_bytes=total))
    return dict(iterations=iterations, warmup=3, samples=records)


def validate_measurement(record):
    p.require(record['iterations'] == 10 and record['warmup'] == 3 and len(record['samples']) == 5,
              'Incomplete operator measurement')
    for s in record['samples']:
        p.require(all(math.isfinite(s[k]) and s[k] > 0 for k in
                  ('cuda_ms', 'wall_ms', 'peak_allocated_gib', 'peak_reserved_gib'))
                  and type(s['incremental_allocated_bytes']) is int and s['incremental_allocated_bytes'] >= 0
                  and type(s['baseline_allocated_bytes']) is int and s['baseline_allocated_bytes'] > 0
                  and s['peak_allocated_gib']*2**30 == s['baseline_allocated_bytes']+s['incremental_allocated_bytes']
                  and s['peak_reserved_gib'] >= s['peak_allocated_gib']
                  and 0 <= s['device_free_before_bytes'] <= s['device_total_bytes'], 'Invalid operator measurement')


def validate(root, arm, plan, worker):
    from op1_a_probe import validate_mapping
    root = Path(root)
    rows = [p.read(root/f'rank{r}.json') for r in range(8)]
    for rank, row in enumerate(rows):
        p.require(row['protocol'] == p.PROTOCOL and row['arm'] == arm and row['rank'] == rank
                  and row['run_id'] == plan['run_id'] and row['completed'] is True
                  and row['build_manifest_sha256'] == plan['build']['manifest_sha256']
                  and row['entrypoint_sha256'] == p.sha(__file__)
                  and row['libraries'] == worker['libraries']
                  and row['codec_library'] == worker['config']['codec_library']
                  and row['stream'] != row['default_stream'] and set(row['roles']) == set(ROLES),
                  'Missing/foreign operator evidence')
        p.require(row['matmul_settings'] == dict(allow_tf32=False,
                  allow_bf16_reduced_precision_reduction=False), 'Operator GEMM reference precision changed')
        if arm != p.ARMS[0]:
            validate_codec_cases(row['codec_cases'])
        else:
            p.require(row['codec_cases'] is None, 'BF16 arm unexpectedly ran an INT8 codec')
        for role, record in row['roles'].items():
            p.require(record['k_local'] == ROLES[role] and set(record['cases']) == set(CASES)
                      and record['retention_passed'] is True and record['retained_outputs'] == 6
                      and record['reuse_calls'] == 7 and record['changed_call_output_different'] is True
                      and record['delayed_rank'] == 7 and record['delay_seconds'] == .02,
                      'Incomplete shape/case/lifetime coverage')
            if arm == p.ARMS[2]:
                validate_mapping(record['mapping'], rank)
            else:
                p.require(record['mapping'] is None, 'Unexpected interleaved mapping')
            for case, value in record['cases'].items():
                p.require(value['passed'] is True and value['finite_partial'] is True
                          and math.isfinite(value['relative_l2_vs_bf16'])
                          and math.isfinite(value['gemm_relative_l2'])
                          and 0 <= value['relative_l2_vs_bf16'] <= .05 and 0 <= value['gemm_relative_l2'] <= .01
                          and all(is_digest(value[k]) for k in ('input_sha256', 'weight_sha256')),
                          'Operator numerical oracle failed')
                if arm != p.ARMS[0]:
                    p.require(value['encode_bitwise'] and value['decode_bitwise'] and value['reduce_bitwise'],
                              'Codec differs from independent Torch oracle')
                    peers = {str(i) for i in range(8) if i != rank}
                    p.require(set(value['tx']) == set(value['rx']) == peers, 'Missing peer packets')
                    p.require(all(is_digest(h) for direction in ('tx', 'rx') for h in value[direction].values()),
                              'Invalid directed packet hash')
                    for peer in peers:
                        other = rows[int(peer)]['roles'][role]['cases'][case]
                        p.require(value['tx'][peer] == other['rx'][str(rank)], 'Directed wire bytes changed')
            state = record['transport']
            expected_calls = len(CASES)+7
            p.require(state['calls'] == expected_calls and state['owns_output'] is True,
                      'Incorrect lifetime probe call count/output ownership')
            if arm != p.ARMS[0]:
                p.require(all(state[k] == expected_calls for k in
                          ('encode_calls', 'decode_calls', 'all_to_all_calls', 'wait_calls'))
                          and state['caller_stream'] == state['current_stream'] == row['stream']
                          and state['pending_exchange'] is False and state['failed'] is False
                          and state['send_split_bytes'] == state['receive_split_bytes'] ==
                          [0 if peer == rank else p.PACKET_BYTES for peer in range(8)],
                          'Missing completed current-stream encode/exchange/wait/decode evidence')
            required = {'gemm', 'operator'} | (set() if arm == p.ARMS[0] else
                {'encode', 'exchange', 'decode', 'decode_reduce'})
            p.require(set(record['timings']) == required, 'Missing component timings')
            for value in record['timings'].values():
                validate_measurement(value)
    return rows


def run(args):
    import torch
    import torch.distributed as dist
    import open_compress_build as builder
    import open_compress_codec as codec
    import open_compress_profile as profile
    from op1_a_probe import loaded_libraries, validate_mapping
    from flux_tensor_taco import FluxTensorTaco
    rank = int(os.environ['LOCAL_RANK'])
    record = dict(protocol=p.PROTOCOL, arm=args.arm, rank=rank, run_id=os.environ['H100_RUN_ID'],
        completed=False, entrypoint_sha256=p.sha(__file__),
        build_manifest_sha256=p.sha(args.build/builder.MANIFEST))
    try:
        torch.cuda.set_device(rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        prop = torch.cuda.get_device_properties(rank)
        p.require(torch.cuda.device_count() == 8 and '5090' in prop.name
                  and (prop.major, prop.minor, prop.multi_processor_count) == (12, 0, 170), 'Expected RTX5090 TP8')
        record.update(gpu_uuid=str(prop.uuid), torch=str(torch.__version__), cuda=torch.version.cuda,
                      nccl=list(torch.cuda.nccl.version()), matmul_settings=dict(
                          allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                          allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
        dist.init_process_group('nccl', timeout=timedelta(seconds=300))
        stream = torch.cuda.Stream()
        record.update(stream=int(stream.cuda_stream), default_stream=int(torch.cuda.default_stream().cuda_stream), roles={})
        native = builder.native_root(args.build, args.arm)
        sorted_lib = None
        if args.arm == p.ARMS[2]:
            sorted_lib = ctypes.CDLL(str(native/'libflux_cuda.so'))
            sorted_lib.op1_a_sorted_mapping.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            sorted_lib.op1_a_sorted_mapping.restype = ctypes.c_int
        with torch.cuda.stream(stream):
            record['codec_cases'] = codec_cases(rank) if args.arm != p.ARMS[0] else None
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            for role, k in ROLES.items():
                op = FluxTensorTaco(dist.group.WORLD, 2048, 4096, k)
                op.profile = op.tensor_codec.profile = False
                mapping = None
                if sorted_lib is not None:
                    mapped = torch.empty((512, 4), dtype=torch.int32, device='cuda')
                    p.require(sorted_lib.op1_a_sorted_mapping(mapped.data_ptr(), k, stream.cuda_stream) == 0,
                              'Actual registered GEMM mapping probe failed')
                    mapping = mapped.cpu().tolist()
                    validate_mapping(mapping, rank)
                    del mapped
                result = dict(k_local=k, mapping=mapping, cases={})
                for case in CASES:
                    generator = torch.Generator(device='cuda').manual_seed(12345+rank+k)
                    x = torch.randn((2048, k), dtype=torch.bfloat16, device='cuda', generator=generator)*.02
                    w = torch.randn((4096, k), dtype=torch.bfloat16, device='cuda', generator=generator)*.02
                    if case == 'zero':
                        x.zero_()
                    elif case == 'outlier':
                        x[::37, ::31] *= 32
                    actual = op.forward(x, w)
                    partial = op.partial
                    p.require(bool(torch.isfinite(partial).all()), 'Nonfinite GEMM output')
                    independent = torch.matmul(x, w.t())
                    delta = (partial.float()-independent.float()).norm()/independent.float().norm().clamp_min(1e-20)
                    reference = torch.empty_like(actual)
                    dist.reduce_scatter_tensor(reference, independent, group=dist.group.WORLD)
                    relative = ((actual.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-20)).item()
                    value = dict(input_sha256=tensor_sha(x), weight_sha256=tensor_sha(w),
                        finite_partial=True, gemm_relative_l2=delta.item(), relative_l2_vs_bf16=relative,
                        passed=relative <= .05 and delta.item() <= .01)
                    if args.arm != p.ARMS[0]:
                        c = op.tensor_codec
                        value.update(encode_bitwise=torch.equal(c.tx, codec.reference_packets(partial, rank)),
                            decode_bitwise=torch.equal(codec.decode(c.rx), codec.reference_decode(c.rx)),
                            reduce_bitwise=torch.equal(actual, codec.reference_reduce(c.rx, partial, rank)))
                        for direction in ('tx', 'rx'):
                            raw = getattr(c, direction).view(7, p.PACKET_BYTES)
                            value[direction] = {str(peer): tensor_sha(raw[peer-int(peer > rank)])
                                                for peer in range(8) if peer != rank}
                        value['passed'] &= all(value[key] for key in ('encode_bitwise', 'decode_bitwise', 'reduce_bitwise'))
                    p.require(value['passed'], f'Operator oracle failed: {role}/{case}: {value}')
                    result['cases'][case] = value
                outputs = [op.forward(x, w) for _ in range(6)]
                copies = [v.clone() for v in outputs]
                # Change the seventh submission's data: identical results
                # could conceal a stale/overwritten output allocation.
                if rank == 7:
                    time.sleep(.02)
                new_output = op.forward(x * 1.5, w)
                changed = not torch.equal(new_output, copies[-1])
                retained = len({v.data_ptr() for v in outputs}) == 6 and all(torch.equal(a,b) for a,b in zip(outputs,copies))
                p.require(retained and changed, 'Output storage reused/overwritten or changed data was not consumed')
                result.update(retention_passed=True, retained_outputs=6, reuse_calls=7,
                              changed_call_output_different=changed, delayed_rank=7, delay_seconds=.02,
                              transport=op.tensor_codec.state())
                del outputs, copies, new_output, actual, partial, independent, reference, delta, generator
                if args.arm != p.ARMS[0]:
                    del c, raw
                stages = {'gemm': lambda: op.gemm.forward(x, w, output_buf=op.partial, fast_accum=False),
                          'operator': lambda: op.forward(x, w)}
                result['timings'] = {name: measure(fn) for name,fn in stages.items()}
                if args.arm != p.ARMS[0]:
                    c = op.tensor_codec
                    decoded = torch.empty((7, 256, 4096), dtype=torch.bfloat16, device='cuda')
                    reduced = torch.empty((256, 4096), dtype=torch.bfloat16, device='cuda')
                    def exchange():
                        dist.all_to_all_single(c.rx, c.tx, output_split_sizes=c.splits,
                            input_split_sizes=c.splits, async_op=True).wait()
                    stages = dict(encode=lambda: codec.encode(op.partial, c.tx, rank), exchange=exchange,
                        decode=lambda: codec.decode(c.rx, decoded),
                        decode_reduce=lambda: codec.decode_reduce(c.rx, op.partial, reduced, rank))
                    result['timings'].update({name: measure(fn) for name,fn in stages.items()})
                if args.arm != p.ARMS[0]:
                    del decoded, reduced, c
                record['roles'][role] = result
                del op, x, w, stages
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            # A separate instrumented invocation; never pooled into operator timing.
            operators = {role: FluxTensorTaco(dist.group.WORLD, 2048, 4096, k) for role,k in ROLES.items()}
            # Warm lazy GEMM scratch/collective state before capture, using the
            # same stream as profiling. Constructors do not warm GEMM kernels.
            inputs = {role: (torch.zeros((2048, k), dtype=torch.bfloat16, device='cuda'),
                            torch.zeros((4096, k), dtype=torch.bfloat16, device='cuda')) for role,k in ROLES.items()}
            for role, op in operators.items():
                op.forward(*inputs[role])
                op.profile = op.tensor_codec.profile = True
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for role, op in operators.items():
                    with torch.profiler.record_function(f'scope.{role}.flux_tensor'):
                        op.forward(*inputs[role])
                torch.cuda.synchronize()
            trace = args.out/f'rank{rank}-profile.json'
            prof.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())['traceEvents']
            record['profile'] = profile.analyze(events, args.arm, rank, expected_calls=1)
            with gzip.open(str(trace)+'.gz', 'wb') as output:
                output.write(trace.read_bytes())
            trace.unlink()
            record['profile_sha256'] = p.sha(str(trace)+'.gz')
        record['libraries'] = loaded_libraries(native)
        import megatron_taco
        record['codec_library'] = megatron_taco.loaded_library()
        record['completed'] = True
        p.write(args.out/f'rank{rank}.json', record)
        dist.destroy_process_group()
    except BaseException as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        p.write(args.out/f'rank{rank}.json', record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=p.ARMS, required=True)
    run(parser.parse_args())

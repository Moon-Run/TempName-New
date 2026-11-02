#!/usr/bin/env python3
"""TP8 combined projection operator: gradients, latency and stage diagnostics.

Only op3-combined is timed. The unfused autograd path is a correctness oracle,
not a second benchmark arm. No MA8/MA9/MA11 training work is attributed to this
projection microbenchmark; those mechanisms are exercised by full-step workers.
"""
import argparse
from datetime import timedelta
import gzip
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import sys
import time

sys.dont_write_bytecode = True
from rtx5090_loader import HERE, REPO
import op3_protocol as p
import op3_profile as profile
import op3_validation as validation
import op1_abc_probe as original

PROTOCOL = 'rtx5090-op3-combined-operator-v1'
SETTINGS = dict(warmup=20, iterations=100, windows=1, profile_calls=10, gradient_steps=3)
ROLES = {'attn': 512, 'mlp': 2048}
PARAMETERS = ('input', 'weight', 'bias', 'residual')
require, read, write, sha = p.require, p.read, p.write, p.sha


def expected_calls():
    total = 2*SETTINGS['gradient_steps']+SETTINGS['warmup']+SETTINGS['windows']*SETTINGS['iterations']+SETTINGS['profile_calls']
    return total, total-SETTINGS['gradient_steps']


def validate_rows(rows, plan, hardware, libraries, codec_library):
    require(len(rows) == 8, 'Eight operator rank records required')
    calls, fused = expected_calls()
    for rank, row in enumerate(rows):
        require(row['protocol'] == PROTOCOL and row['arm'] == p.COMBINED and row['rank'] == rank
                and row['completed'] is True and row['settings'] == SETTINGS and set(row['roles']) == set(ROLES)
                and row['run_id'] == plan['run_id'] and row['entrypoint_sha256'] == sha(__file__)
                and row['build_manifest_sha256'] == plan['build']['manifest_sha256'], 'Foreign/incomplete operator run')
        require(row['gpu_uuid'] == hardware['devices'][rank]['uuid']
                and all(row[key] == hardware['environment'][key] for key in ('torch', 'cuda', 'nccl'))
                and row['libraries'] == libraries and row['codec_library'] == codec_library,
                'Operator GPU/environment/library differs from preflight')
        require(type(row['stream']) is int and row['stream'] > 0 and row['stream'] != row['default_stream'],
                'Operator requires its own non-default stream')
        for role, k in ROLES.items():
            item = row['roles'][role]
            require(item['shape'] == [2048, 4096, k] and len(item['gradient_checks']) == SETTINGS['gradient_steps'],
                    'Operator shape/gradient coverage changed')
            for step, check in enumerate(item['gradient_checks']):
                require(check['step'] == step and check['output_bitwise'] is True
                        and check['inputs_unchanged'] is True
                        and set(check['gradients']) == set(check['updates']) == set(PARAMETERS),
                        'Missing operator output/gradient/update checks')
                for kind in ('gradients', 'updates'):
                    for value in check[kind].values():
                        require(value['bitwise'] is True and original.is_digest(value['actual_sha256'])
                                and value['actual_sha256'] == value['reference_sha256'],
                                'Fused operator gradient or parameter update differs from reference')
                require(check['output_sha256'] == check['reference_output_sha256']
                        and original.is_digest(check['output_sha256']), 'Operator output hash mismatch')
            windows = item['windows']
            require(len(windows) == SETTINGS['windows'], 'Missing operator timing windows')
            for index, window in enumerate(windows):
                require(window['index'] == index and window['iterations'] == SETTINGS['iterations']
                        and all(math.isfinite(window[key]) and window[key] > 0 for key in
                                ('gpu_us', 'wall_us')), 'Invalid operator timing sample')
            validation.validate_codec_state(item['state'], p.COMBINED, calls, rank, fused_calls=fused)
            require(item['state']['caller_stream'] == row['stream'], 'Operator changed stream')
    return rows


def analyze(events, role, rank):
    require(role in ROLES and type(rank) is int and 0 <= rank < 8, 'Wrong operator profile identity')
    result = profile.chunk_records(events, p.COMBINED, rank,
        role_counts={role: SETTINGS['profile_calls']}, operator=True)
    return dict(protocol=PROTOCOL, arm=p.COMBINED, rank=rank, role=role,
                execution_validated=True, forward=result[role])


def summarize(rows, profiles):
    roles = {}
    for role in ROLES:
        # Retain all windows; the distributed latency is the slowest rank in
        # each matching window, not an average of independently fast ranks.
        windows = [dict(index=i, **{key: max(row['roles'][role]['windows'][i][key] for row in rows)
                    for key in ('gpu_us', 'wall_us')}) for i in range(SETTINGS['windows'])]
        roles[role] = dict(shape=[2048, 4096, ROLES[role]], all_windows=windows,
            **{key: dict(mean=mean(w[key] for w in windows), median=median(w[key] for w in windows),
                        minimum=min(w[key] for w in windows), maximum=max(w[key] for w in windows))
               for key in ('gpu_us', 'wall_us')},
            profile_median_us={key: median(profiles[f'{role}/rank{rank}']['forward']['median_stage_us'][key]
                                           for rank in range(8))
                               for key in ('gemm_us', 'encode_us', 'nccl_us', 'decode_us', 'cpu_us', 'gpu_span_us')})
    return dict(protocol=PROTOCOL, completed=True, arm=p.COMBINED, settings=SETTINGS,
        roles=roles, operator_gradients_validated=True, source_order_preserved=True,
        timed_arms=[p.COMBINED], full_step_measured=False, speedup_evaluated=False,
        latency_scope='GemmOnly + warp H128 encode + completed NCCL exchange + R decode + BF16 bias/residual',
        limitation='Preallocated inputs and one persistent operator per role. No LayerNorm, backward, '
                   'optimizer, MA8/MA9/MA11 work or CUDA Graph in the timing interval. '
                   'Forward/backward and three updates are separately checked against unfused autograd. '
                   'Profile kernel times are diagnostics, not additive full-step savings.')


def markdown(result):
    require(result['protocol'] == PROTOCOL and result['completed'], 'Incomplete operator report')
    lines = ['# OP3 组合组：独立投影算子', '', p.LABELS[p.COMBINED], '',
        '每角色 20 次预热＋1 个窗口×100 次调用；每窗口分别取八 rank 最大 CUDA-event/同步墙钟平均延迟。',
        '正确性阶段检查融合/非融合输出、四类梯度和三步更新逐位一致；只有组合路径进入计时。', '',
        '| 角色 | GPU 平均 μs | GPU 中位 μs | GPU 最慢窗口 μs | 墙钟平均 μs |',
        '| --- | ---: | ---: | ---: | ---: |']
    for role, v in result['roles'].items():
        lines.append(f"| {role} | {v['gpu_us']['mean']:.3f} | {v['gpu_us']['median']:.3f} | "
                     f"{v['gpu_us']['maximum']:.3f} | {v['wall_us']['mean']:.3f} |")
    lines += ['', '独立 profile：先取每 rank 的调用中位数，再取八 rank 中位数。', '',
        '| 角色 | GEMM μs | 编码 μs | NCCL kernel 合计 μs | 解码含融合 μs | CPU 提交 μs | GPU 起止跨度 μs |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for role, v in result['roles'].items():
        times = v['profile_median_us']
        lines.append('| '+role+' | '+' | '.join(f'{times[k]:.3f}' for k in
            ('gemm_us', 'encode_us', 'nccl_us', 'decode_us', 'cpu_us', 'gpu_span_us'))+' |')
    return '\n'.join(lines+['', result['limitation'], ''])


def replay(root, plan, hardware):
    import op3_build as builder
    root = Path(root)
    build = Path(plan['build']['root'])
    rows = validate_rows([read(root/f'rank{rank}.json') for rank in range(8)], plan, hardware,
        builder.libraries(build, p.build_arm(p.COMBINED)), builder.codec_library(build))
    profiles, files = {}, {}
    for rank, row in enumerate(rows):
        files[f'rank{rank}.json'] = sha(root/f'rank{rank}.json')
        for role in ROLES:
            name = f'{role}-rank{rank}-profile.json.gz'
            require(row['roles'][role]['trace_sha256'] == sha(root/name), 'Operator trace changed')
            with gzip.open(root/name, 'rt') as handle:
                profiles[f'{role}/rank{rank}'] = analyze(json.load(handle)['traceEvents'], role, rank)
            files[name] = sha(root/name)
    return dict(summarize(rows, profiles), run_id=plan['run_id'], profiles=profiles, evidence_files=files,
                build_manifest_sha256=plan['build']['manifest_sha256'])


def evidence(root, plan, hardware):
    root = Path(root)/'operator'
    result = replay(root, plan, hardware)
    require(result == read(root/'report.json'), 'Operator report differs from recorded evidence')
    return result


def run_stage(args, build, plan):
    import op3 as controller
    from tp8_dp2_train import run_process
    controller.base.verify_controllers(args.out)
    controller.base.gpu_idle()
    dest = args.out/'operator'
    dest.mkdir()
    job = dict(arm=p.COMBINED, mode='timing', trial=0, block=-1, index=-1)
    scripts = args.out/'workers'/p.COMBINED/'scripts'
    controller.base.check_worker(args.out, p.COMBINED, read(args.out/'workers.json')[p.COMBINED])
    env = controller.worker_environment(args, build, scripts, job, dest, plan)
    command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
        '--max-restarts=0', HERE/'op3_operator.py', '--out', dest, '--execute']
    write(dest/'command.json', dict(command=list(map(str, command)), job=job))
    try:
        run_process(command, REPO, env, dest/'process.log', max(900, args.worker_timeout), controller.base.Peer())
    except Exception:
        controller.base.print_smoke_failures(dest)
        raise
    result = replay(dest, plan, read(args.out/'environment-preflight.json'))
    write(dest/'report.json', result)
    (dest/'report.md').write_text(markdown(result))
    return result


def compare_tensor(torch, actual, reference):
    require(torch.isfinite(actual).all().item() and torch.isfinite(reference).all().item(),
            'Nonfinite operator value')
    same = torch.equal(actual.detach().contiguous().view(torch.uint8), reference.detach().contiguous().view(torch.uint8))
    require(same, 'Fused operator tensor differs bitwise from unfused reference')
    return dict(bitwise=True, actual_sha256=original.tensor_sha(actual), reference_sha256=original.tensor_sha(reference))


def gradient_checks(torch, adapter, op, role, rank):
    k = ROLES[role]
    generator = torch.Generator(device='cuda').manual_seed(20261102+rank*97+k)
    shapes = [(2048, 1, k), (4096, k), (4096,), (256, 1, 4096)]
    actual = [torch.randn(shape, generator=generator, dtype=torch.bfloat16, device='cuda').mul_(.03).requires_grad_()
              for shape in shapes]
    reference = [value.detach().clone().requires_grad_() for value in actual]
    results = []
    for step in range(SETTINGS['gradient_steps']):
        before = [original.tensor_sha(x) for x in actual+reference]
        grad = torch.randn(shapes[-1], generator=generator, dtype=torch.bfloat16, device='cuda').mul_(.03)
        raw = adapter.GemmRSFunction.apply(reference[0], reference[1], op, None)
        expected = reference[3]+(raw+reference[2])
        output = adapter.ConsumerGemmRSFunction.apply(*actual, op, None)
        check = compare_tensor(torch, output, expected)
        expected_grads = torch.autograd.grad(expected, reference, grad)
        grads = torch.autograd.grad(output, actual, grad)
        checks = {name: compare_tensor(torch, a, b) for name, a, b in zip(PARAMETERS, grads, expected_grads)}
        require(before == [original.tensor_sha(x) for x in actual+reference], 'Operator modified its inputs')
        # Exercise both graphs through multiple updates, including bias/residual.
        with torch.no_grad():
            for a, b, da, db in zip(actual, reference, grads, expected_grads):
                a.add_(da, alpha=-1/128)
                b.add_(db, alpha=-1/128)
        updates = {name: compare_tensor(torch, a, b) for name, a, b in zip(PARAMETERS, actual, reference)}
        results.append(dict(step=step, output_bitwise=True, inputs_unchanged=True,
            output_sha256=check['actual_sha256'], reference_output_sha256=check['reference_sha256'],
            gradients=checks, updates=updates))
    return results


def measure_forward(torch, dist, forward, stream, rank):
    for _ in range(SETTINGS['warmup']):
        output = forward()
    windows = []
    for i in range(SETTINGS['windows']):
        dist.barrier(device_ids=[rank])
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record(stream)
        for _ in range(SETTINGS['iterations']):
            output = forward()
        stop.record(stream)
        stop.synchronize()
        elapsed = time.perf_counter()-begin
        windows.append(dict(index=i, iterations=SETTINGS['iterations'],
            gpu_us=start.elapsed_time(stop)*1000/SETTINGS['iterations'],
            wall_us=elapsed*1e6/SETTINGS['iterations']))
    return windows


def run(args):
    import torch
    import torch.distributed as dist
    import op3_build as builder
    rank = int(os.environ['LOCAL_RANK'])
    require(os.environ['WORLD_SIZE'] == os.environ['LOCAL_WORLD_SIZE'] == '8'
            and os.environ['E2E_POLICY'] == p.COMBINED, 'Expected eight combined operator ranks')
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    root = args.out.parent
    plan = read(root/'plan.json')
    build = Path(plan['build']['root'])
    native = builder.native_root(build, p.build_arm(p.COMBINED))
    scripts = root/'workers'/p.COMBINED/'scripts'
    sys.path.insert(0, str(native/'python'))
    sys.path.insert(0, str(scripts))
    record = dict(protocol=PROTOCOL, arm=p.COMBINED, rank=rank, completed=False,
        settings=SETTINGS, run_id=plan['run_id'], entrypoint_sha256=sha(__file__),
        build_manifest_sha256=sha(build/builder.MANIFEST), roles={})
    initialized = False
    try:
        dist.init_process_group('nccl', timeout=timedelta(seconds=300))
        initialized = True
        from megatron.core import parallel_state
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=8, distributed_timeout_minutes=5)
        import flux  # Load the registered native libraries before checking /proc/self/maps.
        import adapter
        from taco_support import TacoForward
        import megatron_taco as codec
        stream = torch.cuda.Stream()
        prop = torch.cuda.get_device_properties(rank)
        require('5090' in prop.name and (prop.major, prop.minor, prop.multi_processor_count) == (12, 0, 170),
                'Expected RTX 5090 SM120')
        record.update(gpu_uuid=str(prop.uuid), torch=str(torch.__version__), cuda=torch.version.cuda,
            nccl=list(torch.cuda.nccl.version()), libraries=original.loaded_libraries(native),
            stream=int(stream.cuda_stream), default_stream=int(torch.cuda.default_stream().cuda_stream))
        with torch.cuda.stream(stream):
            for role, k in ROLES.items():
                op = TacoForward(parallel_state.get_tensor_model_parallel_group(), 2048, 4096, k, p.COMBINED, role)
                checks = gradient_checks(torch, adapter, op, role, rank)
                generator = torch.Generator(device='cuda').manual_seed(509003+k+rank)
                x = torch.randn((2048, k), generator=generator, device='cuda', dtype=torch.bfloat16)*.03
                weight = torch.randn((4096, k), generator=generator, device='cuda', dtype=torch.bfloat16)*.03
                bias = torch.randn((4096,), generator=generator, device='cuda', dtype=torch.bfloat16)*.03
                residual = torch.randn((256, 4096), generator=generator, device='cuda', dtype=torch.bfloat16)*.03
                def forward():
                    return op.forward_consumer(x, weight, bias, residual)
                with torch.no_grad():
                    windows = measure_forward(torch, dist, forward, stream, rank)
                    torch.cuda.synchronize()
                    op.codec.profile = op.codec.tensor_codec.profile = True
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(SETTINGS['profile_calls']):
                            with torch.profiler.record_function(f'scope.{role}.flux_tensor'):
                                output = forward()
                    torch.cuda.synchronize()
                    op.codec.profile = op.codec.tensor_codec.profile = False
                    plain = args.out/f'{role}-rank{rank}-profile.json'
                    trace = plain.with_suffix('.json.gz')
                    prof.export_chrome_trace(str(plain))
                    with plain.open('rb') as src, gzip.open(trace, 'wb') as dest:
                        import shutil
                        shutil.copyfileobj(src, dest)
                    plain.unlink()
                record['roles'][role] = dict(shape=[2048, 4096, k], gradient_checks=checks, windows=windows,
                    state=op.codec.tensor_codec.state(), trace_sha256=sha(trace))
                op.codec.tensor_codec.close()
                del op, x, weight, bias, residual, output
            record['codec_library'] = codec.loaded_library()
        record['completed'] = True
        write(args.out/f'rank{rank}.json', record)
        dist.barrier(device_ids=[rank])
        parallel_state.destroy_model_parallel()
    except BaseException as exc:
        record.update(completed=False, error=f'{type(exc).__name__}: {exc}')
        write(args.out/f'rank{rank}.json', record)
        raise
    finally:
        if initialized:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(protocol=PROTOCOL, arm=p.COMBINED, settings=SETTINGS, roles=ROLES)))
        return
    args.out = args.out.resolve()
    run(args)


if __name__ == '__main__':
    main()

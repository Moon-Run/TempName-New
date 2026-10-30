#!/usr/bin/env python3
"""Fresh same-codec MA20 calibration, followed by an independent held-out gate.

The measured API includes producer, publication, ordered decode, bias/residual
and output ownership. It is an operator selection stage, not a training result.
"""
import argparse
import ast
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import uuid

import ma5_bench as helpers

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PROTOCOL = 'h100-ma20-codec-first-v1-operator'
ARMS = ('MA11', 'Q0', 'Q7', 'Q14', 'Q28', 'Q56', 'Q112')
ROLES = {'attn': [2048, 4096, 512], 'mlp': [2048, 4096, 2048]}
CASES, SEEDS = ['zero', 'small', 'outlier', 'random'], [817, 919]
TIMING = dict(warmup=20, iterations=100)
ENVIRONMENT = {'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'FLUX_GEMM_RS_GEMM_ONLY': '1',
               'NCCL_NVLS_ENABLE': '0', 'NCCL_DEBUG': 'WARN', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1'}
read, write, sha, require = helpers.read, helpers.write, helpers.sha, helpers.require


def schedule():
    training = [list(ARMS[i:] + ARMS[:i]) for i in range(7)]
    orders = training + [row[::-1] for row in training[::-1]]
    return [dict(block=b, arm=arm, index=b * 7 + i,
                 split='train' if b < 7 else 'heldout', directory=f'block{b:02d}-{arm}')
            for b, order in enumerate(orders) for i, arm in enumerate(order)]


def smoke_jobs():
    return [dict(block=-1, arm=arm, index=i, split='smoke', directory='smoke-' + arm)
            for i, arm in enumerate(ARMS)]


def libraries(build):
    result = {}
    for arm in ARMS:
        root = (Path(build) / arm).resolve()
        paths = [root / 'python/flux/lib' / name for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')]
        paths += sorted((root / 'python').glob('flux_ths_pybind*.so'))
        require(len(paths) >= 3 and all(p.is_file() for p in paths), 'Missing MA20 linked runtime: ' + arm)
        mapping = root / 'check_mapping'
        require(mapping.is_file(), 'Missing MA20 mapping executable')
        result[arm] = dict(root=str(root), files={str(p.resolve()): sha(p) for p in paths},
                           mapping=dict(path=str(mapping), sha256=sha(mapping)))
    return result


def identity(build, builder_root=None):
    if builder_root is not None:
        sys.path.insert(1, str(builder_root))
    import ma20_build as builder
    root = Path(build).resolve()
    record = builder.verify(root, built=True)
    return dict(builder_root=str(Path(builder.__file__).resolve().parent),
                manifest_sha256=sha(root / 'manifest.json'),
                ma20_manifest_sha256=sha(root / builder.MANIFEST),
                bank_sha256=sha(root / builder.PLAN), builder_inputs=record['inputs'])


def freeze_reference(scripts):
    import ma4_f_adapter
    original = (REPO / 'tools/a800/phase5/e2e/taco_support.py').read_text()
    source = ma4_f_adapter.taco_source(original)
    functions = {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef)}
    preamble = 'import torch\nimport torch.distributed as dist\nfrom reference import roundtrip\n\n'
    (scripts / 'codec_reference.py').write_text(preamble + functions['masked_roundtrip'] +
        '\n\n@torch.no_grad()\n' + functions['check_consumer_output'] + '\n')
    shutil.copy2(REPO / 'tools/a800/phase4/reference.py', scripts / 'reference.py')


def prepare(args):
    root, build = args.root.resolve(), args.build.resolve()
    provenance, libs = identity(build), libraries(build)
    bank = read(build / 'ma20-plan.json')
    require(tuple(bank['configurations']) == ARMS, 'Wrong MA20 bank')
    root.mkdir(parents=True, exist_ok=False)
    scripts = root / 'scripts'; scripts.mkdir(); (root / 'results').mkdir()
    shutil.copy2(__file__, scripts / 'ma20_probe.py')
    shutil.copy2(HERE / 'ma5_bench.py', scripts / 'ma5_bench.py')
    freeze_reference(scripts)
    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex, build=str(build),
                build_identity=provenance, libraries=libs, bank=bank, arms=list(ARMS), roles=ROLES,
                timing=TIMING, cases=CASES, seeds=SEEDS, retained_output_calls=6,
                environment=ENVIRONMENT, schedule=schedule(), smoke_jobs=smoke_jobs(),
                source_files={str(p.relative_to(root)): sha(p) for p in scripts.iterdir()},
                training_metric='sum_role(max_rank_complete_forward_consumer_wall_us)',
                selection_rule='train_best_positive_then_same_candidate_heldout_ci95_lower_positive',
                hardware_requirement='single_node_tp8_132sm_h100', gpu_validated=False)
    write(root / 'plan.json', plan)
    print(root)


def verify(root, offline=False):
    root = Path(root).resolve(); plan = read(root / 'plan.json')
    require(plan['protocol'] == PROTOCOL and plan['arms'] == list(ARMS) and plan['roles'] == ROLES and
            plan['schedule'] == schedule() and plan['smoke_jobs'] == smoke_jobs() and
            plan['timing'] == TIMING and plan['cases'] == CASES and plan['seeds'] == SEEDS and
            plan['retained_output_calls'] == 6 and plan['environment'] == ENVIRONMENT,
            'Changed MA20 calibration protocol')
    require(tuple(plan['bank']['configurations']) == ARMS and set(plan['libraries']) == set(ARMS),
            'Incomplete MA20 bank coverage')
    for name, digest in plan['source_files'].items():
        require(sha(root / name) == digest, 'Frozen MA20 helper changed: ' + name)
    require(sha(__file__) == plan['source_files']['scripts/ma20_probe.py'], 'MA20 probe controller changed')
    if not offline:
        require(identity(plan['build'], plan['build_identity']['builder_root']) == plan['build_identity'],
                'MA20 calibration build changed')
        require(libraries(plan['build']) == plan['libraries'], 'MA20 calibration libraries changed')
        require(read(Path(plan['build']) / 'ma20-plan.json') == plan['bank'], 'MA20 plan changed')
    return plan


def environment(root, plan, arm, python, cuda_home):
    lib = Path(plan['libraries'][arm]['root'])
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('NCCL_', 'TORCH_NCCL_', 'H100_MA', 'H100_CR2_', 'H100_OP', 'H100_CLOSURE_'))}
    env.update(ENVIRONMENT, H100_MA20_PROBE_EXECUTE='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               PYTHONPATH=f'{root}/scripts:{lib}/python')
    for name in ('FLUX_FORCE_NVLINK', 'FLUX_TUNE_CONFIG_FILE', 'PYTHONOPTIMIZE'):
        env.pop(name, None)
    env['PATH'] = str(Path(python).parent) + ':' + env.get('PATH', '')
    ld = [str(lib / 'python/flux/lib'), str(lib), str(lib / 'build/lib')]
    if cuda_home:
        env['CUDA_HOME'] = str(cuda_home)
        env['PATH'] = str(Path(cuda_home) / 'bin') + ':' + env['PATH']
        ld.append(str(Path(cuda_home) / 'lib64'))
    env['LD_LIBRARY_PATH'] = ':'.join(ld) + ':' + env.get('LD_LIBRARY_PATH', '')
    return env


def mapping_evidence(root, plan):
    files = {}
    for arm in ARMS:
        for role in ROLES:
            path = Path(root) / 'results' / f'mapping-{arm}-{role}.csv'
            with path.open() as handle:
                rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(handle)]
            require(len(rows) == 4096 and {r['rank'] for r in rows} == set(range(8)), 'Incomplete compiled mapping')
            expected = plan['bank']['configurations'][arm]['roles'][role]['entry']['coords']
            for rank in range(8):
                rr = sorted((r for r in rows if r['rank'] == rank), key=lambda r: r['index'])
                require([r['index'] for r in rr] == list(range(512)) and
                        [r['m'] * 32 + r['n'] for r in rr] == expected[rank] and
                        all(r['destination'] == r['m'] // 2 and r['sm_occupancy'] > 0 for r in rr),
                        'Compiled MA20 mapping disagrees with mask/order plan')
            files[str(path.relative_to(root))] = sha(path)
    return files


def run(args):
    root = args.root.resolve(); plan = verify(root)
    require(type(args.worker_timeout) is int and args.worker_timeout > 0, 'Positive worker timeout required')
    python = args.python.expanduser().absolute()
    jobs = smoke_jobs() + schedule()
    prefix = [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
              str(root / 'scripts/ma20_probe.py'), 'worker', str(root)]
    if not args.execute:
        print(json.dumps(dict(dry_run=True, jobs=jobs, command=prefix), indent=2)); return
    write(root / 'started.json', dict(run_id=plan['run_id'], jobs=jobs))
    for arm in ARMS:
        library = plan['libraries'][arm]
        flux = next(p for p in library['files'] if Path(p).name == 'libflux_cuda.so')
        for role, (m, n, k) in ROLES.items():
            with (root / 'results' / f'mapping-{arm}-{role}.csv').open('x') as log:
                helpers.run_worker([library['mapping']['path'], str(m), str(n), str(k * 8), '8', flux],
                    root, environment(root, plan, arm, python, args.cuda_home), log, args.worker_timeout)
    mapping_evidence(root, plan)
    for job in jobs:
        verify(root)
        if job == schedule()[0]:
            collect(root, plan, smoke_jobs())
        dest = root / 'results' / job['directory']; dest.mkdir()
        command = prefix + ['--job', json.dumps(job, sort_keys=True)]
        write(dest / 'command.json', dict(command=command, job=job))
        with (dest / 'process.log').open('x') as log:
            helpers.run_worker(command, root, environment(root, plan, job['arm'], python, args.cuda_home),
                               log, args.worker_timeout)
    return select(root)


def validate_gpu_mask(value, record):
    require(isinstance(value, list) and len(value) == 2, 'Missing GPU mask readback for both slots')
    for slot, item in enumerate(value):
        require(item == dict(slot=slot, device_readback=True, dtype='torch.uint8', shape=[8, 512],
                            mask_sha256=record['mask_sha256'], selected_per_source=record['selected_per_source']),
                'GPU mask readback/count differs from frozen MA20 plan')


def hardware_identity(hardware):
    """Normalize common TP8 identity, ignoring optional diagnostic properties."""
    require(isinstance(hardware, dict) and hardware.get('host') and
            isinstance(hardware.get('devices'), list) and len(hardware['devices']) == 8,
            'Missing MA20 hardware identity')
    devices = [{key: device.get(key) for key in ('uuid', 'name', 'sm_count')}
               for device in hardware['devices']]
    require(len({device['uuid'] for device in devices}) == 8 and
            all(isinstance(device['uuid'], str) and device['uuid'] and
                isinstance(device['name'], str) and 'H100' in device['name'] and
                device['sm_count'] == 132 for device in devices),
            'MA20 hardware identity requires ordered eight full H100s')
    return dict(host=hardware['host'], devices=devices)


def validate_role(value, record, smoke):
    validate_gpu_mask(value.get('gpu_mask_before'), record)
    validate_gpu_mask(value.get('gpu_mask_after'), record)
    expected_cases = [(seed, case) for seed in SEEDS for case in CASES] if smoke else []
    require([(x['seed'], x['case']) for x in value['checks']] == expected_cases, 'Incomplete MA20 smoke cases')
    for check in value['checks']:
        helpers.validate_correctness(check['correctness'])
    retained = value['retained']
    require(len(retained) == (6 if smoke else 0), 'Incomplete retained output check')
    require(all(row['generation'] == i and row['passed'] is True and row['output'] == row['reference']
                for i, row in enumerate(retained)), 'MA20 slot reuse changed retained output')
    if smoke:
        require(len({row['pointer'] for row in retained}) == 6, 'Retained outputs alias')
    else:
        require(math.isfinite(value['wall_us']) and value['wall_us'] > 0, 'Invalid complete operator timing')
        helpers.validate_correctness(value['final_correctness'])
    states = value['consumer']
    calls = 14 if smoke else TIMING['warmup'] + TIMING['iterations']
    require(len(states) == 2 and sum(s['successful_calls'] for s in states) == calls and
            abs(states[0]['successful_calls'] - states[1]['successful_calls']) <= 1,
            'Wrong MA20 consumer call count')
    require(all(s['slot'] == i and s['protocol_version'] == 1 and s['failed'] is False and
                s['initializations'] == 1 and s['successful_calls'] == s['bias_residual_calls'] and
                s['bound_k'] == value['shape'][2] and s['local_packet_ptr'] > 0
                for i, s in enumerate(states)), 'MA20 fused-consumer fallback or failed slot')


def collect(root, plan, jobs):
    root = Path(root); files = {}; metrics = {}; hardware = software = None
    inputs = {}
    for job in jobs:
        rows = []
        for rank in range(8):
            path = root / 'results' / job['directory'] / f'rank{rank}.json'
            require(path.is_file(), 'Missing MA20 rank record: ' + str(path))
            row = read(path)
            require(row['protocol'] == PROTOCOL and row['run_id'] == plan['run_id'] and
                    row['plan_sha256'] == sha(root / 'plan.json') and row['job'] == job and
                    row['rank'] == rank and row['completed'] is True and
                    row['loaded_libraries'] == plan['libraries'][job['arm']]['files'] and
                    row['environment'] == ENVIRONMENT and set(row['roles']) == set(ROLES),
                    'Wrong MA20 rank identity, library or execution coverage')
            observed_hardware = hardware_identity(row['hardware'])
            hardware = observed_hardware if hardware is None else hardware
            software = row['software'] if software is None else software
            require(observed_hardware == hardware and row['software'] == software, 'MA20 hardware/software changed')
            for role, value in row['roles'].items():
                require(value['shape'] == ROLES[role], 'MA20 role shape changed')
                rec = plan['bank']['configurations'][job['arm']]['roles'][role]
                validate_role(value, rec, job['split'] == 'smoke')
                if job['split'] != 'smoke':
                    key = (job['block'], rank, role)
                    expected = inputs.setdefault(key, value['inputs'])
                    require(value['inputs'] == expected, 'MA20 arm inputs changed within paired block')
            files[str(path.relative_to(root))] = sha(path); rows.append(row)
        if job['split'] != 'smoke':
            metrics.setdefault(job['block'], {})[job['arm']] = sum(
                max(row['roles'][role]['wall_us'] for row in rows) for role in ROLES)
    return metrics, files


def choose(metrics):
    require(set(metrics) == set(range(14)) and all(set(v) == set(ARMS) for v in metrics.values()),
            'Selection needs all fourteen complete seven-arm blocks')
    require(all(math.isfinite(v) and v > 0 for values in metrics.values() for v in values.values()),
            'Invalid selection latency')
    train = {arm: helpers.paired([metrics[b][arm] for b in range(7)],
                                [metrics[b]['MA11'] for b in range(7)]) for arm in ARMS[1:]}
    chosen = max(ARMS[1:], key=lambda arm: (train[arm]['percent'], -ARMS.index(arm)))
    if train[chosen]['percent'] <= 0:
        return dict(candidate=None, proposed_candidate=None, training=train, heldout=None,
                    reason='no_positive_training_candidate')
    heldout = helpers.paired([metrics[b][chosen] for b in range(7, 14)],
                             [metrics[b]['MA11'] for b in range(7, 14)])
    accepted = heldout['percent'] > 0 and heldout['ci95'][0] > 0
    return dict(candidate=chosen if accepted else None, proposed_candidate=chosen,
                training=train, heldout=heldout,
                reason='independent_heldout_confirmed' if accepted else 'heldout_rejected_no_reselection')


def selection_result(root, offline=False):
    root = Path(root).resolve(); plan = verify(root, offline)
    maps = mapping_evidence(root, plan)
    # A single collection binds smoke, selection and held-out timing to the same
    # ordered physical GPUs and software. Separate calls would reset identity.
    metrics, raw = collect(root, plan, smoke_jobs() + schedule())
    first_path = Path('results') / schedule()[0]['directory'] / 'rank0.json'
    first = read(root / first_path)
    result = dict(protocol=PROTOCOL, completed=True, run_id=plan['run_id'],
                  plan_sha256=sha(root / 'plan.json'), bank_sha256=plan['build_identity']['bank_sha256'],
                  bundle_sha256=plan['build_identity']['bank_sha256'],
                  build_manifest_sha256=plan['build_identity']['ma20_manifest_sha256'],
                  **choose(metrics), metrics_us={str(k): v for k, v in metrics.items()},
                  evidence=dict(maps, **raw),
                  hardware=hardware_identity(first['hardware']), software=first['software'],
                  first_timing_record=dict(path=str(first_path), sha256=sha(root / first_path)),
                  metric=plan['training_metric'], fullstep_performance_validated=False)
    return result


def verify_selection(root, offline=False):
    result = selection_result(root, offline)
    require(read(Path(root) / 'selection.json') == result, 'Frozen MA20 selection differs from raw evidence')
    return result


def select(root, offline=False):
    root = Path(root).resolve()
    result = selection_result(root, offline)
    helpers.save_unchanged(root / 'selection.json', result)
    print(json.dumps({k: result[k] for k in ('candidate', 'proposed_candidate', 'reason', 'heldout')}, indent=2))
    return result


def worker(args):
    require(os.environ.get('H100_MA20_PROBE_EXECUTE') == '1', 'Use run --execute')
    root = args.root.resolve(); plan = verify(root); job = json.loads(args.job)
    require(job in smoke_jobs() + schedule(), 'Unregistered MA20 probe job')
    require(all(os.environ.get(k) == v for k, v in ENVIRONMENT.items()), 'Changed MA20 worker environment')
    import torch
    import torch.distributed as dist
    from codec_reference import check_consumer_output
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    require(world == 8 and rank == local and torch.cuda.device_count() == 8, 'Need local TP8')
    torch.cuda.set_device(local); torch.set_num_threads(1)
    cpus = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, {cpus[rank * len(cpus) // 8]})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    devices = []
    for i in range(8):
        p = torch.cuda.get_device_properties(i)
        require('H100' in p.name and (p.major, p.minor) == (9, 0) and p.multi_processor_count == 132,
                'Need eight full 132-SM H100s')
        devices.append(dict(name=p.name, uuid=str(p.uuid), sm_count=p.multi_processor_count))
    require(len({d['uuid'] for d in devices}) == 8 and
            all(torch.cuda.can_device_access_peer(i, j) for i in range(8) for j in range(8) if i != j),
            'Need eight unique peer-accessible H100s')
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10)); group = dist.group.WORLD
    import flux
    from flux.gemm_rs_taco import GemmRSTacoDoubleBuffered
    library = plan['libraries'][job['arm']]
    require(Path(flux.__file__).resolve().is_relative_to(Path(library['root'])), 'Wrong Flux Python loaded')
    mapped = {str(Path(line.split()[-1]).resolve()) for line in Path('/proc/self/maps').read_text().splitlines()
              if line.split() and line.split()[-1].startswith('/')}
    require(all(path in mapped and sha(path) == digest for path, digest in library['files'].items()),
            'Frozen MA20 native library was not loaded')
    flux.init_flux_shm(group)
    result = dict(protocol=PROTOCOL, run_id=plan['run_id'], plan_sha256=sha(root / 'plan.json'),
                  job=job, rank=rank, completed=False, loaded_libraries=library['files'],
                  environment=ENVIRONMENT, hardware=dict(host=os.uname().nodename, devices=devices),
                  software=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                nccl=list(torch.cuda.nccl.version()), python=sys.version), roles={})
    def tensor_hash(tensor):
        return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    def extra(shape, seed):
        generator = torch.Generator(device='cuda').manual_seed(seed)
        return ((torch.randn(shape[1], generator=generator, device='cuda') * .125).bfloat16(),
                (torch.randn(shape[0] // 8, shape[1], generator=generator, device='cuda') * .25).bfloat16())
    def gpu_masks(op, record):
        values = []
        expected = torch.tensor(record['entry']['mask'], dtype=torch.uint8)
        for index, slot in enumerate(op.slots):
            require(slot.selected.is_cuda and slot.selected.dtype == torch.uint8,
                    'MA20 mask is not a device uint8 tensor')
            cpu = slot.selected.detach().cpu().reshape(8, 512).contiguous()
            require(torch.equal(cpu, expected), 'Actual GPU mask differs from MA20 bank')
            values.append(dict(slot=index, device_readback=True, dtype=str(cpu.dtype), shape=list(cpu.shape),
                mask_sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                selected_per_source=[int(v) for v in cpu.sum(1).tolist()]))
        validate_gpu_mask(values, record); return values
    with torch.no_grad():
        for role_index, (role, shape) in enumerate(ROLES.items()):
            record = plan['bank']['configurations'][job['arm']]['roles'][role]
            kwargs = dict(placement='fused', selected=record['entry']['mask'], quantization_mode='targeted')
            op = GemmRSTacoDoubleBuffered(group, *shape, **kwargs)
            reference = GemmRSTacoDoubleBuffered(group, *shape, **kwargs) if job['split'] == 'smoke' else None
            mask = torch.tensor(record['entry']['mask'], dtype=torch.uint8)
            value = dict(shape=shape, gpu_mask_before=gpu_masks(op, record), checks=[], retained=[], wall_us=None)
            if job['split'] == 'smoke':
                for seed in SEEDS:
                    for case in CASES:
                        x, w = helpers.make_inputs(torch, shape, seed + rank + role_index * 10000, case)
                        bias, residual = extra(shape, seed + rank + 91)
                        y = op.forward_consumer(x, w, bias, residual)
                        check = check_consumer_output(x @ w.t(), y, group, mask, bias, residual)
                        value['checks'].append(dict(seed=seed, case=case, correctness=check))
                retained = []
                for generation in range(6):
                    x, w = helpers.make_inputs(torch, shape, 8171 + generation * 100 + rank, 'random')
                    bias, residual = extra(shape, 9911 + generation + rank)
                    if rank in (0, 7) and generation % 2 == rank // 7:
                        torch.cuda._sleep(2000000)
                    retained.append((x, w, bias, residual, op.forward_consumer(x, w, bias, residual)))
                torch.cuda.synchronize(); helpers.require_independent_outputs([v[-1] for v in retained])
                for generation, (x, w, bias, residual, y) in enumerate(retained):
                    expected = residual + (reference.forward(x, w) + bias)
                    left, right = tensor_hash(y), tensor_hash(expected)
                    require(left == right, 'MA20 retained output differs after slot reuse')
                    value['retained'].append(dict(generation=generation, passed=True, output=left,
                                                  reference=right, pointer=y.data_ptr()))
                del retained
            else:
                x, w = helpers.make_inputs(torch, shape, 18001 + 100 * job['block'] + rank + role_index * 10000, 'random')
                bias, residual = extra(shape, 19001 + job['block'] * 100 + rank)
                value['inputs'] = [tensor_hash(t) for t in (x, w, bias, residual)]
                for _ in range(TIMING['warmup']):
                    y = op.forward_consumer(x, w, bias, residual)
                torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(TIMING['iterations']):
                    y = op.forward_consumer(x, w, bias, residual)
                torch.cuda.synchronize()
                value['wall_us'] = (time.perf_counter() - started) * 1e6 / TIMING['iterations']
                value['final_correctness'] = check_consumer_output(x @ w.t(), y, group, mask, bias, residual)
            value['gpu_mask_after'] = gpu_masks(op, record)
            value['consumer'] = op.ma4_consumer_state()
            validate_role(value, record, job['split'] == 'smoke')
            result['roles'][role] = value
            torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()
            del op, reference
    result['completed'] = True
    write(root / 'results' / job['directory'] / f'rank{rank}.json', result)
    dist.barrier(); dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('root', type=Path); p.add_argument('--build', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('root', type=Path); p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--cuda-home', type=Path); p.add_argument('--worker-timeout', type=int, default=1800); p.add_argument('--execute', action='store_true')
    p = sub.add_parser('select'); p.add_argument('root', type=Path); p.add_argument('--offline', action='store_true')
    p = sub.add_parser('worker'); p.add_argument('root', type=Path); p.add_argument('--job', required=True)
    args = parser.parse_args(argv)
    if args.action == 'prepare': prepare(args)
    elif args.action == 'run': run(args)
    elif args.action == 'select': select(args.root, args.offline)
    else: worker(args)


if __name__ == '__main__':
    main()

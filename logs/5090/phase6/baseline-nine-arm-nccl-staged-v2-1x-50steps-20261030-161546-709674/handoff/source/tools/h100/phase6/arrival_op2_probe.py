"""Measure fixed production schedules with A's unchanged codec, then freeze one.

This is calibration, never full-step acceptance. Every timing window uses a new
eight-rank process group and the complete operator including A's output clone.
No profiler, arrival sampler, decoder replacement, or generated timing is used.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import uuid

import ablation4_runtime as communication
from calibrate import environment
from common import HERE, REPO, read, require, sha, verify_files, write
from scope_calibrate import startup
from scope_plan import hardware_identity, validate_hardware

PROTOCOL = 'h100-gpt67-tp8-arrival-op2-order-v2'
PROBE_PROTOCOL = 'h100-arrival-op2-samecodec-probe-v1'
A = 'interleaved_all_attn_mlp'
S = 'interleaved_pair_order_all_attn_mlp'
ROLES = {'attn': [2048, 4096, 512], 'mlp': [2048, 4096, 2048]}
CATALOG = (('A', 32), ('identity32', 32), ('g1', 1), ('g4', 4), ('g8', 8), ('g16', 16))
CANDIDATES = CATALOG[2:]
TIMING = dict(warmup=100, iterations=200, batches_per_worker=1)
BOOTSTRAP = dict(resamples=10000, seed=20261020, family_alpha=0.05,
                 comparisons=8, confidence=0.99375, unit='complete paired process block')
INPUT_SEED = 817
SOURCE_INPUTS = ('arrival_op2_probe.py', 'arrival_op2_build.py',
                 'arrival_op2_plan.py', 'arrival_op2_protocol.py',
                 'runtime.py', 'ablation4_runtime.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _builder():
    import arrival_op2_build
    return arrival_op2_build


def schedule():
    """Six complete Latin blocks followed by the exact reverse sequence."""
    labels = [label for label, _ in CATALOG]
    rows = [labels[i:]+labels[:i] for i in range(len(labels))]
    result = []
    for round_number, round_rows in enumerate((rows, [row[::-1] for row in rows[::-1]]), 1):
        for block, row in enumerate(round_rows):
            for position, label in enumerate(row):
                result.append(dict(round=round_number, block=block, position=position,
                                   index=len(result), label=label,
                                   directory=f'round{round_number}/block{block:02d}-{label}'))
    return result


def _runtime_identities(build_root, source_only):
    builder = _builder()
    root = Path(build_root).resolve()
    fused, improved = (read(root/name) for name in
                       ('parent-fused-manifest.json', 'improved/manifest.json'))
    result = {}
    for policy in (A, S):
        runtime = builder.runtime_root(root, policy).resolve()
        variant = (fused['variants']['interleaved_all'] if policy == A else
                   improved['variants'][runtime.name])
        result[policy] = dict(root=str(runtime), source_sha256=variant['source_sha256'],
                             libraries={} if source_only else {
                                 name: sha(runtime/name) for name in
                                 ('libflux_cuda.so', 'libflux_cuda_ths_op.so')})
    order_files = {'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp', 'phase6_host.hpp'}
    common = [{k: v for k, v in r['source_sha256'].items() if k not in order_files}
              for r in result.values()]
    require(common[0] and common[0] == common[1], 'Probe A/S compute, codec or frontend differs')
    return result


def reference_source():
    """Reuse the independently checked tensor codec without importing routing."""
    original = (REPO/'tools/a800/phase5/e2e/taco_support.py').read_text()
    before = original[:original.index('\nclass TacoForward:')]
    before = '\n'.join(line for line in before.splitlines()
                       if not line.startswith('from routing import'))+'\n'
    return before+original[original.index('\n@torch.no_grad()\ndef check_taco_output'):]


def worker_source(source_only=False):
    disabled = ("raise RuntimeError('Source-only probe worker is permanently non-executable')\n"
                if source_only else '')
    return disabled+'''"""Fresh process, complete-operator same-codec calibration worker."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/'config.json').read_text())
if (os.environ.get('H100_OP2_PROBE_EXECUTE') != '1' or PLAN['source_only'] or
        not PLAN['executable']):
    raise RuntimeError('Executable probe with explicit --execute required')
job = PLAN['schedule'][int(os.environ['H100_OP2_PROBE_JOB'])]
label = job['label']
span = dict(PLAN['catalog'])[label]
if any(os.environ.get(key) != str(span) for key in
       ('H100_OP2_ATTN_SPAN', 'H100_OP2_MLP_SPAN')):
    raise RuntimeError('Probe scheduler environment differs from the frozen candidate')
from communication import verify as verify_communication
comm = verify_communication(PLAN['communication'])
import torch
import torch.distributed as dist
rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
if world != 8 or rank != local:
    raise RuntimeError('Probe requires a complete one-node TP8 group')
torch.cuda.set_device(local)
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from runtime import check_topology
hardware = check_topology(world)
import flux
from flux.gemm_rs_taco import GemmRSTaco
from codec_reference import check_taco_output
dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10))
group = dist.new_group(list(range(world)))
policy = PLAN['A'] if label == 'A' else PLAN['S']
expected_runtime = PLAN['runtime_identities'][policy]
library = (Path(flux.__file__).parent/'lib/libflux_cuda.so').resolve()
library_hashes = {name: hashlib.sha256(library.with_name(name).read_bytes()).hexdigest()
                  for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
if (library.parent != (Path(expected_runtime['root'])/'python/flux/lib').resolve() and
        library != (Path(expected_runtime['root'])/'libflux_cuda.so').resolve()):
    raise RuntimeError('Unexpected isolated Flux runtime')
if library_hashes != expected_runtime['libraries']:
    raise RuntimeError('Probe loaded different library bytes')
def tensor_hash(value):
    data = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()
record = dict(protocol=PLAN['protocol'], probe_protocol=PLAN['probe_protocol'],
              run_id=PLAN['run_id'], plan_sha256=hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),
              rank=rank, world_size=world, job=job, policy=policy, span=span,
              hardware=hardware, communication=comm, library_hashes=library_hashes,
              profiling=False, sampling=False, timing=PLAN['timing'], completed=False,
              software=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                            nccl=list(torch.cuda.nccl.version())),
              results={})
for role_index, (role, shape) in enumerate(PLAN['roles'].items()):
    m, n, k_local = shape
    torch.manual_seed(PLAN['input_seed']+role_index*10000+rank)
    inputs = (torch.randn(m, k_local, device='cuda')*.02).to(torch.bfloat16)
    weights = (torch.randn(n, k_local, device='cuda')*.02).to(torch.bfloat16)
    inputs_sha256, weights_sha256 = tensor_hash(inputs), tensor_hash(weights)
    partial = inputs@weights.t()
    op = GemmRSTaco(group, m, n, k_local, placement='fused')
    first = op.forward(inputs, weights).clone()
    correctness = check_taco_output(partial, first, group, mask=None)
    first_sha256 = tensor_hash(first)
    repeat = op.forward(inputs, weights).clone()
    if not torch.equal(first, repeat):
        raise RuntimeError('Repeated same-input operator output differs')
    del partial, repeat
    for _ in range(PLAN['timing']['warmup']):
        output = op.forward(inputs, weights).clone()
    dist.barrier(group=group)
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    started = time.perf_counter()
    for _ in range(PLAN['timing']['iterations']):
        output = op.forward(inputs, weights).clone()
    end.record()
    end.synchronize()
    elapsed = time.perf_counter()-started
    final_sha256 = tensor_hash(output)
    if not torch.isfinite(output).all() or first_sha256 != final_sha256:
        raise RuntimeError('Probe output changed across the timing window')
    record['results'][role] = dict(shape=shape, inputs_sha256=inputs_sha256,
        weights_sha256=weights_sha256, output_sha256=first_sha256,
        final_output_sha256=final_sha256, repeated_bitwise_equal=True,
        correctness=correctness, wall_us=elapsed*1e6/PLAN['timing']['iterations'],
        gpu_us=begin.elapsed_time(end)*1000/PLAN['timing']['iterations'])
    dist.barrier(group=group)
    torch.cuda.synchronize()
    del op, inputs, weights, first, output
record['completed'] = True
output_dir = Path(os.environ['H100_OP2_PROBE_RESULT'])
with (output_dir/f'rank{rank}.json').open('x') as handle:
    json.dump(record, handle, indent=2)
    handle.write('\\n')
dist.destroy_process_group()
'''


def prepare(out, build_root, source_only=False):
    builder = _builder()
    build_root, out = Path(build_root).resolve(), Path(out).resolve()
    build = builder.verify(build_root, built=not source_only)
    require(build['protocol'] == PROTOCOL and build['source_only'] == source_only,
            'Probe requires a matching same-codec op2 build')
    out.mkdir(parents=True, exist_ok=False)
    (out/'scripts').mkdir()
    (out/'sources').mkdir()
    identities = _runtime_identities(build_root, source_only)
    for name in SOURCE_INPUTS:
        shutil.copy2(HERE/name, out/'sources'/name)
    for original, name in ((HERE/'runtime.py', 'runtime.py'),
                           (HERE/'ablation4_runtime.py', 'communication.py'),
                           (REPO/'tools/a800/phase4/reference.py', 'reference.py')):
        shutil.copy2(original, out/'scripts'/name)
    (out/'scripts/codec_reference.py').write_text(reference_source())
    (out/'scripts/worker.py').write_text(worker_source(source_only))
    shutil.copy2(build_root/builder.MANIFEST, out/'build-manifest.json')
    plan = dict(protocol=PROTOCOL, probe_protocol=PROBE_PROTOCOL, run_id=uuid.uuid4().hex,
                source_only=source_only, executable=not source_only,
                build_root=str(build_root), build_manifest_sha256=sha(build_root/builder.MANIFEST),
                config=build['config'], roles=copy.deepcopy(ROLES), input_seed=INPUT_SEED,
                A=A, S=S, catalog=list(CATALOG), schedule=schedule(), timing=dict(TIMING),
                bootstrap=dict(BOOTSTRAP), communication=communication.capture('0'),
                runtime_identities=identities, profiling=False, sampling=False)
    write(out/'scripts/config.json', plan)
    preparation = dict(protocol=PROTOCOL, probe_protocol=PROBE_PROTOCOL,
                       config_sha256=sha(out/'scripts/config.json'),
                       files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    write(out/'preparation.json', preparation)
    _verify_preparation(out)
    return out


def _verify_preparation(root, execute=False, offline=False):
    root = Path(root).resolve()
    prepared = read(root/'preparation.json')
    require(prepared['protocol'] == PROTOCOL and prepared['probe_protocol'] == PROBE_PROTOCOL,
            'Wrong probe preparation protocol')
    expected_files = ({'build-manifest.json'} |
                      {'sources/'+name for name in SOURCE_INPUTS} |
                      {'scripts/'+name for name in ('runtime.py', 'communication.py', 'reference.py',
                                                   'codec_reference.py', 'worker.py', 'config.json')})
    require(set(prepared['files']) == expected_files, 'Probe source snapshot set changed')
    verify_files(root, prepared['files'])
    require(sha(root/'scripts/config.json') == prepared['config_sha256'], 'Probe config changed')
    plan = read(root/'scripts/config.json')
    require(plan['protocol'] == PROTOCOL and plan['probe_protocol'] == PROBE_PROTOCOL and
            plan['A'] == A and plan['S'] == S and plan['roles'] == ROLES and
            plan['catalog'] == [list(item) for item in CATALOG] and
            plan['schedule'] == schedule() and plan['timing'] == TIMING and
            plan['bootstrap'] == BOOTSTRAP and plan['input_seed'] == INPUT_SEED and
            plan['profiling'] is False and plan['sampling'] is False and
            plan['executable'] == (not plan['source_only']), 'Probe protocol/configuration drift')
    require(plan['build_manifest_sha256'] == sha(root/'build-manifest.json'),
            'Probe build snapshot changed')
    require((root/'scripts/worker.py').read_text() == worker_source(plan['source_only']),
            'Probe worker differs from the reviewed protocol')
    if execute:
        require(plan['executable'] and not plan['source_only'], 'Source-only probe cannot execute')
    if not offline:
        builder = _builder()
        actual = Path(plan['build_root'])
        builder.verify(actual, built=execute)
        require(sha(actual/builder.MANIFEST) == plan['build_manifest_sha256'] and
                _runtime_identities(actual, plan['source_only']) == plan['runtime_identities'],
                'Probe runtime/build changed')
    return plan


def paired_statistics(log_ratios):
    require(len(log_ratios) == 12 and all(math.isfinite(x) for x in log_ratios),
            'Exactly twelve complete calibration blocks are required')
    gain = lambda values: 100*(1-math.exp(statistics.mean(values)))
    rng = random.Random(BOOTSTRAP['seed'])
    count = BOOTSTRAP['resamples']
    boot = sorted(gain([rng.choice(log_ratios) for _ in log_ratios]) for _ in range(count))
    alpha = BOOTSTRAP['family_alpha']/BOOTSTRAP['comparisons']
    lo = int(math.floor(alpha*count/2))
    hi = min(count-1, int(math.ceil((1-alpha/2)*count))-1)
    return dict(percent=gain(log_ratios), ci_family=[boot[lo], boot[hi]],
                confidence=1-alpha, blocks=len(log_ratios), log_ratios=list(log_ratios),
                geometric_time_ratio=math.exp(statistics.mean(log_ratios)))


def select(values):
    """Fixed family of eight comparisons; calibration never claims acceptance."""
    require(set(values) == set(ROLES), 'Both roles are mandatory')
    decisions, selected = {}, {}
    labels = [label for label, _ in CATALOG]
    for role in ROLES:
        data = values[role]
        require(set(data) == set(labels) and
                all(len(xs) == 12 and all(math.isfinite(x) and x > 0 for x in xs)
                    for xs in data.values()), 'Incomplete/invalid complete calibration blocks')
        comparisons = {}
        for label, _ in CATALOG[1:]:
            comparisons[label] = paired_statistics([math.log(x/y) for x, y in zip(data[label], data['A'])])
        admitted = [(label, span) for label, span in CANDIDATES
                    if comparisons[label]['ci_family'][0] > 0]
        winner = min(admitted, key=lambda item: comparisons[item[0]]['geometric_time_ratio']) if admitted else ('A', 32)
        selected[role] = winner[1]
        decisions[role] = dict(label=winner[0], span=winner[1],
                               reason='positive_family_interval' if admitted else 'retain_A_no_admitted_candidate',
                               admitted=[label for label, _ in admitted], comparisons=comparisons)
    return dict(selected_spans=selected, decisions=decisions)


def _records(root, plan):
    """Validate every rank/process, then reduce only whole process blocks."""
    root = Path(root).resolve()
    values = {role: {label: [] for label, _ in CATALOG} for role in ROLES}
    first_hardware = software = None
    expected_outputs, expected_inputs = {}, {}
    files = {}
    # A appears first in the frozen first block, fixing reference hashes.
    for job in plan['schedule']:
        rows = []
        for rank in range(8):
            path = root/'results'/job['directory']/f'rank{rank}.json'
            row = read(path)
            files[str(path.relative_to(root))] = sha(path)
            policy = A if job['label'] == 'A' else S
            require(row['completed'] and row['protocol'] == PROTOCOL and
                    row['probe_protocol'] == PROBE_PROTOCOL and row['run_id'] == plan['run_id'] and
                    row['plan_sha256'] == sha(Path(root)/'scripts/config.json') and
                    row['rank'] == rank and row['world_size'] == 8 and row['job'] == job and
                    row['policy'] == policy and row['span'] == dict(CATALOG)[job['label']] and
                    row['timing'] == TIMING and row['profiling'] is False and row['sampling'] is False and
                    row['communication'] == plan['communication'] and
                    row['library_hashes'] == plan['runtime_identities'][policy]['libraries'],
                    'Incomplete or mismatched probe worker')
            validate_hardware(row['hardware'])
            require(row['hardware']['local_rank'] == rank, 'Probe rank/device mismatch')
            hardware = hardware_identity(row['hardware'])
            if first_hardware is None:
                first_hardware, software = hardware, row['software']
            require(hardware == first_hardware and row['software'] == software,
                    'Mixed probe devices or software')
            require(set(row['results']) == set(ROLES), 'Missing probe role')
            for role, shape in ROLES.items():
                sample = row['results'][role]
                require(sample['shape'] == shape and sample['repeated_bitwise_equal'] is True and
                        sample['output_sha256'] == sample['final_output_sha256'] and
                        all(math.isfinite(sample[k]) and sample[k] > 0 for k in ('wall_us', 'gpu_us')),
                        'Invalid operator timing/output')
                correctness = sample['correctness']
                require(correctness['passed'] is True and
                        all(math.isfinite(correctness[k]) and correctness[k] >= 0
                            for k in ('max_abs', 'relative_l2', 'protocol_relative_l2')) and
                        correctness['relative_l2'] <= .05 and correctness['protocol_relative_l2'] <= .01,
                        'Operator codec-reference correctness failed')
                for name in ('inputs_sha256', 'weights_sha256', 'output_sha256'):
                    require(isinstance(sample[name], str) and len(sample[name]) == 64 and
                            all(c in '0123456789abcdef' for c in sample[name]), 'Invalid tensor hash')
                key = (role, rank)
                inputs = (sample['inputs_sha256'], sample['weights_sha256'])
                require(expected_inputs.setdefault(key, inputs) == inputs, 'Probe input drift')
                require(expected_outputs.setdefault(key, sample['output_sha256']) == sample['output_sha256'],
                        'Scheduler output is not bitwise equal to A')
            rows.append(row)
        for role in ROLES:
            values[role][job['label']].append(max(row['results'][role]['wall_us'] for row in rows))
    actual = {str(p.relative_to(root)) for p in (Path(root)/'results').glob('**/rank*.json')}
    require(actual == set(files), 'Missing or extra probe rank records')
    start = read(Path(root)/'start-gpu-check.json')
    require(start['passed'] and start['run_id'] == plan['run_id'] and start['host'] == first_hardware['host'] and
            start['uuids'] == [d['uuid'].removeprefix('GPU-') for d in first_hardware['devices']],
            'Probe startup/actual device order differs')
    return dict(values=values, hardware=first_hardware, software=software,
                files=files, output_hashes={f'{role}/rank{rank}': value
                                           for (role, rank), value in expected_outputs.items()})


def _selection(root, plan):
    measured = _records(root, plan)
    chosen = select(measured['values'])
    evidence = {**measured['files'], 'preparation.json': sha(Path(root)/'preparation.json'),
                'start-gpu-check.json': sha(Path(root)/'start-gpu-check.json'),
                'execution.json': sha(Path(root)/'execution.json'),
                **read(Path(root)/'preparation.json')['files']}
    record = dict(protocol=PROTOCOL, probe_protocol=PROBE_PROTOCOL,
                  run_id=plan['run_id'], source_only=False, executable=True, gpu_measured=True,
                  status='frozen_for_independent_full_step_confirmation',
                  selection_scope='complete_operator_with_output_clone',
                  build_manifest_sha256=plan['build_manifest_sha256'],
                  runtime_identities=plan['runtime_identities'], config=plan['config'],
                  configuration_sha256=digest(plan), hardware=measured['hardware'],
                  software=measured['software'], timing=dict(TIMING), bootstrap=dict(BOOTSTRAP),
                  values=measured['values'], output_hashes=measured['output_hashes'],
                  evidence_files=evidence, **chosen,
                  limitation='Calibration only; operator gains do not establish a 1.5% full-step gain. '
                             'Freeze these choices before independent full-step confirmation; do not retune on it.')
    record['selection_sha256'] = digest(record)
    return record


def run(out, execute=False, python=Path(sys.executable), cuda_home=None):
    root = Path(out).resolve()
    plan = _verify_preparation(root, execute=execute)
    command = [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
               str(root/'scripts/worker.py')]
    if not execute:
        print('DRY RUN: 72 fresh TP8 workers, 12 complete six-policy blocks; no GPU query or launch.')
        print(command)
        return dict(command=command, schedule=plan['schedule'])
    import fcntl
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (root/'execution.json').exists() and not (root/'selection.json').exists(),
                'Probe already started; use a new directory, without resuming partial selection')
        write(root/'execution.json', dict(status='running', run_id=plan['run_id']))
        try:
            startup(root, plan['run_id'])
            for job in plan['schedule']:
                policy = A if job['label'] == 'A' else S
                library = Path(plan['runtime_identities'][policy]['root'])
                env = communication.apply(environment(python, library, plan['config'],
                                                      plan['run_id'], root/'scripts', cuda_home),
                                          plan['communication'])
                span = dict(CATALOG)[job['label']]
                dest = root/'results'/job['directory']
                dest.mkdir(parents=True, exist_ok=False)
                env.update(H100_OP2_PROBE_EXECUTE='1', H100_OP2_PROBE_JOB=str(job['index']),
                           H100_OP2_PROBE_RESULT=str(dest),
                           H100_OP2_ATTN_SPAN=str(span), H100_OP2_MLP_SPAN=str(span))
                with (dest/'process.log').open('x') as output:
                    subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
            _verify_preparation(root, execute=True)
            _records(root, plan)
            write(root/'execution.json', dict(status='completed', run_id=plan['run_id']))
            selection = _selection(root, plan)
            with (root/'selection.json').open('x') as output:
                json.dump(selection, output, indent=2)
                output.write('\n')
            return selection
        except BaseException:
            write(root/'execution.json', dict(status='failed', run_id=plan['run_id']))
            raise


def make_source_only_selection(path, build_root, selected_spans=None):
    """Explicit CPU fixture; no synthetic latency or measured-gain fields."""
    path, build_root = Path(path).resolve(), Path(build_root).resolve()
    builder = _builder()
    build = builder.verify(build_root)
    require(build['source_only'], 'Diagnostic selection requires a source-only build')
    spans = dict(attn=4, mlp=8) if selected_spans is None else selected_spans
    require(set(spans) == set(ROLES) and all(type(v) is int and v in (1, 4, 8, 16, 32)
                                           for v in spans.values()), 'Invalid diagnostic spans')
    record = dict(protocol=PROTOCOL, probe_protocol=PROBE_PROTOCOL, source_only=True,
                  executable=False, gpu_measured=False, status='cpu_fixture_never_executable',
                  run_id='cpu-fixture', selected_spans=dict(spans), config=build['config'],
                  build_manifest_sha256=sha(build_root/builder.MANIFEST),
                  runtime_identities=_runtime_identities(build_root, True), hardware={},
                  selection_scope='diagnostic_source_only', evidence_files={})
    record['selection_sha256'] = digest(record)
    with path.open('x') as output:
        json.dump(record, output, indent=2)
        output.write('\n')
    return record


def selection_verify(path, build_root, offline=False):
    path, build_root = Path(path).resolve(), Path(build_root).resolve()
    record = read(path)
    require(record['protocol'] == PROTOCOL and record['probe_protocol'] == PROBE_PROTOCOL and
            record['selection_sha256'] == digest({k: v for k, v in record.items() if k != 'selection_sha256'}),
            'Selection protocol/digest changed')
    require(set(record['selected_spans']) == set(ROLES) and
            all(type(v) is int and v in (1, 4, 8, 16, 32) for v in record['selected_spans'].values()),
            'Invalid selected scheduler spans')
    builder = _builder()
    require(sha(build_root/builder.MANIFEST) == record['build_manifest_sha256'],
            'Selection belongs to another build')
    build = read(build_root/builder.MANIFEST) if offline else builder.verify(
        build_root, built=not record['source_only'])
    require(build['protocol'] == PROTOCOL and build['config'] == record['config'] and
            build['source_only'] == record['source_only'], 'Selection build/configuration differs')
    if record['source_only']:
        require(record['executable'] is False and record['gpu_measured'] is False and
                record['status'] == 'cpu_fixture_never_executable' and
                record['selection_scope'] == 'diagnostic_source_only' and
                record['evidence_files'] == {} and record['hardware'] == {},
                'CPU fixture cannot be promoted to measured selection')
    else:
        require(record['executable'] is True and record['gpu_measured'] is True,
                'Measured GPU calibration required')
        root = path.parent
        verify_files(root, record['evidence_files'])
        plan = _verify_preparation(root, execute=not offline, offline=offline)
        require(read(root/'execution.json') == dict(status='completed', run_id=record['run_id']),
                'Probe did not complete')
        require(_selection(root, plan) == record, 'Selection differs from all-sample independent replay')
    if not offline:
        require(_runtime_identities(build_root, record['source_only']) == record['runtime_identities'],
                'Selected scheduler runtime/source differs')
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('out', type=Path)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--source-only', action='store_true')
    p = sub.add_parser('run')
    p.add_argument('out', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--cuda-home', type=Path)
    p = sub.add_parser('verify')
    p.add_argument('selection', type=Path)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare(args.out, args.build, args.source_only))
    elif args.action == 'run':
        run(args.out, args.execute, args.python, args.cuda_home)
    else:
        result = selection_verify(args.selection, args.build, args.offline)
        print(json.dumps(dict(status=result['status'], selected_spans=result['selected_spans'],
                              gpu_measured=result['gpu_measured']), indent=2))


if __name__ == '__main__':
    main()

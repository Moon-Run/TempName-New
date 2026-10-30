"""Explicitly gated BF16 interleaved calibration for the exact selected roles."""
import csv
import hashlib
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import uuid

import calibrate
import plan as fitter
import scope_build
from common import HERE, REPO, read, require, replace_once, sha, verify_files, write
from scope_protocol import PROTOCOL, SHAPES
from scope_plan import model_config, role_record, validate, validate_hardware, hardware_identity


def startup(root, run_id):
    """One fail-fast startup snapshot; new calibration may use a new H100 host."""
    result = dict(run_id=run_id, host=socket.gethostname(), passed=False, monitoring='before_suite_only')
    try:
        result['gpu_csv'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                                                     '--format=csv,noheader,nounits'], text=True, timeout=30)
        result['apps_csv'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                                      '--format=csv,noheader,nounits'], text=True, timeout=30)
        rows = [[v.strip() for v in row] for row in csv.reader(result['gpu_csv'].splitlines())]
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            rows = [next((r for r in rows if key.strip() in (r[0], r[1])), [])
                    for key in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        require(len(rows) == 8 and all(rows) and len({r[1] for r in rows}) == 8 and
                all('H100' in r[2] and int(r[3]) <= 1024 and int(r[4]) <= 5 for r in rows) and
                not result['apps_csv'].strip(), 'Eight idle H100s required; no waiting/retries')
        result['uuids'] = [r[1].removeprefix('GPU-') for r in rows]; result['passed'] = True
    finally:
        write(root/'start-gpu-check.json', result)
    return result


def worker_source(source_only=False):
    source = calibrate.worker_source()
    gate = "if os.environ.get('H100_SCOPE_EXECUTE') != '1':\n raise RuntimeError('Explicit scope --execute required')\n"
    source = replace_once(source, 'world=int(os.environ', gate+'world=int(os.environ')
    source = replace_once(source, "ROWS,FRAGS,FIELDS=2048,16,8;SLOTS=ROWS*FRAGS", '''import hashlib
report.update(protocol=config['protocol'],roles=config['roles'],model_config=config['model_config'],sampling=True,
              library_hashes={name:hashlib.sha256(library.with_name(name).read_bytes()).hexdigest()
                              for name in ('libflux_cuda.so','libflux_cuda_ths_op.so')})
ROWS,FRAGS,FIELDS=2048,16,8;SLOTS=ROWS*FRAGS''')
    if source_only:
        source = "raise RuntimeError('Source-only calibration worker is permanently disabled')\n"+source
    return source


def prepare(out, build_root, source_only=False):
    build_root = Path(build_root).resolve()
    m = scope_build.verify(build_root, built=not source_only, kind='scope_sampler')
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'controllers').mkdir()
    for path in HERE.glob('*.py'):
        if not path.name.startswith('test_'):
            shutil.copy2(path, out/'controllers'/path.name)
    (out/'worker.py').write_text(worker_source(source_only))
    shutil.copy2(HERE/'runtime.py', out/'runtime.py')
    roles = m['calibration_roles']
    write(out/'config.json', dict(protocol=PROTOCOL, model_config=model_config(), roles=roles,
                                  shapes=[SHAPES[r] for r in roles], repeats=0, warmup=100,
                                  trials=3, iters=50, sample_warmup=10))
    run_id = uuid.uuid4().hex
    write(out/'preparation.json', dict(source_only=source_only, run_id=run_id))
    shutil.copy2(build_root/'manifest.json', out/'build-manifest.json')
    record = dict(protocol=PROTOCOL, run_id=run_id, build=str(build_root), build_sha256=sha(build_root/'manifest.json'),
                  roles=roles, config=model_config(), source_only=source_only, executable=not source_only,
                  files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()},
                  controllers={str(p): sha(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')})
    write(out/'plan.json', record)
    return out


def verify(root, execute=False):
    root = Path(root); p = read(root/'plan.json')
    require(p['protocol'] == PROTOCOL and p['config'] == model_config(), 'Wrong calibration protocol/model')
    verify_files(root, p['files'])
    require(read(root/'preparation.json') == dict(source_only=p['source_only'], run_id=p['run_id']) and
            p['executable'] == (not p['source_only']), 'Calibration rehearsal mode changed')
    require(sha(Path(p['build'])/'manifest.json') == p['build_sha256'], 'Sampler build changed')
    scope_build.verify(p['build'], built=execute, kind='scope_sampler')
    if execute:
        require(p['executable'] and not p['source_only'], 'Source-only calibration cannot execute')
        verify_files('/', p['controllers'])
    return p


def run(root, execute=False, python=Path(sys.executable), cuda_home=None):
    root = Path(root).resolve(); p = verify(root)
    lib = Path(p['build'])/'scope-sampler'
    mappings = {role: [str(lib/'check_mapping'), *map(str, SHAPES[role]), '8', str(lib/'libflux_cuda.so')]
                for role in p['roles']}
    cmd = [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', str(root/'worker.py')]
    if not execute:
        print('DRY RUN calibration: no GPU query, mapping or worker; roles', p['roles'])
        print(mappings); print(cmd)
        return mappings, cmd
    verify(root, execute=True)
    import fcntl
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (root/'execution.json').exists(), 'Calibration already started; use a new directory')
        write(root/'execution.json', dict(status='running'))
        try:
            start = startup(root, p['run_id'])
            env = calibrate.environment(python, lib, p['config'], p['run_id'], root, cuda_home)
            env.update(H100_SCOPE_EXECUTE='1', MECH_POLICY='interleaved', MECH_KIND='instrumented', MECH_BLOCK='0',
                       MECH_LIBRARY=str(lib/'libflux_cuda.so'), RESULT_DIR=str(root))
            write(root/'command.json', dict(mappings=mappings, worker=cmd, roles=p['roles']))
            for role, mapping in mappings.items():
                with (root/f'mapping-{role}.csv').open('x') as out:
                    subprocess.run(mapping, env=env, stdout=out, check=True)
            with (root/'process.log').open('x') as out:
                subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
            rows = calibration_records(root, p)
            hardware = rows[0]['hardware']
            require(start['host'] == hardware['host'] and start['uuids'] ==
                    [d['uuid'].removeprefix('GPU-') for d in hardware['devices']], 'Startup/actual device order differs')
            write(root/'calibration-complete.json', dict(protocol=PROTOCOL, completed=True, gpu_calibrated=True,
                  run_id=p['run_id'], roles=p['roles'], hardware=hardware, preparation_sha256=sha(root/'plan.json')))
            write(root/'execution.json', dict(status='completed'))
        except BaseException:
            write(root/'execution.json', dict(status='failed')); raise


def calibration_records(root, p):
    rows = [read(root/f'b0-interleaved-instrumented-rank{r}.json') for r in range(8)]
    reference = hardware_identity(rows[0]['hardware'])
    lib = Path(p['build'])/'scope-sampler'
    hashes = {n: sha(lib/n) for n in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}
    for rank, row in enumerate(rows):
        require(row['completed'] and row['protocol'] == PROTOCOL and row['run_id'] == p['run_id'] and
                row['roles'] == p['roles'] and row['model_config'] == p['config'] and
                row['rank'] == rank and row['world_size'] == 8 and row['policy'] == 'interleaved' and
                row['kind'] == 'instrumented' and row['sampling'] and not row['smoke'], 'Incomplete/mismatched calibration rank')
        require(row['hardware']['local_rank'] == rank and hardware_identity(row['hardware']) == reference and
                row['library_hashes'] == hashes, 'Mixed devices/sampler binaries')
        validate_hardware(row['hardware'])
        require([ [s['M'], s['N'], s['K_global']] for s in row['shapes']] == [SHAPES[r] for r in p['roles']], 'Wrong calibration full shapes')
        for s in row['shapes']:
            require(s['correctness'] and s['fragment_count'] == 8 and s['partition_tiles'] == 64 and
                    'flux_bf16' in s['kernel']['name'] and 'sm80' in s['kernel']['name'].lower(), 'Calibration must execute V2 BF16')
            for mode in ('off', 'receiver_sparse'):
                samples = [x for x in s['samples'] if x['mode'] == mode]
                require(len(samples) == 192 and {x['repeat'] for x in samples} == set(range(192)) and
                        all(x['correctness'] for x in samples), 'Missing pass/tile calibration observations')
    return rows


def fit(root, out, arrival_order='legacy_tail', arrival_options=None):
    root, out = Path(root).resolve(), Path(out).resolve(); p = verify(root)
    require(not p['source_only'] and p['executable'], 'CPU rehearsal cannot produce measured plans')
    verify_files('/', p['controllers'])
    done = read(root/'calibration-complete.json')
    require(done['protocol'] == PROTOCOL and done['completed'] and done['gpu_calibrated'] and
            done['run_id'] == p['run_id'] and done['preparation_sha256'] == sha(root/'plan.json'), 'Completed GPU calibration required')
    rows = calibration_records(root, p)
    require(hardware_identity(done['hardware']) == hardware_identity(rows[0]['hardware']), 'Completion hardware changed')
    inputs = {str(f): sha(f) for f in root.rglob('*') if f.is_file() and f.suffix != '.lock'}
    for f in (HERE/'scope_calibrate.py', HERE/'plan.py', REPO/'tools/a800/arrival/plan.py', REPO/'tools/a800/phase5/plan.py'):
        inputs[str(f)] = sha(f)
    if arrival_order in ('tail_residual', 'tail_assignment'):
        inputs[str(HERE/'arrival_tail.py')] = sha(HERE/'arrival_tail.py')
    if arrival_order == 'tail_assignment':
        inputs[str(HERE/'arrival_assignment.py')] = sha(HERE/'arrival_assignment.py')
    results = {}
    for role in p['roles']:
        with (root/f'mapping-{role}.csv').open() as f:
            mapping = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]
        entry = fitter.fit(rows, mapping, p['config'], 'interleaved', SHAPES[role],
                           arrival_order=arrival_order, arrival_options=arrival_options)
        record = role_record(role, entry, done['hardware'], inputs,
                             dict(kind='scope_bf16_calibration', source=str(root), run_id=p['run_id'],
                                  sampler_manifest_sha256=p['build_sha256'], role=role, shape=SHAPES[role]))
        validate(record, role); results[role] = record
    out.mkdir(parents=True, exist_ok=False)
    for role, record in results.items():
        write(out/f'{role}.json', record)
    return out

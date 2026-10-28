"""MA9: exact registered CR3/F4 control and optimizer zero-count helper arm."""
import argparse
import copy
import hashlib
from pathlib import Path
import shutil

import build
import ma4_f_build as baseline
from ma2_ablation_build import copy_registered
import ma9_sources
from native_exports import verify_native_exports
from common import HERE, read, require, sha, verify_files, write
from scope_plan import model_config

PROTOCOL = 'h100-ma9-f4-zcount-v1'
MANIFEST = 'ma9-manifest.json'
VARIANTS = ma9_sources.VARIANTS
RUNTIME = baseline.RUNTIME
C_POLICY = baseline.C_POLICY
INPUTS = tuple(dict.fromkeys(('ma9_build.py', 'ma9_sources.py', 'ma9_adapter.py',
                             'ma9_zero_count.cuh', 'native_exports.py', *baseline.INPUTS)))
canonical = baseline.canonical


def prepare(out, tuning=None):
    tuning = ma9_sources.tuning_config(tuning)
    out = Path(out).resolve()
    require(not out.exists(), 'Use a fresh MA9 build')
    out.mkdir(parents=True)
    reference = baseline.prepare(out / 'reference')
    old = read(reference / 'manifest.json')
    frozen = old['variants']['F4']['source_sha256']
    updates = ma9_sources.sources(reference / 'F4')
    variants = {}
    for arm in VARIANTS:
        copy_registered(reference / 'F4', out / arm, frozen)
        if arm == 'F4-zcount':
            for name, source in updates.items():
                target = out / arm / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source)
        item = copy.deepcopy(old['variants']['F4'])
        names = set(frozen) | (set(updates) if arm == 'F4-zcount' else set())
        item.update(python=arm + '/python', source_sha256={name: sha(out / arm / name) for name in sorted(names)})
        variants[arm] = item
    for name in ('selection-plan.json', 'config.json', 'TACO_LICENSE.txt'):
        shutil.copy2(reference / name, out / name)
    write(out / 'tuning.json', tuning)
    inputs = {str(HERE / name): sha(HERE / name) for name in INPUTS}
    files = {name: sha(out / name) for name in
             ('selection-plan.json', 'config.json', 'TACO_LICENSE.txt', 'tuning.json')}
    info = dict(protocol=build.PROTOCOL, arch='sm90', kind='baselines', status='prepared',
                config=model_config(), gpu_validated=False, source_only=False, diagnostic_only=False,
                submodules=old['submodules'], source_commit=old.get('source_commit'), inputs=inputs,
                files=files, variants=variants)
    write(out / 'manifest.json', info)
    record = dict(protocol=PROTOCOL, tuning=tuning, inputs=inputs, files=files,
                  control_native_sources_unchanged=True, projection_sources_unchanged=True,
                  candidate_native_addition='optimizer_zero_count', baseline_source_only=False, gpu_validated=False,
                  reference_protocol=baseline.PROTOCOL, baseline_variants={'F4': frozen},
                  variants={arm: item['source_sha256'] for arm, item in variants.items()},
                  build_manifest_sha256=sha(out / 'manifest.json'))
    write(out / MANIFEST, record)
    verify(out)
    return out


def verify(root, built=False):
    root = Path(root).resolve()
    record, info = read(root / MANIFEST), read(root / 'manifest.json')
    require(record['protocol'] == PROTOCOL and record['reference_protocol'] == baseline.PROTOCOL,
            'Wrong MA9 source protocol')
    require(record.get('control_native_sources_unchanged') is True and
            record.get('projection_sources_unchanged') is True and
            record.get('candidate_native_addition') == 'optimizer_zero_count',
            'MA9 must retain unchanged F4 control and projection code')
    verify_files('/', record['inputs'])
    verify_files(root, record['files'])
    require(record['inputs'] == {str(HERE / name): sha(HERE / name) for name in INPUTS},
            'MA9 builder or autograd implementation changed')
    require(sha(root / 'manifest.json') == record['build_manifest_sha256'], 'MA9 build manifest changed')
    require(info['protocol'] == build.PROTOCOL and info['arch'] == 'sm90'
            and info['config'] == model_config() and not info['source_only'] and not info['diagnostic_only']
            and tuple(info['variants']) == VARIANTS and tuple(record['variants']) == VARIANTS
            and info['inputs'] == record['inputs'], 'Wrong MA9 build configuration')
    # This includes compiled libraries and Python extensions after compilation;
    # the pre-build record intentionally lists only immutable configuration.
    verify_files(root, info['files'])
    baseline.verify(root / 'reference')
    ref = read(root / 'reference/manifest.json')
    frozen = ref['variants']['F4']['source_sha256']
    require(record['baseline_variants'] == {'F4': frozen}, 'Frozen F4 changed')
    require(read(root / 'selection-plan.json') == read(root / 'reference/selection-plan.json'),
            'C masks/order changed')
    tuning = ma9_sources.tuning_config(record['tuning'])
    require(read(root / 'tuning.json') == tuning, 'MA9 tuning identity changed')
    updates = ma9_sources.sources(root / 'reference/F4')
    expected = {'F4': frozen, 'F4-zcount': dict(frozen, **{
        name: hashlib.sha256(source.encode()).hexdigest() for name, source in updates.items()})}
    for arm in VARIANTS:
        hashes = info['variants'][arm]['source_sha256']
        require(hashes == record['variants'][arm] == expected[arm],
                arm + ' differs from registered MA9 transform')
        verify_files(root / arm, hashes)
        if built:
            require(all((root / arm / name).is_file() for name in
                        ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping')),
                    'Missing MA9 runtime: ' + arm)
            artifacts = [root / arm / 'python/flux/lib' / name for name in
                         ('libflux_cuda.so', 'libflux_cuda_ths_op.so')]
            artifacts += sorted((root / arm / 'python').glob('flux_ths_pybind*.so'))
            artifacts += [root / arm / 'check_mapping']
            require(len(artifacts) >= 4 and all(path.is_file() and
                    str(path.relative_to(root)) in info['files'] for path in artifacts),
                    'MA9 runtime lacks recorded binary identity: ' + arm)
            helpers = ma9_sources.NATIVE_EXPORTS
            verify_native_exports(root / arm / 'python/flux/lib/libflux_cuda.so',
                required=('taco_ma4f_protocol_version', 'taco_ma4f_decode_reduce') +
                         (helpers if arm == 'F4-zcount' else ()),
                forbidden=helpers if arm == 'F4' else ())
    if built:
        require(info['status'] == 'built', 'Unbuilt MA9 experiment')
    return record


def compile_build(root, cuda_home, jobs=2, nccl_root=None):
    require(type(jobs) is int and jobs > 0, 'Positive jobs required')
    root = Path(root).resolve()
    verify(root)
    try:
        build.compile_build(root, Path(cuda_home), jobs,
                            None if nccl_root is None else Path(nccl_root))
    finally:
        record = read(root / MANIFEST)
        record['build_manifest_sha256'] = sha(root / 'manifest.json')
        write(root / MANIFEST, record)
    verify(root, built=True)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('verify')
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--built', action='store_true')
    p = sub.add_parser('build')
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--cuda-home', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--nccl-root', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        print(prepare(args.out))
    elif args.action == 'verify':
        verify(args.build, args.built)
    else:
        compile_build(args.build, args.cuda_home, args.jobs, args.nccl_root)


if __name__ == '__main__':
    main()

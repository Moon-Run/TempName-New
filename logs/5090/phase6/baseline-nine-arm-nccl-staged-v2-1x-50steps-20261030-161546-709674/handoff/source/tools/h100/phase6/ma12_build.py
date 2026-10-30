"""MA12: exact MA89 control versus MA89 plus tile-scoped publication."""
import argparse
import copy
import hashlib
from pathlib import Path
import shutil

import build
import ma4_f_build as baseline
from ma2_ablation_build import copy_registered
import ma12_sources
import ma9_sources
from native_exports import verify_native_exports
from common import HERE, read, require, sha, verify_files, write
from scope_plan import model_config

PROTOCOL = 'h100-ma12-tile-publication-v1'
MANIFEST = 'ma12-manifest.json'
VARIANTS = ma12_sources.VARIANTS
RUNTIME = baseline.RUNTIME
C_POLICY = baseline.C_POLICY
INPUTS = tuple(dict.fromkeys(('ma12_build.py', 'ma12_sources.py', 'ma12_tile_ready.cuh', 'ma8_adapter.py', 'ma9_adapter.py', 'ma9_sources.py', 'ma9_bench.py',
                             'ma9_zero_count.cuh', 'native_exports.py', *baseline.INPUTS)))
canonical = baseline.canonical


def prepare(out, tuning=None):
    tuning = ma12_sources.tuning_config(tuning)
    out = Path(out).resolve()
    require(not out.exists(), 'Use a fresh MA12 build')
    out.mkdir(parents=True)
    reference = baseline.prepare(out / 'reference')
    old = read(reference / 'manifest.json')
    frozen = old['variants']['F4']['source_sha256']
    updates = {VARIANTS[0]: ma9_sources.sources(reference / 'F4')}
    variants = {}
    for arm in VARIANTS:
        if arm == VARIANTS[1]:
            updates[arm] = dict(updates[VARIANTS[0]], **ma12_sources.sources(out / VARIANTS[0]))
        copy_registered(reference / 'F4', out / arm, frozen)
        for name, source in updates[arm].items():
            target = out / arm / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        item = copy.deepcopy(old['variants']['F4'])
        names = set(frozen) | set(updates[arm])
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
                  activation_policy='ma8_v2_cache_in_both_arms',
                  zero_count_policy='ma9_in_both_arms',
                  control_native_reference='registered_ma89_native',
                  baseline_arm='F4-cache-zcount', candidate_arm='F4-cache-zcount-tilepub',
                  control_native_sources_unchanged=True, projection_arithmetic_unchanged=True, projection_sources_unchanged=False,
                  candidate_native_addition='tile_scoped_publication', baseline_source_only=False, gpu_validated=False,
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
            'Wrong MA12 source protocol')
    require(record.get('activation_policy') == 'ma8_v2_cache_in_both_arms' and
            record.get('zero_count_policy') == 'ma9_in_both_arms' and
            record.get('control_native_reference') == 'registered_ma89_native' and
            record.get('baseline_arm') == 'F4-cache-zcount' and record.get('candidate_arm') == 'F4-cache-zcount-tilepub',
            'MA12 requires the complete MA89 baseline in both arms')
    require(record.get('control_native_sources_unchanged') is True and
            record.get('projection_sources_unchanged') is False and
            record.get('projection_arithmetic_unchanged') is True and
            record.get('candidate_native_addition') == 'tile_scoped_publication',
            'MA12 must retain the exact MA89 control and original projection arithmetic')
    verify_files('/', record['inputs'])
    verify_files(root, record['files'])
    require(record['inputs'] == {str(HERE / name): sha(HERE / name) for name in INPUTS},
            'MA12 builder or autograd implementation changed')
    require(sha(root / 'manifest.json') == record['build_manifest_sha256'], 'MA12 build manifest changed')
    require(info['protocol'] == build.PROTOCOL and info['arch'] == 'sm90'
            and info['config'] == model_config() and not info['source_only'] and not info['diagnostic_only']
            and tuple(info['variants']) == VARIANTS and tuple(record['variants']) == VARIANTS
            and info['inputs'] == record['inputs'], 'Wrong MA12 build configuration')
    # This includes compiled libraries and Python extensions after compilation;
    # the pre-build record intentionally lists only immutable configuration.
    verify_files(root, info['files'])
    baseline.verify(root / 'reference')
    ref = read(root / 'reference/manifest.json')
    frozen = ref['variants']['F4']['source_sha256']
    require(record['baseline_variants'] == {'F4': frozen}, 'Frozen F4 changed')
    require(read(root / 'selection-plan.json') == read(root / 'reference/selection-plan.json'),
            'C masks/order changed')
    tuning = ma12_sources.tuning_config(record['tuning'])
    require(read(root / 'tuning.json') == tuning, 'MA12 tuning identity changed')
    control_updates = ma9_sources.sources(root / 'reference/F4')
    control = dict(frozen, **{name: hashlib.sha256(source.encode()).hexdigest()
                             for name, source in control_updates.items()})
    candidate = dict(control, **{name: hashlib.sha256(source.encode()).hexdigest()
                                for name, source in ma12_sources.sources(root / VARIANTS[0]).items()})
    expected = {VARIANTS[0]: control, VARIANTS[1]: candidate}
    for arm in VARIANTS:
        hashes = info['variants'][arm]['source_sha256']
        require(hashes == record['variants'][arm] == expected[arm],
                arm + ' differs from registered MA12 transform')
        verify_files(root / arm, hashes)
        if built:
            require(all((root / arm / name).is_file() for name in
                        ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping')),
                    'Missing MA12 runtime: ' + arm)
            artifacts = [root / arm / 'python/flux/lib' / name for name in
                         ('libflux_cuda.so', 'libflux_cuda_ths_op.so')]
            artifacts += sorted((root / arm / 'python').glob('flux_ths_pybind*.so'))
            artifacts += [root / arm / 'check_mapping']
            require(len(artifacts) >= 4 and all(path.is_file() and
                    str(path.relative_to(root)) in info['files'] for path in artifacts),
                    'MA12 runtime lacks recorded binary identity: ' + arm)
            helpers = ma12_sources.NATIVE_EXPORTS
            verify_native_exports(root / arm / 'python/flux/lib/libflux_cuda.so',
                required=('taco_ma4f_protocol_version', 'taco_ma4f_decode_reduce') +
                         ma9_sources.NATIVE_EXPORTS +
                         (helpers if arm == 'F4-cache-zcount-tilepub' else ()),
                forbidden=helpers if arm == 'F4-cache-zcount' else ())
    if built:
        require(info['status'] == 'built', 'Unbuilt MA12 experiment')
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

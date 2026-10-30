"""Build the MA20 joint mask/order bank from the exact registered MA11 candidate.

Only two mapping tables differ between native variants. Selection masks remain
runtime inputs, frozen separately in ma20-plan.json; all codec, publication,
decoder, MA8/9/11 and backward implementations retain their MA11 source bytes.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import shutil

import build
import ma11_build as ma11_builder
import ma11_sources
import ma20_plan
import scope_build
from ma2_ablation_build import copy_registered
from native_exports import verify_native_exports
from common import HERE, read, replace_once, require, sha, verify_files, write
from scope_plan import model_config

PROTOCOL = ma20_plan.PROTOCOL
MANIFEST = 'ma20-manifest.json'
PLAN = 'ma20-plan.json'
VARIANTS = ('MA11', 'Q0', 'Q7', 'Q14', 'Q28', 'Q56', 'Q112')
REFERENCE_ARM = ma11_sources.VARIANTS[1]
RUNTIME = ma11_builder.RUNTIME
C_POLICY = ma11_builder.C_POLICY
SWIZZLE = scope_build.SWIZZLE
HOST = 'phase6_host.hpp'
ALLOWED_SOURCE_CHANGES = (SWIZZLE, HOST)
INPUTS = tuple(dict.fromkeys(('ma20_build.py', 'ma20_plan.py',
                             'scope_build.py', *ma11_builder.INPUTS)))
canonical = ma11_builder.canonical


def runtime_root(root, arm):
    require(arm in VARIANTS, 'Unknown MA20 source arm')
    return Path(root).resolve() / arm


def _roles(configuration):
    return {role: configuration['roles'][role]['entry'] for role in ('attn', 'mlp')}


def sources(reference, plans, arm):
    """Return only replacements; MA11 itself has no source transform."""
    require(arm in VARIANTS, 'Unknown MA20 source arm')
    if arm == 'MA11':
        return {}
    reference = Path(reference)
    control = _roles(plans['configurations']['MA11'])
    entries = _roles(plans['configurations'][arm])
    swizzle = replace_once((reference / SWIZZLE).read_text(),
                           scope_build.tables(control, True), scope_build.tables(entries, True))
    host = replace_once((reference / HOST).read_text(),
                        scope_build.tables(control), scope_build.tables(entries))
    return {SWIZZLE: swizzle, HOST: host}


def _plan(selection, plan_path=None):
    expected = ma20_plan.bundle(selection)
    require(tuple(expected['configurations']) == VARIANTS, 'Wrong MA20 configuration bank')
    if plan_path is not None:
        require(read(Path(plan_path)) == expected,
                'Supplied MA20 plan differs from the registered joint-placement planner')
    return expected


def prepare(out, ma11_build=None, plan_path=None):
    out = Path(out).resolve()
    require(not out.exists(), 'Use a fresh MA20 build')
    out.mkdir(parents=True)
    reference = (ma11_builder.prepare(out / 'reference') if ma11_build is None
                 else Path(ma11_build).resolve())
    ma11_builder.verify(reference)
    old = read(reference / 'manifest.json')
    frozen = old['variants'][REFERENCE_ARM]['source_sha256']
    selection = read(reference / 'selection-plan.json')
    plans = _plan(selection, plan_path)
    variants = {}
    for arm in VARIANTS:
        copy_registered(reference / REFERENCE_ARM, out / arm, frozen)
        for name, source in sources(reference / REFERENCE_ARM, plans, arm).items():
            (out / arm / name).write_text(source)
        item = copy.deepcopy(old['variants'][REFERENCE_ARM])
        item.update(python=arm + '/python',
                    source_sha256={name: sha(out / arm / name) for name in frozen})
        variants[arm] = item
    for name in ('selection-plan.json', 'config.json', 'TACO_LICENSE.txt', 'tuning.json'):
        shutil.copy2(reference / name, out / name)
    write(out / PLAN, plans)
    inputs = {str(HERE / name): sha(HERE / name) for name in INPUTS}
    files = {name: sha(out / name) for name in
             ('selection-plan.json', 'config.json', 'TACO_LICENSE.txt', 'tuning.json', PLAN)}
    info = dict(protocol=build.PROTOCOL, arch='sm90', kind='baselines', status='prepared',
                config=model_config(), gpu_validated=False, source_only=False, diagnostic_only=False,
                submodules=old['submodules'], source_commit=old.get('source_commit'),
                inputs=inputs, files=files, variants=variants)
    write(out / 'manifest.json', info)
    record = dict(protocol=PROTOCOL, reference_protocol=ma11_builder.PROTOCOL,
                  reference=str(reference), reference_manifest_sha256=sha(reference / 'manifest.json'),
                  reference_ma11_manifest_sha256=sha(reference / ma11_builder.MANIFEST),
                  reference_arm=REFERENCE_ARM, baseline_arm='MA11',
                  activation_policy='ma8_v2_cache_in_all_arms',
                  zero_count_policy='ma9_in_all_arms', softmax_policy='ma11_in_all_arms',
                  control_native_sources_unchanged=True,
                  allowed_source_changes=list(ALLOWED_SOURCE_CHANGES),
                  mask_binding='runtime_frozen_ma20_plan',
                  baseline_source_only=False, gpu_validated=False,
                  tuning=read(out / 'tuning.json'),
                  inputs=inputs, files=files, baseline_variants={REFERENCE_ARM: frozen},
                  variants={arm: item['source_sha256'] for arm, item in variants.items()},
                  build_manifest_sha256=sha(out / 'manifest.json'))
    write(out / MANIFEST, record)
    verify(out)
    return out


def verify(root, built=False):
    root = Path(root).resolve()
    record, info = read(root / MANIFEST), read(root / 'manifest.json')
    require(record.get('protocol') == PROTOCOL and
            record.get('reference_protocol') == ma11_builder.PROTOCOL and
            record.get('reference_arm') == REFERENCE_ARM and
            record.get('baseline_arm') == 'MA11', 'Wrong MA20 source protocol')
    require(record.get('activation_policy') == 'ma8_v2_cache_in_all_arms' and
            record.get('zero_count_policy') == 'ma9_in_all_arms' and
            record.get('softmax_policy') == 'ma11_in_all_arms' and
            record.get('control_native_sources_unchanged') is True and
            record.get('allowed_source_changes') == list(ALLOWED_SOURCE_CHANGES) and
            record.get('mask_binding') == 'runtime_frozen_ma20_plan' and
            record.get('baseline_source_only') is False and record.get('gpu_validated') is False,
            'MA20 must retain the complete MA11 implementation and frozen plan')
    verify_files('/', record['inputs'])
    require(record['inputs'] == {str(HERE / name): sha(HERE / name) for name in INPUTS},
            'MA20 builder or planner changed')
    verify_files(root, record['files'])
    require(sha(root / 'manifest.json') == record['build_manifest_sha256'],
            'MA20 build manifest changed')
    require(info['protocol'] == build.PROTOCOL and info['arch'] == 'sm90' and
            info['kind'] == 'baselines' and info['config'] == model_config() and
            not info['source_only'] and not info['diagnostic_only'] and
            tuple(info['variants']) == VARIANTS and tuple(record['variants']) == VARIANTS and
            info['inputs'] == record['inputs'], 'Wrong MA20 build configuration')
    verify_files(root, info['files'])
    reference = Path(record['reference'])
    ma11_builder.verify(reference)
    require(sha(reference / 'manifest.json') == record['reference_manifest_sha256'] and
            sha(reference / ma11_builder.MANIFEST) == record['reference_ma11_manifest_sha256'],
            'MA20 registered MA11 reference changed')
    old = read(reference / 'manifest.json')
    frozen = old['variants'][REFERENCE_ARM]['source_sha256']
    require(record['baseline_variants'] == {REFERENCE_ARM: frozen}, 'Frozen MA11 changed')
    selection = read(root / 'selection-plan.json')
    require(selection == read(reference / 'selection-plan.json'), 'MA20 baseline selection changed')
    plans = _plan(selection, root / PLAN)
    tuning = ma11_sources.tuning_config(record['tuning'])
    require(read(root / 'tuning.json') == read(reference / 'tuning.json') == tuning,
            'MA20 cannot change MA11 geometry')
    for arm in VARIANTS:
        updates = sources(reference / REFERENCE_ARM, plans, arm)
        expected = dict(frozen, **{name: hashlib.sha256(value.encode()).hexdigest()
                                  for name, value in updates.items()})
        hashes = info['variants'][arm]['source_sha256']
        require(hashes == record['variants'][arm] == expected,
                arm + ' differs from registered MA20 transform')
        differences = {name for name in set(frozen) | set(hashes)
                       if hashes.get(name) != frozen.get(name)}
        require(not differences if arm == 'MA11' else differences <= set(ALLOWED_SOURCE_CHANGES),
                'MA20 changed native work beyond its mapping tables')
        verify_files(root / arm, hashes)
        if built:
            require(all((root / arm / name).is_file() for name in
                        ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping')),
                    'Missing MA20 runtime: ' + arm)
            artifacts = [root / arm / 'python/flux/lib' / name for name in
                         ('libflux_cuda.so', 'libflux_cuda_ths_op.so')]
            artifacts += sorted((root / arm / 'python').glob('flux_ths_pybind*.so'))
            artifacts += [root / arm / 'check_mapping']
            require(len(artifacts) >= 4 and all(path.is_file() and
                    str(path.relative_to(root)) in info['files'] for path in artifacts),
                    'MA20 runtime lacks recorded binary identity: ' + arm)
            verify_native_exports(root / arm / 'python/flux/lib/libflux_cuda.so',
                required=('taco_ma4f_protocol_version', 'taco_ma4f_decode_reduce') +
                         ma11_sources.ZERO_COUNT_EXPORTS + ma11_sources.NATIVE_EXPORTS)
    if built:
        require(info['status'] == 'built', 'Unbuilt MA20 experiment')
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
    p.add_argument('--ma11-build', type=Path)
    p.add_argument('--plan', type=Path)
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
        print(prepare(args.out, args.ma11_build, args.plan))
    elif args.action == 'verify':
        verify(args.build, args.built)
    else:
        compile_build(args.build, args.cuda_home, args.jobs, args.nccl_root)


if __name__ == '__main__':
    main()

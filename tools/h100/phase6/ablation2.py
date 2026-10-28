"""Five-group H100 ablation: CPU preparation/dry-run; GPU requires run --execute."""
import argparse
import copy
from pathlib import Path
import shutil
import sys

import ablation2_plan as spec
import candidate
from ablation_plan import parent_plan
from common import HERE, REPO, read, require, sha, verify_files, write


def reuse(out, source):
    """Register only four existing builds, never modify or rebuild their trees."""
    source, out = Path(source).resolve(), Path(out).resolve()
    old = read(source/'ablation-manifest.json')
    require(old['parent_freeze'] == spec.FREEZE_ID, 'Wrong reuse freeze')
    parent, path, evidence = parent_plan(source/'plans/parent.json')
    require(sha(path) == old['parent_plan']['sha256'] and parent['config'] == old['config'] and
            parent['hardware'] == old['calibration_hardware'], 'Reuse parent/config/hardware mismatch')
    # Only selected groups are checked: unavailable G0/G5/G6/remote builds are irrelevant.
    entries = {}
    for key, definition in spec.definitions().items():
        identity = 'interleaved-'+key
        v = dict(definition, policy=key)
        if key != 'native_taco':
            prior = old['variants'][identity]
            root = (source/prior['build']).resolve()
            manifest = read(root/'manifest.json')
            require(manifest['status'] == 'built' and not manifest.get('diagnostic_only'), 'Reuse needs complete local builds')
            verify_files(root, manifest['files'])
            verify_files(root/identity, manifest['variants'][identity]['source_sha256'])
            plan = source/prior['plan']
            require(read(plan)['entry'] == spec.derive(parent, definition) and
                    sha(plan) == prior['derived_plan_sha256'] == sha(root/'derived-plan.json'), 'Reuse physical plan changed')
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping'):
                require((root/identity/name).is_file(), f'Missing local reuse binary: {name}')
            v.update(build=str(root), plan=f'plans/{identity}.json',
                     derived_plan_sha256=sha(plan), mask_sha256=read(plan)['entry']['mask_sha256'],
                     build_protocol=manifest['protocol'], reuse_manifest_sha256=sha(root/'manifest.json'))
        entries[identity] = v
    out.mkdir(parents=True, exist_ok=False)
    (out/'plans').mkdir()
    shutil.copy2(path, out/'plans/parent.json')
    write(out/'config.json', parent['config'])
    for identity, v in entries.items():
        if v['group'] != 'native_taco':
            shutil.copy2(source/old['variants'][identity]['plan'], out/v['plan'])
    record = copy.deepcopy(old)
    record.update(protocol=spec.PROTOCOL, bases=['interleaved'], seeds=[], variants=entries,
                  parent_plan=dict(path=str(path), sha256=sha(path), evidence=evidence), gpu_validated=False,
                  reuse=dict(root=str(source), manifest_sha256=sha(source/'ablation-manifest.json'),
                             timing_reused=False),
                  files={str(p.relative_to(out)): sha(p) for p in (out/'plans').iterdir()})
    record['files']['config.json'] = sha(out/'config.json')
    write(out/'ablation-manifest.json', record)
    candidate.verify(out, built=True)
    return out


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('build-prepare'); p.add_argument('out', type=Path); p.add_argument('--plan', type=Path)
    p = sub.add_parser('reuse'); p.add_argument('out', type=Path); p.add_argument('--from-builds', type=Path, required=True)
    p = sub.add_parser('compile'); p.add_argument('builds', type=Path)
    p.add_argument('--cuda-home', type=Path, required=True); p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--nccl-root', type=Path); p.add_argument('--groups', nargs='+', choices=spec.POLICIES[1:])
    p = sub.add_parser('compile-check'); p.add_argument('builds', type=Path)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--cuda-home', type=Path, required=True)
    p.add_argument('--cxx', default='g++')
    p = sub.add_parser('prepare'); p.add_argument('out', type=Path)
    p.add_argument('--builds', type=Path, required=True); p.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    p.add_argument('--blocks', type=int, default=5); p.add_argument('--schedule-seed', type=int, default=20261013)
    p.add_argument('--source-only', action='store_true')
    p = sub.add_parser('run'); p.add_argument('root', type=Path)
    p.add_argument('--stage', choices=('preflight', 'timing', 'all'), default='all')
    p.add_argument('--execute', action='store_true'); p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--cuda-home', type=Path)
    p = sub.add_parser('report'); p.add_argument('root', type=Path)
    p.add_argument('--offline', action='store_true'); p.add_argument('--preflight', action='store_true')
    a = parser.parse_args()
    if a.action == 'build-prepare':
        print(candidate.prepare(a.out, a.plan, seeds=(), bases=spec.BASES, protocol=spec.PROTOCOL))
    elif a.action == 'reuse':
        print(reuse(a.out, a.from_builds))
    elif a.action in ('compile', 'compile-check', 'prepare'):
        data = candidate.verify(a.builds)
        require(data['protocol'] == spec.PROTOCOL, 'Use a five-group build registry')
        if a.action == 'compile':
            import build
            require(a.jobs > 0 and not data.get('reuse'), 'Compile only fresh builds; reuse is read-only')
            for key in a.groups or spec.POLICIES[1:]:
                root = a.builds/data['variants']['interleaved-'+key]['build']
                if read(root/'manifest.json')['status'] != 'built':
                    build.compile_build(root, a.cuda_home, a.jobs, a.nccl_root)
        elif a.action == 'compile-check':
            from ablation_compile_check import check
            check(a.builds, a.out, a.cuda_home, a.cxx)
        else:
            from ablation_prepare import prepare
            print(prepare(a.out, a.builds, a.megatron, a.blocks, a.schedule_seed, a.source_only))
    else:
        require(read(a.root/'submission.json')['protocol'] == spec.PROTOCOL, 'Not an ablation2 comparison')
        if a.action == 'run':
            from ablation_run import run
            run(a.root, a.stage, a.execute, a.python, a.cuda_home)
        else:
            from ablation_report import preflight
            from ablation2_report import report
            (preflight if a.preflight else report)(a.root, a.offline)


if __name__ == '__main__':
    main()

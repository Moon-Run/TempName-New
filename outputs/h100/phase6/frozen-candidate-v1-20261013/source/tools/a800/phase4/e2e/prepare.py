"""Freeze MLP-only quantization comparisons using two isolated codec builds."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(repo, out, job, fused_build=None, separate_build=None, legacy_fused_build=None):
    repo, out = repo.resolve(), out.resolve()
    here = Path(__file__).resolve().parent
    assert repo/'logs/a800' in out.parents
    artifacts = repo/'outputs/a800/phase4'/out.name
    out.mkdir(parents=True)
    (out/'results').mkdir()
    (out/'scripts').mkdir()
    artifacts.mkdir()
    build = artifacts/'build'
    build.mkdir()
    parent = repo/'outputs/a800/phase3/three-way-build'
    fused = (fused_build or repo/'outputs/a800/phase4/taco-fused-20261003').resolve()
    separate = (separate_build or repo/'outputs/a800/phase4/taco-separate-20261003').resolve()
    stock_manifest = json.loads((parent/'manifest.json').read_text())
    roots = {'original': parent/'original', 'original_matched': fused/'taco',
             'taco_fused': fused/'taco', 'taco_separate': separate/'taco'}
    if legacy_fused_build:
        roots['taco_fused_legacy'] = legacy_fused_build.resolve()/'taco'
    manifest = dict(files={}, policies={}, source_manifests={}, reorder=False,
                    scope='MLP-only quantization; original attention BF16 mapping')
    for policy, source in roots.items():
        source_root = parent if policy == 'original' else source.parent
        m = json.loads((source_root/'manifest.json').read_text())
        for name, digest in m['files'].items():
            assert sha(source_root/name) == digest, name
        if policy != 'original':
            assert not m.get('reorder', False), 'Phase4 requires original tile order; use phase5 for joint builds'
            assert m['original_swizzle_sha256'] == stock_manifest['policies']['original']['swizzle_sha256']
            assert sha(source/'overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp') == m['original_swizzle_sha256']
            assert m['placement'] == ('separate' if policy == 'taco_separate' else 'fused')
        (build/policy).symlink_to(source, target_is_directory=True)
        prefix = 'original/' if policy == 'original' else 'taco/'
        for name, digest in m['files'].items():
            if name.startswith(prefix):
                manifest['files'][policy+'/'+name[len(prefix):]] = digest
        manifest['source_manifests'][policy] = dict(path=str(source_root/'manifest.json'), sha256=sha(source_root/'manifest.json'))
        manifest['policies'][policy] = dict(library_sha256=sha(source/'libflux_cuda.so'),
            swizzle_sha256=sha(source/'overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp'))
    (build/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for name in ('adapter.py','worker.py','launch.py','summarize.py','routing.py','taco_support.py','config.json'):
        shutil.copy2(here/name, out/'scripts'/name)
    config_path = out/'scripts/config.json'
    config = json.loads(config_path.read_text())
    if legacy_fused_build:
        config['policies'].append('taco_fused_legacy')
        config['module_policies']['taco_fused_legacy'] = {'attention': 'original', 'mlp': 'taco_fused_legacy'}
        policies = config['policies']
        config['orders'] = [policies[i:]+policies[:i] for _ in range(2) for i in range(len(policies))]
        config_path.write_text(json.dumps(config, indent=2)+'\n')
    shutil.copy2(here.parent/'reference.py', out/'scripts/reference.py')
    shutil.copy2(here.parent/'validate.py', out/'scripts/validate.py')
    for name in ('campaign.py','report.py','prepare.py','verify.py'):
        shutil.copy2(here/name, out/name)
    for p in out.rglob('*.py'):
        ast.parse(p.read_text())
    files = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    info = dict(job_id=job, repo=str(repo), artifact_root=str(artifacts), files=files,
                build_manifest_sha256=sha(build/'manifest.json'), repetitions=2, blocks=len(config['orders']),
                policies=list(json.loads((out/'scripts/config.json').read_text())['policies']),
                scope='Two process repetitions in one held allocation; not independent jobs',
                original_campaign=str(repo/'outputs/a800/arrival-v2-tp4-20261003'))
    (out/'submission.json').write_text(json.dumps(info, indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('repo', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--job-id', type=int, required=True)
    p.add_argument('--fused-build', type=Path)
    p.add_argument('--separate-build', type=Path)
    p.add_argument('--legacy-fused-build', type=Path)
    a = p.parse_args()
    main(a.repo, a.out, a.job_id, a.fused_build, a.separate_build, a.legacy_fused_build)

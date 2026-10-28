"""Freeze four MLP-only policies: results in logs, build cache in outputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def main(repo, out, job, artifact_root=None):
    repo, out = repo.resolve(), out.resolve()
    here = Path(__file__).resolve().parent
    source = here/'model'
    artifact_root = (artifact_root or repo/'outputs/a800'/out.name).resolve()
    assert repo/'logs/a800' in out.parents, 'Store experiment results under logs/a800'
    assert repo/'outputs/a800' in artifact_root.parents, 'Store build caches under outputs/a800'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    artifact_root.mkdir()
    (out/'results').mkdir()
    shutil.copytree(source, out/'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('build.py', 'campaign.py', 'report.py', 'routing.py', 'prepare.py', 'test_routing.py', 'test_headers.py', 'verify.py'):
        shutil.copy2(here/name, out/name)
    shutil.copy2(here/'routing.py', out/'scripts/routing.py')
    check = repo/'tools/a800/arrival/check_mapping.cu'
    shutil.copy2(check, out/'check_mapping.cu')
    scripts = out/'scripts'
    cfg = json.loads((scripts/'config.json').read_text())
    cfg['policies'] = ['native', 'original', 'mlp_remote', 'mlp_remote_arrival']
    policies = cfg['policies']
    cfg['orders'] = [policies[i:]+policies[:i] for i in range(4)] * 2
    cfg['scope'] = 'MLP-only rank+1 and rank+1+tail mapping; attention remains original Flux mapping. TP4 full optimizer steps, eight paired blocks per round; synthetic tokens, not convergence evidence.'
    cfg['module_policies'] = {'native': {'attention': 'native', 'mlp': 'native'},
                              'original': {'attention': 'original', 'mlp': 'original'},
                              'mlp_remote': {'attention': 'original', 'mlp': 'remote_first'},
                              'mlp_remote_arrival': {'attention': 'original', 'mlp': 'remote_arrival'}}
    cfg['mapping_policy_ids'] = {'original': {'2048': 0, '8192': 0},
                                 'mlp_remote': {'2048': 0, '8192': 1},
                                 'mlp_remote_arrival': {'2048': 0, '8192': 3}}
    (scripts/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    original = repo/'outputs/a800/arrival-v2-tp4-20261003'
    plan = original/'artifacts/plan.json'
    shutil.copy2(plan, artifact_root/'plan.json')
    for p in list(out.glob('*.py'))+list(scripts.glob('*.py')):
        ast.parse(p.read_text())
    files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in out.rglob('*') if p.is_file()}
    info = dict(job_id=job, repo=str(repo), policies=policies, files=files,
                parent_scripts=str(source), repetitions=2, blocks=8,
                artifact_root=str(artifact_root), plan_source=str(plan),
                plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
                scope='Fresh processes within one existing allocation; not independent jobs',
                original_campaign=str(repo/'outputs/a800/arrival-v2-tp4-20261003'))
    (out/'submission.json').write_text(json.dumps(info, indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('repo', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--artifact-root', type=Path)
    args = parser.parse_args()
    main(args.repo, args.out, args.job_id, args.artifact_root)

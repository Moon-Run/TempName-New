"""CPU-only, non-overwriting archive of the Phase6 source and measured plans.

The archive contains buildable source inputs, not runtime libraries or complete
measurement logs. Calibration records remain explicit, hashed external inputs.
No import of torch, CUDA initialization, Git mutation, or GPU query is performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


REPO = Path(__file__).resolve().parents[3]
DEFAULT = 'outputs/h100/phase7/phase6-baseline-20261017'
FREEZE = 'outputs/h100/phase6/frozen-candidate-v1-20261013'
SCOPE = 'outputs/h100/phase6/TTention-MLP-gpt67-tp8-20261014-231236'
ABLATION = 'outputs/h100/phase6/ablation3-gpt67-tp8-scope-20261016-221018-release1'
REPEATS = ['TTention-MLP-gpt67-tp8-20261015-012238-repeat'+str(i) for i in range(1, 5)]
SKIP_DIRS = {'.git', '__pycache__', 'build', 'dist', '.pytest_cache'}
SKIP_SUFFIXES = {'.pyc', '.o', '.so', '.a', '.cubin', '.fatbin'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe relative archive path: '+str(name))
    return path


def inventory(repo):
    """Return a fixed allowlist; generated libraries never enter the archive."""
    repo = Path(repo).resolve()
    files, artifacts, references = {}, {}, {}

    def add(relative, expected=None):
        relative = safe_relative(relative)
        path = repo / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError('Missing/non-regular snapshot input: '+str(relative))
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Changed historical input: '+str(relative))
        files[relative.as_posix()] = dict(sha256=actual, bytes=path.stat().st_size)

    def tree(relative):
        root = repo / relative
        if not root.is_dir():
            raise ValueError('Missing snapshot tree: '+str(relative))
        for path in sorted(root.rglob('*')):
            tail = path.relative_to(root)
            if path.is_file() and not path.is_symlink() and not (set(tail.parts) & SKIP_DIRS) and path.suffix not in SKIP_SUFFIXES:
                add(path.relative_to(repo))

    def metadata(relative):
        for path in sorted((repo / relative).iterdir()):
            if path.is_file() and not path.is_symlink() and path.suffix in {'.json', '.txt', '.md', '.py', '.sh'}:
                add(path.relative_to(repo))

    def build(relative):
        manifest = read(repo / relative / 'manifest.json')
        metadata(relative)
        for name, variant in manifest['variants'].items():
            for source, expected in variant['source_sha256'].items():
                add(Path(relative) / name / source, expected)
        artifacts[relative] = dict(manifest_sha256=digest(repo / relative / 'manifest.json'),
                                   source_commit=manifest.get('source_commit'),
                                   parent_source_commit=manifest.get('parent_source_commit'),
                                   submodules=manifest.get('submodules'),
                                   variants=list(manifest['variants']))

    def evidence(record):
        for name, expected in record.get('inputs', {}).items():
            relative = name.removeprefix('/workspace/TempName/')
            if Path(relative).is_absolute():
                try:
                    relative = str(Path(relative).relative_to(repo))
                except ValueError:
                    raise ValueError('Evidence outside repository: '+name) from None
            safe_relative(relative)
            candidates = [relative, str(Path(FREEZE) / 'source' / relative)]
            match = next((x for x in candidates if (repo / x).is_file() and digest(repo / x) == expected), None)
            if match is None:
                raise ValueError('Missing/changed plan evidence: '+name)
            references[name] = dict(repository_path=match, sha256=expected,
                                    bytes=(repo / match).stat().st_size)
            # Retain small metadata and controllers; full calibration measurements
            # stay in the historical evidence tree rather than duplicating logs.
            if (repo / match).stat().st_size <= 1024 * 1024:
                add(match, expected)

    for relative in ('src', 'include', 'cmake', 'python', 'tools/a800', 'tools/h100/phase6', FREEZE):
        tree(relative)
    for relative in ('CMakeLists.txt', 'setup.py', 'LICENSE', 'README.md', '.gitmodules', 'build.sh'):
        add(relative)
    frozen = read(repo / FREEZE / 'manifest.json')
    measured = frozen['candidate']['build']
    build(measured)
    if artifacts[measured]['manifest_sha256'] != frozen['candidate']['manifest_sha256']:
        raise ValueError('Frozen measured-candidate manifest changed')
    evidence(read(repo / FREEZE / 'evidence/selection-plan.json'))
    for relative in (SCOPE+'/candidates', SCOPE+'/calibration', ABLATION+'/candidates/base', ABLATION+'/candidates/arrival'):
        build(relative)
    for relative in (SCOPE, ABLATION+'/candidates'):
        metadata(relative)
    tree(SCOPE+'/controller-source')
    tree(SCOPE+'/plans')
    scope_plans = read(repo / SCOPE / 'candidates/plans.json')
    for record in scope_plans['roles'].values():
        evidence(record)
    for session in REPEATS+[Path(ABLATION).name]:
        compare = 'logs/h100/phase6/'+session+'/compare'
        metadata(compare)
        for name in ('scripts', 'provenance'):
            tree(compare+'/'+name)
        out = 'outputs/h100/phase6/'+session
        metadata(out)
        if (repo / out / 'controller-source').is_dir():
            tree(out+'/controller-source')
    for record in references.values():
        record['archived'] = record['repository_path'] in files
    return dict(files=dict(sorted(files.items())), artifacts=artifacts, plan_inputs=references)


def verify(out, repo=None):
    out = Path(out).resolve()
    manifest = read(out / 'manifest.json')
    sums = {}
    for line in (out / 'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split('  ', 1)
        safe_relative(name)
        if name in sums:
            raise ValueError('Duplicate checksum entry: '+name)
        sums[name] = expected
        path = out / name
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError('Snapshot checksum mismatch: '+name)
    expected_files = {'manifest.json', 'README.md'} | {'repo/'+p for p in manifest['files']}
    actual_files = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() or p.is_symlink()}
    if set(sums) != expected_files or actual_files != expected_files | {'SHA256SUMS.txt'}:
        raise ValueError('Snapshot has missing or unexpected files')
    for name, record in manifest['files'].items():
        path = out / 'repo' / safe_relative(name)
        if sums['repo/'+name] != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('Snapshot manifest mismatch: '+name)
    checked = 0
    if repo is not None:
        repo = Path(repo).resolve()
        for name, record in manifest['files'].items():
            path = repo / safe_relative(name)
            if not path.is_file() or digest(path) != record['sha256']:
                raise ValueError('Original input changed: '+name)
        for record in manifest['plan_inputs'].values():
            path = repo / safe_relative(record['repository_path'])
            if not path.is_file() or digest(path) != record['sha256']:
                raise ValueError('External plan evidence changed: '+str(path))
            checked += 1
    return dict(status='verified', files=len(manifest['files']), bytes=manifest['source_bytes'],
                original_inputs_checked=repo is not None, plan_inputs_checked=checked,
                gpu_used=False, snapshot=str(out))


def create(out, repo):
    out, repo = Path(out).resolve(), Path(repo).resolve()
    if out.exists():
        raise ValueError('Refusing to overwrite existing snapshot; use verify: '+str(out))
    data = inventory(repo)
    data.update(protocol='h100-phase7-phase6-source-snapshot-v1',
                created_at=datetime.now(timezone.utc).isoformat(), repository=str(repo),
                source_commit=git(repo, 'rev-parse', 'HEAD'),
                submodule_status=git(repo, 'submodule', 'status', '--recursive'),
                controller_status=git(repo, 'status', '--porcelain', '--', 'tools/h100/phase6', 'tools/a800'),
                source_bytes=sum(v['bytes'] for v in data['files'].values()),
                megatron_commit='3ea68ad6042cc1204386ae9364358f7c4de1bc37', gpu_used=False,
                excludes=['runtime .so/.a/.o and build trees', 'Git/submodule objects and environments',
                          'complete training/profile logs', 'large calibration inputs: see plan_inputs where archived=false'],
                restore_policy='Copy into an independent checkout. Never edit this snapshot or overwrite old sessions.')
    out.mkdir(parents=True, exist_ok=False)
    for name, record in data['files'].items():
        target = out / 'repo' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
        if digest(target) != record['sha256']:
            raise ValueError('Source changed during snapshot copy: '+name)
    (out / 'manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    (out / 'README.md').write_text('''# Phase6 source archive before Phase7

This is a non-overwriting source/evidence archive, not an executable build.
`manifest.json` records every copied input, repository HEAD, submodules, measured
artifact identities and calibration input hashes. `SHA256SUMS.txt` covers all
payload files and the manifest. The original outputs/logs were not modified.

Verify from the repository root:

```bash
python3 tools/h100/phase7/snapshot.py verify SNAPSHOT_PATH
# Also verify current source inputs and externally retained calibration evidence:
python3 tools/h100/phase7/snapshot.py verify SNAPSHOT_PATH --original
```

For reconstruction, make a separate checkout of the recorded source_commit,
restore repo/ at its repository root, and restore external plan_inputs using
their repository_path and SHA256. Initialize CUTLASS/NCCL at recorded revisions
and install Megatron at megatron_commit. No third-party repository, environment,
library, or complete result log is embedded. Historical manifests keep their
original paths and library hashes; rebuilding produces a new identity and must
use a new session. Do not change old built manifests to make them executable.

Use archived tools/h100/phase6/README-mlp-attention.md and README-ablation3.md
for new calibration/build/compare commands. Recalibrate both roles if hardware,
GPU order, host, or source changes. The measured candidate manifest is under
TTention-MLP-gpt67-tp8-20261014-231236/candidates; the repeat1-4 sessions reuse it.
Keep this archive as evidence and do all rebuilding in a separate directory.
''')
    paths = sorted(p for p in out.rglob('*') if p.is_file())
    (out / 'SHA256SUMS.txt').write_text(''.join(digest(p)+'  '+p.relative_to(out).as_posix()+'\n' for p in paths))
    return verify(out, repo)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('inventory', 'create', 'verify'))
    parser.add_argument('out', nargs='?', type=Path, default=REPO / DEFAULT)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--original', action='store_true', help='also check original archived inputs and external calibration evidence')
    args = parser.parse_args()
    if args.action == 'inventory':
        data = inventory(args.repo)
        result = dict(files=len(data['files']), bytes=sum(v['bytes'] for v in data['files'].values()),
                      external_plan_inputs=sum(not r['archived'] for r in data['plan_inputs'].values()),
                      artifacts=data['artifacts'], gpu_used=False)
    elif args.action == 'create':
        result = create(args.out, args.repo)
    else:
        result = verify(args.out, args.repo if args.original else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Freeze and run the two-route GEMM tuning diagnostic in an existing TP4 job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    repo = Path(__file__).resolve().parents[3]
    root, build = args.root.resolve(), args.build.resolve()
    root.mkdir(parents=True, exist_ok=False)
    scripts = root/'scripts'
    scripts.mkdir()
    for name, source in {
        'bench_gemm_tuning.py': Path(__file__).with_name('bench_gemm_tuning.py'),
        'taco_support.py': repo/'tools/a800/phase5/e2e/taco_support.py',
        'routing.py': repo/'tools/a800/phase5/e2e/routing.py',
        'reference.py': repo/'tools/a800/phase4/reference.py',
        'plan.json': args.plan.resolve(),
    }.items():
        shutil.copy2(source, scripts/name)
    shutil.copy2(__file__, root/'run.py')
    info = dict(build=str(build), job_id=args.job_id, blocks=args.blocks,
        stages=args.stages, sms=args.sms,
        build_manifests={p:sha(build/p/'manifest.json') for p in
                         ('remote_arrival','interleaved_arrival')},
        files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()})
    (root/'manifest.json').write_text(json.dumps(info, indent=2)+'\n')
    print(root)


def run(root):
    root = root.resolve()
    info = json.loads((root/'manifest.json').read_text())
    assert os.environ.get('SLURM_JOB_ID') == str(info['job_id'])
    for name, digest in info['files'].items():
        assert sha(root/name) == digest, name
    build, scripts = Path(info['build']), root/'scripts'
    state = dict(completed=False, job_id=info['job_id'], step_id=os.environ['SLURM_STEP_ID'], results=[])
    for policy, folder in (('remote_first','remote_arrival'), ('interleaved','interleaved_arrival')):
        assert sha(build/folder/'manifest.json') == info['build_manifests'][folder]
        manifest = json.loads((build/folder/'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            assert sha(build/folder/name) == digest, name
        library = build/folder/'taco'
        env = dict(os.environ, PYTHONPATH=f'{scripts}:{library}/python',
            PATH=f'{Path(sys.executable).parent}:/data/apps/cuda/12.4/bin:'+os.environ['PATH'],
            LD_LIBRARY_PATH=f'{library}:/data/apps/cuda/12.4/lib64',
            CUDA_DEVICE_MAX_CONNECTIONS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            PYTHONNOUSERSITE='1', NCCL_DEBUG='WARN', NCCL_IB_DISABLE='1',
            NCCL_P2P_DISABLE='0', NCCL_P2P_LEVEL='NVL', FLUX_FORCE_NVLINK='1')
        out = root/policy
        out.mkdir()
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
               str(scripts/'bench_gemm_tuning.py'), str(out), '--plan', str(scripts/'plan.json'),
               '--policy', policy, '--blocks', str(info['blocks']), '--stages',
               *map(str,info['stages']), '--sms', *map(str,info['sms'])]
        print('START', policy, flush=True)
        with (out/'process.log').open('w') as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        (out/'exit-code.txt').write_text(str(result.returncode)+'\n')
        if result.returncode:
            print((out/'process.log').read_text()[-10000:], flush=True)
            raise RuntimeError(cmd)
        rows = [json.loads((out/f'rank{r}.json').read_text()) for r in range(4)]
        assert all(r['completed'] for r in rows)
        state['results'].append(policy)
        (root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        print('DONE', policy, flush=True)
    state['completed'] = True
    (root/'state.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--job-id', type=int)
    parser.add_argument('--blocks', type=int, default=8)
    parser.add_argument('--stages', type=int, nargs='+', default=[3,4])
    parser.add_argument('--sms', type=int, nargs='+', default=[0,104,100,96,92,88,84,80,72,64])
    args = parser.parse_args()
    if args.build:
        assert args.plan and args.job_id
        prepare(args)
    else:
        run(args.root)

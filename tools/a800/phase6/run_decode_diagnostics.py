"""Run frozen decoder correctness/performance checks in one existing TP4 step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main(root):
    root = root.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    build = Path(manifest['build'])
    scripts = root/'scripts'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert os.environ.get('SLURM_JOB_ID') == str(manifest.get('job_id',179147))
    for name, digest in manifest['files'].items():
        assert sha(root/name) == digest, name
    results = []
    for policy, folder in (('remote_first', 'remote_arrival'), ('interleaved', 'interleaved_arrival')):
        assert sha(build/folder/'manifest.json') == manifest['build_manifests'][folder]
        info = json.loads((build/folder/'manifest.json').read_text())
        for name, digest in info['files'].items():
            assert sha(build/folder/name) == digest, name
        library = build/folder/'taco'
        env = dict(os.environ, PYTHONPATH=f'{scripts}:{library}/python',
                   LD_LIBRARY_PATH=f'{library}:/data/apps/cuda/12.4/lib64',
                   CUDA_DEVICE_MAX_CONNECTIONS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONNOUSERSITE='1', NCCL_DEBUG='WARN', NCCL_IB_DISABLE='1',
                   NCCL_P2P_DISABLE='0', NCCL_P2P_LEVEL='NVL', FLUX_FORCE_NVLINK='1')
        stages = [('validate', ['--double-buffered','--model-shape','8192','2048','2048']),
                  ('bench_decode_variants', [])]
        if manifest.get('validation_only'):
            stages = stages[:1]
        for kind, extra in stages:
            if kind == 'validate' and 'validation_variant' in manifest:
                extra += ['--decode-variant', str(manifest['validation_variant'])]
            if kind == 'validate' and manifest.get('compact_boundaries'):
                extra += ['--compact-boundaries']
            if kind == 'bench_decode_variants' and 'variants' in manifest:
                extra += ['--variants', *map(str,manifest['variants'])]
            out = root/f'{kind}-{policy}'
            out.mkdir()
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc_per_node=4', str(scripts/f'{kind}.py'), str(out),
                       '--plan', str(scripts/'plan.json'), '--policy', policy, *extra]
            print('START', kind, policy, flush=True)
            with (out/'process.log').open('w') as log:
                run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            (out/'exit-code.txt').write_text(str(run.returncode)+'\n')
            if run.returncode:
                print((out/'process.log').read_text()[-12000:], flush=True)
                raise RuntimeError(command)
            rows = [json.loads((out/f'rank{rank}.json').read_text()) for rank in range(4)]
            assert all(r['completed'] for r in rows)
            results.append(dict(kind=kind, policy=policy, completed=True))
            (root/'state.json').write_text(json.dumps(dict(completed=False, results=results),indent=2)+'\n')
            print('DONE', kind, policy, flush=True)
    (root/'state.json').write_text(json.dumps(dict(completed=True, results=results),indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('root', type=Path)
    main(p.parse_args().root)

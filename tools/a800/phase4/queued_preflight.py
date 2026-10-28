"""Run TACO validation after another experiment releases the GPUs logically.

The dependency is a completed campaign state in the same held allocation.
No job cancellation, submission, or allocation release is performed here.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def main(repo, root, dependency):
    repo, root = repo.resolve(), root.resolve()
    info = json.loads((root/'submission.json').read_text())
    assert os.environ.get('SLURM_JOB_ID') == str(info['job_id'])
    now = lambda: datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    state = dict(job_id=info['job_id'], step_id=os.environ.get('SLURM_STEP_ID'),
                 status='waiting', dependency=str(dependency), started_at=now())
    def save():
        state['updated_at'] = now()
        (root/'state.json').write_text(json.dumps(state, indent=2)+'\n')
    save()
    try:
        while True:
            other = json.loads(dependency.read_text())
            assert other['job_id'] == info['job_id']
            if other['status'] == 'completed':
                break
            if other['status'] in ('failed','interrupted'):
                raise RuntimeError('Dependency did not complete; do not overlap a failed GPU workload')
            time.sleep(10)
        for name, digest in info['files'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
        build = repo/info['build']
        manifest = json.loads((build/'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            assert hashlib.sha256((build/name).read_bytes()).hexdigest() == digest
        state.update(status='running', validation_started_at=now())
        save()
        python = repo/'outputs/a800/venv/bin/python'
        library = build/'taco'
        env = dict(os.environ, PYTHONPATH=str(library/'python'),
                   LD_LIBRARY_PATH=f'{library}:/data/apps/cuda/12.8/lib64',
                   FLUX_FORCE_NVLINK='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', NCCL_IB_DISABLE='1',
                   NCCL_P2P_DISABLE='0', NCCL_P2P_LEVEL='NVL', NCCL_DEBUG='WARN')
        command = [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   str(root/'scripts/validate.py'), str(root/'results'), '--model-shapes']
        state['command'] = command
        save()
        with (root/'validation.log').open('w') as log:
            result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        state['exit_code'] = result.returncode
        assert result.returncode == 0, (root/'validation.log').read_text()[-12000:]
        rows = [json.loads((root/f'results/rank{i}.json').read_text()) for i in range(4)]
        assert all(r['completed'] and r['passed'] for r in rows)
        state.update(status='completed', finished_at=now(), checks=sum(len(r['checks']) for r in rows))
        save()
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo','root','dependency'):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    main(a.repo, a.root, a.dependency)

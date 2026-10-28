"""Run the audited TP4 experiment serially inside one existing allocation.

No sbatch/salloc/scancel calls: the outer allocation owns resource lifetime.
Each repetition has fresh processes and a separate directory, but is explicitly
NOT an independent Slurm allocation. Failed stages stop dependent work.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_gate(rounds):
    """Require the same candidate to improve both actual model shapes twice."""
    accepted = []
    shapes = {(2048, 2048, 2048), (2048, 2048, 8192)}
    for policy, base in [('remote_arrival', 'remote_first'), ('interleaved_arrival', 'interleaved')]:
        passed = len(rounds) == 2
        for report in rounds:
            points = [r for r in report['comparisons'] if r['policy'] == policy and r['baseline'] == base
                      and (r['M'], r['N'], r['K_global']) in shapes]
            passed &= (len(points) == 2 and all(r['ci95_low'] > 0 for r in points))
        if passed:
            accepted.append(policy)
    return accepted


def main(run_root, repo):
    run_root = run_root.resolve()
    repo = repo.resolve()
    assert os.environ.get('SLURM_JOB_ID'), 'Run only inside the dedicated allocation'
    submission = json.loads((run_root/'submission.json').read_text())
    assert str(submission['job_id']) == os.environ['SLURM_JOB_ID'], 'Wrong allocation'
    tools = run_root/'source/tools/a800/arrival'
    artifacts = run_root/'artifacts'
    results = run_root/'results'
    results.mkdir(exist_ok=True)
    state = dict(job_id=submission['job_id'], started_at=now(), status='running', stages=[],
                 repetition_scope='fresh processes within one allocation; not independent jobs')

    def save():
        state['updated_at'] = now()
        tmp = run_root/'campaign-state.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(run_root/'campaign-state.json')

    def verify_snapshot():
        for name, digest in json.loads((run_root/'snapshot.json').read_text())['files'].items():
            assert sha(run_root/name) == digest, ('Frozen script changed', name)

    operator_python = repo/'outputs/a800/venv/bin/python'
    model_python = repo.parents[1]/'conda_envs/flux-megatron-a800/bin/python'
    assert operator_python.is_file() and model_python.is_file()
    common = dict(os.environ, SLURM_SUBMIT_DIR=str(repo), ARRIVAL_OUTPUT_ROOT=str(artifacts),
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
                  NCCL_DEBUG='WARN', TORCH_NCCL_ASYNC_ERROR_HANDLING='1', NCCL_IB_DISABLE='1',
                  NCCL_P2P_DISABLE='0', NCCL_P2P_LEVEL='NVL', FLUX_FORCE_NVLINK='1')

    def execute(name, command, env, directory):
        verify_snapshot()
        directory.mkdir(exist_ok=True)
        entry = dict(name=name, started_at=now(), command=list(map(str, command)), result=str(directory))
        state['stage'] = name
        state['stages'].append(entry)
        save()
        print(now(), 'START', name, flush=True)
        with (directory/'stage.log').open('w') as log:
            result = subprocess.run(entry['command'], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        entry.update(exit_code=result.returncode, finished_at=now())
        (directory/'exit-code.txt').write_text(str(result.returncode)+'\n')
        save()
        if result.returncode:
            print((directory/'stage.log').read_text()[-12000:], flush=True)
            raise RuntimeError(f'{name} failed with exit code {result.returncode}')
        print(now(), 'DONE', name, flush=True)

    def stage(name, stage_name, reverse=0, preflight=None, model=False):
        directory = results/name
        directory.mkdir()  # Never overwrite a completed/failed repetition.
        scripts = directory/'scripts'
        shutil.copytree(artifacts/'e2e-scripts' if model else tools, scripts,
                        ignore=shutil.ignore_patterns('__pycache__'))
        cuda = '/data/apps/cuda/12.4' if model else '/data/apps/cuda/12.8'
        python = model_python if model else operator_python
        env = dict(common, RESULT_DIR=str(directory), ARRIVAL_STAGE=stage_name, ARRIVAL_REVERSE=str(reverse),
                   CUDA_HOME=cuda, PATH=f'{python.parent}:{cuda}/bin:'+common['PATH'])
        if preflight:
            env['E2E_SCALE_PREFLIGHT'] = str(preflight)
        if model:
            env.update(E2E_CASE='l12-h2048-s2048-tp4', E2E_SCALE_STAGE=stage_name,
                       CUDA_DEVICE_MAX_CONNECTIONS='1', MAX_JOBS='2')
        execute(name, [python, scripts/'launch.py'], env, directory)
        return directory

    try:
        verify_snapshot()
        save()
        # Full physical-tile coverage on four sources, twice in fresh processes.
        calibration = stage('calibration-fit', 'calibrate')
        validation = stage('calibration-validation', 'calibrate')
        state['independent_calibration'] = str(validation)
        state['independent_calibration_used_for_fit'] = False
        fit_dir = results/'fit'
        execute('fit-tail', [operator_python, tools/'plan.py', calibration, artifacts/'plan.json', '--score', 'tail'], common, fit_dir)
        execute('fit-gap-ablation', [operator_python, tools/'plan.py', calibration, artifacts/'plan-gap.json', '--score', 'gap'], common, results/'fit-gap')
        execute('validate-labels', [operator_python, tools/'validate_calibration.py', calibration, validation,
                                   artifacts/'plan.json', results/'label-validation.json'], common, results/'validate-labels')
        execute('build', [operator_python, tools/'build.py', artifacts/'plan.json', artifacts/'build'], common, results/'build')

        checked = stage('operator-preflight', 'operator-preflight')
        assert json.loads((checked/'analysis.json').read_text())['completed']
        preflight = stage('model-preflight', 'preflight', model=True)
        assert json.loads((preflight/'preflight.json').read_text())['completed']
        operators = [stage(f'operator-{r+1}', 'operator', reverse=r, preflight=preflight) for r in range(2)]
        accepted = model_gate([json.loads((p/'analysis.json').read_text()) for p in operators])
        state['model_gate'] = dict(accepted_policies=accepted,
                                  rule='positive paired CI lower bound on BOTH model shapes in BOTH repetitions')
        save()
        if accepted:
            for r in range(2):
                stage(f'model-{r+1}', 'timing', reverse=r, preflight=preflight, model=True)
        else:
            state['model_timing'] = 'skipped: no repeatable increment on both model shapes; preflight completed'
        execute('report', [operator_python, tools/'campaign_report.py', run_root], common, results/'report')
        state.update(status='completed', stage='completed', finished_at=now())
        save()
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_root', type=Path)
    parser.add_argument('repo', type=Path)
    args = parser.parse_args()
    main(args.run_root, args.repo)

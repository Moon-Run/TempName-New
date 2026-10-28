"""Run build, full-model preflight, and two unconditional end-to-end rounds."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
from report import write_report

ROOT = Path(__file__).resolve().parent
INFO = json.loads((ROOT/'submission.json').read_text())
REPO = Path(INFO['repo'])
PYTHON = REPO.parents[1]/'conda_envs/flux-megatron-a800/bin/python'
now = lambda: datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert os.environ.get('SLURM_JOB_ID') == str(INFO['job_id'])
assert not (Path(INFO['original_campaign'])/'RELEASE').exists()
state = dict(job_id=INFO['job_id'], step_id=os.environ.get('SLURM_STEP_ID'),
             status='running', started_at=now(), stages=[], scope=INFO['scope'])


def save():
    state['updated_at'] = now()
    p = ROOT/'campaign-state.tmp'
    p.write_text(json.dumps(state, indent=2)+'\n')
    p.replace(ROOT/'campaign-state.json')


def execute(name, command, env, directory):
    for path, digest in INFO['files'].items():
        assert sha(ROOT/path) == digest, path
    assert sha(Path(INFO['artifact_root'])/'plan.json') == INFO['plan_sha256']
    directory.mkdir(exist_ok=True)
    entry = dict(name=name, started_at=now(), command=list(map(str, command)), result=str(directory))
    state['stages'].append(entry)
    state['stage'] = name
    save()
    print(now(), 'START', name, flush=True)
    with (directory/'stage.log').open('w') as log:
        result = subprocess.run(entry['command'], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    entry.update(exit_code=result.returncode, finished_at=now())
    (directory/'exit-code.txt').write_text(str(result.returncode)+'\n')
    save()
    if result.returncode:
        print((directory/'stage.log').read_text()[-16000:], flush=True)
        raise RuntimeError(entry)
    print(now(), 'DONE', name, flush=True)


def model_stage(name, stage, reverse=0):
    directory = ROOT/'results'/name
    directory.mkdir()
    shutil.copytree(ROOT/'scripts', directory/'scripts')
    cuda = '/data/apps/cuda/12.4'
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(REPO), ARRIVAL_OUTPUT_ROOT=INFO['artifact_root'],
               E2E_SCALE_PREFLIGHT=str(ROOT/'results/preflight'), RESULT_DIR=str(directory),
               ARRIVAL_REVERSE=str(reverse), E2E_CASE='l12-h2048-s2048-tp4', E2E_SCALE_STAGE=stage,
               CUDA_HOME=cuda, PATH=f'{PYTHON.parent}:{cuda}/bin:'+os.environ['PATH'],
               CUDA_DEVICE_MAX_CONNECTIONS='1', MAX_JOBS='2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', NCCL_DEBUG='WARN',
               TORCH_NCCL_ASYNC_ERROR_HANDLING='1', NCCL_IB_DISABLE='1', NCCL_P2P_DISABLE='0',
               NCCL_P2P_LEVEL='NVL', FLUX_FORCE_NVLINK='1')
    execute(name, [PYTHON, directory/'scripts/launch.py'], env, directory)
    artifact = 'preflight.json' if stage == 'preflight' else 'analysis.json'
    assert json.loads((directory/artifact).read_text())['completed']


def verify_mappings():
    previous = Path(INFO['original_campaign'])/'results/model-preflight'
    current = ROOT/'results/preflight'
    def coords(path):
        with path.open() as f:
            return [tuple(int(r[k]) for k in ('rank', 'index', 'm', 'n', 'destination'))
                    for r in csv.DictReader(f)]
    verified = []
    for policy in ('original', 'mlp_remote', 'mlp_remote_arrival'):
        for k in (2048, 8192):
            baseline = 'remote_first' if policy == 'mlp_remote' and k == 8192 else 'original'
            if policy == 'mlp_remote_arrival' and k == 8192:
                baseline = 'remote_arrival'
            actual = current/f'mapping-{policy}-k{k}.csv'
            reference = previous/f'mapping-{baseline}-k{k}.csv'
            assert coords(actual) == coords(reference), (policy, k, baseline)
            verified.append(dict(policy=policy, K_global=k, reference_policy=baseline,
                                 rows=len(coords(actual)), actual_sha256=sha(actual), reference=str(reference)))
    (ROOT/'mapping-equivalence.json').write_text(json.dumps(dict(completed=True, checks=verified), indent=2)+'\n')


try:
    save()
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    artifacts = Path(INFO['artifact_root'])
    execute('build', [PYTHON, ROOT/'build.py', REPO, artifacts/'build', ROOT/'check_mapping.cu',
                      artifacts/'plan.json'], env, ROOT/'results/build')
    model_stage('preflight', 'preflight')
    verify_mappings()
    for i in (1, 2):
        model_stage(f'model-{i}', 'timing', reverse=i-1)
        write_report(ROOT)
    state.update(status='completed', stage='completed', finished_at=now())
    save()
except BaseException:
    state.update(status='failed', error=traceback.format_exc())
    save()
    raise

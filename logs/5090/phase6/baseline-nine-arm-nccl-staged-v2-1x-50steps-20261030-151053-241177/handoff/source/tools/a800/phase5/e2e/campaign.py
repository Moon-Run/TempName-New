"""Validate codec variants, then measure full optimizer steps serially."""
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
BUILD = Path(INFO['artifact_root'])/'build'
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
    assert sha(BUILD/'manifest.json') == INFO['build_manifest_sha256']
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


def base_env():
    cuda = '/data/apps/cuda/12.4'
    return dict(os.environ, SLURM_SUBMIT_DIR=str(REPO), ARRIVAL_OUTPUT_ROOT=INFO['artifact_root'],
                CUDA_HOME=cuda, PATH=f'{PYTHON.parent}:{cuda}/bin:'+os.environ['PATH'],
                CUDA_DEVICE_MAX_CONNECTIONS='1', MAX_JOBS='2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', NCCL_DEBUG='WARN',
                TORCH_NCCL_ASYNC_ERROR_HANDLING='1', NCCL_IB_DISABLE='1', NCCL_P2P_DISABLE='0',
                NCCL_P2P_LEVEL='NVL', FLUX_FORCE_NVLINK='1')


def codec_stage(placement):
    name = 'codec-'+placement
    directory = ROOT/'results'/name
    directory.mkdir()
    if placement=='native_taco':
        env=dict(base_env(),PYTHONPATH=str(REPO.parent/'Megatron-LM'),
                 LD_LIBRARY_PATH='/data/apps/cuda/12.4/lib64')
        command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                 ROOT/'scripts/validate_megatron.py',directory]
        execute(name,command,env,directory)
        rows=[json.loads((directory/f'rank{i}.json').read_text()) for i in range(4)]
        assert all(r['completed'] and r['passed'] and r['ste_gradient_gather_exact'] for r in rows)
        return
    policy = placement+'_selective' if placement.endswith('_arrival') else {
        'remote_first':'remote_all','interleaved':'interleaved_all','separate':'taco_separate','fused':'taco_fused'}[placement]
    library = BUILD/policy
    env = dict(base_env(), PYTHONPATH=str(library/'python'),
               LD_LIBRARY_PATH=f'{library}:/data/apps/cuda/12.4/lib64')
    command = [PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4']
    if placement in ('separate','fused'):
        command += [ROOT/'scripts/validate_separate.py',directory,'--model-shapes','--placement',placement]
    else:
        base='remote_first' if placement=='remote_arrival' else placement.removesuffix('_arrival')
        command += [ROOT/'scripts/validate.py',directory,'--policy',base,'--plan',ROOT/'scripts/selection-plan.json']
        if json.loads((ROOT/'scripts/config.json').read_text()).get('graph_forward'):
            command += ['--graph']
        if json.loads((ROOT/'scripts/config.json').read_text()).get('double_buffered'):
            command += ['--double-buffered']
        config=json.loads((ROOT/'scripts/config.json').read_text())
        if config.get('scenario_extension'):
            case=config['cases'][0]
            command += ['--model-shape',str(case['sequence']*config['common']['micro_batch']),str(case['hidden']),str(case['ffn']//case['tp'])]
    execute(name, command, env, directory)
    rows = [json.loads((directory/f'rank{i}.json').read_text()) for i in range(4)]
    assert all(r['completed'] and r['passed'] for r in rows)


def model_stage(name, stage, reverse=0):
    directory = ROOT/'results'/name
    directory.mkdir()
    shutil.copytree(ROOT/'scripts', directory/'scripts')
    env = dict(base_env(), E2E_SCALE_PREFLIGHT=str(ROOT/'results/preflight'), RESULT_DIR=str(directory),
               ARRIVAL_REVERSE=str(reverse), E2E_CASE=json.loads((ROOT/'scripts/config.json').read_text())['cases'][0]['name'], E2E_SCALE_STAGE=stage)
    execute(name, [PYTHON, directory/'scripts/launch.py'], env, directory)
    artifact = 'preflight.json' if stage == 'preflight' else 'analysis.json'
    assert json.loads((directory/artifact).read_text())['completed']


try:
    save()
    directory=ROOT/'results/mapping'
    directory.mkdir()
    execute('mapping',[PYTHON,ROOT/'check_mappings.py',ROOT],base_env(),directory)
    for placement in INFO.get('codec_checks',['remote_first','interleaved','separate']):
        codec_stage(placement)
    model_stage('preflight', 'preflight')
    for i in (1, 2):
        model_stage(f'model-{i}', 'timing', reverse=i-1)
        write_report(ROOT)
    state.update(status='completed', stage='completed', finished_at=now())
    save()
except BaseException:
    state.update(status='failed', error=traceback.format_exc())
    save()
    raise

"""Validated four-policy end-to-end experiment in an ordinary Slurm allocation."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(os.environ['SLURM_SUBMIT_DIR'])
OUT = Path(os.environ['RESULT_DIR'])
SCRIPTS = OUT/'scripts'
CONFIG = json.loads((SCRIPTS/'config.json').read_text())
BUILD = ROOT/'outputs/a800/phase3/three-way-build'
MEGATRON = ROOT.parent/'Megatron-LM'
STAGE = os.environ.get('E2E_STAGE','validate')
EXPLORATORY = os.environ.get('E2E_ALLOW_EXPLORATORY') == '1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((BUILD/'manifest.json').read_text())
assert not manifest['sampling']
for name, expected in manifest['files'].items():
    assert sha(BUILD/name) == expected, name
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=MEGATRON,text=True).strip()==CONFIG['megatron_commit']
shutil.copy2(BUILD/'manifest.json',OUT/'flux-manifest.json')
protocol = dict(config=CONFIG,stage=STAGE,exploratory=EXPLORATORY,script_sha256={p.name:sha(p) for p in SCRIPTS.iterdir() if p.is_file()},
                flux_manifest_sha256=sha(BUILD/'manifest.json'))
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
commands=[]
args=[
 '--num-layers','4','--hidden-size','1024','--ffn-hidden-size','4096','--num-attention-heads','16',
 '--seq-length','1024','--max-position-embeddings','1024','--transformer-impl','local',
 '--position-embedding-type','learned_absolute','--untie-embeddings-and-output-weights',
 '--no-persist-layer-norm','--no-masked-softmax-fusion','--no-bias-gelu-fusion','--no-bias-dropout-fusion','--no-rope-fusion',
 '--micro-batch-size','1','--global-batch-size','4','--train-iters','1000','--bf16','--optimizer','adam',
 '--lr','1e-4','--min-lr','1e-4','--lr-decay-style','constant','--weight-decay','0.01',
 '--adam-beta1','0.9','--adam-beta2','0.95','--clip-grad','1.0','--init-method-std','0.02',
 '--seed','1234','--hidden-dropout','0','--attention-dropout','0','--no-gradient-accumulation-fusion',
 '--tensor-model-parallel-size','4','--pipeline-model-parallel-size','1','--context-parallel-size','1',
 '--sequence-parallel','--mock-data','--tokenizer-type','NullTokenizer','--vocab-size','8192',
 '--split','100,0,0','--num-workers','0','--eval-iters','0','--timing-log-level','0',
 '--no-check-for-nan-in-loss-and-grad']

def run(policy,mode,name,extra=None,allow_numeric_failure=False):
    directory=OUT/name;directory.mkdir()
    env=dict(os.environ,E2E_POLICY=policy,E2E_MODE=mode,E2E_PROCESS_OUT=str(directory),E2E_BUILD=str(BUILD),
        PYTHONPATH=f'{MEGATRON}:{BUILD/policy/"python"}' if policy!='native' else str(MEGATRON),
        LD_LIBRARY_PATH=f'{BUILD/policy}:{os.environ["CUDA_HOME"]}/lib64' if policy!='native' else f'{os.environ["CUDA_HOME"]}/lib64')
    env.update(extra or {})
    process_args = args.copy()
    if env.get('E2E_FP32')=='1':
        process_args.remove('--bf16')
    if env.get('E2E_SEED'):
        process_args[process_args.index('--seed')+1] = env['E2E_SEED']
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(SCRIPTS/'worker.py'),*process_args]
    commands.append(dict(name=name,command=cmd,environment={k:env[k] for k in ['E2E_POLICY','E2E_MODE','PYTHONPATH','LD_LIBRARY_PATH']},extra=extra or {}))
    (OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    print('START',name,flush=True)
    with (directory/'process.log').open('w') as log:
        result=subprocess.run(cmd,env=env,cwd=MEGATRON,stdout=log,stderr=subprocess.STDOUT)
    (directory/'exit-code.txt').write_text(str(result.returncode)+'\n')
    complete_reports = all((directory/f'rank{rank}.json').exists() for rank in range(4))
    numeric_failure = allow_numeric_failure and mode=='validate' and complete_reports and all(json.loads((directory/f'rank{rank}.json').read_text()).get('completed') for rank in range(4))
    if result.returncode and not numeric_failure:
        print((directory/'process.log').read_text()[-12000:],flush=True)
        raise RuntimeError(f'{name} failed with {result.returncode}')
    for rank in range(4):
        r=json.loads((directory/f'rank{rank}.json').read_text())
        assert r['completed'] and (r['passed'] or numeric_failure)
    print('DONE',name,flush=True)

if STAGE=='calibrate':
    for seed in [101,202]:
        with tempfile.TemporaryDirectory(prefix='flux-e2e-calibrate-',dir='/tmp') as reference:
            common=dict(E2E_REFERENCE=reference,E2E_SEED=str(seed))
            run('native','calibrate',f'calibration-{seed}-bf16',dict(common,E2E_WRITE_REFERENCE='1'))
            run('native','calibrate',f'calibration-{seed}-fp32',dict(common,E2E_FP32='1'))
    # Isolate the adapter's backward implementation with an unchanged native forward.
    with tempfile.TemporaryDirectory(prefix='flux-e2e-autograd-',dir='/tmp') as reference:
        run('native','validate','autograd-native',dict(E2E_REFERENCE=reference,E2E_WRITE_REFERENCE='1'))
        run('native','validate','autograd-control',dict(E2E_REFERENCE=reference,E2E_CONTROL_BACKWARD='1'))
    print('CALIBRATION_COMPLETE: derive budgets from native-only records before validation',flush=True)
elif STAGE=='validate':
    for policy in CONFIG['policies'][1:]:
        for m,n,k in CONFIG['shapes']:
            stem=f'mapping-{policy}-m{m}-n{n}-k{k}'
            with (OUT/(stem+'.csv')).open('w') as out,(OUT/(stem+'.log')).open('w') as err:
                subprocess.run([str(BUILD/policy/'check_mapping'),str(m),str(n),str(k),'4',
                                str(CONFIG['policies'].index(policy)-1)],stdout=out,stderr=err,check=True)
    with tempfile.TemporaryDirectory(prefix='flux-e2e-reference-',dir='/tmp') as reference:
        run('native','validate','validation-native',dict(E2E_REFERENCE=reference,E2E_WRITE_REFERENCE='1'))
        run('native','validate','validation-native-repeat',dict(E2E_REFERENCE=reference))
        for policy in CONFIG['policies'][1:]:
            run(policy,'validate',f'validation-{policy}',dict(E2E_REFERENCE=reference),allow_numeric_failure=True)
    admission=[dict(directory=p.parent.name,rank=r['rank'],passed=r['passed']) for p in sorted(OUT.glob('validation-*/rank*.json')) for r in [json.loads(p.read_text())]]
    (OUT/'admission.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in admission),records=admission),indent=2)+'\n')
    if not all(r['passed'] for r in admission) and not EXPLORATORY:
        raise RuntimeError('Numerical admission failed; all policies checked, no performance run authorized')
    for policy in CONFIG['policies']:
        run(policy,'pilot',f'pilot-{policy}')
    # All diagnostics run separately after the pilot, never in formal timing windows.
    for policy in CONFIG['policies']:
        run(policy,'profile',f'profile-{policy}')
    subprocess.run([sys.executable,str(SCRIPTS/'summarize.py'),str(OUT)],check=True)
else:
    assert STAGE in ('timing','exploratory-timing')
    assert (STAGE=='exploratory-timing') == EXPLORATORY
    validated=Path(os.environ['E2E_VALIDATION_DIR'])
    gate=json.loads((validated/'validation.json').read_text())
    assert (gate['passed'] or (EXPLORATORY and gate.get('execution_passed'))) and gate['flux_manifest_sha256']==protocol['flux_manifest_sha256']
    # Changes to implementation/config after validation require validating again.
    for name in ['adapter.py','worker.py','launch.py','config.json']:
        assert gate['script_sha256'][name]==protocol['script_sha256'][name],name
    shutil.copy2(validated/'validation.json',OUT/'validation-gate.json')
    (OUT/'validation-source.txt').write_text(str(validated)+'\n')
    window=0
    for block,order in enumerate(CONFIG['orders']):
        for policy in order:
            run(policy,'timing',f'window-{window:02d}-{policy}',dict(E2E_BLOCK=str(block),E2E_WINDOW=str(window)))
            window+=1
    subprocess.run([sys.executable,str(SCRIPTS/'summarize.py'),str(OUT)],check=True)
print('E2E_JOB_PASS',STAGE,OUT,flush=True)

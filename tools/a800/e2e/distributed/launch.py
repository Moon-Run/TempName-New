"""Single coordinator: each window launches one torchrun task per node via srun."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from config import validate
from summarize import analyze, check_window

ROOT=Path(os.environ['E2E_DIST_ROOT'])
OUT=Path(os.environ['E2E_DIST_OUT'])
SCRIPTS=OUT/'scripts'
C=validate(json.loads((OUT/'config.json').read_text()))
MODEL=C
MEGATRON=ROOT.parent/'Megatron-LM'
BUILD=ROOT/'outputs/a800/phase3/three-way-build'
MAPPING=ROOT/'outputs/a800/megatron-e2e/scale-mapping'
STAGE=os.environ['E2E_DIST_STAGE']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

assert int(os.environ['SLURM_JOB_NUM_NODES']) == C['nodes']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=MEGATRON,text=True).strip() == '3ea68ad6042cc1204386ae9364358f7c4de1bc37'
flux_policies=[p for p in C['policies'] if p!='native']
manifest_hash=None
if flux_policies:
    manifest=json.loads((BUILD/'manifest.json').read_text())
    for name,value in manifest['files'].items():
        assert sha(BUILD/name)==value,name
    manifest_hash=sha(BUILD/'manifest.json')
    shutil.copy2(BUILD/'manifest.json', OUT/'flux-manifest.json')
    mapping=json.loads((MAPPING/'manifest.json').read_text())
    for name,value in mapping['files'].items():
        assert sha(Path(name))==value,name
protocol=dict(config=C, stage=STAGE, flux_manifest_sha256=manifest_hash,
              mapping_manifest_sha256=sha(MAPPING/'manifest.json') if flux_policies else None,
              scripts={p.name:sha(p) for p in SCRIPTS.iterdir() if p.is_file()})
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
if STAGE=='timing':
    preflight=Path(os.environ['E2E_DIST_PREFLIGHT'])
    checked=json.loads((preflight/'preflight.json').read_text())
    assert (preflight/'exit-code.txt').read_text().strip()=='0'
    assert checked['completed'] and checked['config']==C
    assert checked['mapping_manifest_sha256']==protocol['mapping_manifest_sha256']
    assert checked['scripts']==protocol['scripts'] and checked['flux_manifest_sha256']==manifest_hash
    shutil.copy2(preflight/'preflight.json',OUT/'preflight-source.json')
args=[]
for flag,key in [('num-layers','layers'),('hidden-size','hidden'),('ffn-hidden-size','ffn'),
                 ('num-attention-heads','heads'),('seq-length','sequence'),('max-position-embeddings','sequence'),
                 ('micro-batch-size','micro_batch'),('global-batch-size','global_batch'),('seed','seed'),
                 ('vocab-size','vocab'),('tensor-model-parallel-size','tp')]:args += ['--'+flag,str(MODEL[key])]
args += ['--transformer-impl','local','--position-embedding-type','learned_absolute',
 '--untie-embeddings-and-output-weights','--no-persist-layer-norm','--no-masked-softmax-fusion',
 '--no-bias-gelu-fusion','--no-bias-dropout-fusion','--no-rope-fusion','--train-iters',str(max(1000,C['warmup_steps']+C['timed_steps']+3)),'--bf16',
 '--optimizer','adam','--lr','1e-4','--min-lr','1e-4','--lr-decay-style','constant','--weight-decay','0.01',
 '--adam-beta1','0.9','--adam-beta2','0.95','--clip-grad','1.0','--init-method-std','0.02',
 '--hidden-dropout','0','--attention-dropout','0','--no-gradient-accumulation-fusion',
 '--pipeline-model-parallel-size','1','--context-parallel-size','1','--sequence-parallel',
 '--mock-data','--tokenizer-type','NullTokenizer','--split','100,0,0','--num-workers','0','--eval-iters','0',
 '--timing-log-level','0','--no-check-for-nan-in-loss-and-grad']

commands=[]
def run(policy,mode,name,block=-1):
    directory=OUT/name
    directory.mkdir()
    env=dict(os.environ,E2E_POLICY=policy,E2E_MODE=mode,E2E_PROCESS_OUT=str(directory),
             E2E_BUILD=str(BUILD),E2E_BLOCK=str(block),E2E_MEGATRON_ARGS=json.dumps(args))
    env['PYTHONPATH']=str(MEGATRON)+(f':{BUILD/policy/"python"}' if policy!='native' else '')
    env['LD_LIBRARY_PATH']=f'{BUILD/policy}:' if policy!='native' else ''
    env['LD_LIBRARY_PATH']+=f'{os.environ["CUDA_HOME"]}/lib64:'+os.environ.get('LD_LIBRARY_PATH','')
    cmd=['srun','--kill-on-bad-exit=1',f'--nodes={C["nodes"]}',f'--ntasks={C["nodes"]}',
         '--ntasks-per-node=1','--gpu-bind=none',sys.executable,str(SCRIPTS/'node.py')]
    commands.append(dict(name=name,command=cmd,policy=policy,mode=mode))
    (OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    print('START',name,flush=True)
    result=subprocess.run(cmd,env=env,cwd=MEGATRON)
    (directory/'exit-code.txt').write_text(str(result.returncode)+'\n')
    if result.returncode:
        raise RuntimeError(f'{name} failed; inspect {directory}/node*.log')
    check_window(directory,C,policy,mode)
    print('DONE',name,flush=True)

if STAGE=='preflight':
    for policy in flux_policies:
        for k in [C['hidden'],C['ffn']]:
            # Each node validates its physical devices; K argument reconstructs local K.
            cmd=['srun','--kill-on-bad-exit=1','--ntasks-per-node=1','--gpu-bind=none',
                 f'--output={OUT}/mapping-{policy}-k{k}-node%t.csv',
                 f'--error={OUT}/mapping-{policy}-k{k}-node%t.log',
                 str(MAPPING/f'check_mapping-{policy}'),str(C['sequence']*C['micro_batch']//C['tp_nodes']),
                 str(C['hidden']),str(k//C['tp_nodes']),str(C['local_tp']),
                 str(['original','remote_first','interleaved'].index(policy))]
            subprocess.run(cmd,check=True)
    for policy in C['policies']:
        run(policy,'smoke','smoke-'+policy)
        run(policy,'profile','profile-'+policy)
    protocol.update(completed=True, scope='Mapping, actual kernel profile and forward reference checks; finite short training and DP replica consistency. Not a convergence or paired-gradient certificate.')
    (OUT/'preflight.json').write_text(json.dumps(protocol,indent=2)+'\n')
else:
    for block,order in enumerate(C['orders']):
        for policy in order:
            run(policy,'timing',f'block-{block:03d}-{policy}',block)
    analyze(OUT)
print('DISTRIBUTED_E2E_COMPLETE',OUT,flush=True)

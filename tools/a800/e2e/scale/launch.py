"""Run one explicitly configured model/TP case with all four implementations."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['SLURM_SUBMIT_DIR'])
OUT=Path(os.environ['RESULT_DIR']);SCRIPTS=OUT/'scripts'
PLAN=json.loads((SCRIPTS/'config.json').read_text())
CASE=next(c for c in PLAN['cases'] if c['name']==os.environ['E2E_CASE'])
MODEL=dict(PLAN['common'],**CASE);TP=CASE['tp']
BUILD=ROOT/'outputs/a800/phase3/three-way-build'
MAPPING=ROOT/'outputs/a800/megatron-e2e/scale-mapping'
MEGATRON=ROOT.parent/'Megatron-LM'
STAGE=os.environ.get('E2E_SCALE_STAGE','preflight')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=MEGATRON,text=True).strip()==PLAN['megatron_commit']
manifest=json.loads((BUILD/'manifest.json').read_text())
for name,value in manifest['files'].items():assert sha(BUILD/name)==value,name
mapping_manifest=json.loads((MAPPING/'manifest.json').read_text())
for name,value in mapping_manifest['files'].items():assert sha(Path(name))==value,name
shutil.copy2(BUILD/'manifest.json',OUT/'flux-manifest.json')
shutil.copy2(MAPPING/'manifest.json',OUT/'mapping-build.json')
protocol=dict(plan=PLAN,case=CASE,model=MODEL,stage=STAGE,
    flux_manifest_sha256=sha(BUILD/'manifest.json'),script_sha256={p.name:sha(p) for p in SCRIPTS.iterdir() if p.is_file()},
    measurement='Metrics experiment; previous small-model numerical budgets are not applied.')
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
args=[]
for flag,key in [('num-layers','layers'),('hidden-size','hidden'),('ffn-hidden-size','ffn'),
                 ('num-attention-heads','heads'),('seq-length','sequence'),('max-position-embeddings','sequence'),
                 ('micro-batch-size','micro_batch'),('global-batch-size','global_batch'),('seed','seed'),
                 ('vocab-size','vocab'),('tensor-model-parallel-size','tp')]:args += ['--'+flag,str(MODEL[key])]
args += ['--transformer-impl','local','--position-embedding-type','learned_absolute',
 '--untie-embeddings-and-output-weights','--no-persist-layer-norm','--no-masked-softmax-fusion',
 '--no-bias-gelu-fusion','--no-bias-dropout-fusion','--no-rope-fusion','--train-iters','1000','--bf16',
 '--optimizer','adam','--lr','1e-4','--min-lr','1e-4','--lr-decay-style','constant','--weight-decay','0.01',
 '--adam-beta1','0.9','--adam-beta2','0.95','--clip-grad','1.0','--init-method-std','0.02',
 '--hidden-dropout','0','--attention-dropout','0','--no-gradient-accumulation-fusion',
 '--pipeline-model-parallel-size','1','--context-parallel-size','1','--sequence-parallel',
 '--mock-data','--tokenizer-type','NullTokenizer','--split','100,0,0','--num-workers','0','--eval-iters','0',
 '--timing-log-level','0','--no-check-for-nan-in-loss-and-grad']
commands=[]
def run(policy,mode,name,extra=None):
    directory=OUT/name;directory.mkdir()
    env=dict(os.environ,E2E_POLICY=policy,E2E_MODE=mode,E2E_PROCESS_OUT=str(directory),E2E_BUILD=str(BUILD),
        PYTHONPATH=f'{MEGATRON}:{BUILD/policy/"python"}' if policy!='native' else str(MEGATRON),
        LD_LIBRARY_PATH=f'{BUILD/policy}:{os.environ["CUDA_HOME"]}/lib64' if policy!='native' else f'{os.environ["CUDA_HOME"]}/lib64')
    env.update(extra or {})
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={TP}',str(SCRIPTS/'worker.py'),*args]
    commands.append(dict(name=name,command=cmd,policy=policy,mode=mode,extra=extra or {}))
    (OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    print('START',CASE['name'],name,flush=True)
    with (directory/'process.log').open('w') as log:result=subprocess.run(cmd,cwd=MEGATRON,env=env,stdout=log,stderr=subprocess.STDOUT)
    (directory/'exit-code.txt').write_text(str(result.returncode)+'\n')
    if result.returncode:
        print((directory/'process.log').read_text()[-16000:],flush=True)
        raise RuntimeError(f'{name} failed: {result.returncode}')
    for rank in range(TP):
        r=json.loads((directory/f'rank{rank}.json').read_text());assert r['passed'] and r['completed']
    print('DONE',CASE['name'],name,flush=True)

if STAGE=='preflight':
    for policy in PLAN['policies'][1:]:
        for k in [MODEL['hidden'],MODEL['ffn']]:
            stem=f'mapping-{policy}-k{k}'
            cmd=[str(MAPPING/f'check_mapping-{policy}'),str(MODEL['sequence']),str(MODEL['hidden']),str(k),str(TP),str(PLAN['policies'].index(policy)-1)]
            with (OUT/(stem+'.csv')).open('w') as f,(OUT/(stem+'.log')).open('w') as err:
                subprocess.run(cmd,stdout=f,stderr=err,check=True)
    for policy in PLAN['policies']:run(policy,'smoke','smoke-'+policy)
    for policy in PLAN['policies']:run(policy,'profile','profile-'+policy)
elif STAGE=='timing':
    preflight=Path(os.environ['E2E_SCALE_PREFLIGHT'])
    checked=json.loads((preflight/'preflight.json').read_text())
    assert checked['completed'] and checked['case']==CASE and checked['flux_manifest_sha256']==protocol['flux_manifest_sha256']
    for name in ['worker.py','adapter.py','launch.py','config.json']:
        assert checked['script_sha256'][name]==protocol['script_sha256'][name],name
    shutil.copy2(preflight/'preflight.json',OUT/'preflight.json')
    window=0
    for block,order in enumerate(PLAN['orders']):
        for policy in order:
            run(policy,'timing',f'window-{window:02d}-{policy}',dict(E2E_BLOCK=str(block),E2E_WINDOW=str(window)))
            window+=1
else:raise ValueError(STAGE)
subprocess.run([sys.executable,str(SCRIPTS/'summarize.py'),str(OUT)],check=True)
print('SCALE_JOB_COMPLETE',CASE['name'],STAGE,OUT,flush=True)

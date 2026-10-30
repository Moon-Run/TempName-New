"""Standalone single-node torchrun campaign. Defaults to printing commands only."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import MEGATRON_COMMIT, orders, read, require, shape, sha, verify_files, write
from calibrate import environment
from report import preflight, report, verify_inputs, window


def arguments(c):
    args=[]
    for flag,key in [('num-layers','layers'),('hidden-size','hidden'),('ffn-hidden-size','ffn'),
                     ('num-attention-heads','heads'),('seq-length','sequence'),('max-position-embeddings','sequence'),
                     ('micro-batch-size','micro_batch'),('global-batch-size','global_batch'),('seed','seed'),
                     ('vocab-size','vocab'),('tensor-model-parallel-size','tp')]:
        args += ['--'+flag,str(c[key])]
    return args+['--transformer-impl','local','--position-embedding-type','learned_absolute',
        '--untie-embeddings-and-output-weights','--no-persist-layer-norm','--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion','--no-bias-dropout-fusion','--no-rope-fusion','--train-iters','1000','--bf16',
        '--optimizer','adam','--lr','1e-4','--min-lr','1e-4','--lr-decay-style','constant','--weight-decay','0.01',
        '--adam-beta1','0.9','--adam-beta2','0.95','--clip-grad','1.0','--init-method-std','0.02',
        '--hidden-dropout','0','--attention-dropout','0','--no-gradient-accumulation-fusion',
        '--pipeline-model-parallel-size','1','--context-parallel-size','1','--sequence-parallel',
        '--mock-data','--tokenizer-type','NullTokenizer','--split','100,0,0','--num-workers','0','--eval-iters','0',
        '--timing-log-level','0','--no-check-for-nan-in-loss-and-grad']


def schedule(info,stage):
    if stage in ('preflight','all'):
        for mode in ('smoke','profile'):
            for p in info['policies']:yield mode,p,f'{mode}-{p}',-1,-1
    if stage in ('timing','all'):
        for rep,blocks in enumerate(orders(info['config']),1):
            index=0
            for block,ps in enumerate(blocks):
                for p in ps:
                    yield 'timing',p,f'round{rep}/window-{index:03d}-{p}',block,index
                    index+=1


def main(root,stage,execute,python,cuda_home):
    root=root.resolve(); info=verify_inputs(root);c=info['config'];scripts=root/'scripts'
    verify_files('/',info['controllers'])
    megatron=Path(info['megatron'])
    require(subprocess.check_output(['git','-C',str(megatron),'rev-parse','HEAD'],text=True).strip()==MEGATRON_COMMIT,'Megatron revision changed')
    require(not subprocess.check_output(['git','-C',str(megatron),'status','--porcelain','--untracked-files=no'],text=True).strip(),'Megatron working tree changed')
    cmd=[str(python),'-m','torch.distributed.run','--standalone',f'--nproc_per_node={c["tp"]}',str(scripts/'worker.py'),*arguments(c)]
    jobs=list(schedule(info,stage))
    if not execute:
        print('GPU execution is disabled; pass --execute on the H100 host only.')
        print('Command:',cmd)
        for mode,p,name,block,index in jobs:print(mode,p,name,'block',block,'window',index)
        return
    # Exclusive run lock; a failed run stays frozen for inspection.
    import fcntl
    lock=(root/'.run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=root/f'{stage}-state.json'
    require(not state.exists(),'This stage already started; prepare a new run instead of overwriting')
    write(state,dict(status='running',stage=stage))
    gated=False
    try:
        if stage in ('preflight','all'):
            for p,v in info['variants'].items():
                if v['role']!='candidate':continue
                lib=Path(v['root']);env=environment(python,lib,c,info['run_id'],scripts,cuda_home)
                with (root/'results'/f'mapping-{p}.csv').open('w') as f:
                    subprocess.run([str(lib/'check_mapping'),*map(str,shape(c)),str(c['tp']),str(lib/'libflux_cuda.so')],env=env,stdout=f,check=True)
        for mode,p,name,block,index in jobs:
            if mode=='timing' and not gated:
                gate=preflight(root) if stage=='all' else read(root/'preflight.json')
                require(gate['completed'] and gate['submission_sha256']==sha(root/'submission.json'),'Preflight required before timing')
                verify_files(root,gate['files']);gated=True
            verify_inputs(root)
            directory=root/'results'/name;directory.mkdir(parents=True,exist_ok=False)
            lib=Path(info['variants'][p]['root']) if p in info['variants'] else root/'no-flux-library'
            env=environment(python,lib,c,info['run_id'],scripts,cuda_home)
            env.update(E2E_POLICY=p,E2E_MODE=mode,E2E_CASE='h100-phase6',E2E_BUILD=str(root/'build'),
                E2E_PROCESS_OUT=str(directory),E2E_BLOCK=str(block),E2E_WINDOW=str(index))
            env['PYTHONPATH']=f'{scripts}:{megatron}'+(f':{lib}/python' if p in info['variants'] else '')
            write(directory/'command.json',dict(command=cmd,policy=p,mode=mode,block=block,window=index))
            print('START',name,flush=True)
            with (directory/'process.log').open('w') as log:
                subprocess.run(cmd,cwd=megatron,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            window(root,directory,p,mode,block,index)
        if stage=='preflight':preflight(root)
        else:report(root)
        write(state,dict(status='completed',stage=stage))
    except BaseException:
        write(state,dict(status='failed',stage=stage));raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path)
    p.add_argument('--stage',choices=['preflight','timing','all'],default='all');p.add_argument('--execute',action='store_true')
    p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--cuda-home',type=Path)
    a=p.parse_args();main(a.root,a.stage,a.execute,a.python,a.cuda_home)

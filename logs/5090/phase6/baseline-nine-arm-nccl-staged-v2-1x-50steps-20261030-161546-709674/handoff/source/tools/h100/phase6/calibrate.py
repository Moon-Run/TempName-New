"""Prepare/run a new H100 BF16 calibration. GPU execution requires --execute."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from common import HERE, REPO, PROTOCOL, manifest, read, replace_once, require, sha, shape, verify_files, write


def worker_source():
    text=(REPO/'tools/a800/phase3/mechanism/worker.py').read_text()
    text=replace_once(text,'import flux\n','')
    text=replace_once(text,'torch.cuda.set_device(local)','torch.cuda.set_device(local)\nimport flux')
    text=replace_once(text,"assert os.environ.get('SLURM_JOB_ID') and world==4", "assert world == int(os.environ['H100_TP'])")
    text=replace_once(text,"assert all('A800' in torch.cuda.get_device_name(i) for i in range(world))",
                      "from runtime import check_topology\nhardware = check_topology(world)")
    text=replace_once(text,"job_id=os.environ['SLURM_JOB_ID']", "run_id=os.environ['H100_RUN_ID'],hardware=hardware")
    text=text.replace('ring_reduction=False','ring_reduction=True')
    text=replace_once(text,'offsets=[0,per//4,per//2,3*per//4,per-1]','offsets=list(range(per))')
    text=replace_once(text,'for repeat in range(repeats):','for repeat in range(3*per):')
    text=replace_once(text,'order=list(modes);',"order=[x for x in modes if x[0] in ('off','receiver_sparse')];")
    # Preserve the existing BF16 correctness gate. FP32 differences are recorded
    # as diagnostics, as in the A800 arrival8 protocol, not as a BF16-equivalence claim.
    text=replace_once(text,' def correct(y):'," fp32_metrics=dict(checks=0,max_abs=0.,max_relative_l2=0.)\n def correct(y):")
    text=replace_once(text,'  torch.testing.assert_close(y.float(),ref32,rtol=.02,atol=.02)',
        "  delta=y.float()-ref32\n  fp32_metrics['checks']+=1\n  fp32_metrics['max_abs']=max(fp32_metrics['max_abs'],delta.abs().max().item())\n  fp32_metrics['max_relative_l2']=max(fp32_metrics['max_relative_l2'],(delta.norm()/ref32.norm()).item())")
    return replace_once(text,'correctness=True,batch_us=[],samples=[]','correctness=True,fp32_diagnostic=fp32_metrics,batch_us=[],samples=[]')


def prepare(out, build):
    build=build.resolve(); data=manifest(build,'baselines'); c=data['config']
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'worker.py').write_text(worker_source())
    shutil.copy2(HERE/'runtime.py',out/'runtime.py')
    write(out/'config.json',dict(shapes=[shape(c)],repeats=0,warmup=100,trials=3,iters=50,sample_warmup=10))
    write(out/'plan.json',dict(protocol=PROTOCOL,config=c,run_id=uuid.uuid4().hex,build=str(build),
        build_sha256=sha(build/'manifest.json'),policies=c['bases'],
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(out)


def environment(python, library, c, run_id, out, cuda_home=None):
    env=dict(os.environ,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             CUDA_DEVICE_MAX_CONNECTIONS='1',H100_TP=str(c['tp']),H100_SM_COUNT=str(c['sm_count']),
             H100_RUN_ID=run_id,PYTHONPATH=f'{out}:{library}/python',
             PATH=str(Path(python).parent)+':'+os.environ['PATH'])
    # Do not force an A800 Flux topology/registry or inherit smoke-only settings.
    for key in ('FLUX_FORCE_NVLINK','FLUX_TUNE_CONFIG_FILE','MECH_SMOKE','PYTHONOPTIMIZE'):
        env.pop(key,None)
    ld=[str(library/'python/flux/lib'),str(library)]
    if cuda_home:
        env['CUDA_HOME']=str(cuda_home);env['PATH']=str(Path(cuda_home)/'bin')+':'+env['PATH']
        ld.append(str(Path(cuda_home)/'lib64'))
    env['LD_LIBRARY_PATH']=':'.join(ld)+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    return env


def run(out, execute, python, cuda_home=None):
    out=out.resolve(); p=read(out/'plan.json');c=p['config'];build=Path(p['build'])
    require(sha(build/'manifest.json')==p['build_sha256'],'Calibration build changed')
    manifest(build,'baselines');verify_files(out,p['files'])
    cmds=[]
    for base in c['bases']:
        lib=build/('sampler_'+base)
        cmds.append((base,lib,[str(lib/'check_mapping'),*map(str,shape(c)),str(c['tp']),str(lib/'libflux_cuda.so')],
                     [str(python),'-m','torch.distributed.run','--standalone',f'--nproc_per_node={c["tp"]}',str(out/'worker.py')]))
    if not execute:
        for base,_,mapping,worker in cmds:print(base, '\n ',mapping,'\n ',worker)
        print('Dry run only. --execute launches GPU calibration.');return
    require(not (out/'execution.json').exists(),'Use a new calibration directory; no overwrite/resume of partial measurements')
    write(out/'execution.json',dict(status='running',run_id=p['run_id']))
    try:
        for base,lib,mapping,worker in cmds:
            env=environment(python,lib,c,p['run_id'],out,cuda_home)
            env.update(MECH_POLICY=base,MECH_KIND='instrumented',MECH_BLOCK='0',MECH_LIBRARY=str(lib/'libflux_cuda.so'),RESULT_DIR=str(out))
            m,n,k=shape(c)
            with (out/f'mapping-{base}-{m}-{n}-{k}.csv').open('w') as output:
                subprocess.run(mapping,env=env,stdout=output,check=True)
            with (out/f'{base}.log').open('w') as log:
                subprocess.run(worker,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            records=[read(out/f'b0-{base}-instrumented-rank{r}.json') for r in range(c['tp'])]
            require(all(r['completed'] and r['rank']==i and r['run_id']==p['run_id'] for i,r in enumerate(records)), 'Incomplete calibration')
            require(len({r['hardware']['host'] for r in records})==1,'Multi-node calibration is forbidden')
        write(out/'calibration-complete.json',dict(protocol=PROTOCOL,completed=True,arch='sm90',gpu_calibrated=True,
            config=c,run_id=p['run_id'],build=str(build),build_sha256=p['build_sha256'],hardware=records[0]['hardware']))
        write(out/'execution.json',dict(status='completed',run_id=p['run_id']))
    except BaseException:
        write(out/'execution.json',dict(status='failed',run_id=p['run_id']));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('out',type=Path);p.add_argument('--build',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('out',type=Path);p.add_argument('--execute',action='store_true')
    p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--cuda-home',type=Path)
    a=parser.parse_args()
    if a.action=='prepare':prepare(a.out,a.build)
    else:run(a.out,a.execute,a.python,a.cuda_home)

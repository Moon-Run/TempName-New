"""Run unchanged controls around each sampled process; keep all diagnostics separate."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['SLURM_SUBMIT_DIR']);OUT=Path(os.environ['RESULT_DIR']);SCRIPTS=OUT/'scripts'
BASE=ROOT/'outputs/a800/phase3/three-way-build';BUILD=ROOT/'outputs/a800/phase3/mechanism-build'
config=json.loads((SCRIPTS/'config.json').read_text());smoke=os.environ.get('MECH_SMOKE')=='1'
for name,path in [('parent',BASE),('instrumented',BUILD)]:
 manifest=json.loads((path/'manifest.json').read_text())
 for file,digest in manifest['files'].items():assert hashlib.sha256((path/file).read_bytes()).hexdigest()==digest,file
 if name=='instrumented':assert hashlib.sha256((SCRIPTS/'config.json').read_bytes()).hexdigest()==manifest['config_sha256']
 shutil.copy2(path/'manifest.json',OUT/(name+'-manifest.json'))
shutil.copy2(BUILD/'instrumentation.patch',OUT/'instrumentation.patch')
(OUT/'experiment.json').write_text(json.dumps(dict(config,smoke=smoke,sample_warmup=config.get('sample_warmup',10)),indent=2)+'\n')
commands=[]
if not smoke:
 # Permission can differ between nodes; probe the actual measurement allocation.
 args=['/data/apps/cuda/12.8/bin/ncu','--metrics',
       'dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct',
       '--csv','--log-file',str(OUT/'counter-probe.csv'),str(ROOT/'outputs/a800/phase3/counter_probe')]
 with (OUT/'counter-probe.log').open('w') as f:
  result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=90)
 text=(OUT/'counter-probe.csv').read_text() if (OUT/'counter-probe.csv').exists() else ''
 (OUT/'counter-capability.json').write_text(json.dumps(dict(returncode=result.returncode,
      permission_denied='ERR_NVGPUCTRPERM' in text,available=result.returncode==0),indent=2)+'\n')
 print('COUNTER PROBE',result.returncode,flush=True)
 # Separate processes keep the cold and prewarmed marker paths independent.
 path=BUILD/'interleaved'
 env=dict(os.environ,MECH_LIBRARY=str(path/'libflux_cuda.so'),PYTHONPATH=str(path/'python'),
          LD_LIBRARY_PATH=f'{path}:/data/apps/cuda/12.8/lib64')
 for mode in ('cold','warm'):
  with (OUT/f'startup-{mode}.log').open('w') as f:
   subprocess.run([sys.executable,str(SCRIPTS/'startup_probe.py'),mode],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('STARTUP PROBE',mode,(OUT/f'startup-{mode}.json').read_text().strip(),flush=True)
for block,order in enumerate(config['orders'][:1] if smoke else config['orders']):
 for policy in order:
  for kind in ['vanilla_before','instrumented','vanilla_after']:
   path=(BUILD if kind=='instrumented' else BASE)/policy
   env=dict(MECH_POLICY=policy,MECH_KIND=kind,MECH_BLOCK=str(block),MECH_LIBRARY=str(path/'libflux_cuda.so'),PYTHONPATH=str(path/'python'),LD_LIBRARY_PATH=f'{path}:/data/apps/cuda/12.8/lib64')
   args=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(SCRIPTS/'worker.py')]
   commands.append(dict(args=args,env=env));(OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
   print('START',block,policy,kind,flush=True)
   with (OUT/f'b{block}-{policy}-{kind}.log').open('w') as f:
    subprocess.run(args,env=dict(os.environ,**env),stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(SCRIPTS/'summarize.py'),str(OUT)],check=True)
subprocess.run([sys.executable,str(SCRIPTS/'report.py'),str(OUT)],check=True)
print('MECHANISM COMPLETE',OUT,flush=True)

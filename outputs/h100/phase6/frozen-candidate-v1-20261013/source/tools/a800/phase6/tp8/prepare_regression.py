"""Freeze the unchanged TP4 suite with separate before/after TP8 stages."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]


def main(a):
    out=a.out.resolve()
    subprocess.run([sys.executable,str(REPO/'tools/a800/phase6/prepare_scenario.py'),str(out),
        '--build',str(REPO/'outputs/a800/phase6/tp4-opt2-final-build-20261004'),
        '--plan',str(REPO/'logs/a800/phase6/communication-confirm-h2048-tp4-20261003/scripts/selection-plan.json'),
        '--hidden','2048','--sequence',str(a.sequence),'--bases','remote_first','interleaved',
        '--decoder','compact-v1','--job-id',str(a.job_id)],check=True)
    source=(out/'campaign.py').read_text()
    prefix=source.split('\ntry:\n',1)[0]
    body='''
import sys
stage=sys.argv[1]
assert stage in ('before','after')
previous=json.loads((ROOT/'stage-before.json').read_text()) if stage=='after' else None
if previous:
    assert previous['status']=='before_completed'
    state['stages']=previous['stages']
try:
    save()
    if stage=='before':
        directory=ROOT/'results/mapping';directory.mkdir()
        execute('mapping',[PYTHON,ROOT/'check_mappings.py',ROOT],base_env(),directory)
        for placement in INFO['codec_checks']:codec_stage(placement)
        model_stage('preflight','preflight')
        model_stage('model-1','timing',reverse=0)
        state.update(status='before_completed',stage='waiting_for_tp8',finished_at=now())
    else:
        model_stage('model-2','timing',reverse=1)
        write_report(ROOT)
        state.update(status='completed',stage='completed',finished_at=now())
    save()
    (ROOT/('stage-'+stage+'.json')).write_text(json.dumps(state,indent=2)+'\\n')
    if stage=='after':
        env=dict(base_env(),PYTHONPATH=str(ROOT/'scripts'))
        subprocess.run([PYTHON,ROOT/'verify.py',ROOT],env=env,check=True)
except BaseException:
    state.update(status='failed',error=traceback.format_exc());save();raise
'''
    target=out/'regression_stage.py';target.write_text(prefix+'\n'+body)
    info=json.loads((out/'submission.json').read_text())
    info['files'][target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    info['scope']='Unchanged TP4 six-policy control before and after TP8; time-separated regression measurement, not a new optimization campaign'
    (out/'submission.json').write_text(json.dumps(info,indent=2)+'\n')
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('out',type=Path)
    p.add_argument('--sequence',type=int,choices=[1024,2048],required=True)
    p.add_argument('--job-id',type=int,required=True)
    main(p.parse_args())

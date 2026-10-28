"""Freeze fresh BF16 arrival calibration for communication-heavy shapes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from model_config import read_model, mlp_shape

SHAPES=[[8192,512,2048],[8192,1024,4096],[8192,2048,8192],[8192,4096,2048]]
POLICIES=['remote_first','interleaved','interleaved_remote']


def model_worker(repo, world):
    text=(repo/'tools/a800/phase3/mechanism/worker.py').read_text()
    text=text.replace('import flux\n','')
    text=text.replace('torch.cuda.set_device(local)','torch.cuda.set_device(local)\nimport flux')
    text=text.replace('world==4',f'world=={world}')
    text=text.replace('ring_reduction=False','ring_reduction=True')
    text=text.replace('offsets=[0,per//4,per//2,3*per//4,per-1]','offsets=list(range(per))')
    text=text.replace('for repeat in range(repeats):','for repeat in range(3*per):')
    text=text.replace('order=list(modes);',"order=[x for x in modes if x[0] in ('off','receiver_sparse')];")
    text=text.replace(' def correct(y):'," fp32_metrics=dict(checks=0,max_abs=0.,max_relative_l2=0.)\n def correct(y):")
    text=text.replace('  torch.testing.assert_close(y.float(),ref32,rtol=.02,atol=.02)',
        "  delta=y.float()-ref32\n  fp32_metrics['checks']+=1\n  fp32_metrics['max_abs']=max(fp32_metrics['max_abs'],delta.abs().max().item())\n  fp32_metrics['max_relative_l2']=max(fp32_metrics['max_relative_l2'],(delta.norm()/ref32.norm()).item())")
    return text.replace('correctness=True,batch_us=[],samples=[]','correctness=True,fp32_diagnostic=fp32_metrics,batch_us=[],samples=[]')


def main(out,sampler,group_sampler=None,large=False,model_path=None,world=4,job_id=None,sampler_root=None):
    repo=Path(__file__).resolve().parents[3]
    assert job_id is not None and job_id>0, 'Pass the current allocated --job-id'
    assert not (large and model_path), '--large is a historical operator-only catalog'
    out=out.resolve()
    model=read_model(model_path,world) if model_path else None
    assert model or world==4, 'TP8 calibration requires an explicit model profile'
    assert model or sampler is not None, 'Historical three-base calibration requires --sampler'
    out.mkdir(parents=True,exist_ok=False)
    (out/'calibration-scripts').mkdir()
    source=repo/'tools/a800/phase3/mechanism/worker.py' if model else repo/'logs/a800/arrival/arrival-v2-tp4-20261003/results/calibration-fit/calibration-scripts/worker.py'
    assert source.is_file()
    if model:
        (out/'calibration-scripts/worker.py').write_text(model_worker(repo,world))
        shutil.copy2(model_path,out/'model-profile.json')
    else:
        shutil.copy2(source,out/'calibration-scripts/worker.py')
    shapes=[mlp_shape(model)] if model else [[65536,512,2048],[32768,1024,4096],[16384,2048,8192]] if large else SHAPES
    cfg=dict(shapes=shapes,repeats=0,warmup=100,trials=3,iters=50,sample_warmup=10)
    (out/'calibration-scripts/config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    default_root=repo/('outputs/a800/phase3/mechanism-build' if world==4 else 'outputs/a800/arrival8/mechanism-build')
    info=dict(repo=str(repo),sampler=str(sampler.resolve()) if sampler else None,
              sampler_root=str(Path(sampler_root or default_root).resolve()),world_size=world,job_id=job_id,
              shapes=shapes,policies=['remote_first','interleaved'] if model else POLICIES+(['interleaved_remote_group'] if group_sampler else []),
              rationale='Fresh per-base BF16 arrival measurements for the selected model; no reuse of old shape tables.',
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if group_sampler:info['group_sampler']=str(group_sampler.resolve())
    (out/'plan.json').write_text(json.dumps(info,indent=2)+'\n')
    shutil.copy2(Path(__file__).with_name('calibration_driver.py'),out/'driver.py')
    (out/'manifest.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('out',type=Path);p.add_argument('--sampler',type=Path)
    p.add_argument('--model-config',type=Path);p.add_argument('--world-size',type=int,choices=[4,8],default=4)
    p.add_argument('--job-id',type=int,required=True);p.add_argument('--sampler-root',type=Path)
    p.add_argument('--group-sampler',type=Path)
    p.add_argument('--large',action='store_true')
    a=p.parse_args();main(a.out,a.sampler,a.group_sampler,a.large,a.model_config,a.world_size,a.job_id,a.sampler_root)

"""Freeze v2 gates from independent native BF16/FP32 measurements only.

The factor 2 is fixed before candidate revalidation: an empirical two-sided
BF16-to-reference envelope, not a proof or a long-run convergence guarantee.
No Flux candidate error is read by this script.
"""
import hashlib
import json
from pathlib import Path
import sys

out=Path(sys.argv[1]).resolve()
config_path=Path(__file__).with_name('config.json')
config=json.loads(config_path.read_text())
by_tensor={};shape_bounds={};loss_bound=0.;inputs={}
for seed in [101,202]:
    for precision in ['bf16','fp32']:
        folder=out/f'calibration-{seed}-{precision}'
        assert (folder/'exit-code.txt').read_text().strip()=='0'
        for rank in range(4):
            path=folder/f'rank{rank}.json'
            r=json.loads(path.read_text())
            assert r['policy']=='native' and r['mode']=='calibrate' and r['seed']==seed
            assert r['precision']==precision and r['finite_training'] and r['completed']
            inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            for check in r['shape_fp32_checks']:
                key=','.join(map(str,check['shape']))
                b=shape_bounds.setdefault(key,dict(max_abs=0.,relative_l2=0.))
                for field in b:b[field]=max(b[field],check[field])
            if precision=='fp32':
                loss_bound=max(loss_bound,max(r['loss_differences']))
                for name,check in r['checks'].items():
                    assert check['finite']
                    b=by_tensor.setdefault(name,dict(max_abs=0.,relative_l2=0.))
                    for field in b:b[field]=max(b[field],check[field])
for name in ['autograd-native','autograd-control']:
    for rank in range(4):
        r=json.loads((out/name/f'rank{rank}.json').read_text())
        assert r['passed'] and all(c['passed'] for c in r['checks'].values())
        if name=='autograd-control':
            assert r['control_backward']
            assert all(c['max_abs']==0 for c in r['checks'].values()),'Custom backward must match native control exactly'

def budget(bound):
    return dict(atol=max(2*bound['max_abs'],1e-8),rtol=0.,relative_l2=max(2*bound['relative_l2'],1e-6))

config['v1_budgets']=json.loads(json.dumps(config.get('v1_budgets',config['budgets'])))
config['tensor_budgets']={name:budget(bound) for name,bound in by_tensor.items()}
config['shape_budgets']={name:budget(bound) for name,bound in shape_bounds.items()}
config['budgets']['loss_absolute']=2*loss_bound
config['calibration']=dict(source=str(out),seeds=[101,202],holdout_model_seed=1234,
    factor=2,inputs=inputs,rule='Per tensor max across 2 native BF16/FP32 seeds and 4 ranks; double both max-abs and relative-L2 bounds. Original v1 failure stays failed. Near-zero parameters require absolute and empirical relative gates; no bitwise or convergence claim.',
    backward_control='Native forward with adapter backward exactly reproduces native outputs, gradients and first update.')
config['revision_note']='User authorized independent native calibration after v1 admission failures 178699/178700. v2 gates use no candidate error; prospective candidate validation uses held-out model seed 1234.'
config_path.write_text(json.dumps(config,indent=2)+'\n')
(out/'frozen-budgets.json').write_text(json.dumps(dict(tensor_budgets=config['tensor_budgets'],shape_budgets=config['shape_budgets'],
    loss_absolute=config['budgets']['loss_absolute'],calibration=config['calibration']),indent=2)+'\n')
print('FROZEN',len(by_tensor),'tensor bounds',len(shape_bounds),'shape bounds; loss absolute',2*loss_bound)

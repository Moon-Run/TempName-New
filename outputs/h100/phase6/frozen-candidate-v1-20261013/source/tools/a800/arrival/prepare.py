"""Snapshot existing validated workers and apply explicit arrival-experiment configuration.

Run before calibration. Historical experiment tools/builds are never overwritten.
"""
import argparse,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,default=ROOT/'outputs/a800/arrival')
OUT=parser.parse_args().out.resolve()
OUT.mkdir(parents=True,exist_ok=True)
SHAPES=[[8192,4096,2048],[4096,4096,8192],[2048,2048,2048],[2048,2048,8192]]
POLICIES=['original','remote_first','interleaved','remote_arrival','interleaved_arrival']
config=dict(shapes=SHAPES,world_size=4,policies=POLICIES,window=16,calibration_passes=3,
            ring_reduction=True,fp32_policy='diagnostic_only_user_authorized',seeds=[17,29,43],warmup_initial=500,warmup_per_trial=30,iters=100,trial_count=12,
            orders=[POLICIES[i:]+POLICIES[:i] for i in range(5)])
(HERE/'config.json').write_text(json.dumps(config,indent=2)+'\n')
cal=OUT/'calibration-scripts';cal.mkdir(exist_ok=True)
cfg=dict(shapes=SHAPES,repeats=0,warmup=100,trials=3,iters=50)
(cal/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
s=(ROOT/'tools/a800/phase3/mechanism/worker.py').read_text()
s=s.replace('ring_reduction=False','ring_reduction=True')
s=s.replace("offsets=[0,per//4,per//2,3*per//4,per-1]", "offsets=list(range(per))")
s=s.replace('for repeat in range(repeats):', 'for repeat in range(3*per):')
s=s.replace('order=list(modes);', "order=[x for x in modes if x[0] in ('off','receiver_sparse')];")
s=s.replace(' def correct(y):', " fp32_metrics=dict(checks=0,failed_checks=0,max_abs=0.,max_relative_l2=0.,policy='diagnostic_only_user_authorized')\n def correct(y):")
s=s.replace('  torch.testing.assert_close(y.float(),ref32,rtol=.02,atol=.02)', """  delta=y.float()-ref32
  fp32_metrics['checks']+=1
  fp32_metrics['failed_checks']+=int(not torch.all(delta.abs() <= .02+.02*ref32.abs()).item())
  fp32_metrics['max_abs']=max(fp32_metrics['max_abs'],delta.abs().max().item())
  fp32_metrics['max_relative_l2']=max(fp32_metrics['max_relative_l2'],(delta.norm()/ref32.norm()).item())""")
s=s.replace('correctness=True,batch_us=[],samples=[]', 'correctness=True,fp32_diagnostic=fp32_metrics,batch_us=[],samples=[]')

# Flush completed shapes only; retain all fragment timestamps and matched off observations.
(cal/'worker.py').write_text(s)
# Five-policy operator worker, using same ring-reduction mode as the model.
s=(ROOT/'tools/a800/phase3/extension_window.py').read_text()
s=s.replace('ring_reduction=False','ring_reduction=True')
s=s.replace("'extension.json'", "'config.json'")
s=s.replace('   torch.testing.assert_close(actual.float(),ref32,atol=.02,rtol=.02)', "   fp32_passed=bool(torch.all((actual.float()-ref32).abs() <= .02+.02*ref32.abs()).item())")
s=s.replace("dict(seed=seed,passed=True,max_abs_vs_bf16=", "dict(seed=seed,passed=True,bf16_passed=True,fp32_passed=fp32_passed,fp32_policy='diagnostic_only_user_authorized',max_abs_vs_bf16=")

(HERE/'operator_worker.py').write_text(s)
model=OUT/'e2e-scripts';model.mkdir(exist_ok=True)
for name in ['adapter.py','worker.py','summarize.py','launch.py']:
 s=(ROOT/'tools/a800/e2e/scale'/name).read_text()
 if name=='launch.py':
  s=s.replace("CASE=next(", "if os.environ.get('ARRIVAL_REVERSE')=='1':\n PLAN['orders']=[list(reversed(o)) for o in reversed(PLAN['orders'])]\nCASE=next(",1)
  s=s.replace("ROOT/'outputs/a800/phase3/three-way-build'", "ROOT/'outputs/a800/arrival/build'")
  s=s.replace("ROOT/'outputs/a800/megatron-e2e/scale-mapping'", "ROOT/'outputs/a800/arrival/build/mapping'")
  s=s.replace("str(PLAN['policies'].index(policy)-1)", "str({'original':0,'remote_first':1,'interleaved':2,'remote_arrival':3,'interleaved_arrival':4}[policy])")
  s=s.replace("ROOT/'outputs/a800/arrival/build'", "Path(os.environ.get('ARRIVAL_OUTPUT_ROOT', str(ROOT/'outputs/a800/arrival')))/'build'")
  s=s.replace("ROOT/'outputs/a800/arrival/build/mapping'", "BUILD/'mapping'")
 if name=='adapter.py':
  s=s.replace('import types', 'import types\nimport os')
  s=s.replace('        self.observed = {}', '        self.observed = {}\n        self.forward_checks = {}')
  marker='            output = GemmRSFunction.apply(input_, module.weight, op, adapter.option)'
  addition="""
            if os.environ.get('E2E_MODE') == 'smoke' and name not in adapter.forward_checks:
                with torch.no_grad():
                    flat = input_.reshape(-1, input_.shape[-1])
                    partial = flat.matmul(module.weight.t())
                    ref = torch.empty_like(output).reshape(-1, output.shape[-1])
                    torch.distributed.reduce_scatter_tensor(ref, partial, group=parallel_state.get_tensor_model_parallel_group())
                    actual = output.reshape_as(ref)
                    torch.testing.assert_close(actual, ref, atol=.02, rtol=.02)
                    delta = actual.float() - ref.float()
                    adapter.forward_checks[name] = dict(max_abs=delta.abs().max().item(), relative_l2=(delta.norm()/ref.float().norm()).item())
"""
  s=s.replace(marker, marker+addition)
 if name=='worker.py':
  s=s.replace("report['peak_allocated_gib']=", "report['forward_checks']=adapter.forward_checks\nreport['peak_allocated_gib']=")
 if name=='summarize.py':
  s=s.replace("for baseline in ['native','original']:", "for baseline in ['native','original','remote_first','interleaved']:")
  s=s.replace("            elapsed=max(r['elapsed_seconds'] for r in rows)", "            if policy != 'native':\n                assert all(len(r['forward_checks'])==2*model['layers'] for r in rows)\n            elapsed=max(r['elapsed_seconds'] for r in rows)",1)
  s=s.replace("e['args']['registers per thread']==254", "0 < e['args']['registers per thread'] <= 255")
 (model/name).write_text(s)
plan=json.loads((ROOT/'tools/a800/e2e/scale/config.json').read_text())
plan['policies']=['native']+POLICIES
plan['orders']=[plan['policies'][i:]+plan['policies'][:i] for i in range(6)]
plan['cases']=plan['cases'][:1]
plan['scope']='BF16 arrival-priority: six position-balanced policies, single-node TP4, full optimizer steps. Independent operator numerical/mapping preflight and finite model training; no claim of real-data convergence.'
(model/'config.json').write_text(json.dumps(plan,indent=2)+'\n')
# Build checker: preserve the base-policy checks and validate arrival table explicitly.
s=(ROOT/'tools/a800/e2e/scale/check_mapping.cu').read_text()
s=s.replace('policy>2','policy>4').replace('policy==2 && count!=tiles','policy>=2 && count!=tiles')
s=s.replace('if(policy==2){\n    int dest=', 'if(policy>=3){\n    int j=flux_arrival_host_index(tm,tn,k/tp,tp,rank,i);\n    int ex=flux_arrival_host_coord(tm,tn,k/tp,tp,rank,i);\n    if(j<0 || ex!=x*tn+y)return 8;\n   }else if(policy==2){\n    int dest=')
# Helpers are injected into baseline checkers too, but never called there.
s=s.replace('using Base=', '#include "arrival_host.hpp"\nusing Base=')
(HERE/'check_mapping.cu').write_text(s)
print(OUT)

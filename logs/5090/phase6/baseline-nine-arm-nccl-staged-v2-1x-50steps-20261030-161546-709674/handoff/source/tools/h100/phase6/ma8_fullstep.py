#!/usr/bin/env python3
"""F4 / F4-cache full optimizer-step comparison with identical native forward."""
from pathlib import Path
import ast


def registered_replace(source,old,new,count=1):
    if source.count(old)!=count:
        raise ValueError('MA8 registered source anchor changed: '+old[:100])
    return source.replace(old,new)


_HERE=Path(__file__).resolve().parent
_BASE_PATH=_HERE/'ma8_fullstep_base.py'
_FROZEN=_BASE_PATH.is_file()
if not _BASE_PATH.is_file():_BASE_PATH=_HERE/'ma5_fullstep.py'
_F4_PATH=_HERE/'ma8_fullstep_f4.py'
if not _F4_PATH.is_file():_F4_PATH=_HERE/'ma4_f_fullstep.py'
_source=_BASE_PATH.read_text()
_source=_source[:_source.index('\nif __name__ == "__main__":')]
_source=registered_replace(_source,'ma5','ma8',25)
_source=registered_replace(_source,'MA5','MA8',14)
if _FROZEN:
    # The returned controller is a read-only replay entry. Its generated
    # worker/adapter already exist and are hashed; source-generation imports
    # and server launch dependencies must not leak in from another checkout.
    _source=registered_replace(_source,'import ablation4_runtime as communication','import communication')
    _source=registered_replace(_source,'import ablation4_protocol as routing','import routing')
    _source=registered_replace(_source,'import ablation4_sources as sources','sources = None')
    _source=registered_replace(_source,'from run import arguments as model_arguments',
        "def model_arguments(*args, **kwargs):\n    raise RuntimeError('Frozen MA8 controller is read-only; use repository entry to launch')")
_source=registered_replace(_source,'MA8 tile/block tuning versus unchanged CR2; full optimizer steps.',
    'MA8 F4 / F4-cache; full optimizer steps; retained column activations.')
_source=registered_replace(_source,'equivalence_pairs=[["T","CR2"]]','equivalence_pairs=bench.EQUIVALENCE_PAIRS',2)
_source=registered_replace(_source,'                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},',
    "                activation_policy='retain_owned_column_input_uninstrumented',\n"
    '                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},')
_source=registered_replace(_source,'    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,',
                        '    freeze_helpers(scripts)\n    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,')
exec(compile(_source,str(_BASE_PATH),'exec'),globals())
__doc__ = 'One or two balanced F4 / F4-cache full optimizer-step comparisons.'

import ma4_f_adapter as f4_adapter
import ma8_adapter
import ma8_profile

PROTOCOL=bench.SOURCE_PROTOCOL+'-fullstep'
TARGET_MS=None
TARGET_PERCENT=2.0
ACCEPTANCE_FIELDS=('increment_over_F4_validated', 'target_point_observed')
ACCEPTANCE_RULE=('Within every requested independent trial, positive increment requires paired point>0 and CI95 lower>0. '
    'The 2% target is validated only when paired point>=2 and CI95 lower>=2 in every trial; point>=2 alone is reported separately. '
    'Require exact full-model gradient/update fingerprints, identical trajectories, owned column input reuse and actual collective removal. '
    'All balanced blocks retained; one trial is exploratory.')
_prepare,_verify,_window,_preflight,_report=prepare,verify,window,preflight,report
_worker_source,_adapter_source,_taco_source=worker_source,adapter_source,taco_source


def _f4_function(name):
    text=_F4_PATH.read_text()
    nodes=[node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name==name]
    require(len(nodes)==1,'Missing or duplicate frozen F4 function: '+name)
    counts=dict(worker_source=4,validate_parameter_trajectory=0,window=6,preflight=2,trace_projection_records=0)
    return registered_replace(ast.get_source_segment(text,nodes[0]),'ma4_f','ma8',counts[name])


# Reuse the exact strong smoke and consumer coverage checks, frozen separately.
exec(compile(_f4_function('worker_source'),str(_F4_PATH),'exec'),globals())
_consumer_worker_source=worker_source
exec(compile(_f4_function('validate_parameter_trajectory'),str(_F4_PATH),'exec'),globals())
_window_source=registered_replace(_f4_function('window'),"if job['arm']=='F4':","if job['arm'] in bench.ARMS:")
exec(compile(_window_source,str(_F4_PATH),'exec'),globals())
_consumer_window=window
exec(compile(_f4_function('preflight'),str(_F4_PATH),'exec'),globals())


def freeze_helpers(scripts):
    bench.freeze_helpers(scripts)
    for source,name in [(_BASE_PATH,'ma8_fullstep_base.py'),(_F4_PATH,'ma8_fullstep_f4.py'),
                        (Path(bench.__file__),'ma8_bench.py'),
                        (Path(ma8_profile.__file__),'ma8_profile.py')]:
        shutil.copy2(source,scripts/name)


def verify(root,offline=False):
    require(not _FROZEN or offline,'Frozen controller supports offline replay only')
    plan=_verify(root,offline)
    bench.validate_experiment(plan)
    worker=(Path(root)/'scripts/worker.py').read_text()
    require(activation_setup_source() in worker and
            'ma8_cache_tracker' not in worker and 'ma8_activation_cache_state' not in worker,
            'Plan/generated worker must use setup-only metadata and candidate-only installation')
    for source,name in [(_BASE_PATH,'ma8_fullstep_base.py'),(_F4_PATH,'ma8_fullstep_f4.py'),
                        (bench.TEMPLATE,'ma8_bench_base.py'),(bench.base_path,'ma4_f_bench_base.py'),
                        (Path(f4_adapter.__file__),'ma4_f_adapter.py'),
                        (Path(ma8_adapter.__file__),'ma8_adapter.py'),
                        (Path(ma8_profile.__file__),'ma8_profile.py'),
                        (Path(bench.ma8_sources.__file__),'ma8_sources.py'),
                        *[(_HERE/name,name) for name in ('common.py','ma4_f_sources.py','ma2_cr2_sources.py')]]:
        require(sha(source)==plan['source_files']['scripts/'+name],'Changed controller dependency: '+name)
    return plan


def adapter_source():
    source=f4_adapter.adapter_source(_adapter_source())
    source=registered_replace(source, 'H100_MA4_F_ARM', 'H100_MA8_ARM')
    return registered_replace(source, "os.environ.get('H100_MA8_ARM') == 'F4'",
        "os.environ.get('H100_MA8_ARM') in ('F4', 'F4-cache')")


def taco_source():
    return f4_adapter.taco_source(_taco_source())


def activation_setup_source():
    return """import ma8_adapter
if os.environ['H100_MA8_ARM'] == 'F4-cache':
    ma8_activation_setup = ma8_adapter.install_activation_cache(raw, 'F4-cache')
else:
    ma8_activation_setup = ma8_adapter.inspect_activation_setup(raw, 'F4')"""


def worker_source():
    source = _consumer_worker_source()
    anchor = "adapter = Adapter(raw, POLICY, SPEC['shapes'], SPEC['ring_reduction'])"
    source = replace_once(source, anchor, anchor + '\n' + activation_setup_source())
    return replace_once(source, "report['ma8_consumer_state'] = adapter.consumer_state()",
        "report['ma8_activation_setup'] = ma8_activation_setup\n"
        "report['ma8_consumer_state'] = adapter.consumer_state()")


def validate_activation_setup(metadata, arm, target_modules):
    bench.family(arm)
    expected={name.replace('.self_attention.linear_proj', '.self_attention.linear_qkv')
             .replace('.mlp.linear_fc2', '.mlp.linear_fc1'):
             'qkv' if name.endswith('.self_attention.linear_proj') else 'fc1'
             for name in target_modules}
    require(len(expected)==64 and metadata==dict(
        protocol=ma8_adapter.PROTOCOL, arm=arm, installed=arm=='F4-cache',
        function='_MA8CachedRawLinear' if arm=='F4-cache' else 'LinearWithGradAccumulationAndAsyncCommunication',
        module_roles=expected,
        source_sha256=dict(class_source=ma8_adapter.CLASS_SHA256,
                           column_forward=ma8_adapter.COLUMN_FORWARD_SHA256,
                           linear_wrapper=ma8_adapter.LINEAR_WRAPPER_SHA256)),
        'Wrong setup-only activation metadata or column module coverage')


def window(root,plan,job):
    rows=_consumer_window(root,plan,job)
    for row in rows:
        require(len(row['ma8_trainable_parameters'])==389, 'Changed trainable parameter universe')
        require('ma8_activation_cache_state' not in row, 'MA8-v1 per-call counter evidence is not accepted')
        validate_activation_setup(row['ma8_activation_setup'],job['arm'],row['target_modules'])
    return rows


def comparison_acceptance(estimates,mean_ms):
    require(set(estimates)=={'F4-cache/F4'} and set(mean_ms)==set(bench.ARMS), 'Missing paired comparison')
    value=estimates['F4-cache/F4']
    require(len(value['ci95'])==2 and all(math.isfinite(v) for v in [value['percent'],*value['ci95']])
            and value['ci95'][0]<=value['ci95'][1], 'Invalid paired interval')
    require(all(math.isfinite(v) and v>0 for v in mean_ms.values()), 'Invalid mean step time')
    return dict(increment_over_F4_validated={'F4-cache':value['percent']>0 and value['ci95'][0]>0},
        target_point_observed=value['percent']>=TARGET_PERCENT,
        target_validated=value['percent']>=TARGET_PERCENT and value['ci95'][0]>=TARGET_PERCENT)


def aggregate_acceptance(comparisons):
    require(len(comparisons) in (1,2), 'One or two independent trials required')
    return dict(increment_over_F4_validated={'F4-cache':all(v['increment_over_F4_validated']['F4-cache'] for v in comparisons.values())},
        target_point_observed=all(v['target_point_observed'] for v in comparisons.values()),
        target_validated=all(v['target_validated'] for v in comparisons.values()))


def report(root,offline=False,save=True):
    result=_report(root,offline,save=False)
    plan=verify(root,offline);fingerprints={};memory=defaultdict(list)
    for job in jobs(plan,'timing'):
        rows=window(Path(root),plan,job)
        key=(job['trial'],job['block']);value=[r['ma8_final_parameters_sha256'] for r in rows]
        fingerprints.setdefault(key,value);require(fingerprints[key]==value,'Final paired model parameters differ')
        for row in rows:
            require(all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) and row[k]>0
                        for k in ('peak_allocated_gib','peak_reserved_gib')), 'Missing peak CUDA memory evidence')
        memory[job['arm']].append({k:max(row[k] for row in rows)
                                  for k in ('peak_allocated_gib','peak_reserved_gib')})
    result.update(target_evaluated=True,target_percent=TARGET_PERCENT,consumer_parameter_gradients_validated=True,
        activation_policy=plan['activation_policy'],
        peak_cuda_memory_gib={arm:{key:dict(mean_window_rank_max=statistics.mean(v[key] for v in values),
            max_window_rank_max=max(v[key] for v in values)) for key in ('peak_allocated_gib','peak_reserved_gib')}
            for arm,values in memory.items()},
        acceptance_status={arm:'positive_F4_increment' if good else 'F4_increment_not_validated'
                           for arm,good in result['increment_over_F4_validated'].items()},
        main_contrasts=[c+'/'+b for c,b in bench.MAIN_COMPARISONS],
        activation_setup_validated=True, per_call_instrumentation=False)
    result['bootstrap']['unit']='complete fresh-process F4/F4-cache block'
    result.pop('target_ms',None)
    if save:
        bench.save_unchanged(Path(root)/'report.json',result)
        (Path(root)/'report.md').write_text(report_markdown(result))
        print(json.dumps(dict(comparisons=result['comparisons'],increment_over_F4_validated=result['increment_over_F4_validated']),indent=2))
    return result


def report_markdown(result):
    lines=['# MA8：CR3/F4 与列并行激活保留的完整训练结果', '',
        '| trial | F4 / F4-cache 平均 ms | 收益 %［95% CI］ | 点估计 ≥2% | 区间下界 ≥2% |',
        '| --- | --- | --- | --- | --- |']
    for name,comparison in result['comparisons'].items():
        combined=comparison['rounds']['combined']
        means=' / '.join(f"{combined['mean_ms'][arm]:.3f}" for arm in bench.ARMS)
        value=combined['comparisons']['F4-cache/F4']
        lines.append(f"| {name} | {means} | {value['percent']:+.3f} [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}] | {comparison['target_point_observed']} | {comparison['target_validated']} |")
    lines+=['', f"保留全部 {len(result['all_samples'])} 个窗口、{result['formal_optimizer_steps']} 个正式 optimizer step。",
        '每个 trial 分别验收；单轮仍为探索性结果。profile 必须另行通过，完整实验状态以 batch-state.json 为准。',
        '比较双方保持原 CR3 行投影；候选保留列并行前向完整输入，删除其反向重复 AllGather。',
        '完整梯度/更新指纹保持一致；显存峰值与全部慢窗口一并报告。']
    return '\n'.join(lines)+'\n'


_trace_source=_f4_function('trace_projection_records')
_trace_source=registered_replace(_trace_source,'(events,arm,mechanism=None)', '(events,arm,tuning=None)')
_trace_source=registered_replace(_trace_source,'bench.validate_mechanism(mechanism)','bench.validate_tuning(tuning)')
_trace_source=registered_replace(_trace_source,'        gemm,barrier,decoder=path',
    "        gemm,barrier,decoder=path\n        role=scope['name'].split('.')[1]\n"
    "        launch=bench.validate_launch(gemm,decoder,arm,role,tuning)")
_trace_source=registered_replace(_trace_source,"0 if arm=='F4' else 2", '0',2)
_trace_source=registered_replace(_trace_source,"fused_consumer=arm=='F4'", 'fused_consumer=True')
_trace_source=registered_replace(_trace_source,".append(dict(cpu_us=call['dur']", ".append(dict(launch=launch,cpu_us=call['dur']")
_trace_source=registered_replace(_trace_source,'layernorm_successors=128,fused_consumer=True,',
                                    "layernorm_successors=128,fused_consumer=True,launch=values[0]['launch'],")
exec(compile(_trace_source,str(_F4_PATH),'exec'),globals())
_trace_forward=trace_projection_records


def trace_projection_records(events,arm,tuning=None):
    forward=_trace_forward(events,arm,tuning)
    backward=ma8_profile.trace_row_backward_records(events,arm)
    result={role:dict(value,backward=backward[role]) for role,value in forward.items()}
    result['activation_cache']=ma8_profile.trace_activation_cache_records(events,arm)
    return result


if _FROZEN:
    _replay_report,_replay_preflight,_replay_profile=report,preflight,profile_summary

    def _require_replay_gate(root,offline):
        require(offline,'Frozen controller supports offline replay only')
        require((Path(root)/'preflight.json').is_file(),
                'Frozen replay requires an existing preflight.json; it never creates evidence')

    def preflight(root,offline=True):
        _require_replay_gate(root,offline)
        return _replay_preflight(root,offline=True)

    def report(root,offline=True,save=False):
        _require_replay_gate(root,offline)
        # Even an explicit save=True cannot mutate the returned evidence.
        return _replay_report(root,offline=True,save=False)

    def profile_summary(root,offline=True):
        _require_replay_gate(root,offline)
        return _replay_profile(root,offline=True)

    def _frozen_source(name):
        root=_HERE.parent
        plan=verify(root,offline=True)
        path=_HERE/(name+'.py')
        require(sha(path)==plan['source_files']['scripts/'+name+'.py'],'Frozen generated source changed')
        return path.read_text()

    def adapter_source():return _frozen_source('adapter')
    def worker_source():return _frozen_source('worker')
    def taco_source():return _frozen_source('taco_support')

    def prepare(args):
        raise RuntimeError('Frozen MA8 controller is read-only; use repository entry to prepare')

    def run(args):
        raise RuntimeError('Frozen MA8 controller is read-only; use repository entry to launch')

    def main(argv=None):
        parser=argparse.ArgumentParser(description='Read-only replay of frozen MA8 evidence; JSON is printed to stdout.')
        sub=parser.add_subparsers(dest='action',required=True)
        for action in ('report','profile-report'):
            command=sub.add_parser(action)
            command.add_argument('root',type=Path)
            command.add_argument('--offline',action='store_true',required=True)
        args=parser.parse_args(argv)
        value=(report(args.root,offline=True,save=False) if args.action=='report' else
               profile_summary(args.root,offline=True))
        print(json.dumps(value,ensure_ascii=False,indent=2))
        return value


if __name__=='__main__':main()

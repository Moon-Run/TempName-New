#!/usr/bin/env python3
"""Frozen F4, F4-mask and F4-layout full optimizer-step comparison."""
from pathlib import Path
import ast


def registered_replace(source,old,new,count=1):
    if source.count(old)!=count:
        raise ValueError('MA6 registered source anchor changed: '+old[:100])
    return source.replace(old,new)


_HERE=Path(__file__).resolve().parent
_BASE_PATH=_HERE/'ma6_fullstep_base.py'
_FROZEN=_BASE_PATH.is_file()
if not _BASE_PATH.is_file():_BASE_PATH=_HERE/'ma5_fullstep.py'
_F4_PATH=_HERE/'ma6_fullstep_f4.py'
if not _F4_PATH.is_file():_F4_PATH=_HERE/'ma4_f_fullstep.py'
_source=_BASE_PATH.read_text()
_source=_source[:_source.index('\nif __name__ == "__main__":')]
_source=registered_replace(_source,'ma5','ma6',25)
_source=registered_replace(_source,'MA5','MA6',14)
if _FROZEN:
    # The returned controller is a read-only replay entry. Its generated
    # worker/adapter already exist and are hashed; source-generation imports
    # and server launch dependencies must not leak in from another checkout.
    _source=registered_replace(_source,'import ablation4_runtime as communication','import communication')
    _source=registered_replace(_source,'import ablation4_protocol as routing','import routing')
    _source=registered_replace(_source,'import ablation4_sources as sources','sources = None')
    _source=registered_replace(_source,'from run import arguments as model_arguments',
        "def model_arguments(*args, **kwargs):\n    raise RuntimeError('Frozen MA6 controller is read-only; use repository entry to launch')")
_source=registered_replace(_source,'MA6 tile/block tuning versus unchanged CR2; full optimizer steps.',
    'MA6 F4 / F4-mask / F4-layout; full optimizer steps; no combined candidate.')
_source=registered_replace(_source,'args.blocks >= 8 and args.blocks % 8 == 0','args.blocks >= 12 and args.blocks % 12 == 0')
_source=registered_replace(_source,'plan["blocks"] >= 8 and plan["blocks"] % 8 == 0','plan["blocks"] >= 12 and plan["blocks"] % 12 == 0')
_source=registered_replace(_source,'at least eight complete blocks, divisible by eight','complete multiples of twelve three-arm blocks')
_source=registered_replace(_source,'equivalence_pairs=[["T","CR2"]]','equivalence_pairs=bench.EQUIVALENCE_PAIRS',2)
_source=registered_replace(_source,'default=8)','default=12)')
_source=registered_replace(_source,'                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},',
    "                backward_schedule='baseline',\n"
    '                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},')
_source=registered_replace(_source,'    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,',
                        '    freeze_helpers(scripts)\n    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,')
exec(compile(_source,str(_BASE_PATH),'exec'),globals())
__doc__ = 'One or two balanced F4 / F4-mask / F4-layout full optimizer-step comparisons.'

import ma4_f_adapter as f4_adapter
import ma6_adapter

PROTOCOL=bench.SOURCE_PROTOCOL+'-fullstep'
TARGET_MS=None
ACCEPTANCE_FIELDS=('increment_over_F4_validated',)
ACCEPTANCE_RULE=('Each candidate/F4 is accepted only when its paired point and CI95 lower are positive '
    'within every requested trial, with exact consumer outputs, full-model gradient/update fingerprints '
    'and actual native/backward execution paths. Three complete arms, all blocks retained; one trial is exploratory. '
    'Mask/layout comparison is secondary. No combined candidate, no historical C/R target.')
_prepare,_verify,_window,_preflight,_report=prepare,verify,window,preflight,report
_worker_source,_adapter_source,_taco_source=worker_source,adapter_source,taco_source


def _f4_function(name):
    text=_F4_PATH.read_text()
    nodes=[node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name==name]
    require(len(nodes)==1,'Missing or duplicate frozen F4 function: '+name)
    counts=dict(worker_source=4,validate_parameter_trajectory=0,window=6,preflight=2,trace_projection_records=0)
    return registered_replace(ast.get_source_segment(text,nodes[0]),'ma4_f','ma6',counts[name])


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
    for source,name in [(_BASE_PATH,'ma6_fullstep_base.py'),(_F4_PATH,'ma6_fullstep_f4.py'),
                        (Path(bench.__file__),'ma6_bench.py')]:
        shutil.copy2(source,scripts/name)


def verify(root,offline=False):
    require(not _FROZEN or offline,'Frozen controller supports offline replay only')
    plan=_verify(root,offline)
    bench.validate_backward_schedule(plan)
    require("BACKWARD_SCHEDULE = 'baseline'\n" in
            (Path(root)/'scripts/adapter.py').read_text(), 'Baseline plan/generated adapter schedule mismatch')
    for source,name in [(_BASE_PATH,'ma6_fullstep_base.py'),(_F4_PATH,'ma6_fullstep_f4.py'),
                        (bench.TEMPLATE,'ma6_bench_base.py'),(bench.base_path,'ma4_f_bench_base.py'),
                        (Path(f4_adapter.__file__),'ma4_f_adapter.py'),
                        (Path(ma6_adapter.__file__),'ma6_adapter.py'),
                        (Path(bench.ma6_sources.__file__),'ma6_sources.py'),
                        *[(_HERE/name,name) for name in ('common.py','ma4_f_sources.py','ma2_cr2_sources.py')]]:
        require(sha(source)==plan['source_files']['scripts/'+name],'Changed controller dependency: '+name)
    return plan


def adapter_source():
    return ma6_adapter.adapter_source(f4_adapter.adapter_source(_adapter_source()))


def taco_source():
    return f4_adapter.taco_source(_taco_source())


def worker_source():
    return replace_once(_consumer_worker_source(),"report['ma6_consumer_state'] = adapter.consumer_state()",
        "report['ma6_backward_state'] = adapter.ma6_backward_state()\n"
        "report['ma6_consumer_state'] = adapter.consumer_state()")


def window(root,plan,job):
    rows=_consumer_window(root,plan,job)
    count=1 if job['mode']=='profile' else 3 if job['mode']=='smoke' else 50
    warmup=2 if job['mode']=='smoke' else 10
    for row in rows:
        require(len(row['ma6_trainable_parameters'])==389,'Changed trainable parameter universe')
        states=row['ma6_backward_state']
        require(set(states)==set(row['target_modules']),'Missing backward module coverage')
        for state in states.values():bench.validate_backward_state(state,job['arm'],(warmup+count)*4)
    return rows


def comparison_acceptance(estimates,mean_ms):
    require(set(estimates)=={c+'/'+b for c,b in bench.COMPARISONS} and set(mean_ms)==set(bench.ARMS),'Missing three-arm comparison')
    accepted={}
    for candidate,baseline in bench.MAIN_COMPARISONS:
        value=estimates[candidate+'/'+baseline]
        require(len(value['ci95'])==2 and all(math.isfinite(v) for v in [value['percent'],*value['ci95']]) and
                value['ci95'][0]<=value['ci95'][1],'Invalid paired interval')
        accepted[candidate]=value['percent']>0 and value['ci95'][0]>0
    return dict(increment_over_F4_validated=accepted,target_validated=False)


def aggregate_acceptance(comparisons):
    require(len(comparisons) in (1,2),'One or two independent trials required')
    return dict(increment_over_F4_validated={arm:all(v['increment_over_F4_validated'][arm] for v in comparisons.values())
        for arm,_ in bench.MAIN_COMPARISONS},target_validated=False)


def report(root,offline=False,save=True):
    result=_report(root,offline,save=False)
    plan=verify(root,offline);fingerprints={}
    for job in jobs(plan,'timing'):
        rows=window(Path(root),plan,job)
        key=(job['trial'],job['block']);value=[r['ma6_final_parameters_sha256'] for r in rows]
        fingerprints.setdefault(key,value);require(fingerprints[key]==value,'Final three-arm model parameters differ')
    result.update(target_evaluated=False,target_percent=None,consumer_parameter_gradients_validated=True,
        backward_schedule=plan['backward_schedule'],
        acceptance_status={arm:'positive_F4_increment' if good else 'F4_increment_not_validated'
                           for arm,good in result['increment_over_F4_validated'].items()},
        main_contrasts=[c+'/'+b for c,b in bench.MAIN_COMPARISONS],
        secondary_contrast='F4-mask/F4-layout',backward_schedule_validated=True)
    result['bootstrap']['unit']='complete fresh-process F4/F4-mask/F4-layout block'
    result.pop('target_ms',None)
    if save:
        bench.save_unchanged(Path(root)/'report.json',result)
        (Path(root)/'report.md').write_text(report_markdown(result))
        print(json.dumps(dict(comparisons=result['comparisons'],increment_over_F4_validated=result['increment_over_F4_validated']),indent=2))
    return result


def report_markdown(result):
    lines=['# MA6：F4 / F4-mask / F4-layout 完整训练结果','',
        '| trial | F4 / F4-mask / F4-layout 平均 ms | 比较 | 收益 %［95% CI］ |',
        '| --- | --- | --- | --- |']
    for name,comparison in result['comparisons'].items():
        combined=comparison['rounds']['combined']
        means=' / '.join(f"{combined['mean_ms'][arm]:.3f}" for arm in bench.ARMS)
        for label,value in combined['comparisons'].items():
            lines.append(f"| {name} | {means} | {label} | {value['percent']:+.3f} [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}] |")
    lines+=['',f"三臂反向调度：`{result['backward_schedule']}`。",
        f"全部保留 {len(result['all_samples'])} 个窗口、{result['formal_optimizer_steps']} 个正式 optimizer step。",
        '每个候选分别判断相对同期 F4 的正增量；单轮为探索性结果。未测试 mask+layout 合并候选或 C/R。',
        '两种接收端机制分别评估；完整训练与独立 profile 共同验证路径。慢窗口不删除。']
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


def trace_backward_records(events,arm):
    """Associate each backward's CPU launches to GPU work without assuming AG order."""
    bench.family(arm)
    end=lambda event:event['ts']+event.get('dur',0)
    scopes=defaultdict(list);runtimes=defaultdict(list);by_corr=defaultdict(list);by_external=defaultdict(list)
    forwards=[];backwards=[]
    for event in events:
        category=event.get('cat');name=event.get('name','');args=event.get('args',{})
        if category=='user_annotation' and name in ('scope.attn.candidate','scope.mlp.candidate'):scopes[event['tid']].append(event)
        if category in ('cuda_runtime','cuda_driver'):runtimes[event['tid']].append(event)
        if category in ('kernel','gpu_memcpy','gpu_memset'):
            by_corr[args.get('correlation')].append(event);by_external[args.get('External id')].append(event)
        if category=='cpu_op' and name=='ConsumerGemmRSFunction':forwards.append(event)
        if category=='cpu_op' and name=='ConsumerGemmRSFunctionBackward':backwards.append(event)
    for values in scopes.values():values.sort(key=lambda e:e['ts'])
    for values in runtimes.values():values.sort(key=lambda e:e['ts'])
    starts={tid:[e['ts'] for e in values] for tid,values in scopes.items()}
    rt_starts={tid:[e['ts'] for e in values] for tid,values in runtimes.items()}
    sequence_roles={}
    for call in forwards:
        tid=call['tid'];i=bisect_right(starts[tid],call['ts'])-1
        require(i>=0 and end(call)<=end(scopes[tid][i])+.01,'Missing forward role scope')
        seq=call['args']['Sequence number'];require(seq not in sequence_roles,'Duplicate autograd sequence')
        sequence_roles[seq]=scopes[tid][i]['name'].split('.')[1]
    require(len(forwards)==len(backwards)==256,'Incomplete forward/backward profile coverage')
    result=defaultdict(list);used=set();seen=set()
    for call in backwards:
        seq=call['args']['Sequence number'];require(seq in sequence_roles and seq not in seen,'Unmatched/duplicate backward')
        seen.add(seq);role=sequence_roles[seq];tid=call['tid']
        lo=bisect_right(rt_starts[tid],call['ts']-.001);hi=bisect_right(rt_starts[tid],end(call)+.01)
        inside=[e for e in runtimes[tid][lo:hi] if end(e)<=end(call)+.01]
        launch_by_corr={e.get('args',{}).get('correlation'):e for e in inside}
        associated={id(e):e for e in by_external[call['args']['External id']]}
        for corr in launch_by_corr:associated.update({id(e):e for e in by_corr[corr]})
        require(not any(e.get('cat')=='gpu_memcpy' for e in associated.values()),'Backward introduced a GPU copy')
        kernels=[e for e in associated.values() if e.get('cat')=='kernel']
        bias=[e for e in kernels if 'reduce_kernel' in e['name']]
        gathers=[e for e in kernels if 'nccl' in e['name'].lower() and 'AllGather' in e['name']]
        gemms=sorted((e for e in kernels if 'gemm' in e['name'].lower()),key=lambda e:e['ts'])
        require(len(kernels)==4 and len(bias)==len(gathers)==1 and len(gemms)==2,'Backward kernels changed')
        bias,ag=bias[0],gathers[0];dgrad,wgrad=gemms
        require(not used & {id(e) for e in kernels},'GPU kernel attributed twice');used.update(map(id,kernels))
        require(bias['args']['stream']==dgrad['args']['stream']==wgrad['args']['stream'] and
                ag['args']['stream']!=bias['args']['stream'],'Backward compute/communication streams changed')
        require(end(ag)<=dgrad['ts']+.01 and end(bias)<=dgrad['ts']+.01 and end(dgrad)<=wgrad['ts']+.01,
                'Backward consumed unfinished gather/bias or changed GEMM order')
        bias_launch=launch_by_corr[bias['args']['correlation']];ag_launch=launch_by_corr[ag['args']['correlation']]
        dgrad_launch=launch_by_corr[dgrad['args']['correlation']]
        require(end(bias_launch)<=ag_launch['ts']+.01 and end(bias)<=ag['ts']+.01,
                'Baseline backward order changed')
        overlap=max(0.0,min(end(bias),end(ag))-max(bias['ts'],ag['ts']))
        result[role].append(dict(backward_cpu_us=call['dur'],
            gather_launch_end_to_bias_launch_us=bias_launch['ts']-end(ag_launch),
            gather_launch_end_to_dgrad_launch_us=dgrad_launch['ts']-end(ag_launch),
            # Signed: negative means CPU submitted dgrad before AG completed.
            allgather_end_to_dgrad_launch_us=dgrad_launch['ts']-end(ag),
            dgrad_launch_to_gpu_start_us=dgrad['ts']-dgrad_launch['ts'],
            bias_us=bias['dur'],allgather_us=ag['dur'],dgrad_us=dgrad['dur'],wgrad_us=wgrad['dur'],
            bias_allgather_overlap_us=overlap,local_span_us=end(wgrad)-min(bias['ts'],ag['ts']),
            allgather_end_to_dgrad_start_us=dgrad['ts']-end(ag)))
    require(set(result)==set(bench.ROLES) and all(len(rows)==128 for rows in result.values()),'Missing role backward calls')
    return {role:dict(backward_calls=len(rows),bias_kernels=128,allgather_kernels=128,dgrad_kernels=128,wgrad_kernels=128,
        gather_schedule='baseline',
        gather_submitted_before_bias=False,overlap_calls=sum(r['bias_allgather_overlap_us']>0 for r in rows),
        overlap_fraction=sum(r['bias_allgather_overlap_us']>0 for r in rows)/len(rows),
        upstream_gradient_ready_event_identified=False,
        median_us={key:statistics.median(row[key] for row in rows) for key in rows[0]}) for role,rows in result.items()}


def trace_projection_records(events,arm,tuning=None):
    forward=_trace_forward(events,arm,tuning)
    backward=trace_backward_records(events,arm)
    return {role:dict(value,backward=backward[role]) for role,value in forward.items()}


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
        raise RuntimeError('Frozen MA6 controller is read-only; use repository entry to prepare')

    def run(args):
        raise RuntimeError('Frozen MA6 controller is read-only; use repository entry to launch')

    def main(argv=None):
        parser=argparse.ArgumentParser(description='Read-only replay of frozen MA6 evidence; JSON is printed to stdout.')
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

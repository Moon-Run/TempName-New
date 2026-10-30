#!/usr/bin/env python3
"""F4 / F4-wgrad full optimizer-step comparison with identical native forward."""
from pathlib import Path
from collections import Counter
import ast


def registered_replace(source,old,new,count=1):
    if source.count(old)!=count:
        raise ValueError('MA7 registered source anchor changed: '+old[:100])
    return source.replace(old,new)


_HERE=Path(__file__).resolve().parent
_BASE_PATH=_HERE/'ma7_fullstep_base.py'
_FROZEN=_BASE_PATH.is_file()
if not _BASE_PATH.is_file():_BASE_PATH=_HERE/'ma5_fullstep.py'
_F4_PATH=_HERE/'ma7_fullstep_f4.py'
if not _F4_PATH.is_file():_F4_PATH=_HERE/'ma4_f_fullstep.py'
_source=_BASE_PATH.read_text()
_source=_source[:_source.index('\nif __name__ == "__main__":')]
_source=registered_replace(_source,'ma5','ma7',25)
_source=registered_replace(_source,'MA5','MA7',14)
if _FROZEN:
    # The returned controller is a read-only replay entry. Its generated
    # worker/adapter already exist and are hashed; source-generation imports
    # and server launch dependencies must not leak in from another checkout.
    _source=registered_replace(_source,'import ablation4_runtime as communication','import communication')
    _source=registered_replace(_source,'import ablation4_protocol as routing','import routing')
    _source=registered_replace(_source,'import ablation4_sources as sources','sources = None')
    _source=registered_replace(_source,'from run import arguments as model_arguments',
        "def model_arguments(*args, **kwargs):\n    raise RuntimeError('Frozen MA7 controller is read-only; use repository entry to launch')")
_source=registered_replace(_source,'MA7 tile/block tuning versus unchanged CR2; full optimizer steps.',
    'MA7 F4 / F4-wgrad; full optimizer steps; independent weight-gradient stream.')
_source=registered_replace(_source,'equivalence_pairs=[["T","CR2"]]','equivalence_pairs=bench.EQUIVALENCE_PAIRS',2)
_source=registered_replace(_source,'                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},',
    "                backward_schedule='independent_weight_stream',\n"
    '                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},')
_source=registered_replace(_source,'    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,',
                        '    freeze_helpers(scripts)\n    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,')
exec(compile(_source,str(_BASE_PATH),'exec'),globals())
__doc__ = 'One or two balanced F4 / F4-wgrad full optimizer-step comparisons.'

import ma4_f_adapter as f4_adapter
import ma7_adapter

PROTOCOL=bench.SOURCE_PROTOCOL+'-fullstep'
TARGET_MS=None
ACCEPTANCE_FIELDS=('increment_over_F4_validated',)
ACCEPTANCE_RULE=('Each candidate/F4 is accepted only when its paired point and CI95 lower are positive '
    'within every requested trial, with exact consumer outputs, full-model gradient/update fingerprints '
    'and actual native/backward execution paths. Two complete arms, all blocks retained; one trial is exploratory. '
    'No historical C/R target; forward native sources are identical.')
_prepare,_verify,_window,_preflight,_report=prepare,verify,window,preflight,report
_worker_source,_adapter_source,_taco_source=worker_source,adapter_source,taco_source


def _f4_function(name):
    text=_F4_PATH.read_text()
    nodes=[node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name==name]
    require(len(nodes)==1,'Missing or duplicate frozen F4 function: '+name)
    counts=dict(worker_source=4,validate_parameter_trajectory=0,window=6,preflight=2,trace_projection_records=0)
    return registered_replace(ast.get_source_segment(text,nodes[0]),'ma4_f','ma7',counts[name])


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
    for source,name in [(_BASE_PATH,'ma7_fullstep_base.py'),(_F4_PATH,'ma7_fullstep_f4.py'),
                        (Path(bench.__file__),'ma7_bench.py')]:
        shutil.copy2(source,scripts/name)


def verify(root,offline=False):
    require(not _FROZEN or offline,'Frozen controller supports offline replay only')
    plan=_verify(root,offline)
    bench.validate_backward_schedule(plan)
    require("BACKWARD_SCHEDULE = 'independent_weight_stream'\n" in
            (Path(root)/'scripts/adapter.py').read_text(), 'Plan/generated adapter schedule mismatch')
    for source,name in [(_BASE_PATH,'ma7_fullstep_base.py'),(_F4_PATH,'ma7_fullstep_f4.py'),
                        (bench.TEMPLATE,'ma7_bench_base.py'),(bench.base_path,'ma4_f_bench_base.py'),
                        (Path(f4_adapter.__file__),'ma4_f_adapter.py'),
                        (Path(ma7_adapter.__file__),'ma7_adapter.py'),
                        (Path(bench.ma7_sources.__file__),'ma7_sources.py'),
                        *[(_HERE/name,name) for name in ('common.py','ma4_f_sources.py','ma2_cr2_sources.py')]]:
        require(sha(source)==plan['source_files']['scripts/'+name],'Changed controller dependency: '+name)
    return plan


def adapter_source():
    return ma7_adapter.adapter_source(f4_adapter.adapter_source(_adapter_source()))


def taco_source():
    return f4_adapter.taco_source(_taco_source())


def worker_source():
    source = _consumer_worker_source()
    anchor = "adapter = Adapter(raw, POLICY, SPEC['shapes'], SPEC['ring_reduction'])"
    source = replace_once(source, anchor, anchor +
        "\nimport megatron.core.pipeline_parallel.schedules as ma7_schedules\n"
        "adapter.install_wgrad_schedule(ma7_schedules, config, model)")
    return replace_once(source,"report['ma7_consumer_state'] = adapter.consumer_state()",
        "report['ma7_backward_state'] = adapter.ma7_backward_state()\n"
        "report['ma7_consumer_state'] = adapter.consumer_state()")


def window(root,plan,job):
    rows=_consumer_window(root,plan,job)
    count=1 if job['mode']=='profile' else 3 if job['mode']=='smoke' else 50
    warmup=2 if job['mode']=='smoke' else 10
    for row in rows:
        require(len(row['ma7_trainable_parameters'])==389,'Changed trainable parameter universe')
        states=row['ma7_backward_state']
        require(set(states)==set(row['target_modules']),'Missing backward module coverage')
        for state in states.values():bench.validate_backward_state(state,job['arm'],(warmup+count)*4,training=True)
    return rows


def comparison_acceptance(estimates,mean_ms):
    require(set(estimates)=={c+'/'+b for c,b in bench.COMPARISONS} and set(mean_ms)==set(bench.ARMS),'Missing paired comparison')
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
    plan=verify(root,offline);fingerprints={};memory=defaultdict(list)
    for job in jobs(plan,'timing'):
        rows=window(Path(root),plan,job)
        key=(job['trial'],job['block']);value=[r['ma7_final_parameters_sha256'] for r in rows]
        fingerprints.setdefault(key,value);require(fingerprints[key]==value,'Final paired model parameters differ')
        for row in rows:
            require(all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) and row[k]>0
                        for k in ('peak_allocated_gib','peak_reserved_gib')), 'Missing peak CUDA memory evidence')
        memory[job['arm']].append({k:max(row[k] for row in rows)
                                  for k in ('peak_allocated_gib','peak_reserved_gib')})
    result.update(target_evaluated=False,target_percent=None,consumer_parameter_gradients_validated=True,
        backward_schedule=plan['backward_schedule'],
        peak_cuda_memory_gib={arm:{key:dict(mean_window_rank_max=statistics.mean(v[key] for v in values),
            max_window_rank_max=max(v[key] for v in values)) for key in ('peak_allocated_gib','peak_reserved_gib')}
            for arm,values in memory.items()},
        acceptance_status={arm:'positive_F4_increment' if good else 'F4_increment_not_validated'
                           for arm,good in result['increment_over_F4_validated'].items()},
        main_contrasts=[c+'/'+b for c,b in bench.MAIN_COMPARISONS],
        backward_schedule_validated=True)
    result['bootstrap']['unit']='complete fresh-process F4/F4-wgrad block'
    result.pop('target_ms',None)
    if save:
        bench.save_unchanged(Path(root)/'report.json',result)
        (Path(root)/'report.md').write_text(report_markdown(result))
        print(json.dumps(dict(comparisons=result['comparisons'],increment_over_F4_validated=result['increment_over_F4_validated']),indent=2))
    return result


def report_markdown(result):
    lines=['# MA7：F4 / F4-wgrad 完整训练结果','',
        '| trial | F4 / F4-wgrad 平均 ms | 比较 | 收益 %［95% CI］ |',
        '| --- | --- | --- | --- |']
    for name,comparison in result['comparisons'].items():
        combined=comparison['rounds']['combined']
        means=' / '.join(f"{combined['mean_ms'][arm]:.3f}" for arm in bench.ARMS)
        for label,value in combined['comparisons'].items():
            lines.append(f"| {name} | {means} | {label} | {value['percent']:+.3f} [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}] |")
    lines+=['',f"反向实验：`{result['backward_schedule']}`；两臂 native 前向相同。",
        f"全部保留 {len(result['all_samples'])} 个窗口、{result['formal_optimizer_steps']} 个正式 optimizer step。",
        '按每个独立 trial 的完整块配对判断同期 F4 正增量；单轮为探索性结果。',
        'profile 核对wgrad 提交、microbatch drain 及 CUDA stream；观察到重叠本身不代表整步收益。慢窗口不删除。']
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
    """Attribute each projection backward and its deferred work by launch correlation.

    CPU drain markers delimit microbatches. Only continuation kernels submitted
    after the node returns and before its drain count toward observed overlap.
    No positive overlap or performance result is required to validate the path.
    """
    bench.family(arm)
    candidate=arm=='F4-wgrad'
    end=lambda event:event['ts']+event.get('dur',0)
    scopes=defaultdict(list);runtimes=defaultdict(list);by_corr=defaultdict(list);by_external=defaultdict(list)
    forwards=[];backwards=[];submits=[];drains=[];all_kernels=[];launches={}
    for event in events:
        category=event.get('cat');name=event.get('name','');args=event.get('args',{})
        if category=='user_annotation':
            if name in ('scope.attn.candidate','scope.mlp.candidate'):scopes[event['tid']].append(event)
            elif name=='ma7.wgrad':submits.append(event)
            elif name=='ma7.drain':drains.append(event)
        if category in ('cuda_runtime','cuda_driver'):
            runtimes[event['tid']].append(event)
            correlation=args.get('correlation')
            if correlation is not None:launches[correlation]=event
        if category in ('kernel','gpu_memcpy','gpu_memset'):
            by_corr[args.get('correlation')].append(event);by_external[args.get('External id')].append(event)
            if category=='kernel':all_kernels.append(event)
        if category=='cpu_op' and name=='ConsumerGemmRSFunction':forwards.append(event)
        if category=='cpu_op' and name=='ConsumerGemmRSFunctionBackward':backwards.append(event)
    for values in scopes.values():values.sort(key=lambda e:e['ts'])
    for values in runtimes.values():values.sort(key=lambda e:e['ts'])
    drains.sort(key=lambda e:e['ts'])
    require(len(drains)==4,'Expected exactly four microbatch drain boundaries')
    require(len(submits)==(256 if candidate else 0),'Missing or unexpected deferred wgrad submissions')
    drain_waits=0
    for drain in drains:
        inside=[e for e in runtimes[drain['tid']] if drain['ts']<=e['ts'] and end(e)<=end(drain)+.01]
        waits=[e for e in inside if 'StreamWaitEvent' in e.get('name','')]
        require(len(waits)==int(candidate),'Each microbatch drain must issue exactly one candidate join and zero baseline joins')
        require(not any('Synchronize' in e.get('name','') for e in inside),'Drain introduced a device or stream synchronization')
        device_work=[gpu for launch in inside for gpu in by_corr[launch.get('args',{}).get('correlation')]]
        require(not device_work,'Microbatch drain unexpectedly launches kernels or copies')
        drain_waits+=len(waits)
    starts={tid:[e['ts'] for e in values] for tid,values in scopes.items()}
    rt_starts={tid:[e['ts'] for e in values] for tid,values in runtimes.items()}
    sequence_roles={}
    for call in forwards:
        tid=call['tid'];i=bisect_right(starts[tid],call['ts'])-1
        require(i>=0 and end(call)<=end(scopes[tid][i])+.01,'Missing forward role scope')
        seq=call['args']['Sequence number'];require(seq not in sequence_roles,'Duplicate autograd sequence')
        sequence_roles[seq]=scopes[tid][i]['name'].split('.')[1]
    require(len(forwards)==len(backwards)==256,'Incomplete forward/backward profile coverage')
    result=defaultdict(list);used=set();seen=set();used_submits=set();drain_coverage=Counter();weight_streams=set()
    for call in backwards:
        seq=call['args']['Sequence number'];require(seq in sequence_roles and seq not in seen,'Unmatched/duplicate backward')
        seen.add(seq);role=sequence_roles[seq];tid=call['tid']
        lo=bisect_right(rt_starts[tid],call['ts']-.001);hi=bisect_right(rt_starts[tid],end(call)+.01)
        inside=[e for e in runtimes[tid][lo:hi] if end(e)<=end(call)+.01]
        launch_by_corr={e.get('args',{}).get('correlation'):e for e in inside}
        associated={id(e):e for e in by_external[call['args']['External id']]}
        for corr in launch_by_corr:associated.update({id(e):e for e in by_corr[corr]})
        require(not any(e.get('cat') in ('gpu_memcpy','gpu_memset') for e in associated.values()),
                'Backward introduced a GPU copy or memset')
        kernels=[e for e in associated.values() if e.get('cat')=='kernel']
        bias=[e for e in kernels if 'reduce_kernel' in e['name']]
        gathers=[e for e in kernels if 'nccl' in e['name'].lower() and 'AllGather' in e['name']]
        gemms=sorted((e for e in kernels if 'gemm' in e['name'].lower()),key=lambda e:e['ts'])
        adds=[e for e in kernels if bench.is_add(e)]
        require(len(kernels)==(5 if candidate else 4) and len(bias)==len(gathers)==1 and
                len(gemms)==2 and len(adds)==int(candidate),'Backward kernel coverage changed')
        bias,ag=bias[0],gathers[0];dgrad,wgrad=gemms
        require(not used & {id(e) for e in kernels},'GPU kernel attributed twice');used.update(map(id,kernels))
        main_stream=bias['args']['stream'];weight_stream=wgrad['args']['stream'];weight_streams.add(weight_stream)
        require(main_stream==dgrad['args']['stream'] and ag['args']['stream']!=main_stream,
                'Backward compute/communication streams changed')
        require(end(bias)<=ag['ts']+.01 and end(ag)<=dgrad['ts']+.01 and end(dgrad)<=wgrad['ts']+.01,
                'Backward consumed unfinished gather/dgrad or changed original order')
        bias_launch=launch_by_corr[bias['args']['correlation']];ag_launch=launch_by_corr[ag['args']['correlation']]
        require(end(bias_launch)<=ag_launch['ts']+.01,'Original bias-before-gather submission changed')
        matching_drains=[d for d in drains if d['ts']>=end(call)-.01]
        require(matching_drains,'Backward has no following microbatch drain')
        drain=matching_drains[0];drain_coverage[id(drain)]+=1
        if candidate:
            add=adds[0]
            require(weight_stream!=main_stream and weight_stream!=ag['args']['stream'] and
                    add['args']['stream']==weight_stream and end(wgrad)<=add['ts']+.01,
                    'Deferred weight gradient/accumulation must use a separate ordered CUDA stream')
            marks=[m for m in submits if m['tid']==tid and call['ts']<=m['ts'] and end(m)<=end(call)+.01]
            require(len(marks)==1 and id(marks[0]) not in used_submits,'Missing/duplicate per-node wgrad marker')
            mark=marks[0];used_submits.add(id(mark))
            submission_runtime=[r for r in inside if mark['ts']<=r['ts'] and end(r)<=end(mark)+.01]
            waits=[r for r in submission_runtime if 'StreamWaitEvent' in r.get('name','')]
            require(len(waits)==1,'Deferred submit must wait on the producer stream exactly once')
            for gpu in (wgrad,add):
                launch=launch_by_corr[gpu['args']['correlation']]
                require(mark['ts']<=launch['ts'] and end(launch)<=end(mark)+.01,
                        'Weight work was not submitted inside the deferred scope')
        else:
            require(weight_stream==main_stream,'Baseline weight gradient moved off the original stream')
        continuation=[]
        for gpu in all_kernels:
            launch=launches.get(gpu.get('args',{}).get('correlation'))
            if (id(gpu) not in associated and gpu['args'].get('stream')==main_stream and launch is not None and
                    launch.get('tid')==tid and end(call)<=launch['ts'] and end(launch)<=drain['ts']+.01 and
                    gpu['ts']>=end(dgrad)-.01):
                continuation.append(gpu)
        # Main-stream kernels are ordered and disjoint; this sum never counts
        # host gaps or NCCL waiting as hidden weight-gradient computation.
        overlap=sum(max(0.,min(end(wgrad),end(gpu))-max(wgrad['ts'],gpu['ts'])) for gpu in continuation)
        result[role].append(dict(backward_cpu_us=call['dur'],bias_us=bias['dur'],allgather_us=ag['dur'],
            dgrad_us=dgrad['dur'],wgrad_us=wgrad['dur'],
            weight_accumulation_us=adds[0]['dur'] if adds else 0.,
            wgrad_main_continuation_overlap_us=overlap,
            continuation_kernels_before_drain=len(continuation),
            main_release_to_wgrad_end_us=end(wgrad)-end(dgrad),
            drain_cpu_us=drain['dur'],local_span_us=max(end(e) for e in kernels)-bias['ts'],
            allgather_end_to_dgrad_start_us=dgrad['ts']-end(ag)))
    require(set(result)==set(bench.ROLES) and all(len(rows)==128 for rows in result.values()),'Missing role backward calls')
    require(len(drain_coverage)==4 and set(drain_coverage.values())=={64},'Each microbatch must drain exactly 64 projection backwards')
    require(len(used_submits)==len(submits),'Unattributed deferred wgrad marker')
    require(len(weight_streams)==1,'All weight gradients must share one CUDA stream')
    return {role:dict(backward_calls=len(rows),bias_kernels=128,allgather_kernels=128,dgrad_kernels=128,wgrad_kernels=128,
        main_grad_add_kernels=128 if candidate else 0,
        backward_schedule='deferred_weight_stream' if candidate else 'baseline',
        separate_weight_stream=candidate,microbatch_drains=4,drain_stream_waits=drain_waits,weight_stream_count=len(weight_streams),
        overlap_calls=sum(r['wgrad_main_continuation_overlap_us']>0 for r in rows),
        overlap_fraction=sum(r['wgrad_main_continuation_overlap_us']>0 for r in rows)/len(rows),
        median_us={key:statistics.median(row[key] for row in rows) for key in rows[0] if key.endswith('_us')},
        continuation_calls=sum(r['continuation_kernels_before_drain']>0 for r in rows)) for role,rows in result.items()}

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
        raise RuntimeError('Frozen MA7 controller is read-only; use repository entry to prepare')

    def run(args):
        raise RuntimeError('Frozen MA7 controller is read-only; use repository entry to launch')

    def main(argv=None):
        parser=argparse.ArgumentParser(description='Read-only replay of frozen MA7 evidence; JSON is printed to stdout.')
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

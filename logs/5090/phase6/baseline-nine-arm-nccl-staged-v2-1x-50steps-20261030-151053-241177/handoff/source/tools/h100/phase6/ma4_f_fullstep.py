#!/usr/bin/env python3
"""Registered MA4-F overlay on the existing paired full-step controller.

The base controller is copied and hashed with every run. The overlay changes
only experiment identity, consumer integration, strong smoke evidence and the
consumer-to-layernorm profile. Original MA5 code and experiments are untouched.
"""
from pathlib import Path
import hashlib

_HERE = Path(__file__).resolve().parent
_BASE_PATH = _HERE/'ma4_f_fullstep_base.py'
if not _BASE_PATH.is_file():
    _BASE_PATH = _HERE/'ma5_fullstep.py'
_source = _BASE_PATH.read_text()
_main = '\nif __name__ == "__main__":'
assert _source.count(_main) == 1
_source = _source[:_source.index(_main)]
_source = (_source.replace('ma5', 'ma4_f').replace('MA5', 'MA4_F')
           .replace('H100_MA4_F', 'H100_MA4_F').replace("'T'", "'F4'").replace('"T"', '"F4"')
           .replace('T/CR2', 'F4/CR2').replace('tuning', 'mechanism').replace('tile/block', 'consumer fusion'))
_anchor = '    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,'
assert _source.count(_anchor) == 1
_source = _source.replace(_anchor, '    freeze_helpers(scripts)\n'+_anchor)
exec(compile(_source, str(_BASE_PATH), 'exec'), globals())

import ma4_f_adapter as consumer_adapter

PROTOCOL = 'h100-ma4-f-fullstep-v1'
TARGET_MS = None
ACCEPTANCE_FIELDS = ('increment_over_CR2_validated',)
ACCEPTANCE_RULE = ('Within every requested trial, F4/CR2 point speedup and CI95 lower must be >0. '
                   'All blocks retained; one trial is exploratory. Require exact consumer outputs, '
                   'every smoke parameter gradient and update fingerprint, and measured consumer removal. '
                   'C/R are not measured; the original 2% target is not evaluated.')

_prepare, _verify, _window, _preflight, _report = prepare, verify, window, preflight, report
_worker_source, _adapter_source, _taco_source = worker_source, adapter_source, taco_source


def freeze_helpers(scripts):
    shutil.copy2(_BASE_PATH, scripts/'ma4_f_fullstep_base.py')
    bench.freeze_helpers(scripts)
    shutil.copy2(Path(bench.__file__), scripts/'ma4_f_bench.py')


def verify(root, offline=False):
    plan = _verify(root, offline)
    for current, frozen in ((_BASE_PATH, 'ma4_f_fullstep_base.py'),
                            (bench.base_path, 'ma4_f_bench_base.py'),
                            (Path(consumer_adapter.__file__), 'ma4_f_adapter.py')):
        require(sha(current) == plan['source_files']['scripts/'+frozen], 'Consumer controller dependency changed: '+frozen)
    return plan


def adapter_source():
    return consumer_adapter.adapter_source(_adapter_source())


def taco_source():
    return consumer_adapter.taco_source(_taco_source())


def worker_source():
    source = _worker_source()
    source = replace_once(source, '\ndef step():', '''
consumer_smoke_step = 0
consumer_parameter_trajectory = []
consumer_parameter_universe = {name:dict(shape=list(parameter.shape),dtype=str(parameter.dtype))
    for name,parameter in raw.named_parameters() if parameter.requires_grad}
def parameter_fingerprints(gradients=False):
    result = {}
    for name, parameter in raw.named_parameters():
        if not parameter.requires_grad:
            continue
        value = getattr(parameter, 'main_grad', None) if gradients else parameter
        if gradients and value is None:
            value = parameter.grad
        assert value is not None, 'Missing trainable gradient: '+name
        result[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
            sha256=hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
    assert set(result) == set(consumer_parameter_universe)
    return result

def step():''')
    source = replace_once(source, '''    losses = forward_backward_step()
    successful, grad_norm = update()''', '''    global consumer_smoke_step
    losses = forward_backward_step()
    audit_parameters = MODE == 'smoke' and consumer_smoke_step >= SPEC['smoke_warmup_steps']
    gradients = parameter_fingerprints(True) if audit_parameters else None
    successful, grad_norm = update()
    if audit_parameters:
        consumer_parameter_trajectory.append(dict(step=len(consumer_parameter_trajectory),
            gradients=gradients, parameters=parameter_fingerprints()))
    consumer_smoke_step += 1''')
    return replace_once(source, "report['forward_checks']=adapter.forward_checks", """report['ma4_f_consumer_state'] = adapter.consumer_state()
report['ma4_f_parameter_trajectory'] = consumer_parameter_trajectory
report['ma4_f_trainable_parameters'] = consumer_parameter_universe
report['ma4_f_final_parameters_sha256'] = fingerprint(raw.named_parameters())
report['forward_checks']=adapter.forward_checks""")


def validate_parameter_trajectory(trajectory, universe):
    require(isinstance(trajectory,list) and len(trajectory)==3, 'Three full-model smoke gradient/update steps required')
    expected_names = set(universe)
    for i, record in enumerate(trajectory):
        require(set(record)=={'step','gradients','parameters'} and record['step']==i, 'Wrong smoke gradient/update step')
        keys=set(record['parameters'])
        require(keys==set(record['gradients'])==expected_names and keys, 'Missing parameter gradient or update')
        for part in ('gradients','parameters'):
            for name,value in record[part].items():
                require(set(value)=={'shape','dtype','sha256'} and isinstance(value['shape'],list) and
                        all(type(n)is int and n>0 for n in value['shape']) and value['dtype'] in ('torch.bfloat16','torch.float32') and
                        isinstance(value['sha256'],str) and len(value['sha256'])==64 and
                        all(c in '0123456789abcdef' for c in value['sha256']), 'Invalid full-parameter fingerprint')
                require(value['shape']==universe[name]['shape'],'Gradient/parameter shape differs from model')
                if part=='parameters':require(value['dtype']==universe[name]['dtype'],'Parameter dtype differs from model')


def window(root,plan,job):
    rows=_window(root,plan,job)
    count=1 if job['mode']=='profile' else 3 if job['mode']=='smoke' else 50
    warmup=2 if job['mode']=='smoke' else 10
    for row in rows:
        universe=row['ma4_f_trainable_parameters']
        required={name+suffix for name in row['target_modules'] for suffix in ('.weight','.bias')}
        require(required<=set(universe), 'Missing target weight/bias from parameter universe')
        require(all(isinstance(v,dict) and set(v)=={'shape','dtype'} and v['dtype'] in ('torch.bfloat16','torch.float32') and
                    isinstance(v['shape'],list) and v['shape'] and all(type(n)is int and n>0 for n in v['shape'])
                    for v in universe.values()), 'Invalid trainable parameter universe')
        consumer=row['ma4_f_consumer_state']
        if job['arm']=='F4':
            require(set(consumer)==set(row['target_modules']), 'Missing consumer coverage')
            for name,state in consumer.items():
                require(state['projection_calls']==state['bda_calls']==(warmup+count)*4 and state['bias_parameter_id']>0,
                        'Fused projection/BDA coverage differs')
                bench.validate_consumer(state['native'],row['ma4_f_submission_state'][name],'F4')
        else:require(consumer=={}, 'CR2 unexpectedly bypassed BDA')
        digest=row['ma4_f_final_parameters_sha256']
        require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest), 'Missing final parameter identity')
        if job['mode']=='smoke':validate_parameter_trajectory(row['ma4_f_parameter_trajectory'],universe)
        else:require(row['ma4_f_parameter_trajectory']==[], 'Per-parameter checks leaked into timing/profile')
    return rows


def preflight(root,offline=False):
    plan=verify(root,offline); reference=None
    for job in jobs(plan,'preflight'):
        rows=window(Path(root),plan,job)
        current=[(row['ma4_f_trainable_parameters'],row['ma4_f_parameter_trajectory']) for row in rows]
        reference=current if reference is None else reference
        require(current==reference,'F4/CR2 full parameter gradients or updates differ')
    return _preflight(root,offline)


def comparison_acceptance(estimates,mean_ms):
    require(set(estimates)=={'F4/CR2'} and set(mean_ms)==set(bench.ARMS),'Missing F4/CR2 comparison')
    value=estimates['F4/CR2']
    require(len(value['ci95'])==2 and all(math.isfinite(v) for v in [value['percent'],*value['ci95']]) and
            value['ci95'][0]<=value['ci95'][1], 'Invalid confidence interval')
    return dict(increment_over_CR2_validated=value['percent']>0 and value['ci95'][0]>0, target_validated=False)


def aggregate_acceptance(comparisons):
    require(len(comparisons) in (1,2),'One or two trials required')
    return dict(increment_over_CR2_validated=all(v['increment_over_CR2_validated'] for v in comparisons.values()),target_validated=False)


def report(root,offline=False,save=True):
    result=_report(root,offline,save=False)
    # The base timing statistics remain unchanged; final model bytes are also
    # compared per paired block, outside the measurement window.
    plan=verify(root,offline);fingerprints={}
    for job in jobs(plan,'timing'):
        rows=window(Path(root),plan,job)
        key=(job['trial'],job['block']);value=[r['ma4_f_final_parameters_sha256'] for r in rows]
        fingerprints.setdefault(key,value);require(fingerprints[key]==value,'Final F4/CR2 parameters differ')
    result.update(target_evaluated=False,target_percent=None,consumer_parameter_gradients_validated=True,
                  acceptance_status='positive_CR2_increment' if result['increment_over_CR2_validated'] else 'CR2_increment_not_validated')
    result.pop('target_ms',None)
    if save:
        bench.save_unchanged(Path(root)/'report.json',result)
        (Path(root)/'report.md').write_text(report_markdown(result))
        print(json.dumps(dict(comparisons=result['comparisons'],increment_over_CR2_validated=result['increment_over_CR2_validated']),indent=2))
    return result


def report_markdown(result):
    lines=['# MA4-F：融合消费链与 CR2 完整训练结果','','| 轮次 | F4 / CR2 平均 ms | F4/CR2 收益 %［95% CI］ | 正增量 |','| --- | --- | --- | --- |']
    for name,comparison in result['comparisons'].items():
        c=comparison['rounds']['combined'];value=c['comparisons']['F4/CR2']
        means=' / '.join(f"{c['mean_ms'][arm]:.3f}" for arm in bench.ARMS)
        lines.append(f"| {name} | {means} | {value['percent']:+.3f} [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}] | {'通过' if comparison['increment_over_CR2_validated'] else '未通过'} |")
    lines+=['',f"正式窗口 {len(result['all_samples'])}，正式 optimizer step {result['formal_optimizer_steps']}；全部窗口保留。",
            'F4/CR2 点估计和95%区间下界均>0才确认增量。未测C/R，原2%目标未评估。',
            'smoke检查全部参数的梯度和更新指纹；profile验证 decoder 开始至下一 LayerNorm 结束的完整消费链。']
    return '\n'.join(lines)+'\n'


def trace_projection_records(events,arm,mechanism=None):
    bench.family(arm);bench.validate_mechanism(mechanism)
    scopes=defaultdict(list);calls=[];by_external=defaultdict(list);by_correlation=defaultdict(list)
    stream_events=defaultdict(list);host_events=defaultdict(list);host_external=defaultdict(list)
    for event in events:
        category=event.get('cat');name=event.get('name','');args=event.get('args',{})
        if category=='user_annotation' and name in ('scope.attn.candidate','scope.mlp.candidate'):
            scopes[event['tid']].append(event)
        if category in ('cpu_op','cuda_runtime','cuda_driver'):
            host_events[event['tid']].append(event)
            if category=='cpu_op':host_external[args.get('External id')].append(event)
            if category=='cpu_op' and name in ('GemmRSFunction','ConsumerGemmRSFunction'):calls.append(event)
        if category in ('kernel','gpu_memcpy','gpu_memset'):
            by_external[args.get('External id')].append(event)
            by_correlation[args.get('correlation')].append(event)
            stream_events[args.get('stream')].append(event)
    for values in scopes.values():values.sort(key=lambda e:e['ts'])
    starts={key:[e['ts'] for e in values] for key,values in scopes.items()}
    for values in host_events.values():values.sort(key=lambda e:e['ts'])
    host_starts={key:[e['ts'] for e in values] for key,values in host_events.items()}
    positions={}
    for stream,values in stream_events.items():
        values.sort(key=lambda e:e['ts']);positions.update({id(e):(stream,i) for i,e in enumerate(values)})
    result=defaultdict(list);used=set()
    for call in calls:
        tid=call['tid'];i=bisect_right(starts.get(tid,[]),call['ts'])-1
        require(i>=0,'Projection lacks scope');scope=scopes[tid][i]
        require(call['ts']+call['dur']<=scope['ts']+scope['dur']+.01 and (tid,i) not in used,'Invalid projection scope')
        used.add((tid,i))
        begin=bisect_right(host_starts[tid],call['ts']-.001)
        finish=bisect_right(host_starts[tid],call['ts']+call['dur']+.01)
        inside=[e for e in host_events[tid][begin:finish] if e.get('pid')==call.get('pid') and
                e['ts']>=call['ts'] and e['ts']+e.get('dur',0)<=call['ts']+call['dur']+.01]
        require(not any(e.get('name') in ('aten::clone','aten::copy_','aten::_to_copy') or
                        e.get('name','').startswith(('cudaMemcpy','cuMemcpy')) for e in inside), 'Unexpected projection clone/copy')
        correlations={e.get('args',{}).get('correlation') for e in inside if e.get('cat') in ('cuda_runtime','cuda_driver')}-{None}
        associated={id(e):e for e in by_external[call['args']['External id']]}
        for corr in correlations:associated.update({id(e):e for e in by_correlation[corr]})
        require(not any(e.get('cat')=='gpu_memcpy' for e in associated.values()),'Unexpected projection GPU memcpy')
        path=sorted((e for e in associated.values() if e.get('cat')=='kernel'),key=lambda e:e['ts'])
        require(len(path)==3,'Projection must contain exactly GEMM, publication and decoder')
        gemm,barrier,decoder=path
        require('flux_bf16' in gemm['name'] and 'reducescatter' in gemm['name'] and
                'CudaIpcBarrierAllKernel' in barrier['name'] and bench.decoder_matches(decoder['name'],arm),
                'Wrong projection consumer path')
        stream,index=positions[id(decoder)];sequence=stream_events[stream]
        add_count=0 if arm=='F4' else 2
        require(index+add_count+1<len(sequence),'Consumer LayerNorm missing')
        adds=sequence[index+1:index+1+add_count];norm=sequence[index+1+add_count]
        require(all(bench.is_add(e) for e in adds),'CR2 bias/residual additions missing')
        name=norm.get('name','').lower()
        require(norm.get('cat')=='kernel' and ('layernorm' in name or 'layer_norm' in name),
                'Consumer does not immediately feed LayerNorm')
        norm_cpu=host_external[norm.get('args',{}).get('External id')]
        require(any(e.get('name')=='FusedLayerNormAffineFunction' for e in norm_cpu),
                'LayerNorm kernel lacks its actual fused LayerNorm CPU association')
        chain=[gemm,barrier,decoder,*adds,norm]
        require(stream is not None and len({e.get('args',{}).get('stream') for e in chain})==1 and
                all(a['ts']+a['dur']<=b['ts']+.01 for a,b in zip(chain,chain[1:])), 'Consumer stream ordering changed')
        # Count the entire original/fused decoder, including all newly fused work.
        end=norm['ts']+norm['dur']
        result[scope['name'].split('.')[1]].append(dict(cpu_us=call['dur'],decoder_us=decoder['dur'],
            decoder_start_to_layernorm_end_us=end-decoder['ts'],
            gemm_end_to_layernorm_end_us=end-(gemm['ts']+gemm['dur']),consumer_add_kernels=add_count))
    require(len(used)==sum(map(len,scopes.values())) and set(result)==set(bench.ROLES) and
            all(len(values)==128 for values in result.values()),'Incomplete layer/microbatch consumer coverage')
    return {role:dict(projection_calls=128,publication_barriers=128,consumer_add_kernels=128*(0 if arm=='F4' else 2),
                      layernorm_successors=128,fused_consumer=arm=='F4',
                      **{key:statistics.median(row[key] for row in values) for key in
                         ('cpu_us','decoder_us','decoder_start_to_layernorm_end_us','gemm_end_to_layernorm_end_us')})
            for role,values in result.items()}


if __name__=='__main__':main()

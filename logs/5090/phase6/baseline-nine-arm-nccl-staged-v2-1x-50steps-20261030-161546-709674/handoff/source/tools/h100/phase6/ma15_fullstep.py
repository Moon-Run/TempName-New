#!/usr/bin/env python3
"""Complete MA11 versus fixed-shape attention CUDA Graph replay."""
from pathlib import Path
import ast


def registered_replace(source,old,new,count=1):
    if source.count(old)!=count:
        raise ValueError('MA15 registered source anchor changed: '+old[:100])
    return source.replace(old,new)


_HERE=Path(__file__).resolve().parent
_BASE_PATH=_HERE/'ma15_fullstep_base.py'
_FROZEN=_BASE_PATH.is_file()
if not _BASE_PATH.is_file():_BASE_PATH=_HERE/'ma5_fullstep.py'
_F4_PATH=_HERE/'ma15_fullstep_f4.py'
if not _F4_PATH.is_file():_F4_PATH=_HERE/'ma4_f_fullstep.py'
_source=_BASE_PATH.read_text()
_source=_source[:_source.index('\nif __name__ == "__main__":')]
_source=registered_replace(_source,'ma5','ma15',25)
_source=registered_replace(_source,'MA5','MA15',14)
if _FROZEN:
    # The returned controller is a read-only replay entry. Its generated
    # worker/adapter already exist and are hashed; source-generation imports
    # and server launch dependencies must not leak in from another checkout.
    _source=registered_replace(_source,'import ablation4_runtime as communication','import communication')
    _source=registered_replace(_source,'import ablation4_protocol as routing','import routing')
    _source=registered_replace(_source,'import ablation4_sources as sources','sources = None')
    _source=registered_replace(_source,'from run import arguments as model_arguments',
        "def model_arguments(*args, **kwargs):\n    raise RuntimeError('Frozen MA15 controller is read-only; use repository entry to launch')")
_source=registered_replace(_source,'MA15 tile/block tuning versus unchanged CR2; full optimizer steps.',
    'MA15 versus MA11; full optimizer steps; fixed-shape attention CUDA Graph replay.')
_source=registered_replace(_source,'equivalence_pairs=[["T","CR2"]]','equivalence_pairs=bench.EQUIVALENCE_PAIRS',2)
_source=registered_replace(_source,'                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},',
    "                activation_policy='retain_owned_column_input_uninstrumented',\n"
    "                zero_count_policy='fused_optimizer_zero_count',\n"
    '                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},')
_source=registered_replace(_source,'    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,',
                        '    freeze_helpers(scripts)\n    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,')
exec(compile(_source,str(_BASE_PATH),'exec'),globals())
__doc__ = 'MA11 versus MA15; fixed-shape attention CUDA Graph replay; complete optimizer steps.'

import ma4_f_adapter as f4_adapter
import ma8_adapter
import ma9_adapter
import ma11_adapter
import ma15_profile
import ma15_adapter

PROTOCOL=bench.SOURCE_PROTOCOL+'-fullstep'
TARGET_MS=None
TARGET_PERCENT=2.0
ACCEPTANCE_FIELDS=('increment_over_MA11_validated', 'target_point_observed')
ACCEPTANCE_RULE=('Within every requested independent trial, positive increment requires paired point>0 and CI95 lower>0. '
    'The 2% incremental target over MA11 is validated only when paired point>=2 and CI95 lower>=2 in every trial; point>=2 alone is reported separately. '
    'Require exact full-model gradient/update fingerprints, identical trajectories, shared MA8-v2/MA9 paths, exact softmax smoke and independent launch-correlated profile. '
    'All balanced blocks retained; one trial is exploratory.')
_prepare,_verify,_window,_preflight,_report=prepare,verify,window,preflight,report
_worker_source,_adapter_source,_taco_source=worker_source,adapter_source,taco_source


def _f4_function(name):
    text=_F4_PATH.read_text()
    nodes=[node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name==name]
    require(len(nodes)==1,'Missing or duplicate frozen F4 function: '+name)
    counts=dict(worker_source=4,validate_parameter_trajectory=0,window=6,preflight=2,trace_projection_records=0)
    return registered_replace(ast.get_source_segment(text,nodes[0]),'ma4_f','ma15',counts[name])


# Reuse the exact strong smoke and consumer coverage checks, frozen separately.
exec(compile(_f4_function('worker_source'),str(_F4_PATH),'exec'),globals())
_consumer_worker_source=worker_source
exec(compile(_f4_function('validate_parameter_trajectory'),str(_F4_PATH),'exec'),globals())
_window_source=registered_replace(_f4_function('window'),"if job['arm']=='F4':","if job['arm'] in bench.ARMS:")
_window_source=registered_replace(_window_source,
    "bench.validate_consumer(state['native'],row['ma15_submission_state'][name],'F4')",
    "bench.validate_consumer(state['native'],row['ma15_submission_state'][name],job['arm'])")
exec(compile(_window_source,str(_F4_PATH),'exec'),globals())
_consumer_window=window
exec(compile(_f4_function('preflight'),str(_F4_PATH),'exec'),globals())


def freeze_helpers(scripts):
    bench.freeze_helpers(scripts)
    for source,name in [(_BASE_PATH,'ma15_fullstep_base.py'),(_F4_PATH,'ma15_fullstep_f4.py'),
                        (Path(bench.__file__),'ma15_bench.py'),
                        (Path(ma15_profile.__file__),'ma15_profile.py'),
                        (Path(ma15_adapter.__file__),'ma15_adapter.py'),
                        (_HERE/'ma11_profile.py','ma11_profile.py'),
                        (_HERE/'ma8_profile.py','ma8_profile.py'),
                        (_HERE/'ma9_profile.py','ma9_profile.py'),
                        (Path(ma11_adapter.__file__),'ma11_adapter.py'),
                        (Path(ma8_adapter.__file__),'ma8_adapter.py'),
                        (Path(ma9_adapter.__file__),'ma9_adapter.py')]:
        shutil.copy2(source,scripts/name)


def verify(root,offline=False):
    require(not _FROZEN or offline,'Frozen controller supports offline replay only')
    plan=_verify(root,offline)
    bench.validate_experiment(plan)
    require("ma9_adapter.install_zero_count(optimizer," in
            (Path(root)/'scripts/worker.py').read_text(), 'Plan/generated worker mechanism mismatch')
    validate_baseline_sources((Path(root)/'scripts/adapter.py').read_text(),
        (Path(root)/'scripts/taco_support.py').read_text(),
        (Path(root)/'scripts/worker.py').read_text())
    for source,name in [(_BASE_PATH,'ma15_fullstep_base.py'),(_F4_PATH,'ma15_fullstep_f4.py'),
                        (bench.TEMPLATE,'ma15_bench_base.py'),(bench.base_path,'ma4_f_bench_base.py'),
                        (bench.DRIVER,'ma15_bench_driver.py'),
                        (Path(f4_adapter.__file__),'ma4_f_adapter.py'),
                        (Path(ma15_profile.__file__),'ma15_profile.py'),
                        (Path(ma15_adapter.__file__),'ma15_adapter.py'),
                        (_HERE/'ma11_profile.py','ma11_profile.py'),
                        (_HERE/'ma8_profile.py','ma8_profile.py'),
                        (_HERE/'ma9_profile.py','ma9_profile.py'),
                        (Path(ma11_adapter.__file__),'ma11_adapter.py'),
                        (Path(ma8_adapter.__file__),'ma8_adapter.py'),
                        (Path(ma9_adapter.__file__),'ma9_adapter.py'),
                        (Path(bench.ma15_sources.__file__),'ma15_sources.py'),
                        (_HERE/'ma9_sources.py','ma9_sources.py'),
                        *[(_HERE/name,name) for name in ('common.py','ma4_f_sources.py','ma2_cr2_sources.py')]]:
        require(sha(source)==plan['source_files']['scripts/'+name],'Changed controller dependency: '+name)
    return plan


def adapter_source():
    # Exact original CR3 adapter in BOTH arms, including original env variable.
    return f4_adapter.adapter_source(_adapter_source())


def taco_source():
    # Keep even the unreachable policy error text identical to historical F4.
    return f4_adapter.taco_source(_taco_source()).replace(
        'MA15 requires the selective CR2 policy', 'MA4_F requires the selective CR2 policy')


BASELINE_ADAPTER_SHA256 = 'a66c0124e514bc222a6b48eb2cd3da47263205ce393fb0ca6d1a0625073597c9'
BASELINE_TACO_SHA256 = 'd994d174d2a367d2171f45f5c6dd977c84f697e20ee0728fb018b07728f66755'
# Actual historical ma4-f-1x-50steps-20261025-115124-977713 worker, not MA8.
# AST checks ignore only formatting; every operator, call and statement counts.
BASELINE_FUNCTIONS = {
    'fingerprint': '88753042ebba0fddfb21d236df4da461f2e83eb4187fffa9621abf33509d0dc5',
    'forward_step': '9f92bfcb7794bf834689382a75e261487a71a44c0ba79226c6a88154b1c12313',
    'forward_backward_step': '79d6a48241f0c523c830c44ea1c5a282fa06c65180bf01852629ee8f0bbc5931',
    'update': 'fc167ce4b4dbda996b701b32865b0e9e81fa19f0cd8cf8a6edbd3bf9a039e092',
    'parameter_fingerprints': '385008369c6d0b3fb3a824205ca7e08c83c32a63e7deb7a0812abda7637b88ef',
    'step': '922ca58b6095784e9205e22804232fc1a01333d69534a12a0c083ba3800bfd33',
    'clear_gradients': 'bc7d91287017387363c32220e94dee782bc65d2b09126ba803925e94bb1b4f29',
    'loss_number': 'f03bda81a85e5cbf9e21bf5a697cc8b0d2d1da189016a942afe62ad8d002d4ed',
}
BASELINE_TIMING_SHA256 = '8fa72ca9d33e273f626ae082901facace07355fcfe6ddb4f87d85350699117b3'


def validate_baseline_sources(adapter, taco, worker):
    require(hashlib.sha256(adapter.encode()).hexdigest() == BASELINE_ADAPTER_SHA256,
            'Model adapter is not the original CR3/F4 row bytes')
    require(hashlib.sha256(taco.encode()).hexdigest() == BASELINE_TACO_SHA256,
            'TACO frontend is not the original CR3/F4 row bytes')
    tree = ast.parse(worker)
    functions = {n.name:hashlib.sha256(ast.dump(n).encode()).hexdigest()
                 for n in tree.body if isinstance(n, ast.FunctionDef)}
    require(functions == BASELINE_FUNCTIONS, 'Original CR3 training-step function changed')
    regions = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test)=="MODE == 'profile'"]
    require(len(regions)==1 and hashlib.sha256(ast.dump(regions[0]).encode()).hexdigest()==BASELINE_TIMING_SHA256,
            'Original CR3 timing/profile loop changed')
    cache_calls = [n for n in tree.body if isinstance(n, ast.Assign) and
                   ast.unparse(n) == "ma15_activation_setup = ma8_adapter.install_activation_cache(raw, 'F4-cache')"]
    zero_calls = [n for n in tree.body if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id == 'ma15_zero_tracker' for t in n.targets)]
    softmax_setup = [n for n in tree.body if isinstance(n, ast.Assign) and
                     ast.unparse(n) == 'ma15_softmax_tracker = ma11_adapter.install_softmax(raw, ma15_library, MODE)']
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import) and
               any(a.name == 'ma11_adapter' for a in n.names)]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Attribute) and n.func.attr == 'install_softmax']
    require(len(cache_calls) == len(zero_calls) == len(softmax_setup) == len(imports) == len(calls) == 1,
            'Both MA15 arms require exactly the same MA8/MA9/MA11 installation')
    require(isinstance(zero_calls[0].value, ast.Call) and
            ast.unparse(zero_calls[0].value.func) == 'ma9_adapter.install_zero_count' and
            tree.body.index(cache_calls[0]) < tree.body.index(zero_calls[0]) < tree.body.index(softmax_setup[0]),
            'Shared MA11 mechanism setup changed')
    candidate = [n for n in tree.body if isinstance(n, ast.If) and
                 ast.unparse(n.test) == "os.environ['H100_MA15_ARM'] == 'F4-cache-zcount-csoftmax-agraph'"]
    expected_graph = ast.parse(
        "if os.environ['H100_MA15_ARM'] == 'F4-cache-zcount-csoftmax-agraph':\n"
        "    import ma15_adapter\n"
        "    ma15_attention_graph_tracker = ma15_adapter.install_attention_graphs(raw, ma15_library, MODE)\n").body[0]
    graph_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                   isinstance(n.func, ast.Attribute) and n.func.attr == 'install_attention_graphs']
    require(len(candidate) == len(graph_calls) == 1 and
            ast.dump(candidate[0]) == ast.dump(expected_graph) and
            tree.body.index(softmax_setup[0]) < tree.body.index(candidate[0]),
            'MA15 requires candidate-only attention graph installation after original MA11')
    require(not any(token in worker for token in ('ma8_cache_tracker','ma8_activation_cache_state','ma8.column.')),
            'Per-column instrumentation is forbidden in MA15')
    return dict(original_adapter_sha256=BASELINE_ADAPTER_SHA256,
                original_taco_sha256=BASELINE_TACO_SHA256,
                original_training_functions=True, original_timing_loop=True,
                candidate_only_graph_installation=True, shared_softmax_installation=True, shared_cache_installation=True,
                shared_zero_count_installation=True,
                control=bench.ARMS[0], candidate=bench.ARMS[1])


def worker_source():
    source = _consumer_worker_source()
    anchor = "adapter = Adapter(raw, POLICY, SPEC['shapes'], SPEC['ring_reduction'])"
    source = replace_once(source, anchor,
        "os.environ['H100_MA4_F_ARM'] = 'F4'\n" + anchor +
        "\nimport ma8_adapter\nimport ma9_adapter\nimport ctypes\n"
        "ma15_activation_setup = ma8_adapter.install_activation_cache(raw, 'F4-cache')\n"
        "ma15_library = ctypes.CDLL(str((Path(os.environ['E2E_BUILD']) / POLICY / 'libflux_cuda.so').resolve()))\n"
        "ma15_zero_tracker = ma9_adapter.install_zero_count(optimizer, ma15_library, MODE)\n"
        "import ma11_adapter\n"
        "ma15_softmax_tracker = ma11_adapter.install_softmax(raw, ma15_library, MODE)")
    source = replace_once(source,
        "ma15_softmax_tracker = ma11_adapter.install_softmax(raw, ma15_library, MODE)",
        "ma15_softmax_tracker = ma11_adapter.install_softmax(raw, ma15_library, MODE)\n"
        "ma15_attention_graph_tracker = None\n"
        "if os.environ['H100_MA15_ARM'] == 'F4-cache-zcount-csoftmax-agraph':\n"
        "    import ma15_adapter\n"
        "    ma15_attention_graph_tracker = ma15_adapter.install_attention_graphs(raw, ma15_library, MODE)")
    return replace_once(source, "report['ma15_consumer_state'] = adapter.consumer_state()",
        "report['ma15_activation_setup'] = ma15_activation_setup\n"
        "report['ma15_zero_count_state'] = ma15_zero_tracker.state()\n"
        "report['ma15_softmax_state'] = ma15_softmax_tracker.state()\n"
        "report['ma15_attention_graph_state'] = (ma15_attention_graph_tracker.state() if ma15_attention_graph_tracker is not None else {'installed': False})\n"
        "report['ma15_consumer_state'] = adapter.consumer_state()")


def validate_zero_count_state(state, arm, calls, mode, rank):
    require(arm in bench.ARMS and state.get('protocol') == ma9_adapter.PROTOCOL
            and state.get('installed') is True and state.get('failed') is False,
            'Wrong MA9 optimizer statistics identity')
    require(state.get('calls') == calls and state.get('kernel_calls') == 2*calls
            and state.get('initializations') == 1, 'Wrong zero-count execution or buffer reuse')
    require(state.get('selected_tensors') == (389 if rank == 0 else 194)
            and state.get('gradient_elements', 0) > 0 and state.get('chunks', 0) > 0,
            'Wrong original shared/TP parameter filtering')
    require(state.get('audits') == (calls if mode == 'smoke' else 0),
            'Missing exact statistic audit or audit leaked into timing')
    if mode == 'smoke':
        require(state.get('self_test') == dict(passed=True, cases=6, ordered_above_2pow24=True,
                    nan_inf_subnormal_signed_zero=True), 'Native statistic edge cases not validated')
    else:
        require(state.get('self_test') is None, 'Self-test leaked into measured window')


def validate_activation_setup(metadata, target_modules):
    expected={name.replace('.self_attention.linear_proj', '.self_attention.linear_qkv')
             .replace('.mlp.linear_fc2', '.mlp.linear_fc1'):
             'qkv' if name.endswith('.self_attention.linear_proj') else 'fc1'
             for name in target_modules}
    require(len(expected)==64 and metadata==dict(
        protocol=ma8_adapter.PROTOCOL, arm='F4-cache', installed=True,
        function='_MA8CachedRawLinear', module_roles=expected,
        source_sha256=dict(class_source=ma8_adapter.CLASS_SHA256,
                           column_forward=ma8_adapter.COLUMN_FORWARD_SHA256,
                           linear_wrapper=ma8_adapter.LINEAR_WRAPPER_SHA256)),
        'Both MA15 arms must install the exact MA8-v2 cache with all 64 column modules')


def validate_softmax_state(state, arm, steps, mode):
    require(arm in bench.ARMS, 'Unknown MA15 softmax arm')
    helper = ma11_adapter
    require(state.get('protocol') == helper.PROTOCOL and
            state.get('installed') is True and state.get('shape') == list(helper.SHAPE) and
            state.get('original_class_sha256') == helper.CLASS_SHA256 and
            state.get('original_mask_sha256') == helper.MASK_SHA256,
            'Invalid shared MA11 softmax identity')
    modules = state.get('modules', [])
    expected = {f'decoder.layers.{i}.self_attention.core_attention.scale_mask_softmax' for i in range(32)}
    require(set(modules) == expected and len(modules) == 32, 'MA15 attention module coverage changed')
    expected_audits = steps * 128 if mode == 'smoke' else 0
    require(state.get('forward_audits') == state.get('backward_audits') == expected_audits,
            'Softmax smoke audit missing or audit leaked into timing')
    expected_test = dict(passed=True, cases=6, forward_bitwise=True,
                         backward_bitwise=True, rng_unchanged=True)
    require(state.get('self_test') == (expected_test if mode == 'smoke' else None),
            'Softmax native arithmetic self-test missing or leaked into timing')


def window(root,plan,job):
    rows=_consumer_window(root,plan,job)
    count=1 if job['mode']=='profile' else 3 if job['mode']=='smoke' else 50
    warmup=2 if job['mode']=='smoke' else 10
    for row in rows:
        require(len(row['ma15_trainable_parameters'])==389, 'Changed trainable parameter universe')
        validate_activation_setup(row['ma15_activation_setup'],row['target_modules'])
        validate_softmax_state(row['ma15_softmax_state'],job['arm'],warmup+count,job['mode'])
        validate_zero_count_state(row['ma15_zero_count_state'], job['arm'], warmup+count, job['mode'], row['rank'])
        state = row['ma15_attention_graph_state']
        if job['arm'] == bench.ARMS[0]:
            require(state == {'installed': False}, 'MA11 control unexpectedly installed attention graphs')
        else:
            ma15_adapter.validate_state(state, warmup + count, job['mode'])
    return rows



def comparison_acceptance(estimates,mean_ms):
    require(set(estimates)=={'F4-cache-zcount-csoftmax-agraph/F4-cache-zcount-csoftmax'} and set(mean_ms)==set(bench.ARMS), 'Missing paired comparison')
    value=estimates['F4-cache-zcount-csoftmax-agraph/F4-cache-zcount-csoftmax']
    require(len(value['ci95'])==2 and all(math.isfinite(v) for v in [value['percent'],*value['ci95']])
            and value['ci95'][0]<=value['ci95'][1], 'Invalid paired interval')
    require(all(math.isfinite(v) and v>0 for v in mean_ms.values()), 'Invalid mean step time')
    return dict(increment_over_MA11_validated={'F4-cache-zcount-csoftmax-agraph':value['percent']>0 and value['ci95'][0]>0},
        target_point_observed=value['percent']>=TARGET_PERCENT,
        target_validated=value['percent']>=TARGET_PERCENT and value['ci95'][0]>=TARGET_PERCENT)


def aggregate_acceptance(comparisons):
    require(len(comparisons) in (1,2), 'One or two independent trials required')
    return dict(increment_over_MA11_validated={'F4-cache-zcount-csoftmax-agraph':all(v['increment_over_MA11_validated']['F4-cache-zcount-csoftmax-agraph'] for v in comparisons.values())},
        target_point_observed=all(v['target_point_observed'] for v in comparisons.values()),
        target_validated=all(v['target_validated'] for v in comparisons.values()))


def report(root,offline=False,save=True):
    result=_report(root,offline,save=False)
    plan=verify(root,offline);fingerprints={};memory=defaultdict(list)
    for job in jobs(plan,'timing'):
        rows=window(Path(root),plan,job)
        key=(job['trial'],job['block']);value=[r['ma15_final_parameters_sha256'] for r in rows]
        fingerprints.setdefault(key,value);require(fingerprints[key]==value,'Final paired model parameters differ')
        for row in rows:
            require(all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) and row[k]>0
                        for k in ('peak_allocated_gib','peak_reserved_gib')), 'Missing peak CUDA memory evidence')
        memory[job['arm']].append({k:max(row[k] for row in rows)
                                  for k in ('peak_allocated_gib','peak_reserved_gib')})
    result.update(target_evaluated=True,target_percent=TARGET_PERCENT,consumer_parameter_gradients_validated=True,
        baseline_identity=validate_baseline_sources((Path(root)/'scripts/adapter.py').read_text(),
            (Path(root)/'scripts/taco_support.py').read_text(),(Path(root)/'scripts/worker.py').read_text()),
        activation_policy=plan['activation_policy'], zero_count_policy=plan['zero_count_policy'],
        peak_cuda_memory_gib={arm:{key:dict(mean_window_rank_max=statistics.mean(v[key] for v in values),
            max_window_rank_max=max(v[key] for v in values)) for key in ('peak_allocated_gib','peak_reserved_gib')}
            for arm,values in memory.items()},
        acceptance_status={arm:'positive_MA11_increment' if good else 'MA11_increment_not_validated'
                           for arm,good in result['increment_over_MA11_validated'].items()},
        main_contrasts=[c+'/'+b for c,b in bench.MAIN_COMPARISONS],
        zero_count_counters_validated=True, zero_statistics_exact_smoke_validated=True,
        activation_setup_validated=True, softmax_setup_validated=True, softmax_exact_smoke_validated=True,
        attention_graph_state_validated=True,
        column_per_call_instrumentation=False,
        target_reference='F4-cache-zcount-csoftmax')
    result['bootstrap']['unit']='complete fresh-process F4-cache-zcount-csoftmax/F4-cache-zcount-csoftmax-agraph block'
    result.pop('target_ms',None)
    if save:
        bench.save_unchanged(Path(root)/'report.json',result)
        (Path(root)/'report.md').write_text(report_markdown(result))
        print(json.dumps(dict(comparisons=result['comparisons'],increment_over_MA11_validated=result['increment_over_MA11_validated']),indent=2))
    return result


def report_markdown(result):
    lines=['# MA15：相对 MA11 的 attention core CUDA Graph 重放', '',
        '| trial | F4-cache-zcount-csoftmax / F4-cache-zcount-csoftmax-agraph 平均 ms | 收益 %［95% CI］ | 点估计 ≥2% | 区间下界 ≥2% |',
        '| --- | --- | --- | --- | --- |']
    for name,comparison in result['comparisons'].items():
        combined=comparison['rounds']['combined']
        means=' / '.join(f"{combined['mean_ms'][arm]:.3f}" for arm in bench.ARMS)
        value=combined['comparisons']['F4-cache-zcount-csoftmax-agraph/F4-cache-zcount-csoftmax']
        lines.append(f"| {name} | {means} | {value['percent']:+.3f} [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}] | {comparison['target_point_observed']} | {comparison['target_validated']} |")
    lines+=['', f"保留全部 {len(result['all_samples'])} 个窗口、{result['formal_optimizer_steps']} 个正式 optimizer step。",
        '每个 trial 分别验收；单轮仍为探索性结果。profile 必须另行通过，完整实验状态以 batch-state.json 为准。',
        '两组均使用原 MA8-v2、MA9 与 MA11 前后向；候选仅重放 attention core 的固定前后向 CUDA Graph。',
        '2% 为相对同期 F4-cache-zcount-csoftmax 的增量目标，不是相对原 CR3 的总收益。',
        '两组 smoke 核对零统计及 softmax 前后向字节，attention core smoke 核对图重放前后向；审计不进入正式计时。',
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
    evidence=ma15_profile.trace_composed_records(events,arm)
    result={role:dict(value,backward=evidence['row_backward'][role]) for role,value in forward.items()}
    result.update(activation_cache=evidence['activation_cache'],zero_count=evidence['zero_count'],
                  attention_softmax=evidence['attention_softmax'],
                  attention_graph=evidence['attention_graph'])
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
        raise RuntimeError('Frozen MA15 controller is read-only; use repository entry to prepare')

    def run(args):
        raise RuntimeError('Frozen MA15 controller is read-only; use repository entry to launch')

    def main(argv=None):
        parser=argparse.ArgumentParser(description='Read-only replay of frozen MA15 evidence; JSON is printed to stdout.')
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

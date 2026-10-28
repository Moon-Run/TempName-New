#!/usr/bin/env python3
"""Original CR3/F4 versus exact DDP gradient accumulation fusion."""
from pathlib import Path
from collections import Counter
import ast


def registered_replace(source,old,new,count=1):
    if source.count(old)!=count:
        raise ValueError('MA10 registered source anchor changed: '+old[:100])
    return source.replace(old,new)


_HERE=Path(__file__).resolve().parent
_BASE_PATH=_HERE/'ma10_fullstep_base.py'
_FROZEN=_BASE_PATH.is_file()
if not _BASE_PATH.is_file():_BASE_PATH=_HERE/'ma5_fullstep.py'
_F4_PATH=_HERE/'ma10_fullstep_f4.py'
if not _F4_PATH.is_file():_F4_PATH=_HERE/'ma4_f_fullstep.py'
_source=_BASE_PATH.read_text()
_source=_source[:_source.index('\nif __name__ == "__main__":')]
_source=registered_replace(_source,'ma5','ma10',25)
_source=registered_replace(_source,'MA5','MA10',14)
if _FROZEN:
    # The returned controller is a read-only replay entry. Its generated
    # worker/adapter already exist and are hashed; source-generation imports
    # and server launch dependencies must not leak in from another checkout.
    _source=registered_replace(_source,'import ablation4_runtime as communication','import communication')
    _source=registered_replace(_source,'import ablation4_protocol as routing','import routing')
    _source=registered_replace(_source,'import ablation4_sources as sources','sources = None')
    _source=registered_replace(_source,'from run import arguments as model_arguments',
        "def model_arguments(*args, **kwargs):\n    raise RuntimeError('Frozen MA10 controller is read-only; use repository entry to launch')")
_source=registered_replace(_source,'MA10 tile/block tuning versus unchanged CR2; full optimizer steps.',
    'MA10 F4 / F4-gacc; full optimizer steps; bounded exact DDP gradient accumulation.')
_source=registered_replace(_source,'equivalence_pairs=[["T","CR2"]]','equivalence_pairs=bench.EQUIVALENCE_PAIRS',2)
_source=registered_replace(_source,'                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},',
    "                gradient_batching_policy='bounded_ddp_gradient_batching',\n"
    '                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},')
_source=registered_replace(_source,'    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,',
                        '    freeze_helpers(scripts)\n    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,')
exec(compile(_source,str(_BASE_PATH),'exec'),globals())
__doc__ = 'One or two balanced F4 / F4-gacc full optimizer-step comparisons.'

import ma4_f_adapter as f4_adapter
import ma10_adapter
import ma10_profile

PROTOCOL=bench.SOURCE_PROTOCOL+'-fullstep'
TARGET_MS=None
TARGET_PERCENT=2.0
ACCEPTANCE_FIELDS=('increment_over_F4_validated', 'target_point_observed')
ACCEPTANCE_RULE=('Within every requested independent trial, positive increment requires paired point>0 and CI95 lower>0. '
    'The 2% target is validated only when paired point>=2 and CI95 lower>=2 in every trial; point>=2 alone is reported separately. '
    'Require exact full-model gradient/update fingerprints, identical trajectories, original F4 model path and exact microbatch gradient accumulation. '
    'All balanced blocks retained; one trial is exploratory.')
_prepare,_verify,_window,_preflight,_report=prepare,verify,window,preflight,report
_worker_source,_adapter_source,_taco_source=worker_source,adapter_source,taco_source


def _f4_function(name):
    text=_F4_PATH.read_text()
    nodes=[node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name==name]
    require(len(nodes)==1,'Missing or duplicate frozen F4 function: '+name)
    counts=dict(worker_source=4,validate_parameter_trajectory=0,window=6,preflight=2,trace_projection_records=0)
    return registered_replace(ast.get_source_segment(text,nodes[0]),'ma4_f','ma10',counts[name])


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
    for source,name in [(_BASE_PATH,'ma10_fullstep_base.py'),(_F4_PATH,'ma10_fullstep_f4.py'),
                        (Path(bench.__file__),'ma10_bench.py'),
                        (Path(ma10_profile.__file__),'ma10_profile.py')]:
        shutil.copy2(source,scripts/name)


def verify(root,offline=False):
    require(not _FROZEN or offline,'Frozen controller supports offline replay only')
    plan=_verify(root,offline)
    bench.validate_experiment(plan)
    require("ma10_adapter.install_gradient_batching(ma10_gradient_tracker," in
            (Path(root)/'scripts/worker.py').read_text(), 'Plan/generated worker mechanism mismatch')
    validate_baseline_sources((Path(root)/'scripts/adapter.py').read_text(),
        (Path(root)/'scripts/taco_support.py').read_text(),
        (Path(root)/'scripts/worker.py').read_text())
    for source,name in [(_BASE_PATH,'ma10_fullstep_base.py'),(_F4_PATH,'ma10_fullstep_f4.py'),
                        (bench.TEMPLATE,'ma10_bench_base.py'),(bench.base_path,'ma4_f_bench_base.py'),
                        (Path(f4_adapter.__file__),'ma4_f_adapter.py'),
                        (Path(ma10_adapter.__file__),'ma10_adapter.py'),
                        (Path(ma10_profile.__file__),'ma10_profile.py'),
                        (Path(bench.ma10_sources.__file__),'ma10_sources.py'),
                        *[(_HERE/name,name) for name in ('common.py','ma4_f_sources.py','ma2_cr2_sources.py')]]:
        require(sha(source)==plan['source_files']['scripts/'+name],'Changed controller dependency: '+name)
    return plan


def adapter_source():
    # Exact original CR3 adapter in BOTH arms, including original env variable.
    return f4_adapter.adapter_source(_adapter_source())


def taco_source():
    # Keep even the unreachable policy error text identical to historical F4.
    return f4_adapter.taco_source(_taco_source()).replace(
        'MA10 requires the selective CR2 policy', 'MA4_F requires the selective CR2 policy')


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
            'Model adapter is not the original CR3/F4 bytes')
    require(hashlib.sha256(taco.encode()).hexdigest() == BASELINE_TACO_SHA256,
            'TACO frontend is not the original CR3/F4 bytes')
    tree = ast.parse(worker)
    functions = {n.name:hashlib.sha256(ast.dump(n).encode()).hexdigest()
                 for n in tree.body if isinstance(n, ast.FunctionDef)}
    require(functions == BASELINE_FUNCTIONS, 'Original CR3 training-step function changed')
    regions = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test)=="MODE == 'profile'"]
    require(len(regions)==1 and hashlib.sha256(ast.dump(regions[0]).encode()).hexdigest()==BASELINE_TIMING_SHA256,
            'Original CR3 timing/profile loop changed')
    guarded = [n for n in tree.body if isinstance(n, ast.If) and
               ast.unparse(n.test) == "os.environ['H100_MA10_ARM'] == 'F4-gacc'"]
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import) and
               any(alias.name == 'ma10_adapter' for alias in n.names)]
    prepares = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                isinstance(n.func, ast.Attribute) and n.func.attr == 'prepare_gradient_batching']
    installs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                isinstance(n.func, ast.Attribute) and n.func.attr == 'install_gradient_batching']
    setups = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
              and isinstance(n.value.func, ast.Name) and n.value.func.id == 'setup_model_and_optimizer']
    require(len(guarded) == 2 and len(imports) == len(prepares) == len(installs) == len(setups) == 1 and
            imports[0] in list(ast.walk(guarded[0])) and prepares[0] in list(ast.walk(guarded[0])) and
            installs[0] in list(ast.walk(guarded[1])) and
            tree.body.index(guarded[0]) < tree.body.index(setups[0]) < tree.body.index(guarded[1]),
            'MA10 hook preparation and installation must be candidate-only around model construction')
    return dict(original_adapter_sha256=BASELINE_ADAPTER_SHA256,
                original_taco_sha256=BASELINE_TACO_SHA256,
                original_training_functions=True, original_timing_loop=True,
                candidate_only_installation=True)


def worker_source():
    source = _consumer_worker_source()
    setup = 'model, optimizer, scheduler = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)'
    source = replace_once(source, setup,
        "ma10_gradient_tracker = None\n"
        "if os.environ['H100_MA10_ARM'] == 'F4-gacc':\n"
        "    import ma10_adapter\n"
        "    ma10_gradient_tracker = ma10_adapter.prepare_gradient_batching(MODE)\n" + setup +
        "\nif os.environ['H100_MA10_ARM'] == 'F4-gacc':\n"
        "    import ctypes\n"
        "    ma10_adapter.install_gradient_batching(ma10_gradient_tracker, model,\n"
        "        ctypes.CDLL(str((Path(os.environ['E2E_BUILD']) / POLICY / 'libflux_cuda.so').resolve())), MODE)")
    anchor = "adapter = Adapter(raw, POLICY, SPEC['shapes'], SPEC['ring_reduction'])"
    source = replace_once(source, anchor, "os.environ['H100_MA4_F_ARM'] = 'F4'\n" + anchor)
    return replace_once(source, "report['ma10_consumer_state'] = adapter.consumer_state()",
        "report['ma10_gradient_batching_state'] = (ma10_gradient_tracker.state() if ma10_gradient_tracker is not None else {'installed': False})\n"
        "report['ma10_consumer_state'] = adapter.consumer_state()")


def validate_gradient_batching_state(state, arm, calls, mode, rank):
    require(type(rank) is int and 0 <= rank < 8, 'Invalid TP rank')
    if arm == 'F4':
        require(state == {'installed': False}, 'F4 must have no MA10 adapter or runtime counters')
        return
    require(arm == 'F4-gacc' and state.get('protocol') == ma10_adapter.PROTOCOL
            and state.get('installed') is True and state.get('bound') is True
            and state.get('failed') is False and state.get('original_factory_restored') is True,
            'Wrong MA10 gradient accumulation identity or un-restored hook factory')
    microbatches = calls * 4
    require(state.get('parameter_count') == 389 and state.get('microbatches') == microbatches
            and state.get('hooks') == state.get('tensor_commits') == 389 * microbatches,
            'Incomplete gradient hook execution or commit coverage')
    require(type(state.get('flushes')) is int and 0 < state['flushes'] < state['tensor_commits']
            and state.get('kernel_calls') == state['flushes'],
            'Gradient batches were not executed or fell back to per-tensor kernels')
    require(state.get('pending_tensors') == state.get('pending_bytes') == 0,
            'Gradient writes remain pending at the full-step boundary')
    require(state.get('tensor_limit') == 32 and state.get('byte_limit') == 67108864
            and type(state.get('max_pending_tensors')) is int and 1 <= state['max_pending_tensors'] <= 32
            and type(state.get('largest_source_bytes')) is int and state['largest_source_bytes'] > 0
            and type(state.get('max_pending_bytes')) is int
            and 0 < state['max_pending_bytes'] <= max(state['byte_limit'], state['largest_source_bytes']),
            'Gradient queue exceeded its registered tensor/byte bound')
    require(state.get('source_dtype') == 'torch.bfloat16' and state.get('target_dtype') == 'torch.float32',
            'Gradient accumulation precision changed')
    require(state.get('audits') == (state['flushes'] if mode == 'smoke' else 0),
            'Missing exact accumulation audit or audit leaked into timing')
    if mode == 'smoke':
        test = state.get('self_test')
        require(isinstance(test, dict) and test.get('passed') is True and
                type(test.get('cases')) is int and test['cases'] >= 6 and
                test.get('signed_zero_subnormal_nan_inf') is True and test.get('scratch_free') is True,
                'Native gradient accumulation self-test missing')
    else:
        require(state.get('self_test') is None, 'Self-test leaked into measured window')


def window(root,plan,job):
    rows=_consumer_window(root,plan,job)
    count=1 if job['mode']=='profile' else 3 if job['mode']=='smoke' else 50
    warmup=2 if job['mode']=='smoke' else 10
    for row in rows:
        require(len(row['ma10_trainable_parameters'])==389, 'Changed trainable parameter universe')
        validate_gradient_batching_state(row['ma10_gradient_batching_state'], job['arm'], warmup+count, job['mode'], row['rank'])
    return rows


def comparison_acceptance(estimates,mean_ms):
    require(set(estimates)=={'F4-gacc/F4'} and set(mean_ms)==set(bench.ARMS), 'Missing paired comparison')
    value=estimates['F4-gacc/F4']
    require(len(value['ci95'])==2 and all(math.isfinite(v) for v in [value['percent'],*value['ci95']])
            and value['ci95'][0]<=value['ci95'][1], 'Invalid paired interval')
    require(all(math.isfinite(v) and v>0 for v in mean_ms.values()), 'Invalid mean step time')
    return dict(increment_over_F4_validated={'F4-gacc':value['percent']>0 and value['ci95'][0]>0},
        target_point_observed=value['percent']>=TARGET_PERCENT,
        target_validated=value['percent']>=TARGET_PERCENT and value['ci95'][0]>=TARGET_PERCENT)


def aggregate_acceptance(comparisons):
    require(len(comparisons) in (1,2), 'One or two independent trials required')
    return dict(increment_over_F4_validated={'F4-gacc':all(v['increment_over_F4_validated']['F4-gacc'] for v in comparisons.values())},
        target_point_observed=all(v['target_point_observed'] for v in comparisons.values()),
        target_validated=all(v['target_validated'] for v in comparisons.values()))


def report(root,offline=False,save=True):
    result=_report(root,offline,save=False)
    plan=verify(root,offline);fingerprints={};memory=defaultdict(list)
    for job in jobs(plan,'timing'):
        rows=window(Path(root),plan,job)
        key=(job['trial'],job['block']);value=[r['ma10_final_parameters_sha256'] for r in rows]
        fingerprints.setdefault(key,value);require(fingerprints[key]==value,'Final paired model parameters differ')
        for row in rows:
            require(all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) and row[k]>0
                        for k in ('peak_allocated_gib','peak_reserved_gib')), 'Missing peak CUDA memory evidence')
        memory[job['arm']].append({k:max(row[k] for row in rows)
                                  for k in ('peak_allocated_gib','peak_reserved_gib')})
    result.update(target_evaluated=True,target_percent=TARGET_PERCENT,consumer_parameter_gradients_validated=True,
        baseline_identity=validate_baseline_sources((Path(root)/'scripts/adapter.py').read_text(),
            (Path(root)/'scripts/taco_support.py').read_text(),(Path(root)/'scripts/worker.py').read_text()),
        gradient_batching_policy=plan['gradient_batching_policy'],
        peak_cuda_memory_gib={arm:{key:dict(mean_window_rank_max=statistics.mean(v[key] for v in values),
            max_window_rank_max=max(v[key] for v in values)) for key in ('peak_allocated_gib','peak_reserved_gib')}
            for arm,values in memory.items()},
        acceptance_status={arm:'positive_F4_increment' if good else 'F4_increment_not_validated'
                           for arm,good in result['increment_over_F4_validated'].items()},
        main_contrasts=[c+'/'+b for c,b in bench.MAIN_COMPARISONS],
        gradient_batching_counters_validated=True)
    result['bootstrap']['unit']='complete fresh-process F4/F4-gacc block'
    result.pop('target_ms',None)
    if save:
        bench.save_unchanged(Path(root)/'report.json',result)
        (Path(root)/'report.md').write_text(report_markdown(result))
        print(json.dumps(dict(comparisons=result['comparisons'],increment_over_F4_validated=result['increment_over_F4_validated']),indent=2))
    return result


def report_markdown(result):
    lines=['# MA10：CR3/F4 与 DDP 梯度入库批量融合的完整训练结果', '',
        '| trial | F4 / F4-gacc 平均 ms | 收益 %［95% CI］ | 点估计 ≥2% | 区间下界 ≥2% |',
        '| --- | --- | --- | --- | --- |']
    for name,comparison in result['comparisons'].items():
        combined=comparison['rounds']['combined']
        means=' / '.join(f"{combined['mean_ms'][arm]:.3f}" for arm in bench.ARMS)
        value=combined['comparisons']['F4-gacc/F4']
        lines.append(f"| {name} | {means} | {value['percent']:+.3f} [{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}] | {comparison['target_point_observed']} | {comparison['target_validated']} |")
    lines+=['', f"保留全部 {len(result['all_samples'])} 个窗口、{result['formal_optimizer_steps']} 个正式 optimizer step。",
        '每个 trial 分别验收；单轮仍为探索性结果。profile 必须另行通过，完整实验状态以 batch-state.json 为准。',
        'F4 保留原 CR3 模型与训练步；候选在每次 microbatch backward 内批量执行原 FP32 main_grad += BF16 grad，保持原通信和优化器。',
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
    result=dict(forward)
    result['gradient_batching']=ma10_profile.trace_gradient_batching_records(events,arm)
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
        raise RuntimeError('Frozen MA10 controller is read-only; use repository entry to prepare')

    def run(args):
        raise RuntimeError('Frozen MA10 controller is read-only; use repository entry to launch')

    def main(argv=None):
        parser=argparse.ArgumentParser(description='Read-only replay of frozen MA10 evidence; JSON is printed to stdout.')
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

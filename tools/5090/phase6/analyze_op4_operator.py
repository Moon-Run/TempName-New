#!/usr/bin/env python3
"""Read-only OP3-combined operator timing and launch-correlated gap analysis.

Formal windows and separate profiler calls remain separate. GPU gaps describe
intervals; submission_late_us is a diagnostic bound, not a causal decomposition
of host overhead, network transfer or rank synchronization.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parents[3]
ROLES = ('attn', 'mlp')
STAGES = ('gemm', 'encode', 'nccl', 'decode')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def end(event):
    return event['ts'] + event['dur']


def contains(parent, child):
    return (parent['ts'] <= child['ts'] and end(child) <= end(parent) + .01
            and parent.get('pid') == child.get('pid') and parent.get('tid') == child.get('tid'))


def stats(values):
    require(values and all(math.isfinite(value) for value in values), 'Invalid sample')
    return dict(count=len(values), mean=mean(values), median=median(values),
                minimum=min(values), maximum=max(values))


def classify(event):
    name = event['name']
    if 'Op1AInterleavedSwizzle' in name:
        return 'gemm'
    if 'op3_warp_encode_kernel' in name:
        return 'encode'
    if 'ncclDevKernel_SendRecv' in name:
        return 'nccl'
    if 'op3_r_decode_kernel' in name:
        return 'decode'
    raise ValueError('Unexpected kernel in combined operator: ' + name)


def analyze_trace(path, role, rank, count):
    with gzip.open(path, 'rt') as handle:
        events = json.load(handle)['traceEvents']
    scopes = sorted((event for event in events if event.get('cat') == 'user_annotation'
                     and event.get('name') == f'scope.{role}.flux_tensor'), key=lambda event: event['ts'])
    require(len(scopes) == count, 'Incorrect projection scope count')
    launches = [event for event in events if event.get('cat') in ('cuda_runtime', 'cuda_driver')
                and 'launch' in event.get('name', '').lower()]
    kernels = [event for event in events if event.get('cat') == 'kernel']
    markers = [event for event in events if event.get('cat') == 'user_annotation']
    names = dict(gemm='ablation6.flux_gemm_only', encode='op3.encode.0',
                 nccl='op3.exchange.0', decode='op3.decode.0')
    claimed, calls, geometries = set(), [], {stage: Counter() for stage in STAGES}
    previous_decode, previous_scope = None, None
    for index, scope in enumerate(scopes):
        own_launches = [event for event in launches if contains(scope, event)]
        work, launch_for, cpu_stages = {}, {}, {}
        for kernel in kernels:
            matching = [launch for launch in own_launches if
                        launch['args'].get('correlation') == kernel['args'].get('correlation')]
            if not matching:
                continue
            require(len(matching) == 1 and id(kernel) not in claimed, 'Ambiguous kernel launch ownership')
            claimed.add(id(kernel))
            stage = classify(kernel)
            require(stage not in work and kernel['args'].get('device') == rank,
                    'Duplicate stage or wrong device')
            work[stage], launch_for[stage] = kernel, matching[0]
            dims = (tuple(kernel['args']['grid']), tuple(kernel['args']['block']))
            geometries[stage][dims] += 1
        require(set(work) == set(STAGES), 'Missing combined operator stage')
        for stage, name in names.items():
            matching = [event for event in markers if event['name'] == name and contains(scope, event)]
            require(len(matching) == 1 and contains(matching[0], launch_for[stage]),
                    'Kernel launch is outside its CPU stage')
            cpu_stages[stage] = matching[0]['dur']
        ordered = [work[stage] for stage in STAGES]
        require(all(end(left) <= right['ts'] + .01 for left, right in zip(ordered, ordered[1:])),
                'Projection dependency overlap/order changed')
        gaps, late = {}, {}
        for first, second in zip(STAGES, STAGES[1:]):
            key = first + '_to_' + second
            gaps[key] = max(0., work[second]['ts'] - end(work[first]))
            late[key] = max(0., launch_for[second]['ts'] - end(work[first]))
            require(late[key] <= gaps[key] + .01, 'Submission gap exceeds measured GPU gap')
        nccl = work['nccl']['args']
        require(nccl.get('In msg nelems') == nccl.get('Out msg nelems') == 7798784
                and nccl.get('dtype') == 'Byte' and nccl.get('Group size') == 8,
                'Unexpected SendRecv payload or group')
        call = dict(index=index, kernel_us={stage: work[stage]['dur'] for stage in STAGES},
                    gpu_span_us=end(work['decode']) - work['gemm']['ts'],
                    gpu_gaps_us=gaps, submission_late_us=late,
                    cpu_scope_us=scope['dur'], cpu_stages_us=cpu_stages,
                    cpu_outside_stages_us=scope['dur'] - sum(cpu_stages.values()),
                    inter_call_gpu_gap_us=None if previous_decode is None else
                        max(0., work['gemm']['ts'] - end(previous_decode)),
                    inter_call_cpu_gap_us=None if previous_scope is None else
                        max(0., scope['ts'] - end(previous_scope)))
        require(abs(call['gpu_span_us'] - sum(call['kernel_us'].values()) - sum(gaps.values())) < .02,
                'GPU stage and gap arithmetic does not reconcile')
        calls.append(call)
        previous_decode, previous_scope = work['decode'], scope
    require(len(claimed) == len(kernels) == 4 * count, 'Unowned or missing operator kernel')

    def summarize(items):
        result = {}
        for field in ('kernel_us', 'gpu_gaps_us', 'submission_late_us', 'cpu_stages_us'):
            result[field] = {key: stats([item[field][key] for item in items]) for key in items[0][field]}
        for field in ('gpu_span_us', 'cpu_scope_us', 'cpu_outside_stages_us',
                      'inter_call_gpu_gap_us', 'inter_call_cpu_gap_us'):
            result[field] = stats([item[field] for item in items if item[field] is not None])
        result['gpu_total_gap_us'] = stats([sum(item['gpu_gaps_us'].values()) for item in items])
        result['submission_late_total_us'] = stats([sum(item['submission_late_us'].values()) for item in items])
        return result

    return dict(rank=rank, role=role, calls=calls,
                all_calls=summarize(calls), after_first_call_diagnostic=summarize(calls[1:]),
                launch_geometries={stage: [dict(grid=list(grid), block=list(block), calls=number)
                                          for (grid, block), number in sorted(values.items())]
                                   for stage, values in geometries.items()})


def analyze(batch):
    batch = Path(batch).resolve()
    root = batch / 'operator'
    report = json.loads((root / 'report.json').read_text())
    require(report['completed'] and report['operator_gradients_validated']
            and report['arm'] == 'op3-combined', 'Incomplete operator result')
    evidence = {'operator/report.json': sha(root / 'report.json')}
    records = []
    for rank in range(8):
        path = root / f'rank{rank}.json'
        require(sha(path) == report['evidence_files'][path.name], 'Changed operator rank record')
        record = json.loads(path.read_text())
        require(record['rank'] == rank and record['completed'] and record['run_id'] == report['run_id'],
                'Foreign or incomplete operator rank')
        evidence[str(path.relative_to(batch))] = sha(path)
        records.append(record)
    roles = {}
    for role in ROLES:
        profiles = []
        for rank, record in enumerate(records):
            checks = record['roles'][role]['gradient_checks']
            require(len(checks) == report['settings']['gradient_steps'], 'Missing gradient-check steps')
            for step, check in enumerate(checks):
                require(check['step'] == step and check['output_bitwise'] and check['inputs_unchanged']
                        and check['output_sha256'] == check['reference_output_sha256'],
                        'Output or input-preservation check failed')
                for kind in ('gradients', 'updates'):
                    require(set(check[kind]) == {'input', 'weight', 'bias', 'residual'}
                            and all(item['bitwise'] and item['actual_sha256'] == item['reference_sha256']
                                    for item in check[kind].values()),
                            'Incomplete or non-identical operator gradient/update')
            state = record['roles'][role]['state']
            settings = report['settings']
            expected = (2 * settings['gradient_steps'] + settings['warmup']
                        + settings['windows'] * settings['iterations'] + settings['profile_calls'])
            require(all(state[key] == expected for key in
                        ('calls', 'encode_calls', 'exchange_calls', 'decode_calls', 'wait_calls'))
                    and state['fused_consumer_calls'] == expected - settings['gradient_steps']
                    and state['projection_calls'] == settings['gradient_steps']
                    and not state['failed'] and not state['pending_exchange']
                    and state['wire_bytes_per_rank'] == 7798784,
                    'Incomplete operator execution state')
            path = root / f'{role}-rank{rank}-profile.json.gz'
            digest = sha(path)
            require(digest == report['evidence_files'][path.name]
                    == record['roles'][role]['trace_sha256'], 'Changed operator trace')
            evidence[str(path.relative_to(batch))] = digest
            profile = analyze_trace(path, role, rank, report['settings']['profile_calls'])
            prior = report['profiles'][f'{role}/rank{rank}']['forward']['median_stage_us']
            for stage in STAGES:
                require(abs(profile['all_calls']['kernel_us'][stage]['median'] - prior[stage + '_us']) < .001,
                        'Replayed kernel median differs from source report')
            require(abs(profile['all_calls']['cpu_scope_us']['median'] - prior['cpu_us']) < .001
                    and abs(profile['all_calls']['gpu_span_us']['median'] - prior['gpu_span_us']) < .001,
                    'Replayed span median differs from source report')
            profiles.append(profile)
        windows = []
        for index in range(report['settings']['windows']):
            rows = [dict(rank=rank, **record['roles'][role]['windows'][index])
                    for rank, record in enumerate(records)]
            maximum = {metric: max(row[metric] for row in rows) for metric in ('gpu_us', 'wall_us')}
            require(dict(index=index, **maximum) == report['roles'][role]['all_windows'][index],
                    'Formal distributed timing differs from raw rank windows')
            windows.append(dict(index=index, ranks=rows, maximum_rank_us=maximum,
                                rank_gpu_us=stats([row['gpu_us'] for row in rows]),
                                rank_wall_us=stats([row['wall_us'] for row in rows])))
        summary = {}
        for scope in ('all_calls', 'after_first_call_diagnostic'):
            out = {}
            for field in ('kernel_us', 'gpu_gaps_us', 'submission_late_us', 'cpu_stages_us'):
                out[field] = {key: median(profile[scope][field][key]['median'] for profile in profiles)
                              for key in profiles[0][scope][field]}
            for field in ('gpu_span_us', 'cpu_scope_us', 'cpu_outside_stages_us',
                          'inter_call_gpu_gap_us', 'inter_call_cpu_gap_us', 'gpu_total_gap_us',
                          'submission_late_total_us'):
                out[field] = median(profile[scope][field]['median'] for profile in profiles)
            summary[scope + '_median_of_rank_medians'] = out
        roles[role] = dict(shape=report['roles'][role]['shape'], formal_windows=windows,
                           profile_summary=summary, profiles=profiles,
                           correctness=dict(ranks=8, steps_per_rank=report['settings']['gradient_steps'],
                                            outputs_inputs_gradients_updates_checked=True),
                           calls_per_rank=expected, wire_bytes_per_rank_per_call=7798784)
    result = dict(schema='op4-historical-operator-analysis-v1',
                  source_batch_relative_to_repository=os.path.relpath(batch, REPO),
                  settings=report['settings'], run_id=report['run_id'], evidence_files=evidence,
                  roles=roles, operator_gradients_validated=True,
                  method='Each GPU kernel is matched to a unique actual runtime launch by correlation ID, '
                         'then to its enclosing role and stage CPU scopes. All ten calls are retained. '
                         'Separate after-first summaries describe profiler startup sensitivity only.',
                  limitations=[
                      'One formal window of 100 forward calls per role; no repeat-window stability claim.',
                      'Profile calls are separate from formal timing and cannot be subtracted from it.',
                      'NCCL kernel duration includes rank waiting and instrumentation effects; it is not pure wire-transfer time.',
                      'GPU gap includes dependency and submission effects; submission_late_us is not causal host-cost accounting.',
                      'Kernel entry-point names do not establish the actual NCCL algorithm or protocol.',
                      'Rank GPU clocks are not recorded here; differing kernel times do not prove a hardware fault or clock cause.',
                      'The operator contains SendRecv but no forward AllGather/ReduceScatter, LayerNorm, backward or optimizer.',
                      'Global channel limits may change SendRecv scheduling; actual CTA count must be checked in candidate traces.',
                  ],
                  references=dict(
                      p2p_channels='https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/graph/paths.cc#L767-L795',
                      p2p_scheduler='https://github.com/NVIDIA/nccl/blob/v2.27.3-1/src/enqueue.cc#L960-L1025'))
    full_report = batch / 'report.json'
    if full_report.is_file():
        full = json.loads(full_report.read_text())
        evidence['report.json'] = sha(full_report)
        result['historical_full_step'] = dict(timing_windows=full['timing_windows'],
            comparisons=full['comparisons'], source='report.json', concurrent_op4_comparison=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    batch, out = args.batch.resolve(), args.out.resolve()
    require(batch != out and batch not in out.parents, 'Do not write inside the historical batch')
    result = analyze(batch)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(out)


if __name__ == '__main__':
    main()

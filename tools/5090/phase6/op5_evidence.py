"""OP5 admission: keep OP4 channels and require actual receive-side CE work."""
from bisect import bisect_left
import gzip
import json
import math
from pathlib import Path
from statistics import mean

import op3_protocol as p
import op4
import op4_evidence as previous

require = p.require
CHANNEL_ENV = previous.CHANNEL_ENV
CE_ENV = {'NCCL_SHM_USE_CUDA_MEMCPY': '1', 'NCCL_SHM_MEMCPY_MODE': '2',
          'NCCL_SHM_LOCALITY': '2'}
OVERRIDES = dict(CHANNEL_ENV, **CE_ENV)
# Ring all-gather receives seven 2-MiB contributions on each rank.
GATHER_RECEIVED_BYTES = 7 * 1048576 * 2


def compare_communication(reference, candidate):
    normalized = dict(candidate, environment=dict(candidate['environment']))
    for key, value in CE_ENV.items():
        require(key not in reference['environment'], 'OP5 reference has a hidden SHM override: '+key)
        require(normalized['environment'].pop(key, None) == value, 'OP5 requires '+key+'='+value)
    require(reference == normalized, 'Communication changed beyond OP5 receive-side CE settings')
    require(all(reference['environment'].get(key) == value for key, value in CHANNEL_ENV.items()),
            'OP5 requires the OP4 16-channel reference')
    forbidden = set(op4.EXTERNAL_OVERRIDES) - set(CHANNEL_ENV)
    require(not forbidden.intersection(reference['environment']), 'Hidden NCCL algorithm/protocol override')


def reference_record(root):
    """Bind both OP4 windows, rederive timing from ranks; never reload old binaries."""
    root = Path(root)
    report, plan = p.read(root/'report.json'), p.read(root/'plan.json')
    require(p.read(root/'batch-state.json')['status'] == 'completed'
            and report['protocol'] == plan['protocol'] == op4.PROTOCOL
            and report['completed'] is True and report['full_step_completed'] is True
            and report['operator_only'] is False and report['arms'] == ['candidate']
            and report['correctness_passed'] is True and report['execution_path_validated'] is True
            and plan['nccl_overrides'] == report['nccl_overrides'] == op4.OVERRIDES
            and plan['model'] == p.MODEL and not plan['operator_only'],
            'OP5 needs a completed full-step OP4 parent batch')
    require(p.sha(root/'plan.json') == report['plan_sha256']
            and p.sha(root/'candidate-evidence.json') == report['candidate_evidence_sha256'],
            'Historical OP4 parent evidence changed')
    jobs = op4.schedule(plan['trials'], plan['blocks'])
    require(plan['schedule'] == jobs and report['trials'] == plan['trials']
            and report['blocks'] == plan['blocks']
            and report['timing_windows'] == len(jobs)
            and report['formal_optimizer_steps'] == 50*len(jobs)
            and len(report['all_samples']) == len(jobs), 'Incomplete historical OP4 windows')
    files = {name: p.sha(root/name) for name in ('plan.json', 'report.json', 'candidate-evidence.json', 'batch-state.json')}
    expected = {job['directory']+suffix for job in jobs for suffix in ('/report.json', '-launch.json')}
    require(set(report['child_reports']) == expected, 'Missing/foreign historical OP4 child reports')
    for name, digest in report['child_reports'].items():
        require(p.sha(root/name) == digest, 'Historical OP4 child evidence changed: '+name)
        files[name] = digest
    children, values = [], []
    for job, recorded in zip(jobs, report['all_samples']):
        child = root/job['directory']
        require(p.read(child/'batch-state.json')['status'] == 'completed', 'Incomplete OP4 child')
        child_plan = p.read(child/'plan.json')
        env = child_plan['communication']['environment']
        require(all(env.get(k) == v for k, v in CHANNEL_ENV.items())
                and not any(k in env for k in CE_ENV), 'Historical OP4 transport changed')
        record = op4.reference_record(child)
        timing_job = p.schedule(1, 1)[0]
        rows = [p.read(child/'results'/timing_job['directory']/f'rank{rank}.json') for rank in range(8)]
        require(all(row.get('completed') is True and row.get('rank') == rank
                    and row.get('mode') == 'timing' and row.get('timed_steps') == 50
                    and row.get('warmup_steps') == 10 for rank, row in enumerate(rows)),
                'Incomplete historical OP4 timing rank')
        sample = p.summarize(rows, timing_job)
        require(sample == p.read(child/'report.json')['all_samples'][0]
                and dict(sample, **job) == recorded, 'OP4 reported time differs from raw ranks')
        require(record['historical_mean_ms'] == sample['ms_per_step'], 'OP4 child timing summary changed')
        values.append(sample['ms_per_step'])
        children.append(dict(directory=job['directory'], **record))
    for trial in range(1, plan['trials']+1):
        own = [row['ms_per_step'] for row in report['all_samples'] if row['trial'] == trial]
        require(report['comparisons'][f'compare{trial}']['arms']['candidate']['mean_ms'] == mean(own),
                'OP4 aggregate timing summary changed')
    return dict(root=str(root.resolve()), report_sha256=p.sha(root/'report.json'),
        evidence_sha256=p.digest(files), historical_mean_ms=mean(values),
        timing_windows=len(values), all_ms=values, children=children,
        numerical_reference_directory=jobs[0]['directory'], read_only=True, reference_gpu_jobs=0,
        historical_binaries_revalidated=False)


def copy_evidence(events, rank, expected_calls):
    """Admit actual HtoD activity within every AllGather, including full payload.

    Proxy-thread copies often have no record_param_comms External id. Attribute
    by same device and complete temporal containment in serialized NCCL kernels.
    Do not add memcpy duration to its enclosing NCCL duration.
    """
    gathers = sorted((e for e in events if e.get('cat') == 'kernel'
                      and 'nccl' in e.get('name', '').lower() and 'AllGather' in e['name']), key=lambda e: e['ts'])
    require(len(gathers) == expected_calls, 'Missing OP5 AllGather calls')
    copies = sorted((e for e in events if e.get('cat') == 'gpu_memcpy'
                     and ('HtoD' in e.get('name', '') or 'Host -> Device' in e.get('name', ''))
                     and e.get('args', {}).get('device') == rank), key=lambda e: e['ts'])
    starts = [e['ts'] for e in copies]
    coverage = []
    previous_end = -math.inf
    for gather in gathers:
        args = gather.get('args', {})
        start, end = gather['ts'], gather['ts']+gather['dur']
        require(math.isfinite(start) and math.isfinite(end) and end > start
                and start >= previous_end-.05 and args.get('device') == rank
                and args.get('grid') == [16, 1, 1], 'Wrong/overlapping OP5 AllGather launch')
        previous_end = end
        inside = copies[bisect_left(starts, start-.05):bisect_left(starts, end)]
        require(inside and all(math.isfinite(e['dur']) and e['dur'] > 0
                and e['ts']+e['dur'] <= end+.05 and type(e['args'].get('bytes')) is int
                and e['args']['bytes'] > 0 for e in inside),
                'AllGather lacks valid contained HtoD copies; CE path not demonstrated')
        size = sum(e['args']['bytes'] for e in inside)
        require(size == GATHER_RECEIVED_BYTES,
                f'AllGather CE payload is {size}, expected {GATHER_RECEIVED_BYTES}; inspect trace')
        coverage.append(dict(copies=len(inside), bytes=size))
    return dict(rank=rank, allgather_calls=expected_calls, ce_covered_calls=len(coverage),
        received_bytes_per_call=GATHER_RECEIVED_BYTES,
        htod_copies=sum(item['copies'] for item in coverage),
        htod_bytes=sum(item['bytes'] for item in coverage),
        attribution='same-device complete containment in nonoverlapping AllGather kernels',
        memcpy_time_is_not_additive=True)


def audit_candidate(child_root, reference_root):
    reference_root, child = Path(reference_root), Path(child_root)
    history = reference_record(reference_root)
    reference = reference_root/history['numerical_reference_directory']
    result = previous.audit_candidate(child, reference, communication_check=compare_communication,
        reference_channels=16, intervention=CE_ENV)
    result.update(historical_full_step_ms=history['historical_mean_ms'],
        historical_op4_windows=history['timing_windows'],
        historical_op4_report_sha256=history['report_sha256'],
        numerical_reference_directory=history['numerical_reference_directory'], ce_profiles=[])
    if not result['operator_only']:
        for rank in range(8):
            name = f'results/profile-{p.COMBINED}/rank{rank}-profile.json.gz'
            with gzip.open(child/name, 'rt') as stream:
                result['ce_profiles'].append(copy_evidence(json.load(stream)['traceEvents'], rank, 524))
        result['full_model_allgather_ce_validated'] = True
    else:
        result['full_model_allgather_ce_validated'] = False
    return result

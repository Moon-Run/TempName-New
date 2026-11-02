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
# NCCL 2.27.3 selects SendRecv LL independently of collective protocols.
# Keep these 1,114,112-byte peer packets off the Simple/SHM-CE path that hung.
P2P_ENV = {'NCCL_P2P_LL_THRESHOLD': '2097152'}
INTERVENTION_ENV = dict(CE_ENV, **P2P_ENV)
OVERRIDES = dict(CHANNEL_ENV, **INTERVENTION_ENV)
# Ring all-gather receives seven 2-MiB contributions on each rank.
GATHER_RECEIVED_BYTES = 7 * 1048576 * 2
# Temporal containment can attribute copies to adjacent calls. Retain a bounded
# diagnostic range, not a claim that each remote payload was fully accounted for.
MIN_GATHER_HTOD_BYTES = GATHER_RECEIVED_BYTES * 3 // 4
MAX_GATHER_HTOD_BYTES = GATHER_RECEIVED_BYTES * 5 // 4


def compare_communication(reference, candidate):
    normalized = dict(candidate, environment=dict(candidate['environment']))
    for key, value in INTERVENTION_ENV.items():
        require(key not in reference['environment'], 'OP5 reference has a hidden transport override: '+key)
        require(normalized['environment'].pop(key, None) == value, 'OP5 requires '+key+'='+value)
    require(reference == normalized, 'Communication changed beyond OP5 receive-CE/P2P-LL settings')
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
                and not any(k in env for k in INTERVENTION_ENV), 'Historical OP4 transport changed')
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
    """Admit actual HtoD activity in a bounded range within every AllGather.

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
        require(all(math.isfinite(e['dur']) and e['dur'] > 0
                and type(e['args'].get('bytes')) is int
                and e['args']['bytes'] > 0 for e in inside),
                'AllGather has invalid HtoD copy records')
        contained = [e for e in inside if e['ts']+e['dur'] <= end+.05]
        require(contained, 'AllGather lacks valid contained HtoD copies; CE path not demonstrated')
        size = sum(e['args']['bytes'] for e in contained)
        require(MIN_GATHER_HTOD_BYTES <= size <= MAX_GATHER_HTOD_BYTES,
                f'AllGather contained HtoD bytes are {size}, expected {MIN_GATHER_HTOD_BYTES}'
                f'..{MAX_GATHER_HTOD_BYTES}; inspect trace')
        coverage.append(dict(copies=len(contained), bytes=size, excluded_cross_boundary=len(inside)-len(contained)))
    return dict(rank=rank, allgather_calls=expected_calls, ce_covered_calls=len(coverage),
        expected_remote_bytes_per_call=GATHER_RECEIVED_BYTES,
        contained_htod_bytes_range=[MIN_GATHER_HTOD_BYTES, MAX_GATHER_HTOD_BYTES],
        contained_htod_bytes_per_call=[item['bytes'] for item in coverage],
        htod_copies=sum(item['copies'] for item in coverage),
        htod_bytes=sum(item['bytes'] for item in coverage),
        excluded_cross_boundary_copies_per_call=[item['excluded_cross_boundary'] for item in coverage],
        attribution='same-device complete containment in nonoverlapping AllGather kernels',
        attribution_limitation='Temporal containment does not prove copy ownership or complete payload coverage',
        memcpy_time_is_not_additive=True)


def sendrecv_evidence(events, rank, expected_calls, *, isolated=True):
    """Observe absence of CE copies during SendRecv; not a direct LL trace ID.

    NCCL 2.27.3 dispatches LL and Simple inside the same SendRecv entry. With
    SHM LL buffers and the pinned threshold, its planner selects LL for these
    packets. Require no CE in isolated probes/operators. Full-model HtoD may
    belong to other collectives: report temporal intersections without turning
    that ambiguous attribution into a new correctness gate.
    """
    require(type(rank) is int and 0 <= rank < 8 and type(expected_calls) is int
            and expected_calls > 0 and type(isolated) is bool, 'Invalid OP5 SendRecv profile identity')
    kernels = [e for e in events if e.get('cat') == 'kernel'
               and 'nccl' in e.get('name', '').lower() and 'SendRecv' in e['name']]
    require(len(kernels) == expected_calls, 'Missing OP5 SendRecv calls')
    copies = sorted((e for e in events if e.get('cat') == 'gpu_memcpy'
                     and ('HtoD' in e.get('name', '') or 'Host -> Device' in e.get('name', ''))
                     and e.get('args', {}).get('device') == rank), key=lambda e: e['ts'])
    # Prefix maximum end makes both crossing and contained copies detectable.
    starts, end_prefix = [], []
    for copy in copies:
        start, duration = copy['ts'], copy['dur']
        require(math.isfinite(start) and math.isfinite(duration) and duration > 0,
                'Invalid HtoD timing in SendRecv profile')
        starts.append(start)
        end_prefix.append(max(start+duration, end_prefix[-1] if end_prefix else -math.inf))
    intersections, per_call = set(), []
    for event in kernels:
        start, duration = event['ts'], event['dur']
        require(event.get('args', {}).get('device') == rank and math.isfinite(start)
                and math.isfinite(duration) and duration > 0, 'Invalid OP5 SendRecv kernel')
        index = bisect_left(starts, start+duration)-1
        overlapping = []
        while index >= 0 and end_prefix[index] > start:
            if copies[index]['ts']+copies[index]['dur'] > start:
                overlapping.append(index)
            index -= 1
        intersections.update(overlapping)
        per_call.append(dict(ts=start, duration_us=duration, overlapping_htod_copies=len(overlapping),
            overlapping_htod_bytes=sum(copies[i]['args'].get('bytes', 0) for i in overlapping)))
    require(not isolated or not intersections,
            'OP5 SendRecv overlaps HtoD; P2P compatibility path not demonstrated')
    return dict(rank=rank, sendrecv_calls=expected_calls, overlapping_htod_copies=len(intersections),
        isolated_profile=isolated, no_htod_overlap_required=isolated, per_call=per_call,
        actual_protocol_inferred_from_kernel_name=False,
        ll_selection_basis='NCCL 2.27.3 SHM LL buffers and fixed P2P threshold; source-based inference',
        limitation='SendRecv entry name and absence of HtoD alone do not identify its protocol; '
                   'full-model temporal overlap does not establish copy ownership')


def audit_candidate(child_root, reference_root):
    reference_root, child = Path(reference_root), Path(child_root)
    history = reference_record(reference_root)
    reference = reference_root/history['numerical_reference_directory']
    result = previous.audit_candidate(child, reference, communication_check=compare_communication,
        reference_channels=16, intervention=INTERVENTION_ENV)
    result.update(historical_full_step_ms=history['historical_mean_ms'],
        historical_op4_windows=history['timing_windows'],
        historical_op4_report_sha256=history['report_sha256'],
        numerical_reference_directory=history['numerical_reference_directory'], ce_profiles=[],
        sendrecv_profiles=[], operator_sendrecv_profiles=[])
    from op3_operator import SETTINGS
    for role in ('attn', 'mlp'):
        for rank in range(8):
            with gzip.open(child/f'operator/{role}-rank{rank}-profile.json.gz', 'rt') as stream:
                checked = sendrecv_evidence(json.load(stream)['traceEvents'], rank, SETTINGS['profile_calls'])
            result['operator_sendrecv_profiles'].append(dict(role=role, **checked))
    if not result['operator_only']:
        for rank in range(8):
            name = f'results/profile-{p.COMBINED}/rank{rank}-profile.json.gz'
            with gzip.open(child/name, 'rt') as stream:
                events = json.load(stream)['traceEvents']
            result['ce_profiles'].append(copy_evidence(events, rank, 524))
            result['sendrecv_profiles'].append(sendrecv_evidence(events, rank, 256, isolated=False))
        result['full_model_allgather_ce_validated'] = True
    else:
        result['full_model_allgather_ce_validated'] = False
    return result

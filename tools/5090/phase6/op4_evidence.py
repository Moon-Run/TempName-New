"""Audit one OP4 candidate against read-only historical OP3 evidence.

The controller must replay the candidate with op3.replay first. Historical
records and hashes are checked here; historical binaries are not revalidated.
The historical batch is not a concurrently measured performance control.
"""
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
from statistics import median

from rtx5090_loader import load
import op3_protocol as p

require = p.require
CHANNEL_ENV = {'NCCL_MIN_NCHANNELS': '16', 'NCCL_MAX_NCHANNELS': '16'}
EXPECTED_GATHERS = 524


def compare_communication(reference, candidate):
    normalized = dict(candidate, environment=dict(candidate['environment']))
    for key, value in CHANNEL_ENV.items():
        require(key not in reference['environment'], 'Historical OP3 reference must use automatic channel selection')
        require(normalized['environment'].pop(key, None) == value, 'OP4 candidate must request 16 channels')
    require(reference == normalized, 'Communication changed beyond NCCL channel count')
    forbidden = ('NCCL_PROTO', 'NCCL_ALGO', 'NCCL_TUNER_PLUGIN', 'NCCL_MIN_NRINGS',
                 'NCCL_MAX_NRINGS', 'NCCL_MIN_CTAS', 'NCCL_MAX_CTAS')
    require(not any(key in reference['environment'] for key in forbidden),
            'OP4 cannot attribute results with a hidden protocol, algorithm or channel override')


def _histogram(events, geometry=True):
    # Addresses, timestamps and numeric stream IDs differ between processes.
    # Keep entry kernel, launch geometry, payload and process-group identity.
    fields = ('Collective name', 'In msg nelems',
              'Out msg nelems', 'Group size', 'dtype', 'In split size', 'Out split size',
              'Process Group Description', 'Process Group Ranks')
    if geometry:
        fields += ('grid', 'block', 'shared memory')
    counter = Counter(json.dumps(dict(name=e['name'].split('(')[0],
        **{key: e['args'].get(key) for key in fields}), sort_keys=True) for e in events)
    return [dict(signature=json.loads(key), calls=count) for key, count in sorted(counter.items())]


def profile_evidence(events, rank, channels):
    require(channels in (4, 16) and type(rank) is int and 0 <= rank < 8,
            'Invalid OP4 profile identity')
    kernels = [e for e in events if e.get('cat') == 'kernel' and 'nccl' in e.get('name', '').lower()]
    gathers = [e for e in kernels if 'allgather' in e['name'].lower()]
    require(len(gathers) == EXPECTED_GATHERS, 'OP4 requires all 524 AllGather kernels')
    launches = defaultdict(list)
    for event in events:
        if (event.get('cat') in ('cuda_runtime', 'cuda_driver')
                and 'launch' in event.get('name', '').lower()):
            launches[event.get('args', {}).get('correlation')].append(event)
    correlations = set()
    for event in gathers:
        args = event['args']
        name = event['name'].split('(')[0].upper()
        # NCCL 2.27 maps multiple algorithms/protocols to this entry kernel.
        # The suffix is NOT evidence that LL was selected. Grid is observable.
        require(name.endswith('_ALLGATHER_RING_LL'), 'Unexpected NCCL AllGather entry kernel')
        require(args.get('grid') == [channels, 1, 1],
                'AllGather did not execute the required '+str(channels)+' channel grid')
        require(args.get('device') == rank and args.get('Group size') == 8
                and args.get('dtype') == 'BFloat16' and args.get('Collective name') == '_allgather_base'
                and args.get('In msg nelems') == 1048576 and args.get('Out msg nelems') == 8388608
                and args.get('Process Group Description') == 'TENSOR_MODEL_PARALLEL_GROUP'
                and json.loads(args.get('Process Group Ranks', 'null')) == list(range(8)),
                'AllGather device, TP group or BF16 payload changed')
        correlation = args.get('correlation')
        require(correlation is not None and len(launches[correlation]) == 1 and correlation not in correlations,
                'AllGather is missing a unique actual launch')
        correlations.add(correlation)
        launch = launches[correlation][0]
        require(all(math.isfinite(float(item[key])) for item in (event, launch) for key in ('ts', 'dur'))
                and launch['ts'] <= event['ts']+.01 and event['dur'] > 0 and launch['dur'] >= 0,
                'Invalid AllGather launch timing')
    other = [e for e in kernels if e not in gathers]
    require(any('ReduceScatter' in e['name'] for e in other)
            and any('AllReduce' in e['name'] for e in other)
            and sum('SendRecv' in e['name'] for e in other) == 256,
            'Missing original reduction or projection communication')
    return dict(rank=rank, allgather_calls=len(gathers), allgather_grid_channels=channels,
        algorithm_protocol_inferred_from_kernel_name=False,
        allgather_total_ms=sum(e['dur'] for e in gathers)/1000,
        allgather_median_us=median(e['dur'] for e in gathers),
        allgather_launches=_histogram(gathers), other_nccl_launches=_histogram(other),
        other_nccl_payloads=_histogram(other, geometry=False))


def compare_rows(reference, candidate, mode, *, communication_check=None):
    require(mode in ('smoke', 'profile', 'timing') and len(reference) == len(candidate) == 8,
            'OP4 requires eight historical and candidate rank records')
    for rows in (reference, candidate):
        require([row['rank'] for row in rows] == list(range(8)), 'Historical/candidate ranks are missing/reordered')
    identity = load('validation').identity
    left, right = identity(reference), identity(candidate)
    for a, b in zip(left, right):
        (communication_check or compare_communication)(a.pop('communication'), b.pop('communication'))
    require(left == right, 'OP4 historical/candidate hardware/software/model/weights/RNG/input changed')
    gradient_differences = []
    compared = candidate
    if mode == 'smoke':
        # Compare and record gradient hashes, but exclude them from the
        # historical equality gate. Individual worker records are validated
        # by the OP3 replay before this cross-run comparison.
        compared = []
        for a, b in zip(reference, candidate):
            projected = dict(b)
            left_steps, right_steps = a['parameter_trajectory'], b['parameter_trajectory']
            require(len(left_steps) == len(right_steps), 'OP4 smoke trajectory length changed')
            projected_steps = []
            for index, (left_step, right_step) in enumerate(zip(left_steps, right_steps)):
                left_grad, right_grad = left_step['gradients'], right_step['gradients']
                names = sorted(name for name in left_grad.keys() | right_grad.keys()
                               if left_grad.get(name) != right_grad.get(name))
                if names:
                    gradient_differences.append(dict(rank=a['rank'], step=index,
                                                     count=len(names), tensors=names))
                projected_steps.append(dict(right_step, gradients=left_grad))
            projected['parameter_trajectory'] = projected_steps
            compared.append(projected)
    try:
        load('validation').equivalent(reference, compared, smoke=mode == 'smoke')
    except ValueError as exc:
        # Hash differences do not reveal their numerical magnitude. Report the
        # affected fields and tensors while preserving the exact equality gate.
        differences = []
        for a, b in zip(reference, candidate):
            for key in ('losses', 'grad_norms', 'final_parameters_sha256', 'trainable_parameters'):
                if (key != 'trainable_parameters' or mode == 'smoke') and a.get(key) != b.get(key):
                    differences.append(f"rank {a['rank']} {key}")
            if mode == 'smoke':
                left_steps, right_steps = a.get('parameter_trajectory'), b.get('parameter_trajectory')
                if not isinstance(left_steps, list) or not isinstance(right_steps, list) or len(left_steps) != len(right_steps):
                    differences.append(f"rank {a['rank']} parameter_trajectory structure")
                    continue
                for index, (left_step, right_step) in enumerate(zip(left_steps, right_steps)):
                    if left_step.get('step') != right_step.get('step'):
                        differences.append(f"rank {a['rank']} trajectory step {index} identity")
                    for part in ('parameters',):
                        left_values, right_values = left_step.get(part), right_step.get(part)
                        if not isinstance(left_values, dict) or not isinstance(right_values, dict):
                            differences.append(f"rank {a['rank']} step {index} {part} structure")
                            continue
                        names = sorted(name for name in left_values.keys() | right_values.keys()
                                       if left_values.get(name) != right_values.get(name))
                        if names:
                            differences.append(f"rank {a['rank']} step {index} {part}: {len(names)} tensors; first={names[0]}")
        detail = '; '.join(differences) if differences else 'unclassified trajectory difference'
        raise ValueError(f'{exc}; OP4 historical comparison ({mode}): {detail}') from exc
    for a, b in zip(reference, candidate):
        # Hashes cover all local parameters, rather than just sampled tensors.
        keys = ['initial_loss', 'warmup_last_loss', 'skips', 'cpu_affinity_policy', 'cpu_affinity',
                'seed', 'precision', 'target_modules', 'op3_implementation', 'op3_optimization_stack',
                'codec_identity']
        if mode == 'smoke':
            keys += ['smoke_output_sha256']
        require(all(a[key] == b[key] for key in keys),
                'OP4 changed numerical checks, CPU affinity or the OP3 mechanism stack')
    return dict(mode=mode, ranks=8, initial_identity_equal=True,
                numerical_trajectory_identical=not gradient_differences,
                historical_gradient_hash_equality_required=False,
                historical_gradient_hash_differences=gradient_differences,
                full_parameter_trajectory_checked=mode == 'smoke', projection_hashes_checked=mode == 'smoke')


def _bind(root, name, label, files, expected=None):
    path = root/name
    require(path.is_file(), 'Missing '+label+' evidence: '+name)
    digest = p.sha(path)
    require(expected is None or digest == expected, 'Changed '+label+' evidence SHA256: '+name)
    files[label+'/'+name] = digest
    return path


def _verify_files(root, manifest, names, label, files, prefix=''):
    require(isinstance(manifest, dict) and set(manifest) == set(names),
            'Missing or foreign '+label+' evidence manifest')
    for name in names:
        _bind(root, prefix+name, label, files, manifest[name])


def _operator_evidence(root, label, files, expected_report=None):
    import op3_operator as operator
    report_path = _bind(root, 'operator/report.json', label, files, expected_report)
    report = p.read(report_path)
    require(report.get('protocol') == operator.PROTOCOL and report.get('completed') is True
            and report.get('arm') == p.COMBINED and report.get('settings') == operator.SETTINGS
            and report.get('operator_gradients_validated') is True
            and report.get('source_order_preserved') is True
            and report.get('timed_arms') == [p.COMBINED]
            and report.get('full_step_measured') is False and set(report.get('roles', {})) == {'attn', 'mlp'},
            'Foreign or incomplete '+label+' combined operator report')
    names = {f'rank{rank}.json' for rank in range(8)} | {
        f'{role}-rank{rank}-profile.json.gz' for rank in range(8) for role in ('attn', 'mlp')}
    _verify_files(root, report.get('evidence_files'), names, label, files, 'operator/')
    diagnostics, rows = [], []
    for rank in range(8):
        row = p.read(root/f'operator/rank{rank}.json')
        require(row.get('rank') == rank and row.get('completed') is True
                and row.get('protocol') == operator.PROTOCOL and row.get('arm') == p.COMBINED,
                'Foreign or incomplete '+label+' operator rank')
        rows.append(row)
        for role in ('attn', 'mlp'):
            name = f'operator/{role}-rank{rank}-profile.json.gz'
            with gzip.open(root/name, 'rt') as handle:
                events = json.load(handle)['traceEvents']
            kernels = [e for e in events if e.get('cat') == 'kernel'
                       and 'nccl' in e.get('name', '').lower()]
            require(len(kernels) == operator.SETTINGS['profile_calls']
                    and all('SendRecv' in e['name'] and e['args'].get('device') == rank for e in kernels),
                    'Operator profile must contain only the original ten SendRecv exchanges')
            diagnostics.append(dict(rank=rank, role=role, sendrecv_calls=len(kernels),
                launches=_histogram(kernels), diagnostic_only=True, allgather_grid_checked=False))
    return report, diagnostics, rows


def audit_candidate(child_root, reference_root, *, communication_check=None,
                    reference_channels=4, intervention=None):
    """Validate a new candidate; never execute or mutate the historical batch."""
    child, reference = Path(child_root), Path(reference_root)
    files = {}
    plans = []
    for label, root in (('reference', reference), ('candidate', child)):
        plan = p.read(_bind(root, 'plan.json', label, files))
        require(plan.get('protocol') == p.PROTOCOL and plan.get('arms') == [p.COMBINED]
                and plan.get('trials') == plan.get('blocks') == 1,
                'OP4 requires one-window combined OP3 candidate and historical reference')
        plans.append(plan)
    reference_plan, plan = plans
    operator_only = bool(plan.get('operator_only'))
    (communication_check or compare_communication)(reference_plan['communication'], plan['communication'])
    for key in ('model', 'environment', 'cuda_visible_devices'):
        require(reference_plan[key] == plan[key], 'OP4 historical/candidate plan changed: '+key)

    reference_report = None
    if not operator_only:
        require(not reference_plan.get('operator_only'), 'Historical reference has no full-step evidence')
        reference_report = p.read(_bind(reference, 'report.json', 'reference', files))
        require(reference_report.get('protocol') == p.PROTOCOL and reference_report.get('completed') is True
                and reference_report.get('arms') == [p.COMBINED]
                and reference_report.get('trials') == reference_report.get('blocks') == 1
                and reference_report.get('correctness_passed') is True
                and reference_report.get('execution_path_validated') is True,
                'Foreign or incomplete historical combined full-step report')
        _bind(reference, 'plan.json', 'reference', files, reference_report.get('plan_sha256'))
        require(reference_report.get('plan_sha256') == files['reference/plan.json'],
                'Historical report is not bound to its plan')
    historical_operator, reference_operator_profiles, reference_operator_rows = _operator_evidence(
        reference, 'reference', files,
        reference_report.get('operator_report_sha256') if reference_report else None)
    if reference_report is not None:
        require(reference_report.get('operator_report_sha256') == files['reference/operator/report.json'],
                'Historical full-step report is not bound to its operator report')
    candidate_operator, candidate_operator_profiles, candidate_operator_rows = _operator_evidence(child, 'candidate', files)
    require(historical_operator['settings'] == candidate_operator['settings'],
            'Historical/candidate operator measurement settings changed')
    for historical, current in zip(reference_operator_rows, candidate_operator_rows):
        identity_keys = ('rank', 'protocol', 'arm', 'gpu_uuid', 'torch', 'cuda', 'nccl',
                         'settings', 'entrypoint_sha256')
        require(all(historical[key] == current[key] for key in identity_keys),
                'Historical/candidate operator hardware, software or source identity changed')
        for role in ('attn', 'mlp'):
            require(historical['roles'][role]['shape'] == current['roles'][role]['shape']
                    and historical['roles'][role]['gradient_checks'] == current['roles'][role]['gradient_checks'],
                    'Historical/candidate operator output, gradients or diagnostic updates changed')
    result = dict(passed=True, sole_intervention=dict(CHANNEL_ENV if intervention is None else intervention), operator_only=operator_only,
        full_step_completed=not operator_only, historical_reference=True,
        concurrent_control_measured=False, historical_numerical_regression_checked=not operator_only,
        reference_binary_revalidated=False, candidate_correctness_replayed_by_controller=True,
        historical_build=reference_plan['build'], candidate_build=plan['build'],
        build_paths_equal=reference_plan['build']['root'] == plan['build']['root'],
        operator_historical_identity_and_numerical_regression_checked=True,
        operator_binary_records=[dict(rank=rank, **{
            label: dict(libraries=row['libraries'], codec_library=row['codec_library'])
            for label, row in (('reference', historical), ('candidate', current))})
            for rank, (historical, current) in enumerate(zip(reference_operator_rows, candidate_operator_rows))],
        other_collective_launch_geometry_may_change=True, reduction_changes_allowed=False,
        full_training_cross_arm_comparison=False, numerical_checks=[], profiles=[],
        historical_operator_roles=historical_operator['roles'],
        reference_operator_report_sha256=files['reference/operator/report.json'],
        operator_profiles=dict(reference=reference_operator_profiles, candidate=candidate_operator_profiles),
        evidence_files=files, profile_durations_are_formal_speedups=False,
        limitation='Only the candidate is newly executed. Historical data hashes and numerical records '
                   'are checked without rerunning its GPU work or revalidating its binaries. '
                   'Historical timing differences are not paired or concurrent performance gains.')
    if operator_only:
        return result

    names = {f"results/{job['directory']}/rank{rank}.json" for job in p.jobs(1, 1) for rank in range(8)}
    _verify_files(reference, reference_report.get('worker_records'), names, 'reference', files)
    profile_path = _bind(reference, 'profile-summary.json', 'reference', files,
                         reference_report.get('profile_summary_sha256'))
    require(reference_report.get('profile_summary_sha256') == files['reference/profile-summary.json'],
            'Historical report is not bound to its profile summary')
    historical_profile = p.read(profile_path)
    require(historical_profile.get('protocol') == p.PROTOCOL and historical_profile.get('completed') is True
            and historical_profile.get('execution_validated') is True,
            'Incomplete historical full-step profile summary')
    names = {f'results/profile-{p.COMBINED}/rank{rank}-profile.json.gz' for rank in range(8)}
    _verify_files(reference, historical_profile.get('evidence_files'), names, 'reference', files)
    for job in p.jobs(1, 1):
        records = []
        for label, root in (('reference', reference), ('candidate', child)):
            records.append([p.read(_bind(root, f"results/{job['directory']}/rank{rank}.json", label, files))
                            for rank in range(8)])
        check = compare_rows(*records, job['mode'], communication_check=communication_check)
        check.update(comparison='historical_numerical_regression', concurrent=False)
        result['numerical_checks'].append(check)
    for rank in range(8):
        record = {}
        for label, root, channels in (('reference', reference, reference_channels), ('candidate', child, 16)):
            name = f'results/profile-{p.COMBINED}/rank{rank}-profile.json.gz'
            with gzip.open(_bind(root, name, label, files), 'rt') as handle:
                record[label] = profile_evidence(json.load(handle)['traceEvents'], rank, channels)
        require(record['reference']['other_nccl_payloads'] == record['candidate']['other_nccl_payloads'],
                'Non-AllGather NCCL entry kernel, payload or call count changed')
        result['profiles'].append(record)
    historical_ms = reference_report['comparisons']['compare1']['arms'][p.COMBINED]['mean_ms']
    require(math.isfinite(historical_ms) and historical_ms > 0, 'Invalid historical full-step time')
    result.update(reference_report_sha256=files['reference/report.json'], historical_full_step_ms=historical_ms)
    return result

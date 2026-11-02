"""Strict, concurrent OP4/OP6 evidence after both child replays succeed.

The controller replays each child through op6_child before calling audit_pair.
This module binds the raw files and compares numerical trajectories and NCCL
work. Unlike the historical OP4 check, smoke gradient differences fail here.
"""
import gzip
import json
from pathlib import Path

from rtx5090_loader import load
import op3_protocol as p
from op4_evidence import profile_evidence

require, read, sha = p.require, p.read, p.sha
CHANNELS = {'NCCL_MIN_NCHANNELS': '16', 'NCCL_MAX_NCHANNELS': '16'}
FORBIDDEN = ('NCCL_SHM_USE_CUDA_MEMCPY', 'NCCL_SHM_MEMCPY_MODE',
             'NCCL_SHM_LOCALITY', 'NCCL_P2P_LL_THRESHOLD')


def compare_rows(control, candidate, mode):
    require(mode in ('smoke', 'profile', 'timing'), 'Unknown OP6 comparison mode')
    for rows in (control, candidate):
        require(len(rows) == 8 and [row['rank'] for row in rows] == list(range(8)),
                'OP6 comparison requires exactly eight ordered ranks')
    validation = load('validation')
    require(validation.identity(control) == validation.identity(candidate),
            'OP6 changed hardware/software/model/initial weights/RNG/input/communication')
    validation.equivalent(control, candidate, smoke=mode == 'smoke')
    keys = ['initial_loss', 'warmup_last_loss', 'skips', 'cpu_affinity_policy',
            'cpu_affinity', 'seed', 'precision', 'target_modules',
            'op3_implementation', 'op3_optimization_stack', 'codec_identity']
    if mode == 'smoke':
        keys.append('smoke_output_sha256')
    for left, right in zip(control, candidate):
        require(all(left[key] == right[key] for key in keys),
                'OP6 changed projection output, numerical settings, affinity or OP3 stack')
    return dict(mode=mode, ranks=8, initial_identity_equal=True,
                numerical_trajectory_identical=True,
                full_parameter_and_gradient_trajectory_checked=mode == 'smoke',
                projection_hashes_checked=mode == 'smoke', concurrent=True,
                smoke_gradient_hash_equality_required=True)


def audit_pair(op4_child, op6_child, recompute_profiles=True):
    roots = {'op4': Path(op4_child), 'op6': Path(op6_child)}
    files, plans, reports = {}, {}, {}

    def bind(arm, name, expected=None):
        path = roots[arm]/name
        require(path.is_file(), 'Missing OP6 pair evidence: '+str(path))
        value = sha(path)
        require(expected is None or value == expected,
                'Changed OP6 pair evidence: '+str(path))
        files[arm+'/'+name] = value
        return path

    expected_records = {f"results/{job['directory']}/rank{rank}.json"
                        for job in p.jobs(1, 1) for rank in range(8)}
    expected_traces = {f'results/profile-{p.COMBINED}/rank{rank}-profile.json.gz'
                       for rank in range(8)}
    for arm, root in roots.items():
        plan = plans[arm] = read(bind(arm, 'plan.json'))
        result = reports[arm] = read(bind(arm, 'report.json'))
        require(read(bind(arm, 'batch-state.json'))['status'] == 'completed',
                'OP6 paired child is incomplete')
        require(plan.get('op6_variant') == arm and plan['protocol'] == p.PROTOCOL
                and plan['arms'] == [p.COMBINED] and plan['trials'] == plan['blocks'] == 1
                and plan.get('operator_only') is False,
                'OP6 paired child has the wrong variant or workload')
        require(result['protocol'] == p.PROTOCOL and result['completed'] is True
                and result['correctness_passed'] is True
                and result['execution_path_validated'] is True
                and result['operator_gradients_validated'] is True
                and result['timing_windows'] == 1 and result['formal_optimizer_steps'] == 50,
                'OP6 requires completed full-step and operator checks')
        bind(arm, 'plan.json', result['plan_sha256'])
        require(set(result['worker_records']) == expected_records,
                'OP6 pair has incomplete or foreign rank records')
        for name, digest in result['worker_records'].items():
            bind(arm, name, digest)
        profile = read(bind(arm, 'profile-summary.json', result['profile_summary_sha256']))
        require(profile['completed'] is True and profile['execution_validated'] is True
                and set(profile['evidence_files']) == expected_traces,
                'OP6 pair lacks validated eight-rank profiles')
        for name, digest in profile['evidence_files'].items():
            bind(arm, name, digest)
        operator = read(bind(arm, 'operator/report.json', result['operator_report_sha256']))
        for name, digest in operator['evidence_files'].items():
            bind(arm, 'operator/'+name, digest)
        env = plan['communication']['environment']
        require(all(env.get(key) == value for key, value in CHANNELS.items())
                and not any(key in env for key in FORBIDDEN),
                'OP6 requires unchanged OP4 channels without OP5 settings')
    for key in ('build', 'model', 'environment', 'communication', 'cuda_visible_devices'):
        require(plans['op4'][key] == plans['op6'][key], 'OP6 paired plan differs: '+key)
    for compat in ('ma9_compat', 'ma11_compat'):
        # These small helpers are built separately by the original child runner.
        # Paths/logs differ; require identical source, compiler and recipe.
        for key in ('protocol', 'source_sha256', 'builder_sha256', 'nvcc_sha256',
                    'torch', 'architecture'):
            require(plans['op4'][compat][key] == plans['op6'][compat][key],
                    'OP6 paired compatibility build differs: '+compat+'/'+key)
    numerical = []
    for job in p.jobs(1, 1):
        rows = [[read(roots[arm]/f"results/{job['directory']}/rank{rank}.json")
                 for rank in range(8)] for arm in ('op4', 'op6')]
        numerical.append(compare_rows(*rows, job['mode']))
    profiles = []
    # Always derive these comparisons from raw traces. recompute_profiles controls
    # the caller's full child replay, never permission to trust a stored histogram.
    for rank in range(8):
        record = {}
        for arm in roots:
            path = roots[arm]/f'results/profile-{p.COMBINED}/rank{rank}-profile.json.gz'
            with gzip.open(path, 'rt') as stream:
                record[arm] = profile_evidence(json.load(stream)['traceEvents'], rank, 16)
        for key in ('allgather_launches', 'other_nccl_launches', 'other_nccl_payloads'):
            require(record['op4'][key] == record['op6'][key],
                    'OP6 changed actual NCCL calls/payload/launch geometry: '+key)
        profiles.append(record)
    return dict(passed=True, concurrent_control_measured=True,
                child_correctness_replayed_by_controller=True,
                smoke_gradient_hash_equality_required=True, numerical_checks=numerical,
                profiles=profiles, evidence_files=files, build=plans['op4']['build'],
                sole_intervention='local attention forward/backward CUDA Graph submission',
                nccl_configuration_and_launches_equal=True,
                profile_durations_are_formal_speedups=False)

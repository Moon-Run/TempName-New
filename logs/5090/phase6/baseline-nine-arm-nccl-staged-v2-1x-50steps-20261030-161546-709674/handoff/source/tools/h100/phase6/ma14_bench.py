#!/usr/bin/env python3
"""MA14 epoch publication barrier comparison; mechanism timing belongs to fullstep.

Reuse the registered MA9 controller in an isolated module namespace. These
bounded substitutions change experiment identity and composition only; MA9's
arithmetic and helper ABI stay fixed; the candidate changes only publication.
"""
from pathlib import Path
import hashlib


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError('MA14 registered controller anchor changed: ' + old[:100])
    return source.replace(old, new)


_HERE = Path(__file__).resolve().parent
DRIVER = _HERE / 'ma14_bench_driver.py'
if not DRIVER.is_file():
    DRIVER = _HERE / 'ma9_bench.py'
DRIVER_SHA256 = '1d7c47d335be58ecd9b4d21268697a9a6d287d562cc2d0e8293332f8c327e5f8'
_driver = DRIVER.read_text()
if hashlib.sha256(_driver.encode()).hexdigest() != DRIVER_SHA256:
    raise ValueError('MA14 requires the registered unchanged MA9 controller')
_driver = _driver[:_driver.index("\nif __name__=='__main__':")]
for old, new, count in (
        ('ma9_bench', 'ma14_bench', 8), ('ma9_build', 'ma14_build', 3),
        ('ma9_sources', 'ma14_sources', 7), ('ma9_manifest_sha256', 'ma14_manifest_sha256', 1),
        ('H100_MA9_', 'H100_MA14_', 1),
        ("'h100-ma9-f4-zcount-v1'", "'h100-ma14-epoch-reuse-v2'", 1),
        ("ARMS = ('F4', 'F4-zcount')", "ARMS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-epoch')", 1),
        ("MAIN_COMPARISONS = (('F4-zcount', 'F4'),)",
         "MAIN_COMPARISONS = (('F4-cache-zcount-csoftmax-epoch', 'F4-cache-zcount-csoftmax'),)", 1),
        ("if arm == 'F4-zcount':", "if arm in ARMS:", 1),
        ("zero_count_policy='fused_optimizer_zero_count', roles=ROLES",
         "zero_count_policy='fused_optimizer_zero_count', activation_policy='retain_owned_column_input_uninstrumented', roles=ROLES", 1)):
    _driver = _replace(_driver, old, new, count)
_driver = _replace(_driver,
    "LABELS = {'F4':'Unchanged CR3/F4',\n"
    "          'F4-zcount':'Identical native projection; optimizer zero-count fusion is evaluated only in fullstep'}",
    "LABELS = {'F4-cache-zcount-csoftmax':'Unchanged complete MA11 baseline',\n"
    "          'F4-cache-zcount-csoftmax-epoch':'MA11 plus monotonic epoch publication barrier'}")
# Exercise 32 complete ping-pong reuse cycles under rank skew, outside timing.
# The original driver reads its own frozen template; register each change here
# so replay freezes the same base bytes and deterministic transform.
_driver = _replace(_driver, "_source = TEMPLATE.read_text()", """_source = TEMPLATE.read_text()
_source = registered_replace(_source, 'for generation in range(6):', 'for generation in range(64):')
_source = registered_replace(_source, "len(value['retained'])==6", "len(value['retained'])==64")
_source = registered_replace(_source, 'expected=16+(3', 'expected=74+(3')
_source = registered_replace(_source, "for r in value['retained']})==6", "for r in value['retained']})==64")""")
_driver = _replace(_driver, "'CudaIpcBarrierAllKernel' in path[1]['name']",
                   "barrier_name(arm) in path[1]['name']")
_driver = _replace(_driver,
    "result['roles'][role]=dict(native_tuning=actual_tuning, shape=shape",
    "result['roles'][role]=dict(epoch_state=epoch_state(op,job['arm']), native_tuning=actual_tuning, shape=shape")
_driver = _replace(_driver,
    "        path=sorted((e for e in kernels if e.get('args',{}).get('correlation') in correlations),key=lambda e:e['ts'])",
    "        import ma14_profile\n"
    "        ma14_profile.validate_projection_work([e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and e.get('args',{}).get('correlation') in correlations], arm)\n"
    "        path=sorted((e for e in kernels if e.get('args',{}).get('correlation') in correlations),key=lambda e:e['ts'])")
_driver = _replace(_driver, "durations.append(dict(cpu_us=marker['dur'],decoder_us=path[2]['dur']",
    "durations.append(dict(**ma14_profile.submission_intervals(marker,path,runtime),cpu_us=marker['dur'],barrier_us=path[1]['dur'],decoder_us=path[2]['dur']")
_driver = _replace(_driver, "launch=launches[0],median_us=",
    "launch=launches[0],submission_attribution=ma14_profile.SUBMISSION_LIMITATION,"
    "sum_us={k:sum(v[k] for v in durations) for k in ma14_profile.SUBMISSION_FIELDS},median_us=")
exec(compile(_driver, str(DRIVER), 'exec'), globals())

_driver_freeze, _driver_verify = freeze_helpers, verify
_driver_native_tuning = native_tuning
_driver_report, _driver_profile = report, profile_summary
_driver_validate_rows = validate_rows
EXTRA_DEPENDENCIES = ('ma8_adapter.py', 'ma9_sources.py', 'ma11_sources.py',
                      'ma11_adapter.py', 'ma14_profile.py', 'ma11_profile.py', 'ma8_profile.py', 'ma9_profile.py')


def validate_experiment(plan):
    require(plan.get('activation_policy') == 'retain_owned_column_input_uninstrumented' and
            plan.get('zero_count_policy') == 'fused_optimizer_zero_count',
            'MA14 requires unchanged MA8-v2 activation caching and MA9 zero counting in both arms')


def freeze_helpers(scripts):
    _driver_freeze(scripts)
    shutil.copy2(DRIVER, scripts / 'ma14_bench_driver.py')
    for name in EXTRA_DEPENDENCIES:
        shutil.copy2(HERE / name, scripts / name)


def verify(root, offline=False):
    plan = _driver_verify(root, offline)
    for current, name in [(DRIVER, 'ma14_bench_driver.py'),
                          *[(HERE / name, name) for name in EXTRA_DEPENDENCIES]]:
        require(sha(current) == plan['source_files']['scripts/' + name],
                'Changed MA14 composition dependency: ' + name)
    return plan


def native_tuning(op, arm, plan):
    expected = _driver_native_tuning(op, arm, plan)
    for slot in op.slots:
        from ma11_sources import NATIVE_EXPORTS as ma11_exports
        original = slot.lib.ma11_softmax_protocol
        original.argtypes = []; original.restype = ctypes.c_int
        require(original() == 11 and all(hasattr(slot.lib, name) for name in ma11_exports),
                'Both arms require the complete original MA11 native implementation')
        if arm == ARMS[1]:
            protocol = slot.lib.taco_ma14_epoch_protocol
            protocol.argtypes = []; protocol.restype = ctypes.c_int
            require(protocol() == 1402, 'Wrong MA14-v2 native pair/publication helper')
            require(all(hasattr(slot.lib, name) for name in ma14_sources.NATIVE_EXPORTS),
                    'Missing MA14 epoch publication exports')
        else:
            require(not any(hasattr(slot.lib, name) for name in ma14_sources.NATIVE_EXPORTS),
                    'MA11 control unexpectedly contains the MA14 helper')
    return expected


def epoch_state(op, arm):
    require(arm in ARMS, 'Unknown MA14 arm')
    return op.ma14_epoch_state() if arm == ARMS[1] else []


def validate_epoch_state(states, submissions, arm):
    require(arm in ARMS, 'Unknown MA14 arm')
    if arm == ARMS[0]:
        require(states == [], 'MA11 control unexpectedly has epoch state')
        return
    require(len(states) == len(submissions) == 2, 'Missing two epoch slots')
    require(len({s['local_ready_ptr'] for s in states}) == 2, 'Epoch slots share counters')
    for state, submission in zip(states, submissions):
        require(state['protocol_version'] == 1402 and state['initializations'] == 1 and
                state['slot'] == submission['slot'] and state['failed'] is False and
                state['epoch'] == submission['successful_calls'] and
                0 < state['epoch'] < 2**63 and state['local_ready_ptr'] > 0,
                'Epoch initialization, generation or call count drift')


def validate_rows(root, plan, jobs, stage, reference=None):
    result = _driver_validate_rows(root, plan, jobs, stage, reference)
    for job in jobs:
        for rank in range(8):
            row = read(Path(root)/'results'/job['directory']/f'rank{rank}.json')
            for value in row['roles'].values():
                validate_epoch_state(value['epoch_state'], value['submission'], job['arm'])
    return result


def forward_consumer(op, arm, x, w, bias, residual):
    # Both arms exercise the same arithmetic and their registered publication protocol.
    return op.forward_consumer(x, w, bias, residual)


def barrier_name(arm):
    require(arm in ARMS, 'Unknown MA14 arm')
    return ('taco_ma14_epoch_barrier_kernel' if arm == ARMS[1]
            else 'CudaIpcBarrierAllKernel')


LIMITATION = ('Complete row projection and publication measurement. GPU profile spans are '
              'perturbed and cannot substitute for paired unprofiled optimizer-step wall time.')


def report(root, save=True, offline=False):
    result = _driver_report(root, save=False, offline=offline)
    result['activation_policy'] = verify(root, offline)['activation_policy']
    result['limitation'] = LIMITATION
    if save:
        save_unchanged(Path(root) / 'report.json', result)
        print(json.dumps(result['roles'], indent=2))
    return result


def profile_summary(root, offline=False):
    result = _driver_profile(root, offline)
    result['limitation'] = LIMITATION
    return result


if __name__ == '__main__':
    main()

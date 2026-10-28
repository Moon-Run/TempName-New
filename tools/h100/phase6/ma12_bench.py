#!/usr/bin/env python3
"""MA12 complete row-projection harness; end-to-end timing belongs to fullstep.

Reuse the registered MA9 controller in an isolated module namespace. These
bounded substitutions preserve MA9's native helper ABI and the original
projection arithmetic. The candidate changes native publication and waiting.
"""
from pathlib import Path
import hashlib


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError('MA12 registered controller anchor changed: ' + old[:100])
    return source.replace(old, new)


_HERE = Path(__file__).resolve().parent
DRIVER = _HERE / 'ma12_bench_driver.py'
if not DRIVER.is_file():
    DRIVER = _HERE / 'ma9_bench.py'
DRIVER_SHA256 = '1d7c47d335be58ecd9b4d21268697a9a6d287d562cc2d0e8293332f8c327e5f8'
_driver = DRIVER.read_text()
if hashlib.sha256(_driver.encode()).hexdigest() != DRIVER_SHA256:
    raise ValueError('MA12 requires the registered unchanged MA9 controller')
_driver = _driver[:_driver.index("\nif __name__=='__main__':")]
for old, new, count in (
        ('ma9_bench', 'ma12_bench', 8), ('ma9_build', 'ma12_build', 3),
        ('ma9_sources', 'ma12_sources', 7), ('ma9_manifest_sha256', 'ma12_manifest_sha256', 1),
        ('H100_MA9_', 'H100_MA12_', 1),
        ("'h100-ma9-f4-zcount-v1'", "'h100-ma12-tile-publication-v1'", 1),
        ("ARMS = ('F4', 'F4-zcount')", "ARMS = ('F4-cache-zcount', 'F4-cache-zcount-tilepub')", 1),
        ("MAIN_COMPARISONS = (('F4-zcount', 'F4'),)",
         "MAIN_COMPARISONS = (('F4-cache-zcount-tilepub', 'F4-cache-zcount'),)", 1),
        ("if arm == 'F4-zcount':", "if arm in ARMS:", 1),
        ("zero_count_policy='fused_optimizer_zero_count', roles=ROLES",
         "zero_count_policy='fused_optimizer_zero_count', activation_policy='retain_owned_column_input_uninstrumented', roles=ROLES", 1)):
    _driver = _replace(_driver, old, new, count)
_driver = _replace(_driver,
    "LABELS = {'F4':'Unchanged CR3/F4',\n"
    "          'F4-zcount':'Identical native projection; optimizer zero-count fusion is evaluated only in fullstep'}",
    "LABELS = {'F4-cache-zcount':'Unchanged MA89 activation cache and zero count',\n"
    "          'F4-cache-zcount-tilepub':'MA89 plus tile-scoped producer publication and receiver acquires'}")
exec(compile(_driver, str(DRIVER), 'exec'), globals())

_driver_freeze, _driver_verify = freeze_helpers, verify
_driver_native_tuning = native_tuning
_driver_report, _driver_profile = report, profile_summary
EXTRA_DEPENDENCIES = ('ma8_adapter.py', 'ma9_sources.py', 'ma12_profile.py', 'ma8_profile.py', 'ma9_profile.py')


def validate_experiment(plan):
    require(plan.get('activation_policy') == 'retain_owned_column_input_uninstrumented' and
            plan.get('zero_count_policy') == 'fused_optimizer_zero_count',
            'MA12 requires unchanged MA8-v2 activation caching and MA9 zero counting in both arms')


def freeze_helpers(scripts):
    _driver_freeze(scripts)
    shutil.copy2(DRIVER, scripts / 'ma12_bench_driver.py')
    for name in EXTRA_DEPENDENCIES:
        shutil.copy2(HERE / name, scripts / name)


def verify(root, offline=False):
    plan = _driver_verify(root, offline)
    for current, name in [(DRIVER, 'ma12_bench_driver.py'),
                          *[(HERE / name, name) for name in EXTRA_DEPENDENCIES]]:
        require(sha(current) == plan['source_files']['scripts/' + name],
                'Changed MA12 composition dependency: ' + name)
    return plan


def native_tuning(op, arm, plan):
    expected = _driver_native_tuning(op, arm, plan)
    for slot in op.slots:
        if arm == ARMS[1]:
            protocol = slot.lib.taco_ma12_protocol_version
            protocol.argtypes = []; protocol.restype = ctypes.c_int
            require(protocol() == 1, 'Wrong MA12 tile-scoped publication helper')
            require(all(hasattr(slot.lib, name) for name in ma12_sources.NATIVE_EXPORTS),
                    'Missing MA12 tile-scoped publication exports')
        else:
            require(not any(hasattr(slot.lib, name) for name in ma12_sources.NATIVE_EXPORTS),
                    'MA89 control unexpectedly contains the MA12 helper')
    return expected


def forward_consumer(op, arm, x, w, bias, residual):
    # Both arms retain the original Python API and exact ordered arithmetic.
    return op.forward_consumer(x, w, bias, residual)


LIMITATION = ('Complete row projection, tile publication, decoder, bias and residual API. '
              'Operator savings are not end-to-end training savings; the fullstep comparison '
              'measures the incremental benefit on the common MA89 baseline.')


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


def decoder_name(arm, role, tuning):
    family(arm); validate_tuning(tuning)
    require(role in ROLES, 'Unknown decoder role')
    return 'taco_ma12_decode_consumer_kernel' if arm == ARMS[1] else 'taco_ma4f_decode_consumer_kernel'


def trace_path(events, arm, role, calls=4, tuning=None):
    import ma12_profile
    return ma12_profile.trace_operator_path(events, arm, role, calls=calls, tuning=validate_tuning(tuning))


if __name__ == '__main__':
    main()

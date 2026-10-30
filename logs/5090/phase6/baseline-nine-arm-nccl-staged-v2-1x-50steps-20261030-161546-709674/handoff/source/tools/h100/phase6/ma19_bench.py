#!/usr/bin/env python3
"""MA19 row-projection sanity harness; mechanism timing belongs to fullstep.

Reuse the registered MA9 controller in an isolated module namespace. These
bounded substitutions change experiment identity and composition only; MA9's
native helper ABI and the original F4 row projection stay unchanged.
"""
from pathlib import Path
import hashlib


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError('MA19 registered controller anchor changed: ' + old[:100])
    return source.replace(old, new)


_HERE = Path(__file__).resolve().parent
DRIVER = _HERE / 'ma19_bench_driver.py'
if not DRIVER.is_file():
    DRIVER = _HERE / 'ma9_bench.py'
DRIVER_SHA256 = '1d7c47d335be58ecd9b4d21268697a9a6d287d562cc2d0e8293332f8c327e5f8'
_driver = DRIVER.read_text()
if hashlib.sha256(_driver.encode()).hexdigest() != DRIVER_SHA256:
    raise ValueError('MA19 requires the registered unchanged MA9 controller')
_driver = _driver[:_driver.index("\nif __name__=='__main__':")]
for old, new, count in (
        ('ma9_bench', 'ma19_bench', 8), ('ma9_build', 'ma19_build', 3),
        ('ma9_sources', 'ma19_sources', 7), ('ma9_manifest_sha256', 'ma19_manifest_sha256', 1),
        ('H100_MA9_', 'H100_MA19_', 1),
        ("'h100-ma9-f4-zcount-v1'", "'h100-ma19-parameter-publish-v1'", 1),
        ("ARMS = ('F4', 'F4-zcount')", "ARMS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-pcopy')", 1),
        ("MAIN_COMPARISONS = (('F4-zcount', 'F4'),)",
         "MAIN_COMPARISONS = (('F4-cache-zcount-csoftmax-pcopy', 'F4-cache-zcount-csoftmax'),)", 1),
        ("if arm == 'F4-zcount':", "if arm in ARMS:", 1),
        ("zero_count_policy='fused_optimizer_zero_count', roles=ROLES",
         "zero_count_policy='fused_optimizer_zero_count', activation_policy='retain_owned_column_input_original_backward', roles=ROLES", 1)):
    _driver = _replace(_driver, old, new, count)
_driver = _replace(_driver,
    "LABELS = {'F4':'Unchanged CR3/F4',\n"
    "          'F4-zcount':'Identical native projection; optimizer zero-count fusion is evaluated only in fullstep'}",
    "LABELS = {'F4-cache-zcount-csoftmax':'Unchanged complete MA11 baseline',\n"
    "          'F4-cache-zcount-csoftmax-pcopy':'MA11 plus fixed-descriptor master-to-model parameter publication, evaluated only in fullstep'}")
exec(compile(_driver, str(DRIVER), 'exec'), globals())

_driver_freeze, _driver_verify = freeze_helpers, verify
_driver_native_tuning = native_tuning
_driver_report, _driver_profile = report, profile_summary
EXTRA_DEPENDENCIES = ('ma8_adapter.py', 'ma9_sources.py', 'ma11_sources.py',
                      'ma11_adapter.py', 'ma19_adapter.py', 'ma19_profile.py', 'ma19_param_copy.cu',
                      'ma11_profile.py', 'ma8_profile.py', 'ma9_profile.py')


def validate_experiment(plan):
    require(plan.get('activation_policy') == 'retain_owned_column_input_original_backward' and
            plan.get('zero_count_policy') == 'fused_optimizer_zero_count',
            'MA19 requires unchanged MA8-v2 activation caching and MA9 zero counting in both arms')


def freeze_helpers(scripts):
    _driver_freeze(scripts)
    shutil.copy2(DRIVER, scripts / 'ma19_bench_driver.py')
    for name in EXTRA_DEPENDENCIES:
        shutil.copy2(HERE / name, scripts / name)


def verify(root, offline=False):
    plan = _driver_verify(root, offline)
    for current, name in [(DRIVER, 'ma19_bench_driver.py'),
                          *[(HERE / name, name) for name in EXTRA_DEPENDENCIES]]:
        require(sha(current) == plan['source_files']['scripts/' + name],
                'Changed MA19 composition dependency: ' + name)
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
            native = slot.lib.ma19_copy_protocol
            native.argtypes = []; native.restype = ctypes.c_int
            require(native() == 19 and all(hasattr(slot.lib, name) for name in ma19_sources.NATIVE_EXPORTS),
                    'MA19 candidate parameter-publication native implementation is missing')
            chunk_size = slot.lib.ma19_copy_chunk_size
            chunk_size.argtypes = []; chunk_size.restype = ctypes.c_int
            require(chunk_size() == 32768, 'MA19 candidate chunk size changed')
        else:
            require(not any(hasattr(slot.lib, name) for name in ma19_sources.NATIVE_EXPORTS),
                    'MA11 control unexpectedly exports MA19 parameter-publication operations')
    return expected


def forward_consumer(op, arm, x, w, bias, residual):
    # This benchmark deliberately exercises only the unchanged row projection.
    return op.forward_consumer(x, w, bias, residual)


LIMITATION = ('Identical row-projection sanity measurement only. It exercises neither column '
              'activation retention nor optimizer zero-count fusion; complete training measures '
              'the fixed-descriptor master-to-model parameter publication increment on the common MA11 baseline.')


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

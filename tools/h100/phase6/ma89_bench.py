#!/usr/bin/env python3
"""MA89 row-projection sanity harness; mechanism timing belongs to fullstep.

Reuse the registered MA9 controller in an isolated module namespace. These
bounded substitutions change experiment identity and composition only; MA9's
native helper ABI and the original F4 row projection stay unchanged.
"""
from pathlib import Path
import hashlib


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError('MA89 registered controller anchor changed: ' + old[:100])
    return source.replace(old, new)


_HERE = Path(__file__).resolve().parent
DRIVER = _HERE / 'ma89_bench_driver.py'
if not DRIVER.is_file():
    DRIVER = _HERE / 'ma9_bench.py'
DRIVER_SHA256 = '1d7c47d335be58ecd9b4d21268697a9a6d287d562cc2d0e8293332f8c327e5f8'
_driver = DRIVER.read_text()
if hashlib.sha256(_driver.encode()).hexdigest() != DRIVER_SHA256:
    raise ValueError('MA89 requires the registered unchanged MA9 controller')
_driver = _driver[:_driver.index("\nif __name__=='__main__':")]
for old, new, count in (
        ('ma9_bench', 'ma89_bench', 8), ('ma9_build', 'ma89_build', 3),
        ('ma9_sources', 'ma89_sources', 7), ('ma9_manifest_sha256', 'ma89_manifest_sha256', 1),
        ('H100_MA9_', 'H100_MA89_', 1),
        ("'h100-ma9-f4-zcount-v1'", "'h100-ma89-cache-zcount-v1'", 1),
        ("ARMS = ('F4', 'F4-zcount')", "ARMS = ('F4-cache', 'F4-cache-zcount')", 1),
        ("MAIN_COMPARISONS = (('F4-zcount', 'F4'),)",
         "MAIN_COMPARISONS = (('F4-cache-zcount', 'F4-cache'),)", 1),
        ("if arm == 'F4-zcount':", "if arm == 'F4-cache-zcount':", 1),
        ("zero_count_policy='fused_optimizer_zero_count', roles=ROLES",
         "zero_count_policy='fused_optimizer_zero_count', activation_policy='retain_owned_column_input_uninstrumented', roles=ROLES", 1)):
    _driver = _replace(_driver, old, new, count)
_driver = _replace(_driver,
    "LABELS = {'F4':'Unchanged CR3/F4',\n"
    "          'F4-zcount':'Identical native projection; optimizer zero-count fusion is evaluated only in fullstep'}",
    "LABELS = {'F4-cache':'Original MA8-v2 cache with original optimizer zero count',\n"
    "          'F4-cache-zcount':'Same MA8-v2 cache with original MA9 fused zero count'}")
exec(compile(_driver, str(DRIVER), 'exec'), globals())

_driver_freeze, _driver_verify = freeze_helpers, verify
_driver_report, _driver_profile = report, profile_summary
EXTRA_DEPENDENCIES = ('ma8_adapter.py', 'ma9_sources.py')


def validate_experiment(plan):
    require(plan.get('activation_policy') == 'retain_owned_column_input_uninstrumented' and
            plan.get('zero_count_policy') == 'fused_optimizer_zero_count',
            'MA89 requires the unchanged MA8-v2 cache in both arms and MA9 candidate zero count')


def freeze_helpers(scripts):
    _driver_freeze(scripts)
    shutil.copy2(DRIVER, scripts / 'ma89_bench_driver.py')
    for name in EXTRA_DEPENDENCIES:
        shutil.copy2(HERE / name, scripts / name)


def verify(root, offline=False):
    plan = _driver_verify(root, offline)
    for current, name in [(DRIVER, 'ma89_bench_driver.py'),
                          *[(HERE / name, name) for name in EXTRA_DEPENDENCIES]]:
        require(sha(current) == plan['source_files']['scripts/' + name],
                'Changed MA89 composition dependency: ' + name)
    return plan


def forward_consumer(op, arm, x, w, bias, residual):
    # This benchmark deliberately exercises only the unchanged row projection.
    return op.forward_consumer(x, w, bias, residual)


LIMITATION = ('Identical row-projection sanity measurement only. It exercises neither column '
              'activation retention nor optimizer zero-count fusion; complete training measures '
              'the MA9 increment on the common MA8-v2 baseline.')


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

"""Shared build parallelism and narrow compatibility for the default change."""
import os
from pathlib import Path
import subprocess


def default_build_jobs():
    """Match ``--jobs \"$(nproc)\"``; stay usable when coreutils is absent."""
    try:
        value = int(subprocess.check_output(['nproc'], text=True, stderr=subprocess.DEVNULL).strip())
        if value > 0:
            return value
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        value = len(os.sched_getaffinity(0))
        if value > 0:
            return value
    except (AttributeError, OSError):
        pass
    return max(1, os.cpu_count() or 1)


def resolve_build_jobs(jobs):
    if jobs is None:
        return default_build_jobs()
    if type(jobs) is not int or jobs <= 0:
        raise ValueError('Build jobs must be a positive integer')
    return jobs


# Exact before/after recipe identities for the resource-default-only change.
# Any later recipe edit invalidates this compatibility entry. It never admits
# changes to other input fields, registered source files, binaries or ABI.
DEFAULT_JOBS_RECIPE_COMPAT = {'baseline_build.py': ('04825ceee88a00c2f2bedb2e268971e8fad6466d9713d6e91eebb16a1f21bbbd',
                       '4db3eefbbf870bf79b0cd41a6576a99bbc0bee598ce307eedd613bfca3d8f3ba'),
 'flux_baseline_build.py': ('e1bc2d3a58daf344cc13291d8d4f62806b482cc6eb13f54402830c4cc8073a42',
                            'fadeab57cdc6cf679fa151aed8df66e326415b03972847fdcd84cdee93922c7d'),
 'op1_a_build.py': ('f58ca54f094d53403fa7bdb1cd91897ecb1b0ae267f5f72be6cc41726a53d48c',
                    '9ca08894fcd28203b075d1ddf0dfa623fbd522cde3202fd915606584e508543a'),
 'op1_abc_build.py': ('e049cc1dc0494905bd9b61f4c78a3b80f6f06383c5334e3eda4493cdde6f5249',
                      '909660ab1f0e070f2d65db6dd27bb2d2fefd075cdfdb82c73dadb94c548dc14d'),
 'op3_build.py': ('dc7e7124bdb07e97ad65fb400263e033ce707a48611deaf746211c2c69416366',
                  '8b52844dc8f0240b1988c5111690515d7cb2b9d8ab9cc7f3a19db24a1931e567')}

# Exact identities for the relocation-only change. Other provenance checks are
# unchanged, and a later recipe edit must not inherit this exception.
RELOCATION_RECIPE_COMPAT = {
    'baseline_build.py': (DEFAULT_JOBS_RECIPE_COMPAT['baseline_build.py'],
        '8db614035bf50b7521b5ce35536661d7c69ce88b33394415056c91c954ebc238'),
    'flux_baseline_build.py': (DEFAULT_JOBS_RECIPE_COMPAT['flux_baseline_build.py'],
        'b1e6dc96a80c9f0a1276c4f12ab97169f14bb60c41bb2729443f7ec9f7d2b40a'),
}


def recipe_matches(path, saved, current):
    prior, relocated = RELOCATION_RECIPE_COMPAT.get(Path(path).name, ((), None))
    return (saved == current or
            DEFAULT_JOBS_RECIPE_COMPAT.get(Path(path).name) == (saved, current) or
            (current == relocated and saved in prior))


def build_inputs_match(path, saved, current):
    if not isinstance(saved, dict) or set(saved) != set(current):
        return False
    return (recipe_matches(path, saved['recipe_sha256'], current['recipe_sha256'])
            and all(saved[key] == value for key, value in current.items() if key != 'recipe_sha256'))

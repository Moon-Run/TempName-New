"""Check defined dynamic function exports without loading CUDA or querying GPUs."""
from pathlib import Path
import subprocess

from common import require


def verify_native_exports(library, required=(), forbidden=()):
    library = Path(library)
    result = subprocess.run(
        ['nm', '-D', '--defined-only', '--format=posix', str(library)],
        capture_output=True, text=True, check=False)
    require(result.returncode == 0,
            'Cannot inspect native exports: ' + str(library) + '\n' + result.stderr.strip())
    entries = [line.split() for line in result.stdout.splitlines()]
    defined = {entry[0] for entry in entries if len(entry) >= 2}
    functions = {entry[0] for entry in entries if len(entry) >= 2 and entry[1] in ('T', 'W')}
    missing = sorted(set(required) - functions)
    unexpected = sorted(set(forbidden) & defined)
    require(not missing, 'Missing exported native functions in ' + str(library) + ': ' + ', '.join(missing))
    require(not unexpected, 'Unexpected candidate exports in F4 control ' + str(library) + ': ' + ', '.join(unexpected))

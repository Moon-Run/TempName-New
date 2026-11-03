"""Export model profiles on the output filesystem and publish complete gzip files.

Called only after profiler capture has stopped, outside formal timing. Neither
the system /tmp directory nor TMPDIR is used for the large uncompressed trace.
"""
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile

PROTOCOL = 'op6-output-filesystem-atomic-profile-v1'


def storage_status(directory):
    try:
        info = os.statvfs(directory)
        return f'available_bytes={info.f_bavail*info.f_frsize}, available_inodes={info.f_favail}'
    except OSError as exc:
        return f'filesystem_status_unavailable={exc}'


def validate_trace(path):
    if path.stat().st_size == 0:
        raise RuntimeError('profiler did not produce a nonempty trace.json')
    # Kineto can log a write error without raising a Python exception. Reject
    # truncated JSON as well as an empty/missing export before publishing it.
    with path.open('r', encoding='utf-8') as stream:
        trace = json.load(stream)
    if not isinstance(trace, dict) or not isinstance(trace.get('traceEvents'), list) or not trace['traceEvents']:
        raise RuntimeError('profiler output lacks a nonempty traceEvents array')


def export_profile(profiler, directory, rank):
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError('OP6 profile export requires rank 0..7')
    directory = Path(directory).resolve()
    destination = directory/f'rank{rank}-profile.json.gz'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite an existing profile: {destination}')
    try:
        with tempfile.TemporaryDirectory(prefix=f'.op6-profile-rank{rank}-', dir=directory) as temporary:
            temporary = Path(temporary)
            plain = temporary/'trace.json'
            pending = temporary/'profile.json.gz.part'
            # Probe file creation in Python so ENOSPC/EACCES/EDQUOT is retained
            # when possible, instead of only Kineto's generic C++ log message.
            with plain.open('xb'):
                pass
            profiler.export_chrome_trace(str(plain))
            validate_trace(plain)
            with plain.open('rb') as source, gzip.open(pending, 'wb') as target:
                shutil.copyfileobj(source, target, length=1024*1024)
            # Closing gzip above must succeed before the final filename appears.
            # Both paths are on the output filesystem, so publication is atomic.
            os.replace(pending, destination)
        return destination
    except Exception as exc:
        raise RuntimeError(f'OP6 rank{rank} profile export failed on output filesystem '
                           f'{directory}: {exc}; {storage_status(directory)}') from exc

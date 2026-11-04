"""Keep compiler/runtime temporary files on the OP6 output filesystem.

This scope starts before any hardware probe or build (including nvcc identity
queries), and is inherited by all worker processes. It does not change build
recipes, model code, profiling windows or timing. Only its own private directory
is removed; the system /tmp and other experiments are never cleaned here.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from op6_profile_io import storage_status

VARIABLES = ('TMPDIR', 'TMP', 'TEMP')
# Python multiprocessing may append /pymp-xxxxxxxx/listener-xxxxxxxx and bind
# an AF_UNIX socket. A full experiment/batch path can exceed sun_path's limit.
MAX_DIRECTORY_BYTES = 64


def temporary_base(root):
    """Choose the nearest writable, short ancestor on the output filesystem."""
    device = root.stat().st_dev
    for ancestor in (root, *root.parents):
        if ancestor.stat().st_dev != device:
            break
        base = ancestor / f'.op6-tmp-{os.getuid()}'
        if len(os.fsencode(base / 'run-xxxxxxxx')) > MAX_DIRECTORY_BYTES:
            continue
        try:
            base.mkdir(mode=0o700, exist_ok=True)
        except PermissionError:
            continue
        # Never follow an existing link onto the full root disk, or share a
        # private namespace owned by a different user.
        if base.is_symlink() or base.stat().st_uid != os.getuid() or base.stat().st_dev != device:
            raise RuntimeError(f'Unsafe OP6 temporary base: {base}')
        return base
    raise RuntimeError(f'No writable short temporary directory on output filesystem {root}; '
                       'use a shorter --out path on that filesystem')


def probe(directory):
    path = directory / 'write-probe'
    with path.open('xb') as handle:
        handle.write(b'\0' * 65536)
        handle.flush()
        os.fsync(handle.fileno())
    path.unlink()


@contextmanager
def session(root):
    root = Path(root).resolve(strict=True)
    previous = {key: os.environ.get(key) for key in VARIABLES}
    previous_tempdir = tempfile.tempdir
    directory = None
    try:
        try:
            base = temporary_base(root)
            directory = Path(tempfile.mkdtemp(prefix='run-', dir=base))
            probe(directory)
            os.environ.update({key: str(directory) for key in VARIABLES})
            # tempfile may already have cached /tmp before this context starts.
            tempfile.tempdir = str(directory)
            record = dict(protocol='op6-output-filesystem-temporary-v1', pid=os.getpid(),
                          output=str(root), directory=str(directory),
                          filesystem_device=root.stat().st_dev,
                          environment={key: str(directory) for key in VARIABLES},
                          storage_status=storage_status(root))
            (root / 'handoff').mkdir(exist_ok=True)
            (root / 'handoff/temporary-storage.json').write_text(json.dumps(record, indent=2) + '\n')
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f'OP6 temporary storage initialization failed on {root}: {exc}; '
                               f'{storage_status(root)}; no fallback to system /tmp') from exc
        print(f'OP6 TMPDIR={directory} ({storage_status(root)})', flush=True)
        yield directory
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = previous_tempdir
        if directory is not None:
            try:
                shutil.rmtree(directory)
            except OSError as exc:
                # Preserve the actual training/build failure if cleanup also
                # fails. Identify exactly the owned directory left behind.
                print(f'OP6 temporary cleanup failed for {directory}: {exc}', file=sys.stderr)

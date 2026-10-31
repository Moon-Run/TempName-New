#!/usr/bin/env python3
"""Upgrade registered staging Python frontends in an already compiled build.

The source and destination receive full build verification. CUDA sources and binaries
are unchanged; original frontend files/manifests are retained under repairs/.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import uuid

sys.dont_write_bytecode = True
from rtx5090_loader import load

builder = load('build')
LEGACY_SHA256 = '61bb8c847e41774648629cc1cffcb35729601bd6ad64e6fa9f73a8d6c0ebab22'
FIXED_RANK_SHA256 = 'ba668cc182904040bce8592d5d98840e0a286dde9dbcec4742874ddbd65eca8c'
LEGACY_FRONTEND_SHA256 = frozenset((LEGACY_SHA256, FIXED_RANK_SHA256))
RANK_INITIALIZATION = '        self.rank = rank\n'
FRONTEND = 'python/flux/gemm_rs_taco.py'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False)+'\n').encode()


def atomic_write(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def legacy_source(identity=LEGACY_SHA256):
    builder.require(identity in LEGACY_FRONTEND_SHA256, 'Unregistered legacy staging frontend')
    current = (builder.HERE/'legacy_nccl_transport_v2.txt').read_text()
    builder.require(digest(current.encode()) == FIXED_RANK_SHA256,
                    'Frozen staging frontend fixture changed')
    builder.require(current.count(RANK_INITIALIZATION) == 1, 'Expected exactly the native rank initialization fix')
    original = current.replace(RANK_INITIALIZATION, '', 1) if identity == LEGACY_SHA256 else current
    builder.require(digest(original.encode()) == identity,
                    'This repair accepts only the registered staging frontend revisions')
    return original


def legacy_verifier(directory, original):
    """Run the unmodified verifier with the exact original frontend generator."""
    for name in (*builder.TRANSPORT_INPUTS, 'gemm_only_generator.cc'):
        if name == 'nccl_transport.py':
            (directory/name).write_text(original)
        else:
            (directory/name).symlink_to(builder.HERE/name)

    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value

    verifier = module('rtx5090_rank_repair_legacy_verifier', builder.__file__)
    verifier.HERE = directory
    verifier.nccl_transport = module('rtx5090_rank_repair_legacy_transport', directory/'nccl_transport.py')
    return verifier


def repair(root, *, require_built=True):
    # False is used only for CPU source-tree tests; status is never promoted.
    root = Path(root).resolve()
    transport_sha = builder.sha(builder.HERE/'nccl_transport.py')
    record = builder.read(root/'baseline-build.json')
    previous_sha = record['transport_inputs']['nccl_transport.py']
    if previous_sha == transport_sha:
        builder.verify(root, built=require_built)
        return dict(status='already_repaired', build=str(root))
    builder.require(previous_sha in LEGACY_FRONTEND_SHA256,
                    'Build does not contain a registered staging frontend')
    original = legacy_source(previous_sha)
    with tempfile.TemporaryDirectory(prefix='rtx5090-rank-verifier-') as tmp:
        legacy_verifier(Path(tmp), original).verify(root, built=require_built)

    selective = builder.read(root/'selective/manifest.json')
    updates = {}
    for arm in builder.NATIVE_ARMS:
        name = f'selective/{arm}/{FRONTEND}'
        data = builder.nccl_transport.frontend_source(arm).encode()
        updates[name] = data
        selective['variants'][arm]['source_sha256'][FRONTEND] = digest(data)
        if f'{arm}/{FRONTEND}' in selective['files']:
            selective['files'][f'{arm}/{FRONTEND}'] = digest(data)
    updates['selective/manifest.json'] = encoded(selective)
    record['transport_inputs']['nccl_transport.py'] = transport_sha
    for name, data in updates.items():
        if name in record.get('files', {}):
            record['files'][name] = digest(data)
    updates['baseline-build.json'] = encoded(record)
    originals = {name: (root/name).read_bytes() for name in updates}
    backup = root/'repairs'/('native-rank-'+uuid.uuid4().hex)
    backup.mkdir(parents=True)
    for name, data in originals.items():
        path = backup/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    receipt = dict(status='repairing', build=str(root), backup=str(backup),
                   previous_frontend_sha256=previous_sha, frontend_sha256=transport_sha,
                   old_sha256={name: digest(data) for name, data in originals.items()},
                   new_sha256={name: digest(data) for name, data in updates.items()},
                   native_binaries_verified=False, gpu_validated=False)
    atomic_write(backup/'repair.json', encoded(receipt))
    try:
        for name, data in updates.items():
            atomic_write(root/name, data)
        builder.verify(root, built=require_built)
        receipt.update(status='repaired', native_binaries_verified=require_built)
        atomic_write(backup/'repair.json', encoded(receipt))
    except BaseException as exc:
        for name, data in originals.items():
            atomic_write(root/name, data)
        receipt.update(status='rolled_back', error=f'{type(exc).__name__}: {exc}')
        atomic_write(backup/'repair.json', encoded(receipt))
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args(argv)
    lock_path = builder.REPO/'logs/5090/.baseline.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('A 5090 campaign is running; repair only after it exits') from None
        result = repair(args.build)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print('FRONTEND REPAIR PASSED', args.build.resolve())


if __name__ == '__main__':
    main()

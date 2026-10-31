#!/usr/bin/env python3
"""Compile and test the RTX 5090 MA11 softmax without launching the model."""
import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
from rtx5090_loader import REPO, load

compat = load('ma11_compat')


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--prepared-root', type=Path,
                        help='Verify and test an already compiled MA11 compatibility library')
    args = parser.parse_args(argv)
    if args.out is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        args.out = REPO.parent/'Load-5090/logs/5090/phase6'/('ma11-softmax-probe-'+stamp)
    args.out, args.cuda_home = args.out.expanduser().resolve(), args.cuda_home.expanduser().resolve()
    if args.prepared_root is not None:
        args.prepared_root = args.prepared_root.expanduser().resolve()
    if args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    return args


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out/'probe-result.json'
    print('Probe:', args.out, flush=True)
    report = dict(protocol=compat.PROTOCOL, completed=False, performance_measured=False, gpu=args.gpu)
    try:
        import torch
        compat.require(str(torch.__version__) == compat.TORCH, 'Expected the frozen NGC 25.06 PyTorch')
        compat.require(torch.cuda.is_available() and args.gpu < torch.cuda.device_count(), 'Selected GPU unavailable')
        torch.cuda.set_device(args.gpu)
        device = torch.cuda.get_device_properties(args.gpu)
        compat.require('5090' in device.name and (device.major, device.minor) == (12, 0)
                       and device.multi_processor_count == 170, 'Expected full RTX 5090 SM120')
        report.update(torch=str(torch.__version__), cuda=torch.version.cuda, name=device.name,
                      capability=[device.major, device.minor], sm_count=device.multi_processor_count,
                      uuid=str(device.uuid))
        if args.prepared_root is None:
            record = compat.prepare(args.out, args.cuda_home, str(torch.__version__))
        else:
            record = json.loads((args.prepared_root/'ma11-compat/manifest.json').read_text())
            compat.verify(record, args.prepared_root, str(torch.__version__))
        report['library'] = record
        sources = load('sources')
        adapter = ModuleType('rtx5090_probe_ma11_adapter')
        exec(compile(sources.softmax_adapter_source({'torch': str(torch.__version__)}),
                     'rtx5090-probe-ma11-adapter.py', 'exec'), adapter.__dict__)
        native = adapter.NativeSoftmax(torch, ctypes.CDLL(record['library']))
        result = adapter.native_self_test(torch, native)
        torch.cuda.synchronize()
        report.update(completed=True, self_test=result)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        raise
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print('MA11 SOFTMAX PROBE PASSED', path, flush=True)


def main(argv=None):
    args = arguments(argv)
    if not args.execute:
        print(json.dumps(dict(preview=True, protocol=compat.PROTOCOL, out=str(args.out), gpu=args.gpu,
                              cuda_home=str(args.cuda_home), torch=compat.TORCH,
                              performance_measured=False), indent=2))
        return
    run(args)


if __name__ == '__main__':
    main()

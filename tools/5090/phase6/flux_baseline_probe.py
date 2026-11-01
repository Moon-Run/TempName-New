#!/usr/bin/env python3
"""Retain old A/C packet, reference and ownership probes with all masks removed."""
from pathlib import Path
from types import ModuleType
from rtx5090_loader import HERE
from flux_baseline_native import replace


def probe_module():
    source = (HERE/'local_smoke.py').read_text()
    source = replace(source, "PROTOCOL = 'rtx5090-local-transport-smoke-v1'",
                     "PROTOCOL = 'rtx5090-flux-four-native-smoke-v1'")
    source = replace(source, "ARMS = ('ma11', 'ma89', 'ma8', 'ma9', 'cr3', 'r', 'c', 'a')", "ARMS = ('c', 'a')")
    source = replace(source, 'FUSED = ARMS[:5]', 'FUSED = ()')
    old = 'constructor = frontend.GemmRSTacoDoubleBuffered if selected is not None else frontend.GemmRSTaco'
    if source.count(old) != 2:
        raise ValueError('Old shared/role probe constructors changed')
    source = source.replace(old, "constructor = frontend.GemmRSTacoDoubleBuffered if arm == 'c' else frontend.GemmRSTaco")
    source = replace(source, "if key == 'mask' and count == 1:", "if key == 'mask':")
    source = replace(source, "        builder = load('build')", '        import flux_baseline_build as builder')
    source = source.replace("args.build/'baseline-build.json'", "args.build/'flux-build.json'")
    source = replace(source, "        selection = read(args.build/'selective/selection-plan.json')\n"
                     "        policy = 'interleaved_arrival_selective_attn_mlp'\n", '')
    source = replace(source, "                selected = None if args.arm in ('a', 'r') else selection['variants'][policy]['roles'][role]['entry']['mask']",
                     '                selected = None  # Both old native variants quantize every remote tile.')
    module = ModuleType('rtx5090_flux_four_native_probe')
    module.__file__ = str(Path(__file__).resolve())
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


probe = probe_module()
validate_reports = probe.validate_reports


if __name__ == '__main__':
    probe.main()

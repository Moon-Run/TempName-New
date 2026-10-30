"""GPU preflight of baseline identity, D8, H8, mixed packets and ordered BF16 fold."""
import math
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import basic_quant_validation as baseline

CASES = ('zeros', 'small', 'outliers', 'tiny', 'mixed', 'rounding_midpoint', 'full_shape')
SELECTOR = dict(relative_l2=.025, max_abs_over_amax=.0625)
# User-approved D8-to-reference tolerance. Packet decoding still uses 0.5%,
# and reconstruction against the BF16 input still uses 5%.
D8_REFERENCE_L2_LIMIT = .008


def validate_nonfinite(codec, policy='mav2'):
    """Device assertions poison a CUDA context: probe in short isolated processes."""
    import torch
    program = r'''
import importlib.util, json, sys, torch
path, identity, device, value, policy = sys.argv[1:]
spec = importlib.util.spec_from_file_location('mav2_assert_probe', path)
codec = importlib.util.module_from_spec(spec); spec.loader.exec_module(codec)
torch.cuda.set_device(int(device)); codec.load_library(json.loads(identity))
x = torch.ones((1,128), device='cuda', dtype=torch.bfloat16)
x[0,0] = float(value)
try:
    codec.pack_rows(x, 1, policy=policy)
    torch.cuda.synchronize()
except RuntimeError as exc:
    if 'device-side assert' in str(exc):
        print('MAV2_ASSERT_CONFIRMED', flush=True)
        sys.exit(0)
    raise
raise RuntimeError('Nonfinite input was silently accepted')
'''
    checks = []
    for value in ('nan', 'inf'):
        env = os.environ.copy(); env['CUDA_LAUNCH_BLOCKING'] = '1'
        proc = subprocess.run([sys.executable, '-c', program, str(Path(codec.__file__).resolve()),
            json.dumps(codec.loaded_library()), str(torch.cuda.current_device()), value, policy],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if proc.returncode != 0 or 'MAV2_ASSERT_CONFIRMED' not in proc.stdout:
            raise RuntimeError('MAV2 nonfinite probe failed: '+proc.stderr[-2000:])
        checks.append(dict(value=value, rejected=True, separate_cuda_context=True))
    return checks


def _baseline_codec(codec):
    return SimpleNamespace(loaded_library=codec.loaded_library,
        pack_rows=lambda x, chunks: codec.pack_rows(x, chunks, policy='basic'),
        unpack_rows=lambda p, rows, n: codec.unpack_rows(p, rows, n, policy='basic'),
        decode_reduce=lambda p, local, rank: codec.decode_reduce(p, local, rank, policy='basic'))


def validate(codec, reference, rank):
    import torch
    import mav2_reference as oracle
    record = baseline.validate(_baseline_codec(codec), reference, rank)
    gen = torch.Generator(device='cuda').manual_seed(20261030 + rank)
    checks, direct_checks, retained = [], [], []
    for case in CASES:
        rows, n = (256, 4096) if case == 'full_shape' else (8, 128)
        partials = torch.randn((8, rows, n), generator=gen, device='cuda', dtype=torch.bfloat16)
        if case == 'zeros': partials.zero_()
        elif case == 'small': partials.mul_(1e-6)
        elif case == 'outliers': partials[:, :, ::128] = 64
        elif case == 'tiny':
            partials.zero_(); partials[:, :, ::2] = 2.**-133
            partials[:, :, 1::4] = -2.**-126
        elif case == 'mixed':
            partials.zero_(); partials[:, rows//2:, :] = .51953125
            partials[:, rows//2:, ::128] = 1
        elif case == 'rounding_midpoint':
            # Both constants are exact BF16 values. Correctly rounded scale
            # makes x/s == 368, an E4M3 midpoint that rounds to 384.
            # float32 amax * (1/448) moves below the midpoint and rounds to 352.
            partials.fill_(2.0563602447509766e-6)
            partials[:, :, 0] = 2.5033950805664062e-6
        remote = torch.cat([partials[s] for s in range(8) if s != rank])
        packets = codec.pack_rows(remote, 7, policy='mav2')
        actual = codec.unpack_rows(packets, rows, n, policy='mav2')
        decoded_wire = oracle.decode_packets(packets, rows, n)
        expected, _ = oracle.roundtrip(remote)
        # FP64 metrics retain tiny BF16 information instead of underflowing its norm.
        denom = remote.double().norm().clamp_min(1e-300)
        protocol = float((actual.reshape_as(remote).double()-expected.double()).norm()/denom)
        wire_error = float((actual.double()-decoded_wire.double()).norm()/denom)
        input_error = float((actual.reshape_as(remote).double()-remote.double()).norm()/denom)
        diag = codec.packet_diagnostics(packets, rows*n)
        if case == 'mixed' and not (diag['d8'] and diag['h8']):
            raise RuntimeError('Mixed preflight did not exercise both modes')
        fold = torch.zeros_like(partials[rank])
        for offset in range(1, 9):
            src = (rank+offset) % 8
            value = partials[src] if src == rank else actual[src-int(src > rank)]
            fold = (fold+value).to(torch.bfloat16)
        fused = codec.decode_reduce(packets, partials[rank], rank, policy='mav2')
        ordered_equal = torch.equal(fold.view(torch.uint8), fused.view(torch.uint8))
        # H8 fallback must retain the original encoder bytes, not merely a similar error.
        baseline_packet = codec.pack_rows(remote, 7, policy='basic')
        forced_h8 = codec.pack_rows(remote, 7, policy='mav2', force_mode=0)
        h8_equal = torch.equal(baseline_packet, forced_h8)
        baseline_h8_decoded = codec.unpack_rows(baseline_packet, rows, n, policy='basic')
        candidate_h8_decoded = codec.unpack_rows(forced_h8, rows, n, policy='mav2')
        h8_decode_equal = torch.equal(baseline_h8_decoded.view(torch.uint8),
                                     candidate_h8_decoded.view(torch.uint8))
        baseline_h8_fold = codec.decode_reduce(baseline_packet, partials[rank], rank, policy='basic')
        candidate_h8_fold = codec.decode_reduce(forced_h8, partials[rank], rank, policy='mav2')
        h8_fold_equal = torch.equal(baseline_h8_fold.view(torch.uint8),
                                   candidate_h8_fold.view(torch.uint8))
        # Exercise the actual formal D8 policy, including inputs rejected by MAV2's gate.
        forced_d8 = codec.pack_rows(remote, 7, policy='d8')
        direct_actual = codec.unpack_rows(forced_d8, rows, n, policy='d8').reshape_as(remote)
        direct_expected, _ = oracle.direct_roundtrip(remote)
        direct_error = float((direct_actual.double()-direct_expected.double()).norm()/denom)
        direct_wire = oracle.decode_packets(forced_d8, rows, n).reshape_as(remote)
        direct_wire_error = float((direct_actual.double()-direct_wire.double()).norm()/denom)
        direct_input_error = float((direct_actual.double()-remote.double()).norm()/denom)
        direct_diag = codec.packet_diagnostics(forced_d8, rows*n)
        midpoint_check = None
        if case == 'rounding_midpoint':
            elements, groups = rows*n, rows*n//128
            wire_scale = forced_d8[:, elements:elements+groups*4].contiguous().view(torch.float32)
            wire_q = forced_d8[:, :elements].contiguous().view(torch.float8_e4m3fn).float().reshape(7, rows, n)
            expected_q = torch.full_like(wire_q, 384.)
            expected_q[:, :, 0] = 448.
            midpoint_check = dict(scale_exact=bool((wire_scale == 5.587935447692871e-9).all()),
                                  payload_exact=bool((wire_q == expected_q).all()),
                                  reference_equal=torch.equal(direct_actual, direct_expected))
        direct_fold = torch.zeros_like(partials[rank])
        direct_sources = direct_actual.reshape(7,rows,n)
        for offset in range(1, 9):
            src = (rank+offset) % 8
            value = partials[src] if src == rank else direct_sources[src-int(src > rank)]
            direct_fold = (direct_fold+value).to(torch.bfloat16)
        direct_fused = codec.decode_reduce(forced_d8, partials[rank], rank, policy='d8')
        direct_ordered_equal = torch.equal(direct_fold.view(torch.uint8), direct_fused.view(torch.uint8))
        direct_finite = bool(torch.isfinite(direct_actual).all() and torch.isfinite(direct_fused).all())
        formal_equals_diagnostic = torch.equal(forced_d8,
            codec.pack_rows(remote, 7, policy='mav2', force_mode=1))
        direct_passed = (direct_finite and direct_ordered_equal and formal_equals_diagnostic and
            direct_error <= D8_REFERENCE_L2_LIMIT and direct_wire_error <= .005 and direct_input_error <= .05 and
            direct_diag['d8'] == direct_diag['total'] and direct_diag['h8'] == 0 and
            (midpoint_check is None or all(midpoint_check.values())))
        direct_row = dict(case=case, shape=[rows,n], passed=direct_passed, finite=direct_finite,
            relative_l2=direct_input_error, protocol_relative_l2=direct_error,
            packet_relative_l2=direct_wire_error, ordered_bf16_fold_equal=direct_ordered_equal,
            formal_equals_diagnostic=formal_equals_diagnostic, diagnostics=direct_diag,
            policy='d8', quality_selection=False, h8_fallback_allowed=False,
            budget_vs_reference=D8_REFERENCE_L2_LIMIT, rounding_midpoint=midpoint_check)
        if not direct_passed: raise RuntimeError('Formal D8 preflight failed: '+repr(direct_row))
        direct_checks.append(direct_row); retained.append((direct_fused, direct_fused.clone()))
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(fused).all())
        passed = (finite and ordered_equal and h8_equal and h8_decode_equal and h8_fold_equal
                  and protocol <= .01 and wire_error <= .005
                  and input_error <= .05 and direct_error <= D8_REFERENCE_L2_LIMIT)
        row = dict(case=case, shape=[rows,n], passed=passed, finite=finite,
            relative_l2=input_error, protocol_relative_l2=protocol, packet_relative_l2=wire_error,
            forced_d8_relative_l2=direct_error, forced_d8_reference_budget=D8_REFERENCE_L2_LIMIT,
            forced_h8_packet_equal=h8_equal,
            forced_h8_decode_equal=h8_decode_equal, forced_h8_fold_equal=h8_fold_equal,
            ordered_bf16_fold_equal=ordered_equal, diagnostics=diag)
        if not passed: raise RuntimeError('MAV2 codec preflight failed: '+repr(row))
        checks.append(row); retained.append((fused, fused.clone()))
    preserved = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a,b in retained)
    if not preserved: raise RuntimeError('MAV2 overwrote a retained output')
    record['selective_transform'] = dict(passed=True, rank=rank, cases=checks,
        retained_outputs_unchanged=True, synthetic=False, selector=SELECTOR,
        nonfinite_input_policy='CUDA device assertion', nonfinite_checks=validate_nonfinite(codec),
        quantized_remote_fraction=1.0, bytes_per_group=136)
    record['forced_direct'] = dict(passed=True, rank=rank, cases=direct_checks,
        retained_outputs_unchanged=True, synthetic=False, policy='d8', quality_selection=False,
        h8_fallback_allowed=False, nonfinite_input_policy='CUDA device assertion',
        nonfinite_checks=validate_nonfinite(codec, policy='d8'),
        quantized_remote_fraction=1.0, bytes_per_group=136)
    return record


def verify(record, library, rank):
    from common import require
    baseline.verify(record, library, rank)
    check = record.get('selective_transform', {})
    require(check.get('passed') is True and check.get('synthetic') is False and
            check.get('rank') == rank and check.get('retained_outputs_unchanged') is True and
            check.get('selector') == SELECTOR and check.get('quantized_remote_fraction') == 1.0 and
            check.get('nonfinite_checks') == [dict(value=v, rejected=True, separate_cuda_context=True)
                                              for v in ('nan', 'inf')] and
            check.get('bytes_per_group') == 136 and [r['case'] for r in check.get('cases', [])] == list(CASES),
            'Missing MAV2 GPU preflight')
    for row in check['cases']:
        expected_shape = [256,4096] if row['case'] == 'full_shape' else [8,128]
        require(row['passed'] and row['finite'] and row['shape'] == expected_shape and
                row['forced_h8_packet_equal'] and row['forced_h8_decode_equal'] and
                row['forced_h8_fold_equal'] and row['ordered_bf16_fold_equal'] and
                row['forced_d8_reference_budget'] == D8_REFERENCE_L2_LIMIT, 'Incomplete MAV2 codec evidence')
        for key, limit in (('relative_l2',.05), ('protocol_relative_l2',.01),
                           ('packet_relative_l2',.005), ('forced_d8_relative_l2',D8_REFERENCE_L2_LIMIT)):
            require(math.isfinite(row[key]) and 0 <= row[key] <= limit, 'MAV2 codec numerical gate failed')
        d = row['diagnostics']; total = 7*expected_shape[0]*expected_shape[1]//128
        require(d['total'] == total and d['d8'] + d['h8'] == total and min(d['d8'],d['h8']) >= 0 and
                d['nonfinite'] == d['invalid_tags'] == 0 and d['selector'] == SELECTOR and
                d['bytes_per_group'] == 136, 'Invalid mode evidence')
        if row['case'] == 'mixed': require(d['d8'] > 0 and d['h8'] > 0, 'Both modes were not exercised')
    direct = record.get('forced_direct', {})
    require(direct.get('passed') is True and direct.get('synthetic') is False and
            direct.get('rank') == rank and direct.get('retained_outputs_unchanged') is True and
            direct.get('policy') == 'd8' and direct.get('quality_selection') is False and
            direct.get('h8_fallback_allowed') is False and direct.get('quantized_remote_fraction') == 1.0 and
            direct.get('nonfinite_checks') == [dict(value=v,rejected=True,separate_cuda_context=True)
                                               for v in ('nan','inf')] and
            direct.get('bytes_per_group') == 136 and
            [r['case'] for r in direct.get('cases',[])] == list(CASES), 'Missing formal D8 GPU preflight')
    for row in direct['cases']:
        expected_shape = [256,4096] if row['case'] == 'full_shape' else [8,128]
        require(row['passed'] and row['finite'] and row['shape'] == expected_shape and
                row['ordered_bf16_fold_equal'] and row['formal_equals_diagnostic'] and
                row['policy'] == 'd8' and row['quality_selection'] is False and
                row['h8_fallback_allowed'] is False and row['budget_vs_reference'] == D8_REFERENCE_L2_LIMIT,
                'Incomplete formal D8 evidence')
        if row['case'] == 'rounding_midpoint':
            require(row['rounding_midpoint'] == dict(scale_exact=True, payload_exact=True, reference_equal=True),
                    'D8 scale/FP8 midpoint rounding changed')
        else:
            require(row['rounding_midpoint'] is None, 'Unexpected rounding-midpoint evidence')
        for key,limit in (('relative_l2',.05),('protocol_relative_l2',D8_REFERENCE_L2_LIMIT),('packet_relative_l2',.005)):
            require(math.isfinite(row[key]) and 0 <= row[key] <= limit, 'Formal D8 numerical gate failed')
        d = row['diagnostics']; total = 7*expected_shape[0]*expected_shape[1]//128
        require(d['total'] == d['d8'] == total and d['h8'] == 0 and d['nonfinite'] == d['invalid_tags'] == 0 and
                d['selector'] == SELECTOR and d['bytes_per_group'] == 136, 'D8 representation was substituted')

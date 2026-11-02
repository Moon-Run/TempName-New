"""Synthetic trace/evidence audits, never treated as measured GPU evidence."""
import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import open_compress_probe as probe
import open_compress_profile as profile

GEMM = ('cutlass::Kernel2<flux_bf16_bf16_void_bf16_fp32_fp32_sm89_rtx5090_'
        'commnone_rcr_gemmv2_streamksk_{swizzle}>')


def fixture(arm):
    events, correlation = [], 0
    quantized = arm != 'flux-bf16'

    def host(name, ts, dur, cat='user_annotation', **args):
        value = dict(cat=cat, name=name, ph='X', ts=ts, dur=dur, tid=1, args=args)
        events.append(value)
        return value

    def gpu(name, cpu, ts, dur, stream=7, cat='kernel'):
        nonlocal correlation
        correlation += 1
        host('cudaLaunchKernel', cpu, .1, 'cuda_runtime', correlation=correlation)
        event = dict(cat=cat, name=name, ph='X', ts=ts, dur=dur, tid=stream,
                     args=dict(correlation=correlation, stream=stream, device=0))
        events.append(event)
        return event

    for i, role in enumerate(('attn', 'mlp')):
        cpu, clock = 200*i, 1000+200*i
        host(f'scope.{role}.flux_tensor', cpu, 100)
        host('ablation6.flux_gemm_only', cpu+1, 10)
        gpu('Memset (Device)', cpu+2, clock, 1, cat='gpu_memset')
        gpu(GEMM.format(swizzle='Op1AInterleavedSwizzle' if arm.endswith('interleaved') else 'Original'),
            cpu+3, clock+2, 5)
        if quantized:
            host('open_compress.encode', cpu+12, 5)
            gpu('oc_int8_encode_kernel', cpu+13, clock+8, 2)
            host('open_compress.exchange', cpu+20, 20)
            host('c10d::alltoall_base_', cpu+21, 8, 'cpu_op')
            gpu('ncclDevKernel_SendRecv', cpu+22, clock+12, 10, stream=26)
            host('record_param_comms', cpu+31, 5, 'cpu_op', **{'Collective name': 'wait'})
            host('cudaStreamWaitEvent', cpu+32, .1, 'cuda_runtime')
            host('open_compress.decode_reduce', cpu+41, 5)
            gpu('oc_int8_decode_reduce_kernel', cpu+42, clock+23, 3)
        else:
            host('flux_baseline.bf16_reduce_scatter', cpu+20, 20)
            host('c10d::_reduce_scatter_base_', cpu+21, 8, 'cpu_op')
            gpu('ncclDevKernel_ReduceScatter', cpu+22, clock+12, 10)
    return events


def first(events, name, category=None):
    return next(e for e in events if name in e['name'] and (category is None or e['cat'] == category))


class ProfileAudit(unittest.TestCase):
    def test_three_actual_paths_with_streamk_memset(self):
        for arm in profile.p.ARMS:
            with self.subTest(arm=arm):
                result = profile.analyze(fixture(arm), arm, 0, expected_calls=1)
                self.assertTrue(result['execution_validated'])
                self.assertEqual(result['roles']['attn']['calls'], 1)
                self.assertEqual(result['interleaved'], arm.endswith('interleaved'))

    def test_wait_must_establish_actual_cuda_dependency(self):
        for missing in ('record_param_comms', 'cudaStreamWaitEvent'):
            events = fixture('flux-int8')
            events.remove(first(events, missing))
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                profile.analyze(events, 'flux-int8', 0, 1)

    def test_wrong_stream_device_order_and_swizzle_rejected(self):
        for mutation in ('stream', 'device', 'early_decode', 'swizzle', 'wait_before_exchange'):
            events = fixture('flux-int8')
            decoder = first(events, 'oc_int8_decode_reduce_kernel')
            if mutation == 'stream':
                decoder['args']['stream'] = 30
            elif mutation == 'device':
                decoder['args']['device'] = 1
            elif mutation == 'early_decode':
                decoder['ts'] = first(events, 'nccl')['ts']
            elif mutation == 'swizzle':
                first(events, 'cutlass')['name'] = GEMM.format(swizzle='Op1AInterleavedSwizzle')
            else:
                first(events, 'record_param_comms')['ts'] -= 5
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                profile.analyze(events, 'flux-int8', 0, 1)

    def test_launch_and_all_work_must_belong_to_stages(self):
        for mutation in ('foreign_launch', 'extra_copy', 'unowned_marker', 'wrong_collective'):
            events = fixture('flux-int8')
            if mutation == 'foreign_launch':
                decoder = first(events, 'oc_int8_decode_reduce_kernel')
                launch = next(e for e in events if e['cat'] == 'cuda_runtime' and
                              e['args'].get('correlation') == decoder['args']['correlation'])
                launch['ts'] += 10
            elif mutation == 'extra_copy':
                extra = copy.deepcopy(first(events, 'oc_int8_encode_kernel'))
                extra.update(cat='gpu_memcpy', name='Memcpy DtoD')
                events.append(extra)
            elif mutation == 'unowned_marker':
                extra = copy.deepcopy(first(events, 'open_compress.encode'))
                extra['ts'] = 90000
                events.append(extra)
            else:
                first(events, 'c10d::alltoall')['name'] = 'c10d::_reduce_scatter_base_'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                profile.analyze(events, 'flux-int8', 0, 1)

    def test_taco_and_codec_outside_scopes_are_rejected(self):
        for name in ('taco_decode', 'oc_int8_decode_kernel'):
            events = fixture('flux-bf16')
            events.append(dict(cat='kernel', name=name, ph='X', ts=99999, dur=1,
                               args=dict(device=0, stream=7), tid=7))
            with self.subTest(name=name), self.assertRaises(ValueError):
                profile.analyze(events, 'flux-bf16', 0, 1)


def arithmetic_fixture():
    cases = {case: dict(finite_input=True, encode_bitwise=True, decode_bitwise=True,
                       reduce_bitwise=True, finite_output=True, packet_sha256='a'*64)
             for case in probe.CODEC_CASES}
    cases['signed_halfway']['expected_signed_bytes'] = [127, -127, 0, 2, 2, 4, 0, -2, -2, -4]
    return dict(completed=True, cases=cases, nonfinite=dict(nan_inf_negative_inf=True,
        affected_remote_groups=7, nan_scale_propagated=True, decode_nan_propagated=True,
        reduction_nan_propagated=True, unaffected_values_zero=True))


def measurement_fixture():
    return dict(iterations=10, warmup=3, samples=[dict(cuda_ms=1., wall_ms=2.,
        peak_allocated_gib=1024/2**30, peak_reserved_gib=2048/2**30,
        incremental_allocated_bytes=512, baseline_allocated_bytes=512,
        device_free_before_bytes=100, device_total_bytes=1000) for _ in range(5)])


def rank_fixture(rank):
    def wire(source, destination):
        return hashlib.sha256(f'{source}:{destination}'.encode()).hexdigest()
    record = dict(protocol=probe.p.PROTOCOL, arm='flux-int8', rank=rank, run_id='test', completed=True,
        build_manifest_sha256='build', entrypoint_sha256=probe.p.sha(probe.__file__), libraries={},
        codec_library={}, stream=7, default_stream=0, codec_cases=arithmetic_fixture(), roles={})
    record['matmul_settings'] = dict(allow_tf32=False, allow_bf16_reduced_precision_reduction=False)
    for role, k in probe.ROLES.items():
        cases = {case: dict(passed=True, finite_partial=True, relative_l2_vs_bf16=.01,
            gemm_relative_l2=.001, input_sha256='b'*64, weight_sha256='c'*64, encode_bitwise=True,
            decode_bitwise=True, reduce_bitwise=True,
            tx={str(peer): wire(rank, peer) for peer in range(8) if peer != rank},
            rx={str(peer): wire(peer, rank) for peer in range(8) if peer != rank}) for case in probe.CASES}
        transport = dict(calls=10, owns_output=True, encode_calls=10, decode_calls=10,
            all_to_all_calls=10, wait_calls=10, caller_stream=7, current_stream=7,
            pending_exchange=False, failed=False,
            send_split_bytes=[0 if peer == rank else probe.p.PACKET_BYTES for peer in range(8)],
            receive_split_bytes=[0 if peer == rank else probe.p.PACKET_BYTES for peer in range(8)])
        record['roles'][role] = dict(k_local=k, cases=cases, retention_passed=True, retained_outputs=6,
            reuse_calls=7, mapping=None, transport=transport, changed_call_output_different=True,
            delayed_rank=7, delay_seconds=.02,
            timings={name: measurement_fixture() for name in
                     ('gemm', 'operator', 'encode', 'exchange', 'decode', 'decode_reduce')})
    return record


class ProbeEvidenceAudit(unittest.TestCase):
    def validate_rows(self, rows):
        def read(path):
            return rows[int(Path(path).stem.removeprefix('rank'))]
        with patch.object(probe.p, 'read', read):
            return probe.validate(Path('/fake'), 'flux-int8',
                dict(run_id='test', build=dict(manifest_sha256='build')),
                dict(libraries={}, config=dict(codec_library={})))

    def test_complete_directed_wire_and_stream_evidence(self):
        self.assertEqual(len(self.validate_rows([rank_fixture(r) for r in range(8)])), 8)

    def test_packet_corruption_bad_hash_and_missing_wait_fail(self):
        for mutation in ('packet', 'hash', 'wait', 'stream', 'nonfinite', 'negative_error'):
            rows = [rank_fixture(r) for r in range(8)]
            role = rows[0]['roles']['attn']
            if mutation == 'packet':
                role['cases']['random']['tx']['1'] = 'a'*64
            elif mutation == 'hash':
                role['cases']['random']['tx']['1'] = ''
            elif mutation == 'wait':
                role['transport']['wait_calls'] = 9
            elif mutation == 'stream':
                role['transport']['current_stream'] = 30
            elif mutation == 'nonfinite':
                rows[0]['codec_cases']['nonfinite']['nan_scale_propagated'] = False
            else:
                role['cases']['random']['relative_l2_vs_bf16'] = -1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate_rows(rows)

    def test_bad_rounding_and_inconsistent_peak_memory_fail(self):
        arithmetic = arithmetic_fixture()
        arithmetic['cases']['signed_halfway']['expected_signed_bytes'][3] = 1
        with self.assertRaises(ValueError):
            probe.validate_codec_cases(arithmetic)
        measurement = measurement_fixture()
        measurement['samples'][0]['incremental_allocated_bytes'] = 0
        with self.assertRaises(ValueError):
            probe.validate_measurement(measurement)


if __name__ == '__main__':
    unittest.main()

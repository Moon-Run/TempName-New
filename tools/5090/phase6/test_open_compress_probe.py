"""Synthetic trace/evidence audits, never treated as measured GPU evidence."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import open_compress_probe as probe
import open_compress_profile as profile

GEMM = ('cutlass::Kernel2<flux_bf16_bf16_void_bf16_fp32_fp32_sm89_rtx5090_'
        'commnone_rcr_gemmv2_streamksk_{swizzle}>')


def fixture(arm):
    events, correlation = [], 0
    quantized = probe.is_quantized(arm)
    native = probe.is_megatron(arm)

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
        host(probe.projection_scope(arm, role), cpu, 100)
        host(profile.NATIVE_GEMM_STAGE if native else profile.GEMM_STAGE, cpu+1, 10)
        if native:
            host('LinearWithGradAccumulationAndAsyncCommunication', cpu+1.2, 9.5, 'cpu_op')
            host('aten::mm', cpu+1.5, 9, 'cpu_op')
        gpu('Memset (Device)', cpu+2, clock, 1, cat='gpu_memset')
        gemm = ('void cublasLt::blackwell_bf16_gemm_kernel' if native else GEMM.format(swizzle=
            'Op1AInterleavedSwizzle' if arm.endswith('interleaved') else
            'OpenCompressRemoteFirstSwizzle' if arm.endswith('remote-first') else 'Original'))
        gpu(gemm, cpu+3, clock+2, 5)
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
            host(profile.NATIVE_RS_STAGE if native else 'flux_baseline.bf16_reduce_scatter', cpu+20, 20)
            host('c10d::_reduce_scatter_base_', cpu+21, 8, 'cpu_op')
            gpu('ncclDevKernel_ReduceScatter', cpu+22, clock+12, 10)
    return events


def first(events, name, category=None):
    return next(e for e in events if name in e['name'] and (category is None or e['cat'] == category))


class ProfileAudit(unittest.TestCase):
    def test_all_actual_paths_with_streamk_or_cublas_memset(self):
        for arm in profile.p.ARMS:
            with self.subTest(arm=arm):
                result = profile.analyze(fixture(arm), arm, 0, expected_calls=1)
                self.assertTrue(result['execution_validated'])
                self.assertEqual(result['roles']['attn']['calls'], 1)
                self.assertEqual(result['interleaved'], arm.endswith('interleaved'))
                self.assertEqual(result['remote_first'], arm.endswith('remote-first'))
                self.assertEqual(result['native_megatron'], probe.is_megatron(arm))

    def test_megatron_requires_actual_aten_owner_and_rejects_flux(self):
        for arm in ('megatron-bf16', 'megatron-int8'):
            for mutate in ('missing_mm', 'mm_launch_outside_owner', 'flux', 'wrong_route'):
                events = fixture(arm)
                if mutate == 'missing_mm':
                    events.remove(first(events, 'aten::mm'))
                elif mutate == 'mm_launch_outside_owner':
                    first(events, 'aten::mm')['ts'] += 3
                elif mutate == 'flux':
                    first(events, 'blackwell_bf16_gemm')['name'] = GEMM.format(swizzle='Original')
                else:
                    first(events, 'scope.attn.').update(name='scope.attn.flux_tensor')
                with self.subTest(arm=arm, mutation=mutate), self.assertRaises(ValueError):
                    profile.analyze(events, arm, 0, 1)

    def test_remote_first_kernel_cannot_be_reported_as_interleaved_or_original(self):
        for arm in ('flux-int8', 'flux-int8-interleaved'):
            events = fixture(arm)
            first(events, 'cutlass')['name'] = GEMM.format(swizzle='OpenCompressRemoteFirstSwizzle')
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                profile.analyze(events, arm, 0, 1)

    def test_remote_first_maps_all_remote_tiles_before_local(self):
        for rank in range(8):
            original = [[slot//32, slot%32] for slot in range(512)]
            rows = [[((rank+1+slot//64)%8)*2+(slot%64)//32, slot%32, *original[slot]] for slot in range(512)]
            probe.validate_remote_first_mapping(rows, rank)
            rows[0], rows[448] = rows[448], rows[0]
            with self.assertRaises(ValueError):
                probe.validate_remote_first_mapping(rows, rank)

    def test_native_cublas_split_k_epilogue_is_owned_by_the_same_mm(self):
        events = fixture('megatron-int8')
        gemm = first(events, 'blackwell_bf16_gemm')
        launch = next(e for e in events if e['cat'] == 'cuda_runtime' and
                      e['args'].get('correlation') == gemm['args']['correlation'])
        extra, owner = copy.deepcopy(gemm), copy.deepcopy(launch)
        extra.update(name='void cublasLt::splitKreduce_kernel', ts=gemm['ts']+gemm['dur'], dur=.5)
        extra['args']['correlation'] += 1000
        owner['args']['correlation'] += 1000
        owner['ts'] += 1
        events.extend((owner,extra))
        self.assertTrue(profile.analyze(events, 'megatron-int8', 0, 1)['execution_validated'])

    def test_native_process_rejects_loaded_flux_binary(self):
        with patch.object(probe.Path, 'read_text', return_value='/usr/lib/libc.so\n'):
            self.assertEqual(probe.loaded_libraries(None),{})
        for filename in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
            with patch.object(probe.Path, 'read_text', return_value='/foreign/'+filename+'\n'), self.assertRaises(ValueError):
                probe.loaded_libraries(None)

    def test_native_autograd_owner_name_is_not_a_reduce_scatter_collective(self):
        events = fixture('megatron-int8')
        events.append(dict(cat='cpu_op',name='OpenCompressNativeReduceScatter',ph='X',
                           ts=11.5,dur=35,tid=1,args={}))
        self.assertTrue(profile.analyze(events,'megatron-int8',0,1)['execution_validated'])
        events.append(dict(cat='cpu_op',name='c10d::_reduce_scatter_base_',ph='X',
                           ts=48,dur=2,tid=1,args={}))
        with self.assertRaises(ValueError):
            profile.analyze(events,'megatron-int8',0,1)

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


def rank_fixture(rank, arm='flux-int8'):
    def wire(source, destination):
        return hashlib.sha256(f'{source}:{destination}'.encode()).hexdigest()
    record = dict(protocol=probe.p.PROTOCOL, arm=arm, rank=rank, run_id='test', completed=True,
        build_manifest_sha256='build', entrypoint_sha256=probe.p.sha(probe.__file__), libraries={},
        codec_library={}, stream=7, default_stream=0, codec_cases=arithmetic_fixture(), roles={})
    record['matmul_settings'] = dict(allow_tf32=False, allow_bf16_reduced_precision_reduction=False)
    record['operator_grad_enabled'] = False
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
            native_execution=(dict(arm=arm, calls=10, linear_calls=10, native_forward_preserved=True,
                native_linear_backward_preserved=True, native_rs_backward_preserved=True,
                original_rs_forward=arm == 'megatron-bf16', private_function_globals=True, flux_loaded=False,
                source_sha256={'layers.py':'d'*64, 'mappings.py':'e'*64}) if probe.is_megatron(arm) else None),
            delayed_rank=7, delay_seconds=.02,
            timings={name: measurement_fixture() for name in
                     ('gemm', 'operator', 'encode', 'exchange', 'decode', 'decode_reduce')})
    return record


class ProbeEvidenceAudit(unittest.TestCase):
    def test_native_operator_close_happens_after_device_completion(self):
        events=[]
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:events.append('synchronize')))
        op=SimpleNamespace(close=lambda:events.append('close'))
        with patch.dict('sys.modules',{'torch':torch}):
            probe.close_operator(op,'megatron-int8')
            probe.close_operator(op,'megatron-bf16')
            probe.close_operator(op,'flux-int8')
        self.assertEqual(events,['synchronize','close','synchronize','close','synchronize'])

    def validate_rows(self, rows, arm='flux-int8'):
        def read(path):
            return rows[int(Path(path).stem.removeprefix('rank'))]
        with patch.object(probe.p, 'read', read):
            return probe.validate(Path('/fake'), arm,
                dict(run_id='test', build=dict(manifest_sha256='build')),
                dict(libraries={}, config=dict(codec_library={}, native_source_sha256={'layers.py':'d'*64, 'mappings.py':'e'*64})))

    def test_complete_directed_wire_and_stream_evidence(self):
        self.assertEqual(len(self.validate_rows([rank_fixture(r) for r in range(8)])), 8)

    def test_megatron_quantization_requires_native_forward_and_backward(self):
        rows = [rank_fixture(r, 'megatron-int8') for r in range(8)]
        self.assertEqual(len(self.validate_rows(rows, 'megatron-int8')), 8)
        for key in ('native_forward_preserved', 'native_linear_backward_preserved', 'native_rs_backward_preserved'):
            broken = copy.deepcopy(rows)
            broken[0]['roles']['attn']['native_execution'][key] = False
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.validate_rows(broken, 'megatron-int8')

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

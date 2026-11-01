"""Synthetic launch/dependency fixtures, never stored as GPU evidence."""
import copy
import unittest

import op1_abc_profile as profile

GEMM = ('cutlass::Kernel2<flux_bf16_bf16_void_bf16_fp32_fp32_sm89_rtx5090_'
        'commnone_rcr_gemmv2_streamksk_Op1AInterleavedSwizzle>')


def fixture(arm='flux-basic-arrival', overlap=True):
    events, correlation = [], 0
    chunks = 1 if arm == 'flux-basic-interleaved' else 4

    def host(category, name, timestamp, duration, external=None):
        event = dict(cat=category, name=name, ph='X', ts=timestamp, dur=duration, tid=1, args={})
        if external is not None:
            event['args']['External id'] = external
        events.append(event)
        return event

    def gpu(cpu, timestamp, name, duration=1, stream=7, external=None):
        nonlocal correlation
        correlation += 1
        launch = host('cuda_runtime', 'cudaLaunchKernel', cpu, .1)
        launch['args']['correlation'] = correlation
        event = dict(cat='kernel', name=name, ph='X', ts=timestamp, dur=duration, tid=stream,
                     args=dict(correlation=correlation, device=0, stream=stream))
        if external is not None:
            event['args']['External id'] = external
        events.append(event)
        return event

    for projection in range(256):
        cpu, start = projection*300, 1e6+projection*300
        role = 'attn' if projection < 128 else 'mlp'
        host('user_annotation', f'scope.{role}.flux_tensor', cpu, 150)
        host('user_annotation', 'ablation6.flux_gemm_only', cpu+1, 5)
        gpu(cpu+2, start, GEMM, duration=5)
        decoders = []
        for chunk in range(chunks):
            offset = cpu+10+chunk*30
            host('user_annotation', f'op1_abc.encode.{chunk}', offset, 4)
            gpu(offset+1, start+10+chunk*5, profile.ENCODER)
            host('user_annotation', f'op1_abc.exchange.{chunk}', offset+5, 10)
            host('cpu_op', 'c10d::alltoall_base_', offset+6, 7, projection*10+chunk)
            gpu(offset+7, start+35+chunk*15, 'ncclDevKernel_SendRecv', duration=10,
                stream=26, external=projection*10+chunk)
            host('user_annotation', f'op1_abc.decode.{chunk}', offset+16, 5)
            decoders.append(gpu(offset+17,
                start+(46+chunk*15 if overlap else 100+chunk*15), profile.DECODER,
                duration=12, stream=30 if chunks > 1 else 7))
        if chunks > 1:
            host('user_annotation', 'op1_abc.join', cpu+135, 5)
        consumer = max(profile.end(e) for e in decoders)+2
        for i in range(2):
            external = 10000+projection*3+i
            host('cpu_op', 'aten::add', cpu+160+i*5, 3, external)
            gpu(cpu+161+i*5, consumer+i*3,
                'vectorized_elementwise_kernel<AddFunctor<c10::BFloat16>>', external=external)
        external = 10000+projection*3+2
        host('cpu_op', 'FusedLayerNormAffineFunction', cpu+175, 4, external)
        gpu(cpu+176, consumer+6, 'cuApplyLayerNorm<c10::BFloat16>', external=external)
    return events


def find(events, name, category='kernel'):
    return next(e for e in events if e['cat'] == category and name in e['name'])


class ProfileTest(unittest.TestCase):
    def test_actual_chunks_and_observed_overlap(self):
        for arm in profile.p.BASIC_ARMS:
            value = profile.analyze(fixture(arm), arm, 0)
            self.assertTrue(value['execution_validated'])
            row = value['row_forward']['attn']
            self.assertEqual(row['chunks_per_call'], 1 if arm.endswith('interleaved') else 4)
            self.assertEqual(row['calls_with_observed_decode_communication_overlap'],
                             0 if arm.endswith('interleaved') else 128)

    def test_serialized_pipeline_does_not_claim_overlap(self):
        rows = profile.chunk_records(fixture(overlap=False), 'flux-basic-arrival', 0)
        self.assertEqual(rows['attn']['calls_with_observed_decode_communication_overlap'], 0)
        self.assertEqual(rows['attn']['median_decode_communication_overlap_us'], 0)

    def test_early_decode_missing_ownership_and_early_consumer_fail(self):
        for mutation in ('early_decode', 'foreign_launch', 'early_consumer', 'wrong_stream', 'wrong_gemm'):
            events = fixture()
            decoder = find(events, profile.DECODER)
            if mutation == 'early_decode':
                decoder['ts'] = find(events, 'nccl')['ts']
            elif mutation == 'foreign_launch':
                launch = next(e for e in events if e['cat'] == 'cuda_runtime'
                              and e['args']['correlation'] == decoder['args']['correlation'])
                launch['ts'] += 20
            elif mutation == 'early_consumer':
                find(events, 'AddFunctor')['ts'] = decoder['ts']+.1
            elif mutation == 'wrong_stream':
                decoder['args']['stream'] = 7
            else:
                find(events, 'cutlass')['name'] = GEMM.replace('Op1AInterleavedSwizzle', 'Original')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                profile.chunk_records(events, 'flux-basic-arrival', 0)

    def test_chunk_join_and_consumer_owner_are_mandatory(self):
        for token, category in [('op1_abc.join', 'user_annotation'),
                                ('op1_abc.exchange.3', 'user_annotation'),
                                ('FusedLayerNormAffineFunction', 'cpu_op')]:
            events = fixture()
            events.remove(find(events, token, category))
            with self.subTest(token=token), self.assertRaises(ValueError):
                profile.chunk_records(events, 'flux-basic-arrival', 0)


if __name__ == '__main__':
    unittest.main()

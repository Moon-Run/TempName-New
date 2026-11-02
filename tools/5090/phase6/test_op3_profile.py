"""Synthetic OP3 ownership/dependency failures; these are not GPU evidence."""
import copy
import unittest

import op3_profile as profile
from test_op1_abc_profile import fixture as old_fixture, find


def forward_fixture(arm):
    events = old_fixture('flux-basic-interleaved')
    for event in events:
        event['name'] = event['name'].replace('op1_abc.', 'op3.')
        if event['cat'] == 'kernel':
            if 'op1_abc_encode_kernel' in event['name']:
                event['name'] = profile.ENCODERS[arm]
            elif 'op1_abc_decode_kernel' in event['name']:
                event['name'] = profile.DECODERS[arm] + ('<true>' if profile.p.fused(arm) else '')
    if profile.p.fused(arm):
        adds = [e for e in events if e['cat'] == 'kernel' and 'AddFunctor' in e['name']]
        correlations = {e['args']['correlation'] for e in adds}
        externals = {e['args']['External id'] for e in adds}
        events = [e for e in events if e not in adds
                  and e.get('args', {}).get('correlation') not in correlations
                  and e.get('args', {}).get('External id') not in externals]
    for exchange in [e for e in events if e['name'] == 'op3.exchange.0']:
        events += [dict(cat='cpu_op', name='record_param_comms', ph='X',
                        ts=exchange['ts']+8.2, dur=1, tid=1, args={'Collective name': 'wait'}),
                   dict(cat='cuda_runtime', name='cudaStreamWaitEvent', ph='X',
                        ts=exchange['ts']+8.4, dur=.2, tid=1, args={})]
    return events


def backward_fixture(fused=True, splitk=False):
    events, correlation = [], 0
    function = 'ConsumerGemmRSFunction' if fused else 'GemmRSFunction'

    def host(name, start, duration, category='cpu_op', **args):
        event = dict(cat=category, name=name, ph='X', ts=start, dur=duration, tid=1, args=args)
        events.append(event)
        return event

    def gpu(name, cpu, timestamp):
        nonlocal correlation
        correlation += 1
        launch = host('cudaLaunchKernel', cpu, .1, 'cuda_runtime', correlation=correlation)
        event = dict(cat='kernel', name=name, ph='X', ts=timestamp, dur=1, tid=7,
                     args=dict(correlation=correlation, stream=7, device=0))
        events.append(event)
        return event

    for i in range(256):
        cpu, start = i*300, 1e6+i*300
        role = 'attn' if i < 128 else 'mlp'
        host(f'scope.{role}.flux_tensor', cpu, 50, 'user_annotation')
        host(function, cpu+1, 10, **{'Sequence number': i})
        host(function+'Backward', cpu+100, 100, **{'Sequence number': i})
        if fused:
            host('aten::sum', cpu+110, 8)
            gpu('void at::native::reduce_kernel<BFloat16Bias>', cpu+111, start+1)
        gpu('ncclDevKernel_AllGather', cpu+121, start+3)
        host('aten::mm', cpu+130, 10)
        gpu('cublas_bf16_gemm_dgrad', cpu+131, start+6)
        if splitk and role == 'attn':
            gpu('void cublasLt::splitKreduce_kernel<__nv_bfloat16>', cpu+133, start+8)
        host('aten::mm', cpu+150, 10)
        gpu('cublas_bf16_gemm_wgrad', cpu+151, start+10)
    return events


class ForwardAudit(unittest.TestCase):
    def test_actual_specialization_consumer_and_single_exchange_counts(self):
        for arm in profile.p.ARMS:
            result = profile.chunk_records(forward_fixture(arm), arm, 0)
            for role in ('attn', 'mlp'):
                row = result[role]
                self.assertEqual(row['calls'], 128)
                self.assertEqual(row['chunks_per_call'], 1)
                self.assertEqual(row['all_to_all_calls'], 128)
                self.assertEqual(row['consumer_add_kernels'], 0 if profile.p.fused(arm) else 256)
                self.assertEqual(row['fused_consumer'], profile.p.fused(arm))

    def test_wrong_decoder_specialization_and_wrong_encoder_rejected(self):
        for mutation in ('raw_r', 'wrong_encoder', 'extra_native'):
            events = forward_fixture('op3-r')
            if mutation == 'raw_r':
                find(events, 'op3_r_decode_kernel')['name'] = 'op3_r_decode_kernel<false>'
            elif mutation == 'wrong_encoder':
                find(events, 'op1_abc_encode_kernel')['name'] = 'op3_warp_encode_kernel'
            else:
                events.append(copy.deepcopy(find(events, 'op3_r_decode_kernel')))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                profile.chunk_records(events, 'op3-r', 0)

    def test_wait_and_stream_dependency_are_both_required(self):
        for mutation in ('missing_wait', 'missing_stream_wait', 'early_wait'):
            events = forward_fixture('op3-improved')
            wait = find(events, 'record_param_comms', 'cpu_op')
            if mutation == 'missing_wait':
                events.remove(wait)
            elif mutation == 'missing_stream_wait':
                events.remove(find(events, 'cudaStreamWaitEvent', 'cuda_runtime'))
            else:
                wait['ts'] -= 4
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                profile.chunk_records(events, 'op3-improved', 0)

    def test_launch_ownership_order_and_consumer_form_are_strict(self):
        for mutation in ('early_decode', 'foreign_launch', 'wrong_stream', 'consumer_owner', 'wrong_consumer'):
            events = forward_fixture('op3-improved')
            decoder = find(events, 'op1_abc_decode_kernel')
            if mutation == 'early_decode':
                decoder['ts'] = find(events, 'nccl')['ts']
            elif mutation == 'foreign_launch':
                launch = next(e for e in events if e['cat'] == 'cuda_runtime'
                              and e['args'].get('correlation') == decoder['args']['correlation'])
                launch['ts'] += 20
            elif mutation == 'wrong_stream':
                decoder['args']['stream'] = 30
            elif mutation == 'consumer_owner':
                events.remove(find(events, 'FusedLayerNormAffineFunction', 'cpu_op'))
            else:
                find(events, 'AddFunctor')['name'] = 'foreign_kernel'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                profile.chunk_records(events, 'op3-improved', 0)


class BackwardAudit(unittest.TestCase):
    def test_both_autograd_paths_preserve_gathers_and_cublas_splitk(self):
        for arm in profile.p.ARMS:
            result = profile.row_backward(backward_fixture(profile.p.fused(arm), True), arm, 0)
            self.assertEqual(result['attn']['splitk_reduction_kernels'], 128)
            self.assertEqual(result['mlp']['splitk_reduction_kernels'], 0)
            self.assertEqual(result['attn']['dgrad_calls'], 128)
            if profile.p.fused(arm):
                self.assertEqual(result['attn']['bias_reduction_kernels'], 128)

    def test_bias_order_owner_and_missing_work_are_rejected(self):
        for mutation in ('gather_before_bias', 'wrong_bias_stream', 'missing_bias_owner', 'missing_bias'):
            events = backward_fixture()
            bias = find(events, 'BFloat16Bias')
            if mutation == 'gather_before_bias':
                find(events, 'nccl')['ts'] = bias['ts']-.5
            elif mutation == 'wrong_bias_stream':
                bias['args']['stream'] = 26
            elif mutation == 'missing_bias_owner':
                events.remove(find(events, 'aten::sum', 'cpu_op'))
            else:
                events.remove(bias)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                profile.row_backward(events, 'op3-r', 0)

    def test_splitk_must_belong_to_dgrad_and_complete_before_wgrad(self):
        for mutation in ('unowned_splitk', 'late_splitk'):
            events = backward_fixture(splitk=True)
            reduction = find(events, 'splitKreduce_kernel')
            if mutation == 'unowned_splitk':
                launch = next(e for e in events if e['cat'] == 'cuda_runtime'
                              and e['args']['correlation'] == reduction['args']['correlation'])
                launch['ts'] += 20
            else:
                reduction['ts'] = find(events, 'gemm_wgrad')['ts']+1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                profile.row_backward(events, 'op3-r', 0)


if __name__ == '__main__':
    unittest.main()

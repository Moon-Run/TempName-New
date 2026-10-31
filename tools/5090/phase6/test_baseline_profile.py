"""Synthetic trace contracts only: these fixtures are not GPU measurements."""
import copy
import unittest

from rtx5090_loader import load

p = load('protocol')
profile = load('profile')

GEMM = ('cutlass::Kernel2<flux_bf16_bf16_void_bf16_fp32_fp32_sm80_a100_'
        'reducescatter_rcr_gemmv2_false_false_intranode>')
TE_NORM = ('void transformer_engine::normalization::ln_fwd_tuned_kernel<'
           'transformer_engine::normalization::Kernel_traits<__nv_bfloat16, '
           '__nv_bfloat16, __nv_bfloat16, float, unsigned int, 4096u, '
           '1u, 1u, 4u, 16u, transformer_engine::normalization::Kernel_traits_base<'
           '4096u, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, unsigned int, '
           '128u> > >(transformer_engine::normalization::ForwardKernelParams)')


def fixture(arm='ma11', memcpy=False, multiple_nccl=False, te_final=False):
    events = []
    correlation = 0
    fused = arm in p.CR3_FAMILY
    decoder = ('taco_ma4f_decode_consumer_kernel(...)' if fused else
               'taco_ma2_decode_kernel(...)' if arm == 'r' else
               'taco_decode_ring_kernel<false>(...)' if arm == 'c' else
               'taco_decode_ring_kernel(...)')

    def host(cat, name, start, duration, external=None):
        value = dict(cat=cat, name=name, ph='X', ts=start, dur=duration, tid=1, pid=100, args={})
        if external is not None:
            value['args']['External id'] = external
        events.append(value)
        return value

    def device(cpu, gpu, name, duration=1, stream=7, external=None, cat='kernel', **args):
        nonlocal correlation
        correlation += 1
        launch = host('cuda_runtime', 'cudaMemcpyAsync' if cat == 'gpu_memcpy' else 'cudaLaunchKernel', cpu, .1)
        launch['args']['correlation'] = correlation
        value = dict(cat=cat, name=name, ph='X', ts=gpu, dur=duration, tid=stream, pid=200,
                     args=dict(correlation=correlation, stream=stream, device=0, **args))
        if external is not None:
            value['args']['External id'] = external
        events.append(value)
        return value

    for number in range(256):
        cpu, gpu = number*100, 1000000+number*100
        role = 'attn' if number < 128 else 'mlp'
        host('user_annotation', f'scope.{role}.candidate', cpu, 60)
        host('cpu_op', 'ConsumerGemmRSFunction' if fused else 'GemmRSFunction', cpu+1, 55, number)
        if arm == 'a':
            host('user_annotation', profile.CONFIGURE, cpu+2, 1)
        host('user_annotation', profile.PRODUCE, cpu+5, 10)
        device(cpu+6, gpu, GEMM, duration=4, external=number)
        device(cpu+12, gpu+6, profile.PACK_KERNEL+'(...)', external=number)
        host('user_annotation', profile.EXCHANGE, cpu+20, 15)
        host('cpu_op', 'c10d::alltoall_base_', cpu+21, 6, 10000+number)
        device(cpu+22, gpu+10, 'ncclDevKernel_SendRecv', duration=2 if multiple_nccl else 5,
               stream=26, external=10000+number)
        if multiple_nccl:
            device(cpu+23, gpu+13, 'ncclDevKernel_SendRecv', duration=2,
                   stream=26, external=10000+number)
        if memcpy:
            device(cpu+24, gpu+11, 'Memcpy DtoH (Device -> Pinned)', duration=1,
                   stream=26, external=10000+number, cat='gpu_memcpy', bytes=4096)
            device(cpu+25, gpu+13, 'Memcpy HtoD (Pinned -> Device)', duration=1,
                   stream=26, external=10000+number, cat='gpu_memcpy', bytes=4096)
        host('user_annotation', profile.CONSUMER if fused else profile.DECODE, cpu+40, 10)
        device(cpu+42, gpu+16, profile.UNPACK_KERNEL+'(...)', external=number)
        device(cpu+45, gpu+18, decoder, duration=2, external=number)
        next_gpu = gpu+21
        if arm == 'a':
            host('cpu_op', 'aten::clone', cpu+51, 4, 20000+number)
            device(cpu+52, next_gpu, 'Memcpy DtoD (Device -> Device)', external=20000+number,
                   cat='gpu_memcpy', bytes=256*4096*2)
            next_gpu += 2
        if not fused:
            for offset in range(2):
                external = 30000+number*2+offset
                host('cpu_op', 'aten::add', cpu+63+offset*4, 2, external)
                device(cpu+64+offset*4, next_gpu, 'vectorized_elementwise_kernel<AddFunctor<c10::BFloat16>>',
                       external=external)
                next_gpu += 2
        final_norm = te_final and role == 'mlp' and (number+1) % 32 == 0
        host('cpu_op', '_OperationFuserAutogradFunction' if final_norm else
             'FusedLayerNormAffineFunction', cpu+71, 4, 40000+number)
        device(cpu+72, next_gpu, TE_NORM if final_norm else 'cuApplyLayerNorm<c10::BFloat16>',
               duration=2, external=40000+number)
    return events


def find(events, token, cat='kernel'):
    return next(event for event in events if event['cat'] == cat and token in event['name'])


class StagedProfiles(unittest.TestCase):
    def test_5090_ma89_profile_accepts_pytorch28_warp_forward(self):
        from test_ma11_profile import fixture as attention_fixture
        old = attention_fixture()
        events = copy.deepcopy(old)
        for event in events:
            if event.get('cat') == 'kernel' and 'cunn_SoftMaxForwardSmem' in event['name']:
                event['name'] = 'softmax_warp_forward<c10::BFloat16, c10::BFloat16, float, 11>'
        result = profile.ma11_profile.trace_attention_softmax_records(events, 'F4-cache-zcount')
        self.assertEqual(result['softmax_compute_kernel_calls'], 640)
        with self.assertRaisesRegex(ValueError, 'Original BF16 softmax'):
            profile.ma11_profile.trace_attention_softmax_records(old, 'F4-cache-zcount')
        # The isolated adaptation must not modify the original H100 audit.
        self.assertEqual(profile.h100_ma11_profile.trace_attention_softmax_records(
            old, 'F4-cache-zcount')['softmax_compute_kernel_calls'], 640)

    def test_all_eight_staged_arms_have_real_kernel_and_consumer_paths(self):
        for arm in p.ARMS[:-1]:
            with self.subTest(arm=arm):
                result = profile.staged_projection_records(fixture(arm), arm, 0)
                self.assertEqual(set(result), {'attn', 'mlp'})
                for role in result.values():
                    self.assertEqual(role['projection_calls'], 128)
                    self.assertEqual(role['packet_pack_kernels'], 128)
                    self.assertEqual(role['packet_unpack_kernels'], 128)
                    self.assertEqual(role['native_ipc_barriers'], 0)
                    self.assertEqual(role['consumer_add_kernels'], 0 if arm in p.CR3_FAMILY else 256)
                    self.assertEqual(role['output_clones'], 128 if arm == 'a' else 0)

    def test_nccl_internal_copies_and_multiple_device_kernels_are_allowed(self):
        result = profile.staged_projection_records(fixture(memcpy=True, multiple_nccl=True), 'ma11', 0)
        self.assertEqual(result['attn']['nccl_memcpy_events'], 256)
        self.assertEqual(result['attn']['nccl_kernels'], 256)

    def test_te_final_layernorm_preserves_all_eight_consumer_paths(self):
        for arm in p.ARMS[:-1]:
            with self.subTest(arm=arm):
                result = profile.staged_projection_records(fixture(arm, te_final=True), arm, 0)
                self.assertEqual(result['attn']['layernorm_backends'], {'apex': 128})
                self.assertEqual(result['mlp']['layernorm_backends'],
                                 {'apex': 124, 'transformer_engine': 4})

    def test_te_final_layernorm_requires_bf16_4096_forward_kernel(self):
        for old, new in [('ln_fwd_tuned_kernel', 'ln_bwd_tuned_kernel'),
                         ('ln_fwd_tuned_kernel', 'rmsnorm_fwd_tuned_kernel'),
                         ('__nv_bfloat16', 'float'), ('4096u', '2048u')]:
            with self.subTest(mutation=new):
                events = fixture(te_final=True)
                norm = find(events, 'ln_fwd_tuned_kernel')
                norm['name'] = norm['name'].replace(old, new)
                with self.assertRaisesRegex(ValueError, 'immediately feed LayerNorm'):
                    profile.staged_projection_records(events, 'ma11', 0)

    def test_te_layernorm_cannot_replace_an_inner_norm(self):
        for role in ('attn', 'mlp'):
            events = fixture(te_final=True)
            norms = [event for event in events if event['cat'] == 'kernel' and
                     'cuApplyLayerNorm' in event['name']]
            norms[0 if role == 'attn' else 128]['name'] = TE_NORM
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'final MLP'):
                profile.staged_projection_records(events, 'ma11', 0)

    def test_te_final_layernorm_requires_exact_owner_and_launch(self):
        for mutation in ('name', 'external', 'launch', 'extra_work'):
            events = fixture(te_final=True)
            norm = find(events, 'ln_fwd_tuned_kernel')
            owner = find(events, '_OperationFuserAutogradFunction', 'cpu_op')
            if mutation == 'name':
                owner['name'] = 'FusedLayerNormAffineFunction'
            elif mutation == 'external':
                norm['args']['External id'] += 1
            elif mutation == 'launch':
                launch = next(event for event in events if event['cat'] == 'cuda_runtime'
                              and event['args']['correlation'] == norm['args']['correlation'])
                launch['ts'] = owner['ts'] + owner['dur'] + 1
            else:
                extra = copy.deepcopy(norm)
                extra.update(name='unexpected_fused_work', ts=norm['ts']+3)
                events.append(extra)
            message = 'unexpected device work' if mutation == 'extra_work' else 'CPU/launch association'
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                profile.staged_projection_records(events, 'ma11', 0)

    def test_missing_marker_or_pack_rejected(self):
        for token, cat in ((profile.PRODUCE, 'user_annotation'), (profile.EXCHANGE, 'user_annotation'),
                           (profile.CONSUMER, 'user_annotation'), (profile.PACK_KERNEL, 'kernel')):
            events = fixture()
            events.remove(find(events, token, cat))
            with self.subTest(token=token), self.assertRaises(ValueError):
                profile.staged_projection_records(events, 'ma11', 0)

    def test_wrong_decoder_and_legacy_ipc_rejected(self):
        events = fixture()
        find(events, 'taco_ma4f_decode')['name'] = 'taco_ma2c_decode_kernel(...)'
        with self.assertRaisesRegex(ValueError, 'Wrong original decoder'):
            profile.staged_projection_records(events, 'ma11', 0)
        events = fixture()
        find(events, 'nccl')['name'] = 'CudaIpcBarrierAllKernel'
        with self.assertRaisesRegex(ValueError, 'Legacy CUDA IPC'):
            profile.staged_projection_records(events, 'ma11', 0)
        events = fixture('c')
        find(events, 'taco_decode_ring')['name'] = 'taco_decode_ring_kernel(...)'
        with self.assertRaisesRegex(ValueError, 'Wrong original decoder'):
            profile.staged_projection_records(events, 'c', 0)

    def test_cpu_annotations_without_launch_ownership_do_not_pass(self):
        events = fixture()
        kernel = find(events, profile.PACK_KERNEL)
        launch = next(event for event in events if event['cat'] == 'cuda_runtime'
                      and event['args']['correlation'] == kernel['args']['correlation'])
        launch['ts'] = 18  # Inside outer projection, outside the producer stage.
        with self.assertRaisesRegex(ValueError, 'pack kernel'):
            profile.staged_projection_records(events, 'ma11', 0)
        events = fixture()
        find(events, 'c10d::alltoall', 'cpu_op')['name'] = 'unrelated_cpu_op'
        with self.assertRaisesRegex(ValueError, 'All-to-All submission'):
            profile.staged_projection_records(events, 'ma11', 0)

    def test_early_exchange_or_decode_wrong_stream_and_device_rejected(self):
        mutations = [lambda events: find(events, 'nccl').update(ts=1000005),
                     lambda events: find(events, profile.UNPACK_KERNEL).update(ts=1000014),
                     lambda events: find(events, profile.UNPACK_KERNEL)['args'].update(stream=8),
                     lambda events: find(events, profile.PACK_KERNEL)['args'].update(device=1)]
        for mutation in mutations:
            events = fixture()
            mutation(events)
            with self.assertRaises(ValueError):
                profile.staged_projection_records(events, 'ma11', 0)

    def test_model_consumer_must_be_bf16_and_have_real_cpu_association(self):
        events = fixture('r')
        find(events, 'AddFunctor')['name'] = 'vectorized_elementwise_kernel<AddFunctor<float>>'
        with self.assertRaisesRegex(ValueError, 'BF16'):
            profile.staged_projection_records(events, 'r', 0)
        events = fixture()
        find(events, 'FusedLayerNormAffineFunction', 'cpu_op')['name'] = 'unrelated_cpu_op'
        with self.assertRaisesRegex(ValueError, 'CPU/launch association'):
            profile.staged_projection_records(events, 'ma11', 0)
        events = fixture()
        kernel = find(events, 'cuApplyLayerNorm')
        launch = next(event for event in events if event['cat'] == 'cuda_runtime'
                      and event['args']['correlation'] == kernel['args']['correlation'])
        launch['ts'] = 76
        with self.assertRaisesRegex(ValueError, 'CPU/launch association'):
            profile.staged_projection_records(events, 'ma11', 0)

    def test_a_clone_and_reconfigure_are_part_of_registered_path(self):
        events = fixture('a')
        events.remove(find(events, 'aten::clone', 'cpu_op'))
        with self.assertRaisesRegex(ValueError, 'clone count'):
            profile.staged_projection_records(events, 'a', 0)
        events = fixture('a')
        events.remove(find(events, profile.CONFIGURE, 'user_annotation'))
        with self.assertRaisesRegex(ValueError, 'markers'):
            profile.staged_projection_records(events, 'a', 0)

    def test_extra_cuda_work_and_duplicate_projection_rejected(self):
        events = fixture()
        event = copy.deepcopy(find(events, profile.PACK_KERNEL))
        event['name'] = 'unexpected_fallback_kernel'
        events.append(event)
        with self.assertRaisesRegex(ValueError, 'Unexpected producer kernel'):
            profile.staged_projection_records(events, 'ma11', 0)
        events = fixture()
        events.append(copy.deepcopy(find(events, 'scope.attn.candidate', 'user_annotation')))
        with self.assertRaisesRegex(ValueError, 'both 32-layer'):
            profile.staged_projection_records(events, 'ma11', 0)


if __name__ == '__main__':
    unittest.main()

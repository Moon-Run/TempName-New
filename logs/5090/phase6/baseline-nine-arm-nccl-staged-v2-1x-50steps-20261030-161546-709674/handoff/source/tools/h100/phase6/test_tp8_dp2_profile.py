"""CPU trace fixtures; they test admission logic, never certify GPU execution."""
import unittest
import tp8_dp2_profile as profile


def dp_fixture(kind='allreduce_coalesced'):
    return [
        dict(cat='cpu_op',name='record_param_comms',ts=1000000,dur=100,tid=1,args={
            'Collective name':kind,'Group size':2,'In msg nelems':849207296,'Out msg nelems':849207296,
            'dtype':'Float','Process Group Description':'DATA_PARALLEL_GROUP_WITH_CP'}),
        dict(cat='cuda_runtime',name='cudaLaunchKernel',ts=1000010,dur=1,tid=1,args={'correlation':900000}),
        dict(cat='kernel',name='ncclDevKernel_AllReduce_Sum_f32',ts=1000200,dur=10,tid=7,
             args={'correlation':900000,'stream':7})]


def projection_fixture(arm):
    events=[]
    for number in range(128):
        role='attn' if number<64 else 'mlp'
        route='flux_tensor' if arm=='flux-basic' else 'tensor'
        base=number*1000
        events.append(dict(cat='user_annotation',ph='X',name=f'scope.{role}.{route}',ts=base,dur=200,tid=1))
        markers=['megatron_taco.encode','megatron_taco.exchange','megatron_taco.decode_reduce']
        if arm=='flux-basic': markers.append('ablation6.flux_gemm_only')
        for name in markers:
            events.append(dict(cat='user_annotation',ph='X',name=name,ts=base+2,dur=180,tid=1))
        events.append(dict(cat='cpu_op',name='c10d::alltoall_base_',ts=base+70,dur=10,tid=1))
        if arm=='megatron-cuda':
            events += [dict(cat='cpu_op',name='aten::add',ts=base+120+i*2,dur=1,tid=1) for i in range(8)]
        for i,name in enumerate(['flux_bf16_sm90_commnone' if arm=='flux-basic' else 'sm90_gemm',
                                 'fused_ash_compress_kernel','ncclDevKernel_SendRecv',
                                 'basic_quant_decode_reduce_kernel' if arm=='flux-basic' else 'fused_ash_decompress_kernel']):
            correlation=number*10+i+1
            events.append(dict(cat='cuda_runtime',name='cudaLaunchKernel',ts=base+5+i*20,dur=1,tid=1,
                               args={'correlation':correlation}))
            events.append(dict(cat='kernel',name=name,ts=base+300+i*20,dur=2,tid=7,
                               args={'correlation':correlation,'stream':7}))
    return events


class DPProfileTests(unittest.TestCase):
    def test_regular_and_pytorch26_coalesced_collectives(self):
        for name in ('allreduce','allreduce_coalesced'):
            result=profile.dp_gradient_records(dp_fixture(name))
            self.assertEqual(result['dp_collective_names'],{name:1})
            self.assertEqual(result['dp_gradient_elements'],849207296)

    def test_metadata_alone_or_unowned_gpu_kernel_cannot_pass(self):
        for change in ('no_gpu','outside_launch','wrong_collective'):
            events=dp_fixture()
            if change=='no_gpu': events.pop()
            elif change=='outside_launch': events[1]['ts']=1000500
            else: events[2]['name']='ncclDevKernel_ReduceScatter'
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'correlated NCCL'):
                profile.dp_gradient_records(events)

    def test_scalar_tp_or_wrong_group_is_not_dp_gradient_evidence(self):
        for key,value in [('Group size',8),('In msg nelems',1),('Out msg nelems',1),
                          ('dtype','Long'),('Process Group Description','MODEL_PARALLEL_GROUP'),
                          ('Collective name','reduce_scatter')]:
            events=dp_fixture();events[0]['args'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): profile.dp_gradient_records(events)

    def test_both_baseline_profiles_cover_two_microbatches(self):
        for arm in ('flux-basic','megatron-cuda'):
            result=profile.analyze(dp_fixture()+projection_fixture(arm),arm,8)
            self.assertEqual(result['dp_collective_names'],{'allreduce_coalesced':1})
            self.assertEqual(result['roles']['attn']['calls'],64)
            self.assertEqual(result['roles']['mlp']['separate_bf16_adds'],0 if arm=='flux-basic' else 512)

    def test_missing_projection_and_wrong_receiver_are_rejected(self):
        for arm in ('flux-basic','megatron-cuda'):
            for change in ('scope','receiver','clone'):
                events=projection_fixture(arm)
                if change=='scope': events.pop(0)
                elif change=='receiver':
                    kernel=next(e for e in events if e['cat']=='kernel' and
                                ('decode' in e['name'] or 'decompress' in e['name']))
                    kernel['name']='wrong_receiver'
                else: events.append(dict(cat='cpu_op',name='aten::clone',ts=10,dur=1,tid=1))
                with self.subTest(arm=arm,change=change),self.assertRaises(ValueError):
                    profile.analyze(dp_fixture()+events,arm,0)


if __name__=='__main__': unittest.main()

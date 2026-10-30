"""Independent negative cases for MA4 graph execution-path evidence."""
import copy
import unittest

import ma4_g_bench as bench


def replay_fixture():
    call = dict(cat='cpu_op', name='GemmRSFunction', pid=1, tid=2,
                ts=10., dur=10., args={'External id': 50})
    events = [call]
    for offset, name in ((1., 'cudaGraphExecKernelNodeSetParams'),
                         (2., 'cudaGraphExecKernelNodeSetParams'),
                         (3., 'cudaGraphLaunch')):
        events.append(dict(cat='cuda_runtime', name=name, pid=1, tid=2,
                           ts=10.+offset, dur=.1, args={'correlation': 70}))
    for index, (category, name) in enumerate((
            ('gpu_memset', 'Memset (Device)'),
            ('kernel', 'flux_bf16_reducescatter'),
            ('kernel', 'CudaIpcBarrierAllKernel'),
            ('kernel', 'taco_ma2c_decode_kernel'))):
        events.append(dict(cat=category, name=name, pid=8, tid=7,
                           ts=30.+index, dur=.5,
                           args={'correlation': 70, 'stream': 7}))
    return events, call


class IndependentGraphProfileReview(unittest.TestCase):
    def test_gpu_queue_delay_does_not_require_host_gpu_time_overlap(self):
        events, call = replay_fixture()
        paths = bench.graph_call_paths(events, [call])
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(paths[0]), 4)

    def test_unrelated_correlation_cannot_hide_copy_in_projection(self):
        for api in ('cudaMemcpyAsync', 'cuMemcpyDtoDAsync_v2'):
            with self.subTest(api=api):
                events, call = replay_fixture()
                events.extend([
                    dict(cat='cuda_runtime', name=api, pid=1, tid=2,
                         ts=14., dur=.1, args={'correlation': 99}),
                    dict(cat='gpu_memcpy', name='Memcpy DtoD', pid=8, tid=7,
                         ts=34., dur=.5, args={'correlation': 99, 'stream': 7}),
                ])
                with self.assertRaises(ValueError):
                    bench.graph_call_paths(events, [call])

    def test_parameter_update_after_launch_is_rejected(self):
        events, call = replay_fixture()
        events[1]['ts'] = 14.
        with self.assertRaises(ValueError):
            bench.graph_call_paths(events, [call])

    def test_parameter_update_overlapping_launch_is_rejected(self):
        events, call = replay_fixture()
        events[2]['dur'] = 2.
        with self.assertRaises(ValueError):
            bench.graph_call_paths(events, [call])

    def test_missing_association_does_not_fall_back_to_time_proximity(self):
        events, call = replay_fixture()
        for event in events:
            if event['cat'] in ('kernel', 'gpu_memset'):
                event['args'].pop('correlation')
        with self.assertRaises(ValueError):
            bench.graph_call_paths(events, [call])

    def test_same_nodes_cannot_count_as_two_replays(self):
        events, call = replay_fixture()
        with self.assertRaises(ValueError):
            bench.graph_call_paths(events, [call, copy.deepcopy(call)])


if __name__ == '__main__':
    unittest.main()

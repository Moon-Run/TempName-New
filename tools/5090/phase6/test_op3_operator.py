"""Operator evidence corruption, distributed statistics and actual timing lifecycle."""
from contextlib import ExitStack
import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import op3_operator as op
from test_op3_validation import state
from test_op3_profile import forward_fixture


def fixture():
    calls, fused = op.expected_calls()
    plan = dict(run_id='test', build={'manifest_sha256': 'b'*64})
    hardware = dict(devices=[{'uuid': f'GPU-{rank}'} for rank in range(8)],
                    environment=dict(torch='test-torch', cuda='test-cuda', nccl=[2, 27, 3]))
    rows = []
    for rank in range(8):
        roles = {}
        for role, k in op.ROLES.items():
            checks = []
            for step in range(op.SETTINGS['gradient_steps']):
                equality = dict(bitwise=True, actual_sha256='a'*64, reference_sha256='a'*64)
                checks.append(dict(step=step, output_bitwise=True, inputs_unchanged=True,
                    output_sha256='a'*64, reference_output_sha256='a'*64,
                    gradients={name: dict(equality) for name in op.PARAMETERS},
                    updates={name: dict(equality) for name in op.PARAMETERS}))
            roles[role] = dict(shape=[2048,4096,k], gradient_checks=checks,
                state=state(op.p.COMBINED, rank, calls, fused_calls=fused),
                windows=[dict(index=i, iterations=op.SETTINGS['iterations'],
                              gpu_us=100+rank+i, wall_us=120+rank+i)
                         for i in range(op.SETTINGS['windows'])])
        rows.append(dict(protocol=op.PROTOCOL, arm=op.p.COMBINED, rank=rank, completed=True,
            settings=dict(op.SETTINGS), roles=roles, run_id='test', entrypoint_sha256=op.sha(op.__file__),
            build_manifest_sha256='b'*64, gpu_uuid=f'GPU-{rank}', libraries={}, codec_library={},
            stream=100, default_stream=0, **hardware['environment']))
    return rows, plan, hardware


class Evidence(unittest.TestCase):
    def test_full_gradient_coverage_and_combined_call_count(self):
        rows, plan, hardware = fixture()
        self.assertEqual(op.expected_calls(), (136, 133))
        self.assertEqual(len(op.validate_rows(rows, plan, hardware, {}, {})), 8)
        for mutate in (
            lambda r: r['roles']['attn']['gradient_checks'][0]['gradients']['bias'].update(bitwise=False),
            lambda r: r['roles']['mlp']['gradient_checks'][2]['updates']['weight'].update(actual_sha256='f'*64),
            lambda r: r['roles']['attn']['gradient_checks'][0]['gradients'].pop('residual'),
            lambda r: r['roles']['attn']['state'].update(encoder='upstream-cta128'),
            lambda r: r['roles']['attn']['state'].update(fused_consumer_calls=132),
            lambda r: r['roles']['attn']['windows'].pop(),
            lambda r: r['roles']['attn']['windows'][0].update(gpu_us=float('nan')),
            lambda r: r.update(stream=99),
            lambda r: r.update(arm='op3-r'),
            lambda r: r.update(gpu_uuid='wrong'),
        ):
            damaged = copy.deepcopy(rows)
            mutate(damaged[0])
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, KeyError)):
                op.validate_rows(damaged, plan, hardware, {}, {})

    def test_statistics_use_slowest_rank_per_window_and_preserve_every_window(self):
        rows, _, _ = fixture()
        keys = ('gemm_us', 'encode_us', 'nccl_us', 'decode_us', 'cpu_us', 'gpu_span_us')
        profiles = {f'{role}/rank{rank}': {'forward': {'median_stage_us': dict.fromkeys(keys, rank+1)}}
                    for role in op.ROLES for rank in range(8)}
        result = op.summarize(rows, profiles)
        self.assertEqual(result['roles']['attn']['gpu_us']['mean'], 107)
        self.assertEqual(result['roles']['attn']['profile_median_us']['encode_us'], 4.5)
        self.assertEqual(len(result['roles']['mlp']['all_windows']), 1)
        self.assertFalse(result['full_step_measured'])
        self.assertFalse(result['speedup_evaluated'])
        self.assertEqual(result['timed_arms'], ['op3-combined'])
        self.assertIn('解码含融合', op.markdown(result))

    def test_operator_audit_requires_warp_encoder_fused_r_and_completed_exchange(self):
        # Keep real launch/correlation relationships from the projection fixture;
        # operator execution ends at the decoder, with no fake LayerNorm demand.
        events = forward_fixture(op.p.COMBINED)
        result = op.profile.chunk_records(events, op.p.COMBINED, 0, operator=True)
        self.assertEqual(result['attn']['consumer_add_kernels'], 0)
        self.assertEqual(result['mlp']['median_stage_us']['encode_us'], 1)
        short = [e for e in events if e['ts'] < 3000 or 1e6 <= e['ts'] < 1e6+3000]
        short = [e for e in short if e['name'] not in ('cuApplyLayerNorm<c10::BFloat16>', 'FusedLayerNormAffineFunction')]
        checked = op.analyze(short, 'attn', 0)
        self.assertEqual(checked['forward']['calls'], 10)
        with self.assertRaises(ValueError):
            op.analyze(short[:-1], 'attn', 0)
        for token, replacement in (('op3_warp_encode_kernel', 'op1_abc_encode_kernel'),
                                   ('op3_r_decode_kernel<true>', 'op3_r_decode_kernel<false>')):
            damaged = copy.deepcopy(events)
            next(e for e in damaged if token in e['name'])['name'] = replacement
            with self.assertRaises(ValueError):
                op.profile.chunk_records(damaged, op.p.COMBINED, 0, operator=True)


class TimingLifecycle(unittest.TestCase):
    def test_actual_timed_loop_syncs_each_window_and_excludes_warmup(self):
        events, calls = [], [0]
        class Event:
            def record(self, stream):
                events.append(('event', calls[0], stream))
            def synchronize(self):
                events.append(('event_sync', calls[0]))
            def elapsed_time(self, other):
                return 5.0
        torch = SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kwargs: Event(),
            synchronize=lambda: events.append(('device_sync', calls[0]))))
        dist = SimpleNamespace(barrier=lambda **kwargs: events.append(('barrier', calls[0])))
        def forward():
            calls[0] += 1
            return object()
        ticks = iter(float(i) for i in range(10))
        with patch.object(op.time, 'perf_counter', side_effect=lambda: next(ticks)):
            windows = op.measure_forward(torch, dist, forward, 'stream', 3)
        self.assertEqual(calls[0], 120)
        self.assertEqual([e[1] for e in events if e[0] == 'barrier'], [20])
        self.assertEqual([e[1] for e in events if e[0] == 'event_sync'], [120])
        self.assertTrue(all(w['iterations'] == 100 and w['gpu_us'] == 50 and w['wall_us'] == 10000
                            for w in windows))
        for i in range(1):
            self.assertEqual([e[0] for e in events[i*5:(i+1)*5]],
                             ['barrier','device_sync','event','event','event_sync'])


if __name__ == '__main__':
    unittest.main()

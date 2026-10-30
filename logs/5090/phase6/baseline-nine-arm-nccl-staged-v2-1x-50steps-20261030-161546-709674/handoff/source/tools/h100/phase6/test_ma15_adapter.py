"""CPU checks of graph ownership, failure handling and autograd smoke oracles.

No GPU graph is executed here. CUDA capture and the exact MA11 kernels are
validated by the H100 launcher's mandatory full-model smoke/profile stages.
"""
import copy
from contextlib import contextmanager
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import ma15_adapter as adapter


def completed_state(steps=2, mode='smoke'):
    module = dict(initializations=1, phase='idle', forward_replays=steps * 4,
                  backward_replays=steps * 4, forward_audits=steps * 4 if mode == 'smoke' else 0,
                  backward_audits=steps * 4 if mode == 'smoke' else 0, rng_unchanged=True,
                  private_graph_pair=True, owned_outputs=True,
                  inputs=[dict(shape=list(adapter.QKV_SHAPE), strides=[1536, 1536, 384, 1],
                               dtype='torch.bfloat16')] * 3)
    return dict(protocol=adapter.PROTOCOL, installed=True, original_core_sha256=adapter.CORE_SHA256,
                scope='local_attention_only', communication_captured=False, optimizer_captured=False,
                profiler_setup=(dict(primed=True, sessions=1, before_capture=True,
                                     environment=dict(adapter.PROFILER_ENVIRONMENT))
                                if mode == 'profile' else {'primed': False}),
                modules={name: copy.deepcopy(module) for name in adapter.MODULES})


class ProfilerSetupTests(unittest.TestCase):
    def test_smoke_and_timing_do_not_touch_profiler_or_environment(self):
        with patch.dict(os.environ, TEARDOWN_CUPTI='1', DISABLE_CUPTI_LAZY_REINIT='0'):
            expected = dict(os.environ)
            for mode in ('smoke', 'timing'):
                self.assertEqual(adapter.prime_profiler(object(), mode), {'primed': False})
            self.assertEqual(dict(os.environ), expected)

    def test_profile_attaches_once_and_keeps_cupti_alive_for_later_capture(self):
        events = []

        @contextmanager
        def profile(*, activities):
            self.assertEqual(activities, ('CPU', 'CUDA'))
            self.assertEqual({key: os.environ[key] for key in adapter.PROFILER_ENVIRONMENT},
                             adapter.PROFILER_ENVIRONMENT)
            events.append('attach')
            yield
            events.append('stop_collection_without_teardown')

        fake = SimpleNamespace(
            profiler=SimpleNamespace(ProfilerActivity=SimpleNamespace(CPU='CPU', CUDA='CUDA'),
                                     supported_activities=lambda: {'CPU', 'CUDA'}, profile=profile),
            int64='int64',
            empty=lambda *a, **kw: SimpleNamespace(zero_=lambda: events.append('zero')),
            cuda=SimpleNamespace(synchronize=lambda: events.append('synchronize')))
        with patch.dict(os.environ, TEARDOWN_CUPTI='1', DISABLE_CUPTI_LAZY_REINIT='0'):
            state = adapter.prime_profiler(fake, 'profile')
            self.assertEqual(events, ['attach', 'zero', 'synchronize',
                                      'stop_collection_without_teardown'])
            self.assertEqual(state, completed_state(mode='profile')['profiler_setup'])
            self.assertEqual({key: os.environ[key] for key in adapter.PROFILER_ENVIRONMENT},
                             adapter.PROFILER_ENVIRONMENT)

    def test_missing_early_attach_and_timing_contamination_are_rejected(self):
        for mode in ('smoke', 'timing', 'profile'):
            state = completed_state(mode=mode)
            state['profiler_setup'] = ({'primed': False} if mode == 'profile' else
                                       completed_state(mode='profile')['profiler_setup'])
            with self.assertRaisesRegex(RuntimeError, 'CUPTI must attach'):
                adapter.validate_state(state, 2, mode)


class GuardTests(unittest.TestCase):
    def test_repeated_forward_backward_and_owned_identity(self):
        guard = adapter.InvocationGuard()
        for _ in range(8):
            guard.begin(0, 7, ('strided-qkv',))
            guard.finish_forward()
            guard.check_backward(0, 7)
            guard.finish_backward()
        self.assertEqual((guard.phase, guard.forward_replays, guard.backward_replays), ('idle', 8, 8))

    def test_second_forward_poisoned_before_reuse(self):
        guard = adapter.InvocationGuard()
        guard.begin(0, 7, ('layout',))
        with self.assertRaisesRegex(RuntimeError, 'one forward'):
            guard.begin(0, 7, ('layout',))
        self.assertEqual(guard.phase, 'failed')
        with self.assertRaises(RuntimeError):
            guard.check_backward(0, 7)

    def test_wrong_stream_device_or_layout_permanently_invalidates(self):
        for actual in ((1, 7, ('layout',)), (0, 8, ('layout',)), (0, 7, ('new-layout',))):
            with self.subTest(actual=actual):
                guard = adapter.InvocationGuard()
                guard.begin(0, 7, ('layout',))
                guard.finish_forward()
                guard.finish_backward()
                with self.assertRaises(RuntimeError):
                    guard.begin(*actual)
                self.assertEqual(guard.phase, 'failed')
                with self.assertRaises(RuntimeError):
                    guard.begin(0, 7, ('layout',))

    def test_backward_requires_matching_live_forward(self):
        for device, stream in ((1, 7), (0, 8)):
            guard = adapter.InvocationGuard()
            guard.begin(0, 7, ())
            with self.assertRaises(RuntimeError):
                guard.check_backward(device, stream)
            self.assertEqual(guard.phase, 'failed')
        with self.assertRaises(RuntimeError):
            adapter.InvocationGuard().check_backward(0, 7)

    def test_state_rejects_missing_audit_replay_rng_coverage_and_leaked_capture(self):
        good = completed_state()
        adapter.validate_state(good, 2, 'smoke')
        mutations = [
            ('initializations', 2), ('phase', 'live'), ('forward_replays', 7),
            ('backward_replays', 7), ('forward_audits', 0), ('backward_audits', 0),
            ('rng_unchanged', False), ('owned_outputs', False), ('private_graph_pair', False),
            ('inputs', [dict(shape=list(adapter.QKV_SHAPE), strides=[512, 512, 128, 1],
                             dtype='torch.bfloat16')] * 3),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                bad = copy.deepcopy(good)
                bad['modules'][adapter.MODULES[0]][key] = value
                with self.assertRaises(RuntimeError):
                    adapter.validate_state(bad, 2, 'smoke')
        for key in ('communication_captured', 'optimizer_captured'):
            bad = copy.deepcopy(good)
            bad[key] = True
            with self.assertRaises(RuntimeError):
                adapter.validate_state(bad, 2, 'smoke')
        bad = copy.deepcopy(good)
        bad['modules'].pop(adapter.MODULES[0])
        with self.assertRaises(RuntimeError):
            adapter.validate_state(bad, 2, 'smoke')

    def test_timing_and_profile_require_no_smoke_audits(self):
        for mode in ('timing', 'profile'):
            good = completed_state(60, mode)
            adapter.validate_state(good, 60, mode)
            good['modules'][adapter.MODULES[0]]['forward_audits'] = 1
            with self.assertRaises(RuntimeError):
                adapter.validate_state(good, 60, mode)


class CPUReplayOracle:
    """Replace only the CUDA graph transport; exercise real MA15 autograd code."""
    def __init__(self, mode='smoke', corrupt=None):
        self.mode, self.corrupt = mode, corrupt
        self.guard = adapter.InvocationGuard()
        self.forward_audits = self.backward_audits = self.eager_calls = 0

    def eager(self, query, key, value, mask):
        self.eager_calls += 1
        return query * key + value

    def forward(self, values, mask):
        self.guard.begin(0, 7, tuple(adapter.layout(x)[:4] for x in values))
        self.values = tuple(x.detach().clone() for x in values)
        self.guard.finish_forward()
        query, key, value = self.values
        result = query * key + value
        return result + 1 if self.corrupt == 'forward' else result

    def backward(self, gradient):
        self.guard.check_backward(0, 7)
        query, key, _ = self.values
        values = (gradient * key, gradient * query, gradient.clone())
        return tuple(x + 1 for x in values) if self.corrupt == 'backward' else values


class AutogradTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def values(self):
        # Noncontiguous inputs exercise the oracle without silently normalizing
        # layouts. Small exact integers keep this CPU test independent of CUDA.
        values = tuple((torch.arange(48, dtype=torch.float32) + i).reshape(4, 12)[:, ::3]
                       .detach().requires_grad_() for i in range(3))
        self.assertTrue(all(not value.is_contiguous() and value.stride() == (12, 3)
                            for value in values))
        return values

    def test_smoke_forward_and_three_gradients_match_eager(self):
        slot = CPUReplayOracle()
        function = adapter.make_function(torch, slot)
        outputs = []
        for _ in range(8):
            values = self.values()
            output = function.apply(*values, torch.zeros(1, dtype=torch.bool))
            saved = output.detach().clone()
            gradients = torch.autograd.grad(output, values, torch.ones_like(output))
            self.assertTrue(torch.equal(gradients[0], values[1]))
            self.assertTrue(torch.equal(gradients[1], values[0]))
            self.assertTrue(torch.equal(gradients[2], torch.ones_like(values[2])))
            outputs.append((output, saved))
        self.assertTrue(all(torch.equal(a, b) for a, b in outputs))
        self.assertEqual((slot.guard.forward_replays, slot.guard.backward_replays,
                          slot.forward_audits, slot.backward_audits, slot.eager_calls), (8, 8, 8, 8, 8))

    def test_timing_runs_no_eager_reference_or_marker(self):
        slot = CPUReplayOracle('timing')
        function = adapter.make_function(torch, slot)
        values = self.values()
        with patch.object(torch.autograd.profiler, 'record_function',
                          side_effect=AssertionError('marker leaked')):
            function.apply(*values, torch.zeros(1, dtype=torch.bool)).sum().backward()
        self.assertEqual((slot.eager_calls, slot.forward_audits, slot.backward_audits), (0, 0, 0))
        self.assertEqual(slot.guard.phase, 'idle')

    def test_smoke_rejects_corrupt_output_and_gradient_and_poison_slot(self):
        for corrupt in ('forward', 'backward'):
            slot = CPUReplayOracle(corrupt=corrupt)
            function = adapter.make_function(torch, slot)
            with self.assertRaisesRegex(RuntimeError, 'differ.*from eager'):
                function.apply(*self.values(), torch.zeros(1, dtype=torch.bool)).sum().backward()
            self.assertEqual(slot.guard.phase, 'failed')

    def test_original_input_version_checks_are_retained(self):
        slot = CPUReplayOracle('timing')
        function = adapter.make_function(torch, slot)
        values = self.values()
        output = function.apply(*values, torch.zeros(1, dtype=torch.bool))
        with torch.no_grad():
            values[0].add_(1)
        with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
            output.sum().backward()
        self.assertEqual(slot.guard.phase, 'failed')

    def test_two_live_autograd_contexts_cannot_share_graph_storage(self):
        slot = CPUReplayOracle('timing')
        function = adapter.make_function(torch, slot)
        values = self.values()
        function.apply(*values, torch.zeros(1, dtype=torch.bool))
        with self.assertRaisesRegex(RuntimeError, 'one forward'):
            function.apply(*values, torch.zeros(1, dtype=torch.bool))
        self.assertEqual(slot.guard.phase, 'failed')


if __name__ == '__main__':
    unittest.main()

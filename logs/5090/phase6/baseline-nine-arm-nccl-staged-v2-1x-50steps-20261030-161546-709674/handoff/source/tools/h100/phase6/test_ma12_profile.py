"""Reject fallback, premature reuse, missing launches and changed successors."""
import copy
import unittest

import ma12_profile as profile


def fixture(candidate=True, full=False):
    events, correlation = [], 0

    def event(cat, name, ts, dur, **args):
        value = dict(cat=cat, name=name, ts=ts, dur=dur, pid=1, tid=1, args=args)
        events.append(value)
        return value

    def kernel(ts, name, grid, block, external=None):
        nonlocal correlation
        correlation += 1
        event('cuda_runtime', 'cudaLaunchKernel', ts, .1, correlation=correlation)
        args = dict(correlation=correlation, stream=7, grid=[grid, 1, 1], block=[block, 1, 1])
        if external is not None:
            args['External id'] = external
        return event('kernel', name, ts + 1000000, 1, **args)

    for i in range(256 if full else 4):
        t = i * 100
        role = 'attn' if i % 2 == 0 else 'mlp'
        if full:
            event('user_annotation', f'scope.{role}.candidate', t, 80)
            event('cpu_op', 'ConsumerGemmRSFunction', t + 1, 50, **{'External id': i + 1})
        else:
            event('user_annotation', f'ma4f.attn.{i}', t, 50)
        if candidate:
            kernel(t + 5, profile.WAIT, 1, 32)
        clear = kernel(t + 7, 'Memset (Device)', 1, 1)
        clear['cat'] = 'gpu_memset'
        clear['args']['bytes'] = 128
        kernel(t + 10, 'flux_bf16_gemm_reducescatter', 512, 128)
        if not candidate:
            kernel(t + 20, 'CudaIpcBarrierAllKernel', 1, 32)
        kernel(t + 30, profile.DECODER if candidate else profile.BASE_DECODER, 2048, 128)
        if candidate:
            kernel(t + 40, profile.DONE, 1, 32)
        if full:
            event('cpu_op', 'FusedLayerNormAffineFunction', t + 60, 10,
                  **{'External id': 1000 + i})
            kernel(t + 61, 'layernorm_forward', 1, 128, 1000 + i)
    return events


def find(events, token, cat='kernel'):
    return next(e for e in events if e['cat'] == cat and token in e['name'])


class ProfileTests(unittest.TestCase):
    def test_both_operator_paths(self):
        for candidate in (False, True):
            result = profile.trace_operator_path(fixture(candidate), profile.ARMS[int(candidate)], 'attn')
            self.assertEqual(result['publication_barriers_per_call'], int(not candidate))

    def test_both_full_projection_paths(self):
        for candidate in (False, True):
            result = profile.trace_forward_records(fixture(candidate, True), profile.ARMS[int(candidate)])
            for role in ('attn', 'mlp'):
                self.assertEqual(result[role]['projection_calls'], 128)
                self.assertEqual(result[role]['completion_publications'], 128 if candidate else 0)

    def test_candidate_cannot_run_old_barrier_path(self):
        with self.assertRaises(ValueError):
            profile.trace_operator_path(fixture(False), profile.ARMS[1], 'attn')

    def test_control_cannot_install_candidate(self):
        with self.assertRaises(ValueError):
            profile.trace_operator_path(fixture(True), profile.ARMS[0], 'attn')

    def test_missing_handoff(self):
        for token in (profile.WAIT, profile.DONE):
            events = fixture()
            events.remove(find(events, token))
            with self.assertRaisesRegex(ValueError, 'kernel count'):
                profile.trace_operator_path(events, profile.ARMS[1], 'attn')

    def test_done_cannot_precede_decoder(self):
        events = fixture()
        find(events, profile.DONE)['ts'] -= 15
        with self.assertRaises(ValueError):
            profile.trace_operator_path(events, profile.ARMS[1], 'attn')

    def test_overlap_or_other_stream_rejected(self):
        for field, value in [('stream', 8), ('dur', 100)]:
            events = fixture()
            decoder = find(events, profile.DECODER)
            (decoder['args'] if field == 'stream' else decoder)[field] = value
            with self.assertRaises(ValueError):
                profile.trace_operator_path(events, profile.ARMS[1], 'attn')

    def test_wrong_launch_geometry(self):
        events = fixture()
        find(events, profile.DECODER)['args']['block'] = [256, 1, 1]
        with self.assertRaisesRegex(ValueError, 'layout'):
            profile.trace_operator_path(events, profile.ARMS[1], 'attn')

    def test_flags_cannot_be_cleared_each_call(self):
        events = fixture()
        find(events, 'Memset', 'gpu_memset')['args']['bytes'] = 16388
        with self.assertRaisesRegex(ValueError, 'workspace clear'):
            profile.trace_operator_path(events, profile.ARMS[1], 'attn')

    def test_workspace_clear_cannot_run_after_producer(self):
        events = fixture()
        find(events, 'Memset', 'gpu_memset')['ts'] += 10
        with self.assertRaisesRegex(ValueError, 'clear moved'):
            profile.trace_operator_path(events, profile.ARMS[1], 'attn')

    def test_unowned_launch_and_duplicate_rejected(self):
        events = fixture()
        events.append(copy.deepcopy(find(events, profile.DONE)))
        with self.assertRaises(ValueError):
            profile.trace_operator_path(events, profile.ARMS[1], 'attn')
        events = fixture()
        find(events, 'cudaLaunchKernel', 'cuda_runtime')['ts'] = -1
        with self.assertRaises(ValueError):
            profile.trace_operator_path(events, profile.ARMS[1], 'attn')

    def test_missing_layernorm(self):
        events = fixture(True, True)
        find(events, 'layernorm_forward')['name'] = 'standalone_bias_add'
        with self.assertRaisesRegex(ValueError, 'LayerNorm'):
            profile.trace_forward_records(events, profile.ARMS[1])

    def test_layernorm_must_have_original_owner(self):
        events = fixture(True, True)
        find(events, 'FusedLayerNormAffineFunction', 'cpu_op')['name'] = 'other_norm'
        with self.assertRaisesRegex(ValueError, 'CPU owner'):
            profile.trace_forward_records(events, profile.ARMS[1])

    def test_missing_projection(self):
        events = fixture(True, True)
        events.remove(find(events, 'ConsumerGemmRSFunction', 'cpu_op'))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            profile.trace_forward_records(events, profile.ARMS[1])


if __name__ == '__main__':
    unittest.main()

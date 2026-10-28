"""A-derived op4 lifecycle and source-boundary checks; no device claims."""
import ast
from collections import deque
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest

import arrival_op4_sources as op4
import arrival_op3_v2_sources as op3
import build
from common import HERE
from test_arrival_op3_sources import body, definition


class Op4SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='op4-sources-')
        cls.root = Path(cls.temp.name)
        cls.original = {}
        for relative in (op4.HEADER, op4.RUNTIME, op4.EPILOGUE, op4.WRAPPER, op4.FRONTEND):
            text = (HERE/'fused_v2'/relative).read_text()
            if relative == op4.WRAPPER:
                text = build.wrapper(text, legacy_fused=True)
            elif relative == op4.FRONTEND:
                text = build.legacy_fused_python(text).replace('(8, 0)', '(9, 0)')
            cls.original[relative] = text
            path = cls.root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        cls.generated = {v: op4.sources(cls.root, v) for v in ('D', 'L', 'F')}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_factorial_boundaries_keep_A_codec_and_production_unchanged(self):
        self.assertEqual(self.generated['D'], op3.sources(self.root, decoder_only=True))
        self.assertEqual(set(self.generated['L']), {op4.HEADER, op4.RUNTIME, op4.WRAPPER, op4.FRONTEND})
        self.assertEqual(set(self.generated['F']), set(self.generated['L']) | {op4.DECODER})
        for variant, files in self.generated.items():
            self.assertNotIn(op4.EPILOGUE, files)
            self.assertNotIn('src/gemm_rs/taco_codec.cuh', files)
            runtime = files[op4.RUNTIME]
            for anchor in ('__global__ void taco_encode_scatter_kernel(',
                           '__global__ void taco_decode_ring_kernel(',
                           'extern "C" int taco_configure('):
                self.assertEqual(definition(runtime, anchor), definition(self.original[op4.RUNTIME], anchor))
            self.assertNotIn('taco_op3_try', runtime)
            self.assertNotIn('taco_op3_tail', runtime)
        self.assertEqual(self.generated['L'][op4.WRAPPER], self.generated['F'][op4.WRAPPER])
        self.assertEqual(self.generated['L'][op4.HEADER], self.generated['F'][op4.HEADER])
        self.assertEqual(self.generated['F'][op4.DECODER], self.generated['D'][op4.DECODER])
        for relative, text in self.original.items():
            self.assertEqual((self.root/relative).read_text(), text)

    def test_L_scalar_F_warp_and_both_preserve_publication_barrier(self):
        self.assertEqual(body(self.generated['L'][op4.RUNTIME], 'extern "C" int taco_decode_reduce('),
                         body(self.original[op4.RUNTIME], 'extern "C" int taco_decode_reduce('))
        self.assertEqual(body(self.generated['F'][op4.RUNTIME], 'extern "C" int taco_decode_reduce('),
                         body(self.generated['D'][op4.RUNTIME], 'extern "C" int taco_decode_reduce('))
        wrapper = self.generated['L'][op4.WRAPPER]
        for anchor in ('forward_gemm_impl(', 'forward_barrier(torch::Tensor input',
                       'forward_impl('):
            # Production, publication and call order stay byte identical.
            self.assertEqual(definition(wrapper, anchor), definition(self.original[op4.WRAPPER], anchor))
        reduction = body(wrapper, 'local_reduction(torch::Tensor buffer')
        self.assertIn('auto output = torch::empty(', reduction)
        self.assertIn('if (!owned_taco.enabled) group_barrier.barrier_all(stream);', reduction)
        self.assertNotIn('clone(', reduction)

    def test_cpp_scopes_configuration_and_rejects_shape_stream_dtype_capture(self):
        wrapper = self.generated['L'][op4.WRAPPER]
        start = wrapper.index('    // Scope the immutable instance config in C++')
        scope = wrapper[start:wrapper.index('    return forward_impl(', start)]
        self.assertIn('~RestoreOp4() { taco_op4_set_config(previous); }', scope)
        for requirement in ('current == taco_stream', 'status == cudaStreamCaptureStatusNone',
                            'input.scalar_type() == at::ScalarType::BFloat16',
                            'weight.scalar_type() == at::ScalarType::BFloat16',
                            'input.size(1) == owned_taco.op4_k_local',
                            'weight.size(1) == owned_taco.op4_k_local'):
            self.assertIn(requirement, scope)
        self.assertLess(scope.index('TORCH_CHECK(current'), scope.index('taco_op4_set_config(owned_taco)'))
        for variant in ('L', 'F'):
            ast.parse(self.generated[variant][op4.FRONTEND])
            self.assertIn("os.environ.get('H100_OP4_PROTOCOL') != 'h100-arrival-op4'",
                          self.generated[variant][op4.FRONTEND])

    def frontend_namespace(self):
        tree = ast.parse(self.generated['L'][op4.FRONTEND])
        public = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GemmRSTaco')
        calls, slots = [], []

        class Slot:
            def __init__(self, group, m, n, k_local, placement):
                self.group, self.m, self.n, self.k_local = group, m, n, k_local
                self.rank, self.world, self.device, self.stream = 0, 8, 0, 97
                self.placement = placement
                self.wire_bytes_per_rank, self.bf16_wire_bytes_per_rank = 7, 14
                self.index = len(slots)
                self.peers = [SimpleNamespace(data_ptr=lambda: self.index+100)]
                self.fail = False
                slots.append(self)

            def forward(self, input, weight):
                calls.append((self.index, input, weight))
                if self.fail:
                    raise ValueError('producer may already be enqueued')
                return object()

        namespace = dict(_GemmRSTacoSlot=Slot, threading=threading)
        exec(compile(ast.Module(body=[public], type_ignores=[]), '<op4-public>', 'exec'), namespace)
        return namespace, calls, slots

    def test_actual_public_frontend_alternates_and_retains_independent_outputs(self):
        namespace, calls, slots = self.frontend_namespace()
        op = namespace['GemmRSTaco'](None, 2048, 4096, 512, 'fused')
        outputs = [op.forward(i, -i) for i in range(8)]
        self.assertEqual([c[0] for c in calls], [0, 1]*4)
        self.assertEqual(len({id(x) for x in outputs}), 8)
        self.assertTrue(op.owns_output and op.owns_configuration)
        self.assertEqual(len(slots), 2)
        self.assertEqual([s.peers[0].data_ptr() for s in slots], [100, 101])
        self.assertEqual(op.next_slot, 0)

    def test_failed_forward_poisoning_prevents_unsafe_slot_reuse(self):
        namespace, calls, slots = self.frontend_namespace()
        op = namespace['GemmRSTaco'](None, 2048, 4096, 2048, 'fused')
        slots[0].fail = True
        with self.assertRaises(ValueError):
            op.forward(1, 2)
        self.assertEqual(op.next_slot, 0)
        with self.assertRaisesRegex(RuntimeError, 'after a failed forward'):
            op.forward(3, 4)
        self.assertEqual(len(calls), 1)

    @unittest.skipUnless(shutil.which('g++'), 'host compiler needed')
    def test_actual_runtime_configuration_code_rejects_bad_shape_and_restores(self):
        header = self.generated['L'][op4.HEADER]
        runtime = self.generated['L'][op4.RUNTIME]
        source = '#include <cassert>\n#include <cstdint>\n'
        source += 'enum {cudaSuccess=0, cudaErrorInvalidValue=1};\n'
        source += definition(header, 'struct FluxTacoConfig') + ';\n'
        source += 'static thread_local FluxTacoConfig active;\n'
        for anchor in ('extern "C" FluxTacoConfig taco_config()', 'extern "C" void taco_reset()',
                       'extern "C" int taco_placement()', 'extern "C" int taco_configure(',
                       'extern "C" int taco_op4_bind(', 'extern "C" void taco_op4_set_config('):
            source += definition(runtime, anchor) + '\n'
        source += '''
int main() {
  unsigned char storage[8]; void* peers[8];
  for (int i=0;i<8;i++) peers[i]=storage+i;
  assert(taco_op4_bind(512)!=0);
  assert(taco_configure(peers,0,8,2048,4096)==0);
  assert(taco_op4_bind(1024)!=0);
  assert(taco_op4_bind(512)==0);
  const auto a=taco_config();
  taco_reset(); assert(!taco_config().enabled);
  assert(taco_configure(peers,0,8,2048,4096)==0);
  assert(taco_op4_bind(2048)==0);
  const auto b=taco_config();
  taco_op4_set_config(a); assert(taco_config().op4_k_local==512);
  taco_op4_set_config(b); assert(taco_config().op4_k_local==2048);
  taco_reset(); assert(taco_config().op4_k_local==0);
  assert(taco_configure(peers,0,4,2048,4096)==0);
  assert(taco_op4_bind(512)!=0);
}
'''
        with tempfile.TemporaryDirectory(prefix='op4-runtime-test-') as temp:
            path = Path(temp)/'test.cc'
            path.write_text(source)
            subprocess.run(['g++', '-std=c++17', '-O2', str(path), '-o', str(Path(temp)/'test')], check=True)
            subprocess.run([str(Path(temp)/'test')], check=True)


class TwoSlotOrderTests(unittest.TestCase):
    """Enumerate asynchronous rank schedules of producer/barrier/decode phases.

    This checks the lifetime argument, not GPU fence implementation. A phase is
    G=producer, B=post-GEMM all-rank barrier, D=decode. Each slot has its own
    barrier, and calls use the same stream on every rank.
    """
    @staticmethod
    def explore(slots, calls=5, world=2):
        initial = (0,)*world
        queue, seen = deque([initial]), {initial}
        while queue:
            state = queue.popleft()
            for rank, phase in enumerate(state):
                call, stage = divmod(phase, 3)
                if call == calls:
                    continue
                if stage == 0 and call >= slots:
                    # Producer reuses a packet slot: every receiver must have
                    # completed the previous decode that read this slot.
                    if any(p < 3*(call-slots+1) for p in state):
                        return False, state
                if stage == 1 and any(p < 3*call+1 for p in state):
                    continue
                next_state = list(state)
                next_state[rank] += 1
                next_state = tuple(next_state)
                if next_state not in seen:
                    seen.add(next_state)
                    queue.append(next_state)
        return True, len(seen)

    def test_two_slots_cover_every_enumerated_delayed_rank_schedule(self):
        safe, count = self.explore(2)
        self.assertTrue(safe)
        self.assertGreater(count, 50)

    def test_removing_reuse_barrier_with_one_slot_has_a_counterexample(self):
        safe, counterexample = self.explore(1)
        self.assertFalse(safe)
        self.assertIn(3, counterexample)


if __name__ == '__main__':
    unittest.main()

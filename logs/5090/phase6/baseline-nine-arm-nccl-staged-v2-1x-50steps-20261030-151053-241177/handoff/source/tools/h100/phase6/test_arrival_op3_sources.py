"""Regression checks against the actual pinned A -> O source transformation.

Host execution tests configuration and frontend lifecycle; none of these checks
establish CUDA execution, cross-device memory visibility, or measured overlap.
"""
import ast
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest

import arrival_op3_sources as op3
import build
from common import HERE


def definition(source, signature):
    """Extract a definition by balanced braces, including nested control flow."""
    begin = source.index(signature)
    opening = source.index('{', begin)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end]


def body(source, signature):
    text = definition(source, signature)
    return text[text.index('{') + 1:-1]


class ArrivalOp3SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='arrival-op3-sources-')
        cls.root = Path(cls.temp.name)
        cls.original = {}
        for relative in (op3.HEADER, op3.RUNTIME, op3.EPILOGUE, op3.WRAPPER, op3.FRONTEND):
            text = (HERE/'fused_v2'/relative).read_text()
            if relative == op3.WRAPPER:
                text = build.wrapper(text, legacy_fused=True)
            elif relative == op3.FRONTEND:
                text = build.legacy_fused_python(text).replace('(8, 0)', '(9, 0)')
            cls.original[relative] = text
            target = cls.root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        cls.generated = op3.sources(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_six_changed_files_and_original_input_tree_remains_identical(self):
        self.assertEqual(set(self.generated), {
            op3.HEADER, op3.RUNTIME, op3.EPILOGUE, op3.WRAPPER, op3.FRONTEND, op3.HELPER})
        for relative, original in self.original.items():
            self.assertEqual(hashlib.sha256((self.root/relative).read_bytes()).hexdigest(),
                             hashlib.sha256(original.encode()).hexdigest())
            self.assertNotEqual(self.generated[relative], original)
        self.assertFalse((self.root/op3.HELPER).exists())

    def test_decode_arithmetic_is_extracted_verbatim_from_pinned_A(self):
        original = body(self.original[op3.RUNTIME], '__global__ void taco_decode_ring_kernel(')
        original_arithmetic = original[original.index('  const int nt = '):]
        extracted = body(self.generated[op3.RUNTIME],
                         '__device__ __forceinline__ void taco_op3_decode_group(')
        self.assertEqual(extracted.lstrip('\n'), original_arithmetic)
        self.assertNotIn('decode_warp', extracted)
        # Retaining the original kernel also provides an unmodified reference
        # definition within O; only scheduling calls use the extracted helper.
        self.assertEqual(definition(self.generated[op3.RUNTIME],
                                    '__global__ void taco_decode_ring_kernel('),
                         definition(self.original[op3.RUNTIME],
                                    '__global__ void taco_decode_ring_kernel('))
        for name in ('taco_op3_try_decode(', 'taco_op3_tail_decode('):
            self.assertIn('taco_op3_decode_group(', body(self.generated[op3.RUNTIME], name))

    def test_every_payload_writer_fences_before_local_and_remote_release(self):
        epilogue = self.generated[op3.EPILOGUE]
        remote = body(epilogue, 'taco_end_step(int step_idx)')
        local = body(epilogue, 'end_step(int step_idx)')
        self.assertLess(remote.rindex('flux_taco::encode_warp('),
                        remote.index('taco_op3_publish('))
        self.assertLess(local.rindex('cutlass::arch::global_store'),
                        local.index('taco_op3_publish('))
        self.assertLess(local.index('taco_end_step(step_idx)'),
                        local.index('return;  // No remote BF16 store'))
        self.assertEqual(epilogue.count('taco_op3_publish('), 2)
        publication = body(self.generated[op3.HELPER], 'void taco_op3_publish(')
        self.assertLess(publication.index('__threadfence_system()'),
                        publication.index('__syncthreads()'))
        self.assertLess(publication.index('__syncthreads()'),
                        publication.index('if (threadIdx.x == 0)'))
        self.assertLess(publication.index('if (threadIdx.x == 0)'),
                        publication.index('taco_op3_release('))

    def test_event_dependencies_allow_try_overlap_but_tail_follows_both_streams(self):
        wrapper = self.generated[op3.WRAPPER]
        producer = body(wrapper, 'forward_gemm_impl(')
        positions = [producer.index(token) for token in (
            'op3_output_ = torch::empty',
            'cudaEventRecord(op3_input_ready_, stream)',
            'cutlass_op->run(args, workspace, stream)',
            'cudaStreamWaitEvent(op3_stream_, op3_input_ready_, 0)',
            'taco_op3_try(',
            'cudaEventRecord(op3_try_done_, op3_stream_)')]
        self.assertEqual(positions, sorted(positions))
        reduction = body(wrapper, 'local_reduction(')
        positions = [reduction.index(token) for token in (
            'auto output = op3_output_',
            'cudaStreamWaitEvent(stream, op3_try_done_, 0)',
            'taco_op3_tail(',
            'group_barrier.barrier_all(stream)',
            'op3_output_ = torch::Tensor()',
            'return output')]
        self.assertEqual(positions, sorted(positions))

    def test_publication_barrier_only_bypassed_for_O_taco_reuse_stays(self):
        old = body(self.original[op3.WRAPPER], 'forward_barrier(torch::Tensor input')
        new = body(self.generated[op3.WRAPPER], 'forward_barrier(torch::Tensor input')
        self.assertNotIn('taco_config().enabled', old)
        self.assertIn('group_barrier.barrier_all(stream)', old)
        self.assertLess(new.index('if (taco_config().enabled) return;'),
                        new.index('group_barrier.barrier_all(stream)'))
        for variant in (self.original, self.generated):
            reduction = body(variant[op3.WRAPPER], 'local_reduction(')
            self.assertIn('group_barrier.barrier_all(stream)', reduction)

    def test_try_is_bounded_and_done_is_published_after_all_fragment_groups(self):
        runtime = self.generated[op3.RUNTIME]
        attempt = body(runtime, 'taco_op3_try_decode(')
        self.assertNotIn('while (', attempt)
        self.assertLess(attempt.index('if (!can_decode) return;'),
                        attempt.index('taco_op3_decode_group('))
        self.assertLess(attempt.index('taco_op3_decode_group('),
                        attempt.index('taco_op3_done(c)[fragment] = c.op3_epoch'))
        completed_loop = attempt[attempt.index('taco_op3_decode_group('):
                                 attempt.index('taco_op3_done(c)[fragment] = c.op3_epoch')]
        self.assertIn('__syncthreads()', completed_loop)
        tail = body(runtime, 'taco_op3_tail_decode(')
        self.assertLess(tail.index('taco_op3_done(c)[fragment] == c.op3_epoch'),
                        tail.index('while (!taco_op3_is_ready'))
        self.assertLess(tail.index('__syncthreads()'), tail.index('taco_op3_decode_group('))
        prepare = body(runtime, 'int taco_op3_prepare(')
        self.assertNotIn('taco_op3_ready(', prepare)
        self.assertNotIn('taco_op3_done(', prepare)

    def test_source_drift_and_repeat_application_fail_instead_of_silent_partial_patch(self):
        mutations = (
            (op3.header, op3.HEADER, '  int n = 0;\n};'),
            (op3.runtime, op3.RUNTIME, '__global__ void taco_decode_ring_kernel('),
            (op3.epilogue, op3.EPILOGUE,
             '      // The existing GemmRS group barrier publishes completion before DQ.'),
            (op3.wrapper, op3.WRAPPER, '    cutlass_op->run(args, workspace, stream);'),
            (op3.frontend, op3.FRONTEND, '        self._lock = threading.Lock()'))
        for transform, relative, anchor in mutations:
            with self.subTest(file=relative), self.assertRaises(ValueError):
                transform(self.original[relative].replace(anchor, '// anchor changed', 1))
        with tempfile.TemporaryDirectory(prefix='arrival-op3-reapply-') as directory:
            root = Path(directory)
            for relative, text in self.generated.items():
                target = root/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            with self.assertRaises(ValueError):
                op3.sources(root)

    def test_frontend_ast_constructor_does_not_publish_an_epoch(self):
        module = ast.parse(self.generated[op3.FRONTEND])
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
                   and node.name == 'GemmRSTaco')
        methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
        init = methods['__init__']
        constructor_op3_calls = [node for node in ast.walk(init)
                                if isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Attribute)
                                and node.func.attr == 'taco_op3_configure']
        self.assertFalse(constructor_op3_calls)
        self.assertTrue(any(isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                            and node.value.value == 0 and any(
                                isinstance(target, ast.Attribute) and target.attr == 'op3_epoch'
                                for target in node.targets) for node in ast.walk(init)))

    def test_executed_frontend_increments_epoch_once_and_resets_config_on_failure(self):
        module = ast.parse(self.generated[op3.FRONTEND])
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef))
        forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                       and node.name == 'forward')
        fake_torch = SimpleNamespace(
            bfloat16='bf16',
            cuda=SimpleNamespace(current_device=lambda: 0,
                                 current_stream=lambda: SimpleNamespace(cuda_stream=99),
                                 is_current_stream_capturing=lambda: False))
        namespace = {'torch': fake_torch}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), '<op3 forward>', 'exec'), namespace)
        calls = []
        fail = [False]

        def configure_epoch(*args):
            calls.append(('epoch', args))
            return int(fail[0])

        state = SimpleNamespace(
            device=0, stream=99, m=2048, n=4096, k_local=512, world=8, rank=0,
            pointers=None, option=None, op3_epoch=0, op3_mode='overlap', op3_rounds=2,
            op3_instrument=0, _lock=threading.Lock(),
            lib=SimpleNamespace(
                taco_configure=lambda *args: calls.append(('base', args)) or 0,
                taco_op3_configure=configure_epoch,
                taco_reset=lambda: calls.append(('reset', ()))),
            op=SimpleNamespace(forward=lambda *args, **kwargs:
                               SimpleNamespace(reshape=lambda *shape: shape)))

        def tensor(shape):
            return SimpleNamespace(is_cuda=True, device=SimpleNamespace(index=0),
                                   dtype='bf16', shape=shape, is_contiguous=lambda: True,
                                   requires_grad=False)

        inputs, weights = tensor((2048, 512)), tensor((4096, 512))
        for _ in range(2):
            self.assertEqual(namespace['forward'](state, inputs, weights), (256, 4096))
        self.assertEqual([value for kind, value in calls if kind == 'epoch'],
                         [(1, 1, 2, 0), (2, 1, 2, 0)])
        self.assertEqual(sum(kind == 'reset' for kind, _ in calls), 2)
        fail[0] = True
        with self.assertRaisesRegex(RuntimeError, 'configuration rejected'):
            namespace['forward'](state, inputs, weights)
        self.assertEqual(calls[-1][0], 'reset')
        state.op3_epoch = 0x7fffffff
        before = list(calls)
        with self.assertRaisesRegex(RuntimeError, 'epoch exhausted'):
            namespace['forward'](state, inputs, weights)
        self.assertEqual(calls, before)

    @unittest.skipUnless(shutil.which('g++'), 'g++ needed to execute actual host configuration')
    def test_compiled_real_C_configuration_shape_modes_and_generation_gates(self):
        runtime = self.generated[op3.RUNTIME]
        config_struct = definition(self.generated[op3.HEADER], 'struct FluxTacoConfig')
        helpers = '\n'.join(definition(self.generated[op3.HEADER], name)
                            for name in ('__host__ __device__ inline int64_t taco_groups(',
                                         '__host__ __device__ inline int64_t taco_source_bytes('))
        helpers = helpers.replace('__host__', '').replace('__device__', '')
        definitions = '\n'.join(definition(runtime, name) for name in (
            'extern "C" void taco_reset()', 'extern "C" int taco_configure(',
            'extern "C" int taco_op3_configure('))
        code = '#include <cstdint>\n#include <cassert>\n' + config_struct + ';\n' + helpers + '''
static thread_local FluxTacoConfig active;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1 };
constexpr int kOp3ControlBytes = 18472;
int placement = 1;
int taco_placement() { return placement; }
''' + definitions + r'''
int main() {
  void* peers[8];
  for(int i=0;i<8;++i) peers[i]=reinterpret_cast<void*>(uintptr_t(0x10000+i*0x10000));
  assert(taco_op3_configure(1,1,2,0)!=0);  // No active base configuration.
  assert(taco_configure(peers,0,8,2048,4096)==0);
  assert(active.op3_epoch==0 && active.op3_overlap==0);  // Construction is not a launch.
  assert(taco_op3_configure(1,1,2,0)==0);
  assert(active.op3_epoch==1 && active.op3_overlap==1 && active.op3_rounds==2);
  assert(taco_op3_configure(0x7fffffffU,0,4,1)==0);
  assert(active.op3_epoch==0x7fffffffU && active.op3_overlap==0 && active.op3_diagnostics==1);
  struct Args { uint32_t epoch; int overlap, rounds, diagnostic; };
  Args invalid[]={{0,1,2,0},{0x80000000U,1,2,0},{1,2,2,0},
                  {1,1,0,0},{1,1,5,0},{1,1,2,2}};
  for(auto x:invalid)
    assert(taco_op3_configure(x.epoch,x.overlap,x.rounds,x.diagnostic)!=0);
  placement=2; assert(taco_op3_configure(1,1,2,0)!=0); placement=1;
  active.world=4; assert(taco_op3_configure(1,1,2,0)!=0); active.world=8;
  active.m=1024; assert(taco_op3_configure(1,1,2,0)!=0); active.m=2048;
  active.n=2048; assert(taco_op3_configure(1,1,2,0)!=0); active.n=4096;
  taco_reset();
  assert(taco_configure(peers,0,8,2048,4096)==0);
  assert(active.op3_epoch==0);  // A former epoch cannot leak into a new instance.
}
'''
        with tempfile.TemporaryDirectory(prefix='arrival-op3-host-config-') as directory:
            source = Path(directory)/'config.cc'
            binary = source.with_suffix('')
            source.write_text(code)
            subprocess.run(['g++', '-std=c++17', '-O2', str(source), '-o', str(binary)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()

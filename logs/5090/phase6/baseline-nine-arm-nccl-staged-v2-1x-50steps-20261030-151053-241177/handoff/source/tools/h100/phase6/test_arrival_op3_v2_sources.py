"""Regression checks of the actual pinned A -> D/O source transformations.

These inspect generated sources and execute frontend gates / host configuration;
they do not establish device execution, cumulative publication or performance.
"""
import ast
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import arrival_op3_v2_sources as op3
import build
from common import HERE
from test_arrival_op3_sources import body, definition


class ArrivalOp3V2SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='arrival-op3-v2-sources-')
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
        cls.decoder_only = op3.sources(cls.root, decoder_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_A_is_not_mutated_and_D_changes_only_decoder_runtime_and_resource(self):
        self.assertEqual(set(self.generated), {
            op3.HEADER, op3.RUNTIME, op3.EPILOGUE, op3.WRAPPER, op3.FRONTEND,
            op3.HELPER, op3.DECODER})
        self.assertEqual(set(self.decoder_only), {op3.RUNTIME, op3.DECODER})
        self.assertEqual(self.generated[op3.DECODER], self.decoder_only[op3.DECODER])
        for relative, original in self.original.items():
            self.assertEqual(hashlib.sha256((self.root/relative).read_bytes()).hexdigest(),
                             hashlib.sha256(original.encode()).hexdigest())
        self.assertNotIn('taco_op3_publish(', self.decoder_only[op3.RUNTIME])
        self.assertNotIn('taco_op3_stop(', self.decoder_only[op3.RUNTIME])
        # Both new runtimes keep the independent scalar implementation available.
        for runtime in (self.generated[op3.RUNTIME], self.decoder_only[op3.RUNTIME]):
            for function in ('__global__ void taco_encode_scatter_kernel(',
                             '__global__ void taco_decode_ring_kernel(',
                             'extern "C" int taco_configure('):
                self.assertEqual(definition(runtime, function),
                                 definition(self.original[op3.RUNTIME], function))

    def test_D_uses_four_warps_but_rejects_unvalidated_shape(self):
        runtime = self.decoder_only[op3.RUNTIME]
        launch = body(runtime, 'extern "C" int taco_decode_reduce(')
        self.assertIn('c.world != 8 || c.m != 2048 || c.n != 4096', launch)
        self.assertIn('taco_op3_v2_decode<<<taco_groups(c) / 4, 128, 0, stream>>>', launch)
        kernel = body(runtime, '__global__ void taco_op3_v2_decode(')
        self.assertIn('int64_t(blockIdx.x) * 4 + threadIdx.x / 32', kernel)
        self.assertIn('op3_v2_decode_warp(c, local, output, group)', kernel)
        self.assertNotIn('__syncthreads', self.generated[op3.DECODER])
        self.assertNotIn('__shared__', self.generated[op3.DECODER])

    def test_cumulative_publication_follows_all_local_and_remote_writers(self):
        helper = self.generated[op3.HELPER]
        publish = body(helper, 'void taco_op3_publish(')
        self.assertLess(publish.index('__syncthreads()'), publish.index('if (threadIdx.x == 0)'))
        self.assertLess(publish.index('if (threadIdx.x == 0)'), publish.index('taco_op3_release('))
        self.assertEqual(publish.count('__syncthreads()'), 1)
        self.assertNotIn('__threadfence_system()', publish)
        self.assertIn('st.release.sys.global.u32', body(helper, 'void taco_op3_release('))
        self.assertIn('ld.acquire.sys.global.u32', body(helper, 'uint32_t taco_op3_acquire('))
        ready = body(helper, 'bool taco_op3_is_ready(')
        self.assertIn('source < 8', ready)
        self.assertIn('source * kOp3Fragments + fragment', ready)
        for name, writer in (('taco_end_step(int step_idx)', 'flux_taco::encode_warp('),
                             ('end_step(int step_idx)', 'cutlass::arch::global_store')):
            epilogue = body(self.generated[op3.EPILOGUE], name)
            self.assertLess(epilogue.rindex(writer), epilogue.rindex('taco_op3_publish('))

    def test_try_is_bounded_and_stops_between_warp_groups_without_CTA_deadlock(self):
        runtime = self.generated[op3.RUNTIME]
        attempt = body(runtime, '__global__ void taco_op3_v2_try_decode(')
        self.assertNotIn('while (', attempt)
        self.assertIn('const int fragment = blockIdx.x;', attempt)
        self.assertIn('for (int compact = warp; compact < 16; compact += 4)', attempt)
        self.assertLess(attempt.index('taco_op3_is_ready('), attempt.index('__syncthreads()'))
        self.assertLess(attempt.index('__syncthreads()'), attempt.index('if (!ready) return;'))
        per_group = attempt[attempt.index('for (int compact = warp;'):]
        self.assertNotIn('__syncthreads()', per_group)
        ordered = [per_group.index(token) for token in (
            'stopped = taco_op3_stopped(c)', '__shfl_sync(0xffffffff, stopped, 0)',
            'if (stopped) return;', 'if (done) continue;',
            'op3_v2_decode_warp(', '__syncwarp()', 'taco_op3_done(c)[group] = c.op3_epoch')]
        self.assertEqual(ordered, sorted(ordered))
        launch = body(runtime, 'extern "C" int taco_op3_try(')
        self.assertIn('round < c.op3_rounds', launch)
        self.assertIn('<<<kOp3EarlyCtas, 128, 0, stream>>>', launch)
        self.assertNotIn('round *', attempt)

    def test_stop_is_independent_of_aux_and_tail_follows_local_GEMM_and_join(self):
        runtime = self.generated[op3.RUNTIME]
        stop = body(runtime, '__global__ void taco_op3_v2_stop(')
        self.assertIn('st.release.gpu.global.u32', stop)
        self.assertIn('ld.acquire.gpu.global.u32',
                      body(self.generated[op3.HELPER], 'bool taco_op3_stopped('))
        wrapper = self.generated[op3.WRAPPER]
        producer = body(wrapper, 'forward_gemm_impl(')
        ordered = [producer.index(token) for token in (
            'cudaEventRecord(op3_input_ready_, stream)',
            'cutlass_op->run(args, workspace, stream)',
            'taco_op3_stop(taco, stream)',
            'cudaStreamWaitEvent(op3_stream_, op3_input_ready_, 0)',
            'taco_op3_try(', 'cudaEventRecord(op3_try_done_, op3_stream_)')]
        self.assertEqual(ordered, sorted(ordered))
        # Aux waits only on pre-GEMM inputs. Stop has no wait on aux; main joins
        # aux afterward. This makes input -> {GEMM->stop, try} -> tail acyclic.
        self.assertEqual(producer.count('cudaStreamWaitEvent('), 1)
        reduction = body(wrapper, 'local_reduction(')
        ordered = [reduction.index(token) for token in (
            'cudaStreamWaitEvent(stream, op3_try_done_, 0)',
            'taco_op3_tail(', 'group_barrier.barrier_all(stream)')]
        self.assertEqual(ordered, sorted(ordered))
        barrier = body(wrapper, 'forward_barrier(torch::Tensor input')
        self.assertLess(barrier.index('if (taco_config().enabled) return;'),
                        barrier.index('group_barrier.barrier_all(stream)'))

    def test_tail_checks_group_done_and_only_spins_before_its_CTA_barrier(self):
        tail = body(self.generated[op3.RUNTIME], '__global__ void taco_op3_v2_tail_decode(')
        self.assertIn('taco_op3_done(c)[taco_op3_group(fragment, compact)]', tail)
        self.assertLess(tail.index('while (!taco_op3_is_ready'), tail.index('__syncthreads()'))
        per_group = tail[tail.index('for (int compact = warp;'):]
        self.assertIn('taco_op3_done(c)[group] == c.op3_epoch', per_group)
        self.assertIn('if (!done) op3_v2_decode_warp(', per_group)
        self.assertNotIn('__syncthreads()', per_group)
        prepare = body(self.generated[op3.RUNTIME], 'extern "C" int taco_op3_prepare(')
        self.assertNotIn('taco_op3_ready(', prepare)
        self.assertNotIn('taco_op3_done(', prepare)
        self.assertNotIn('taco_op3_stop_ptr(', prepare)
        self.assertIn('c.op3_diagnostics ?', prepare)

    def test_frontend_AST_executes_strict_environment_and_library_protocol_gates(self):
        source = self.generated[op3.FRONTEND]
        module = ast.parse(source)
        cls = next(node for node in module.body
                   if isinstance(node, ast.ClassDef) and node.name == 'GemmRSTaco')
        constructor = next(node for node in cls.body
                           if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        gates = [node for node in ast.walk(constructor) if isinstance(node, ast.If)
                 and any(isinstance(child, ast.Constant) and child.value in (
                     'H100_OP3_PROTOCOL', 'Wrong op3 library protocol')
                         for child in ast.walk(node))]
        self.assertEqual(len(gates), 2)
        environment_gate = next(node for node in gates if 'H100_OP3_PROTOCOL' in ast.unparse(node))
        library_gate = next(node for node in gates if 'taco_op3_protocol_version' in ast.unparse(node))
        for protocol, accepted in ((None, False), ('h100-arrival-op3', False),
                                   ('h100-arrival-op3-v2', True)):
            namespace = {'os': SimpleNamespace(environ={'H100_OP3_PROTOCOL': protocol})}
            executable = compile(ast.Module(body=[environment_gate], type_ignores=[]), '<env gate>', 'exec')
            if accepted:
                exec(executable, namespace)
            else:
                with self.assertRaisesRegex(RuntimeError, 'own experiment protocol'):
                    exec(executable, namespace)
        for version in (0, 1, 2, 3):
            namespace = {'self': SimpleNamespace(
                lib=SimpleNamespace(taco_op3_protocol_version=lambda: version))}
            executable = compile(ast.Module(body=[library_gate], type_ignores=[]), '<library gate>', 'exec')
            if version == 2:
                exec(executable, namespace)
            else:
                with self.assertRaisesRegex(RuntimeError, 'Wrong op3 library protocol'):
                    exec(executable, namespace)
        self.assertLess(source.index("os.environ.get('H100_OP3_PROTOCOL')"),
                        source.index('self.lib = ctypes.CDLL'))
        self.assertLess(source.index('self.lib.taco_op3_protocol_version()'),
                        source.index('cpp_mod.init_flux_shm(group)'))
        self.assertNotIn('taco_op3_configure(', ast.unparse(constructor))

    def test_frontend_control_slice_and_diagnostics_use_group_units(self):
        source = self.generated[op3.FRONTEND]
        self.assertIn('start:start+49216', source)
        self.assertIn('self.packet_bytes_per_source + 49160', source)
        self.assertIn('start:start+56', source)
        self.assertIn("work_unit='h128_group'", source)
        self.assertIn('try_groups=count, max_try_groups=1024, early_scan_ctas=64', source)
        self.assertIn('stop_on_local_gemm=True', source)
        self.assertNotIn('try_fragments=', source)

    @unittest.skipUnless(shutil.which('g++'), 'g++ needed for actual host configuration')
    def test_compiled_control_layout_mapping_and_configuration_gates(self):
        header = self.generated[op3.HEADER]
        helper = self.generated[op3.HELPER]
        config_struct = definition(header, 'struct FluxTacoConfig') + ';\n'
        helpers = '\n'.join(definition(header, signature) for signature in (
            '__host__ __device__ inline int64_t taco_groups(',
            '__host__ __device__ inline int64_t taco_source_bytes('))
        control = helper[helper.index('constexpr int kOp3Fragments'):helper.index('#ifdef __CUDACC__')]
        mapping = definition(helper, '__device__ __forceinline__ int taco_op3_group(')
        runtime = self.generated[op3.RUNTIME]
        configuration = '\n'.join(definition(runtime, signature) for signature in (
            'extern "C" void taco_reset()', 'extern "C" int taco_configure(',
            'extern "C" int taco_op3_configure(', 'extern "C" int taco_op3_protocol_version()'))
        code = '''#include <cstdint>
#include <cassert>
#include <vector>
#define __host__
#define __device__
#define __forceinline__ inline
''' + config_struct + helpers + control + mapping + '''
static thread_local FluxTacoConfig active;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1 };
int placement = 1;
int taco_placement() { return placement; }
''' + configuration + r'''
int main() {
  static_assert(kOp3StatsOffset == 49160 && kOp3ControlBytes == 49216);
  static_assert(kOp3StatsOffset % 8 == 0 && kOp3StatsCount == 7);
  static_assert(kOp3EarlyCtas == 64 && kOp3MaxTryGroups == 1024);
  assert(taco_op3_protocol_version() == 2);
  assert(taco_op3_configure(1,1,2,0) != 0);
  std::vector<unsigned char> packet(8192 * 136 * 8);
  void* peers[8];
  for (int rank=0; rank<8; ++rank) peers[rank] = packet.data();
  assert(taco_configure(peers,0,8,2048,4096) == 0);
  assert(active.op3_epoch == 0 && active.op3_overlap == 0);
  assert(taco_source_bytes(active) >= kOp3ControlBytes);
  auto* base = reinterpret_cast<unsigned char*>(taco_op3_ready(active,0));
  assert(reinterpret_cast<unsigned char*>(taco_op3_done(active))-base == 16384);
  assert(reinterpret_cast<unsigned char*>(taco_op3_stop_ptr(active))-base == 49152);
  assert(reinterpret_cast<unsigned char*>(taco_op3_stats(active))-base == kOp3StatsOffset);
  int seen[8192]{};
  for (int fragment=0; fragment<512; ++fragment)
    for (int compact=0; compact<16; ++compact) {
      int group = taco_op3_group(fragment,compact);
      assert(group>=0 && group<8192 && ++seen[group]==1);
      int row=group/32, inverse=((row/128)*32+group%32)*8+(row%64)/8;
      assert(inverse==fragment);
    }
  for (int count:seen) assert(count==1);
  assert(taco_op3_configure(1,1,2,0) == 0);
  assert(taco_op3_configure(0x7fffffffU,0,4,1) == 0);
  struct Args { uint32_t epoch; int overlap, rounds, diagnostic; };
  Args invalid[]={{0,1,2,0},{0x80000000U,1,2,0},{1,2,2,0},
                  {1,1,0,0},{1,1,5,0},{1,1,2,2}};
  for (auto x:invalid)
    assert(taco_op3_configure(x.epoch,x.overlap,x.rounds,x.diagnostic) != 0);
  placement=2; assert(taco_op3_configure(1,1,2,0)!=0); placement=1;
  active.world=4; assert(taco_op3_configure(1,1,2,0)!=0); active.world=8;
  active.m=1024; assert(taco_op3_configure(1,1,2,0)!=0); active.m=2048;
  active.n=2048; assert(taco_op3_configure(1,1,2,0)!=0); active.n=4096;
  taco_reset(); assert(taco_configure(peers,0,8,2048,4096)==0);
  assert(active.op3_epoch==0);
}
'''
        with tempfile.TemporaryDirectory(prefix='arrival-op3-v2-host-') as directory:
            source = Path(directory)/'control.cc'
            binary = source.with_suffix('')
            source.write_text(code)
            subprocess.run(['g++', '-std=c++17', '-O2', str(source), '-o', str(binary)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    def test_transforms_fail_closed_on_source_drift_and_reapplication(self):
        for transform, relative, anchor in (
            (op3.header, op3.HEADER, '  int n = 0;\n};'),
            (op3.decoder_only_runtime, op3.RUNTIME, '#include "gemm_rs/taco_codec.cuh"'),
            (op3.epilogue, op3.EPILOGUE,
             '      // The existing GemmRS group barrier publishes completion before DQ.'),
            (op3.wrapper, op3.WRAPPER, '    cutlass_op->run(args, workspace, stream);'),
            (op3.frontend, op3.FRONTEND, '        self._lock = threading.Lock()')):
            with self.subTest(file=relative), self.assertRaises(ValueError):
                transform(self.original[relative].replace(anchor, '// contract drift', 1))
        with tempfile.TemporaryDirectory(prefix='arrival-op3-v2-reapply-') as directory:
            root = Path(directory)
            for relative, text in self.generated.items():
                target = root/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            with self.assertRaises(ValueError):
                op3.sources(root)


if __name__ == '__main__':
    unittest.main()

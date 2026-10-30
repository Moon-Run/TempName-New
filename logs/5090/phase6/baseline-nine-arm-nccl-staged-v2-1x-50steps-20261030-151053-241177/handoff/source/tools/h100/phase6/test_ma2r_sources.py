"""MA2-R ownership, TLS isolation and unchanged-device-path checks; CPU only."""
import ast
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest

import build
import ma2_sources
import ma2r_sources as revision
from common import HERE
from test_arrival_op3_sources import definition


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ma2r-sources-')
        cls.root = Path(cls.temp.name)
        cls.original = {}
        for relative in (revision.HEADER, revision.RUNTIME, revision.WRAPPER, revision.FRONTEND):
            text = (HERE/'fused_v2'/relative).read_text()
            if relative == revision.WRAPPER:
                text = build.wrapper(text, legacy_fused=True)
            elif relative == revision.FRONTEND:
                text = build.legacy_fused_python(text).replace('(8, 0)', '(9, 0)')
            cls.original[relative] = text
            path = cls.root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        cls.generated = {a: revision.sources(cls.root, a) for a in ('V', 'R')}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_V_is_exact_old_MA2_and_R_changes_only_host_submission_files(self):
        self.assertEqual(self.generated['V'], ma2_sources.sources(self.root, 'V'))
        self.assertEqual(set(self.generated['R']), {revision.HEADER, revision.RUNTIME,
            revision.WRAPPER, revision.FRONTEND, revision.DECODER})
        self.assertEqual(self.generated['V'][revision.DECODER], self.generated['R'][revision.DECODER])
        self.assertEqual(definition(self.generated['R'][revision.HEADER], 'struct FluxTacoConfig'),
                         definition(self.original[revision.HEADER], 'struct FluxTacoConfig'))
        for relative, text in self.original.items():
            self.assertEqual((self.root/relative).read_text(), text)

    def test_device_arithmetic_producer_and_both_barriers_stay_identical(self):
        for anchor in ('extern "C" int taco_configure(', '__global__ void taco_encode_scatter_kernel(',
                       '__global__ void taco_decode_ring_kernel(', 'extern "C" int taco_decode_reduce('):
            self.assertEqual(definition(self.generated['R'][revision.RUNTIME], anchor),
                             definition(self.generated['V'][revision.RUNTIME], anchor))
        wrapper = self.generated['R'][revision.WRAPPER]
        for anchor in ('forward_gemm_impl(', 'local_reduction(torch::Tensor buffer',
                       'forward_barrier(torch::Tensor input', 'forward_impl('):
            self.assertEqual(definition(wrapper, anchor), definition(self.original[revision.WRAPPER], anchor))
        reduction = definition(wrapper, 'local_reduction(torch::Tensor buffer')
        self.assertIn('auto output = torch::empty(', reduction)
        self.assertIn('group_barrier.barrier_all(stream);', reduction)
        self.assertNotIn('clone(', reduction)
        self.assertNotIn('cudaDeviceSynchronize', wrapper)

    def test_native_validation_and_exception_restore_before_existing_dispatch(self):
        source = self.generated['R'][revision.WRAPPER]
        start = source.index('    // Every exit, including validation/launch exceptions')
        scope = source[start:source.index('    return forward_impl(', start)]
        for required in ('~RestoreMA2R() { taco_ma2r_restore(previous, previous_k); }',
                         'current == ma2r_stream', 'status == cudaStreamCaptureStatusNone',
                         'input.device() == weight.device()', 'input.is_contiguous()',
                         'input.size(1) == ma2r_k_local', 'weight.size(1) == ma2r_k_local',
                         '!at::GradMode::is_enabled()'):
            self.assertIn(required, scope)
        self.assertLess(scope.index('TORCH_CHECK(current'), scope.index('taco_ma2r_restore(ma2r_config'))

    def test_constructor_binds_before_native_creation_and_always_resets(self):
        tree = ast.parse(self.generated['R'][revision.FRONTEND])
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        tries = [n for n in ast.walk(constructor) if isinstance(n, ast.Try) and n.finalbody]
        self.assertEqual(len(tries), 1)
        text = ast.unparse(tries[0])
        self.assertLess(text.index('taco_ma2r_bind(k_local)'), text.index('cpp_mod.GemmRS('))
        self.assertEqual(ast.unparse(tries[0].finalbody[0]), 'self.lib.taco_reset()')
        self.assertNotIn('self.slots', ast.unparse(tree))
        self.assertIn('self.peers = cpp_mod.create_tensor_list', ast.unparse(tree))

    def make_frontend(self, function):
        tree = ast.parse(self.generated['R'][revision.FRONTEND])
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        namespace = {}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), '<ma2r-frontend>', 'exec'), namespace)
        instance = namespace['GemmRSTaco'].__new__(namespace['GemmRSTaco'])
        instance._failed, instance._lock = False, threading.Lock()
        instance.option = object()
        instance.op = SimpleNamespace(forward=function)
        return instance

    def test_actual_frontend_directly_returns_distinct_native_outputs(self):
        calls, outputs = [], []
        def forward(input, weight, **kwargs):
            calls.append((input, weight, kwargs))
            output = object()
            outputs.append(output)
            return output
        op = self.make_frontend(forward)
        actual = [op.forward(i, 10+i) for i in range(6)]
        self.assertEqual(actual, outputs)
        self.assertEqual(len({id(o) for o in actual}), 6)
        self.assertEqual([x[:2] for x in calls], [(i, 10+i) for i in range(6)])
        self.assertTrue(op.owns_output and op.owns_configuration)
        self.assertTrue(all(call[2] == {'reduce_scatter_option': op.option} for call in calls))

    def test_native_failure_poisoning_prevents_packet_reuse(self):
        calls = []
        def fail(*args, **kwargs):
            calls.append(args)
            raise ValueError('producer may have been enqueued')
        op = self.make_frontend(fail)
        with self.assertRaisesRegex(ValueError, 'producer may'):
            op.forward(1, 2)
        with self.assertRaisesRegex(RuntimeError, 'after a failed forward'):
            op.forward(3, 4)
        self.assertEqual(len(calls), 1)
        other = self.make_frontend(lambda *args, **kwargs: 9)
        self.assertEqual(other.forward(1, 2), 9)

    def test_source_anchors_and_unknown_arm_fail_closed(self):
        with self.assertRaises(ValueError):
            revision.sources(self.root, 'P')
        with self.assertRaises(ValueError):
            revision.wrapper(self.original[revision.WRAPPER].replace(
                '  const bool h100_taco = taco_config().enabled;', '// unsupported wrapper'))

    @unittest.skipUnless(shutil.which('g++'), 'host compiler required')
    def test_actual_host_runtime_and_native_guard_multiple_instances_and_errors(self):
        """Compile the generated runtime functions and actual forward scope.

        Small tensor/stream stubs exercise host decisions and exception cleanup;
        this does not execute a CUDA kernel or replace the eight-rank GPU checks.
        """
        runtime, wrapper = self.generated['R'][revision.RUNTIME], self.generated['R'][revision.WRAPPER]
        source = '#include <cassert>\n#include <cstdint>\n#include <stdexcept>\n#include <thread>\n'
        source += 'enum {cudaSuccess=0, cudaErrorInvalidValue=1};\n'
        source += definition(self.original[revision.HEADER], 'struct FluxTacoConfig')+';\n'
        source += 'static thread_local FluxTacoConfig active;\nstatic thread_local int ma2r_bound_k=0;\n'
        for anchor in ('extern "C" FluxTacoConfig taco_config()', 'extern "C" void taco_reset()',
                       'extern "C" int taco_placement()', 'extern "C" int taco_configure(',
                       'extern "C" int taco_ma2r_bind(', 'extern "C" int taco_ma2r_bound_k()',
                       'extern "C" void taco_ma2r_restore('):
            source += definition(runtime, anchor)+'\n'
        source += r'''
namespace at { enum ScalarType {BFloat16, Float};
struct GradMode { static bool enabled; static bool is_enabled() { return enabled; } };
bool GradMode::enabled=false;
}
struct Device { int value; int index() const{return value;} bool operator==(Device b)const{return value==b.value;} };
struct Tensor {
  int rows, cols, dev=0, dtype=at::ScalarType::BFloat16, ndim=2;
  bool cuda=true, contiguous=true, grad=false;
  bool is_cuda()const{return cuda;} Device device()const{return {dev};}
  int scalar_type()const{return dtype;} bool is_contiguous()const{return contiguous;}
  int dim()const{return ndim;} int size(int i)const{return i?cols:rows;}
  bool requires_grad()const{return grad;}
};
struct Stream {int value, device; int device_index()const{return device;}
  bool operator==(Stream b)const{return value==b.value && device==b.device;} };
Stream now{3,0};
namespace c10 { namespace cuda { Stream getCurrentCUDAStream(){return now;} } }
enum cudaStreamCaptureStatus {cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive};
cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
int cudaStreamIsCapturing(Stream, cudaStreamCaptureStatus* out){*out=capture;return 0;}
#define CUDA_CHECK(code) do { if((code)!=0) throw std::runtime_error("cuda"); } while(0)
#define TORCH_CHECK(condition, ...) do { if(!(condition)) throw std::runtime_error("check"); } while(0)
struct Native {
  const FluxTacoConfig ma2r_config;
  const int ma2r_k_local;
  const Stream ma2r_stream;
  bool fail=false;
  int forward(Tensor input, Tensor weight) {
'''
        start = wrapper.index('    // Every exit, including validation/launch exceptions')
        end = wrapper.index('#endif\n    return forward_impl(', start)
        source += wrapper[start:end]
        source += r'''
    assert(taco_config().peers[0]==ma2r_config.peers[0]);
    assert(taco_ma2r_bound_k()==ma2r_k_local);
    if(fail) throw std::runtime_error("dispatch");
    return ma2r_k_local;
  }
};
int main() {
  unsigned char storage[16]; void *peers[8];
  for(int i=0;i<8;++i) peers[i]=storage+i;
  assert(taco_ma2r_bind(512)!=0);
  assert(taco_configure(peers,0,8,2048,4096)==0);
  assert(taco_ma2r_bind(1024)!=0);
  assert(taco_ma2r_bind(512)==0);
  Native a{taco_config(),taco_ma2r_bound_k(),now};
  taco_reset(); assert(!taco_config().enabled && taco_ma2r_bound_k()==0);
  for(int i=0;i<8;++i) peers[i]=storage+8+i;
  assert(taco_configure(peers,0,8,2048,4096)==0);
  assert(taco_ma2r_bind(2048)==0);
  Native b{taco_config(),taco_ma2r_bound_k(),now};
  taco_reset();
  Tensor ai{2048,512}, aw{4096,512}, bi{2048,2048}, bw{4096,2048};
  for(int repeat=0;repeat<6;++repeat) {
    assert(a.forward(ai,aw)==512); assert(b.forward(bi,bw)==2048);
    assert(!taco_config().enabled && taco_ma2r_bound_k()==0);
  }
  taco_ma2r_restore(b.ma2r_config,2048);
  assert(a.forward(ai,aw)==512);
  assert(taco_config().peers[0]==b.ma2r_config.peers[0] && taco_ma2r_bound_k()==2048);
  a.fail=true; try {a.forward(ai,aw); assert(false);} catch(const std::runtime_error&) {}
  assert(taco_config().peers[0]==b.ma2r_config.peers[0] && taco_ma2r_bound_k()==2048);
  a.fail=false;
  auto rejected=[&](Tensor input,Tensor weight) {
    try {a.forward(input,weight); assert(false);} catch(const std::runtime_error&) {}
    assert(taco_config().peers[0]==b.ma2r_config.peers[0] && taco_ma2r_bound_k()==2048);
  };
  rejected(bi,bw);
  auto wrong=ai; wrong.dtype=at::ScalarType::Float; rejected(wrong,aw);
  wrong=ai; wrong.cuda=false; rejected(wrong,aw);
  wrong=ai; wrong.dev=1; rejected(wrong,aw);
  wrong=ai; wrong.ndim=1; rejected(wrong,aw);
  wrong=ai; wrong.contiguous=false; rejected(wrong,aw);
  now.value=4; rejected(ai,aw); now.value=3;
  now.device=1; rejected(ai,aw); now.device=0;
  capture=cudaStreamCaptureStatusActive; rejected(ai,aw); capture=cudaStreamCaptureStatusNone;
  aw.grad=true; at::GradMode::enabled=true; rejected(ai,aw);
  at::GradMode::enabled=false; assert(a.forward(ai,aw)==512);
  // The real binding is thread_local; another host thread cannot inherit it.
  std::thread t([] {assert(!taco_config().enabled && taco_ma2r_bound_k()==0);}); t.join();
  assert(taco_ma2r_bound_k()==2048);
  taco_reset(); assert(taco_configure(peers,0,4,2048,4096)==0);
  assert(taco_ma2r_bind(512)!=0); taco_reset();
}
'''
        with tempfile.TemporaryDirectory(prefix='ma2r-native-host-') as temp:
            path = Path(temp)/'scope.cc'
            path.write_text(source)
            subprocess.run(['g++', '-std=c++17', '-O2', '-pthread', str(path),
                            '-o', str(Path(temp)/'scope')], check=True)
            subprocess.run([str(Path(temp)/'scope')], check=True)


if __name__ == '__main__':
    unittest.main()

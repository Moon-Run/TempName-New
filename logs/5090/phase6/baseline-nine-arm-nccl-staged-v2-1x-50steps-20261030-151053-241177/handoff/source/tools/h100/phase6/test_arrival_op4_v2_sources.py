"""CPU/host validation of the actual closure source transforms; no GPU claims."""
import ast
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import arrival_op4_v2_sources as source
import arrival_op4_v2_plan as plan
import build
from common import HERE
from test_arrival_op3_sources import definition


class ClosureSourcesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='closure-source-')
        cls.root = Path(cls.temp.name)
        cls.original = {}
        for relative in (source.HEADER, source.RUNTIME, source.EPILOGUE,
                         source.WRAPPER, source.FRONTEND):
            text = (HERE / 'fused_v2' / relative).read_text()
            if relative == source.WRAPPER:
                text = build.wrapper(text, legacy_fused=True)
            elif relative == source.FRONTEND:
                text = build.legacy_fused_python(text).replace('(8, 0)', '(9, 0)')
            cls.original[relative] = text
            target = cls.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        cls.generated = {arm: source.sources(cls.root, arm) for arm in source.VARIANTS}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_codec_lifetime_producer_schedule_and_packet_layout_stay_common(self):
        for arm, files in self.generated.items():
            self.assertEqual(set(files), {source.HEADER, source.RUNTIME, source.EPILOGUE,
                                          source.FRONTEND, source.DECODER})
            self.assertNotIn(source.WRAPPER, files)
            self.assertNotIn('src/gemm_rs/taco_codec.cuh', files)
            epilogue = files[source.EPILOGUE]
            self.assertEqual(definition(epilogue, 'taco_end_step(int step_idx)'),
                             definition(self.original[source.EPILOGUE], 'taco_end_step(int step_idx)'))
            self.assertEqual(definition(files[source.RUNTIME], '__global__ void taco_encode_scatter_kernel('),
                             definition(self.original[source.RUNTIME], '__global__ void taco_encode_scatter_kernel('))
            self.assertEqual(definition(files[source.HEADER], 'int64_t taco_source_bytes('),
                             definition(self.original[source.HEADER], 'int64_t taco_source_bytes('))
            self.assertIn('taco_closure_selected(\n                params.taco, params.rank, dst_rank, tile_idx)', epilogue)
            self.assertNotIn('owns_output', files[source.FRONTEND])
            self.assertNotIn('owns_configuration', files[source.FRONTEND])
            self.assertNotIn('self.slots', files[source.FRONTEND])
            self.assertEqual(files[source.FRONTEND].count('cpp_mod.create_tensor_list('), 1)
        for relative, text in self.original.items():
            self.assertEqual((self.root / relative).read_text(), text)

    def test_two_configure_calls_both_receive_phase_and_abi_is_checked_first(self):
        for arm, files in self.generated.items():
            text = files[source.FRONTEND]
            tree = ast.parse(text)
            configure = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == 'taco_configure']
            self.assertEqual(len(configure), 2)
            self.assertTrue(all(len(node.args) == 6 for node in configure))
            self.assertTrue(all(ast.unparse(node.args[-1]) == 'self.closure_phase' for node in configure))
            self.assertLess(text.index('self.lib.taco_closure_protocol()'),
                            text.index('code = self.lib.taco_configure('))
            self.assertIn(f"self.lib.taco_closure_arm() != b'{arm}'", text)
            self.assertEqual(text.count('self.lib.taco_reset()'), 2)

    def test_actual_python_mask_snapshot_and_state_match_frozen_plan_all_phases(self):
        # Execute the actual generated mask/config block and state method, not
        # an independent implementation substituted for the production source.
        class Torch:
            uint8 = 'uint8'
            @staticmethod
            def tensor(value, dtype):
                return value

        for arm, files in self.generated.items():
            text = files[source.FRONTEND]
            tree = ast.parse(text)
            klass = next(node for node in tree.body if isinstance(node, ast.ClassDef))
            methods = [node for node in klass.body if isinstance(node, ast.FunctionDef)
                       and node.name in ('closure_state', 'wire_bytes_per_rank')]
            stub = ast.ClassDef(name='Snapshot', bases=[], keywords=[], body=methods, decorator_list=[])
            namespace = {}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])),
                         '<closure-snapshot>', 'exec'), namespace)
            start = text.index('        self.closure_phase = int(')
            end = text.index('        library = ', start)
            block = '\n'.join(line[8:] for line in text[start:end].splitlines())
            for phase in range(8):
                instance = namespace['Snapshot']()
                env = SimpleNamespace(environ={'H100_CLOSURE_PHASE': str(phase)})
                exec(block, dict(self=instance, os=env, torch=Torch, hashlib=hashlib))
                self.assertEqual(instance.selection_cpu, plan.mask(arm, phase))
                self.assertEqual(instance.selection_sha256, plan.mask_sha256(arm, phase))
                state = instance.closure_state()
                self.assertEqual(state['protocol'], plan.PROTOCOL)
                self.assertEqual((state['arm'], state['phase']), (arm, phase))
                self.assertEqual(state['inverse_h128_per_rank'], 1024 if arm == 'CM' else 7168)
                self.assertEqual(state['wire_bytes_per_rank'], plan.WIRE_BYTES_PER_RANK)
            for invalid in ('-1', '8', 'non-integer'):
                with self.assertRaises(ValueError):
                    exec(block, dict(self=namespace['Snapshot'](),
                         os=SimpleNamespace(environ={'H100_CLOSURE_PHASE': invalid}),
                         torch=Torch, hashlib=hashlib))

    def test_decoder_factorial_has_same_fp32_accumulation_and_distinct_profile_names(self):
        decoders = {files[source.DECODER] for files in self.generated.values()}
        self.assertEqual(len(decoders), 1)
        decoder = decoders.pop()
        self.assertEqual(decoder.count('__float2bfloat16('), 1)
        self.assertIn('quant[group] / fmaxf(adaptive[group], 1e-12f)', decoder)
        self.assertIn('if (has_compressed) taco_closure_inverse(compressed)', decoder)
        for arm, files in self.generated.items():
            kind = 'merged' if arm[1] == 'M' else 'per_source'
            runtime = files[source.RUNTIME]
            self.assertIn(f'__global__ void taco_closure_decode_{kind}(', runtime)
            self.assertIn(f'taco_closure_decode_{kind}<<<taco_groups(c) / 4, 128, 0, stream>>>(', runtime)

    @unittest.skipUnless(shutil.which('g++'), 'host compiler required')
    def test_actual_cpp_mask_and_configure_all_ranks_phases_shapes(self):
        for arm, files in self.generated.items():
            header, runtime = files[source.HEADER], files[source.RUNTIME]
            code = '#include <cassert>\n#include <cstdint>\n#include <cstring>\n'
            code += '#define __host__\n#define __device__\n'
            code += 'enum {cudaSuccess=0,cudaErrorInvalidValue=1};\n'
            code += definition(header, 'struct FluxTacoConfig') + ';\n'
            code += header[header.index('// GF(8)'):]
            code += 'static thread_local FluxTacoConfig active;\n'
            for anchor in ('extern "C" FluxTacoConfig taco_config()', 'extern "C" void taco_reset()',
                           'extern "C" int taco_placement()', 'extern "C" int taco_configure(',
                           'extern "C" const char* taco_closure_protocol()',
                           'extern "C" const char* taco_closure_arm()'):
                code += definition(runtime, anchor) + '\n'
            code += f'''
int main() {{
  unsigned char storage[8]; void* peers[8];
  for(int i=0;i<8;i++) peers[i]=storage+i;
  assert(std::strcmp(taco_closure_protocol(), "{source.PROTOCOL}")==0);
  assert(std::strcmp(taco_closure_arm(), "{arm}")==0);
  const int twice[8]={{0,2,4,6,3,1,7,5}};
  for(int phase=0;phase<8;phase++) for(int rank=0;rank<8;rank++) {{
    taco_reset(); assert(taco_configure(peers,rank,8,2048,4096,phase)==0);
    assert(taco_configure(peers,rank,8,2048,4096,phase)!=0);
    auto c=taco_config(); assert(c.closure_phase==phase);
    for(int s=0;s<8;s++) {{
      int total=0;
      for(int d=0;d<8;d++) {{
        int link=0;
        for(int t=0;t<64;t++) {{
          int target={'d' if arm[0] == 'C' else '(twice[s]^twice[d]^d)'};
          bool expected=s!=d && ((t+phase)%8)==target;
          bool got=taco_closure_selected(c,s,d,d*64+t);
          assert(got==expected); link+=got;
        }}
        assert(link==(s==d?0:8)); total+=link;
      }}
      assert(total==56);
    }}
  }}
  taco_reset();
  assert(taco_configure(peers,0,8,2048,4096,-1)!=0);
  assert(taco_configure(peers,0,8,2048,4096,8)!=0);
  assert(taco_configure(peers,0,4,2048,4096,0)!=0);
  assert(taco_configure(peers,0,8,4096,4096,0)!=0);
  assert(taco_configure(peers,0,8,2048,2048,0)!=0);
  assert(taco_configure(peers,8,8,2048,4096,0)!=0);
  assert(taco_configure(nullptr,0,8,2048,4096,0)!=0);
  peers[7]=nullptr; assert(taco_configure(peers,0,8,2048,4096,0)!=0);
  assert(!taco_config().enabled);
}}
'''
            with tempfile.TemporaryDirectory(prefix='closure-host-') as directory:
                path = Path(directory)
                (path / 'check.cc').write_text(code)
                subprocess.run(['g++', '-std=c++17', '-O2', str(path / 'check.cc'),
                                '-o', str(path / 'check')], check=True)
                subprocess.run([str(path / 'check')], check=True)


if __name__ == '__main__':
    unittest.main()

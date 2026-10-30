"""MA6-v3 preserves F4 arithmetic, mask bits, and complete warp ownership."""
import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from common import HERE
import ma4_f_sources as f4
import ma6_sources as source


class ReceiverSourceTests(unittest.TestCase):
    def setUp(self):
        self.original = f4.decoder((HERE / 'ma2_c_decode.cuh').read_text())

    def candidate(self, arm):
        text = source.decoder(self.original, arm)
        self.assertTrue(text.startswith(self.original + '\n'))
        self.assertEqual(text.count(f4.CONSUMER_STORE), 1)
        return text[len(self.original) + 1:]

    def test_layout_only_changes_group_ownership_and_keeps_full_f4_arithmetic(self):
        candidate = self.candidate('F4-layout')
        normalized = candidate.replace(source.kernel_name('attn', arm='F4-layout'),
                                       'taco_ma4f_decode_consumer_kernel')
        normalized = normalized.replace(source.GROUP_MAPPING, 'blockIdx.x * 4 + threadIdx.x / 32')
        self.assertEqual(normalized, self.original[self.original.index('__global__ void taco_ma4f'):])
        self.assertNotIn('__syncthreads', candidate)
        self.assertNotIn('row_offset', candidate)

    def test_mask_only_replaces_selection_loads_and_keeps_full_f4_arithmetic(self):
        candidate = self.candidate('F4-mask')
        normalized = candidate.replace(source.kernel_name('attn', arm='F4-mask'),
                                       'taco_ma4f_decode_consumer_kernel')
        normalized = normalized.replace(', const uint8_t* packed_mask', '')
        old_mask = self.original[self.original.index('  bool selected[8];'):
                                 self.original.index('  if (!any_quant) {')]
        normalized = normalized.replace('  const uint8_t source_mask = packed_mask[physical_tile];\n', old_mask)
        normalized = normalized.replace('if (source_mask == 0)', 'if (!any_quant)')
        normalized = normalized.replace('((source_mask >> src) & 1) != 0', 'selected[src]')
        self.assertEqual(normalized, self.original[self.original.index('__global__ void taco_ma4f'):])
        self.assertNotIn('selected[', candidate)
        self.assertEqual(candidate.count('packed_mask['), 1)

    def test_layout_covers_every_group_once_and_ctas_share_physical_tile(self):
        groups = [block // 32 * 128 + block % 32 + warp * 32
                  for block in range(2048) for warp in range(4)]
        self.assertEqual(sorted(groups), list(range(8192)))
        for rank in range(8):
            for block in range(2048):
                tiles = {(rank * 2 + (g // 32) // 128) * 32 + g % 32
                         for g in groups[block * 4:block * 4 + 4]}
                self.assertEqual(len(tiles), 1)

    @unittest.skipUnless(shutil.which('g++'), 'C++ compiler required for production packer')
    def test_actual_cpp_packer_matches_every_legal_mask_byte_and_rejects_local_or_nonbinary(self):
        # Every destination tile and every 7-bit remote-source combination. This
        # tests the exact native helper, not a duplicate Python implementation.
        code = '''#include "ma6_mask.h"
#include <array>
int main() {
  std::array<uint8_t,4096> selected{};
  std::array<uint8_t,512> packed{};
  for (int bits=0; bits<256; ++bits) {
    for (int tile=0; tile<512; ++tile)
      for (int src=0; src<8; ++src)
        selected[src*512+tile] = src!=tile/64 && ((bits>>src)&1);
    if (!taco_ma6_pack_mask(selected.data(),packed.data())) return 1;
    for (int rank=0; rank<8; ++rank) for (int group=0; group<8192; ++group) {
      const int tile=(rank*2+(group/32)/128)*32+group%32;
      for (int src=0; src<8; ++src)
        if (((packed[tile]>>src)&1) != (src!=rank && selected[src*512+tile])) return 2;
    }
  }
  selected.fill(0); selected[0]=1;
  if (taco_ma6_pack_mask(selected.data(),packed.data())) return 3;
  selected.fill(0); selected[512]=2;
  if (taco_ma6_pack_mask(selected.data(),packed.data())) return 4;
  return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='ma6-mask-unit-') as directory:
            root = Path(directory)
            (root/'ma6_mask.h').write_text(source.PACK_MASK)
            (root/'check.cc').write_text(code)
            subprocess.run(['g++','-std=c++17','-O2',str(root/'check.cc'),'-o',str(root/'check')],check=True)
            subprocess.run([str(root/'check')],check=True)

    def test_native_identity_matches_fixed_geometry_and_distinct_mechanisms(self):
        original = 'extern "C" int taco_ma4f_protocol_version() { return 1; }\n'
        for arm in source.VARIANTS[1:]:
            changed = source.runtime(original, arm)
            self.assertTrue(changed.startswith(original))
            self.assertIn(source.kernel_name('attn', arm=arm)+'<<<2048, 128, 0, stream>>>', changed)
            self.assertIn('taco_ma6_protocol_version() { return 3; }', changed)
            self.assertIn('taco_ma6_receiver_variant() { return '+str(source.RECEIVER_VARIANTS[arm])+'; }', changed)
            for value in source.BASELINE_TUNING.values():
                self.assertIn('return k == 512 || k == 2048 ? '+str(value)+' : -2;', changed)

    def test_mask_binding_is_once_only_private_and_outside_forward(self):
        methods = source.MASK_METHODS
        self.assertIn('cr2_calls == 0', methods)
        self.assertIn('cr2_initializations == 0 && ma6_mask_bindings == 0', methods)
        self.assertIn('selected.data_ptr() == owned_taco.selected', methods)
        self.assertIn('selection_cpu.equal(selected.cpu())', methods)
        self.assertIn('stream == taco_stream', methods)
        self.assertIn('packed_cpu.to(selected.device(), at::kByte, false, true)', methods)
        self.assertIn('ma6_selected_owner = std::move(selected)', methods)
        frontend = source.frontend('        self.option = cpp_mod.ReduceScatterOption()')
        self.assertEqual(frontend.count('ma6_bind_packed_mask('), 1)

    def test_v3_rejects_geometry_changes_and_noninteger_values(self):
        for key, value in [('decoder_threads',True),('decoder_threads',256),
                           ('decoder_blocks',512),('decoder_rows',4)]:
            tuning = copy.deepcopy(source.DEFAULT_TUNING)
            tuning['attn'][key] = value
            with self.assertRaisesRegex(ValueError,'fixes decoder geometry'):
                source.tuning_config(tuning)
        with self.assertRaises(ValueError):
            source.tuning_config({'attn':dict(source.BASELINE_TUNING)})
        copied = source.tuning_config()
        copied['attn']['decoder_blocks'] = 1
        self.assertEqual(source.DEFAULT_TUNING['attn']['decoder_blocks'],2048)


if __name__ == '__main__':
    unittest.main()

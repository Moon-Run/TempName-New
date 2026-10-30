"""CPU interoperability tests for the actual native wire contract and transforms."""
import copy
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import rtx5090_loader
import local_sources
import nccl_transport
import ma11_build

HERE = Path(__file__).resolve().parent


class WireContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('g++')
        if compiler is None:
            raise unittest.SkipTest('C++ compiler required for native wire interoperability')
        cls.temporary = tempfile.TemporaryDirectory(prefix='rtx5090-layout-contract-')
        cls.root = Path(cls.temporary.name)
        program = cls.root/'check.cc'
        program.write_text('''#include "local_layout.h"
#include <array>
#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) return 64;
  std::ifstream input(argv[1]);
  int rank, has_mask; int64_t tx_bytes, rx_bytes;
  input >> rank >> has_mask >> tx_bytes >> rx_bytes;
  std::array<unsigned char, 4096> mask{};
  std::array<int64_t, 512> tx{}, rx{};
  if (has_mask) for (auto& value : mask) { int x; input >> x; value = x; }
  for (auto& value : tx) input >> value;
  for (auto& value : rx) input >> value;
  if (!input) return 65;
  try {
    rtx5090_local_layout::validate(rank, has_mask ? mask.data() : nullptr,
        tx.data(), rx.data(), tx_bytes, rx_bytes);
  } catch (const std::exception& error) {
    std::cerr << error.what(); return 3;
  }
  return 0;
}
''')
        cls.binary = cls.root/'check'
        subprocess.run([compiler, '-std=c++17', '-O2', '-I'+str(HERE), str(program), '-o', str(cls.binary)],
                       check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def check_layout(self, rank, mask, layout):
        values = [rank, int(mask is not None), layout['tx_bytes'], layout['rx_bytes']]
        if mask is not None:
            values += [value for row in mask for value in row]
        values += layout['tx_offsets'] + layout['rx_offsets']
        case = self.root/'case.txt'
        case.write_text(' '.join(map(str, values)))
        return subprocess.run([str(self.binary), str(case)], capture_output=True, text=True)

    def test_python_layout_matches_actual_cpp_validator_for_every_rank(self):
        masks = [None, [[0]*512 for _ in range(8)],
                 [[int(tile//64 != src) for tile in range(512)] for src in range(8)],
                 [[int(tile//64 != src and (tile*7+src*13)%19 < 3)
                   for tile in range(512)] for src in range(8)]]
        for mask in masks:
            for rank in range(8):
                result = self.check_layout(rank, mask, nccl_transport.wire_layout(mask, rank))
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_validator_rejects_self_quantization_stale_offsets_and_lengths(self):
        mask = [[int(tile//64 != src and (tile+src)%17 == 0)
                 for tile in range(512)] for src in range(8)]
        original = nccl_transport.wire_layout(mask, 3)
        mutations = [lambda value: value['tx_offsets'].__setitem__(192, 0),
                     lambda value: value['rx_offsets'].__setitem__(64, value['rx_offsets'][64]+1),
                     lambda value: value.update(tx_bytes=value['tx_bytes']-1),
                     lambda value: value.update(rx_offsets=value['tx_offsets'])]
        for change in mutations:
            layout = copy.deepcopy(original)
            change(layout)
            result = self.check_layout(3, mask, layout)
            self.assertEqual(result.returncode, 3)
        for source, tile, value in ((0, 0, 1), (2, 0, 2)):
            changed = copy.deepcopy(mask)
            changed[source][tile] = value
            result = self.check_layout(3, changed, original)
            self.assertEqual(result.returncode, 3)


class SourceTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='rtx5090-local-sources-')
        cls.root = Path(cls.temporary.name)/'reference'
        ma11_build.prepare(cls.root)
        cls.sources = {'ma11': cls.root/'F4-cache-zcount-csoftmax',
                       'cr3': cls.root/'reference/F4'}
        cls.sources.update({arm: cls.root/'reference/reference/reference'/arm.upper()
                            for arm in ('a', 'r', 'c')})

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_original_quantization_decoder_and_abi_are_preserved(self):
        for arm, source in self.sources.items():
            changes = local_sources.sources(source, arm)
            runtime = (source/'src/gemm_rs/taco_runtime.cu').read_text()
            self.assertTrue(changes['src/gemm_rs/taco_runtime.cu'].startswith(runtime))
            self.assertNotIn('src/gemm_rs/taco_runtime.h', changes)
            self.assertNotIn('src/gemm_rs/taco_codec.cuh', changes)
            self.assertNotIn('src/gemm_rs/ma4_f_decode.cuh', changes)
            # Reversing only address/initialization substitutions restores the
            # exact frozen producer, including Hadamard order and BF16 rounding.
            epilogue = changes['src/gemm_rs/epilogue_evt.hpp']
            epilogue = epilogue.replace(
                'unsigned char* slot = c.peers[dst_rank];  // Local tx slot; NCCL moves it after GEMM.',
                'unsigned char* slot = c.peers[dst_rank] + c.rank * taco_source_bytes(c);')
            epilogue = epilogue.replace(
                'int target_line = cur_row % M_lines_per_rank;  // Destination-indexed local staging.',
                'int target_line = params.rank * M_lines_per_rank + cur_row % M_lines_per_rank;')
            epilogue = epilogue.replace('Params params{};', 'Params params;').replace(
                'params.has_barrier_ptr ? params.barrier_ptr_aux[dst_rank] : nullptr;',
                'params.barrier_ptr_aux ? params.barrier_ptr_aux[dst_rank] : nullptr;').replace(
                'The producer stream orders pack after GEMM; NCCL completion orders unpack and DQ.',
                'The existing GemmRS group barrier publishes completion before DQ.')
            self.assertEqual(epilogue, (source/'src/gemm_rs/epilogue_evt.hpp').read_text())
            self.assertIn('#define RTX5090_LOCAL_SELECTIVE '+str(int(arm not in ('a', 'r'))),
                          changes['src/gemm_rs/local_transport.cu'])
        ma11_build.verify(self.root)

    def test_legacy_ipc_constructor_is_not_registered_or_compiled(self):
        for arm, source in self.sources.items():
            changes = local_sources.sources(source, arm)
            self.assertNotIn('py::class_<GemmRSOpCls>', changes['src/pybind/gemm_rs.cc'])
            self.assertIn('list(REMOVE_ITEM THS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/ths_op/gemm_reduce_scatter.cc")',
                          changes['src/gemm_rs/CMakeLists.txt'])
            self.assertIn('rtx5090_local_*;', changes['src/cuda/version.ld'])
            self.assertIn('src/gemm_rs/local_layout.h', changes)

    def test_unknown_source_edit_or_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            local_sources.sources(self.sources['ma11'], 'unknown')
        path = self.sources['ma11']/'src/gemm_rs/epilogue_evt.hpp'
        original = path.read_bytes()
        try:
            path.write_text(path.read_text().replace('Params params;', 'Params params_modified;'))
            with self.assertRaisesRegex(ValueError, 'anchor changed'):
                local_sources.sources(self.sources['ma11'], 'ma11')
        finally:
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()

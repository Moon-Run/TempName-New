"""MA5 work ownership and unchanged per-group numerical code contracts."""
import unittest

from common import HERE
import ma5_sources as source


class SchedulingTests(unittest.TestCase):
    def test_every_output_group_has_exactly_one_warp_owner(self):
        # Full grids and grids with a nonuniform final iteration are covered.
        for threads in (64, 128, 256, 512):
            warps = threads // 32
            for blocks in (1, 132, 264, min(528, 8192 // warps), 8192 // warps):
                with self.subTest(threads=threads, blocks=blocks):
                    owners = [0] * 8192
                    for block in range(blocks):
                        for warp in range(warps):
                            for group in range(block * warps + warp, 8192, blocks * warps):
                                owners[group] += 1
                    self.assertEqual(owners, [1] * 8192)

    def test_numeric_body_is_exactly_the_existing_cr2_body(self):
        original = (HERE / 'ma2_c_decode.cuh').read_text()
        changed = source.decoder(original, source.DEFAULT_TUNING)
        old_body = original.split('  const int lane = threadIdx.x & 31;', 1)[1]
        old_body = old_body.replace('  const int group = blockIdx.x * 4 + threadIdx.x / 32;\n', '')
        new_body = changed.split('  const int lane = threadIdx.x & 31;', 1)[1]
        new_body = new_body.split('// MA5: every warp owns', 1)[0].rstrip()
        self.assertEqual(old_body.rstrip(), new_body)
        # The BF16 return terminates the helper, not the grid-stride kernel.
        helper, kernel = changed.split('__global__ void taco_ma5_decode_kernel', 1)
        self.assertIn('    return;', helper)
        self.assertNotIn('return;', kernel)
        self.assertIn('group += gridDim.x * kWarps', kernel)

    def test_invalid_or_incomplete_tuning_is_rejected(self):
        for fields in (dict(decoder_threads=96), dict(decoder_blocks=0),
                       dict(decoder_blocks=1025), dict(gemm_sms=0),
                       dict(gemm_tile_k=128), dict(decoder_threads=True)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                source.tuning_config(dict(source.DEFAULT_TUNING, **fields))
        with self.assertRaises(ValueError):
            source.tuning_config({'decoder_threads': 256})

    def test_tuning_does_not_change_legacy_rs_num_blocks(self):
        fixture = ('// cr2_op->run(args, workspace, stream);\n'
                   'opt.num_blocks == 6\n'
                   '.barrier_ptrs = barrier_ptrs.data(), .avail_sms = -1,')
        result = source.wrapper(fixture, dict(source.DEFAULT_TUNING, gemm_sms=128))
        self.assertIn('opt.num_blocks == 6', result)
        self.assertIn('.avail_sms = 128', result)

    def test_mapping_probe_uses_candidate_tile_and_scheduling_width(self):
        fixture = ('k%(32*tp)\n{128,128,32},1,active,\n'
                   'prop.multiProcessorCount,prop.multiProcessorCount,2,2,2,8);')
        self.assertEqual(source.mapping(fixture, source.DEFAULT_TUNING), fixture)
        result = source.mapping(fixture, dict(source.DEFAULT_TUNING, gemm_tile_k=64, gemm_sms=128))
        self.assertIn('{128,128,64},1,active,', result)
        self.assertIn('k%(64*tp)', result)
        self.assertIn('prop.multiProcessorCount < 128', result)


if __name__ == '__main__':
    unittest.main()

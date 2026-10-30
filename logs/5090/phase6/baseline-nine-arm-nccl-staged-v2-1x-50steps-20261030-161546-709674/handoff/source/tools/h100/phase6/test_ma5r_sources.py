"""Numerical source preservation and immutable role selection for MA5-R."""
import copy
from types import SimpleNamespace
import unittest

from common import HERE
import ma2_cr2_sources as cr2
import ma5_sources as uniform
import ma5r_sources as split
import ma5r_bench as bench


class RoleSelectionTests(unittest.TestCase):
    def test_mlp_kernel_and_attention_math_are_preserved(self):
        original = (HERE / 'ma2_c_decode.cuh').read_text()
        changed = split.decoder(original, split.DEFAULT_TUNING)
        self.assertTrue(changed.startswith(original + '\n'))
        attention = uniform.decoder(original, uniform.DEFAULT_TUNING)
        attention = attention[attention.index('__device__ __forceinline__ void taco_ma5_decode_group('):]
        attention = attention.replace('taco_ma5_decode_group', 'taco_ma5r_decode_group')
        attention = attention.replace('taco_ma5_decode_kernel', 'taco_ma5r_attention_decode_kernel')
        self.assertEqual(changed[len(original) + 1:], attention)
        self.assertEqual(changed.count('__global__ void taco_ma2c_decode_kernel'), 1)

    def test_role_is_bound_once_instead_of_re_reading_tls(self):
        source = '  const int cr2_k = taco_cr2_bound_k();\n' + cr2.CR2_METHOD
        changed = split.wrapper(source, split.DEFAULT_TUNING)
        self.assertIn('const decltype(&taco_decode_reduce) ma5r_decode', changed)
        self.assertIn('cr2_k == 512 ? &taco_ma5r_decode_attention : &taco_decode_reduce', changed)
        self.assertEqual(changed.count('taco_cr2_bound_k()'), 1)
        body = changed[changed.index('  cr2_forward('):]
        self.assertNotIn('taco_cr2_bound_k()', body)
        self.assertIn('ma5r_decode(\n          owned_taco', body)
        self.assertIn('cr2_k == 512 || cr2_k == 2048', body)
        self.assertIn('group_barrier.barrier_all(stream)', body)

    def test_mlp_and_producer_tuning_cannot_drift(self):
        for role, key, value in (('mlp', 'decoder_blocks', 528), ('mlp', 'decoder_threads', 256),
                                 ('attn', 'gemm_tile_k', 64), ('attn', 'gemm_sms', 128)):
            tuning = copy.deepcopy(split.DEFAULT_TUNING)
            tuning[role][key] = value
            with self.assertRaises(ValueError):
                split.tuning_config(tuning)
        tuning = split.tuning_config()
        tuning['attn']['decoder_blocks'] = 1
        self.assertEqual(split.DEFAULT_TUNING['attn']['decoder_blocks'], 528)

    def test_native_role_evidence_rejects_wrong_instance_k(self):
        class Function:
            def __init__(self, values): self.values = values
            def __call__(self, *args): return self.values[args]
        fields = {'taco_ma5r_protocol_version': Function({(): 1})}
        for key in split.BASELINE_TUNING:
            fields['taco_ma5r_' + key] = Function({(512,): split.DEFAULT_TUNING['attn'][key],
                                                   (2048,): split.BASELINE_TUNING[key]})
        states = [{'bound_k': 512}, {'bound_k': 512}]
        op = SimpleNamespace(slots=[SimpleNamespace(lib=SimpleNamespace(**fields))],
                             cr2_submission_state=lambda: states)
        plan = dict(tuning=split.DEFAULT_TUNING)
        self.assertEqual(bench.native_tuning(op, 'T', plan, 'attn'), split.DEFAULT_TUNING['attn'])
        with self.assertRaisesRegex(ValueError, 'immutable instance K'):
            bench.native_tuning(op, 'T', plan, 'mlp')
        states[:] = [{'bound_k': 2048}, {'bound_k': 2048}]
        self.assertEqual(bench.native_tuning(op, 'T', plan, 'mlp'), split.BASELINE_TUNING)


if __name__ == '__main__':
    unittest.main()

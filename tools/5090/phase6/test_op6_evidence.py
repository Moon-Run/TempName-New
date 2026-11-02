import copy
import unittest

import op6_evidence as evidence


def rows():
    keys = ('gpu_uuid torch cuda nccl python hardware model transport '
            'initial_parameters_sha256 initial_rng_sha256 tokens_sha256 '
            'parameter_count_local padded_vocab_size communication losses grad_norms '
            'final_parameters_sha256 trainable_parameters initial_loss warmup_last_loss '
            'skips cpu_affinity_policy cpu_affinity seed precision target_modules '
            'op3_implementation op3_optimization_stack codec_identity smoke_output_sha256').split()
    result = []
    for rank in range(8):
        row = dict.fromkeys(keys, 'same')
        row.update(rank=rank, parameter_trajectory=[dict(step=0,
            gradients={'bias': 'gradient-hash'}, parameters={'weight': 'weight-hash'})])
        result.append(row)
    return result


class PairEvidenceTests(unittest.TestCase):
    def test_identical_pairs_pass_all_modes(self):
        for mode in ('smoke', 'profile', 'timing'):
            result = evidence.compare_rows(rows(), rows(), mode)
            self.assertTrue(result['concurrent'])
            self.assertTrue(result['smoke_gradient_hash_equality_required'])

    def test_same_run_gradient_mismatch_is_not_historical_exception(self):
        candidate = rows()
        candidate[4]['parameter_trajectory'][0]['gradients']['bias'] = 'different'
        with self.assertRaisesRegex(ValueError, 'gradients'):
            evidence.compare_rows(rows(), candidate, 'smoke')

    def test_rejects_numerical_and_identity_drift(self):
        for key in ('losses', 'grad_norms', 'final_parameters_sha256',
                    'initial_rng_sha256', 'communication', 'cpu_affinity',
                    'smoke_output_sha256', 'op3_implementation'):
            candidate = rows()
            candidate[2][key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                evidence.compare_rows(rows(), candidate, 'smoke')

    def test_rejects_missing_or_reordered_ranks(self):
        for candidate in (rows()[:-1], rows()[::-1]):
            with self.assertRaises(ValueError):
                evidence.compare_rows(rows(), candidate, 'timing')

    def test_inputs_are_not_mutated(self):
        original = rows()
        before = copy.deepcopy(original)
        evidence.compare_rows(original, rows(), 'smoke')
        self.assertEqual(original, before)


if __name__ == '__main__':
    unittest.main()

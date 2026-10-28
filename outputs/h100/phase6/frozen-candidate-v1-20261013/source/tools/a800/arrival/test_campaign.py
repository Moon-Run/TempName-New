import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = load('campaign')
labels = load('validate_calibration')


class CampaignTests(unittest.TestCase):
    def test_model_gate_requires_both_shapes_and_repetitions(self):
        points = [dict(policy='remote_arrival', baseline='remote_first', M=2048, N=2048,
                       K_global=k, ci95_low=.1) for k in (2048, 8192)]
        rounds = [dict(comparisons=points), dict(comparisons=points)]
        self.assertEqual(campaign.model_gate(rounds), ['remote_arrival'])
        self.assertEqual(campaign.model_gate(rounds[:1]), [])
        points[1]['ci95_low'] = -.1
        self.assertEqual(campaign.model_gate(rounds), [])

    def test_ties_and_rank_correlation(self):
        self.assertEqual(labels.ranks([2, 1, 2]), [1.5, 0, 1.5])
        self.assertAlmostEqual(labels.spearman([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(labels.spearman([1, 1], [2, 3]))


if __name__ == '__main__':
    unittest.main()

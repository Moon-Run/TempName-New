"""Scientific acceptance boundaries, separate from the full GPU run verifier."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('assessment', Path(__file__).with_name('assess.py'))
assessment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessment)


class AssessmentTest(unittest.TestCase):
    def evaluate(self, reductions, target=4., lower=1., upper=8.):
        # Only the statistical gate is under test here. Real runs must also pass
        # their frozen verifier; the separate GPU campaigns exercise that path.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'scripts').mkdir()
            policies = list(assessment.REQUIRED + assessment.REFERENCES + assessment.CANDIDATES)
            config = {} if target is None else dict(acceptance=dict(version='test', target_percent=target))
            (root/'scripts/config.json').write_text(json.dumps(config))
            info = dict(policies=policies, blocks=6, repetitions=2,
                        repo=str(root), artifact_root=str(root))
            (root/'submission.json').write_text(json.dumps(info))
            frozen = root/'outputs/a800/phase4/taco-fused-warp-20261003/taco/libflux_cuda.so'
            frozen.parent.mkdir(parents=True)
            frozen.write_bytes(b'test baseline identity')
            (root/'build').mkdir()
            (root/'build/manifest.json').write_text(json.dumps(dict(
                source_manifests={'taco_fused': {}},
                policies={'taco_fused': dict(library_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())})))
            for i in (1, 2):
                rows = []
                for candidate in assessment.CANDIDATES:
                    row = dict(policy=candidate, median_ms_per_step=100.)
                    for baseline in assessment.REQUIRED + assessment.REFERENCES:
                        value = reductions.get((candidate, i, baseline), 4.2)
                        row[f'latency_reduction_vs_{baseline}_percent'] = value
                        row[f'ci95_low_vs_{baseline}'] = lower
                        row[f'ci95_high_vs_{baseline}'] = upper
                    rows.append(row)
                out = root/f'results/model-{i}'
                out.mkdir(parents=True)
                (out/'analysis.json').write_text(json.dumps(dict(summary=rows)))
            verifier = Mock()
            old_path = sys.path[:]
            try:
                with patch.dict(sys.modules, verify=types.SimpleNamespace(verify=verifier)), contextlib.redirect_stdout(io.StringIO()):
                    result = assessment.assess(root)
            finally:
                sys.path[:] = old_path
            verifier.assert_called_once_with(root)
            return result

    def test_point_estimates_and_both_rounds_are_required(self):
        self.assertTrue(self.evaluate({})['target_achieved'])
        # A high CI bound cannot rescue a sub-target point estimate against the
        # harder baseline, even when the other baseline clears the threshold.
        reductions = {(c, i, 'taco_fused'): 3.99 for c in assessment.CANDIDATES for i in (1, 2)}
        self.assertFalse(self.evaluate(reductions, upper=5.)['target_achieved'])
        # Different winners in the two rounds do not establish one repeated win.
        reductions = {(assessment.CANDIDATES[0], 2, 'taco_fused'): 3.99,
                      (assessment.CANDIDATES[1], 1, 'native_taco'): 3.99}
        self.assertFalse(self.evaluate(reductions)['target_achieved'])

    def test_positive_intervals_and_historical_threshold(self):
        self.assertFalse(self.evaluate({}, lower=-.1)['target_achieved'])
        legacy = self.evaluate({}, target=None)
        self.assertEqual(legacy['target_percent'], 5.)
        self.assertFalse(legacy['target_achieved'])
        self.assertEqual(self.evaluate({})['target_percent'], 4.)


if __name__ == '__main__':
    unittest.main()

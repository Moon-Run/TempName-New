import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
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
    def test_model_gate_requires_both_shapes_and_both_rounds(self):
        points = [dict(policy='remote_arrival', baseline='remote_first', M=2048, N=2048,
                       K_global=k, ci95_low=.1) for k in (2048, 8192)]
        rounds = [dict(comparisons=points), dict(comparisons=points)]
        self.assertEqual(campaign.model_gate(rounds), ['remote_arrival'])
        points[1]['ci95_low'] = -.01
        self.assertEqual(campaign.model_gate(rounds), [])
        self.assertEqual(campaign.model_gate(rounds[:1]), [])

    def test_rank_correlation_and_ties(self):
        self.assertEqual(labels.ranks([2, 1, 2]), [1.5, 0, 1.5])
        self.assertAlmostEqual(labels.spearman([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(labels.spearman([1, 1], [2, 3]))

    def test_outer_process_holds_after_success_failure_and_before_ready(self):
        for outcome, expected in [(0, 'campaign_completed_holding'), (7, 'campaign_failed_holding'),
                                  (None, 'waiting_for_campaign')]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bin_dir = root/'bin'
                bin_dir.mkdir()
                commands = dict(scontrol='#!/bin/bash\nexit 0\n',
                    srun='#!/bin/bash\nif [[ "$*" == *campaign-ready.sh* ]]; then exit '+str(outcome or 0)+'; fi\nexit 0\n',
                    sleep='#!/bin/bash\ncat "$TEST_ROOT/allocation-state.txt" > "$TEST_ROOT/observed-state.txt"\ntouch "$TEST_ROOT/RELEASE"\n')
                for name, content in commands.items():
                    p = bin_dir/name
                    p.write_text(content)
                    p.chmod(0o755)
                if outcome is not None:
                    (root/'campaign-ready.sh').write_text('#!/bin/bash\nexit 0\n')
                env = dict(os.environ, PATH=str(bin_dir)+':'+os.environ['PATH'],
                           SLURM_JOB_ID='123', TEST_ROOT=str(root))
                subprocess.run(['bash', str(HERE/'allocation.sbatch'), str(root), str(root)],
                               env=env, check=True, timeout=10)
                self.assertEqual((root/'observed-state.txt').read_text().strip(), expected)
                if outcome is not None:
                    self.assertEqual((root/'allocation-123/campaign-exit-code.txt').read_text().strip(), str(outcome))


if __name__ == '__main__':
    unittest.main()

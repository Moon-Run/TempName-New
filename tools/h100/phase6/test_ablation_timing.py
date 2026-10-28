"""Timing seals must separate new 60-step sessions from historical evidence."""
from pathlib import Path
import tempfile
import unittest

import ablation_timing as timing
from common import write


class TimingTests(unittest.TestCase):
    def test_new_timing_is_sealed_and_rejects_cfg_or_submission_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); info = dict(execution_timing=timing.execution_timing())
            (root/'provenance').mkdir(); (root/'scripts').mkdir()
            write(root/'provenance/preparation.json', info)
            write(root/'scripts/config.json', timing.execution_timing())
            self.assertEqual(timing.validate(root, info)['timed_steps'], 60)
            write(root/'scripts/config.json', dict(warmup_steps=10, timed_steps=20))
            with self.assertRaisesRegex(ValueError, 'sealed'): timing.validate(root, info)
            info['execution_timing']['timed_steps'] = 20
            with self.assertRaisesRegex(ValueError, 'seal changed'): timing.validate(root, info)
            write(root/'scripts/config.json', timing.execution_timing())
            with self.assertRaisesRegex(ValueError, 'seal changed'): timing.validate(root, {})

    def test_legacy_without_seal_accepts_only_twenty_and_cannot_enter_new_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'provenance').mkdir(); (root/'scripts').mkdir()
            write(root/'provenance/preparation.json', {})
            write(root/'scripts/config.json', dict(warmup_steps=10, timed_steps=20))
            self.assertEqual(timing.validate(root, {})['timed_steps'], 20)
            with self.assertRaises(ValueError): timing.require_new({})
            write(root/'scripts/config.json', timing.execution_timing())
            with self.assertRaises(ValueError): timing.validate(root, {})


if __name__ == '__main__': unittest.main()

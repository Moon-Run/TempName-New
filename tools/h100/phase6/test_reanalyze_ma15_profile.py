"""Offline recovery must preserve original evidence and fail before publication."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reanalyze_ma15_profile as replay


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


class ReanalysisTests(unittest.TestCase):
    def state(self):
        return dict(protocol='h100-ma15-attention-graph-detached-v1', status='failed',
                    stage='fullstep-profile', timing_report_verified=True,
                    expected_timing_windows=16, expected_formal_optimizer_steps=800,
                    reports={'fullstep': 'fullstep/report.json'}, error='original failure')

    def test_frozen_source_tamper_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'fullstep'
            source = root / 'scripts/worker.py'
            source.parent.mkdir(parents=True)
            source.write_text('original = True\n')
            put(root / 'plan.json', {'source_files': {'scripts/worker.py': replay.sha(source)}})
            replay.verify_sources(root)
            source.write_text('original = False\n')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                replay.verify_sources(root)
            outside = root.parent / 'outside.py'
            outside.write_text('external = True\n')
            put(root / 'plan.json', {'source_files': {'../outside.py': replay.sha(outside)}})
            with self.assertRaisesRegex(ValueError, 'escaped'):
                replay.verify_sources(root)

    def test_failed_validation_never_changes_failed_state(self):
        with tempfile.TemporaryDirectory() as temp:
            batch = Path(temp)
            put(batch / 'batch-state.json', self.state())
            original = (batch / 'batch-state.json').read_bytes()
            for name in ('fullstep/plan.json', 'operator/plan.json', 'fullstep/report.json',
                         'operator/report.json', 'operator/profile-summary.json'):
                put(batch / name, {})
            reader = batch / 'reader.py'
            reader.write_text('# fixture\n')
            with patch.object(replay, 'invoke', side_effect=ValueError('dependency order changed')):
                with self.assertRaisesRegex(ValueError, 'dependency order'):
                    replay.reanalyze(batch, reader)
            self.assertEqual((batch / 'batch-state.json').read_bytes(), original)
            self.assertFalse((batch / 'profile-reanalysis').exists())
            self.assertFalse((batch / 'fullstep/profile-summary.json').exists())

    def test_publication_preserves_failed_state_and_marks_offline_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            batch = Path(temp)
            state = self.state()
            put(batch / 'batch-state.json', state)
            raw = batch / 'fullstep/results/rank0.json'
            put(raw, {'fixture': True})
            original, raw_bytes = (batch / 'batch-state.json').read_bytes(), raw.read_bytes()
            report = dict(formal_optimizer_steps=800, all_samples=[{}] * 16,
                          acceptance_status={'candidate': 'positive'}, target_evaluated=True,
                          target_validated=True, target_point_observed=True,
                          increment_over_MA11_validated={'candidate': True})
            full = dict(report=report, profile={'execution_validated': True})
            replay.apply(batch, state, full, {'corrected_reader_sha256': 'fixture'})
            result = replay.read(batch / 'batch-state.json')
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['completion_mode'], 'offline_profile_reanalysis')
            self.assertEqual(result['previous_failure']['error'], 'original failure')
            self.assertEqual((batch / 'profile-reanalysis/batch-state.before.json').read_bytes(), original)
            self.assertEqual(raw.read_bytes(), raw_bytes)
            self.assertTrue((batch / 'fullstep/profile-summary.json').is_file())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                replay.apply(batch, state, full, {'corrected_reader_sha256': 'fixture'})


if __name__ == '__main__':
    unittest.main()

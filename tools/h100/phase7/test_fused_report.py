"""Round-aware report orchestration on isolated CPU fixtures, never GPU evidence.

Hardware admission and per-rank GPU validation are mocked explicitly. The real
report still discovers the frozen timing directories, reads all fixture ranks,
forms complete policy blocks and writes its evidence/counts/Markdown outputs.
"""
import contextlib
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import fused_support
import fused_protocol as routing
import fused_report as reporter
from common import read, write, sha


@contextlib.contextmanager
def report_fixture(rounds):
    with tempfile.TemporaryDirectory(prefix='phase7-report-CPU-FIXTURE-') as directory:
        root = Path(directory)
        jobs = routing.schedule(rounds=rounds)
        info = dict(protocol=routing.PROTOCOL, run_id='cpu-report-orchestration-only',
                    executable=True, source_only=False, rounds=rounds, blocks_per_round=9,
                    timing_windows=len(jobs), schedule=jobs, config={}, candidate_source_commit='fixture')
        write(root / 'submission.json', info)
        (root / 'scripts').mkdir()
        write(root / 'scripts/config.json', dict(role_plan_sha256={}))
        write(root / 'preflight.json', dict(completed=True, protocol=routing.PROTOCOL,
              submission_sha256=sha(root / 'submission.json'), files={}, fingerprints=[['cpu-fixture']]))
        for job in jobs:
            folder = root / 'results' / job['directory']
            folder.mkdir(parents=True)
            for rank in range(8):
                ms = 100 + routing.POLICIES.index(job['policy']) + job['block'] + 10*job['round'] + rank/10
                write(folder / f'rank{rank}.json', dict(elapsed_seconds=ms*20/1000))

        def read_window(root, folder, policy, mode, block, index):
            assert mode == 'timing'
            assert any(job['directory'] == str(folder.relative_to(root / 'results')) and
                       (job['policy'], job['block'], job['index']) == (policy, block, index) for job in jobs)
            return [read(folder / f'rank{rank}.json') for rank in range(8)]

        # CI mathematics is unchanged; avoid repeating its 10000 bootstrap draws
        # in tests focused on round selection and report completeness.
        def paired(logs):
            return dict(percent=100*(1-math.exp(sum(logs)/len(logs))), ci95=[-1., 1.], blocks=len(logs))

        with mock.patch.object(reporter, 'verify_inputs', return_value=info), \
                mock.patch.object(reporter, 'measurement_identity', return_value={'fixture': True}), \
                mock.patch.object(reporter, 'window', side_effect=read_window) as windows, \
                mock.patch.object(reporter, 'fingerprints', return_value=[('cpu-fixture',)]), \
                mock.patch.object(reporter, 'paired', side_effect=paired):
            yield root, info, windows


class ReportRoundTests(unittest.TestCase):
    def test_single_and_default_rounds_report_every_frozen_window(self):
        for count in (1, 2):
            with self.subTest(rounds=count), report_fixture(count) as (root, info, windows):
                result = reporter.report(root, offline=True)
                expected = {str(rep) for rep in range(1, count+1)}
                self.assertTrue(result['completed'])
                self.assertEqual(result['round_count'], count)
                self.assertEqual(result['blocks_per_round'], 9)
                self.assertEqual(set(result['rounds']), expected | {'combined'})
                self.assertEqual(set(result['sensitivity']), expected)
                self.assertEqual(len(result['all_samples']), 81*count)
                self.assertEqual(windows.call_count, 81*count)
                self.assertEqual(result['timing_windows'], 81*count)
                self.assertEqual(result['timing_rank_records'], 648*count)
                self.assertEqual(result['preflight_rank_records'], 144)
                self.assertEqual(result['profile_traces'], 72)
                self.assertEqual(result['candidate_shape_mappings'], 12)
                self.assertEqual(len(result['evidence']['timing_files']), 648*count)
                self.assertEqual(result['rounds']['combined']['blocks'], 9*count)
                self.assertEqual({row['round'] for row in result['all_samples']}, set(range(1, count+1)))
                for rep in expected:
                    self.assertEqual(result['rounds'][rep]['blocks'], 9)
                    self.assertEqual(set(result['rounds'][rep]['ms_per_step']), set(routing.POLICIES))
                    self.assertTrue(all(len(v) == 9 for v in result['rounds'][rep]['ms_per_step'].values()))
                first = result['all_samples'][0]
                self.assertAlmostEqual(first['ms_per_step'], 110.7 + routing.POLICIES.index(first['policy']))
                saved = read(root / 'report.json')
                self.assertEqual(saved['timing_windows'], 81*count)
                markdown = (root / 'report.md').read_text()
                self.assertIn('| 1 | fused_', markdown)
                if count == 1:
                    self.assertNotIn('| 2 |', markdown)

    def test_missing_timing_window_is_rejected_before_summary(self):
        with report_fixture(1) as (root, info, windows):
            shutil.rmtree(root / 'results' / info['schedule'][-1]['directory'])
            with self.assertRaises(ValueError):
                reporter.report(root, offline=True)
            windows.assert_not_called()
            self.assertFalse((root / 'report.json').exists())

    def test_extra_second_round_is_rejected_in_single_round_session(self):
        with report_fixture(1) as (root, info, windows):
            (root / 'results/round2/window-000-unrequested').mkdir(parents=True)
            with self.assertRaises(ValueError):
                reporter.report(root, offline=True)
            windows.assert_not_called()
            self.assertFalse((root / 'report.json').exists())

    def test_missing_rank_cannot_be_reported_as_completed_window(self):
        with report_fixture(1) as (root, info, windows):
            (root / 'results' / info['schedule'][-1]['directory'] / 'rank7.json').unlink()
            with self.assertRaises(FileNotFoundError):
                reporter.report(root, offline=True)
            self.assertFalse((root / 'report.json').exists())


if __name__ == '__main__':
    unittest.main()

"""CPU-only end-to-end preparation/dispatch checks; no synthetic GPU claims."""
import ast
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ablation4_build
import ablation7 as cli
import ablation7_build as builder
import ablation7_prepare as preparation
import ablation7_protocol as routing
import ablation7_report as report
import ablation_timing
from common import read, sha, write


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ablation7-CPU-FIXTURE-')
        cls.root = Path(cls.temp.name)
        cls.parent = ablation4_build.prepare(cls.root/'parent', source_only=True)
        cls.build = builder.prepare(cls.root/'build', cls.parent, source_only=True)
        cls.compare = preparation.prepare(cls.root/'compare', cls.build, Path('/not-used'), source_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_preparation_has_only_three_policies_and_50_timed_steps(self):
        info = report.verify_inputs(self.compare)
        config = read(self.compare/'scripts/config.json')
        self.assertEqual(info['execution_timing'], dict(warmup_steps=10, timed_steps=50))
        self.assertEqual(config['execution_timing'], info['execution_timing'])
        self.assertEqual((config['warmup_steps'], config['timed_steps']), (10, 50))
        self.assertEqual(list(info['variants']), list(routing.POLICIES))
        self.assertEqual(config['policies'], list(routing.POLICIES))
        self.assertEqual(info['timing_windows'], 36)
        self.assertEqual(info['budget_fractions'][routing.POLICIES[0]], 1/16)
        self.assertEqual([info['budget_fractions'][p] for p in routing.POLICIES[1:]], [1., 1.])
        self.assertEqual(ablation_timing.execution_timing(), dict(warmup_steps=10, timed_steps=60))
        plans = read(self.compare/'scripts/selection-plan.json')
        self.assertEqual(config['role_plan_sha256'], preparation.role_hashes(plans))
        for file in (self.compare/'scripts').glob('*.py'):
            ast.parse(file.read_text(), filename=str(file))

    def test_dry_run_is_read_only_and_source_only_cannot_execute(self):
        before = sha(self.compare/'submission.json')
        with mock.patch.object(subprocess, 'run', side_effect=AssertionError('Launched a process')), \
             mock.patch.object(subprocess, 'check_output', side_effect=AssertionError('Queried a GPU')), \
             contextlib.redirect_stdout(io.StringIO()):
            tasks = cli.run(self.compare)
            self.assertEqual(len(tasks), 42)
            self.assertEqual(len(list(cli.mapping_jobs(self.compare, read(self.compare/'submission.json')))), 6)
            self.assertEqual({job['policy'] for job in tasks}, set(routing.POLICIES))
            self.assertEqual(sum(job['mode'] == 'timing' for job in tasks), 36)
            with self.assertRaisesRegex(ValueError, 'Source-only'):
                cli.run(self.compare, execute=True)
        self.assertEqual(sha(self.compare/'submission.json'), before)
        self.assertFalse((self.compare/'all-state.json').exists())
        self.assertNotIn('torch', sys.modules)
        with self.assertRaises(ValueError):
            report.preflight(self.compare)
        result = subprocess.run([sys.executable, str(self.compare/'scripts/worker.py')],
            capture_output=True, text=True, env=dict(os.environ, H100_ABLATION7_EXECUTE='1'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Source-only', result.stderr)

    def test_frozen_timing_drift_is_rejected_even_with_updated_file_hash(self):
        paths = [self.compare/'scripts/config.json', self.compare/'submission.json']
        originals = [p.read_text() for p in paths]
        try:
            for field in ('warmup_steps', 'timed_steps', 'execution_timing'):
                cfg, info = read(paths[0]), read(paths[1])
                cfg.update(warmup_steps=10, timed_steps=50, execution_timing=routing.execution_timing())
                if field == 'execution_timing':
                    cfg[field]['timed_steps'] = 60
                else:
                    cfg[field] += 1
                write(paths[0], cfg)
                info['files']['scripts/config.json'] = sha(paths[0])
                write(paths[1], info)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    report.verify_inputs(self.compare)
        finally:
            for p, original in zip(paths, originals):
                p.write_text(original)

    def test_existing_outputs_and_cross_experiment_build_are_rejected(self):
        before = sha(self.compare/'submission.json')
        with self.assertRaises(FileExistsError):
            preparation.prepare(self.compare, self.build, Path('/not-used'), source_only=True)
        self.assertEqual(sha(self.compare/'submission.json'), before)
        with self.assertRaises((FileNotFoundError, ValueError)):
            preparation.prepare(self.root/'wrong', self.parent, Path('/not-used'), source_only=True)

    def test_cli_default_is_six_balanced_blocks_and_two_rounds(self):
        args = cli.parser().parse_args(['prepare', '/new', '--build', '/build', '--source-only'])
        self.assertEqual((args.blocks, args.rounds, args.nvls_enable), (6, 2, '0'))
        for stage, expected in (('preflight', 6), ('timing', 36), ('all', 42)):
            self.assertEqual(len(list(cli.jobs(read(self.compare/'submission.json'), stage))), expected)


if __name__ == '__main__':
    unittest.main()

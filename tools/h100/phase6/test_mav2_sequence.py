"""CPU process tests for sequential execution and surviving a closed terminal."""
import contextlib
import io
import json
import os
from pathlib import Path
import pty
import select
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import detach_mav2_then_ma14_ma18 as sequence
from common import read, write


WORKER = r'''
import json, os, sys, time
from pathlib import Path
role, args = sys.argv[1], sys.argv[2:]
root = Path(os.environ['SEQUENCE_TEST_ROOT'])
assert '--execute' in args and '--foreground' in args
def event(phase):
    with (root/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(role=role, phase=phase, args=args))+'\n')
event('start')
if role == 'mav2':
    assert '--allow-quality-failure' in args
    if os.environ.get('SEQUENCE_TEST_HOLD') == '1':
        (root/'waiting').touch()
        deadline = time.monotonic()+15
        while not (root/'release').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('CPU test release timed out')
            time.sleep(.02)
if os.environ.get('SEQUENCE_TEST_FAIL') == role:
    sys.exit(7)
if role == 'mav2':
    out = Path(args[args.index('--out')+1])
    out.mkdir()
    quality = os.environ.get('SEQUENCE_TEST_QUALITY') != 'failed'
    status = 'completed' if quality else 'completed_with_quality_failure'
    if os.environ.get('SEQUENCE_TEST_INCOMPLETE') == '1':
        status = 'running'
    (out/'batch-state.json').write_text(json.dumps(dict(status=status, stage='done',
        allow_quality_failure=True, trials=int(args[args.index('--trials')+1]),
        quality_within_pairs=quality)))
event('end')
'''


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mav2-sequence-CPU-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        phase = self.root/'tools/h100/phase6'
        phase.mkdir(parents=True)
        driver = self.root/'fixture_worker.py'
        driver.write_text(WORKER)
        for role in sequence.ORDER:
            (phase/f'run_{role}.sh').write_text(
                '#!/usr/bin/env bash\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(driver))+
                ' '+role+' "$@"\n')
        shutil.copyfile(sequence.MA_SCRIPT, phase/sequence.MA_SCRIPT.name)
        self.mav2 = phase/'run_mav2.sh'
        self.ma = phase/sequence.MA_SCRIPT.name
        self.flux = self.root/'flux build'
        self.flux.mkdir()
        write(self.flux/'manifest.json', {})
        self.out = self.root/'batch with spaces'
        self.env = dict(SEQUENCE_TEST_ROOT=str(self.root), SEQUENCE_TEST_FAIL='',
                        SEQUENCE_TEST_HOLD='', SEQUENCE_TEST_QUALITY='', SEQUENCE_TEST_INCOMPLETE='')

    def argv(self, *extra):
        return ['--out', str(self.out), '--flux-build', str(self.flux), *extra]

    def events(self):
        return [json.loads(line) for line in (self.root/'events.jsonl').read_text().splitlines()]

    def run_foreground(self):
        with mock.patch.object(sequence, 'MAV2_SCRIPT', self.mav2), \
                mock.patch.object(sequence, 'MA_SCRIPT', self.ma), \
                mock.patch.dict(os.environ, self.env), contextlib.redirect_stdout(io.StringIO()):
            return sequence.main(self.argv('--foreground', '--execute'))

    def test_preview_has_no_side_effects(self):
        with mock.patch.object(sequence.launcher, 'invoke', side_effect=AssertionError('subprocess')), \
                mock.patch.object(sequence.launcher, 'detach', side_effect=AssertionError('detach')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            sequence.main(self.argv())
        self.assertFalse(self.out.exists())
        self.assertIn('--allow-quality-failure --foreground', output.getvalue())
        self.assertIn('mav2 -> ma14 -> ma15 -> ma16 -> ma17 -> ma18 -> ma19 -> ma20', output.getvalue())

    def test_defaults_match_requested_commands(self):
        args = sequence.arguments([])
        self.assertEqual(args.flux_build, sequence.DEFAULT_FLUX_BUILD)
        self.assertEqual(args.trials, 1)
        steps = sequence.commands(args)
        self.assertEqual([name for name, _ in steps], ['mav2', 'ma14-ma20'])
        self.assertIn('--foreground', steps[0][1])
        self.assertNotIn('--foreground', steps[1][1])  # The suite applies it to all seven children.

    def test_all_eight_finish_in_order(self):
        result = self.run_foreground()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([(e['role'], e['phase']) for e in self.events()],
                         [(role, phase) for role in sequence.ORDER for phase in ('start', 'end')])
        self.assertTrue((self.out/'mav2.log').is_file())
        self.assertTrue((self.out/'ma14-ma20.log').is_file())
        for event in self.events():
            if event['role'].startswith('ma') and event['role'] != 'mav2':
                number = int(event['role'][2:])
                self.assertEqual('--candidate-only' in event['args'], number >= 16)

    def test_ma19_failure_stops_before_ma20(self):
        self.env['SEQUENCE_TEST_FAIL'] = 'ma19'
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_foreground()
        self.assertEqual(self.events()[-1]['role'], 'ma19')
        self.assertFalse(any(event['role'] == 'ma20' for event in self.events()))

    def test_old_ma_sequence_name_forwards_to_the_new_seven_experiment_entry(self):
        old = self.ma.with_name('run_ma14_ma15_ma16_ma17_ma18.sh')
        shutil.copyfile(sequence.HERE / old.name, old)
        result = subprocess.run(['bash', str(old), '--execute'],
                                env=dict(os.environ, **self.env), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([event['role'] for event in self.events() if event['phase'] == 'start'],
                         sequence.ORDER[1:])

    def test_quality_failure_still_runs_all_ma_experiments(self):
        self.env['SEQUENCE_TEST_QUALITY'] = 'failed'
        result = self.run_foreground()
        self.assertEqual(result['status'], 'completed_with_quality_failure')
        self.assertEqual(result['stages']['ma14-ma20']['status'], 'completed')
        self.assertEqual(self.events()[-1]['role'], 'ma20')

    def test_mav2_process_failure_stops_before_ma14(self):
        self.env['SEQUENCE_TEST_FAIL'] = 'mav2'
        with self.assertRaises(subprocess.CalledProcessError): self.run_foreground()
        self.assertEqual([e['role'] for e in self.events()], ['mav2'])
        state = read(self.out/'batch-state.json')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['stages']['mav2']['exit_code'], 7)
        self.assertEqual(state['stages']['ma14-ma20']['status'], 'pending')

    def test_incomplete_mav2_is_not_mistaken_for_completion(self):
        self.env['SEQUENCE_TEST_INCOMPLETE'] = '1'
        with self.assertRaisesRegex(ValueError, 'did not finish'): self.run_foreground()
        self.assertTrue(all(e['role'] == 'mav2' for e in self.events()))

    def test_ma16_failure_stops_remaining_experiments(self):
        self.env['SEQUENCE_TEST_FAIL'] = 'ma16'
        with self.assertRaises(subprocess.CalledProcessError): self.run_foreground()
        self.assertEqual(self.events()[-1]['role'], 'ma16')
        state = read(self.out/'batch-state.json')
        self.assertEqual(state['stage'], 'ma14-ma20')
        self.assertEqual(state['stages']['ma14-ma20']['exit_code'], 7)

    def test_detach_failure_is_recorded(self):
        with mock.patch.object(sequence.launcher, 'detach', side_effect=RuntimeError('readiness failed')), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'readiness failed'):
            sequence.main(self.argv('--execute'))
        self.assertEqual(read(self.out/'batch-state.json')['status'], 'failed')

    def test_background_sequence_survives_terminal_closure(self):
        # Real double-fork and controlling PTY, with bounded CPU fixture children.
        entry = self.root/'launch_fixture.py'
        entry.write_text(f'import sys\nsys.path.insert(0, {str(sequence.HERE)!r})\n'
            'import detach_mav2_then_ma14_ma18 as sequence\nfrom pathlib import Path\n'
            f'sequence.MAV2_SCRIPT = Path({str(self.mav2)!r})\n'
            f'sequence.MA_SCRIPT = Path({str(self.ma)!r})\n'
            f'sequence.main({self.argv("--execute")!r})\n')
        env = dict(os.environ, **dict(self.env, SEQUENCE_TEST_HOLD='1'))
        pid, terminal = pty.fork()
        if pid == 0:
            os.execve(sys.executable, [sys.executable, str(entry)], env)
        output = bytearray()
        reaped = False
        try:
            deadline = time.monotonic()+12
            while time.monotonic() < deadline:
                ready, _, _ = select.select([terminal], [], [], .05)
                if ready:
                    try: output.extend(os.read(terminal, 8192))
                    except OSError: pass  # PTY EOF after the original launcher exits.
                child, status = os.waitpid(pid, os.WNOHANG)
                if child:
                    reaped = True
                    self.assertEqual(os.waitstatus_to_exitcode(status), 0, output.decode())
                    break
            self.assertTrue(reaped, output.decode())
            os.close(terminal)
            terminal = None
            identity = read(self.out/'handoff/supervisor.json')
            self.assertTrue(identity['ready'])
            self.assertEqual(identity['ppid'], 1)
            self.assertNotEqual(identity['sid'], pid)
            deadline = time.monotonic()+5
            while not (self.root/'waiting').exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue((self.root/'waiting').exists())
            self.assertEqual([(e['role'], e['phase']) for e in self.events()], [('mav2', 'start')])
            (self.root/'release').touch()
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                try: state = read(self.out/'batch-state.json')
                except json.JSONDecodeError: continue  # Concurrent state write.
                if state['status'] in ('completed', 'failed'):
                    break
                time.sleep(.02)
            self.assertEqual(state['status'], 'completed', state)
            self.assertEqual([(e['role'], e['phase']) for e in self.events()],
                             [(role, phase) for role in sequence.ORDER for phase in ('start', 'end')])
        finally:
            (self.root/'release').touch()
            if terminal is not None: os.close(terminal)
            if not reaped: os.waitpid(pid, 0)


if __name__ == '__main__':
    unittest.main()

"""CPU-only checks for the launch boundary and requested experiment workload."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import detach_ma5 as launch


class LauncherTests(unittest.TestCase):
    def test_preview_is_read_only_and_one_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'must-not-exist'
            with patch.object(launch.launcher, 'invoke', side_effect=AssertionError('spawned')):
                with patch.object(launch.launcher, 'detach', side_effect=AssertionError('forked')):
                    with contextlib.redirect_stdout(io.StringIO()) as stream:
                        launch.main(['--out', str(target), '--trials', '1'])
            self.assertFalse(target.exists())
            output = stream.getvalue()
            self.assertIn('16 training windows / 800 formal steps', output)
            self.assertIn('--trials 1', output)
            self.assertNotIn('--trials 2', output)
            self.assertNotIn('compare2', output)

    def test_sweep_pairs_are_sequential_and_independent(self):
        args = launch.arguments(['--sweep', '--trials', '2', '--out', '/tmp/ma5-preview'])
        work = launch.commands(args)
        configs = launch.candidates(args)
        self.assertEqual(len(configs), 4)
        for i, (name, tuning) in enumerate(configs):
            group = work[i * 8:(i + 1) * 8]
            self.assertEqual({entry[0] for entry in group}, {name})
            self.assertEqual(group[-1][1], 'fullstep-profile')
            self.assertIn(str(args.out / name / 'build'), group[0][2])
            self.assertEqual(tuning['gemm_sms'], -1)
            self.assertEqual(tuning['gemm_tile_k'], 32)

    def test_custom_gemm_knobs_reach_build(self):
        args = launch.arguments(['--gemm-tile-k', '64', '--gemm-sms', '128', '--decoder-threads', '128'])
        prepare = launch.commands(args)[0][2]
        self.assertEqual(prepare[prepare.index('--gemm-tile-k') + 1], '64')
        self.assertEqual(prepare[prepare.index('--gemm-sms') + 1], '128')
        self.assertEqual(prepare[prepare.index('--decoder-threads') + 1], '128')

    def test_reused_build_skips_compilation_only(self):
        args = launch.arguments(['--ma5-build', '/tmp/existing-ma5'])
        stages = [label for _, label, _ in launch.commands(args)]
        self.assertNotIn('compile', stages)
        self.assertIn('operator-timing', stages)
        self.assertIn('fullstep-run', stages)
        self.assertIn('fullstep-profile', stages)

    def test_ambiguous_sweep_is_rejected(self):
        for argv in (['--sweep', '--ma5-build', '/tmp/reuse'],
                     ['--sweep', '--decoder-threads', '128']):
            with self.assertRaises(ValueError):
                launch.arguments(argv)

    def test_full_grid_values_are_validated_by_configuration(self):
        for threads, blocks in ((64,4096),(512,512)):
            args=launch.arguments(['--decoder-threads',str(threads),'--decoder-blocks',str(blocks)])
            self.assertEqual(launch.candidates(args)[0][1]['decoder_blocks'],blocks)
        for threads,blocks in ((512,513),(256,0)):
            with self.assertRaises(ValueError):
                launch.arguments(['--decoder-threads',str(threads),'--decoder-blocks',str(blocks)])

    def test_controller_snapshot_covers_harness_build_and_worker_templates(self):
        hashes=launch.controller_hashes()
        for relative in ('detach_ma5.py','ma5_bench.py','ma5_fullstep.py','ma5_build.py',
                         'ma5_sources.py','ma5_gemm.py','common.py','ablation4_sources.py',
                         'run_ma5_tile_block.sh'):
            self.assertIn(str(launch.HERE/relative),hashes)
        self.assertIn(str(launch.REPO/'tools/a800/phase5/e2e/megatron_taco.py'),hashes)

    def supervise_fixture(self,failure=None):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'handoff').mkdir()
            args=launch.arguments(['--out',str(root)])
            configs=launch.candidates(args)
            source=root/'source.py';source.write_text('# frozen CPU fixture\n')
            launch.write(root/'handoff/controllers.json',{str(source):launch.sha(source)})
            launch.write(root/'batch-state.json',dict(status='launching',expected_formal_optimizer_steps=800,
                candidates={name:dict(status='pending',tuning=tuning) for name,tuning in configs}))
            seen=[]
            def invoke(command,log,env):
                seen.append(log.stem)
                self.assertEqual(env['CUDA_DEVICE_MAX_CONNECTIONS'],'1')
                if log.stem=='fullstep-run':
                    folder=log.parent/'fullstep';folder.mkdir()
                    launch.write(folder/'report.json',{'synthetic':'CPU-only negative outcome'})
                if failure=='drift' and log.stem=='compile':source.write_text('# changed\n')
            result=dict(comparisons={'compare1':{}},increment_over_CR2_validated=False,
                        absolute_500ms_observed=False,target_validated=False)
            with patch.object(launch.launcher,'invoke',side_effect=invoke), \
                 patch.object(launch,'check_operator',side_effect=ValueError('incorrect') if failure=='correctness' else None,
                              return_value=dict(operator_correctness_passed=True,operator_execution_path_validated=True)), \
                 patch.object(launch,'check_fullstep',return_value=result),contextlib.redirect_stdout(io.StringIO()):
                if failure:
                    with self.assertRaises(ValueError):launch.supervise(args)
                    self.assertEqual(launch.read(root/'batch-state.json')['status'],'failed')
                    self.assertNotIn('fullstep-run',seen)
                else:
                    state=launch.supervise(args)
                    self.assertEqual(state['status'],'completed')
                    self.assertEqual(state['formal_optimizer_steps'],800)
                    self.assertFalse(state['candidates'][configs[0][0]]['target_validated'])
                    self.assertEqual(seen[-1],'fullstep-profile')

    def test_slower_outcomes_complete(self):
        self.supervise_fixture()

    def test_wrong_correctness_stops_before_training(self):
        self.supervise_fixture('correctness')

    def test_controller_drift_stops_after_build(self):
        self.supervise_fixture('drift')

    def test_shell_recovers_environment_and_forwards_quoted_arguments(self):
        with tempfile.TemporaryDirectory(prefix='ma5 shell ') as directory:
            root=Path(directory);phase=root/'tools/h100/phase6';phase.mkdir(parents=True)
            script=phase/'run_ma5_tile_block.sh'
            script.write_text((launch.HERE/'run_ma5_tile_block.sh').read_text())
            recovery=root/'outputs/h100/container-recovery';recovery.mkdir(parents=True)
            (recovery/'env.sh').write_text('export H100_RECOVERY_PYTHON='+shlex.quote(sys.executable)+'\n'
                'export H100_RECOVERY_BUILD_CUDA=/recovered/compiler\nexport CUDA_HOME=/recovered/runtime\n')
            (phase/'detach_ma5.py').write_text('import sys,os,json\nprint(json.dumps(dict(args=sys.argv[1:],cwd=os.getcwd())))\n')
            env=dict(os.environ);env.pop('MA5_PYTHON',None)
            result=subprocess.run(['bash',str(script),'--trials','1','--execute','--out','path with spaces'],
                                  env=env,check=True,text=True,capture_output=True)
            record=json.loads(result.stdout)
            self.assertEqual(record['cwd'],str(root))
            self.assertEqual(record['args'],['--cuda-home','/recovered/runtime','--build-cuda-home',
                '/recovered/compiler','--trials','1','--execute','--out','path with spaces'])


if __name__ == '__main__':
    unittest.main()

"""Regression for unrelated OP5 edits aborting OP6 after dependency builds."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import op3
import op6
import op6_child
import op6_controllers as controllers


class ControllerScope(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.repo = self.root/'repo'
        self.retained = (
            'tools/5090/phase6/op6.py', 'tools/5090/phase6/op6_controllers.py',
            'tools/5090/phase6/op3.py', 'tools/5090/phase6/op3_codec.cu',
            'tools/5090/phase6/op4_evidence.py',
            'tools/h100/phase6/ma15_profile.py', 'tools/h100/phase6/ma11_adapter.py',
            'tools/a800/phase5/e2e/run.py', 'tools/a800/phase4/reference.py',
        )
        for name in (*self.retained, *controllers.EXCLUDED):
            path = self.repo/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# original '+name+'\n')
        for key, value in dict(REPO=self.repo, HERE=self.repo/'tools/5090/phase6',
                               H100=self.repo/'tools/h100/phase6').items():
            self.stack.enter_context(patch.object(op3.base, key, value))
        self.stack.enter_context(patch.object(op6, 'REPO', self.repo))

    def freeze(self, directory, files):
        for name in files:
            target = directory/'handoff/source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.repo/name, target)
        op3.write(directory/'handoff/controllers.json', files)

    def test_only_exact_op5_paths_are_removed(self):
        all_files = op3.base.controller_files()
        selected = controllers.collect(op3.base.controller_files)
        self.assertEqual(set(all_files)-set(selected), controllers.EXCLUDED)
        self.assertEqual(set(selected), set(self.retained))
        self.assertEqual(selected, {name: all_files[name] for name in selected})

    def test_parent_snapshot_survives_op5_edits_after_build(self):
        out = self.root/'parent'
        with patch.object(op6, 'check_overrides'), patch.object(op6, 'supervise'), redirect_stdout(io.StringIO()):
            op6.main(['--execute', '--foreground', '--out', str(out),
                      '--op3-build', str(self.root/'existing-build')])
        manifest = op3.read(out/'handoff/controllers.json')
        self.assertEqual(set(manifest), set(self.retained))
        for name in controllers.EXCLUDED:
            (self.repo/name).write_text('# OP5 edited during OP6 compile\n')
        op3.base.verify_controllers(out)
        self.assertEqual(op3.read(out/'handoff/controllers.json'), manifest)

    def test_real_dependency_and_saved_snapshot_changes_still_fail(self):
        out = self.root/'batch'
        self.freeze(out, controllers.collect(op3.base.controller_files))
        for directory in (self.repo, out/'handoff/source'):
            for name in self.retained:
                path = directory/name
                original = path.read_bytes()
                path.write_text('# changed dependency\n')
                with self.subTest(directory=directory, name=name), self.assertRaisesRegex(ValueError, 'Missing/changed evidence'):
                    op3.base.verify_controllers(out)
                path.write_bytes(original)

    def test_existing_manifests_are_not_filtered_or_rewritten(self):
        out = self.root/'old-batch'
        self.freeze(out, op3.base.controller_files())
        before = (out/'handoff/controllers.json').read_bytes()
        (self.repo/'tools/5090/phase6/op5.py').write_text('# edited\n')
        restore = controllers.install(op3.base)
        try:
            with self.assertRaisesRegex(ValueError, 'op5.py'):
                op3.base.verify_controllers(out)
        finally:
            restore()
        self.assertEqual((out/'handoff/controllers.json').read_bytes(), before)

    def test_both_child_variants_scope_creation_and_restore_original_verifier(self):
        original_collect, original_verify = op3.base.controller_files, op3.base.verify_controllers
        for variant in ('op4', 'op6'):
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                with op6_child.configured(variant):
                    self.assertEqual(set(op3.base.controller_files()), set(self.retained))
                    self.assertIs(op3.base.verify_controllers, original_verify)
                    raise RuntimeError('worker failed')
            self.assertIs(op3.base.controller_files, original_collect)
            self.assertIs(op3.base.verify_controllers, original_verify)

    def test_initial_preflight_uses_scoped_child_entrypoint(self):
        import op1_abc_build
        baseline = op6.load('build')
        args = op6.arguments(['--out', str(self.root/'first-run')])
        commands = []

        def run(command, environment, log):
            commands.append(command)
            dest = Path(command[command.index('--out')+1])
            op3.write(dest/'batch-state.json', {'status': 'preflight_completed'})

        with patch.object(op6, 'run_child', side_effect=run), patch.object(op6, 'child_environment', return_value={}), \
                patch.object(baseline, 'prepare'), patch.object(baseline, 'compile_build'), \
                patch.object(op1_abc_build, 'prepare'), patch.object(op1_abc_build, 'compile_build'), \
                redirect_stdout(io.StringIO()):
            op6.prepare_initial_build(args, args.out, {})
        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(Path(command[1]).name, 'op6_child.py')
        self.assertEqual(command[command.index('--variant')+1], 'op4')
        self.assertIn('--preflight-only', command)
        self.assertIn('--foreground', command)
        self.assertEqual(args.abc_build, args.out/'dependencies/abc')


if __name__ == '__main__':
    unittest.main()

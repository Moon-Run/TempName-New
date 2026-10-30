"""Historical controller identity survives live edits without weakening raw checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ablation_plan as plan
import scope_plan
from common import REPO, read, sha


class InputResolutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='frozen-input-test-')
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)/'repo'
        self.freeze = self.repo/'freeze'
        self.live = self.repo/'tools/h100/phase6/scope_calibrate.py'
        self.archive = self.repo/'logs/h100/phase6/calibration/controllers/scope_calibrate.py'
        for path, text in ((self.live, 'current implementation\n'), (self.archive, 'historical implementation\n')):
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
        self.original = '/workspace/TempName/tools/h100/phase6/scope_calibrate.py'
        self.archived = '/workspace/TempName/logs/h100/phase6/calibration/controllers/scope_calibrate.py'
        self.digest = sha(self.archive)
        self.inputs = {self.original: self.digest, self.archived: self.digest}
        patcher = mock.patch.multiple(plan, REPO=self.repo, FREEZE=self.freeze)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_registered_exact_snapshot_resolves_after_live_edit_without_mutation(self):
        before = {p: p.read_bytes() for p in (self.live, self.archive)}
        self.assertEqual(plan.resolve_input(self.original, self.digest, self.inputs), self.archive)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        with self.assertRaisesRegex(ValueError, 'Missing/changed'):
            plan.resolve_input(self.original, self.digest)

    def test_live_match_has_priority_and_local_absolute_paths_work(self):
        self.assertEqual(plan.resolve_input(self.original, sha(self.live), self.inputs), self.live)
        inputs = {str(self.live): self.digest, str(self.archive): self.digest}
        self.assertEqual(plan.resolve_input(str(self.live), self.digest, inputs), self.archive)

    def test_legacy_source_freeze_remains_supported(self):
        frozen = self.freeze/'source/tools/h100/phase6/scope_calibrate.py'
        frozen.parent.mkdir(parents=True); frozen.write_bytes(self.archive.read_bytes())
        self.assertEqual(plan.resolve_input(self.original, self.digest), frozen)

    def test_unregistered_wrong_hash_wrong_name_and_missing_snapshots_rejected(self):
        other = self.archive.with_name('plan.py'); other.write_bytes(self.archive.read_bytes())
        for inputs in ({self.original: self.digest},
                       {self.original: self.digest, self.archived: '0'*64},
                       {self.original: self.digest, str(other): self.digest},
                       {self.archived: self.digest}):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                plan.resolve_input(self.original, self.digest, inputs)
        self.archive.unlink()
        with self.assertRaises(ValueError):
            plan.resolve_input(self.original, self.digest, self.inputs)

    def test_corrupted_snapshot_is_rehashed_not_accepted_from_metadata(self):
        self.archive.write_text('tampered\n')
        with self.assertRaises(ValueError):
            plan.resolve_input(self.original, self.digest, self.inputs)

    def test_raw_measurements_and_other_tools_cannot_use_controller_aliases(self):
        for name in ('/workspace/TempName/logs/h100/phase6/calibration/scope_calibrate.py',
                     '/workspace/TempName/tools/a800/scope_calibrate.py'):
            inputs = {name: self.digest, self.archived: self.digest}
            with self.subTest(name=name), self.assertRaises(ValueError):
                plan.resolve_input(name, self.digest, inputs)


class ReturnedCalibrationTests(unittest.TestCase):
    def test_real_two_drifted_controllers_use_their_registered_historical_bytes(self):
        root = REPO/'logs/h100/phase6/ablation5-then4-60steps-20261018-230611-029519/ablation4/compare1'
        parent = read(root/'scripts/selection-plan.json')['parent']
        expected = {'scope_calibrate.py': 'c215e1926c928b2fc3189ee19b8adc10edd75ac6be07fb8fe15c437d4c894c00',
                    'plan.py': 'e110229caedff76e28110271c7a468d4dd3d18f000b1c7a59b73d57c293715f1'}
        for role, original in parent['roles'].items():
            # This fixture isolates controller evidence. It is not a complete
            # executable-build audit (returned archives omit .pyc and binaries).
            record = copy.deepcopy(original); record['inputs'] = {}
            for basename, digest in expected.items():
                name = '/workspace/TempName/tools/h100/phase6/'+basename
                archived = str(Path(original['origin']['source'])/'controllers'/basename)
                self.assertEqual(original['inputs'][name], digest)
                self.assertEqual(original['inputs'][archived], digest)
                self.assertNotEqual(sha(REPO/'tools/h100/phase6'/basename), digest)
                record['inputs'].update({name: digest, archived: digest})
                resolved = plan.resolve_input(name, digest, original['inputs'])
                self.assertEqual(resolved.parent.name, 'controllers')
                self.assertEqual(sha(resolved), digest)
            self.assertEqual(scope_plan.validate(record, role), record)
            missing = copy.deepcopy(record)
            del missing['inputs'][str(Path(original['origin']['source'])/'controllers/scope_calibrate.py')]
            with self.assertRaisesRegex(ValueError, 'Missing/changed frozen input'):
                scope_plan.validate(missing, role)


if __name__ == '__main__':
    unittest.main()

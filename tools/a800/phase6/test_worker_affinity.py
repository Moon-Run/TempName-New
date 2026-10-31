"""CPU-only regression for preparing new runs from immutable legacy workers."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest

from worker_affinity import inherited_worker_source, prepare_worker_affinity


LEGACY = '''import os
local_rank = int(os.environ['LOCAL_RANK'])
WORLD = 8
allowed_cpus = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {allowed_cpus[local_rank * len(allowed_cpus) // WORLD]})
import torch
report = dict(rank=local_rank)
def step():
    return model.forward()
'''


class WorkerAffinityTest(unittest.TestCase):
    def test_pre_torch_startup_preserves_large_and_restricted_masks_for_every_rank(self):
        source = inherited_worker_source(LEGACY)
        tree = ast.parse(source)
        prefix = tree.body[1:next(i for i, node in enumerate(tree.body)
                                 if isinstance(node, ast.Import) and node.names[0].name == 'torch')]
        for mask in (set(range(192)), {2, 7, 19}, {31}):
            for rank in range(8):
                def forbidden(*args):
                    self.fail('A prepared worker attempted to override its inherited CPU set')
                fake_os = types.SimpleNamespace(environ={'LOCAL_RANK': str(rank)},
                                               sched_getaffinity=lambda pid: set(mask),
                                               sched_setaffinity=forbidden)
                context = dict(os=fake_os)
                exec(compile(ast.Module(body=prefix, type_ignores=[]), '<startup>', 'exec'), context)
                self.assertEqual(fake_os.sched_getaffinity(0), mask)
        before = next(node for node in ast.parse(LEGACY).body if isinstance(node, ast.FunctionDef))
        after = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        self.assertEqual(ast.dump(before), ast.dump(after))
        self.assertIn("cpu_affinity_policy='inherit'", source)
        self.assertEqual(inherited_worker_source(source), source)

    def test_compact_calibration_and_separate_cpu_variable_are_supported(self):
        compact = LEGACY.replace(
            "allowed_cpus = sorted(os.sched_getaffinity(0))\nos.sched_setaffinity(0, {allowed_cpus[local_rank * len(allowed_cpus) // WORLD]})",
            "cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[local_rank*len(cpus)//WORLD]})")
        separate = LEGACY.replace(
            'os.sched_setaffinity(0, {allowed_cpus[local_rank * len(allowed_cpus) // WORLD]})',
            '_cpu=allowed_cpus[local_rank * len(allowed_cpus) // WORLD]\nos.sched_setaffinity(0,{_cpu})')
        for source in (compact, separate):
            self.assertNotIn('sched_setaffinity', inherited_worker_source(source))

    def test_unknown_runtime_binding_and_shared_code_fail_closed(self):
        sources = [
            LEGACY.replace('import torch\nreport', 'import torch\nos.sched_setaffinity(0, {3})\nreport'),
            LEGACY.replace('os.sched_setaffinity(0, {allowed_cpus[local_rank * len(allowed_cpus) // WORLD]})',
                           'os.sched_setaffinity(0, {3})'),
            LEGACY.replace('allowed_cpus = sorted', 'unrelated = 1; allowed_cpus = sorted'),
            LEGACY.replace('local_rank * len(allowed_cpus) // WORLD', 'initialize_seed()'),
            LEGACY.replace('local_rank * len(allowed_cpus) // WORLD', '3'),
            LEGACY.replace('local_rank * len(allowed_cpus) // WORLD', 'local_rank * len(allowed_cpus) // choose_world()'),
            LEGACY.replace('sched_getaffinity(0)', 'sched_getaffinity()'),
        ]
        for source in sources:
            with self.assertRaises(ValueError):
                inherited_worker_source(source)

    def test_new_snapshot_identity_matches_actual_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp)/'original.py'
            snapshot = Path(tmp)/'new.py'
            original.write_text(LEGACY)
            snapshot.write_bytes(original.read_bytes())
            record = prepare_worker_affinity(snapshot)
            self.assertTrue(record['changed'])
            self.assertEqual(record['source_sha256'], hashlib.sha256(original.read_bytes()).hexdigest())
            self.assertEqual(record['worker_sha256'], hashlib.sha256(snapshot.read_bytes()).hexdigest())
            self.assertEqual(original.read_text(), LEGACY)

    def test_tp4_new_run_transforms_copy_without_modifying_archive(self):
        import tp4_archive
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'archive'
            source = root/'template/scripts/worker.py'
            source.parent.mkdir(parents=True)
            source.write_text(LEGACY)
            build = root/'artifacts/build/manifest.json'
            build.parent.mkdir(parents=True)
            build.write_text('{}\n')
            submission = root/'source-submission.json'
            submission.write_text(json.dumps(dict(original_campaign='historical')))
            files = {str(path.relative_to(root)): tp4_archive.sha(path)
                     for path in (source, build, submission)}
            (root/'archive-manifest.json').write_text(json.dumps(dict(repo=tmp, files=files)))
            out = Path(tmp)/'new-run'
            tp4_archive.prepare(root, out, 1)
            record = json.loads((out/'submission.json').read_text())
            self.assertEqual(source.read_text(), LEGACY)
            self.assertNotIn('sched_setaffinity', (out/'scripts/worker.py').read_text())
            self.assertEqual(record['cpu_affinity']['worker_sha256'], tp4_archive.sha(out/'scripts/worker.py'))
            for path, digest in files.items():
                self.assertEqual(tp4_archive.sha(root/path), digest)


if __name__ == '__main__':
    unittest.main()

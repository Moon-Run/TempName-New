"""Generated worker and evidence admission for graph and eager OP6 variants."""
import ast
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import op3_protocol as p
import op3_sources as original_sources
import op3_validation as original_validation
import op6_sources as sources
import op6_validation as validation
from test_op3_sources import library_plan
from test_op3_validation import fixture as op3_fixture
from test_op6_adapter import completed_state


def fixture(variant, mode='smoke'):
    rows, job, plan, worker = op3_fixture(p.COMBINED, mode)
    plan.update(op6_protocol=sources.PROTOCOL, op6_variant=variant)
    worker['config'].update(op6_protocol=sources.PROTOCOL, op6_variant=variant,
                            op6_implementation=sources.implementation(variant))
    worker['files']['op6_adapter.py'] = p.sha(sources.ADAPTER)
    worker['files']['op6_profile_io.py'] = p.sha(sources.PROFILE_IO)
    for row in rows:
        row.update(op6_variant=variant, op6_implementation=sources.implementation(variant),
                   op6_attention_graph_state=(completed_state(row['warmup_steps']+len(row['losses']), mode)
                                               if variant == 'op6' else {'installed': False}))
    return rows, job, plan, worker


class SourceGeneration(unittest.TestCase):
    def test_only_setup_reporting_and_export_change_training_and_capture_stay_original(self):
        before = ast.parse(sources._worker_source(p.COMBINED))
        generated = sources.worker_source(p.COMBINED)
        after = ast.parse(generated.replace(sources.PROFILE_EXPORT, sources.LEGACY_PROFILE_EXPORT))
        self.assertNotIn("dir='/tmp'", generated)
        self.assertEqual(generated.count('export_profile(profiler, OUT, rank)'), 1)
        functions = lambda tree: {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(functions(before), functions(after))
        timing = lambda tree: [ast.dump(n) for n in tree.body if isinstance(n, (ast.For, ast.With))
                              or isinstance(n, ast.If) and ast.unparse(n.test) == "MODE == 'profile'"]
        self.assertEqual(timing(before), timing(after))
        graph = [n for n in after.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "PLAN['op6_variant'] == 'op6'"]
        self.assertEqual(len(graph), 1)
        softmax = next(n for n in after.body if isinstance(n, ast.Assign)
                       and ast.unparse(n).startswith('ma11_softmax_tracker ='))
        warmup = next(n for n in after.body if isinstance(n, ast.For)
                      and ast.unparse(n.iter) == 'range(warmup)')
        self.assertLess(softmax.lineno, graph[0].lineno)
        self.assertLess(graph[0].lineno, warmup.lineno)

    def test_both_worker_packages_bind_variant_and_adapter_and_restore_originals(self):
        original = original_sources.worker_source, original_sources.prepare_workers
        for variant in sources.VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                native = root/'parent'
                native.mkdir()
                plan = dict(environment={'torch': original_sources.optimizers.compat.TORCH},
                            communication={}, op6_protocol=sources.PROTOCOL, op6_variant=variant,
                            **library_plan(root))
                stack.enter_context(patch.object(original_sources.builder, 'codec_library',
                                                return_value={'path': '/CPU/libop3_codec.so', 'sha256': 'c'*64}))
                stack.enter_context(patch.object(original_sources.builder, 'native_root', return_value=native))
                stack.enter_context(patch.object(original_sources.builder, 'libraries', return_value={}))
                restore = sources.install(variant)
                try:
                    workers = original_sources.prepare_workers(root, root, plan)
                finally:
                    restore()
                for worker in workers.values():
                    scripts = Path(worker['root'])
                    self.assertEqual(worker['config']['op6_implementation'], sources.implementation(variant))
                    self.assertEqual((scripts/'op6_adapter.py').read_bytes(), sources.ADAPTER.read_bytes())
                    self.assertEqual((scripts/'op6_profile_io.py').read_bytes(), sources.PROFILE_IO.read_bytes())
                    self.assertEqual((scripts/'worker.py').read_text(), sources.worker_source(p.COMBINED))
                    self.assertEqual(worker['files'], {path.name: p.sha(path) for path in scripts.iterdir() if path.is_file()})
            self.assertEqual((original_sources.worker_source, original_sources.prepare_workers), original)

    def test_export_exception_does_not_admit_training_or_capture_changes(self):
        source = sources.worker_source(p.COMBINED)
        for old, new in (('losses,success,grad_norm=step()', 'losses,success,grad_norm=other_step()'),
                         ('range(count)', 'range(count+1)'),
                         ('export_profile(profiler, OUT, rank)', 'export_profile(profiler, OUT, 0)')):
            with self.subTest(change=old), self.assertRaises(ValueError):
                sources.validate_worker(source.replace(old, new))


class EvidenceAdmission(unittest.TestCase):
    def test_all_modes_original_op3_and_graph_checks_pass_together(self):
        before = original_validation.validate_rows
        for variant in sources.VARIANTS:
            restore = validation.install(variant)
            try:
                for mode in ('smoke', 'profile', 'timing'):
                    with self.subTest(variant=variant, mode=mode):
                        self.assertEqual(len(original_validation.validate_rows(*fixture(variant, mode))), 8)
            finally:
                restore()
            self.assertIs(original_validation.validate_rows, before)

    def test_missing_graph_stale_replay_or_suppressed_original_audit_fails(self):
        mutations = (
            lambda row: row.pop('op6_attention_graph_state'),
            lambda row: row.update(op6_variant='op4'),
            lambda row: row['op6_attention_graph_state']['modules'].pop(next(iter(row['op6_attention_graph_state']['modules']))),
            lambda row: next(iter(row['op6_attention_graph_state']['modules'].values())).update(backward_replays=0),
            lambda row: row['ma11_softmax_state'].update(forward_audits=0),
            lambda row: row['ma11_zero_count_state'].update(calls=0),
            lambda row: row.update(smoke_output_sha256={}),
        )
        restore = validation.install('op6')
        try:
            for mutate in mutations:
                rows, job, plan, worker = fixture('op6')
                mutate(rows[0])
                with self.subTest(mutation=mutate), self.assertRaises((ValueError, RuntimeError)):
                    original_validation.validate_rows(rows, job, plan, worker)
        finally:
            restore()

    def test_control_rejects_graph_and_candidate_rejects_foreign_adapter(self):
        for variant in sources.VARIANTS:
            rows, job, plan, worker = fixture(variant)
            if variant == 'op4':
                rows[0]['op6_attention_graph_state'] = completed_state(5, 'smoke')
            else:
                worker['files']['op6_adapter.py'] = 'f'*64
            restore = validation.install(variant)
            try:
                with self.assertRaises(ValueError):
                    original_validation.validate_rows(rows, job, plan, worker)
            finally:
                restore()

    def test_both_variants_require_the_frozen_profile_exporter(self):
        for variant in sources.VARIANTS:
            rows, job, plan, worker = fixture(variant)
            worker['files'].pop('op6_profile_io.py')
            restore = validation.install(variant)
            try:
                with self.subTest(variant=variant), self.assertRaises(ValueError):
                    original_validation.validate_rows(rows, job, plan, worker)
            finally:
                restore()


if __name__ == '__main__':
    unittest.main()

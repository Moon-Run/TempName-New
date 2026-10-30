"""CPU-only C/A/B budget, order separation, source identity and admission checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ablation4_build
import ablation4_protocol
import ablation7_build as builder
import ablation7_plan as planner
import ablation7_protocol as routing
import scope_build
import scope_plan
from ablation_plan import mask_sha
from common import read, sha, write


def parent_fixture():
    return dict(protocol=scope_plan.PROTOCOL, roles={r: scope_plan.cpu_fixture(r) for r in routing.SHAPES},
                source_only=True, gpu_calibrated=False)


def dense_plan():
    parent = parent_fixture(); scores = {}
    for role, record in parent['roles'].items():
        values = [[float(tile//64 != source) for tile in range(512)] for source in range(8)]
        entry = record['entry']; entry['mask'] = planner.fitter.selection.select(values, 8, 1/64)
        entry['selected_per_source'] = [7]*8
        record.update(entry_sha256=routing.digest(entry), mask_sha256=mask_sha(entry['mask']))
        scores[role] = dict(kind='cpu_fixture', scores=values, scores_sha256=routing.digest(values),
                            inputs={}, training_passes=[0, 1], heldout_passes=[2])
    plans = planner.bundle(parent, source_only=True); plans['scores'] = scores
    plans['variants'] = {p: planner.derive(parent, scores, plans['orders'], p) for p in routing.POLICIES}
    return planner.validate(plans)


class PlanTests(unittest.TestCase):
    def test_three_groups_balanced_blocks_reversed_rounds_and_independent_timing(self):
        import ablation_timing
        self.assertEqual(routing.execution_timing(), dict(warmup_steps=10, timed_steps=50))
        self.assertEqual(ablation_timing.execution_timing()['timed_steps'], 60)
        self.assertEqual([routing.definition(p)['family'] for p in routing.POLICIES], ['C', 'A', 'B'])
        self.assertEqual(routing.PRIMARY, [(routing.B, routing.A)])
        jobs = routing.schedule(); self.assertEqual(len(jobs), 36)
        rows = [[j['policy'] for j in jobs if j['round'] == r and j['block'] == b]
                for r in (1, 2) for b in range(6)]
        self.assertEqual(rows[6:], [row[::-1] for row in rows[:6][::-1]])
        for position in range(3):
            self.assertEqual(sorted(row[position] for row in rows[:6]), sorted(list(routing.POLICIES)*2))
        for blocks in (0, 1, 5, True):
            with self.assertRaises(ValueError): routing.schedule(blocks)

    def test_budget_28_and_legacy_c_mapping_are_independent_of_b(self):
        plans = dense_plan()
        for role in routing.SHAPES:
            c, a, b = [routing.role_plans(plans, p)[role] for p in routing.POLICIES]
            original = plans['parent']['roles'][role]['entry']
            self.assertEqual(c['entry']['selected_per_source'], [28]*8)
            self.assertEqual(c['entry']['maps'], original['maps'])
            self.assertEqual(c['entry']['coords'], original['coords'])
            self.assertEqual(c['wire_bytes_per_source'], [14680064-28*128*120]*8)
            self.assertEqual(a['entry'], b['entry'])
            self.assertNotEqual(c['entry']['maps'], b['entry']['maps'])
            self.assertEqual(a['entry']['maps'], [list(range(512)) for _ in range(8)])
            for entry in (a['entry'], b['entry']):
                self.assertEqual(entry['selected_per_source'], [448]*8)
            for source, row in enumerate(c['entry']['mask']):
                self.assertFalse(any(row[source*64:(source+1)*64]))
                self.assertTrue(all(not old or new for old, new in zip(original['mask'][source], row)))

    def test_positive_scores_need_not_fill_budget_and_synthetic_cannot_execute(self):
        plans = planner.bundle(parent_fixture(), source_only=True)
        self.assertEqual(routing.role_plans(plans, routing.C)['attn']['entry']['selected_per_source'], [1]*8)
        changed = copy.deepcopy(plans)
        changed.update(source_only=False, gpu_calibrated=True)
        changed['parent'].update(source_only=False, gpu_calibrated=True)
        with self.assertRaises(ValueError): planner.validate(changed)
        with self.assertRaises(ValueError): planner.bundle(parent_fixture(), source_only=False)

    def test_rehashed_mask_or_order_cannot_change_group_semantics(self):
        plans = dense_plan()
        for policy, field in ((routing.C, 'maps'), (routing.B, 'mask')):
            changed = copy.deepcopy(plans); record = changed['variants'][policy]['roles']['attn']
            if field == 'maps':
                record['entry']['maps'] = [list(range(512)) for _ in range(8)]
                record['entry']['coords'] = copy.deepcopy(record['entry']['base_coords'])
            else:
                record['entry']['mask'][0][64] = 0
                record['entry']['selected_per_source'][0] -= 1
            record['entry_sha256'] = routing.digest(record['entry'])
            record['mask_sha256'] = mask_sha(record['entry']['mask'])
            with self.assertRaisesRegex(ValueError, 'Derived C/A/B'): planner.validate(changed)
        changed = copy.deepcopy(plans); changed['orders']['attn']['heldout_passes'] = [1]
        with self.assertRaisesRegex(ValueError, 'split'): planner.validate(changed)
        changed = copy.deepcopy(plans); changed['execution_timing']['timed_steps'] = 60
        with self.assertRaisesRegex(ValueError, 'timing'): planner.validate(changed)


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='ablation7-PLAN-BUILD-CPU-')
        cls.root = Path(cls.tmp.name)
        cls.parent = ablation4_build.prepare(cls.root/'parent', source_only=True)
        cls.parent_hash = sha(cls.parent/'ablation4-manifest.json')
        cls.build = builder.prepare(cls.root/'build', cls.parent, source_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_reuses_c_a_builds_b_only_and_identity_produces_identical_sources(self):
        record = builder.verify(self.build)
        self.assertEqual(sha(self.parent/'ablation4-manifest.json'), self.parent_hash)
        for policy in (routing.C, routing.A):
            self.assertEqual(builder.runtime_root(self.build, policy),
                ablation4_build.runtime_root(self.parent, builder.PARENT_POLICIES[policy]))
        b_root = builder.runtime_root(self.build, routing.B)
        self.assertNotEqual(b_root, builder.runtime_root(self.build, routing.A))
        self.assertFalse((b_root/'libflux_cuda.so').exists())
        fused = read(self.parent/'fused/manifest.json')
        a = fused['variants'][ablation4_build.FUSED_NAMES[ablation4_protocol.POLICIES[2]]]
        improved = read(self.build/'improved/manifest.json')['variants'][builder.IMPROVED_NAME]
        self.assertEqual(improved['source_sha256'], a['source_sha256'])
        with self.assertRaisesRegex(ValueError, 'Unbuilt|diagnostic'): builder.verify(self.build, built=True)
        with self.assertRaises(ValueError): builder.prepare(self.root/'illegal', self.parent)
        # Returned reports need only recorded paths, never the original host directory.
        for policy in routing.POLICIES:
            builder.runtime_root('/nonexistent-original-host-build', policy, record=record)

    def test_b_codec_and_c_source_tampering_rejected_even_with_updated_hashes(self):
        record = read(self.build/'ablation7-manifest.json'); plans = read(self.build/'plans.json')
        candidate = read(self.build/'parent-candidate-manifest.json')
        fused = read(self.build/'parent-fused-manifest.json'); improved = read(self.build/'improved/manifest.json')
        bad = copy.deepcopy(improved); manifest = copy.deepcopy(record)
        path = 'python/flux/gemm_rs_taco.py'
        bad['variants'][builder.IMPROVED_NAME]['source_sha256'][path] = 'f'*64
        manifest['improved_source_sha256'][path] = 'f'*64
        manifest['files']['improved/manifest.json'] = builder.json_sha(bad)
        with self.assertRaisesRegex(ValueError, 'A/B compute/codec'):
            builder.validate_snapshots(manifest, plans, candidate, fused, bad)
        bad = copy.deepcopy(candidate)
        bad['variants']['scope-candidate']['source_sha256'][scope_build.SWIZZLE] = 'f'*64
        with self.assertRaises(ValueError): builder.validate_snapshots(record, plans, bad, fused, improved)

    def test_compile_delegates_only_b_without_touching_legacy_build(self):
        with mock.patch.object(builder.build, 'compile_build') as compile_b:
            builder.compile_build(self.build, '/absent-cuda', jobs=2)
            compile_b.assert_called_once_with(self.build/'improved', Path('/absent-cuda'), 2, None)
        self.assertEqual(sha(self.parent/'ablation4-manifest.json'), self.parent_hash)


if __name__ == '__main__':
    unittest.main()

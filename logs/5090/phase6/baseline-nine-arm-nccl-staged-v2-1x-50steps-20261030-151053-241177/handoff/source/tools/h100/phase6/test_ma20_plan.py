"""CPU-only MA20 plan properties; these tests make no GPU performance claim."""
import copy
import unittest

import ablation4_protocol as routing
import ma2_ablation_build as parent_builder
import ma20_plan as planner
from ablation_plan import mask_sha
from common import validate_entry


def synthetic_entry(window=64, rotation=0):
    """A mathematical permutation fixture, never an executable measured plan."""
    bases, mappings = [], []
    for source in range(8):
        base = [((source + i % 8) % 8) * 64 + i // 8 for i in range(512)]
        mapping = list(range(512))
        for destination in range(8):
            slots = [i for i, tile in enumerate(base) if tile // 64 == destination]
            for start in range(0, 64, window):
                group = slots[start:start + window]
                offset = rotation % len(group)
                for slot, value in zip(group, group[offset:] + group[:offset]):
                    mapping[slot] = value
        bases.append(base)
        mappings.append(mapping)
    return dict(shape=[2048, 4096, 4096], M=2048, N=4096, K_global=4096,
                K_local=512, tiles=512, base_coords=bases, maps=mappings,
                coords=[[base[i] for i in mapping] for base, mapping in zip(bases, mappings)],
                mask=[[0] * 512 for _ in range(8)], selected_per_source=[0] * 8)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read the hash-verified historical selection already used by MA11.
        # Its measured-parent status is inherited, never fabricated by a fixture.
        cls.selection, _, _ = parent_builder.canonical()
        cls.plans = planner.bundle(cls.selection)

    def test_every_remote_contribution_is_eligible_without_legacy_scores(self):
        reached = [set() for _ in range(8)]
        # Rotate each destination through the tail: Q7 must be able to select
        # any of its 64 tiles even though every old mask entry is zero.
        for rotation in range(64):
            original = synthetic_entry(rotation=rotation)
            chosen = planner.derive(original, 7)
            for source, row in enumerate(chosen['mask']):
                reached[source].update(t for t, bit in enumerate(row) if bit)
        for source, tiles in enumerate(reached):
            self.assertEqual(tiles, {t for t in range(512) if t // 64 != source})
        original = synthetic_entry(rotation=3)
        unrelated = copy.deepcopy(original)
        unrelated['scores'] = [[-1000.0] * 512 for _ in range(8)]
        self.assertEqual(planner.derive(original, 112), planner.derive(unrelated, 112))

    def test_counts_increase_masks_are_nested_and_zero_is_valid(self):
        for role in planner.SHAPES:
            previous = None
            for count in planner.DEFAULT_COUNTS:
                item = self.plans['configurations'][f'Q{count}']['roles'][role]
                entry = item['entry']
                self.assertEqual(item['eligible_per_source'], [448] * 8)
                self.assertEqual(item['selected_per_source'], [count] * 8)
                self.assertEqual(item['selected_total'], count * 8)
                self.assertEqual(item['wire_bytes_per_source'], [14680064 - 15360 * count] * 8)
                self.assertEqual(item['changed_vs_ma11_per_rank'], [2 * count] * 8)
                for source, row in enumerate(entry['mask']):
                    self.assertEqual(sum(row), count)
                    self.assertFalse(any(row[source * 64:(source + 1) * 64]))
                    if previous is not None:
                        self.assertTrue(all(not old or new for old, new in zip(previous[source], row)))
                        self.assertNotEqual(row, previous[source])
                previous = entry['mask']
            original = self.selection['variants'][planner.C_POLICY]['roles'][role]['entry']
            zero = self.plans['configurations']['Q0']['roles'][role]['entry']
            self.assertEqual(zero['maps'], original['maps'])
            self.assertEqual(zero['coords'], original['coords'])
            self.assertEqual(sum(map(sum, zero['mask'])), 0)

    def test_balances_each_source_across_all_remote_destinations(self):
        for count in planner.DEFAULT_COUNTS:
            for role in planner.SHAPES:
                mask = self.plans['configurations'][f'Q{count}']['roles'][role]['entry']['mask']
                destination_totals = [0] * 8
                for source, row in enumerate(mask):
                    counts = [sum(row[d * 64:(d + 1) * 64]) for d in range(8)]
                    remote = [q for destination, q in enumerate(counts) if destination != source]
                    self.assertEqual(counts[source], 0)
                    self.assertEqual(remote, [count // 7] * 7)
                    destination_totals = [a + b for a, b in zip(destination_totals, counts)]
                self.assertEqual(destination_totals, [count] * 8)

    def test_codec_first_uses_minimum_swaps_and_stays_in_each_window(self):
        for window in (1, 8, 16, 64):
            original = synthetic_entry(window=window, rotation=3)
            for count in (0, 7, 28, 112):
                derived = planner.derive(original, count, window)
                validate_entry(derived, {'tp': 8, 'window': window, 'fraction': count / 448},
                               original['shape'])
                for source in range(8):
                    base = original['base_coords'][source]
                    mask = derived['mask'][source]
                    self.assertEqual(derived['coords'][source],
                                     [base[i] for i in derived['maps'][source]])
                    self.assertEqual(sorted(derived['maps'][source]), list(range(512)))
                    for destination in range(8):
                        slots = [i for i, tile in enumerate(base) if tile // 64 == destination]
                        for start in range(0, 64, window):
                            group = slots[start:start + window]
                            before = [original['coords'][source][i] for i in group]
                            after = [derived['coords'][source][i] for i in group]
                            selected = sum(mask[t] for t in before)
                            self.assertTrue(all(mask[t] for t in after[:selected]))
                            self.assertTrue(all(not mask[t] for t in after[selected:]))
                            self.assertEqual(set(after), set(before))
                            misplaced = {i for i, tile in enumerate(before)
                                         if bool(mask[tile]) != (i < selected)}
                            changed = {i for i, (a, b) in enumerate(zip(before, after)) if a != b}
                            # These and only these positions must change: a
                            # selected tail swaps with a BF16 prefix vacancy.
                            self.assertEqual(changed, misplaced)
                            self.assertEqual(len(changed),
                                             2 * sum(not mask[t] for t in before[:selected]))

    def test_change_diagnostics_compare_to_ma11_and_base_separately(self):
        original = synthetic_entry(rotation=3)
        for source in range(8):
            original['mask'][source][((source + 1) % 8) * 64] = 1
        original['selected_per_source'] = [1] * 8
        for count in (0, 7, 112):
            derived = planner.derive(original, count)
            item = planner.record(derived, original)
            for source in range(8):
                self.assertEqual(item['changed_vs_ma11_per_rank'][source],
                                 sum(a != b for a, b in zip(derived['coords'][source], original['coords'][source])))
                self.assertEqual(item['changed_vs_base_per_rank'][source],
                                 sum(i != j for i, j in enumerate(derived['maps'][source])))
                self.assertEqual(item['mask_changed_vs_ma11_per_source'][source],
                                 sum(a != b for a, b in zip(derived['mask'][source], original['mask'][source])))
            if count == 0:
                self.assertEqual(item['changed_vs_ma11_per_rank'], [0] * 8)
                self.assertTrue(all(n > 0 for n in item['changed_vs_base_per_rank']))
                self.assertEqual(item['mask_changed_vs_ma11_per_source'], [1] * 8)

    def test_frozen_plan_and_rehashed_mask_tampering_are_rejected(self):
        def changed_mask(plans):
            item = plans['configurations']['Q7']['roles']['attn']
            row = item['entry']['mask'][0]
            selected = next(t for t in range(64, 512) if row[t])
            replacement = next(t for t in range(64, 512) if not row[t])
            row[selected], row[replacement] = 0, 1
            item['entry_sha256'] = planner.digest(item['entry'])
            item['mask_sha256'] = mask_sha(item['entry']['mask'])

        def changed_permutation(plans):
            entry = plans['configurations']['Q7']['roles']['attn']['entry']
            group = [i for i, tile in enumerate(entry['base_coords'][0]) if tile // 64 == 1]
            left, right = group[:2]
            entry['maps'][0][left], entry['maps'][0][right] = entry['maps'][0][right], entry['maps'][0][left]
            entry['coords'][0] = [entry['base_coords'][0][i] for i in entry['maps'][0]]
            original = plans['baseline_selection']['variants'][planner.C_POLICY]['roles']['attn']['entry']
            plans['configurations']['Q7']['roles']['attn'] = planner.record(entry, original)

        mutations = {
            'rehashed_mask': changed_mask,
            'rehashed_legal_permutation': changed_permutation,
            'count': lambda p: p['configurations']['Q7'].update(count=14),
            'eligible': lambda p: p['configurations']['Q7']['roles']['attn']['eligible_per_source'].__setitem__(0, 7),
            'diagnostic': lambda p: p['configurations']['Q7']['roles']['attn']['changed_vs_ma11_per_rank'].__setitem__(0, -1),
            'wire_bytes': lambda p: p['configurations']['Q7']['roles']['attn']['wire_bytes_per_source'].__setitem__(0, 0),
            'provenance': lambda p: p.update(baseline_selection_sha256='0' * 64),
            'gpu_claim': lambda p: p.update(gpu_validated=True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                altered = copy.deepcopy(self.plans)
                mutate(altered)
                with self.assertRaises(ValueError):
                    planner.validate(altered)

    def test_selection_for_preserves_worker_schema_and_routes_real_arm_masks(self):
        before = copy.deepcopy(self.plans)
        for arm in self.plans['configurations']:
            selected = planner.selection_for(self.plans, arm)
            self.assertEqual(selected['parent'], self.selection['parent'])
            if arm == 'MA11':
                self.assertEqual(selected, self.selection)
            for policy in self.selection['variants']:
                if policy != planner.C_POLICY:
                    self.assertEqual(selected['variants'][policy], self.selection['variants'][policy])
            for role, shape in planner.SHAPES.items():
                record = routing.selection_entry(selected, planner.C_POLICY, role, *shape)
                expected = self.plans['configurations'][arm]['roles'][role]
                self.assertEqual(record['entry'], expected['entry'])
                self.assertEqual(record['entry_sha256'], expected['entry_sha256'])
                self.assertEqual(record['mask_sha256'], mask_sha(record['entry']['mask']))
                if arm == 'MA11':
                    self.assertEqual(record['entry'], self.selection['variants'][planner.C_POLICY]['roles'][role]['entry'])
            selected['variants'][planner.C_POLICY]['roles']['attn']['entry']['mask'][0][64] ^= 1
            self.assertEqual(self.plans, before)
        with self.assertRaises(ValueError):
            planner.selection_for(self.plans, 'not-an-arm')

    def test_invalid_parent_or_unregistered_quota_cannot_enter_bank(self):
        for count in (-1, 1, 448, 7.0, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                planner.derive(synthetic_entry(), count)
        for window in (0, -1, True):
            with self.subTest(window=window), self.assertRaises(ValueError):
                planner.derive(synthetic_entry(), 7, window)
        for change in ({'source_only': True}, {'gpu_calibrated': False}):
            altered = copy.deepcopy(self.selection)
            altered.update(change)
            with self.assertRaises(ValueError):
                planner.bundle(altered)
        with self.assertRaises(ValueError):
            planner.bundle(self.selection, (0, 7, 112))
        self.assertFalse(self.plans['gpu_validated'])
        self.assertTrue(self.plans['runtime_calibration_required'])


if __name__ == '__main__':
    unittest.main()

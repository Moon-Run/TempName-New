import unittest

import arrival_op4_v2_plan as plan


class ClosurePlanTests(unittest.TestCase):
    def test_latin_is_idempotent_and_bijective(self):
        for source in range(8):
            self.assertEqual(plan.latin(source, source), source)
            self.assertEqual(sorted(plan.latin(source, d) for d in range(8)), list(range(8)))
        for destination in range(8):
            self.assertEqual(sorted(plan.latin(s, destination) for s in range(8)), list(range(8)))

    def test_all_phases_have_identical_link_and_source_budgets(self):
        for phase in range(8):
            for arm in plan.ARMS:
                record = plan.statistics(arm, phase)
                self.assertEqual(record["selected_per_source"], [56] * 8)
                self.assertEqual(record["wire_bytes_per_rank"], [13819904] * 8)
                self.assertEqual(record["inverse_groups_per_receiver"],
                                 [1024 if arm == "CM" else 7168] * 8)
                for s in range(8):
                    self.assertEqual(record["selected_per_link"][s], [0 if d == s else 8 for d in range(8)])

    def test_source_production_wave_counts_are_matched(self):
        for phase in range(8):
            for source in range(8):
                for within in range(64):
                    values = [sum(plan.selected(arm, phase, source, destination, within)
                                  for destination in range(8)) for arm in plan.ARMS]
                    self.assertEqual(values, [int((within + phase) % 8 != source)] * 4)

    def test_closed_support_attains_lower_bound_scatter_has_no_reuse(self):
        for phase in range(8):
            self.assertEqual(plan.statistics("CM", phase)["fanin_histogram_per_receiver"],
                             [{"0": 56, "7": 8}] * 8)
            self.assertEqual(plan.statistics("SM", phase)["fanin_histogram_per_receiver"],
                             [{"0": 8, "1": 56}] * 8)

    def test_rotation_covers_each_remote_contribution_once(self):
        for arm in plan.ARMS:
            masks = [plan.mask(arm, phase) for phase in range(8)]
            for source in range(8):
                for tile in range(512):
                    self.assertEqual(sum(mask[source][tile] for mask in masks), int(tile // 64 != source))

    def test_same_mask_pairs_share_exact_identity(self):
        for phase in range(8):
            self.assertEqual(plan.mask_sha256("SP", phase), plan.mask_sha256("SM", phase))
            self.assertEqual(plan.mask_sha256("CP", phase), plan.mask_sha256("CM", phase))
            self.assertNotEqual(plan.mask_sha256("SP", phase), plan.mask_sha256("CP", phase))

    def test_reject_invalid_phase_and_arm(self):
        for arm, phase in (("A", 0), ("CM", True), ("SP", 8), ("SM", -1)):
            with self.assertRaises(ValueError):
                plan.mask(arm, phase)


if __name__ == "__main__":
    unittest.main()

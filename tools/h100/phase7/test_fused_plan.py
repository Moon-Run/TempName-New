"""CPU-only fused-table tests; artificial arrivals are not GPU observations."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("phase7_fused_plan_test_subject", Path(__file__).with_name("fused_plan.py"))
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def fixture(role="attn"):
    record = planner.legacy_scope.cpu_fixture(role)
    entry = record["entry"]
    entry["maps"] = [list(range(512)) for _ in range(8)]
    entry["coords"] = copy.deepcopy(entry["base_coords"])
    record["entry_sha256"] = planner.digest(entry)
    return record


def observations(entry, passes=(0, 1, 2), signed_shift=0):
    """Stable synthetic late residuals on four physical tiles per destination."""
    positions = planner._inverse(entry["base_coords"])
    return {destination: {(passed, tile): (
                [signed_shift + 20 + 0.1 * positions[source][tile]
                 + (100 if tile % 64 >= 60 else 0) for source in range(8)], 0.05)
            for passed in passes for tile in range(destination*64, (destination+1)*64)}
            for destination in range(8)}


class FittingTests(unittest.TestCase):
    def test_training_residuals_change_legal_maps(self):
        record = fixture()
        entry = record["entry"]
        before = copy.deepcopy(entry)
        maps, diag = planner.fit_maps(entry, record["config"], observations(entry))
        self.assertEqual(entry, before)
        self.assertEqual(diag["accepted_destinations"], 8)
        self.assertTrue(any(diag["changed_vs_legacy_per_source"]))
        result = planner._entry(entry, maps)
        planner.validate_entry(result, record["config"], entry["shape"])
        self.assertEqual(result["mask"], entry["mask"])
        for source in range(8):
            self.assertEqual([tile // 64 for tile in result["coords"][source]],
                             [tile // 64 for tile in entry["base_coords"][source]])
        self.assertFalse(diag["measured_performance"])
        self.assertTrue(diag["not_recalibrated"])

    def test_missing_training_tile_falls_back_entire_destination(self):
        record = fixture()
        entry = record["entry"]
        rows = observations(entry)
        del rows[2][(1, 128)]
        maps, diag = planner.fit_maps(entry, record["config"], rows)
        self.assertEqual(diag["destinations"][2]["selected"], "legacy")
        self.assertEqual(diag["destinations"][2]["missing_training"], 1)
        self.assertGreater(diag["accepted_destinations"], 0)
        for source in range(8):
            slots = [slot for slot, tile in enumerate(entry["base_coords"][source]) if tile // 64 == 2]
            self.assertEqual([maps[source][slot] for slot in slots], [entry["maps"][source][slot] for slot in slots])

    def test_heldout_values_cannot_change_mapping(self):
        record = fixture()
        rows = observations(record["entry"])
        expected, before = planner.fit_maps(record["entry"], record["config"], rows)
        for destination, values in rows.items():
            for (passed, tile), (arrival, error) in list(values.items()):
                if passed == 2:
                    values[(passed, tile)] = ([1e6 * ((source + tile) % 9) for source in range(8)], error)
        actual, after = planner.fit_maps(record["entry"], record["config"], rows)
        self.assertEqual(actual, expected)
        self.assertEqual([row["selected"] for row in before["destinations"]], [row["selected"] for row in after["destinations"]])
        self.assertNotEqual(before["destinations"][0]["heldout_metrics"], after["destinations"][0]["heldout_metrics"])
        self.assertFalse(after["destinations"][0]["heldout_metrics"]["used_for_selection"])

    def test_legacy_not_identity_is_acceptance_baseline(self):
        record = fixture()
        rows = observations(record["entry"])
        first, _ = planner.fit_maps(record["entry"], record["config"], rows)
        already_improved = planner._entry(record["entry"], first)
        second, diag = planner.fit_maps(already_improved, record["config"], rows)
        self.assertEqual(first, second)
        self.assertEqual(diag["accepted_destinations"], 0)
        self.assertEqual(diag["changed_vs_legacy_per_source"], [0] * 8)

    def test_signed_receiver_relative_offsets_are_kept(self):
        record = fixture()
        positive, _ = planner.fit_maps(record["entry"], record["config"], observations(record["entry"]))
        signed, diag = planner.fit_maps(record["entry"], record["config"], observations(record["entry"], signed_shift=-500))
        self.assertEqual(positive, signed)
        self.assertLess(diag["destinations"][0]["train_metrics"]["legacy"][0]["surrogate_max_join_us"], 0)

    def test_flat_slot_slope_has_no_claimed_gain(self):
        record = fixture()
        rows = observations(record["entry"])
        for values in rows.values():
            for key in values:
                values[key] = ([30.] * 8, 0.1)
        maps, diag = planner.fit_maps(record["entry"], record["config"], rows)
        self.assertEqual(maps, record["entry"]["maps"])
        self.assertEqual(diag["accepted_destinations"], 0)

    def test_one_pass_or_max_regression_rejects_candidate(self):
        legacy = [dict(surrogate_p95_join_us=100., surrogate_max_join_us=120.) for _ in range(2)]
        candidate = [dict(surrogate_p95_join_us=90., surrogate_max_join_us=110.),
                     dict(surrogate_p95_join_us=101., surrogate_max_join_us=110.)]
        self.assertFalse(planner._admissible(candidate, legacy, 0.1))
        candidate[1] = dict(surrogate_p95_join_us=90., surrogate_max_join_us=121.)
        self.assertFalse(planner._admissible(candidate, legacy, 0.1))
        candidate[1] = dict(surrogate_p95_join_us=90., surrogate_max_join_us=119.)
        self.assertTrue(planner._admissible(candidate, legacy, 0.1))

    def test_gain_below_polling_uncertainty_is_rejected(self):
        legacy = [dict(surrogate_p95_join_us=100., surrogate_max_join_us=120.) for _ in range(2)]
        candidate = [dict(surrogate_p95_join_us=99.99, surrogate_max_join_us=119.99) for _ in range(2)]
        self.assertFalse(planner._admissible(candidate, legacy, 0.1))

    def test_invalid_or_cross_destination_observation_is_rejected(self):
        record = fixture()
        rows = observations(record["entry"])
        rows[0][(0, 64)] = rows[0][(0, 0)]
        with self.assertRaisesRegex(ValueError, "outside destination"):
            planner.fit_maps(record["entry"], record["config"], rows)
        rows = observations(record["entry"])
        rows[0][(0, 0)][0][1] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            planner.fit_maps(record["entry"], record["config"], rows)

    def test_observation_free_fit_preserves_legacy_exactly(self):
        record = planner.legacy_scope.cpu_fixture("attn")
        maps, diag = planner.fit_maps(record["entry"], record["config"], {})
        self.assertEqual(maps, record["entry"]["maps"])
        self.assertEqual(diag["accepted_destinations"], 0)
        self.assertTrue(all(row["missing_training"] == 128 for row in diag["destinations"]))


class RawInputTests(unittest.TestCase):
    def test_hash_resolution_accepts_exact_file_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "evidence.json"
            path.write_text('{"source":"cpu_fixture"}')
            expected = planner.sha(path)
            self.assertEqual(planner.resolve_input(str(path), expected), path.resolve())
            path.write_text('{"source":"changed"}')
            with self.assertRaisesRegex(ValueError, "Missing/changed"):
                planner.resolve_input(str(path), expected)

    def test_test_only_records_cannot_masquerade_as_calibration(self):
        with self.assertRaisesRegex(ValueError, "Real frozen"):
            planner.load_observations(fixture())

    def test_missing_rank_reports_fail_instead_of_faking_arrivals(self):
        record = fixture()
        record.update(test_only=False, gpu_calibrated=True, inputs={})
        with self.assertRaisesRegex(ValueError, "all eight ranks"):
            planner.load_observations(record)

    def test_original_fragment_parser_uses_receiver_clock_and_complete_fragments(self):
        sample = dict(mode="receiver_sparse", offset=0, repeat=0, max_poll_cycle_ns=2.,
                      local_markers=[10000, 20000], receiver=[dict(tile=0, source=source, fragment=fragment,
                      observed_ns=9000 + source*10 + fragment) for source in range(8) for fragment in range(8)])
        result = planner._ARRIVAL.observations(dict(partition_tiles=64, samples=[sample]), 0, 8, relative=True)
        self.assertEqual(result[(0, 0)][0][0], -0.993)
        self.assertEqual(result[(0, 0)][1], 0.004)
        sample["receiver"].pop()
        self.assertEqual(planner._ARRIVAL.observations(dict(partition_tiles=64, samples=[sample]), 0, 8, relative=True), {})


class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Source-only mode must not read/fabricate arrival labels. Real parent
        # fixture metadata is validated through the existing scope loader.
        with tempfile.TemporaryDirectory() as root:
            attn, mlp = Path(root) / "attn.json", Path(root) / "mlp.json"
            attn.write_text(json.dumps(planner.legacy_scope.cpu_fixture("attn")))
            mlp.write_text(json.dumps(planner.legacy_scope.cpu_fixture("mlp")))
            with mock.patch.object(planner, "load_observations", side_effect=AssertionError("GPU provenance read in source-only")):
                cls.plans = planner.bundle(attn, mlp, source_only=True)

    def test_nine_variants_and_fixed_physical_masks(self):
        plans = self.plans
        self.assertEqual(len(plans["variants"]), 9)
        self.assertFalse(plans["gpu_calibrated"])
        self.assertTrue(plans["source_only"])
        planner.validate(plans, verify_evidence=False)
        for order in planner.ORDERS:
            for role, entry in planner.order_entries(plans, order).items():
                self.assertEqual(entry["mask"], plans["parent"]["roles"][role]["entry"]["mask"])
        self.assertEqual(planner.order_entries(plans, "improved"), planner.order_entries(plans, "legacy"))

    def test_hash_or_mask_tampering_fails(self):
        plans = copy.deepcopy(self.plans)
        policy = planner.policy_id("improved", "attn")
        record = plans["variants"][policy]["roles"]["attn"]
        record["entry"]["mask"][0][65] ^= 1
        record["entry_sha256"] = planner.digest(record["entry"])
        record["mask_sha256"] = planner.mask_sha(record["entry"]["mask"])
        with self.assertRaisesRegex(ValueError, "physical mask"):
            planner.validate(plans, verify_evidence=False)

    def test_fabricated_recalibration_flag_fails(self):
        plans = copy.deepcopy(self.plans)
        plans["not_recalibrated"] = False
        with self.assertRaisesRegex(ValueError, "cannot claim"):
            planner.validate(plans, verify_evidence=False)

    def test_verify_fit_reconstructs_diagnostics_from_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            attn, mlp = Path(root) / "attn.json", Path(root) / "mlp.json"
            attn.write_text(json.dumps(planner.legacy_scope.cpu_fixture("attn")))
            mlp.write_text(json.dumps(planner.legacy_scope.cpu_fixture("mlp")))
            plans = planner.bundle(attn, mlp, source_only=True)
            self.assertTrue(planner.verify_fit(plans)["reconstructed"])
            plans["diagnostics"]["attn"]["accepted_destinations"] = 1
            # Hash/structure validation is deliberately separate from refitting.
            planner.validate(plans)
            with self.assertRaisesRegex(ValueError, "fitting diagnostic differs"):
                planner.verify_fit(plans)


if __name__ == "__main__":
    unittest.main()

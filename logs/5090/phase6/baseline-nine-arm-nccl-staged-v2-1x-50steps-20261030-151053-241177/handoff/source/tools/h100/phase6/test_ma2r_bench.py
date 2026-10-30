"""CPU-only protocol tests. Synthetic measurements are never GPU evidence."""
import argparse
from collections import Counter
import contextlib
import copy
import io
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ma2r_bench as bench


def fixture_plan():
    return dict(protocol=bench.PROTOCOL, run_id="CPU-fixture-only", arms=list(bench.ARMS), arm_labels=bench.LABELS,
                roles=bench.ROLES, cases=list(bench.CASES), schedule=bench.schedule(6),
                reuse=2, seeds=[817, 919], retained_output_calls=6, instance_isolation_calls=6, warmup=1, iterations=2,
                slow_rank_stress=dict(ranks=[0, 7], cycles=2000000),
                communication={"NCCL_NVLS_ENABLE": "0"},
                shared_environment={"CUDA_DEVICE_MAX_CONNECTIONS": "1", "FLUX_GEMM_RS_GEMM_ONLY": "1"},
                source_files={}, libraries={arm: dict(root=f"/synthetic/{arm}",
                    files={f"/synthetic/{arm}/libflux_cuda.so": arm}) for arm in bench.ARMS})


def fixture_correctness():
    return dict(passed=True, max_abs=.01, relative_l2=.02, protocol_relative_l2=.001,
                budget_vs_bf16=.05, budget_vs_codec=.01)


def fixture_row(plan, job, rank, stage="preflight"):
    roles = {}
    for role in bench.ROLES:
        checks = []
        for epoch, seed, case in bench.check_indices(plan):
            delayed_rank, delay_cycles = bench.correctness_delay(plan, epoch, seed, case)
            checks.append(dict(epoch=epoch, seed=seed, case=case,
                delayed_rank=delayed_rank, delay_cycles=delay_cycles,
                inputs_sha256="a" * 64, weights_sha256="b" * 64,
                output_sha256="c" * 64, correctness=fixture_correctness()))
        burst = [dict(generation=g, inputs_sha256="d" * 64, weights_sha256="e" * 64,
                      output_sha256="f" * 64, correctness=fixture_correctness()) for g in range(6)]
        state = bench.expected_runtime_state(job["arm"])
        timing = dict(wall_us={"A": 100., "V": 95., "R": 102.}[job["arm"]],
                      iterations=2, warmup=1, includes="owned-output forward and final synchronization") if stage == "timing" else None
        isolated = [dict(generation=i, role=role if i % 2 == 0 else next(r for r in bench.ROLES if r != role),
                         correctness=fixture_correctness(), inputs_sha256="1" * 64,
                         weights_sha256="2" * 64, output_sha256="3" * 64) for i in range(6)]
        roles[role] = dict(shape=bench.ROLES[role], checks=checks, retained_output_checks=burst,
                           runtime_state=state, timing=timing,
                           retained_storage_independent=True,
                           instance_workspaces_distinct=stage == "preflight",
                           instance_isolation_checks=isolated if stage == "preflight" else [])
    return dict(protocol=bench.PROTOCOL, run_id=plan["run_id"], plan_sha256="1" * 64,
                completed=True, rank=rank, job=job, sampling=False, profiling=stage == "profile",
                preflight=stage == "preflight", execution_environment=bench.execution_environment(plan, job["arm"]),
                loaded_libraries=plan["libraries"][job["arm"]]["files"], roles=roles,
                hardware={"host": "CPU-only-fixture", "devices": [str(r) for r in range(8)]},
                software={"torch": "synthetic"})


def trace_fixture(arm="R", role="attn", calls=4):
    events = []
    decoder = {"A": "taco_decode_ring_kernel(X)", "V": "void taco_ma2_decode_kernel(X)",
               "R": "void taco_ma2_decode_kernel(X)"}[arm]
    for invocation in range(calls):
        start = invocation * 200
        events.append(dict(cat="user_annotation", name=f"ma2r.{role}.{invocation}",
                           ph="X", pid=10, tid=11, ts=start, dur=80))
        # Real Kineto traces also contain the same record_function name on GPU.
        events.append(dict(cat="gpu_user_annotation", name=f"ma2r.{role}.{invocation}",
                           ph="X", pid=0, tid=7, ts=start + 100, dur=45))
        if arm != "R":
            events.append(dict(cat="cpu_op", name="aten::clone", ph="X", pid=10, tid=11,
                               ts=start + 50, dur=20))
        names = ["flux_bf16_reducescatter(X)", "CudaIpcBarrierAllKernel(X)", decoder,
                 "CudaIpcBarrierAllKernel(X)"]
        if arm != "R":
            names.append("vectorized_elementwise_kernel_copy(X)")
        for index, name in enumerate(names):
            correlation = invocation * 10 + index
            events.append(dict(cat="cuda_runtime", name="cudaLaunchKernel", pid=10, tid=11,
                ts=start + (55 if index == 4 else 5 + 10 * index), dur=1,
                args={"correlation": correlation}))
            events.append(dict(cat="kernel", name=name, pid=0, tid=7, ts=start + 100 + index * 10,
                               dur=5, args={"correlation": correlation, "stream": 7}))
    return events


class SchedulingAndIsolationTests(unittest.TestCase):
    def test_default_run_has_thirty_six_operator_windows_and_distinct_labels(self):
        self.assertEqual(len(bench.schedule()), 36)
        self.assertEqual(bench.COMPARISONS, (("R", "A"), ("R", "V"), ("V", "A")))
        self.assertEqual(set(bench.LABELS), {"A", "V", "R"})
        self.assertIn("Unchanged MA2", bench.LABELS["V"])
        self.assertIn("native", bench.LABELS["R"])

    def test_every_position_and_pairwise_order_is_counterbalanced(self):
        for blocks in (6, 12, 24):
            jobs = bench.schedule(blocks)
            rows = [[j["arm"] for j in jobs if j["block"] == block] for block in range(blocks)]
            self.assertEqual(rows[blocks // 2:], [r[::-1] for r in rows[:blocks // 2][::-1]])
            self.assertTrue(all(set(row) == set(bench.ARMS) for row in rows))
            for position in range(3):
                self.assertEqual(Counter(row[position] for row in rows), {a: blocks // 3 for a in bench.ARMS})
            for a, b in bench.COMPARISONS:
                self.assertEqual(sum(row.index(a) < row.index(b) for row in rows), blocks // 2)
        for blocks in (0, 2, 7, 12.0):
            with self.assertRaises(ValueError): bench.schedule(blocks)

    def test_environment_selects_isolated_libraries_and_strips_old_controls(self):
        plan = fixture_plan()
        inherited = {"NCCL_ALGO": "Tree", "H100_OP3_MODE": "overlap", "H100_MA2R_ARM": "other",
                     "H100_MA2_ARM": "V", "H100_MA2_PROTOCOL": "previous-ma2",
                     "H100_MA1_ARM": "V", "H100_MA1_PROTOCOL": "old-experiment",
                     "FLUX_TUNE_CONFIG_FILE": "/wrong", "CUDA_DEVICE_MAX_CONNECTIONS": "8"}
        with patch.dict(os.environ, inherited):
            for arm in bench.ARMS:
                env = bench.environment(Path("/experiment"), plan, arm, Path("/venv/bin/python"), None)
                self.assertEqual(env["CUDA_DEVICE_MAX_CONNECTIONS"], "1")
                self.assertEqual(env["H100_MA2R_PROTOCOL"], bench.SOURCE_PROTOCOL)
                self.assertEqual(env["H100_MA2R_ARM"], arm)
                self.assertIn(f"/synthetic/{arm}/python", env["PYTHONPATH"])
                self.assertNotIn("NCCL_ALGO", env)
                self.assertNotIn("H100_OP3_MODE", env)
                self.assertNotIn("H100_MA1_ARM", env)
                self.assertNotIn("H100_MA1_PROTOCOL", env)
                self.assertNotIn("H100_MA2_ARM", env)
                self.assertNotIn("H100_MA2_PROTOCOL", env)
                self.assertNotIn("FLUX_TUNE_CONFIG_FILE", env)
                self.assertEqual(env, bench.environment(Path("/experiment"), plan, arm, Path("/venv/bin/python"), None, True))

    def test_prepare_freezes_four_distribution_reference_without_gpu_import(self):
        with tempfile.TemporaryDirectory(prefix="ma2r-prepare-CPU-") as temp:
            root = Path(temp) / "new"
            args = argparse.Namespace(root=root, build=Path(temp) / "build", blocks=6,
                                      warmup=1, iterations=2, reuse=2, seeds=[817, 919])
            with patch.object(bench, "build_identity", return_value={"synthetic": True}), \
                 patch.object(bench, "libraries", return_value=fixture_plan()["libraries"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                bench.prepare(args)
            plan = bench.read(root / "plan.json")
            self.assertEqual(plan["cases"], list(bench.CASES))
            self.assertEqual(plan["arm_labels"], bench.LABELS)
            self.assertEqual(len(plan["schedule"]), 18)
            self.assertFalse(plan["target_validated"])
            self.assertFalse(plan["historical_timings_used"])
            reference = (root / "scripts/codec_reference.py").read_text()
            self.assertIn("def check_taco_output", reference)
            self.assertNotIn("from routing import", reference)
            bench.verify(root, offline=True)

    def test_delays_are_untimed_and_target_two_ranks(self):
        plan = fixture_plan()
        delays = [bench.correctness_delay(plan, *entry) for entry in bench.check_indices(plan)]
        self.assertEqual([d for d in delays if d[0] >= 0], [(0, 2000000), (7, 2000000)])

    def test_timeout_signals_only_owned_process_group(self):
        process = Mock(pid=432123)
        process.wait.side_effect = [subprocess.TimeoutExpired(["torchrun"], 1), 143, 143]
        with patch.object(bench.subprocess, "Popen", return_value=process) as launch, \
             patch.object(bench.os, "killpg") as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                bench.run_worker(["torchrun"], "/tmp", {}, None, 1)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertEqual(kill.call_args_list, [unittest.mock.call(432123, signal.SIGTERM),
                                             unittest.mock.call(432123, signal.SIGKILL)])

    def test_failed_worker_exit_signals_owned_group_and_preserves_exception(self):
        process = Mock(pid=1234)
        process.wait.return_value = 7
        with patch.object(bench.subprocess, "Popen", return_value=process), patch.object(bench.os, "killpg") as kill:
            with self.assertRaises(subprocess.CalledProcessError):
                bench.run_worker(["torchrun"], "/tmp", {}, None, 1)
        self.assertEqual(kill.call_count, 2)


class MappingTests(unittest.TestCase):
    def rows(self):
        return [dict(rank=rank, index=i, m=coordinate // 32, n=coordinate % 32,
                     destination=coordinate // 64, base_m=0, base_n=0, sm_occupancy=1)
                for rank in range(8) for i in range(512)
                for coordinate in [((i % 8 + rank) % 8) * 64 + i // 8]]

    def test_exact_actual_interleaved_mapping_passes(self):
        bench.verify_mapping(self.rows())

    def test_bijection_is_not_enough_to_pass_wrong_order(self):
        rows = self.rows()
        rows[0]["m"], rows[1]["m"] = rows[1]["m"], rows[0]["m"]
        for row in rows[:2]: row["destination"] = row["m"] // 2
        with self.assertRaisesRegex(ValueError, "interleaved"): bench.verify_mapping(rows)

    def test_missing_duplicate_or_zero_occupancy_fails(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, "Incomplete"): bench.verify_mapping(rows[:-1])
        rows[1]["index"] = 0
        with self.assertRaisesRegex(ValueError, "Duplicate"): bench.verify_mapping(rows)
        rows = self.rows()
        rows[0]["sm_occupancy"] = 0
        with self.assertRaisesRegex(ValueError, "occupancy"): bench.verify_mapping(rows)


class RuntimeAbiTests(unittest.TestCase):
    def op(self, owned=False, **functions):
        return SimpleNamespace(placement="fused", wire_bytes_per_rank=7 * 256 * 32 * 136,
                               owns_output=owned, owns_configuration=owned,
                               lib=SimpleNamespace(**{k: Mock(return_value=v) for k, v in functions.items()}))

    def packed(self, owned=False, **override):
        functions = dict(taco_ma2_protocol_version=1, taco_ma2_packed_load_enabled=1)
        if owned:
            functions.update(taco_ma2r_protocol_version=1, taco_ma2r_submission_enabled=1)
        functions.update(override)
        return self.op(owned, **functions)

    def test_a_does_not_require_a_new_protocol_export(self):
        self.assertEqual(bench.runtime_state(self.op(), "A"), bench.expected_runtime_state("A"))

    def test_v_only_requires_original_ma2_abi(self):
        self.assertEqual(bench.runtime_state(self.packed(), "V"), bench.expected_runtime_state("V"))

    def test_r_requires_both_packed_and_native_submission_abis(self):
        self.assertEqual(bench.runtime_state(self.packed(True), "R"), bench.expected_runtime_state("R"))
        for field in ("taco_ma2_protocol_version", "taco_ma2_packed_load_enabled",
                      "taco_ma2r_protocol_version", "taco_ma2r_submission_enabled"):
            for value in (0, 2):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    bench.runtime_state(self.packed(True, **{field: value}), "R")

    def test_runtime_cannot_claim_owned_output_with_legacy_submission(self):
        with self.assertRaisesRegex(ValueError, "ownership"):
            bench.runtime_state(self.packed(), "R")
        op = self.packed(True)
        op.owns_configuration = False
        with self.assertRaisesRegex(ValueError, "ownership"):
            bench.runtime_state(op, "R")
        op = self.packed(True)
        op.slots = [object(), object()]
        with self.assertRaisesRegex(ValueError, "one communication workspace"):
            bench.runtime_state(op, "R")

    def test_complete_forward_delivers_owned_storage_for_all_arms(self):
        for arm in bench.ARMS:
            output = Mock()
            op = SimpleNamespace(owns_output=arm == "R", forward=Mock(return_value=output))
            result = bench.complete_forward(op, "input", "weight")
            op.forward.assert_called_once_with("input", "weight")
            if arm == "R":
                self.assertIs(result, output)
                output.clone.assert_not_called()
            else:
                output.clone.assert_called_once_with()
                self.assertIs(result, output.clone.return_value)

    def test_retained_outputs_must_use_distinct_storage(self):
        outputs = [SimpleNamespace(data_ptr=lambda i=i: i) for i in range(6)]
        bench.require_independent_outputs(outputs)
        outputs[-1] = outputs[0]
        with self.assertRaisesRegex(ValueError, "alias"):
            bench.require_independent_outputs(outputs)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.plan = fixture_plan()
        self.root = Path("/CPU-only-fixture")
        self.jobs = bench.stage_jobs("preflight")
        self.rows = {(job["directory"], rank): fixture_row(self.plan, job, rank)
                     for job in self.jobs for rank in range(8)}

    def validate(self):
        with patch.object(bench, "read", side_effect=lambda p: self.rows[(p.parent.name, int(p.stem[4:]))]), \
             patch.object(bench, "sha", return_value="1" * 64):
            return bench.validate_rows(self.root, self.plan, self.jobs, "preflight")

    def test_four_distributions_and_all_retained_outputs_match_bitwise(self):
        identity, hashes, files, samples = self.validate()
        self.assertEqual(len(hashes), 16)
        self.assertEqual(len(files), 24)
        self.assertEqual(samples, [])

    def test_any_changed_input_weight_or_output_rejects_bitwise_gate(self):
        for field in ("inputs_sha256", "weights_sha256", "output_sha256"):
            with self.subTest(field=field):
                check = self.rows[("preflight-R", 7)]["roles"]["mlp"]["checks"][0]
                before = check[field]
                check[field] = "9" * 64
                with self.assertRaisesRegex(ValueError, "differs bitwise"): self.validate()
                check[field] = before

    def test_retained_clones_are_compared_across_arms(self):
        self.rows[("preflight-V", 3)]["roles"]["attn"]["retained_output_checks"][4]["output_sha256"] = "8" * 64
        with self.assertRaisesRegex(ValueError, "differs bitwise"): self.validate()

    def test_two_live_instance_outputs_are_compared_across_arms(self):
        isolated = self.rows[("preflight-R", 3)]["roles"]["attn"]["instance_isolation_checks"]
        isolated[3]["output_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "instance output differs bitwise"):
            self.validate()

    def test_missing_or_wrong_role_instance_call_is_rejected(self):
        isolated = self.rows[("preflight-R", 3)]["roles"]["attn"]["instance_isolation_checks"]
        isolated[3]["role"] = "attn"
        with self.assertRaisesRegex(ValueError, "instance configuration check"):
            self.validate()

    def test_retained_storage_or_workspace_alias_evidence_is_rejected(self):
        record = self.rows[("preflight-R", 3)]["roles"]["attn"]
        for field in ("retained_storage_independent", "instance_workspaces_distinct"):
            record[field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate()
            record[field] = True

    def test_missing_distribution_or_reuse_epoch_rejected(self):
        self.rows[("preflight-R", 1)]["roles"]["attn"]["checks"].pop(1)
        with self.assertRaisesRegex(ValueError, "distribution or reuse"): self.validate()

    def test_common_error_cannot_hide_behind_bitwise_agreement(self):
        self.rows[("preflight-A", 0)]["roles"]["attn"]["checks"][0]["correctness"].update(relative_l2=.08)
        with self.assertRaisesRegex(ValueError, "codec correctness"): self.validate()

    def test_nan_or_relaxed_codec_budget_is_rejected(self):
        check = self.rows[("preflight-A", 0)]["roles"]["attn"]["checks"][0]["correctness"]
        for field, value in (("protocol_relative_l2", math.nan), ("budget_vs_codec", .1)):
            old = check[field]
            check[field] = value
            with self.assertRaisesRegex(ValueError, "codec correctness"): self.validate()
            check[field] = old

    def test_mixed_hardware_library_or_runtime_flag_is_rejected(self):
        row = self.rows[("preflight-R", 0)]
        row["hardware"]["host"] = "different"
        with self.assertRaisesRegex(ValueError, "Mixed hardware"): self.validate()
        row["hardware"]["host"] = "CPU-only-fixture"
        row["roles"]["attn"]["runtime_state"]["prefetch_enabled"] = False
        with self.assertRaisesRegex(ValueError, "runtime mechanism"): self.validate()

    def test_report_uses_rank_max_and_retains_negative_prefetch_effects(self):
        timing = {(job["directory"], rank): fixture_row(self.plan, job, rank, "timing")
                  for job in self.plan["schedule"] for rank in range(8)}
        for (directory, rank), row in timing.items():
            if rank != 7:
                for role in bench.ROLES: row["roles"][role]["timing"]["wall_us"] = 1.
        identity, hashes, _, _ = self.validate()
        gate = dict(measurement_identity=identity, tensor_fingerprints=hashes, evidence_files={})
        def read(path):
            if path.name == "preflight.json": return gate
            return timing[(path.parent.name, int(path.stem[4:]))]
        with patch.object(bench, "verify", return_value=self.plan) as verify, patch.object(bench, "preflight", return_value=gate) as preflight, \
             patch.object(bench, "read", side_effect=read), patch.object(bench, "sha", return_value="1" * 64):
            report = bench.report(self.root, save=False, offline=True)
        verify.assert_called_once_with(self.root, offline=True)
        preflight.assert_called_once_with(self.root, save=False, offline=True)
        self.assertFalse(report["target_validated"])
        self.assertTrue(report["bitwise_equal"])
        self.assertEqual(len(report["all_samples"]), 18)
        for role in bench.ROLES:
            summary = report["roles"][role]
            self.assertEqual(summary["mean_us"], {"A": 100., "V": 95., "R": 102.})
            self.assertLess(summary["comparisons"]["R/A"]["percent"], 0)
            self.assertLess(summary["comparisons"]["R/V"]["ci95"][1], 0)
            self.assertEqual(summary["comparisons"]["R/A"]["paired_blocks"], 6)

    def test_paired_statistics_rejects_invalid_or_unpaired_values(self):
        for candidate, baseline in (([1], [1]), ([1, 2], [1]), ([0, 2], [1, 2]), ([math.inf, 2], [1, 2])):
            with self.assertRaises(ValueError): bench.paired(candidate, baseline)
        result = bench.paired([1, 4], [2, 2])
        self.assertAlmostEqual(result["percent"], 0)
        self.assertLess(result["ci95"][0], 0)
        self.assertGreater(result["ci95"][1], 0)


class ProfileTests(unittest.TestCase):
    def test_gpu_mirror_cannot_replace_missing_or_duplicate_cpu_marker(self):
        for duplicate in (False, True):
            events = trace_fixture()
            marker = next(e for e in events if e["cat"] == "user_annotation")
            if duplicate:
                events.append(copy.deepcopy(marker))
            else:
                events.remove(marker)
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(ValueError, "invocation marker"):
                bench.trace_path(events, "R", "attn")

    def test_all_three_paths_bind_by_launch_correlation_not_timestamps(self):
        for arm in bench.ARMS:
            for role in bench.ROLES:
                summary = bench.trace_path(trace_fixture(arm, role), arm, role)
                self.assertTrue(summary["execution_path_validated"])
                self.assertTrue(summary["launch_correlations_validated"])
                self.assertEqual(summary["decoder_median_us"], 5)
                self.assertEqual(summary["cpu_submission_median_us"], 80)
                self.assertEqual(summary["publication_barrier_median_us"], 5)
                self.assertEqual(summary["reuse_barrier_median_us"], 5)
                self.assertEqual(summary["barrier_count"], 8)
                self.assertEqual(summary["clone_count"], 0 if arm == "R" else 4)

    def test_v_and_r_preserve_same_packed_decoder_name(self):
        for arm in ("V", "R"):
            for name in ("void taco_ma2_decode_kernel(Config)", "_Z22taco_ma2_decode_kernel14FluxTacoConfig"):
                self.assertTrue(bench.decoder_matches(name, arm))
            for name in ("taco_ma1_decode_kernel<true>(Config)", "taco_decode_ring_kernel(Config)"):
                self.assertFalse(bench.decoder_matches(name, arm))

    def test_r_rejects_v_clone_path_and_v_rejects_missing_clone(self):
        with self.assertRaisesRegex(ValueError, "clone count"):
            bench.trace_path(trace_fixture("V"), "R", "attn")
        with self.assertRaisesRegex(ValueError, "clone count"):
            bench.trace_path(trace_fixture("R"), "V", "attn")

    def test_wrong_correlation_cannot_pass_matching_kernel_counts(self):
        events = trace_fixture()
        decoder = next(e for e in events if e["cat"] == "kernel" and "taco_ma2" in e["name"])
        decoder["args"]["correlation"] = 999999
        with self.assertRaisesRegex(ValueError, "correlations"):
            bench.trace_path(events, "R", "attn")

    def test_wrong_stream_or_decode_before_publication_rejected(self):
        for mutate in (lambda e: e["args"].update(stream=9), lambda e: e.update(ts=101)):
            events = trace_fixture()
            decode = next(e for e in events if e["cat"] == "kernel" and "taco_ma2" in e["name"])
            mutate(decode)
            with self.assertRaisesRegex(ValueError, "stream ordering"):
                bench.trace_path(events, "R", "attn")

    def test_clone_requires_actual_correlated_device_work(self):
        events = trace_fixture("V")
        events = [e for e in events if not (e["cat"] == "kernel" and "copy" in e["name"])]
        with self.assertRaisesRegex(ValueError, "clone lacks correlated"):
            bench.trace_path(events, "V", "attn")

    def test_legacy_kernel_and_extra_decoder_rejected(self):
        events = trace_fixture()
        events.append(dict(cat="kernel", name="taco_op3_v2_try_decode", ts=1, dur=1, args={}))
        with self.assertRaisesRegex(ValueError, "Legacy or arrival"):
            bench.trace_path(events, "R", "attn")
        events = trace_fixture()
        events.append(copy.deepcopy(next(e for e in events if e["cat"] == "kernel" and "taco_ma2" in e["name"])))
        with self.assertRaisesRegex(ValueError, "path or arm"):
            bench.trace_path(events, "R", "attn")


if __name__ == "__main__":
    unittest.main()

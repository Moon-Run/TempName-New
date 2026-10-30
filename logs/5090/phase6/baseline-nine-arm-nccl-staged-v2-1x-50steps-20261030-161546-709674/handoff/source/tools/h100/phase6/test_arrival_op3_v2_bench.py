"""CPU tests only: fabricated rank records live exclusively in temporary dirs."""
import argparse
import ast
import contextlib
import copy
import gzip
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import tarfile
import unittest
from unittest.mock import Mock, patch

import arrival_op3_v2_bench as bench
import arrival_op3_v2_fullstep as full
import ablation4_sources


class FullStepEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="op3-synthetic-cpu-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = dict(protocol=full.PROTOCOL, run_id="synthetic-CPU-test-only",
            trial_ids={"1": "synthetic-trial-1", "2": "synthetic-trial-2"}, trials=2,
            model=full.MODEL, overlap_rounds=2, warmup_steps=10, timed_steps=50,
            shared_environment={"CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_GEMM_RS_GEMM_ONLY": "1"},
            communication_record={"synthetic": True}, schedule=[],
            acceptance_rule="Synthetic CPU test fixture; never GPU evidence.",
            megatron_snapshot={"patch": {"synthetic": True}},
            libraries={arm: {"files": {f"/test/{arm}/{name}": arm + name for name in
                        ("libflux_cuda.so", "libflux_cuda_ths_op.so")}} for arm in ("A", "D", "O")})
        for trial in (1, 2):
            for job in bench.schedule(4):
                self.plan["schedule"].append(dict(job, trial=trial,
                    round=1 if job["block"] < 2 else 2, directory=f"compare{trial}/{job['directory']}"))
        bench.write(self.root / "plan.json", self.plan)
        self.gate_identity = None
        for job in [*full.jobs(self.plan, "preflight"), *full.jobs(self.plan, "timing")]:
            self.make_window(job)
        self.verify_patch = patch.object(full, "verify", return_value=self.plan)
        self.verify_patch.start()
        self.addCleanup(self.verify_patch.stop)

    def make_window(self, job):
        folder = self.root / "results" / job["directory"]
        folder.mkdir(parents=True)
        for rank in range(8):
            count = 3 if job["mode"] == "smoke" else 50
            modules = [f"module-{i}" for i in range(64)]
            arm = job["arm"]
            lib = arm if arm in ("A", "D") else "O"
            row = dict(completed=True, passed=True, rank=rank, local_rank=rank,
                protocol=full.PROTOCOL, policy=full.POLICY, mode=job["mode"], op3_arm=arm,
                world_size=8, run_id=self.plan["trial_ids"].get(str(job["trial"]), self.plan["run_id"]),
                op3_plan_sha256=bench.sha(self.root / "plan.json"),
                op3_shared_environment=self.plan["shared_environment"],
                op3_mode="serial" if arm == "S" else "overlap", op3_rounds=2, op3_diagnostics="0",
                op3_protocol=bench.SOURCE_PROTOCOL,
                communication=self.plan["communication_record"], sampling=False, profiling=False,
                fallback_calls=0, skips=0, timing_protocol="full-step-one-clear-v2",
                warmup_steps=2 if job["mode"] == "smoke" else 10,
                losses=[1.0] * count, grad_norms=[.1] * count,
                tokens_per_optimizer_step=8192, padded_vocab_size=33792,
                target_modules=modules, replaced_modules=64, independent_scope_instances=64,
                timed_calls={name: count * 4 for name in modules},
                post_audit_routes={name: "candidate" for name in modules},
                model=dict(name="h100-phase6", **{k: v for k, v in full.MODEL.items() if k != "sm_count"}),
                observed_shapes={f"decoder.layers.{layer}.{suffix}":
                    dict(M=2048, N=4096, K_local=k, K_global=k*8)
                    for layer in range(32)
                    for suffix, k in (("self_attention.linear_proj", 512), ("mlp.linear_fc2", 2048))},
                hardware={"host": "synthetic-test-only", "devices": [
                    dict(name="NVIDIA H100 synthetic CPU fixture", uuid=str(i), sm_count=132,
                         capability=[9, 0]) for i in range(8)]},
                forward_checks={name: dict(passed=True, relative_l2=.025, protocol_relative_l2=.0002,
                                           budget_vs_bf16=.05, budget_vs_codec=.01)
                                for name in modules} if job["mode"] == "smoke" else {},
                torch="synthetic", cuda="synthetic", nccl=[0, 0], python="synthetic",
                initial_parameters_sha256=str(rank), initial_rng_sha256=str(rank),
                tokens_sha256="synthetic", timed_steps=count,
                elapsed_seconds=(.49 if arm == "O" else .5) * count)
            for name in ("libflux_cuda.so", "libflux_cuda_ths_op.so"):
                row[name] = dict(path=f"/test/{lib}/{name}", sha256=lib + name)
            bench.write(folder / f"rank{rank}.json", row)

    def modify(self, directory, rank, fn):
        path = self.root / "results" / directory / f"rank{rank}.json"
        row = bench.read(path)
        fn(row)
        path.write_text(json.dumps(row))

    def quiet_report(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return full.report(self.root)

    def test_paired_two_trial_fullstep_gate_and_rank_max(self):
        # Lower seven rank times: the max rank must still determine the window.
        job = next(j for j in self.plan["schedule"] if j["arm"] == "A")
        for rank in range(7):
            self.modify(job["directory"], rank, lambda r: r.update(elapsed_seconds=20.0))
        result = self.quiet_report()
        self.assertTrue(result["target_validated"])
        self.assertEqual(result["formal_optimizer_steps"], 32 * 50)
        window = next(s for s in result["all_samples"] if s["directory"] == job["directory"])
        self.assertEqual(window["ms_per_step"], 500.0)
        self.assertFalse(result["historical_timings_used"])

    def test_second_comparison_must_independently_meet_one_percent(self):
        for job in self.plan["schedule"]:
            if job["trial"] == 2 and job["arm"] == "O":
                for rank in range(8):
                    self.modify(job["directory"], rank, lambda r: r.update(elapsed_seconds=24.9))
        result = self.quiet_report()
        self.assertTrue(result["comparisons"]["compare1"]["target_validated"])
        self.assertFalse(result["comparisons"]["compare2"]["target_validated"])
        self.assertFalse(result["target_validated"])

    def test_fast_decoder_control_does_not_pass_overlap_acceptance(self):
        for job in self.plan["schedule"]:
            if job["arm"] in ("O", "D"):
                for rank in range(8):
                    elapsed = 24 if job["arm"] == "D" else 25.1
                    self.modify(job["directory"], rank, lambda r, v=elapsed: r.update(elapsed_seconds=v))
        result = self.quiet_report()
        self.assertFalse(result["target_validated"])
        for comparison in result["comparisons"].values():
            estimates = comparison["rounds"]["combined"]["comparisons"]
            self.assertGreater(estimates["D/A"]["percent"], 1)
            self.assertLess(estimates["O/A"]["percent"], 0)
            self.assertIn("O/D", estimates)
            self.assertIn("O/S", estimates)

    def test_total_speedup_is_distinct_from_incremental_arrival_benefit(self):
        # O clears 1% over A, but S is still faster than O.
        for job in self.plan["schedule"]:
            if job["arm"] == "S":
                for rank in range(8):
                    self.modify(job["directory"], rank, lambda r: r.update(elapsed_seconds=24.0))
        result = self.quiet_report()
        self.assertTrue(result["target_validated"])
        self.assertFalse(result["arrival_benefit_validated"])
        self.assertEqual(result["attribution_status"], "early_consumption_increment_not_proven")

    def test_fullstep_rejects_profile_instrumentation_in_timing(self):
        job = self.plan["schedule"][0]
        self.modify(job["directory"], 2, lambda r: r.update(op3_diagnostics="1"))
        with self.assertRaisesRegex(ValueError, "runtime settings"):
            full.window(self.root, self.plan, dict(job, mode="timing"))

    def test_fullstep_rejects_skipped_optimizer_update(self):
        job = self.plan["schedule"][0]
        self.modify(job["directory"], 2, lambda r: r.update(skips=1))
        with self.assertRaisesRegex(ValueError, "skipped updates"):
            full.window(self.root, self.plan, dict(job, mode="timing"))

    def test_fullstep_rejects_missing_projection_execution(self):
        job = self.plan["schedule"][0]
        self.modify(job["directory"], 2, lambda r: r["timed_calls"].update({"module-0": 199}))
        with self.assertRaisesRegex(ValueError, "model execution"):
            full.window(self.root, self.plan, dict(job, mode="timing"))

    def test_fullstep_rejects_changed_hardware_between_arms(self):
        self.modify("smoke-O", 3, lambda r: r["hardware"]["devices"][0].update(uuid="different"))
        with self.assertRaisesRegex(ValueError, "hardware/input/software"):
            full.preflight(self.root)

    def test_fullstep_rejects_changed_loss_across_arms(self):
        job = next(j for j in self.plan["schedule"] if j["arm"] == "O")
        for rank in range(8):
            self.modify(job["directory"], rank, lambda r: r.update(losses=[.9] * 50))
        with self.assertRaisesRegex(ValueError, "trajectories differ"):
            self.quiet_report()

    def test_fullstep_rejects_missing_rank(self):
        job = self.plan["schedule"][0]
        (self.root / "results" / job["directory"] / "rank7.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.quiet_report()

    def test_fullstep_rejects_different_library(self):
        job = self.plan["schedule"][0]
        self.modify(job["directory"], 4, lambda r: r["libflux_cuda.so"].update(sha256="different"))
        with self.assertRaisesRegex(ValueError, "Loaded runtime"):
            full.window(self.root, self.plan, dict(job, mode="timing"))


class SourceAndPlanTests(unittest.TestCase):
    def test_all_reused_generated_sources_parse_without_gpu_import(self):
        for function in (ablation4_sources.worker_source, ablation4_sources.adapter_source,
                         ablation4_sources.taco_source):
            ast.parse(function())

    def test_schedule_contains_complete_blocks_and_reversed_second_round(self):
        jobs = bench.schedule(12)
        rows = [[j["arm"] for j in jobs if j["block"] == b] for b in range(12)]
        self.assertTrue(all(sorted(row) == sorted(bench.ARMS) for row in rows))
        self.assertEqual(rows[6:], [row[::-1] for row in rows[:6][::-1]])

    def test_same_runtime_parameters_for_all_arms(self):
        plan = dict(libraries={a: {"root": f"/runtime/{a}"} for a in ("A", "D", "O")},
                    communication={"NCCL_NVLS_ENABLE": "0"},
                    shared_environment={"CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_GEMM_RS_GEMM_ONLY": "1"},
                    overlap_rounds=2)
        for arm in bench.ARMS:
            env = bench.environment(Path("/experiment"), plan, arm, Path("/python/bin/python"), None)
            self.assertEqual(env["CUDA_DEVICE_MAX_CONNECTIONS"], "8")
            self.assertEqual(env["FLUX_GEMM_RS_GEMM_ONLY"], "1")
            self.assertEqual(env["H100_OP3_DIAGNOSTICS"], "0")
            self.assertEqual(env["H100_OP3_MODE"], "serial" if arm == "S" else "overlap")
            self.assertEqual(env["H100_OP3_PROTOCOL"], bench.SOURCE_PROTOCOL)
            library = arm if arm in ("A", "D") else "O"
            self.assertIn(f"/runtime/{library}/python", env["PYTHONPATH"])

    def test_diagnostic_build_cannot_become_gpu_evidence(self):
        import arrival_op3_v2_build as builder
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bench.write(root / "manifest.json", dict(source_only=False, diagnostic_only=True))
            with patch.object(builder, "verify", return_value={"baseline_source_only": False}) as checked:
                with self.assertRaisesRegex(ValueError, "Diagnostic/source-only"):
                    bench.build_identity(root)
                checked.assert_called_once_with(root, built=True)

    def test_fullstep_prepare_freezes_usable_worker_and_unique_trials(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            root = parent / "new"
            args = argparse.Namespace(root=root, build=parent / "build", megatron=parent / "megatron",
                                      trials=2, blocks=4, rounds=2)
            identities = {a: dict(root=str(parent / a), files={}) for a in ("A", "D", "O")}
            with patch.object(bench, "build_identity", return_value={"cpu_test": True}), \
                 patch.object(bench, "libraries", return_value=identities), \
                 patch.object(full, "check_megatron"), \
                 patch.object(full, "snapshot_megatron", return_value={"synthetic": True}), \
                 contextlib.redirect_stdout(io.StringIO()):
                full.prepare(args)
            for path in (root / "scripts").glob("*.py"):
                ast.parse(path.read_text())
            plan = bench.read(root / "plan.json")
            self.assertEqual(len(set(plan["trial_ids"].values())), 2)
            self.assertFalse(plan["historical_hardware_reused"])
            self.assertFalse(plan["target_validated"])
            source = (root / "scripts/worker.py").read_text()
            self.assertIn("report['op3_arm']", source)
            self.assertIn("report['op3_shared_environment']", source)
            config = bench.read(root / "scripts/config.json")
            self.assertEqual((config["warmup_steps"], config["timed_steps"]), (10, 50))
            self.assertEqual(config["policies"], [full.POLICY])

    def test_megatron_guard_patch_only_modifies_independent_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "original", Path(temp) / "snapshot"
            guard = source / "megatron/training/arguments.py"
            guard.parent.mkdir(parents=True)
            original = "def f():\n" + full.MEGATRON_GUARD + "\n"
            guard.write_text(original)
            (source / "unchanged.py").write_text("value = 123\n")
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                archive.add(guard, arcname="megatron/training/arguments.py")
                archive.add(source / "unchanged.py", arcname="unchanged.py")
            with patch.object(full.subprocess, "check_output", return_value=stream.getvalue()):
                record = full.snapshot_megatron(source, target)
            self.assertEqual(guard.read_text(), original)
            self.assertEqual(record["changed_files"], ["megatron/training/arguments.py"])
            self.assertEqual((target / "unchanged.py").read_text(), "value = 123\n")
            full.verify_megatron_snapshot(record)
            (target / "unchanged.py").write_text("value = 456\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                full.verify_megatron_snapshot(record)

    def test_delay_stress_targets_two_ranks_only_in_correctness_epochs(self):
        plan = {"seeds": [817, 919, 1231], "slow_rank_stress": {"ranks": [0, 7], "cycles": 2000000}}
        delays = [bench.correctness_delay(plan, epoch, seed)
                  for epoch in range(4) for seed in plan["seeds"]]
        self.assertEqual([d for d in delays if d[0] >= 0], [(0, 2000000), (7, 2000000)])

    def test_timeout_kills_only_new_torchrun_process_group(self):
        process = Mock(pid=432123)
        process.wait.side_effect = [subprocess.TimeoutExpired(["torchrun"], 1), 143, 143]
        with patch.object(bench.subprocess, "Popen", return_value=process) as launch, \
             patch.object(bench.os, "killpg") as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                bench.run_worker(["torchrun"], "/tmp", {}, None, 1)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertEqual(kill.call_args_list, [
            unittest.mock.call(432123, signal.SIGTERM), unittest.mock.call(432123, signal.SIGKILL)])

    def test_success_does_not_signal_any_process(self):
        process = Mock(pid=432123)
        process.wait.return_value = 0
        with patch.object(bench.subprocess, "Popen", return_value=process), \
             patch.object(bench.os, "killpg") as kill:
            self.assertEqual(bench.run_worker(["torchrun"], "/tmp", {}, None, 1), 0)
        kill.assert_not_called()


class OperatorDiagnosticsTests(unittest.TestCase):
    def samples(self, attn=25, mlp=25):
        rows = []
        for block in range(6):
            for arm in bench.ARMS:
                # D is intentionally much faster: its savings must not be attributed to O.
                values = {"attn": 1000.0, "mlp": 2000.0}
                if arm == "O":
                    values = {"attn": 1000.0 - attn, "mlp": 2000.0 - mlp}
                elif arm == "D":
                    values = {"attn": 500.0, "mlp": 1000.0}
                rows.append(dict(block=block, arm=arm, rankmax_wall_us=values))
        return rows

    def diagnostic(self, invocation=0, useful=True):
        return dict(protocol=bench.SOURCE_PROTOCOL, mode="overlap", rounds=2,
                    invocation=invocation, epoch=100 + invocation, work_unit="h128_group",
                    max_try_groups=1024, try_groups=8 if useful else 0, early_scan_ctas=64,
                    stop_on_local_gemm=True, groups_finished_after_stop=0, stop_ns=900,
                    first_decode_complete_ns=100 if useful else 0,
                    last_local_publish_ns=500, useful_overlap=useful)

    def test_estimate_records_negative_role_savings_without_a_gate(self):
        estimate = bench.savings_estimate(self.samples(attn=-1, mlp=60))
        self.assertEqual(estimate["role_mean_saving_us"]["attn"], -1)
        self.assertLess(estimate["role_paired_reduction_percent"]["attn"], 0)
        self.assertFalse({"passed", "failure_reasons", "required_mean_ms"} & estimate.keys())

    def test_decoder_control_does_not_contribute_to_overlap_estimate(self):
        estimate = bench.savings_estimate(self.samples(attn=1, mlp=1))
        self.assertAlmostEqual(estimate["mean_ms"], .256)

    def test_estimate_preserves_pairing_and_uncertain_savings(self):
        rows = self.samples(attn=25, mlp=25)
        for row in rows:
            if row["arm"] == "O":
                row["rankmax_wall_us"]["attn"] += 200 if row["block"] % 2 else -200
        estimate = bench.savings_estimate(rows)
        self.assertAlmostEqual(estimate["mean_ms"], 6.4)
        self.assertLess(estimate["ci95_ms"][0], 0)
        self.assertGreater(estimate["ci95_ms"][1], 0)

    def test_estimate_rejects_duplicate_or_missing_arms(self):
        rows = self.samples()
        rows[0]["arm"] = rows[1]["arm"]
        with self.assertRaisesRegex(ValueError, "complete unique"):
            bench.savings_estimate(rows)

    def test_old_or_unbounded_diagnostics_rejected(self):
        for change in (dict(protocol="h100-arrival-op3-v1"), dict(try_groups=1025),
                       dict(work_unit="fragment"), dict(stop_on_local_gemm=False)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                bench.validate_diagnostic(dict(self.diagnostic(), **change), "overlap", 2)

    def events(self, overlap_indices=(0, 1, 2, 3), rounds=2):
        gemms, decoders = [], []
        for invocation in range(4):
            stamp = invocation * 100
            gemms.append(dict(name="flux_bf16_reducescatter", ts=stamp, dur=20, args={"stream": 1}))
            for scan_round in range(rounds):
                decoders.append(dict(name="taco_op3_v2_try_decode",
                                     ts=stamp + (5 if invocation in overlap_indices else 30) + scan_round * 4,
                                     dur=3, args={"stream": 2}))
        return gemms, decoders

    def test_early_timestamp_and_trace_overlap_must_be_same_call(self):
        gemms, decoders = self.events(overlap_indices=(1,))
        diagnostics = [self.diagnostic(i, useful=i == 0) for i in range(4)]
        matches = bench.matched_early_calls(gemms, decoders, diagnostics, 2)
        self.assertFalse(any(call["validated"] for call in matches))
        diagnostics[1] = self.diagnostic(1, useful=True)
        self.assertEqual(sum(call["validated"] for call in
                             bench.matched_early_calls(gemms, decoders, diagnostics, 2)), 1)

    def test_tail_overlap_cannot_substitute_for_try_overlap(self):
        gemms, decoders = self.events()
        decoders[0]["name"] = "taco_op3_v2_tail_decode"
        with self.assertRaisesRegex(ValueError, "one GEMM"):
            bench.matched_early_calls(gemms, decoders, [self.diagnostic(i) for i in range(4)], 2)

    def test_diagnostics_must_be_associated_with_unique_invocations(self):
        gemms, decoders = self.events()
        with self.assertRaisesRegex(ValueError, "epochs"):
            bench.matched_early_calls(gemms, decoders, [self.diagnostic()] * 4, 2)

    def test_each_configured_round_is_matched_to_the_correct_call(self):
        for rounds in (1, 2, 3, 4):
            with self.subTest(rounds=rounds):
                gemms, decoders = self.events(rounds=rounds)
                diagnostics = [dict(self.diagnostic(i), rounds=rounds) for i in range(4)]
                matches = bench.matched_early_calls(gemms, decoders, diagnostics, rounds)
                self.assertEqual(len(decoders), 4 * rounds)
                self.assertEqual(sum(call["validated"] for call in matches), 4)
                self.assertTrue(all(call["bounded_try_kernels"] == rounds for call in matches))
                with self.assertRaisesRegex(ValueError, "configured bounded-try rounds"):
                    bench.matched_early_calls(gemms, decoders[:-1], diagnostics, rounds)

    def test_profile_acceptance_requires_matching_useful_call_in_each_role(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = dict(run_id="synthetic-profile-only", overlap_rounds=2)
            bench.write(root / "plan.json", plan)
            for arm in bench.ARMS:
                folder = root / "results" / f"profile-{arm}"
                folder.mkdir(parents=True)
                for rank in range(8):
                    row = dict(completed=True, rank=rank, profiling=True,
                               protocol=bench.PROTOCOL, run_id=plan["run_id"],
                               plan_sha256=bench.sha(root / "plan.json"), roles={})
                    for role in bench.ROLES:
                        gemms, decoders = self.events(rounds=2 if arm == "O" else 1)
                        diagnostics = []
                        if arm in ("S", "O"):
                            diagnostics = [self.diagnostic(i, useful=arm == "O" and role == "attn")
                                           for i in range(4)]
                            if arm == "S":
                                for diagnostic in diagnostics:
                                    diagnostic["mode"] = "serial"
                        if arm != "O":
                            for decoder in decoders:
                                decoder["name"] = {"A": "taco_decode", "D": "taco_op3_v2_decode",
                                                   "S": "taco_op3_v2_tail_decode"}[arm]
                        for event in gemms + decoders:
                            event["cat"] = "kernel"
                        with gzip.open(folder / f"rank{rank}-{role}-profile.json.gz", "wt") as handle:
                            json.dump(dict(traceEvents=gemms + decoders), handle)
                        row["roles"][role] = dict(diagnostics=diagnostics)
                    bench.write(folder / f"rank{rank}.json", row)
            with patch.object(bench, "verify", return_value=plan):
                summary = bench.profile_summary(root)
                self.assertGreater(summary["useful_and_overlapped_calls_by_role"]["attn"], 0)
                self.assertEqual(summary["useful_and_overlapped_calls_by_role"]["mlp"], 0)
                self.assertFalse(summary["useful_early_consumption_validated"])
                path = root / "results/profile-O/rank0.json"
                row = bench.read(path)
                row["roles"]["mlp"]["diagnostics"][0] = self.diagnostic(0, useful=True)
                path.write_text(json.dumps(row))
                self.assertTrue(bench.profile_summary(root)["useful_early_consumption_validated"])


if __name__ == "__main__":
    unittest.main()

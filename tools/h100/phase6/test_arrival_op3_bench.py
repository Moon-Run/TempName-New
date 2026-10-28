"""CPU tests only: fabricated rank records live exclusively in temporary dirs."""
import argparse
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import tarfile
import unittest
from unittest.mock import Mock, patch

import arrival_op3_bench as bench
import arrival_op3_fullstep as full
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
                        ("libflux_cuda.so", "libflux_cuda_ths_op.so")}} for arm in ("A", "O")})
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
            lib = "A" if arm == "A" else "O"
            row = dict(completed=True, passed=True, rank=rank, local_rank=rank,
                protocol=full.PROTOCOL, policy=full.POLICY, mode=job["mode"], op3_arm=arm,
                world_size=8, run_id=self.plan["trial_ids"].get(str(job["trial"]), self.plan["run_id"]),
                op3_plan_sha256=bench.sha(self.root / "plan.json"),
                op3_shared_environment=self.plan["shared_environment"],
                op3_mode="serial" if arm == "S" else "overlap", op3_rounds=2, op3_diagnostics="0",
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
        self.assertEqual(result["formal_optimizer_steps"], 24 * 50)
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
        plan = dict(libraries={a: {"root": f"/runtime/{a}"} for a in ("A", "O")},
                    communication={"NCCL_NVLS_ENABLE": "0"},
                    shared_environment={"CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_GEMM_RS_GEMM_ONLY": "1"},
                    overlap_rounds=2)
        for arm in bench.ARMS:
            env = bench.environment(Path("/experiment"), plan, arm, Path("/python/bin/python"), None)
            self.assertEqual(env["CUDA_DEVICE_MAX_CONNECTIONS"], "8")
            self.assertEqual(env["FLUX_GEMM_RS_GEMM_ONLY"], "1")
            self.assertEqual(env["H100_OP3_DIAGNOSTICS"], "0")
            self.assertEqual(env["H100_OP3_MODE"], "serial" if arm == "S" else "overlap")

    def test_diagnostic_build_cannot_become_gpu_evidence(self):
        import arrival_op3_build as builder
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
            identities = {a: dict(root=str(parent / a), files={}) for a in ("A", "O")}
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


if __name__ == "__main__":
    unittest.main()

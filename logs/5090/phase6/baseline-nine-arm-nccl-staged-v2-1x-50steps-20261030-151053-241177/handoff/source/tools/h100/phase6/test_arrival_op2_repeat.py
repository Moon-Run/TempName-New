"""CPU orchestration checks: same-codec probe, frozen selection, held-out repeats."""
import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import arrival_op2_build as builder
import arrival_op2_protocol as protocol
import arrival_op2_report as reporter
import arrival_op2_probe as probe
import detach_arrival_op2 as batch
from common import read, sha, write


class BatchTests(unittest.TestCase):
    def args(self, root, trials=2, extra=()):
        return batch.arguments(["--out", str(root), "--trials", str(trials), *extra])

    def selection(self, path, spans=None, source_only=False):
        result = dict(protocol=protocol.PROTOCOL, probe_protocol=probe.PROBE_PROTOCOL,
                      source_only=source_only, executable=not source_only,
                      gpu_measured=not source_only, run_id="probe/test",
                      status="completed", selected_spans=spans or {"attn": 4, "mlp": 8},
                      evidence_files={"probe.json": "frozen-evidence"})
        result["selection_sha256"] = probe.digest(result)
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, result)
        return result

    def info(self, root, selection_path):
        selected = read(selection_path)
        return dict(
            protocol=protocol.PROTOCOL, run_id="op2/"+root.name,
            executable=True, source_only=False, execution_timing=dict(protocol.TIMING),
            schedule_options=selected["selected_spans"],
            selection_sha256=sha(selection_path),
            selection=dict(path="provenance/selection/selection.json", sha256=sha(selection_path),
                           selected_spans=selected["selected_spans"], gpu_measured=True),
            schedule=protocol.schedule(6, rounds=2), timing_windows=24,
            communication=dict(environment={"NCCL_NVLS_ENABLE": "0"}, nvls_request="0"),
            config={}, calibration_hardware={}, megatron="same", megatron_commit="same",
            candidate_source_commit="same", build_manifest={}, build_files={},
            controllers={}, files={"scripts/worker.py": "same"},
            variants={p: protocol.definition(p) for p in protocol.POLICIES},
        )

    def report(self, root, point=2.0, lower=1.6):
        info = read(root/"submission.json")
        gate = dict(completed=True, protocol=protocol.PROTOCOL,
                    execution_timing=dict(protocol.TIMING),
                    schedule_options=info["schedule_options"],
                    selection_sha256=info["selection_sha256"],
                    submission_sha256=sha(root/"submission.json"),
                    communication=info["communication"])
        write(root/"preflight.json", gate)
        summary = {"comparisons": {
            protocol.S+"/"+protocol.A: {"percent": point, "ci95": [lower, point+0.4]},
        }}
        acceptance = reporter.acceptance(summary, nonidentity=info["schedule_options"] != protocol.IDENTITY_SPANS)
        return dict(
            protocol=protocol.PROTOCOL, run_id=info["run_id"], completed=True,
            correctness_passed=True, execution_timing=dict(protocol.TIMING),
            schedule_options=info["schedule_options"], selection_sha256=info["selection"]["sha256"],
            formal_optimizer_steps=1200, timing_windows=24, all_samples=[{}]*24,
            round_count=2, blocks_per_round=6, timing_rank_records=192,
            preflight_rank_records=32, profile_traces=16, candidate_shape_mappings=4,
            communication=info["communication"], measurement_identity={"same_runtime": True},
            role_plan_sha256={"same_plans": True},
            arrival_schedule={"same_order": True}, acceptance_threshold=1.5,
            primary_comparisons=protocol.PRIMARY, comparisons=protocol.COMPARISONS,
            rounds={key: copy.deepcopy(summary) for key in ("1", "2", "combined")},
            acceptance_by_round={key: copy.deepcopy(acceptance) for key in ("1", "2", "combined")},
            acceptance=acceptance, evidence={"preflight_sha256": sha(root/"preflight.json")},
        )

    def fake_invoke(self, calls, mutate=None, gain=None, selected_spans=None):
        def invoke(command, log, env):
            label = log.stem
            calls.append(label)
            self.assertEqual(Path(command[1]).name, "arrival_op2.py")
            self.assertEqual(env["NCCL_NVLS_ENABLE"], "0")
            if label.startswith("compare"):
                root = Path(command[3])
                if label.endswith("-prepare"):
                    root.mkdir()
                    selection_path = Path(command[command.index("--selection")+1])
                    write(root/"submission.json", self.info(root, selection_path))
                    (root/"scripts").mkdir()
                    write(root/"scripts/config.json", {"role_plan_sha256": {"same_plans": True}})
                elif label.endswith("-controller"):
                    write(root/"all-state.json", {"status": "completed"})
                elif label.endswith("-report"):
                    values = (2.0, 1.6) if gain is None else gain[root.name]
                    write(root/"report.json", self.report(root, *values))
            elif label == "probe-prepare":
                root = Path(command[3])
                root.mkdir()
                write(root/"probe.json", {"retained_probe": True})
            elif label == "probe-controller":
                root = Path(command[3])
                self.selection(root/"selection.json", selected_spans)
                write(root/"execution.json", {"status": "completed", "run_id": "probe/test"})
            else:
                root = log.parent
            if mutate:
                mutate(label, root, command)
        return invoke

    def run_fake(self, args, calls, mutate=None, gain=None, selected_spans=None):
        with mock.patch.object(batch, "resolve_nccl"), \
                mock.patch.object(batch.launcher, "invoke",
                                  side_effect=self.fake_invoke(calls, mutate, gain, selected_spans)), \
                mock.patch.object(batch.reporter, "verify_inputs",
                                  side_effect=lambda root: read(root/"submission.json")), \
                mock.patch.object(probe, "selection_verify", side_effect=lambda path, *a, **kw: read(path)), \
                mock.patch.object(builder, "verify", return_value={"parent_build": str(args.build)}), \
                contextlib.redirect_stdout(io.StringIO()):
            return batch.supervise(args)

    def test_preview_does_not_create_directories_or_spawn_processes(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(os, "fork", side_effect=AssertionError("fork")), \
                mock.patch.object(subprocess, "run", side_effect=AssertionError("subprocess")), \
                mock.patch.object(batch, "resolve_nccl", side_effect=AssertionError("NCCL lookup")), \
                mock.patch.object(builder, "verify", side_effect=AssertionError("build lookup")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            root = Path(tmp)/"absent"
            batch.main(["--out", str(root)])
            self.assertFalse(root.exists())
            self.assertIn("2400 formal steps", output.getvalue())
            self.assertNotIn("compare3", output.getvalue())
            self.assertNotIn("decoder-check", output.getvalue())
            self.assertIn("--selection", output.getvalue())

    def test_probe_before_two_balanced_confirmations_and_same_frozen_selection(self):
        args = self.args(Path("/tmp/op2-command-only"))
        tasks = batch.commands(args)
        labels = [label for label, _ in tasks]
        self.assertEqual(labels[:5], ["build-prepare", "compile", "probe-prepare",
                                     "probe-dry-run", "probe-controller"])
        self.assertEqual(labels.count("compile"), 1)
        self.assertIn("--execute", tasks[4][1])
        self.assertNotIn("--execute", tasks[3][1])
        selections = []
        for trial in ("compare1", "compare2"):
            self.assertEqual([label for label in labels if label.startswith(trial)],
                             [trial+"-"+stage for stage in ("prepare", "dry-run", "controller", "report")])
            command = dict(tasks)[trial+"-prepare"]
            selections.append(command[command.index("--selection")+1])
        self.assertEqual(selections, [str(args.out/"probe/selection.json")]*2)
        schedule = protocol.schedule(6, rounds=2)
        self.assertEqual(len(schedule), 24)
        for rep in (1, 2):
            for position in range(2):
                policies = [row["policy"] for row in schedule
                            if row["round"] == rep and row["index"] % 2 == position]
                self.assertEqual({p: policies.count(p) for p in protocol.POLICIES},
                                 {p: 3 for p in protocol.POLICIES})
        args.op2_build = Path("/tmp/existing-op2-build")
        reused = [label for label, _ in batch.commands(args)]
        self.assertEqual(reused[0], "probe-prepare")
        self.assertNotIn("compile", reused)
        self.assertNotIn("build-prepare", reused)
        self.assertNotIn("decoder-check", reused)

    def test_existing_selection_skips_probe_and_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root/"external/selection.json"
            self.selection(selected)
            args = self.args(root, extra=("--selection", str(selected), "--op2-build", str(root/"build")))
            self.assertTrue(all(label.startswith("compare") for label, _ in batch.commands(args)))
            calls = []
            with mock.patch.object(probe, "selection_verify", side_effect=lambda path, *a, **kw: read(path)) as check:
                # run_fake installs its own verifier; verify explicitly with the same
                # mock in this test by observing calls through a wrapped read.
                with mock.patch.object(batch, "resolve_nccl"), \
                        mock.patch.object(batch.launcher, "invoke", side_effect=self.fake_invoke(calls)), \
                        mock.patch.object(batch.reporter, "verify_inputs", side_effect=lambda path: read(path/"submission.json")), \
                        mock.patch.object(builder, "verify", return_value={"parent_build": str(args.build)}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    result = batch.supervise(args)
            self.assertTrue(check.called)
            self.assertEqual(Path(check.call_args.args[0]), selected)
            self.assertTrue(result["target_validated"])
            self.assertNotIn("probe-controller", calls)

    def test_two_confirmations_complete_only_after_all_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            state = self.run_fake(self.args(Path(tmp)), calls)
            self.assertEqual(calls[:5], ["build-prepare", "compile", "probe-prepare",
                                       "probe-dry-run", "probe-controller"])
            self.assertEqual(calls[-1], "compare2-report")
            self.assertEqual(state["status"], "completed")
            self.assertTrue(state["target_validated"])
            self.assertEqual((state["timing_windows"], state["timing_rank_records"],
                              state["formal_optimizer_steps"]), (48, 384, 2400))

    def test_identity_fallback_keeps_probe_and_starts_no_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            state = self.run_fake(self.args(root), calls, selected_spans=dict(protocol.IDENTITY_SPANS))
            self.assertEqual(calls[-1], "probe-controller")
            self.assertEqual(state["status"], "completed_no_candidate")
            self.assertFalse(state["target_validated"])
            self.assertEqual(state["formal_optimizer_steps"], 0)
            self.assertEqual(state["timing_windows"], 0)
            self.assertFalse((root/"compare1").exists())
            self.assertTrue((root/"probe/selection.json").is_file())
            self.assertEqual(read(root/"probe/probe.json"), {"retained_probe": True})

    def test_missing_target_is_not_execution_failure_and_no_run_is_dropped(self):
        for bad in ((-0.1, -0.3), (1.7, 1.49)):
            with self.subTest(gain=bad), tempfile.TemporaryDirectory() as tmp:
                calls = []
                state = self.run_fake(self.args(Path(tmp)), calls,
                                      gain={"compare1": (2.0, 1.6), "compare2": bad})
                self.assertEqual(calls[-1], "compare2-report")
                self.assertEqual(state["status"], "completed")
                self.assertFalse(state["target_validated"])
                self.assertEqual(state["acceptance_status"], "not_validated_1.5_percent")

    def test_failed_or_incomplete_stage_stops_before_second_confirmation(self):
        for stage in ("compile", "probe-controller", "compare1-controller", "compare1-report"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls = []
                def corrupt(label, folder, command):
                    if label != stage:
                        return
                    if label.endswith("-report"):
                        result = read(folder/"report.json")
                        result["all_samples"].pop()
                        write(folder/"report.json", result)
                    else:
                        raise subprocess.CalledProcessError(1, command)
                with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                    self.run_fake(self.args(root), calls, mutate=corrupt)
                self.assertEqual(calls[-1], stage)
                self.assertNotIn("compare2-prepare", calls)
                self.assertEqual(read(root/"batch-state.json")["status"], "failed")

    def test_repeat_input_drift_or_duplicate_run_id_stops_before_gpu(self):
        for key, value in (("run_id", "op2/compare1"),
                           ("files", {"scripts/worker.py": "changed"}),
                           ("build_files", {"changed": "hash"}),
                           ("schedule_options", {"attn": 1, "mlp": 8}),
                           ("selection_sha256", "changed")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls = []
                def corrupt(label, folder, command):
                    if label == "compare2-prepare":
                        info = read(folder/"submission.json")
                        info[key] = value
                        write(folder/"submission.json", info)
                with self.assertRaises(ValueError):
                    self.run_fake(self.args(root), calls, mutate=corrupt)
                self.assertEqual(calls[-1], "compare2-dry-run")
                self.assertNotIn("compare2-controller", calls)
                self.assertEqual(read(root/"batch-state.json")["status"], "failed")

    def test_identity_drift_after_second_report_cannot_validate_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            def corrupt(label, folder, command):
                if label == "compare2-report":
                    result = read(folder/"report.json")
                    result["measurement_identity"] = {"different": True}
                    write(folder/"report.json", result)
            with self.assertRaises(ValueError):
                self.run_fake(self.args(root), calls, mutate=corrupt)
            self.assertEqual(read(root/"batch-state.json")["status"], "failed")

    def test_four_trial_run_cannot_reuse_second_run_id_for_third(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            def corrupt(label, folder, command):
                if label == "compare3-prepare":
                    info = read(folder/"submission.json")
                    info["run_id"] = "op2/compare2"
                    write(folder/"submission.json", info)
            with self.assertRaises(ValueError):
                self.run_fake(self.args(root, trials=4), calls, mutate=corrupt)
            self.assertEqual(calls[-1], "compare3-dry-run")
            self.assertNotIn("compare3-controller", calls)

    def test_probe_run_id_cannot_be_reused_by_first_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            def corrupt(label, folder, command):
                if label == "compare1-prepare":
                    info = read(folder/"submission.json")
                    info["run_id"] = "probe/test"
                    write(folder/"submission.json", info)
            with self.assertRaises(ValueError):
                self.run_fake(self.args(root), calls, mutate=corrupt)
            self.assertEqual(calls[-1], "compare1-dry-run")
            self.assertNotIn("compare1-controller", calls)

    def test_detach_readiness_failure_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(batch.launcher, "detach", side_effect=RuntimeError("readiness timeout")), \
                contextlib.redirect_stdout(io.StringIO()):
            parent = Path(tmp)/"parent"
            parent.mkdir()
            (parent/"ablation4-manifest.json").write_text("{}")
            toolkit = Path(tmp)/"cuda"
            (toolkit/"bin").mkdir(parents=True)
            (toolkit/"bin/nvcc").write_text("synthetic")
            root = Path(tmp)/"new"
            with self.assertRaisesRegex(RuntimeError, "readiness timeout"):
                batch.main(["--out", str(root), "--build", str(parent),
                            "--build-cuda-home", str(toolkit), "--execute"])
            state = read(root/"batch-state.json")
            self.assertEqual(state["status"], "failed")
            self.assertIn("readiness timeout", state["error"])

    def test_forged_cached_target_cannot_rescue_negative_primary_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def corrupt(label, folder, command):
                if label == "compare1-report":
                    result = read(folder/"report.json")
                    result["rounds"]["combined"]["comparisons"][protocol.S+"/"+protocol.A] = {
                        "percent": -0.5, "ci95": [-0.8, -0.2]}
                    write(folder/"report.json", result)
            with self.assertRaises(ValueError):
                self.run_fake(self.args(Path(tmp)), calls, mutate=corrupt)
            self.assertNotIn("compare2-prepare", calls)

    def test_cached_acceptance_or_frozen_selection_drift_rejected(self):
        corruptions = (
            ("acceptance", {"passed": "false"}),
            ("acceptance", {"candidate": protocol.A}),
            ("acceptance", {"threshold_percent": 0.0}),
            ("acceptance", {"ci95": [2.0, 1.0]}),
            ("schedule_options", {"attn": 32, "mlp": 32}),
            ("selection_sha256", "changed"),
        )
        for key, change in corruptions:
            with self.subTest(key=key, change=change), tempfile.TemporaryDirectory() as tmp:
                calls = []
                def corrupt(label, folder, command):
                    if label == "compare1-report":
                        result = read(folder/"report.json")
                        if key == "acceptance":
                            result[key].update(change)
                        else:
                            result[key] = change
                        write(folder/"report.json", result)
                with self.assertRaises(ValueError):
                    self.run_fake(self.args(Path(tmp)), calls, mutate=corrupt)
                self.assertNotIn("compare2-prepare", calls)

    def test_cpu_fixture_selection_cannot_start_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root/"external/selection.json"
            self.selection(selected, source_only=True)
            calls = []
            with self.assertRaises(ValueError):
                self.run_fake(self.args(root, extra=("--selection", str(selected),
                                                     "--op2-build", str(root/"build"))), calls)
            self.assertFalse(any(label.startswith("compare") for label in calls))

    def test_identity_report_cannot_pass_even_with_large_forged_speedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root/"selection.json"
            self.selection(selected, dict(protocol.IDENTITY_SPANS))
            info = self.info(root, selected)
            write(root/"submission.json", info)
            (root/"scripts").mkdir()
            write(root/"scripts/config.json", {"role_plan_sha256": {"same_plans": True}})
            write(root/"all-state.json", {"status": "completed"})
            write(root/"report.json", self.report(root, 10.0, 9.0))
            checked = batch.check_report(root, info)
            self.assertFalse(checked["acceptance"]["passed"])
            self.assertFalse(checked["acceptance"]["schedule_changed"])


if __name__ == "__main__":
    unittest.main()

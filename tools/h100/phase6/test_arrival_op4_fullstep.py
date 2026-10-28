"""CPU tests only: fabricated rank records live exclusively in temporary dirs."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import types
from unittest.mock import patch

import arrival_op4_bench as bench
import arrival_op4_fullstep as full


class FullStepEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="op4-synthetic-cpu-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = dict(protocol=full.PROTOCOL, run_id="synthetic-CPU-test-only",
            trial_ids={"1": "synthetic-trial-1", "2": "synthetic-trial-2"}, trials=2,
            model=full.MODEL, blocks=4, arms=list(bench.ARMS), roles=bench.ROLES,
            warmup_steps=10, timed_steps=50, source_files={}, target_percent=1.0,
            shared_environment={"CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_GEMM_RS_GEMM_ONLY": "1"},
            communication_record={"synthetic": True}, schedule=[],
            acceptance_rule="Synthetic CPU test fixture; never GPU evidence.",
            megatron_snapshot={"patch": {"synthetic": True}},
            libraries={arm: {"files": {f"/test/{arm}/{name}": arm + name for name in
                        ("libflux_cuda.so", "libflux_cuda_ths_op.so")}} for arm in bench.ARMS})
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
            lib = arm
            row = dict(completed=True, passed=True, rank=rank, local_rank=rank,
                protocol=full.PROTOCOL, policy=full.POLICY, mode=job["mode"], op4_arm=arm,
                world_size=8, run_id=self.plan["trial_ids"].get(str(job["trial"]), self.plan["run_id"]),
                op4_plan_sha256=bench.sha(self.root / "plan.json"),
                op4_shared_environment=self.plan["shared_environment"],
                op4_protocol=bench.SOURCE_PROTOCOL,
                op4_codec_state={name: dict(owns_output=arm in ('L','F'),
                    owns_configuration=arm in ('L','F'), workspaces=2 if arm in ('L','F') else 1)
                    for name in modules},
                workspace_checks={name: [dict(shape=[2048,4096,4096], packet_ptr=1000+i*2+j)
                    for j in range(2 if arm in ('L','F') else 1)] for i,name in enumerate(modules)},
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
                elapsed_seconds=(.49 if arm == "F" else .5) * count)
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
            return full.report(self.root, save=False)

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
            if job["trial"] == 2 and job["arm"] == "F":
                for rank in range(8):
                    self.modify(job["directory"], rank, lambda r: r.update(elapsed_seconds=24.9))
        result = self.quiet_report()
        self.assertTrue(result["comparisons"]["compare1"]["target_validated"])
        self.assertFalse(result["comparisons"]["compare2"]["target_validated"])
        self.assertFalse(result["target_validated"])

    def test_fast_decoder_control_does_not_pass_combined_acceptance(self):
        for job in self.plan["schedule"]:
            if job["arm"] in ("F", "D"):
                for rank in range(8):
                    elapsed = 24 if job["arm"] == "D" else 25.1
                    self.modify(job["directory"], rank, lambda r, v=elapsed: r.update(elapsed_seconds=v))
        result = self.quiet_report()
        self.assertFalse(result["target_validated"])
        for comparison in result["comparisons"].values():
            estimates = comparison["rounds"]["combined"]["comparisons"]
            self.assertGreater(estimates["D/A"]["percent"], 1)
            self.assertLess(estimates["F/A"]["percent"], 0)
            self.assertIn("F/D", estimates)
            self.assertIn("F/L", estimates)

    def test_total_speedup_is_distinct_from_decoder_increment(self):
        # O clears 1% over A, but S is still faster than O.
        for job in self.plan["schedule"]:
            if job["arm"] == "L":
                for rank in range(8):
                    self.modify(job["directory"], rank, lambda r: r.update(elapsed_seconds=24.0))
        result = self.quiet_report()
        self.assertTrue(result["target_validated"])
        self.assertFalse(result["decoder_increment_validated"])
        self.assertTrue(result["lifecycle_benefit_validated"])

    def test_fullstep_rejects_profile_instrumentation_in_timing(self):
        job = self.plan["schedule"][0]
        self.modify(job["directory"], 2, lambda r: r.update(op4_protocol="h100-arrival-op3-v2"))
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
        self.modify("smoke-F", 3, lambda r: r["hardware"]["devices"][0].update(uuid="different"))
        with self.assertRaisesRegex(ValueError, "hardware/input/software"):
            full.preflight(self.root)

    def test_fullstep_rejects_changed_loss_across_arms(self):
        job = next(j for j in self.plan["schedule"] if j["arm"] == "F")
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

    def test_lifecycle_requires_independent_workspaces(self):
        job = next(j for j in self.plan["schedule"] if j["arm"] == "L")
        def alias(row):
            slots = row["workspace_checks"]["module-0"]
            slots[1]["packet_ptr"] = slots[0]["packet_ptr"]
        self.modify(job["directory"], 0, alias)
        with self.assertRaisesRegex(ValueError, "Aliased communication"):
            full.window(self.root, self.plan, dict(job, mode="timing"))

    def test_baseline_cannot_silently_acquire_owned_output(self):
        job = next(j for j in self.plan["schedule"] if j["arm"] == "A")
        self.modify(job["directory"], 0,
                    lambda row: row["op4_codec_state"]["module-0"].update(owns_output=True))
        with self.assertRaisesRegex(ValueError, "output ownership"):
            full.window(self.root, self.plan, dict(job, mode="timing"))

    def test_components_cannot_substitute_for_combined_target(self):
        for job in self.plan["schedule"]:
            if job["arm"] in ("D", "L", "F"):
                for rank in range(8):
                    elapsed = 24 if job["arm"] in ("D", "L") else 25.1
                    self.modify(job["directory"], rank, lambda row, v=elapsed: row.update(elapsed_seconds=v))
        result = self.quiet_report()
        self.assertTrue(result["decoder_control_passed"])
        self.assertTrue(result["lifecycle_benefit_validated"])
        self.assertFalse(result["lifecycle_increment_validated"])
        self.assertFalse(result["decoder_increment_validated"])
        self.assertFalse(result["target_validated"])
        self.assertNotIn("arrival_benefit_validated", result)

    def test_offline_report_recomputation_does_not_load_runtime(self):
        self.verify_patch.stop()
        with patch.object(bench, "build_identity", side_effect=AssertionError("must not load runtime")):
            result = full.report(self.root, offline=True, save=False)
        self.assertTrue(result["completed"])

    def test_missing_complete_block_is_rejected_offline(self):
        self.verify_patch.stop()
        self.plan["schedule"].pop()
        (self.root / "plan.json").write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, "complete comparison"):
            full.verify(self.root, offline=True)

    def test_offline_report_does_not_overwrite_disagreeing_saved_evidence(self):
        result = self.quiet_report()
        bench.write(self.root / "report.json", result)
        self.verify_patch.stop()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(full.report(self.root, offline=True, save=True), result)
        result["target_validated"] = not result["target_validated"]
        (self.root / "report.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "Saved report differs"):
            full.report(self.root, offline=True, save=True)


class SourceTests(unittest.TestCase):
    def test_run_preserves_symlinked_virtualenv_python_and_path(self):
        # All execution hooks are mocked; the fake interpreter is never started.
        with tempfile.TemporaryDirectory(prefix="op4-python-cpu-only-") as temporary:
            root = Path(temporary)
            interpreter = root / "venv/bin/python"
            interpreter.parent.mkdir(parents=True)
            target = root / "system-python"
            target.touch()
            interpreter.symlink_to(target)
            (root / "results").mkdir()
            job = dict(arm="A", mode="smoke", trial=0, block=-1, index=-1, directory="smoke-A")
            plan = dict(run_id="synthetic", trial_ids={}, megatron=str(root),
                        libraries={"A": {"root": str(root / "runtime")}},
                        communication={}, communication_record={},
                        shared_environment={"CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_GEMM_RS_GEMM_ONLY": "1"})
            args = types.SimpleNamespace(root=root, python=interpreter, stage="preflight", execute=False,
                                         cuda_home=None, worker_timeout=1)
            with patch.object(full, "verify", return_value=plan), \
                    patch.object(full, "jobs", return_value=[job]), \
                    patch.object(full, "window"), patch.object(full, "preflight"), \
                    patch.object(full.communication, "apply", side_effect=lambda env, _: env), \
                    patch.object(bench, "run_worker") as worker:
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    full.run(args)
                self.assertEqual(json.loads(captured.getvalue())["command"][0], str(interpreter))
                worker.assert_not_called()
                args.execute = True
                with contextlib.redirect_stdout(io.StringIO()):
                    full.run(args)
                command, _, env, _, _ = worker.call_args.args
                self.assertEqual(command[0], str(interpreter))
                self.assertTrue(env["PATH"].startswith(str(interpreter.parent) + ":"))
                self.assertNotEqual(command[0], str(interpreter.resolve()))

    def test_prepare_freezes_96_windows_and_rejects_offline_worker_changes(self):
        with tempfile.TemporaryDirectory(prefix="op4-prepare-cpu-only-") as temporary:
            base = Path(temporary)
            root = base / "experiment"
            args = types.SimpleNamespace(root=root, build=base / "build", megatron=base / "meg",
                                         trials=2, blocks=12)
            identities = {arm: {"root": str(base / arm), "files": {}} for arm in bench.ARMS}
            with patch.object(bench, "build_identity", return_value={"builder_root": "/synthetic"}), \
                    patch.object(bench, "libraries", return_value=identities), \
                    patch.object(full, "check_megatron"), \
                    patch.object(full, "snapshot_megatron", return_value={"patch": {"synthetic": True}}), \
                    patch.object(full.communication, "capture", return_value={"environment": {"NCCL_NVLS_ENABLE": "0"}}), \
                    contextlib.redirect_stdout(io.StringIO()):
                full.prepare(args)
            plan = full.verify(root, offline=True)
            self.assertEqual(len(plan["schedule"]), 96)
            self.assertEqual(len(plan["schedule"]) * plan["timed_steps"], 4800)
            self.assertEqual(len(set(plan["trial_ids"].values())), 2)
            self.assertIn("scripts/fullstep-controller.py", plan["source_files"])
            self.assertIn("scripts/operator-controller.py", plan["source_files"])
            (root / "scripts/worker.py").write_text("raise RuntimeError('synthetic drift')\n")
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                full.verify(root, offline=True)

    def test_generated_sources_parse(self):
        for function in (full.taco_source, full.adapter_source, full.worker_source):
            ast.parse(function())

    def test_actual_adapter_clone_contract(self):
        tree = ast.parse(full.adapter_source())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GemmRSFunction")
        context = dict(torch=types.SimpleNamespace(autograd=types.SimpleNamespace(Function=object)),
                       parallel_state=types.SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 8))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "synthetic-adapter", "exec"), context)
        class Tensor:
            shape = (2048, 1, 512)
            cloned = 0
            def reshape(self, *args): return self
            def contiguous(self): return self
            def clone(self): self.cloned += 1; return self
        for owns in (False, True):
            output, inputs, weight = Tensor(), Tensor(), Tensor()
            weight.shape = (4096, 512)
            op = types.SimpleNamespace(owns_output=owns, forward=lambda *a, **kw: output)
            ctx = types.SimpleNamespace(save_for_backward=lambda *a: None)
            result = context["GemmRSFunction"].forward(ctx, inputs, weight, op, None)
            self.assertIs(result, output)
            self.assertEqual(output.cloned, int(not owns))

    def test_taco_wrapper_rejects_false_lifecycle_ownership(self):
        tree = ast.parse(full.taco_source())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TacoForward")
        import os
        context = dict(os=os, is_native=lambda _: False, is_fused_v2=lambda _: True,
                       dist=types.SimpleNamespace(get_world_size=lambda _: 8))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "synthetic-taco", "exec"), context)
        fake_module = types.SimpleNamespace(GemmRSTaco=lambda *a, **kw: types.SimpleNamespace())
        with patch.dict(os.environ, {"H100_OP4_ARM": "F"}), \
                patch.dict("sys.modules", {"flux.gemm_rs_taco": fake_module}):
            with self.assertRaisesRegex(RuntimeError, "output ownership"):
                context["TacoForward"](None, 2048, 4096, 512, full.POLICY, "attn")

    def test_all_stage_has_no_performance_admission_or_profile_jobs(self):
        plan = dict(schedule=[dict(arm="F", trial=1, block=0, index=0, directory="test")])
        work = list(full.jobs(plan, "all"))
        self.assertEqual([j["mode"] for j in work], ["smoke"] * 4 + ["timing"])
        self.assertEqual([j["arm"] for j in full.jobs(plan, "profile")], list(bench.ARMS))


def synthetic_trace(arm):
    """CPU fixture, never a GPU evidence artifact."""
    events = []
    for role_index, role in enumerate(("attn", "mlp")):
        for index in range(128):
            external = role_index * 128 + index
            start = external * 1000
            args = {"External id": external, "stream": 7}
            events.extend([
                dict(cat="user_annotation", name=f"scope.{role}.candidate", tid=1, ts=start, dur=100),
                dict(cat="cpu_op", name="GemmRSFunction", tid=1, ts=start+1, dur=90, args=args),
                dict(cat="kernel", name="flux_bf16_reducescatter", tid=7, ts=start+2, dur=20, args=args),
                dict(cat="kernel", name="CudaIpcBarrierAllKernel", tid=7, ts=start+23, dur=2, args=args),
                dict(cat="kernel", name="taco_op3_v2_decode" if arm in ("D", "F") else "taco_decode_ring_kernel",
                     tid=7, ts=start+26, dur=8, args=args),
            ])
            if arm in ("A", "D"):
                events.extend([
                    dict(cat="kernel", name="CudaIpcBarrierAllKernel", tid=7, ts=start+35, dur=2, args=args),
                    dict(cat="cpu_op", name="aten::clone", tid=1, ts=start+38, dur=2, args=args),
                ])
    return events


class ProfileTests(unittest.TestCase):
    def test_all_four_paths_have_required_real_model_counts(self):
        for arm in bench.ARMS:
            result = full.trace_projection_records(synthetic_trace(arm), arm)
            for role in bench.ROLES:
                self.assertEqual(result[role]["projection_calls"], 128)
                self.assertEqual(result[role]["barrier_calls"], 128 if arm in ("L", "F") else 256)
                self.assertEqual(result[role]["output_clones"], 0 if arm in ("L", "F") else 128)
                self.assertEqual(result[role]["warp_decoder_calls"], 128 if arm in ("D", "F") else 0)

    def test_lifecycle_with_old_reuse_barrier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "barrier execution"):
            full.trace_projection_records(synthetic_trace("A"), "L")

    def test_extra_output_clone_is_rejected(self):
        events = synthetic_trace("F")
        events.append(dict(cat="cpu_op", name="aten::clone", tid=1, ts=38, dur=2))
        with self.assertRaisesRegex(ValueError, "clone count"):
            full.trace_projection_records(events, "F")

    def test_wrong_decoder_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "scalar/warp"):
            full.trace_projection_records(synthetic_trace("L"), "F")

    def test_missing_projection_cannot_pass(self):
        events = [e for e in synthetic_trace("F") if e["ts"] < 255000]
        with self.assertRaisesRegex(ValueError, "32 layers"):
            full.trace_projection_records(events, "F")

    def test_stream_mismatch_cannot_pass(self):
        events = synthetic_trace("F")
        next(e for e in events if e["name"] == "taco_op3_v2_decode")["args"] = {"External id": 0, "stream": 8}
        with self.assertRaisesRegex(ValueError, "stream/barrier"):
            full.trace_projection_records(events, "F")


if __name__ == "__main__":
    unittest.main()


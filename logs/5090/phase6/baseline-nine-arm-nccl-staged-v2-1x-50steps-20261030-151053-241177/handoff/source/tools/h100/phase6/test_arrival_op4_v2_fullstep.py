"""CPU-only tests; fabricated measurements never leave temporary directories."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import arrival_op4_v2_bench as bench
import arrival_op4_v2_fullstep as full
import arrival_op4_v2_plan as closure


def fixture_plan():
    jobs = [dict(job, trial=trial, round=1 if job["block"] < 8 else 2,
                 directory=f"compare{trial}/{job['directory']}")
            for trial in (1, 2) for job in bench.schedule(16)]
    return dict(protocol=full.PROTOCOL, run_id="synthetic-CPU-only", trial_ids={"1": "trial1", "2": "trial2"},
        trials=2, blocks=16, model=full.MODEL, roles=bench.ROLES, arms=list(bench.ARMS), schedule=jobs,
        closure_plans=closure.protocol_record(), numerical_limits=full.NUMERICAL_LIMITS,
        warmup_steps=10, timed_steps=50, source_files={}, target_percent=1.0,
        preflight_phases=8, preflight_windows=32, preflight_rank_records=256,
        shared_environment={"CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_GEMM_RS_GEMM_ONLY": "1"},
        communication_record={"synthetic": True}, acceptance_rule="Synthetic fixture only",
        megatron_snapshot={"patch": {"synthetic": True}},
        libraries={arm: {"files": {f"/synthetic/{arm}/{name}": arm + name for name in
                   ("libflux_cuda.so", "libflux_cuda_ths_op.so")}} for arm in bench.ARMS})


def fixture_rows(plan, job, plan_hash):
    arm, phase = job["arm"], job["phase"]
    count = 1 if job["mode"] == "profile" else 3 if job["mode"] == "smoke" else 50
    modules = [f"module-{i}" for i in range(64)]
    budget = closure.statistics(arm, phase)
    state = dict(protocol=closure.PROTOCOL, arm=arm, phase=phase, mask_sha256=budget["mask_sha256"],
        mask_kind="closed" if arm[0] == "C" else "dispersed",
        reduction="merged" if arm[1] == "M" else "per_source", selected_per_source=56,
        selected_per_link=8, inverse_h128_per_rank=1024 if arm == "CM" else 7168,
        wire_bytes_per_rank=closure.WIRE_BYTES_PER_RANK)
    rows = []
    for rank in range(8):
        row = dict(completed=True, passed=True, rank=rank, local_rank=rank, protocol=full.PROTOCOL,
            policy=full.POLICY, mode=job["mode"], closure_arm=arm, closure_phase=phase,
            world_size=8, run_id=plan["trial_ids"].get(str(job["trial"]), plan["run_id"]),
            closure_plan_sha256=plan_hash, closure_shared_environment=plan["shared_environment"],
            closure_protocol=closure.PROTOCOL, closure_budget=budget,
            closure_codec_state={name: dict(owns_output=False, owns_configuration=False, workspaces=1)
                                 for name in modules},
            closure_runtime={name: dict(state=state, mask_sha256=budget["mask_sha256"]) for name in modules},
            workspace_checks={name: [dict(shape=[2048, 4096, 4096], packet_ptr=1000 + i)]
                              for i, name in enumerate(modules)},
            communication=plan["communication_record"], sampling=False, profiling=job["mode"] == "profile",
            fallback_calls=0, skips=0, timing_protocol="full-step-one-clear-v2",
            warmup_steps=2 if job["mode"] == "smoke" else 10, losses=[1.] * count, grad_norms=[.1] * count,
            tokens_per_optimizer_step=8192, padded_vocab_size=33792, target_modules=modules,
            replaced_modules=64, independent_scope_instances=64, timed_calls={name: count * 4 for name in modules},
            post_audit_routes={name: "candidate" for name in modules},
            model=dict(name="h100-phase6", **{k: v for k, v in full.MODEL.items() if k != "sm_count"}),
            observed_shapes={f"decoder.layers.{layer}.{suffix}": dict(M=2048, N=4096, K_local=k, K_global=k * 8)
                for layer in range(32) for suffix, k in (("self_attention.linear_proj", 512), ("mlp.linear_fc2", 2048))},
            hardware={"host": "CPU-fixture", "devices": [dict(name="NVIDIA H100 synthetic", uuid=str(i),
                      sm_count=132, capability=[9, 0]) for i in range(8)]},
            torch="synthetic", cuda="synthetic", nccl=[0, 0], python="synthetic",
            initial_parameters_sha256=str(rank), initial_rng_sha256=str(rank), tokens_sha256="synthetic",
            timed_steps=count, elapsed_seconds=(.49 if arm == "CM" else .5) * count,
            forward_checks={name: dict(passed=True, relative_l2=.025, protocol_relative_l2=.0002,
                           budget_vs_bf16=.05, budget_vs_codec=.01) for name in modules}
                           if job["mode"] == "smoke" else {})
        for name in ("libflux_cuda.so", "libflux_cuda_ths_op.so"):
            row[name] = dict(path=f"/synthetic/{arm}/{name}", sha256=arm + name)
        rows.append(row)
    return rows


class NumericalTests(unittest.TestCase):
    def test_cross_mask_difference_is_explicitly_allowed_within_fixed_limit(self):
        rows = {arm: dict(losses=[1., .00001], grad_norms=[.1, .00001]) for arm in bench.ARMS}
        for arm in ("CP", "CM"):
            rows[arm] = dict(losses=[1.01, .00002], grad_norms=[.102, .00002])
        result = full.numerical_comparisons(rows)
        self.assertAlmostEqual(result["cross_mask"]["CM/SP"]["loss"]["max_abs"], .01)

    def test_same_mask_has_stricter_gate_than_cross_mask(self):
        rows = {arm: dict(losses=[1.], grad_norms=[.1]) for arm in bench.ARMS}
        rows["SM"] = dict(losses=[1.01], grad_norms=[.1])
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            full.numerical_comparisons(rows)

    def test_grad_difference_cannot_hide_behind_identical_losses(self):
        rows = {arm: dict(losses=[1.], grad_norms=[.1]) for arm in bench.ARMS}
        for arm in ("CP", "CM"):
            rows[arm] = dict(losses=[1.], grad_norms=[.2])
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            full.numerical_comparisons(rows)

    def test_nonfinite_and_missing_pair_rejected(self):
        rows = {arm: dict(losses=[1.], grad_norms=[.1]) for arm in bench.ARMS}
        rows["CM"]["losses"] = [float("nan")]
        with self.assertRaises(ValueError): full.numerical_comparisons(rows)
        del rows["CM"]
        with self.assertRaises(ValueError): full.numerical_comparisons(rows)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="closure-fullstep-CPU-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = fixture_plan()
        bench.write(self.root / "plan.json", self.plan)
        self.job = dict(self.plan["schedule"][0], mode="timing")
        self.rows = fixture_rows(self.plan, self.job, bench.sha(self.root / "plan.json"))

    def window(self):
        with patch.object(full, "read", side_effect=lambda path: self.rows[int(path.stem[4:])]):
            return full.window(self.root, self.plan, self.job)

    def test_single_workspace_clone_protocol_and_budget_accepted(self):
        self.assertEqual(len(self.window()), 8)

    def test_wrong_phase_or_mask_rejected(self):
        self.rows[0]["closure_phase"] = 5
        with self.assertRaisesRegex(ValueError, "runtime settings"): self.window()
        self.rows[0]["closure_phase"] = self.job["phase"]
        self.rows[0]["closure_runtime"]["module-0"] = dict(state={}, mask_sha256="changed")
        with self.assertRaisesRegex(ValueError, "Actual codec"): self.window()

    def test_aliasing_and_skip_rejected(self):
        self.rows[0]["skips"] = 1
        with self.assertRaisesRegex(ValueError, "skipped updates"): self.window()
        self.rows[0]["skips"] = 0
        self.rows[0]["workspace_checks"]["module-1"][0]["packet_ptr"] = 1000
        with self.assertRaisesRegex(ValueError, "Aliased"): self.window()

    def test_smoke_requires_all_64_independent_projection_checks(self):
        self.job = next(full.jobs(self.plan, "preflight"))
        self.rows = fixture_rows(self.plan, self.job, bench.sha(self.root / "plan.json"))
        self.assertEqual(len(self.window()[0]["forward_checks"]), 64)
        self.rows[2]["forward_checks"].pop("module-63")
        with self.assertRaisesRegex(ValueError, "Numerical codec"):
            self.window()


def synthetic_trace(arm):
    events = []
    for role_index, role in enumerate(("attn", "mlp")):
        for index in range(128):
            external, start = role_index * 128 + index, (role_index * 128 + index) * 1000
            args = {"External id": external, "stream": 7}
            events.extend([
                dict(cat="user_annotation", name=f"scope.{role}.candidate", tid=1, ts=start, dur=100),
                dict(cat="cpu_op", name="GemmRSFunction", tid=1, ts=start + 1, dur=90, args=args),
                dict(cat="kernel", name="flux_bf16_reducescatter", tid=7, ts=start + 2, dur=20, args=args),
                dict(cat="kernel", name="CudaIpcBarrierAllKernel", tid=7, ts=start + 23, dur=2, args=args),
                dict(cat="kernel", name="taco_closure_decode_merged" if arm[1] == "M" else "taco_closure_decode_per_source",
                     tid=7, ts=start + 26, dur=8, args=args),
                dict(cat="kernel", name="CudaIpcBarrierAllKernel", tid=7, ts=start + 35, dur=2, args=args),
                dict(cat="cpu_op", name="aten::clone", tid=1, ts=start + 38, dur=2, args=args)])
    return events


class ProfileTests(unittest.TestCase):
    def test_every_arm_requires_256_projections_with_two_barriers_and_clone(self):
        for arm in bench.ARMS:
            result = full.trace_projection_records(synthetic_trace(arm), arm)
            for role in bench.ROLES:
                self.assertEqual(result[role]["projection_calls"], 128)
                self.assertEqual(result[role]["barrier_calls"], 256)
                self.assertEqual(result[role]["output_clones"], 128)
                self.assertEqual(result[role]["merged_decoder_calls"], 128 if arm[1] == "M" else 0)

    def test_wrong_decoder_and_missing_reuse_barrier_rejected(self):
        with self.assertRaisesRegex(ValueError, "separate/merged"):
            full.trace_projection_records(synthetic_trace("SP"), "SM")
        events = synthetic_trace("CM")
        events.pop(5)
        with self.assertRaisesRegex(ValueError, "barrier execution"):
            full.trace_projection_records(events, "CM")


class SourceTests(unittest.TestCase):
    def test_sources_parse_and_override_reference(self):
        for function in (full.taco_source, full.adapter_source, full.worker_source):
            ast.parse(function())
        self.assertIn("from closure_reference import check_taco_output", full.taco_source())
        self.assertIn("op.codec.closure_state()", full.worker_source())
        self.assertIn("getattr(op.codec,'selection_cpu',None)", full.adapter_source())

    def test_prepare_freezes_phase_balanced_128_windows_and_limits(self):
        with tempfile.TemporaryDirectory(prefix="closure-prepare-CPU-") as temporary:
            root = Path(temporary) / "experiment"
            args = types.SimpleNamespace(root=root, build=Path(temporary) / "build",
                                         megatron=Path(temporary) / "megatron", trials=2, blocks=16)
            identities = {arm: {"root": str(Path(temporary) / arm), "files": {}} for arm in bench.ARMS}
            with patch.object(bench, "build_identity", return_value={"builder_root": "/synthetic"}), \
                    patch.object(bench, "libraries", return_value=identities), \
                    patch.object(full, "check_megatron"), \
                    patch.object(full, "snapshot_megatron", return_value={"patch": {"synthetic": True}}), \
                    patch.object(full.communication, "capture", return_value={"environment": {"NCCL_NVLS_ENABLE": "0"}}), \
                    contextlib.redirect_stdout(io.StringIO()):
                full.prepare(args)
            plan = full.verify(root, offline=True)
            self.assertEqual(len(plan["schedule"]), 128)
            self.assertEqual(len(plan["schedule"]) * plan["timed_steps"], 6400)
            self.assertEqual(plan["preflight_phases"], 8)
            self.assertEqual(plan["preflight_rank_records"], 256)
            self.assertEqual(plan["numerical_limits"], full.NUMERICAL_LIMITS)
            for name in ("closure_reference.py", "closure_plan.py", "reference.py"):
                self.assertIn("scripts/" + name, plan["source_files"])
            reference = (root / "scripts/closure_reference.py").read_text()
            self.assertIn("dtype=torch.float64", reference)
            self.assertIn("from reference import encode", reference)
            self.assertIn("supplied != plan.mask(arm, phase)", reference)
            for phase in range(8):
                self.assertEqual(sum(j["phase"] == phase for j in plan["schedule"]), 16)
            plan["numerical_limits"]["same_mask"]["loss"]["rtol"] = .1
            (root / "plan.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "numerical limits"):
                full.verify(root, offline=True)

    def test_smoke_covers_all_phases_while_profile_is_phase_zero(self):
        plan = fixture_plan()
        smoke = list(full.jobs(plan, "preflight"))
        self.assertEqual(len(smoke), 32)
        self.assertEqual(len({(job["phase"], job["arm"]) for job in smoke}), 32)
        self.assertEqual(smoke[0]["directory"], "smoke-SP")
        self.assertEqual(smoke[4]["directory"], "smoke-p1-SP")
        profile = list(full.jobs(plan, "profile"))
        self.assertEqual(len(profile), 4)
        self.assertEqual({job["phase"] for job in profile}, {0})


class PreflightTests(unittest.TestCase):
    def test_numerical_checks_are_separate_for_all_eight_phases(self):
        with tempfile.TemporaryDirectory(prefix="closure-phases-CPU-") as temporary:
            root, plan = Path(temporary), fixture_plan()
            bench.write(root / "plan.json", plan)
            def window(_, __, job):
                # Different phase trajectories are legitimate; paired arms in
                # the same phase remain within their fixed numerical bounds.
                return [dict(losses=[1. + job["phase"]], grad_norms=[.1])] * 8
            with patch.object(full, "verify", return_value=plan), \
                    patch.object(full, "window", side_effect=window), \
                    patch.object(full, "identity", return_value=[]):
                result = full.preflight(root)
            self.assertEqual(set(result["numerical_checks"]), {str(i) for i in range(8)})
            self.assertEqual(result["preflight_rank_records"], 256)

    def test_omitted_phase_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory(prefix="closure-missing-phase-CPU-") as temporary:
            root, plan = Path(temporary), fixture_plan()
            bench.write(root / "plan.json", plan)
            work = [job for job in full.jobs(plan, "preflight") if job["phase"] != 6]
            with patch.object(full, "verify", return_value=plan), \
                    patch.object(full, "jobs", return_value=work), \
                    patch.object(full, "window", return_value=[dict(losses=[1.], grad_norms=[.1])] * 8), \
                    patch.object(full, "identity", return_value=[]):
                with self.assertRaisesRegex(ValueError, "missing a phase"):
                    full.preflight(root)


class ReportTests(unittest.TestCase):
    def test_independent_main_target_and_controls_cannot_replace_it(self):
        with tempfile.TemporaryDirectory(prefix="closure-report-CPU-") as temporary:
            root, plan = Path(temporary), fixture_plan()
            gate = dict(rank_identities=[], files={})
            def rows(_, __, job):
                arm = job["arm"]
                # Trial 1 reaches 2%; trial 2 is positive but below 1%.
                time = .49 if arm == "CM" and job["trial"] == 1 else .497 if arm == "CM" else .5
                return [dict(elapsed_seconds=time * 50, losses=[1.] * 50, grad_norms=[.1] * 50)] * 8
            def paired(a, b):
                percent = 100 * (1 - a[0] / b[0])
                return dict(percent=percent, ci95=[percent, percent])
            with patch.object(full, "verify", return_value=plan), patch.object(full, "preflight", return_value=gate), \
                    patch.object(full, "window", side_effect=rows), patch.object(full, "identity", return_value=[]), \
                    patch.object(bench, "paired", side_effect=paired):
                result = full.report(root, save=False)
            self.assertFalse(result["target_validated"])
            self.assertTrue(result["mechanism_validated"])
            self.assertTrue(result["comparisons"]["compare1"]["target_validated"])
            self.assertFalse(result["comparisons"]["compare2"]["target_validated"])
            self.assertTrue(result["precision_control_passed"])
            self.assertTrue(result["budget_matched"])
            self.assertEqual(result["formal_optimizer_steps"], 6400)
            self.assertEqual(result["timing_rank_records"], 1024)
            self.assertEqual(len(result["numerical_checks"]), 32)


if __name__ == "__main__":
    unittest.main()

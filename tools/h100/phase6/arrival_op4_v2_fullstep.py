#!/usr/bin/env python3
"""Fresh-host scattered/closed and separate/merged inverse GPT6.7B full optimizer-step confirmation.

The proven ablation4 model worker and adapter are reused with the same A policy
and codec. The separately recorded arm selects an isolated runtime and mode.
No arrival calibration is used; masks rotate across eight frozen phases.
"""
import argparse
import hashlib
import gzip
from bisect import bisect_right
from collections import defaultdict
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import uuid

import ablation4_runtime as communication
import ablation4_sources as sources
import arrival_op4_v2_bench as bench
import arrival_op4_v2_plan as closure
from common import MEGATRON_COMMIT, REPO, replace_once
from run import arguments as model_arguments

PROTOCOL = "h100-reduction-closure-v1-fullstep"
POLICY = "interleaved_all_attn_mlp"
MODEL = dict(layers=32, hidden=4096, ffn=16384, heads=32, sequence=2048,
             micro_batch=1, global_batch=4, tp=8, sm_count=132,
             seed=1234, token_seed=4321, vocab=32768)
read, write, sha, require = bench.read, bench.write, bench.sha, bench.require
NUMERICAL_LIMITS = dict(
    cross_mask=dict(loss=dict(atol=.01, rtol=.02), grad=dict(atol=.001, rtol=.05)),
    same_mask=dict(loss=dict(atol=.0001, rtol=.001), grad=dict(atol=.0001, rtol=.01)),
    operator=dict(budget_vs_bf16=.05, budget_vs_codec=.01))


def taco_source():
    """Every arm keeps the original single workspace and output clone."""
    return sources.taco_source() + "\nfrom closure_reference import check_taco_output\n"


def adapter_source():
    return sources.adapter_source()


def worker_source():
    worker = replace_once(sources.worker_source(), "report['protocol'] = PLAN['protocol']", """report['closure_arm'] = os.environ['H100_CLOSURE_ARM']
report['closure_protocol'] = os.environ['H100_CLOSURE_PROTOCOL']
report['closure_phase'] = int(os.environ['H100_CLOSURE_PHASE'])
from closure_plan import statistics as closure_statistics
report['closure_budget'] = closure_statistics(report['closure_arm'], report['closure_phase'])
report['closure_plan_sha256'] = hashlib.sha256((HERE.parent/'plan.json').read_bytes()).hexdigest()
report['closure_shared_environment'] = {key:os.environ[key] for key in
    ('CUDA_DEVICE_MAX_CONNECTIONS','FLUX_GEMM_RS_GEMM_ONLY')}
report['protocol'] = PLAN['protocol']""")
    return replace_once(worker, "report['workspace_checks'] = adapter.workspace_checks", """report['workspace_checks'] = adapter.workspace_checks
report['closure_codec_state'] = {name: dict(
    owns_output=bool(op.owns_output),
    owns_configuration=bool(getattr(op.codec, 'owns_configuration', False)),
    workspaces=len(op.workspace_info)) for name,op in adapter.scope_ops.items()}
report['closure_runtime'] = {name: dict(state=op.codec.closure_state(),
    mask_sha256=op.codec.selection_sha256) for name,op in adapter.scope_ops.items()}""")


def check_megatron(path):
    require(subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            == MEGATRON_COMMIT, "Megatron must use the registered core_v0.12.3 commit")
    require(not subprocess.check_output(["git", "-C", str(path), "status", "--porcelain",
                                        "--untracked-files=no"], text=True).strip(),
            "Megatron tracked sources differ from the pinned commit")


MEGATRON_GUARD = '''            assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1", \\
                "Using tensor model parallelism or context parallelism require setting the environment variable " \\
                "CUDA_DEVICE_MAX_CONNECTIONS to 1"'''
MEGATRON_OP4_GUARD = '''            assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "8", \\
                "This isolated op4 SP/CP/SM/CM experiment requires the environment variable " \\
                "CUDA_DEVICE_MAX_CONNECTIONS to 8"'''


def snapshot_megatron(source, target):
    """Archive pinned tracked files and patch just the checked scheduling guard."""
    archive = subprocess.check_output(["git", "-C", str(source), "archive", "--format=tar", MEGATRON_COMMIT])
    target.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        # Pinned git files are trusted inputs; still prevent path traversal or a
        # symlink escaping the independent snapshot during archive extraction.
        members = handle.getmembers()
        for member in members:
            path = Path(member.name)
            require(not path.is_absolute() and ".." not in path.parts and
                    (member.isdir() or member.isfile()), "Unexpected archive entry: " + member.name)
        handle.extractall(target, members=members)
    original = {str(p.relative_to(target)): sha(p) for p in target.rglob("*") if p.is_file()}
    relative = "megatron/training/arguments.py"
    path = target / relative
    before = path.read_text()
    after = replace_once(before, MEGATRON_GUARD, MEGATRON_OP4_GUARD)
    path.write_text(after)
    patched = dict(original)
    patched[relative] = sha(path)
    require(original[relative] != patched[relative], "Megatron scheduling guard was not patched")
    return dict(source_root=str(source), commit=MEGATRON_COMMIT,
                archive_sha256=hashlib.sha256(archive).hexdigest(),
                snapshot_root=str(target), original_files=original, files=patched,
                changed_files=[relative],
                patch=dict(file=relative, original_sha256=original[relative],
                           patched_sha256=patched[relative], before=MEGATRON_GUARD, after=MEGATRON_OP4_GUARD),
                reason="All SP/CP/SM/CM arms share eight CUDA connections for a fixed common scheduling configuration. This changes Megatron's scheduling-performance guard, not tensor arithmetic.")


def verify_megatron_snapshot(record):
    require(record["commit"] == MEGATRON_COMMIT and
            record["changed_files"] == ["megatron/training/arguments.py"], "Wrong Megatron patch provenance")
    root = Path(record["snapshot_root"])
    for relative, digest in record["files"].items():
        require(sha(root / relative) == digest, "Megatron snapshot source changed: " + relative)
    patch_record = record["patch"]
    require(patch_record["before"] == MEGATRON_GUARD and patch_record["after"] == MEGATRON_OP4_GUARD,
            "Megatron guard patch changed")
    restored = replace_once((root / patch_record["file"]).read_text(), MEGATRON_OP4_GUARD, MEGATRON_GUARD)
    require(hashlib.sha256(restored.encode()).hexdigest() == patch_record["original_sha256"],
            "Megatron snapshot contains changes beyond the registered guard")
    differences = [name for name, digest in record["files"].items()
                   if record["original_files"].get(name) != digest]
    require(differences == record["changed_files"], "More than the guard was changed")


def prepare(args):
    root, build, megatron = (p.resolve() for p in (args.root, args.build, args.megatron))
    provenance = bench.build_identity(build)
    identities = bench.libraries(build)
    check_megatron(megatron)
    require(args.trials >= 2, "Two independent full-step comparisons are required")
    require(args.blocks >= 16 and args.blocks % 16 == 0, "Complete balanced blocks must be a positive multiple of sixteen")
    root.mkdir(parents=True, exist_ok=False)
    scripts = root / "scripts"
    scripts.mkdir()
    (root / "results").mkdir()
    megatron_snapshot = snapshot_megatron(megatron, root / "megatron")
    comm = communication.capture("0")
    config = read(REPO / "tools/a800/phase5/e2e/config.json")
    config.update(protocol=PROTOCOL, policies=[POLICY], orders=[], repetitions=2,
                  warmup_steps=10, timed_steps=50, smoke_warmup_steps=2, smoke_steps=3,
                  graph_forward=False, double_buffered=False, ring_reduction=True,
                  cases=[dict(name="h100-phase6", **{k: MODEL[k] for k in ("hidden", "ffn", "heads", "sequence", "tp")})],
                  common={k: MODEL[k] for k in ("layers", "micro_batch", "global_batch", "seed", "token_seed", "vocab")},
                  role_plan_sha256={POLICY: {"attn": None, "mlp": None}}, communication=comm,
                  scope="New same-host op4 SP/CP/SM/CM comparison; full GPT6.7B optimizer steps; fixed-budget closure masks and FP32 reduction; synthetic input.")
    write(scripts / "config.json", config)
    (scripts / "adapter.py").write_text(adapter_source())
    (scripts / "taco_support.py").write_text(taco_source())
    (scripts / "worker.py").write_text(worker_source())
    native = (REPO / "tools/a800/phase5/e2e/megatron_taco.py").read_text()
    (scripts / "megatron_taco.py").write_text(replace_once(native, "self.world==4", "self.world in (4,8)"))
    for source, name in ((Path(__file__).with_name("ablation4_protocol.py"), "routing.py"),
                         (Path(__file__).with_name("runtime.py"), "runtime.py"),
                         (Path(__file__).with_name("ablation4_runtime.py"), "communication.py"),
                         (Path(closure.__file__), "closure_plan.py"),
                         (Path(__file__).with_name("arrival_op4_v2_reference.py"), "closure_reference.py"),
                         (REPO / "tools/a800/phase4/reference.py", "reference.py")):
        shutil.copy2(source, scripts / name)
    shutil.copy2(__file__, scripts / "fullstep-controller.py")
    shutil.copy2(bench.__file__, scripts / "operator-controller.py")
    write(scripts / "selection-plan.json", dict(protocol=PROTOCOL, calibration_used=False,
                                               reason="Fixed Latin/closed masks and FP32 reduction; no arrival calibration."))
    for arm in bench.ARMS:
        parent = root / "build" / arm
        parent.mkdir(parents=True)
        parent.joinpath(POLICY).symlink_to(identities[arm]["root"], target_is_directory=True)
    jobs = []
    for trial in range(args.trials):
        for job in bench.schedule(args.blocks):
            jobs.append(dict(job, phase=job["block"] % 8, trial=trial + 1,
                             round=1 if job["block"] < args.blocks // 2 else 2,
                             directory=f"compare{trial+1}/{job['directory']}"))
    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,
                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},
                trials=args.trials, blocks=args.blocks, model=MODEL,
                kind="full_optimizer_steps", warmup_steps=10, timed_steps=50,
                preflight_phases=8, preflight_windows=32, preflight_rank_records=256,
                build=str(build), libraries=identities, roles=bench.ROLES, arms=list(bench.ARMS),
                build_identity=provenance,
                closure_plans=closure.protocol_record(), numerical_limits=NUMERICAL_LIMITS,
                communication=comm["environment"],
                communication_record=comm,
                shared_environment={"CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_GEMM_RS_GEMM_ONLY": "1"},
                megatron=str(root / "megatron"), megatron_commit=MEGATRON_COMMIT,
                megatron_source=str(megatron), megatron_snapshot=megatron_snapshot,
                schedule=jobs, source_files={str(p.relative_to(root)): sha(p) for p in scripts.iterdir()},
                target_percent=1.0, target_validated=False,
                acceptance_rule="Each independent comparison: CM/SM point and paired complete-block CI95 lower bound >= 1%; fixed numerical and budget checks pass.",
                controller_sha256=sha(Path(__file__)),
                historical_hardware_reused=False, historical_timings_used=False)
    write(root / "plan.json", plan)
    print(root)


def verify(root, offline=False):
    plan = read(root / "plan.json")
    require(plan["protocol"] == PROTOCOL and plan["model"] == MODEL and plan["arms"] == list(bench.ARMS),
            "Wrong full-step plan")
    require(plan["trials"] >= 2 and len(set(plan["trial_ids"].values())) == plan["trials"],
            "Missing independent comparison identities")
    require(plan["warmup_steps"] == 10 and plan["timed_steps"] == 50, "Full-step timing changed")
    require(plan["preflight_phases"] == 8 and plan["preflight_windows"] == 32 and
            plan["preflight_rank_records"] == 256, "Full-model preflight must cover all eight phases")
    require(plan["roles"] == bench.ROLES and plan["blocks"] >= 16 and plan["blocks"] % 16 == 0,
            "Invalid role shapes or comparison blocks")
    expected_schedule = [dict(job, phase=job["block"] % 8, trial=trial + 1,
        round=1 if job["block"] < plan["blocks"] // 2 else 2,
        directory=f"compare{trial+1}/{job['directory']}")
        for trial in range(plan["trials"]) for job in bench.schedule(plan["blocks"])]
    require(plan["schedule"] == expected_schedule and set(plan["libraries"]) == set(bench.ARMS),
            "Missing or changed complete comparison blocks/libraries")
    require(plan["closure_plans"] == closure.protocol_record() and
            plan["numerical_limits"] == NUMERICAL_LIMITS,
            "Frozen mask budgets or numerical limits changed")
    require(plan["shared_environment"] == {"CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_GEMM_RS_GEMM_ONLY": "1"}
            and plan["target_percent"] == 1.0, "Full-step environment or target changed")
    for name, digest in plan["source_files"].items():
        require(sha(root / name) == digest, f"Frozen source changed: {name}")
    if not offline:
        require(sha(Path(__file__)) == plan["controller_sha256"], "Full-step controller source changed")
        require(sha(Path(bench.__file__)) == plan["source_files"]["scripts/operator-controller.py"],
                "Operator controller source changed")
        require(bench.build_identity(Path(plan["build"]), plan["build_identity"]["builder_root"])
                == plan["build_identity"], "Frozen build or frontend changed")
        require(bench.libraries(Path(plan["build"])) == plan["libraries"], "Frozen runtime changed")
        verify_megatron_snapshot(plan["megatron_snapshot"])
    return plan


def trace_projection_records(events, arm):
    """Match real-model CPU projections to kernels and CPU output clones."""
    scopes, kernels, cpu_ops = defaultdict(list), defaultdict(list), defaultdict(list)
    calls = []
    for event in events:
        category, name = event.get("cat"), event.get("name", "")
        if category == "user_annotation" and name in ("scope.attn.candidate", "scope.mlp.candidate"):
            scopes[event["tid"]].append(event)
        elif category == "kernel":
            kernels[event.get("args", {}).get("External id")].append(event)
        elif category == "cpu_op":
            cpu_ops[event["tid"]].append(event)
            if name == "GemmRSFunction":
                calls.append(event)
    for group in scopes.values():
        group.sort(key=lambda event: event["ts"])
    starts = {tid: [e["ts"] for e in group] for tid, group in scopes.items()}
    records, used = defaultdict(list), set()
    for call in calls:
        tid = call["tid"]
        index = bisect_right(starts.get(tid, []), call["ts"]) - 1
        require(index >= 0, "Projection call has no real-model scope")
        scope = scopes[tid][index]
        require(call["ts"] + call["dur"] <= scope["ts"] + scope["dur"] + .01 and
                (tid, index) not in used, "Invalid or repeated projection CPU association")
        used.add((tid, index))
        group = sorted(kernels[call["args"]["External id"]], key=lambda e: e["ts"])
        gemms = [e for e in group if "flux_bf16" in e["name"] and "reducescatter" in e["name"]]
        decoders = [e for e in group if "taco_closure_decode_" in e["name"]]
        barriers = [e for e in group if "CudaIpcBarrierAllKernel" in e["name"]]
        require(len(gemms) == len(decoders) == 1 and len(barriers) == 2,
                "Wrong per-projection production/decode/barrier execution")
        fast = "taco_closure_decode_merged" in decoders[0]["name"]
        expected_decoder = "taco_closure_decode_merged" if arm[1] == "M" else "taco_closure_decode_per_source"
        require(expected_decoder in decoders[0]["name"], "Wrong separate/merged decoder execution")
        require(not any("taco_op3_v2_try" in e["name"] or "taco_op3_v2_stop" in e["name"]
                        for e in group), "Unexpected arrival-consumer work")
        ordered = [gemms[0], barriers[0], decoders[0], barriers[1]]
        require(len({e["args"]["stream"] for e in ordered}) == 1 and all(
            left["ts"] + left["dur"] <= right["ts"] + .01 for left, right in zip(ordered, ordered[1:])),
            "Projection kernels do not follow the registered stream/barrier sequence")
        clones = [e for e in cpu_ops[tid] if e["name"] == "aten::clone" and
                  call["ts"] <= e["ts"] and e["ts"] + e["dur"] <= call["ts"] + call["dur"] + .01]
        require(len(clones) == 1, "Wrong per-projection output clone count")
        records[scope["name"].split(".")[1]].append(dict(
            barriers=len(barriers), output_clones=len(clones), merged_decoder=fast,
            gemm_us=gemms[0]["dur"], decode_us=decoders[0]["dur"]))
    require(len(used) == sum(map(len, scopes.values())) and set(records) == set(bench.ROLES),
            "Missing real-model projection associations")
    require(all(len(records[role]) == 128 for role in bench.ROLES),
            "Profile must contain 32 layers x four microbatches for each role")
    return {role: dict(projection_calls=len(rows), barrier_calls=sum(r["barriers"] for r in rows),
                      output_clones=sum(r["output_clones"] for r in rows),
                      merged_decoder_calls=sum(r["merged_decoder"] for r in rows),
                      gemm_median_us=statistics.median(r["gemm_us"] for r in rows),
                      decode_median_us=statistics.median(r["decode_us"] for r in rows))
            for role, rows in records.items()}


def profile_summary(root, offline=False):
    """Read-only recheck; the run stage is responsible for writing the summary."""
    root = Path(root).resolve()
    plan = verify(root, offline)
    gate = preflight(root, offline)
    profiles, evidence = {}, {}
    for job in jobs(plan, "profile"):
        rows = window(root, plan, job)
        require(identity(rows) == gate["rank_identities"], "Profile hardware/input/software changed")
        for rank in range(8):
            directory = root / "results" / job["directory"]
            path = directory / f"rank{rank}-profile.json.gz"
            with gzip.open(path, "rt") as handle:
                profiles[f"{job['arm']}/rank{rank}"] = trace_projection_records(
                    json.load(handle)["traceEvents"], job["arm"])
            for item in (path, directory / f"rank{rank}.json"):
                evidence[str(item.relative_to(root))] = sha(item)
    return dict(protocol=PROTOCOL, run_id=plan["run_id"], completed=True, execution_validated=True,
                plan_sha256=sha(root / "plan.json"), profiles=profiles, evidence_files=evidence,
                limitation="Independent real-model profile verifies execution counts only. Diagnostic durations "
                           "include profiler perturbation and do not estimate unprofiled step speedup.")


def jobs(plan, stage):
    if stage in ("all", "preflight", "profile"):
        modes = ("smoke",) if stage != "profile" else ("profile",)
        for mode in modes:
            for phase in (range(8) if mode == "smoke" else (0,)):
                for arm in bench.ARMS:
                    directory = f"{mode}-{arm}" if phase == 0 else f"{mode}-p{phase}-{arm}"
                    yield dict(arm=arm, phase=phase, mode=mode, trial=0, block=-1, index=-1,
                               directory=directory)
    if stage in ("all", "timing"):
        for job in plan["schedule"]:
            yield dict(job, mode="timing")


def run(args):
    root = args.root.resolve()
    plan = verify(root)
    work = list(jobs(plan, args.stage))
    command = [str(args.python.expanduser().absolute()), "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=8", str(root / "scripts/worker.py"), *model_arguments(MODEL)]
    if not args.execute:
        print(json.dumps(dict(dry_run=True, command=command, jobs=work), indent=2))
        return
    write(root / f"{args.stage}-started.json", dict(run_id=plan["run_id"], jobs=work))
    if args.stage == "timing":
        preflight(root)
    gated = args.stage == "timing"
    for job in work:
        if job["mode"] == "timing" and not gated:
            preflight(root)
            gated = True
        verify(root)
        arm = job["arm"]
        env = bench.environment(root, plan, arm, args.python.expanduser().absolute(), args.cuda_home,
                                profile=False, phase=job["phase"])
        env = communication.apply(env, plan["communication_record"])
        library = Path(plan["libraries"][arm]["root"])
        dest = root / "results" / job["directory"]
        dest.mkdir(parents=True, exist_ok=False)
        env.update(H100_ABLATION4_EXECUTE="1", H100_TP="8", H100_SM_COUNT="132",
                   H100_RUN_ID=plan["trial_ids"].get(str(job["trial"]), plan["run_id"]),
                   E2E_POLICY=POLICY, E2E_MODE=job["mode"], E2E_CASE="h100-phase6",
                   E2E_BUILD=str(root / "build" / arm), E2E_PROCESS_OUT=str(dest),
                   E2E_BLOCK=str(job["block"]), E2E_WINDOW=str(job["index"]), H100_CLOSURE_ARM=arm,
                   H100_CLOSURE_PHASE=str(job["phase"]),
                   PYTHONPATH=f"{root}/scripts:{plan['megatron']}:{library}/python")
        write(dest / "command.json", dict(command=command, job=job,
            shared_environment=plan["shared_environment"], protocol=env["H100_CLOSURE_PROTOCOL"],
            arm=env["H100_CLOSURE_ARM"], phase=job["phase"], profiling=job["mode"] == "profile"))
        print("START", job["directory"], flush=True)
        try:
            with (dest / "process.log").open("x") as log:
                bench.run_worker(command, plan["megatron"], env, log, args.worker_timeout)
        except BaseException as error:
            write(dest / "failed.json", dict(status="failed", retry=False,
                exception=type(error).__name__, message=str(error), job=job))
            raise
        window(root, plan, job)
    if args.stage == "preflight":
        preflight(root)
    elif args.stage == "profile":
        write(root / "profile-summary.json", profile_summary(root))
    else:
        report(root)


def window(root, plan, job):
    rows = [read(root / "results" / job["directory"] / f"rank{r}.json") for r in range(8)]
    count = 1 if job["mode"] == "profile" else 3 if job["mode"] == "smoke" else 50
    warmup = 2 if job["mode"] == "smoke" else 10
    for rank, row in enumerate(rows):
        require(row["completed"] and row["passed"] and row["rank"] == rank and row["local_rank"] == rank
                and row["protocol"] == PROTOCOL and row["policy"] == POLICY and row["mode"] == job["mode"]
                and row["closure_arm"] == job["arm"] and row["world_size"] == 8
                and row["run_id"] == plan["trial_ids"].get(str(job["trial"]), plan["run_id"])
                and row["closure_plan_sha256"] == sha(root / "plan.json"), "Wrong or incomplete full-step rank record")
        require(row["closure_shared_environment"] == plan["shared_environment"] and
                row["closure_protocol"] == closure.PROTOCOL and row["closure_phase"] == job["phase"],
                "Full-step runtime settings changed")
        require(row["closure_budget"] == plan["closure_plans"]["plans"][str(job["phase"])][job["arm"]],
                "Runtime mask or budget identity changed")
        states = row["closure_codec_state"]
        require(len(states) == 64 and set(states) == set(row["target_modules"]) and all(
            v == dict(owns_output=False, owns_configuration=False, workspaces=1) for v in states.values()),
            "Wrong runtime output ownership or workspace count")
        runtime = row["closure_runtime"]
        expected_state = dict(protocol=closure.PROTOCOL, arm=job["arm"], phase=job["phase"],
            mask_sha256=row["closure_budget"]["mask_sha256"], selected_per_source=56, selected_per_link=8,
            mask_kind="closed" if job["arm"][0] == "C" else "dispersed",
            reduction="merged" if job["arm"][1] == "M" else "per_source",
            inverse_h128_per_rank=1024 if job["arm"] == "CM" else 7168,
            wire_bytes_per_rank=closure.WIRE_BYTES_PER_RANK)
        require(set(runtime) == set(states) and all(
            value == dict(state=expected_state, mask_sha256=row["closure_budget"]["mask_sha256"])
            for value in runtime.values()), "Actual codec mask/state does not match frozen budget")
        workspace = row["workspace_checks"]
        require(set(workspace) == set(states) and all(len(slots) == 1
                for slots in workspace.values()), "Missing actual runtime workspaces")
        pointers = [slot["packet_ptr"] for slots in workspace.values() for slot in slots]
        require(all(isinstance(p, int) and p > 0 for p in pointers) and len(set(pointers)) == len(pointers),
                "Aliased communication workspaces")
        require(row["communication"] == plan["communication_record"] and not row["sampling"] and
                row["profiling"] == (job["mode"] == "profile") and row["fallback_calls"] == row["skips"] == 0,
                "Sampling, fallback, communication drift or skipped updates")
        require(row["timing_protocol"] == "full-step-one-clear-v2" and row["warmup_steps"] == warmup and
                len(row["losses"]) == len(row["grad_norms"]) == count and
                all(math.isfinite(v) for v in row["losses"] + row["grad_norms"]),
                "Invalid full-step trajectory")
        require(row["tokens_per_optimizer_step"] == 8192 and row["padded_vocab_size"] == 33792 and
                len(row["target_modules"]) == row["replaced_modules"] == row["independent_scope_instances"] == 64
                and len(row["timed_calls"]) == 64 and all(v == count * 4 for v in row["timed_calls"].values())
                and set(row["post_audit_routes"].values()) == {"candidate"},
                "Incomplete dual-role 32-layer model execution")
        require(row["model"] == dict(name="h100-phase6", **{k: v for k, v in MODEL.items() if k != "sm_count"}),
                "Model dimensions or seeds differ from GPT6.7B")
        shapes = row["observed_shapes"]
        for role, suffix, k_local in (("attn", ".self_attention.linear_proj", 512),
                                      ("mlp", ".mlp.linear_fc2", 2048)):
            selected = [value for name, value in shapes.items() if name.endswith(suffix)]
            require(len(selected) == 32 and all(
                [v["M"], v["N"], v["K_local"], v["K_global"]] == [2048, 4096, k_local, k_local * 8]
                for v in selected), "Wrong measured role/full shape: " + role)
        devices = row["hardware"]["devices"]
        require(len(devices) == 8 and len({d["uuid"] for d in devices}) == 8 and
                all("H100" in d["name"] and d["sm_count"] == 132 and d["capability"] == [9, 0] for d in devices),
                "Not eight full H100s")
        expected_libs = plan["libraries"][job["arm"]]["files"]
        for name in ("libflux_cuda.so", "libflux_cuda_ths_op.so"):
            entry = row[name]
            require(expected_libs.get(entry["path"]) == entry["sha256"], "Loaded runtime differs from frozen library")
        if job["mode"] == "smoke":
            require(len(row["forward_checks"]) == 64 and all(
                c["passed"] and math.isfinite(c["relative_l2"]) and math.isfinite(c["protocol_relative_l2"])
                and c["relative_l2"] <= .05 and c["protocol_relative_l2"] <= .01
                and c["budget_vs_bf16"] == .05 and c["budget_vs_codec"] == .01
                for c in row["forward_checks"].values()), "Numerical codec validation failed")
        if job["mode"] != "profile":
            require(row["timed_steps"] == count and math.isfinite(row["elapsed_seconds"]) and
                    row["elapsed_seconds"] > 0, "Invalid elapsed time")
    require(all(r["losses"] == rows[0]["losses"] for r in rows), "Ranks disagree on loss trajectory")
    return rows


def identity(rows):
    return [dict(hardware={k: r["hardware"][k] for k in ("host", "devices")},
                 torch=r["torch"], cuda=r["cuda"], nccl=r["nccl"], python=r["python"],
                 initial_parameters_sha256=r["initial_parameters_sha256"],
                 initial_rng_sha256=r["initial_rng_sha256"], tokens_sha256=r["tokens_sha256"])
            for r in rows]


def trajectory_difference(left, right, kind):
    """Fixed symmetric tolerances; preserve each observed delta and its limit."""
    result = {}
    for key, field in (("loss", "losses"), ("grad", "grad_norms")):
        a, b = left[field], right[field]
        require(len(a) == len(b) and len(a) > 0, "Incomplete paired numerical trajectory")
        limits = NUMERICAL_LIMITS[kind][key]
        errors = [abs(x - y) for x, y in zip(a, b)]
        bounds = [limits["atol"] + limits["rtol"] * max(abs(x), abs(y)) for x, y in zip(a, b)]
        passed = all(math.isfinite(e) and math.isfinite(t) and e <= t for e, t in zip(errors, bounds))
        result[key] = dict(passed=passed, max_abs=max(errors), absolute_differences=errors,
                           allowed_differences=bounds, **limits)
    result["passed"] = all(result[key]["passed"] for key in ("loss", "grad"))
    return result


def numerical_comparisons(rows_by_arm):
    require(set(rows_by_arm) == set(bench.ARMS), "Numerical check requires a complete four-arm block")
    result = {"cross_mask": {}, "same_mask": {}}
    for arm in ("CP", "SM", "CM"):
        result["cross_mask"][arm + "/SP"] = trajectory_difference(rows_by_arm[arm], rows_by_arm["SP"], "cross_mask")
    for arm, reference in (("SM", "SP"), ("CM", "CP")):
        result["same_mask"][arm + "/" + reference] = trajectory_difference(
            rows_by_arm[arm], rows_by_arm[reference], "same_mask")
    require(all(row["passed"] for group in result.values() for row in group.values()),
            "Paired training trajectories exceed the pre-registered numerical limits")
    return result


def preflight(root, offline=False):
    root = Path(root).resolve()
    plan = verify(root, offline)
    reference, trajectories, files = None, {}, {}
    for job in jobs(plan, "preflight"):
        rows = window(root, plan, job)
        current = identity(rows)
        reference = current if reference is None else reference
        require(current == reference, "Preflight hardware/input/software differs across SP/CP/SM/CM")
        phase_rows = trajectories.setdefault(job["phase"], {})
        require(job["arm"] not in phase_rows, "Duplicate arm in full-model phase preflight")
        phase_rows[job["arm"]] = rows[0]
        for p in (root / "results" / job["directory"]).glob("rank*.json"):
            files[str(p.relative_to(root))] = sha(p)
    require(set(trajectories) == set(range(8)), "Full-model preflight is missing a phase")
    numerical_checks = {str(phase): numerical_comparisons(trajectories[phase]) for phase in range(8)}
    gate = dict(protocol=PROTOCOL, completed=True, run_id=plan["run_id"],
                plan_sha256=sha(root / "plan.json"), rank_identities=reference,
                numerical_limits=NUMERICAL_LIMITS, numerical_checks=numerical_checks,
                preflight_phases=8, preflight_windows=32, preflight_rank_records=256,
                precision_control_passed=True, budget_matched=True,
                files=files, historical_hardware_reused=False)
    path = root / "preflight.json"
    if path.exists():
        require(read(path) == gate, "Saved preflight differs from raw evidence")
    else:
        write(path, gate)
    return gate


def report(root, offline=False, save=True):
    root = Path(root).resolve()
    plan = verify(root, offline)
    gate = preflight(root, offline)
    samples, trajectories, evidence = [], {}, dict(gate["files"])
    for job in jobs(plan, "timing"):
        rows = window(root, plan, job)
        require(identity(rows) == gate["rank_identities"], "Timing hardware/input/software differs from preflight")
        key = (job["trial"], job["block"])
        trajectories.setdefault(key, {})[job["arm"]] = rows[0]
        samples.append(dict(job, ms_per_step=max(r["elapsed_seconds"] for r in rows) * 1000 / 50))
        for p in (root / "results" / job["directory"]).glob("rank*.json"):
            evidence[str(p.relative_to(root))] = sha(p)
    numerical_checks = {f"compare{trial}/block{block:02d}": numerical_comparisons(values)
                        for (trial, block), values in trajectories.items()}
    comparisons = {}
    for trial in range(1, plan["trials"] + 1):
        summaries = {}
        for label, selected in (("round1", [s for s in samples if s["trial"] == trial and s["round"] == 1]),
                                ("round2", [s for s in samples if s["trial"] == trial and s["round"] == 2]),
                                ("combined", [s for s in samples if s["trial"] == trial])):
            values = {arm: [s["ms_per_step"] for s in sorted(selected, key=lambda x: x["block"]) if s["arm"] == arm]
                      for arm in bench.ARMS}
            summaries[label] = dict(mean_ms={arm: statistics.mean(v) for arm, v in values.items()},
                comparisons={c + "/" + b: bench.paired(values[c], values[b])
                             for c, b in bench.COMPARISONS},
                closure_reuse_interaction=bench.paired(
                    [cm * sp for cm, sp in zip(values["CM"], values["SP"])],
                    [sm * cp for sm, cp in zip(values["SM"], values["CP"])]))
        estimates = summaries["combined"]["comparisons"]
        positive = lambda key: estimates[key]["percent"] > 0 and estimates[key]["ci95"][0] > 0
        interaction = summaries["combined"]["closure_reuse_interaction"]
        comparisons[f"compare{trial}"] = dict(run_id=plan["trial_ids"][str(trial)], rounds=summaries,
            target_validated=estimates["CM/SM"]["percent"] >= 1.0 and estimates["CM/SM"]["ci95"][0] >= 1.0,
            mechanism_validated=positive("CM/SM"), grouping_control_passed=positive("CP/SP"),
            merging_benefit_validated=positive("CM/CP"), zero_reuse_control_passed=positive("SM/SP"),
            closure_reuse_interaction_validated=interaction["percent"] > 0 and interaction["ci95"][0] > 0)
    fields = ("target_validated", "mechanism_validated", "grouping_control_passed",
              "merging_benefit_validated", "zero_reuse_control_passed", "closure_reuse_interaction_validated")
    acceptance = {key: all(c[key] for c in comparisons.values()) for key in fields}
    passed = acceptance["target_validated"]
    result = dict(protocol=PROTOCOL, run_id=plan["run_id"], completed=True, correctness_passed=True,
                  precision_control_passed=True, budget_matched=True,
                  preflight_phases=8, preflight_windows=32, preflight_rank_records=256,
                  numerical_limits=NUMERICAL_LIMITS, numerical_checks=numerical_checks,
                  closure_plans=plan["closure_plans"],
                  interaction_formula="100*(1-exp(mean(log(CM/SM)-log(CP/SP))))",
                  full_optimizer_steps_measured=True, target_percent=1.0, **acceptance,
                  attribution_rule="CM/SM isolates closure grouping at the same compression budget and merged "
                                   "decoder. CP/SP tests grouping without inverse reuse; CM/CP tests inverse "
                                   "reuse on the closed mask; SM/SP is the no-reuse control. Each positive "
                                   "mechanism check requires point and paired CI95 lower >0 in every trial; "
                                   "it is separate from the CM/SM >=1% target.",
                  acceptance_status="validated_for_new_same_host_configuration" if passed else "not_validated_1_percent",
                  acceptance_rule=plan["acceptance_rule"], comparisons=comparisons, all_samples=samples,
                  formal_optimizer_steps=len(samples) * 50, timing_rank_records=len(samples) * 8,
                  model=MODEL, rank_identities=gate["rank_identities"], evidence_files=evidence,
                  shared_environment=plan["shared_environment"],
                  megatron_patch=plan["megatron_snapshot"]["patch"],
                  bootstrap=dict(unit="complete fresh-process SP/CP/SM/CM block", samples=10000, seed=20261020),
                  historical_hardware_reused=False, historical_timings_used=False,
                  limitation="All arms use the new FP32 reduction protocol, with distinct scattered/closed masks. "
                             "Fixed numerical tolerances check short synthetic trajectories, not original bitwise "
                             "semantics or long-term convergence. Endpoint losses near zero do not prove convergence.")
    if save:
        if (root / "report.json").exists():
            require(read(root / "report.json") == result, "Saved report differs from raw evidence")
        else:
            write(root / "report.json", result)
        print(json.dumps(dict(target_validated=passed, comparisons=comparisons), indent=2))
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("root", type=Path)
    q.add_argument("--build", type=Path, required=True)
    q.add_argument("--megatron", type=Path, default=REPO.parent / "Megatron-LM")
    q.add_argument("--trials", type=int, default=2)
    q.add_argument("--blocks", type=int, default=16)
    q = sub.add_parser("run")
    q.add_argument("root", type=Path)
    q.add_argument("--stage", choices=("all", "preflight", "timing", "profile"), default="all")
    q.add_argument("--python", type=Path, default=Path(sys.executable))
    q.add_argument("--cuda-home", type=Path)
    q.add_argument("--execute", action="store_true")
    q.add_argument("--worker-timeout", type=int, default=1800)
    q = sub.add_parser("report")
    q.add_argument("root", type=Path)
    q.add_argument("--offline", action="store_true")
    args = p.parse_args()
    if args.action == "report":
        report(args.root, args.offline)
    else:
        globals()[args.action](args)


if __name__ == "__main__":
    main()

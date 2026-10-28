#!/usr/bin/env python3
"""Fresh-host A/decoder/serial/overlap GPT6.7B full optimizer-step confirmation.

The proven ablation4 model worker and adapter are reused with the same A policy
and codec. The separately recorded arm selects an isolated runtime and mode.
No historical calibration record is relabeled as a new hardware measurement.
"""
import argparse
import hashlib
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
import arrival_op3_v2_bench as bench
from common import MEGATRON_COMMIT, REPO, replace_once
from run import arguments as model_arguments

PROTOCOL = "h100-arrival-op3-v2-independent-fullstep-v1"
POLICY = "interleaved_all_attn_mlp"
MODEL = dict(layers=32, hidden=4096, ffn=16384, heads=32, sequence=2048,
             micro_batch=1, global_batch=4, tp=8, sm_count=132,
             seed=1234, token_seed=4321, vocab=32768)
read, write, sha, require = bench.read, bench.write, bench.sha, bench.require


def check_megatron(path):
    require(subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            == MEGATRON_COMMIT, "Megatron must use the registered core_v0.12.3 commit")
    require(not subprocess.check_output(["git", "-C", str(path), "status", "--porcelain",
                                        "--untracked-files=no"], text=True).strip(),
            "Megatron tracked sources differ from the pinned commit")


MEGATRON_GUARD = '''            assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1", \\
                "Using tensor model parallelism or context parallelism require setting the environment variable " \\
                "CUDA_DEVICE_MAX_CONNECTIONS to 1"'''
MEGATRON_OP3_GUARD = '''            assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "8", \\
                "This isolated op3 A/D/S/O experiment requires the environment variable " \\
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
    after = replace_once(before, MEGATRON_GUARD, MEGATRON_OP3_GUARD)
    path.write_text(after)
    patched = dict(original)
    patched[relative] = sha(path)
    require(original[relative] != patched[relative], "Megatron scheduling guard was not patched")
    return dict(source_root=str(source), commit=MEGATRON_COMMIT,
                archive_sha256=hashlib.sha256(archive).hexdigest(),
                snapshot_root=str(target), original_files=original, files=patched,
                changed_files=[relative],
                patch=dict(file=relative, original_sha256=original[relative],
                           patched_sha256=patched[relative], before=MEGATRON_GUARD, after=MEGATRON_OP3_GUARD),
                reason="All A/D/S/O arms share eight CUDA connections so independent producer/consumer streams can overlap. This changes Megatron's scheduling-performance guard, not tensor arithmetic.")


def verify_megatron_snapshot(record):
    require(record["commit"] == MEGATRON_COMMIT and
            record["changed_files"] == ["megatron/training/arguments.py"], "Wrong Megatron patch provenance")
    root = Path(record["snapshot_root"])
    for relative, digest in record["files"].items():
        require(sha(root / relative) == digest, "Megatron snapshot source changed: " + relative)
    patch_record = record["patch"]
    require(patch_record["before"] == MEGATRON_GUARD and patch_record["after"] == MEGATRON_OP3_GUARD,
            "Megatron guard patch changed")
    restored = replace_once((root / patch_record["file"]).read_text(), MEGATRON_OP3_GUARD, MEGATRON_GUARD)
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
    require(args.blocks >= 4 and args.blocks % 2 == 0, "At least four even complete blocks required")
    require(1 <= args.rounds <= 4, "Overlap rounds must be between one and four")
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
                  scope="New same-host op3 A/D/S/O comparison; full GPT6.7B optimizer steps; all-remote codec; synthetic input.")
    write(scripts / "config.json", config)
    (scripts / "adapter.py").write_text(sources.adapter_source())
    (scripts / "taco_support.py").write_text(sources.taco_source())
    worker = sources.worker_source()
    worker = replace_once(worker, "report['protocol'] = PLAN['protocol']",
                          """report['op3_arm'] = os.environ['OP3_ARM']
report['op3_mode'] = os.environ['H100_OP3_MODE']
report['op3_rounds'] = int(os.environ['H100_OP3_ROUNDS'])
report['op3_protocol'] = os.environ['H100_OP3_PROTOCOL']
report['op3_diagnostics'] = os.environ['H100_OP3_DIAGNOSTICS']
report['op3_plan_sha256'] = hashlib.sha256((HERE.parent/'plan.json').read_bytes()).hexdigest()
report['op3_shared_environment'] = {key:os.environ[key] for key in
    ('CUDA_DEVICE_MAX_CONNECTIONS','FLUX_GEMM_RS_GEMM_ONLY')}
report['protocol'] = PLAN['protocol']""")
    (scripts / "worker.py").write_text(worker)
    native = (REPO / "tools/a800/phase5/e2e/megatron_taco.py").read_text()
    (scripts / "megatron_taco.py").write_text(replace_once(native, "self.world==4", "self.world in (4,8)"))
    for source, name in ((Path(__file__).with_name("ablation4_protocol.py"), "routing.py"),
                         (Path(__file__).with_name("runtime.py"), "runtime.py"),
                         (Path(__file__).with_name("ablation4_runtime.py"), "communication.py"),
                         (REPO / "tools/a800/phase4/reference.py", "reference.py")):
        shutil.copy2(source, scripts / name)
    write(scripts / "selection-plan.json", dict(protocol=PROTOCOL, calibration_used=False,
                                               reason="All arms use the unselected all-remote A codec."))
    for arm in bench.ARMS:
        parent = root / "build" / arm
        parent.mkdir(parents=True)
        parent.joinpath(POLICY).symlink_to(identities[arm if arm in ("A", "D") else "O"]["root"], target_is_directory=True)
    jobs = []
    for trial in range(args.trials):
        for job in bench.schedule(args.blocks):
            jobs.append(dict(job, trial=trial + 1,
                             round=1 if job["block"] < args.blocks // 2 else 2,
                             directory=f"compare{trial+1}/{job['directory']}"))
    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,
                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},
                trials=args.trials, blocks=args.blocks, model=MODEL,
                kind="full_optimizer_steps", warmup_steps=10, timed_steps=50,
                build=str(build), libraries=identities, roles=bench.ROLES, arms=list(bench.ARMS),
                build_identity=provenance,
                overlap_rounds=args.rounds, communication=comm["environment"],
                communication_record=comm,
                shared_environment={"CUDA_DEVICE_MAX_CONNECTIONS": "8", "FLUX_GEMM_RS_GEMM_ONLY": "1"},
                megatron=str(root / "megatron"), megatron_commit=MEGATRON_COMMIT,
                megatron_source=str(megatron), megatron_snapshot=megatron_snapshot,
                schedule=jobs, source_files={str(p.relative_to(root)): sha(p) for p in scripts.iterdir()},
                target_percent=1.0, target_validated=False,
                acceptance_rule="Each independent comparison: O/A point and paired complete-block CI95 lower bound >= 1%; all correctness gates pass.",
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
    for name, digest in plan["source_files"].items():
        require(sha(root / name) == digest, f"Frozen source changed: {name}")
    if not offline:
        require(bench.build_identity(Path(plan["build"]), plan["build_identity"]["builder_root"])
                == plan["build_identity"], "Frozen build or frontend changed")
        require(bench.libraries(Path(plan["build"])) == plan["libraries"], "Frozen runtime changed")
        verify_megatron_snapshot(plan["megatron_snapshot"])
    return plan


def jobs(plan, stage):
    if stage in ("all", "preflight", "profile"):
        modes = ("smoke",) if stage != "profile" else ("profile",)
        for mode in modes:
            for arm in bench.ARMS:
                yield dict(arm=arm, mode=mode, trial=0, block=-1, index=-1,
                           directory=f"{mode}-{arm}")
    if stage in ("all", "timing"):
        for job in plan["schedule"]:
            yield dict(job, mode="timing")


def run(args):
    root = args.root.resolve()
    plan = verify(root)
    work = list(jobs(plan, args.stage))
    command = [str(args.python.resolve()), "-m", "torch.distributed.run", "--standalone",
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
        env = bench.environment(root, plan, arm, args.python.resolve(), args.cuda_home, profile=False)
        env = communication.apply(env, plan["communication_record"])
        library = Path(plan["libraries"][arm if arm in ("A", "D") else "O"]["root"])
        dest = root / "results" / job["directory"]
        dest.mkdir(parents=True, exist_ok=False)
        env.update(H100_ABLATION4_EXECUTE="1", H100_TP="8", H100_SM_COUNT="132",
                   H100_RUN_ID=plan["trial_ids"].get(str(job["trial"]), plan["run_id"]),
                   E2E_POLICY=POLICY, E2E_MODE=job["mode"], E2E_CASE="h100-phase6",
                   E2E_BUILD=str(root / "build" / arm), E2E_PROCESS_OUT=str(dest),
                   E2E_BLOCK=str(job["block"]), E2E_WINDOW=str(job["index"]), OP3_ARM=arm,
                   PYTHONPATH=f"{root}/scripts:{plan['megatron']}:{library}/python")
        write(dest / "command.json", dict(command=command, job=job,
            shared_environment=plan["shared_environment"], mode=env["H100_OP3_MODE"],
            rounds=env["H100_OP3_ROUNDS"], diagnostics=env["H100_OP3_DIAGNOSTICS"]))
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
    elif args.stage != "profile":
        report(root)


def window(root, plan, job):
    rows = [read(root / "results" / job["directory"] / f"rank{r}.json") for r in range(8)]
    count = 1 if job["mode"] == "profile" else 3 if job["mode"] == "smoke" else 50
    warmup = 2 if job["mode"] == "smoke" else 10
    for rank, row in enumerate(rows):
        require(row["completed"] and row["passed"] and row["rank"] == rank and row["local_rank"] == rank
                and row["protocol"] == PROTOCOL and row["policy"] == POLICY and row["mode"] == job["mode"]
                and row["op3_arm"] == job["arm"] and row["world_size"] == 8
                and row["run_id"] == plan["trial_ids"].get(str(job["trial"]), plan["run_id"])
                and row["op3_plan_sha256"] == sha(root / "plan.json"), "Wrong or incomplete full-step rank record")
        require(row["op3_shared_environment"] == plan["shared_environment"] and
                row["op3_mode"] == ("serial" if job["arm"] == "S" else "overlap") and
                row["op3_rounds"] == plan["overlap_rounds"] and row["op3_diagnostics"] == "0" and
                row["op3_protocol"] == "h100-arrival-op3-v2",
                "Full-step runtime settings changed")
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
        expected_libs = plan["libraries"][job["arm"] if job["arm"] in ("A", "D") else "O"]["files"]
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


def preflight(root, offline=False):
    root = Path(root).resolve()
    plan = verify(root, offline)
    reference, losses, files = None, None, {}
    for job in jobs(plan, "preflight"):
        rows = window(root, plan, job)
        current = identity(rows)
        reference = current if reference is None else reference
        require(current == reference, "Preflight hardware/input/software differs across A/D/S/O")
        losses = rows[0]["losses"] if losses is None else losses
        require(rows[0]["losses"] == losses, "A/D/S/O preflight loss trajectories differ")
        for p in (root / "results" / job["directory"]).glob("rank*.json"):
            files[str(p.relative_to(root))] = sha(p)
    gate = dict(protocol=PROTOCOL, completed=True, run_id=plan["run_id"],
                plan_sha256=sha(root / "plan.json"), rank_identities=reference,
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
        trajectory = (rows[0]["losses"], rows[0]["grad_norms"])
        if key not in trajectories:
            trajectories[key] = trajectory
        require(trajectories[key] == trajectory, "A/D/S/O paired training trajectories differ")
        samples.append(dict(job, ms_per_step=max(r["elapsed_seconds"] for r in rows) * 1000 / 50))
        for p in (root / "results" / job["directory"]).glob("rank*.json"):
            evidence[str(p.relative_to(root))] = sha(p)
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
                             for c, b in bench.COMPARISONS})
        estimate = summaries["combined"]["comparisons"]["O/A"]
        arrival = summaries["combined"]["comparisons"]["O/S"]
        decoder = summaries["combined"]["comparisons"]["D/A"]
        comparisons[f"compare{trial}"] = dict(run_id=plan["trial_ids"][str(trial)], rounds=summaries,
            target_validated=estimate["percent"] >= 1.0 and estimate["ci95"][0] >= 1.0,
            arrival_benefit_validated=arrival["percent"] > 0 and arrival["ci95"][0] > 0,
            decoder_control_passed=decoder["percent"] > 0 and decoder["ci95"][0] > 0)
    passed = all(c["target_validated"] for c in comparisons.values())
    arrival_passed = all(c["arrival_benefit_validated"] for c in comparisons.values())
    decoder_passed = all(c["decoder_control_passed"] for c in comparisons.values())
    result = dict(protocol=PROTOCOL, run_id=plan["run_id"], completed=True, correctness_passed=True,
                  full_optimizer_steps_measured=True, target_percent=1.0, target_validated=passed,
                  arrival_benefit_validated=arrival_passed, decoder_control_passed=decoder_passed,
                  attribution_status="positive_overlap_increment_O_vs_S" if arrival_passed
                  else "early_consumption_increment_not_proven",
                  attribution_rule="Each independent comparison: O/S point and paired CI95 lower >0. "
                                   "This is separate from the O/A >=1% total-performance target.",
                  acceptance_status="validated_for_new_same_host_configuration" if passed else "not_validated_1_percent",
                  acceptance_rule=plan["acceptance_rule"], comparisons=comparisons, all_samples=samples,
                  formal_optimizer_steps=len(samples) * 50, timing_rank_records=len(samples) * 8,
                  model=MODEL, rank_identities=gate["rank_identities"], evidence_files=evidence,
                  shared_environment=plan["shared_environment"],
                  megatron_patch=plan["megatron_snapshot"]["patch"],
                  bootstrap=dict(unit="complete fresh-process A/D/S/O block", samples=10000, seed=20261020),
                  historical_hardware_reused=False, historical_timings_used=False,
                  limitation="Fixed synthetic data and the registered configuration only; not a long-term convergence result.")
    if save:
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
    q.add_argument("--blocks", type=int, default=12)
    q.add_argument("--rounds", type=int, default=2)
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

#!/usr/bin/env python3
"""Two independent A/V/R GPT6.7B native-submission comparisons.

A and V retain their existing scalar/packed receivers and one output clone.
R preserves V's receiver, packet layout and two barriers while moving immutable
configuration into each native instance and returning its independent output.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import uuid

import ablation4_runtime as communication
import ablation4_protocol as routing
import ablation4_sources as sources
import ma2r_bench as bench
from common import MEGATRON_COMMIT, REPO, replace_once
from run import arguments as model_arguments

PROTOCOL = "h100-ma2-submit-fullstep-v1"
TARGET_PERCENT = 1.0
ACCEPTANCE_FIELDS = ('one_percent_point_target_validated', 'strict_one_percent_confidence_validated',
                     'submission_benefit_validated', 'ma2_control_passed')
ACCEPTANCE_RULE = ('Each of two independent comparisons: R/A point estimate >= 1% and paired '
                   'complete-block CI95 lower bound > 0; R/V point estimate and CI95 lower bound > 0. '
                   'A CI95 lower bound >= 1% is reported separately. V/A is a diagnostic control. '
                   'Require bitwise operator outputs and identical per-rank loss/gradient trajectories.')
POLICY = "interleaved_all_attn_mlp"
MODEL = dict(layers=32, hidden=4096, ffn=16384, heads=32, sequence=2048,
             micro_batch=1, global_batch=4, tp=8, sm_count=132,
             seed=1234, token_seed=4321, vocab=32768)
SHARED_ENVIRONMENT = {"CUDA_DEVICE_MAX_CONNECTIONS": "1", "FLUX_GEMM_RS_GEMM_ONLY": "1"}
read, write, sha, require = bench.read, bench.write, bench.sha, bench.require


def snapshot_megatron(source, target):
    require(subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            == MEGATRON_COMMIT, "Megatron must use the pinned core_v0.12.3 commit")
    require(not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
                                        "--untracked-files=no"], text=True).strip(),
            "Megatron tracked sources differ from the pinned commit")
    archive = subprocess.check_output(["git", "-C", str(source), "archive", "--format=tar", MEGATRON_COMMIT])
    target.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        members = handle.getmembers()
        for member in members:
            path = Path(member.name)
            require(not path.is_absolute() and ".." not in path.parts and
                    (member.isdir() or member.isfile()), "Unexpected archive entry: " + member.name)
        handle.extractall(target, members=members)
    return dict(source_root=str(source), snapshot_root=str(target), commit=MEGATRON_COMMIT,
                archive_sha256=hashlib.sha256(archive).hexdigest(), changed_files=[],
                files={str(p.relative_to(target)): sha(p) for p in target.rglob("*") if p.is_file()})


def verify_megatron_snapshot(record):
    require(record["commit"] == MEGATRON_COMMIT and record["changed_files"] == [],
            "MA2R requires an unmodified pinned Megatron snapshot")
    root = Path(record["snapshot_root"])
    for relative, digest in record["files"].items():
        require(sha(root / relative) == digest, "Megatron snapshot changed: " + relative)


def worker_source():
    worker = replace_once(sources.worker_source(), 'H100_ABLATION4_EXECUTE', 'H100_MA2R_EXECUTE')
    worker = replace_once(worker, 'GPU worker requires ablation4 --execute', 'GPU worker requires MA2-R --execute')
    worker = replace_once(worker, "report['protocol'] = PLAN['protocol']", """report['ma2r_arm'] = os.environ['H100_MA2R_ARM']
report['ma2r_protocol'] = os.environ['H100_MA2R_PROTOCOL']
report['ma2r_plan_sha256'] = hashlib.sha256((HERE.parent/'plan.json').read_bytes()).hexdigest()
report['ma2r_shared_environment'] = {key:os.environ[key] for key in
    ('CUDA_DEVICE_MAX_CONNECTIONS','FLUX_GEMM_RS_GEMM_ONLY')}
report['protocol'] = PLAN['protocol']""")
    return replace_once(worker, "report['workspace_checks'] = adapter.workspace_checks", """report['workspace_checks'] = adapter.workspace_checks
report['ma2r_codec_state'] = {name: dict(owns_output=bool(op.owns_output),
    owns_configuration=bool(op.owns_configuration),
    workspaces=len(op.workspace_info)) for name,op in adapter.scope_ops.items()}""")


def taco_source():
    """Record real frontend ownership without changing A/V's call path."""
    source = replace_once(sources.taco_source(), 'import json', 'import json\nimport os')
    return replace_once(source,
        '            self.owns_output = False  # Original/A/B retain the same output clone.',
        '''            self.owns_output = bool(getattr(self.codec, 'owns_output', False))
            self.owns_configuration = bool(getattr(self.codec, 'owns_configuration', False))
            arm = os.environ['H100_MA2R_ARM']
            if arm not in ('A', 'V', 'R') or (self.owns_output, self.owns_configuration) != (arm == 'R', arm == 'R'):
                raise RuntimeError('MA2-R actual output/configuration ownership does not match the arm')''')


def adapter_source():
    return replace_once(sources.adapter_source(),
        '                assert self.scope_ops[name].owns_output == (not is_fused_v2(self.policy))',
        '''                assert is_fused_v2(self.policy)
                assert self.scope_ops[name].owns_output == (os.environ['H100_MA2R_ARM'] == 'R')
                assert self.scope_ops[name].owns_configuration == (os.environ['H100_MA2R_ARM'] == 'R')''')


def expected_schedule(trials, blocks):
    return [dict(job, trial=trial + 1,
                 round=1 if job["block"] < blocks // 2 else 2,
                 directory=f"compare{trial+1}/{job['directory']}")
            for trial in range(trials) for job in bench.schedule(blocks)]


def prepare(args):
    root, build, megatron = (p.resolve() for p in (args.root, args.build, args.megatron))
    provenance, identities = bench.build_identity(build), bench.libraries(build)
    require(type(args.trials) is int and args.trials == 2, "Exactly two independent full-step comparisons are required")
    require(args.blocks >= 12 and args.blocks % 6 == 0, "Use at least twelve complete blocks, divisible by six")
    schedule = expected_schedule(args.trials, args.blocks)
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
                  scope="MA2-R A/V/R: scalar baseline, unchanged MA2 packed receiver, native instance submission and independent output; full optimizer steps.")
    write(scripts / "config.json", config)
    (scripts / "adapter.py").write_text(adapter_source())
    (scripts / "taco_support.py").write_text(taco_source())
    (scripts / "worker.py").write_text(worker_source())
    native = (REPO / "tools/a800/phase5/e2e/megatron_taco.py").read_text()
    (scripts / "megatron_taco.py").write_text(replace_once(native, "self.world==4", "self.world in (4,8)"))
    for source, name in ((Path(__file__).with_name("ablation4_protocol.py"), "routing.py"),
                         (Path(__file__).with_name("runtime.py"), "runtime.py"),
                         (Path(__file__).with_name("ablation4_runtime.py"), "communication.py"),
                         (REPO / "tools/a800/phase4/reference.py", "reference.py"),
                         (Path(__file__), "fullstep-controller.py"),
                         (Path(bench.__file__), "operator-controller.py")):
        shutil.copy2(source, scripts / name)
    write(scripts / "selection-plan.json", dict(protocol=PROTOCOL, calibration_used=False,
          reason="All arms use identity interleaved order and all remote packets; no arrival calibration."))
    for arm in bench.ARMS:
        parent = root / "build" / arm
        parent.mkdir(parents=True)
        parent.joinpath(POLICY).symlink_to(identities[arm]["root"], target_is_directory=True)
    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,
                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},
                trials=args.trials, blocks=args.blocks, model=MODEL,
                kind="full_optimizer_steps", warmup_steps=10, timed_steps=50,
                build=str(build), libraries=identities, roles=bench.ROLES, arms=list(bench.ARMS),
                build_identity=provenance, communication=comm["environment"], communication_record=comm,
                shared_environment=dict(SHARED_ENVIRONMENT),
                megatron=str(root / "megatron"), megatron_commit=MEGATRON_COMMIT,
                megatron_snapshot=megatron_snapshot, schedule=schedule,
                source_files={str(p.relative_to(root)): sha(p) for p in scripts.iterdir()},
                controller_sha256=sha(Path(__file__)), target_validated=False,
                target_percent=TARGET_PERCENT, acceptance_rule=ACCEPTANCE_RULE,
                historical_hardware_reused=False, historical_timings_used=False)
    write(root / "plan.json", plan)
    print(root)


def verify(root, offline=False):
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    require(plan["protocol"] == PROTOCOL and plan["model"] == MODEL and plan["arms"] == list(bench.ARMS),
            "Wrong MA2R full-step plan")
    require(type(plan["trials"]) is int and plan["trials"] == 2 and
            set(plan["trial_ids"]) == {str(i+1) for i in range(plan["trials"])} and
            len(set(plan["trial_ids"].values())) == plan["trials"], "Missing independent comparison identities")
    require(plan["warmup_steps"] == 10 and plan["timed_steps"] == 50, "Full-step timing changed")
    require(plan['target_percent'] == TARGET_PERCENT and plan['acceptance_rule'] == ACCEPTANCE_RULE,
            'MA2R point-estimate target or acceptance rule changed')
    require(plan["roles"] == bench.ROLES and plan["blocks"] >= 12 and plan["blocks"] % 6 == 0 and
            plan["schedule"] == expected_schedule(plan["trials"], plan["blocks"]) and
            set(plan["libraries"]) == set(bench.ARMS), "Incomplete balanced blocks or libraries")
    require(plan["shared_environment"] == SHARED_ENVIRONMENT and
            plan["communication_record"]["environment"] == plan["communication"], "Full-step environment changed")
    communication.validate(plan["communication_record"])
    for name, digest in plan["source_files"].items():
        require(sha(root / name) == digest, "Frozen source changed: " + name)
    if not offline:
        require(sha(Path(__file__)) == plan["controller_sha256"] and
                sha(Path(bench.__file__)) == plan["source_files"]["scripts/operator-controller.py"],
                "Frozen controllers changed")
        require(bench.build_identity(Path(plan["build"]), plan["build_identity"]["builder_root"])
                == plan["build_identity"], "Frozen build or frontend changed")
        require(bench.libraries(Path(plan["build"])) == plan["libraries"], "Frozen runtime changed")
        verify_megatron_snapshot(plan["megatron_snapshot"])
    return plan


def jobs(plan, stage):
    if stage in ("all", "preflight", "profile"):
        mode = "profile" if stage == "profile" else "smoke"
        for arm in bench.ARMS:
            yield dict(arm=arm, mode=mode, trial=0, block=-1, index=-1, directory=f"{mode}-{arm}")
    if stage in ("all", "timing"):
        for job in plan["schedule"]:
            yield dict(job, mode="timing")


def run(args):
    root = args.root.resolve()
    plan = verify(root)
    work = list(jobs(plan, args.stage))
    python = args.python.expanduser().absolute()
    command = [str(python), "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=8", str(root / "scripts/worker.py"), *model_arguments(MODEL)]
    if not args.execute:
        print(json.dumps(dict(dry_run=True, command=command, jobs=work), indent=2))
        return
    require(args.worker_timeout > 0, "Positive worker timeout required")
    write(root / f"{args.stage}-started.json", dict(run_id=plan["run_id"], jobs=work))
    gated = args.stage == "timing"
    if gated:
        preflight(root)
    for job in work:
        if job["mode"] == "timing" and not gated:
            preflight(root)
            gated = True
        verify(root)
        arm = job["arm"]
        env = bench.environment(root, plan, arm, python, args.cuda_home, profile=False)
        env = communication.apply(env, plan["communication_record"])
        library = Path(plan["libraries"][arm]["root"])
        dest = root / "results" / job["directory"]
        dest.mkdir(parents=True, exist_ok=False)
        env.update(H100_MA2R_EXECUTE="1", H100_TP="8", H100_SM_COUNT="132",
                   H100_RUN_ID=plan["trial_ids"].get(str(job["trial"]), plan["run_id"]),
                   E2E_POLICY=POLICY, E2E_MODE=job["mode"], E2E_CASE="h100-phase6",
                   E2E_BUILD=str(root / "build" / arm), E2E_PROCESS_OUT=str(dest),
                   E2E_BLOCK=str(job["block"]), E2E_WINDOW=str(job["index"]), H100_MA2R_ARM=arm,
                   PYTHONPATH=f"{root}/scripts:{plan['megatron']}:{library}/python")
        write(dest / "command.json", dict(command=command, job=job,
              shared_environment=plan["shared_environment"], protocol=env["H100_MA2R_PROTOCOL"],
              arm=arm, profiling=job["mode"] == "profile"))
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
                and row["ma2r_arm"] == job["arm"] and row["world_size"] == 8
                and row["run_id"] == plan["trial_ids"].get(str(job["trial"]), plan["run_id"])
                and row["ma2r_plan_sha256"] == sha(root / "plan.json"), "Wrong or incomplete full-step rank record")
        require(row["ma2r_shared_environment"] == plan["shared_environment"] and
                row["ma2r_protocol"] == bench.SOURCE_PROTOCOL, "Full-step runtime settings changed")
        require(row['tp_rank'] == rank and row['dp_rank'] == 0 and
                row['tp_group_ranks'] == list(range(8)) and row['dp_group_ranks'] == [rank] and
                row['block'] == job['block'] and row['window'] == job['index'],
                'TP8 rank groups or timing block identity changed')
        require(row['scope_definition'] == routing.definition(POLICY) and
                row['role_plan_sha256'] == {POLICY: {'attn': None, 'mlp': None}},
                'Wrong fixed interleaved full-remote policy')
        states = row["ma2r_codec_state"]
        require(set(states) == set(row["target_modules"]) and len(states) == 64 and
                all(v == dict(owns_output=job['arm'] == 'R', owns_configuration=job['arm'] == 'R',
                              workspaces=1) for v in states.values()),
                "Wrong runtime output ownership or workspace count")
        workspaces = row["workspace_checks"]
        require(set(workspaces) == set(states) and all(len(slots) == 1 for slots in workspaces.values()),
                "Missing independent runtime workspaces")
        pointers = [slot["packet_ptr"] for slots in workspaces.values() for slot in slots]
        require(all(type(p) is int and p > 0 for p in pointers) and len(set(pointers)) == len(pointers),
                "Aliased communication workspaces")
        require(row["communication"] == plan["communication_record"] and not row["sampling"] and
                row["profiling"] == (job["mode"] == "profile") and row["fallback_calls"] == row["skips"] == 0,
                "Sampling, fallback, communication drift or skipped updates")
        require(row["timing_protocol"] == "full-step-one-clear-v2" and row["warmup_steps"] == warmup and
                len(row["losses"]) == len(row["grad_norms"]) == count and
                all(math.isfinite(v) for v in row["losses"] + row["grad_norms"]), "Invalid full-step trajectory")
        require(row["tokens_per_optimizer_step"] == 8192 and row["padded_vocab_size"] == 33792 and
                len(row["target_modules"]) == row["replaced_modules"] == row["independent_scope_instances"] == 64
                and len(row["timed_calls"]) == 64 and all(v == count * 4 for v in row["timed_calls"].values())
                and set(row["post_audit_routes"].values()) == {"candidate"}, "Incomplete 32-layer dual-role execution")
        routes = routing.module_routes(row['observed_shapes'], POLICY)
        require(row['module_policies'] == row['post_audit_routes'] == routes and
                set(row['target_modules']) == set(routes) and
                row['warmup_calls'] == {name: warmup*4 for name in routes} and
                row['timed_calls'] == {name: count*4 for name in routes},
                'Wrong per-module warmup/timed calls or post-audit routes')
        require(row['enabled_modules'] == sorted(routes) and
                row['forward_preserved'] == {name: False for name in routes} and
                row['codec_placements'] == {name: 'fused' for name in routes},
                'Enabled projection or fused codec path changed')
        for name, slots in workspaces.items():
            role = routing.module_role(name)
            require(set(slots[0]) == {'shape', 'packet_ptr'} and slots[0]['shape'] == routing.SHAPES[role],
                    'Wrong role workspace shape')
        require(row["model"] == dict(name="h100-phase6", **{k: v for k, v in MODEL.items() if k != "sm_count"}),
                "Model dimensions or seeds differ from GPT6.7B")
        for role, suffix, k_local in (("attn", ".self_attention.linear_proj", 512),
                                      ("mlp", ".mlp.linear_fc2", 2048)):
            selected = [value for name, value in row["observed_shapes"].items() if name.endswith(suffix)]
            require(len(selected) == 32 and all(
                [v["M"], v["N"], v["K_local"], v["K_global"]] == [2048, 4096, k_local, k_local * 8]
                for v in selected), "Wrong measured role/full shape: " + role)
        devices = row["hardware"]["devices"]
        require(len(devices) == 8 and len({d["uuid"] for d in devices}) == 8 and
                all("H100" in d["name"] and d["sm_count"] == 132 and d["capability"] == [9, 0] for d in devices),
                "Not eight full H100s")
        require(row['hardware']['local_rank'] == rank and row['gpu_uuid'] == devices[rank]['uuid'],
                'GPU rank/device order changed')
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


def preflight(root, offline=False):
    root = Path(root).resolve()
    plan = verify(root, offline)
    reference, trajectory, files = None, None, {}
    for job in jobs(plan, "preflight"):
        rows = window(root, plan, job)
        current = identity(rows)
        reference = current if reference is None else reference
        require(current == reference, "Preflight hardware/input/software differs across A/V/R")
        current_trajectory = [(row['losses'], row['grad_norms']) for row in rows]
        trajectory = current_trajectory if trajectory is None else trajectory
        require(current_trajectory == trajectory, "A/V/R preflight loss/gradient trajectories differ")
        for p in (root / "results" / job["directory"]).glob("rank*.json"):
            files[str(p.relative_to(root))] = sha(p)
    gate = dict(protocol=PROTOCOL, completed=True, run_id=plan["run_id"],
                plan_sha256=sha(root / "plan.json"), rank_identities=reference,
                identical_trajectories=True, files=files, historical_hardware_reused=False)
    path = root / "preflight.json"
    if path.exists():
        require(read(path) == gate, "Saved preflight differs from raw evidence")
    else:
        write(path, gate)
    return gate


def comparison_acceptance(estimates):
    require(set(estimates) == {'R/A', 'R/V', 'V/A'}, 'Missing paired comparison')
    for value in estimates.values():
        require(len(value['ci95']) == 2 and all(math.isfinite(x) for x in
                [value['percent'], *value['ci95']]) and value['ci95'][0] <= value['ci95'][1],
                'Invalid speedup confidence interval')
    positive = lambda contrast: estimates[contrast]['percent'] > 0 and estimates[contrast]['ci95'][0] > 0
    main = estimates['R/A']['percent'] >= TARGET_PERCENT and estimates['R/A']['ci95'][0] > 0
    strict = estimates['R/A']['percent'] >= TARGET_PERCENT and estimates['R/A']['ci95'][0] >= TARGET_PERCENT
    submission = positive('R/V')
    return dict(target_validated=main and submission, one_percent_point_target_validated=main,
                strict_one_percent_confidence_validated=strict,
                submission_benefit_validated=submission, ma2_control_passed=positive('V/A'))


def report(root, offline=False, save=True):
    root = Path(root).resolve()
    plan, gate = verify(root, offline), preflight(root, offline)
    samples, trajectories, evidence = [], {}, dict(gate["files"])
    for job in jobs(plan, "timing"):
        rows = window(root, plan, job)
        require(identity(rows) == gate["rank_identities"], "Timing hardware/input/software differs from preflight")
        key = (job["trial"], job["block"])
        trajectory = [(row['losses'], row['grad_norms']) for row in rows]
        trajectories.setdefault(key, trajectory)
        require(trajectories[key] == trajectory, "A/V/R paired training trajectories differ")
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
                comparisons={c + "/" + b: bench.paired(values[c], values[b]) for c, b in bench.COMPARISONS})
        estimates = summaries["combined"]["comparisons"]
        comparisons[f"compare{trial}"] = dict(run_id=plan["trial_ids"][str(trial)], rounds=summaries,
                                              **comparison_acceptance(estimates))
    outcomes = {key: all(c[key] for c in comparisons.values()) for key in
                ('target_validated', *ACCEPTANCE_FIELDS)}
    result = dict(protocol=PROTOCOL, run_id=plan["run_id"], completed=True, correctness_passed=True,
                  identical_trajectories=True, full_optimizer_steps_measured=True, **outcomes,
                  target_percent=TARGET_PERCENT,
                  acceptance_status="one_percent_point_target_and_positive_submission_increment" if outcomes["target_validated"] else "not_validated",
                  acceptance_rule=plan["acceptance_rule"], comparisons=comparisons, all_samples=samples,
                  formal_optimizer_steps=len(samples) * 50, timing_rank_records=len(samples) * 8,
                  model=MODEL, rank_identities=gate["rank_identities"], evidence_files=evidence,
                  shared_environment=plan["shared_environment"],
                  bootstrap=dict(unit="complete fresh-process A/V/R block", samples=10000, seed=bench.BOOTSTRAP_SEED),
                  historical_hardware_reused=False, historical_timings_used=False,
                  limitation="Fixed synthetic inputs and this configuration only; no long-term convergence claim.")
    if save:
        path = root / "report.json"
        if path.exists():
            require(read(path) == result, "Saved full-step report differs from raw evidence")
        else:
            write(path, result)
        (root/'report.md').write_text(report_markdown(result))
        print(json.dumps(dict(target_validated=outcomes["target_validated"], comparisons=comparisons), indent=2))
    return result


def report_markdown(result):
    lines = ['# MA2-R 实例配置与独立输出：完整训练步结果', '',
             'A：原交替排序 scalar 解码；V：原 MA2 packed 预取；R：保持 V 解码，使用 native 实例配置并直接返回独立输出。', '',
             'compare1 和 compare2 是两次独立比较，各含 12 个完整 A/V/R 块；每次内部的 round1/round2 是顺序平衡的两个半程。', '',
             '| 比较 | A / V / R 平均步时 (ms) | R/A 收益 % [95% CI] | R/V 收益 % [95% CI] | V/A 收益 % [95% CI] | 目标通过 | R/A CI 下限 ≥1% |',
             '|---|---|---|---|---|---|---|']
    for name, comparison in result['comparisons'].items():
        summary = comparison['rounds']['combined']
        means = ' / '.join(f'{summary["mean_ms"][arm]:.4f}' for arm in bench.ARMS)
        estimates = []
        for contrast in ('R/A', 'R/V', 'V/A'):
            value = summary['comparisons'][contrast]
            estimates.append(f'{value["percent"]:+.4f} [{value["ci95"][0]:+.4f}, {value["ci95"][1]:+.4f}]')
        passed = '是' if comparison['target_validated'] else '否'
        strict = '是' if comparison['strict_one_percent_confidence_validated'] else '否'
        lines.append('| ' + ' | '.join([name, means, *estimates, passed, strict]) + ' |')
    status = '通过' if result['target_validated'] else '未通过'
    lines += ['', f'结论：{status}。两次独立比较均须满足 R/A 点估计 ≥1%、95% CI 下限 >0，以及 R/V 点估计和 95% CI 下限均 >0。',
              'R/A 的 95% CI 下限 ≥1% 是单列的更严格结果，不作为上述点估计目标的附加门槛。',
              '正值表示候选更快，负值表示更慢。V/A 复测原 MA2 效果，只记录，不参与通过判定。', '',
              f'完整优化器步：{result["formal_optimizer_steps"]}；计时窗口：{len(result["all_samples"])}。',
              '全部 rank 的配对 loss/gradient 轨迹必须一致。数据为固定合成输入；单独 profile 时长不用于训练步收益结论。', '']
    return '\n'.join(lines)


def decoder_matches(name, arm):
    return bench.decoder_matches(name, arm)


def trace_projection_records(events, arm):
    require(arm in bench.ARMS, "Unknown MA2R arm")
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
        decoders = [e for e in group if 'taco' in e['name'] and 'decode' in e['name']]
        barriers = [e for e in group if "CudaIpcBarrierAllKernel" in e["name"]]
        require(len(gemms) == len(decoders) == 1 and len(barriers) == 2 and
                decoder_matches(decoders[0]["name"], arm), "Wrong GEMM/decoder/two-barrier path")
        require(not any("taco_op3" in e["name"] or "taco_closure" in e["name"] for e in group),
                "Unexpected arrival/closure runtime")
        ordered = [gemms[0], barriers[0], decoders[0], barriers[1]]
        require(all(e.get("args", {}).get("stream") is not None for e in ordered) and
                len({e["args"]["stream"] for e in ordered}) == 1 and all(
                a["ts"] + a["dur"] <= b["ts"] + .01 for a, b in zip(ordered, ordered[1:])),
                "Wrong projection stream/barrier ordering")
        clones = [e for e in cpu_ops[tid] if e["name"] == "aten::clone" and
                  call["ts"] <= e["ts"] and e["ts"] + e["dur"] <= call["ts"] + call["dur"] + .01]
        require(len(clones) == (0 if arm == 'R' else 1), "Wrong per-projection output clone count")
        records[scope["name"].split(".")[1]].append(dict(cpu_us=call['dur'], gemm_us=gemms[0]["dur"],
                                                       decode_us=decoders[0]["dur"]))
    require(len(used) == sum(map(len, scopes.values())) and set(records) == set(bench.ROLES) and
            all(len(records[role]) == 128 for role in bench.ROLES),
            "Profile must cover every projection in 32 layers and four microbatches")
    return {role: dict(projection_calls=len(rows), barrier_calls=2 * len(rows),
                      output_clones=0 if arm == 'R' else len(rows),
                      projection_cpu_median_us=statistics.median(r['cpu_us'] for r in rows),
                      gemm_median_us=statistics.median(r["gemm_us"] for r in rows),
                      decode_median_us=statistics.median(r["decode_us"] for r in rows))
            for role, rows in records.items()}


def profile_summary(root, offline=False):
    root = Path(root).resolve()
    plan, gate = verify(root, offline), preflight(root, offline)
    profiles, evidence = {}, {}
    for job in jobs(plan, "profile"):
        rows = window(root, plan, job)
        require(identity(rows) == gate["rank_identities"], "Profile hardware/input/software changed")
        directory = root / "results" / job["directory"]
        for rank in range(8):
            path = directory / f"rank{rank}-profile.json.gz"
            with gzip.open(path, "rt") as handle:
                profiles[f"{job['arm']}/rank{rank}"] = trace_projection_records(json.load(handle)["traceEvents"], job["arm"])
            for item in (path, directory / f"rank{rank}.json"):
                evidence[str(item.relative_to(root))] = sha(item)
    return dict(protocol=PROTOCOL, run_id=plan["run_id"], completed=True, execution_validated=True,
                plan_sha256=sha(root / "plan.json"), profiles=profiles, evidence_files=evidence,
                limitation="Separate profiles verify execution counts; profiler durations do not estimate unprofiled step savings.")


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--megatron", type=Path, default=REPO.parent / "Megatron-LM")
    p.add_argument("--trials", type=int, choices=(2,), default=2)
    p.add_argument("--blocks", type=int, default=12)
    p = sub.add_parser("run")
    p.add_argument("root", type=Path)
    p.add_argument("--stage", choices=("all", "preflight", "timing", "profile"), default="all")
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--cuda-home", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--worker-timeout", type=int, default=1800)
    p = sub.add_parser("report")
    p.add_argument("root", type=Path)
    p.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "report":
        report(args.root, args.offline)
    else:
        globals()[args.action](args)


if __name__ == "__main__":
    main()

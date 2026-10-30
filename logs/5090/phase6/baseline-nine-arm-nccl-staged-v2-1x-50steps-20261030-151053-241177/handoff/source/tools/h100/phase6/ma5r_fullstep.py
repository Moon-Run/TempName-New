#!/usr/bin/env python3
"""One or two balanced MA5-R role-specific decoder versus CR2 comparisons."""
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
import ma5r_bench as bench
from common import MEGATRON_COMMIT, REPO, replace_once
from run import arguments as model_arguments

PROTOCOL = "h100-ma5-role-split-fullstep-v1"
TARGET_MS = 500.0
ACCEPTANCE_FIELDS = ('increment_over_CR2_validated', 'absolute_500ms_observed')
ACCEPTANCE_RULE = ('Within EACH requested trial, report T/CR2 paired point and CI95; '
                   'positive increment requires point>0 and CI95 lower>0. '
                   'Separately report candidate arithmetic mean<=500ms. Target requires both. '
                   'One trial is exploratory; no historical timing samples and no discarded blocks. '
                   'Require bitwise operator outputs and identical loss/gradient trajectories.')
R_POLICY = 'interleaved_all_attn_mlp'
C_POLICY = 'interleaved_arrival_selective_attn_mlp'


def policy_for_arm(arm):
    return C_POLICY if bench.family(arm)=='selective' else R_POLICY


def role_hashes(selection):
    return {R_POLICY:{'attn':None,'mlp':None},C_POLICY:{r:p['entry_sha256']
            for r,p in selection['variants'][C_POLICY]['roles'].items()}}


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
            "CR2 requires an unmodified pinned Megatron snapshot")
    root = Path(record["snapshot_root"])
    for relative, digest in record["files"].items():
        require(sha(root / relative) == digest, "Megatron snapshot changed: " + relative)


def worker_source():
    worker = replace_once(sources.worker_source(), 'H100_ABLATION4_EXECUTE', 'H100_MA5R_EXECUTE')
    worker = replace_once(worker, 'GPU worker requires ablation4 --execute', 'GPU worker requires MA5-R tile/block --execute')
    worker = replace_once(worker, "report['protocol'] = PLAN['protocol']", """report['ma5r_arm'] = os.environ['H100_MA5R_ARM']
report['ma5r_protocol'] = os.environ['H100_MA5R_PROTOCOL']
report['ma5r_plan_sha256'] = hashlib.sha256((HERE.parent/'plan.json').read_bytes()).hexdigest()
report['ma5r_shared_environment'] = {key:os.environ[key] for key in
    ('CUDA_DEVICE_MAX_CONNECTIONS','FLUX_GEMM_RS_GEMM_ONLY')}
report['protocol'] = PLAN['protocol']""")
    return replace_once(worker, "report['workspace_checks'] = adapter.workspace_checks", """report['workspace_checks'] = adapter.workspace_checks
report['ma5r_codec_state'] = {name: dict(owns_output=bool(op.owns_output),
    owns_configuration=bool(op.owns_configuration),
    workspaces=len(op.workspace_info)) for name,op in adapter.scope_ops.items()}
import importlib.util
ma5r_spec = importlib.util.spec_from_file_location('ma5r_operator_controller', HERE/'operator-controller.py')
ma5r_operator = importlib.util.module_from_spec(ma5r_spec)
ma5r_spec.loader.exec_module(ma5r_operator)
ma5r_plan = json.loads((HERE.parent/'plan.json').read_text())
from routing import module_role as ma5r_operator_role
report['ma5r_native_tuning'] = {name: ma5r_operator.native_tuning(op.codec, os.environ['H100_MA5R_ARM'], ma5r_plan, ma5r_operator_role(name))
    for name,op in adapter.scope_ops.items()}
report['ma5r_submission_state'] = {name: op.codec.cr2_submission_state()
    for name,op in adapter.scope_ops.items()}""")


def taco_source():
    source=replace_once(sources.taco_source(),
        '            self.owns_output = False  # Original/A/B retain the same output clone.',
        "            raise RuntimeError('MA5-R requires the selective CR2 policy')")
    return replace_once(source,'        self.owns_output = self.codec.owns_configuration',
                        '        self.owns_configuration = self.codec.owns_configuration\n'
                        '        self.owns_output = self.codec.owns_configuration')


def adapter_source():
    return replace_once(sources.adapter_source(),
        '                assert self.scope_ops[name].owns_output == (not is_fused_v2(self.policy))',
        "                assert self.scope_ops[name].owns_output  is True\n"
        "                assert self.scope_ops[name].owns_configuration  is True")


def expected_schedule(trials, blocks):
    return [dict(job, trial=trial + 1,
                 round=1 if job["block"] < blocks // 2 else 2,
                 directory=f"compare{trial+1}/{job['directory']}")
            for trial in range(trials) for job in bench.schedule(blocks)]


def prepare(args):
    root, build, megatron = (p.resolve() for p in (args.root, args.build, args.megatron))
    provenance, identities = bench.build_identity(build), bench.libraries(build)
    selection=read(build/"selection-plan.json")
    bench.verify_selection(selection)
    require(type(args.trials) is int and args.trials in (1,2), "Use one or two independent full-step comparisons")
    require(args.blocks >= 8 and args.blocks % 8 == 0, "Use at least eight complete blocks, divisible by eight")
    schedule = expected_schedule(args.trials, args.blocks)
    root.mkdir(parents=True, exist_ok=False)
    scripts = root / "scripts"
    scripts.mkdir()
    (root / "results").mkdir()
    megatron_snapshot = snapshot_megatron(megatron, root / "megatron")
    comm = communication.capture("0")
    config = read(REPO / "tools/a800/phase5/e2e/config.json")
    config.update(protocol=PROTOCOL, policies=[C_POLICY], orders=[], repetitions=args.trials,
                  warmup_steps=10, timed_steps=50, smoke_warmup_steps=2, smoke_steps=3,
                  graph_forward=False, double_buffered=True, ring_reduction=True,
                  cases=[dict(name="h100-phase6", **{k: MODEL[k] for k in ("hidden", "ffn", "heads", "sequence", "tp")})],
                  common={k: MODEL[k] for k in ("layers", "micro_batch", "global_batch", "seed", "token_seed", "vocab")},
                  role_plan_sha256=role_hashes(selection), communication=comm,
                  scope="MA5-R role-specific decoder versus unchanged CR2; full optimizer steps.")
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
    write(scripts / "selection-plan.json", selection)
    for arm in bench.ARMS:
        parent = root / "build" / arm
        parent.mkdir(parents=True)
        parent.joinpath(policy_for_arm(arm)).symlink_to(identities[arm]["root"], target_is_directory=True)
    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,
                trial_ids={str(i+1): uuid.uuid4().hex for i in range(args.trials)},
                trials=args.trials, blocks=args.blocks, model=MODEL, selection=selection,
                kind="full_optimizer_steps", warmup_steps=10, timed_steps=50,
                build=str(build), libraries=identities, roles=bench.ROLES, arms=list(bench.ARMS),
                build_identity=provenance, tuning=provenance["tuning"], communication=comm["environment"], communication_record=comm,
                shared_environment=dict(SHARED_ENVIRONMENT),
                megatron=str(root / "megatron"), megatron_commit=MEGATRON_COMMIT,
                megatron_snapshot=megatron_snapshot, schedule=schedule,
                source_files={str(p.relative_to(root)): sha(p) for p in scripts.iterdir()},
                controller_sha256=sha(Path(__file__)), target_validated=False,
                target_ms=TARGET_MS, acceptance_rule=ACCEPTANCE_RULE,
                historical_hardware_reused=False, historical_timings_used=False)
    write(root / "plan.json", plan)
    print(root)


def verify(root, offline=False):
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    require(plan["protocol"] == PROTOCOL and plan["model"] == MODEL and plan["arms"] == list(bench.ARMS),
            "Wrong MA5-R full-step plan")
    require(type(plan["trials"]) is int and plan["trials"] in (1,2) and
            set(plan["trial_ids"]) == {str(i+1) for i in range(plan["trials"])} and
            len(set(plan["trial_ids"].values())) == plan["trials"], "Missing independent comparison identities")
    require(plan["warmup_steps"] == 10 and plan["timed_steps"] == 50, "Full-step timing changed")
    require(plan['target_ms'] == TARGET_MS and plan['acceptance_rule'] == ACCEPTANCE_RULE,
            'MA5-R absolute target or acceptance rule changed')
    require(plan["roles"] == bench.ROLES and plan["blocks"] >= 8 and plan["blocks"] % 8 == 0 and
            plan["schedule"] == expected_schedule(plan["trials"], plan["blocks"]) and
            set(plan["libraries"]) == set(bench.ARMS), "Incomplete balanced blocks or libraries")
    require(plan["shared_environment"] == SHARED_ENVIRONMENT and
            plan["communication_record"]["environment"] == plan["communication"], "Full-step environment changed")
    bench.verify_selection(plan["selection"])
    bench.validate_tuning(plan["tuning"])
    require(plan["tuning"] == plan["build_identity"]["tuning"], "Tuning differs from frozen build")
    require(read(root/"scripts/selection-plan.json")==plan["selection"], "Selection snapshot changed")
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
        env.update(H100_MA5R_EXECUTE="1", H100_TP="8", H100_SM_COUNT="132",
                   H100_RUN_ID=plan["trial_ids"].get(str(job["trial"]), plan["run_id"]),
                   E2E_POLICY=policy_for_arm(arm), E2E_MODE=job["mode"], E2E_CASE="h100-phase6",
                   E2E_BUILD=str(root / "build" / arm), E2E_PROCESS_OUT=str(dest),
                   E2E_BLOCK=str(job["block"]), E2E_WINDOW=str(job["index"]), H100_MA5R_ARM=arm,
                   PYTHONPATH=f"{root}/scripts:{plan['megatron']}:{library}/python")
        write(dest / "command.json", dict(command=command, job=job,
              shared_environment=plan["shared_environment"], protocol=env["H100_MA5R_PROTOCOL"],
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
    policy=policy_for_arm(job["arm"])
    count = 1 if job["mode"] == "profile" else 3 if job["mode"] == "smoke" else 50
    warmup = 2 if job["mode"] == "smoke" else 10
    for rank, row in enumerate(rows):
        require(row["completed"] and row["passed"] and row["rank"] == rank and row["local_rank"] == rank
                and row["protocol"] == PROTOCOL and row["policy"] == policy and row["mode"] == job["mode"]
                and row["ma5r_arm"] == job["arm"] and row["world_size"] == 8
                and row["run_id"] == plan["trial_ids"].get(str(job["trial"]), plan["run_id"])
                and row["ma5r_plan_sha256"] == sha(root / "plan.json"), "Wrong or incomplete full-step rank record")
        require(row["ma5r_shared_environment"] == plan["shared_environment"] and
                row["ma5r_protocol"] == bench.SOURCE_PROTOCOL, "Full-step runtime settings changed")
        require(row['tp_rank'] == rank and row['dp_rank'] == 0 and
                row['tp_group_ranks'] == list(range(8)) and row['dp_group_ranks'] == [rank] and
                row['block'] == job['block'] and row['window'] == job['index'],
                'TP8 rank groups or timing block identity changed')
        require(row['scope_definition'] == routing.definition(policy) and
                row['role_plan_sha256'] == role_hashes(plan["selection"]),
                'Wrong fixed interleaved full-remote policy')
        states = row["ma5r_codec_state"]
        require(set(states) == set(row["target_modules"]) and len(states) == 64 and
                all(v == dict(owns_output=True, owns_configuration=True,
                              workspaces=2 if bench.family(job['arm'])=='selective' else 1) for v in states.values()),
                "Wrong runtime output ownership or workspace count")
        workspaces = row["workspace_checks"]
        require(set(workspaces) == set(states) and all(len(slots) == (2 if bench.family(job['arm'])=='selective' else 1) for slots in workspaces.values()),
                "Missing independent runtime workspaces")
        pointers = [slot["packet_ptr"] for slots in workspaces.values() for slot in slots]
        require(all(type(p) is int and p > 0 for p in pointers) and len(set(pointers)) == len(pointers),
                "Aliased communication workspaces")
        require(row["ma5r_native_tuning"] == {name: bench.expected_tuning(plan,job["arm"],routing.module_role(name)) for name in states},
                "Native tuning differs from frozen build")
        submission = row['ma5r_submission_state']
        require(set(submission) == set(states),
                'Missing CR2 per-module cached submission evidence')
        for name, actual in submission.items():
            bench.validate_submission(actual, job['arm'], routing.module_role(name),
                                      [s['packet_ptr'] for s in workspaces[name]], minimum_calls=1)
            require(sum(s['successful_calls'] for s in actual) == (warmup + count) * 4,
                    'CR2 native successful call count differs from full-step projections')
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
        routes = routing.module_routes(row['observed_shapes'], policy)
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
            expected_keys={'shape','packet_ptr','mask_ptr','mask_sha256'} if policy==C_POLICY else {'shape','packet_ptr'}
            require(all(set(slot)==expected_keys and slot['shape']==routing.SHAPES[role] for slot in slots), 'Wrong workspace layout')
            if policy==C_POLICY:
                record=plan['selection']['variants'][C_POLICY]['roles'][role]
                require(all(slot['mask_sha256']==record['mask_sha256'] and slot['mask_ptr']>0 for slot in slots),'Wrong selective mask')
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
    reference, trajectory, files = None, {}, {}
    for job in jobs(plan, "preflight"):
        rows = window(root, plan, job)
        current = identity(rows)
        reference = current if reference is None else reference
        require(current == reference, "Preflight hardware/input/software differs across T/CR2")
        current_trajectory = [(row['losses'], row['grad_norms']) for row in rows]
        key=bench.family(job['arm']); trajectory.setdefault(key,current_trajectory)
        require(current_trajectory == trajectory[key], "T/CR2 preflight loss/gradient trajectories differ")
        for p in (root / "results" / job["directory"]).glob("rank*.json"):
            files[str(p.relative_to(root))] = sha(p)
    gate = dict(protocol=PROTOCOL, completed=True, run_id=plan["run_id"],
                plan_sha256=sha(root / "plan.json"), rank_identities=reference,
                identical_trajectories_within_pairs=True, equivalence_pairs=[["T","CR2"]], files=files, historical_hardware_reused=False)
    path = root / "preflight.json"
    if path.exists():
        require(read(path) == gate, "Saved preflight differs from raw evidence")
    else:
        write(path, gate)
    return gate


def comparison_acceptance(estimates, mean_ms):
    require(set(estimates) == {'T/CR2'} and set(mean_ms) == set(bench.ARMS), 'Missing paired comparison')
    value = estimates['T/CR2']
    require(len(value['ci95']) == 2 and all(math.isfinite(v) for v in [value['percent'],*value['ci95']])
            and value['ci95'][0] <= value['ci95'][1], 'Invalid confidence interval')
    require(all(math.isfinite(v) and v > 0 for v in mean_ms.values()), 'Invalid mean step time')
    increment = value['percent'] > 0 and value['ci95'][0] > 0
    absolute = mean_ms['T'] <= TARGET_MS
    return dict(increment_over_CR2_validated=increment, absolute_500ms_observed=absolute,
                target_validated=increment and absolute)


def aggregate_acceptance(comparisons):
    require(len(comparisons) in (1,2), 'One or two complete comparisons required')
    return {key: all(c[key] for c in comparisons.values())
            for key in (*ACCEPTANCE_FIELDS,'target_validated')}


def report(root, offline=False, save=True):
    root = Path(root).resolve()
    plan, gate = verify(root, offline), preflight(root, offline)
    samples, trajectories, evidence = [], {}, dict(gate["files"])
    numeric={arm:[] for arm in bench.ARMS}
    for job in jobs(plan, "timing"):
        rows = window(root, plan, job)
        require(identity(rows) == gate["rank_identities"], "Timing hardware/input/software differs from preflight")
        key = (job["trial"], job["block"], bench.family(job["arm"]))
        trajectory = [(row['losses'], row['grad_norms']) for row in rows]
        trajectories.setdefault(key, trajectory)
        require(trajectories[key] == trajectory, "T/CR2 paired training trajectories differ")
        numeric[job['arm']].append(dict(mean_loss=statistics.mean(rows[0]['losses']),
            final_loss=rows[0]['losses'][-1],final_grad_norm=rows[0]['grad_norms'][-1]))
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
                median_ms={arm: statistics.median(v) for arm,v in values.items()},
                comparisons={c + "/" + b: bench.paired(values[c], values[b]) for c, b in bench.COMPARISONS})
        estimates = summaries["combined"]["comparisons"]
        comparisons[f"compare{trial}"] = dict(run_id=plan["trial_ids"][str(trial)], rounds=summaries,
                                              **comparison_acceptance(estimates, summaries["combined"]["mean_ms"]))
    outcomes = aggregate_acceptance(comparisons)
    result = dict(protocol=PROTOCOL, run_id=plan["run_id"], completed=True, correctness_passed=True,
                  identical_trajectories_within_pairs=True, equivalence_pairs=[["T","CR2"]], full_optimizer_steps_measured=True, **outcomes,
                  target_ms=TARGET_MS, trials=plan["trials"], tuning=plan["tuning"],
                  exploratory=plan["trials"] == 1,
                  acceptance_status="positive_increment_and_mean_at_most_500ms" if outcomes["target_validated"] else "not_validated",
                  acceptance_rule=plan["acceptance_rule"], comparisons=comparisons, all_samples=samples,
                  formal_optimizer_steps=len(samples) * 50, timing_rank_records=len(samples) * 8,
                  model=MODEL, rank_identities=gate["rank_identities"], evidence_files=evidence,
                  numeric_summary={arm:{key:statistics.mean(row[key] for row in rows) for key in rows[0]} for arm,rows in numeric.items()},
                  shared_environment=plan["shared_environment"],
                  bootstrap=dict(unit="complete fresh-process T/CR2 block", samples=10000, seed=bench.BOOTSTRAP_SEED),
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
    lines = ['# MA5-R tile/block 与 CR2 完整训练结果', '',
             '全部窗口保留；500 ms 为候选均值目标，配对收益单独判断。', '',
             '| 比较 | T / CR2 平均 ms | T / CR2 中位 ms | T/CR2 收益 % [95% CI] | 均值≤500 ms | 正增量 |',
             '| --- | --- | --- | --- | --- | --- |']
    for name,c in result['comparisons'].items():
        combined=c['rounds']['combined']; estimate=combined['comparisons']['T/CR2']
        means=' / '.join(f"{combined['mean_ms'][a]:.3f}" for a in bench.ARMS)
        medians=' / '.join(f"{combined['median_ms'][a]:.3f}" for a in bench.ARMS)
        effect=f"{estimate['percent']:+.3f} [{estimate['ci95'][0]:+.3f}, {estimate['ci95'][1]:+.3f}]"
        lines.append(f"| {name} | {means} | {medians} | {effect} | {c['absolute_500ms_observed']} | {c['increment_over_CR2_validated']} |")
    lines += ['', '一次比较为探索性结果；最终收益仅适用于本硬件和固定合成输入。',
              '算子输出和逐 rank loss/gradient 轨迹必须与同场 CR2 一致。', '']
    return '\n'.join(lines)


def decoder_matches(name, arm, role):
    return bench.decoder_matches(name, arm, role)


def trace_projection_records(events, arm, tuning=None):
    require(arm in bench.ARMS, "Unknown CR2 arm")
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
        role = scope["name"].split(".")[1]
        require(call["ts"] + call["dur"] <= scope["ts"] + scope["dur"] + .01 and
                (tid, index) not in used, "Invalid or repeated projection CPU association")
        used.add((tid, index))
        group = sorted(kernels[call["args"]["External id"]], key=lambda e: e["ts"])
        gemms = [e for e in group if "flux_bf16" in e["name"] and "reducescatter" in e["name"]]
        decoders = [e for e in group if 'taco' in e['name'] and 'decode' in e['name']]
        barriers = [e for e in group if "CudaIpcBarrierAllKernel" in e["name"]]
        require(len(gemms) == len(decoders) == 1 and len(barriers) == bench.barriers_per_call(arm) and
                decoder_matches(decoders[0]["name"], arm, role), "Wrong GEMM/decoder/two-barrier path")
        require(not any("taco_op3" in e["name"] or "taco_closure" in e["name"] for e in group),
                "Unexpected arrival/closure runtime")
        ordered = [gemms[0], barriers[0], decoders[0]] + (barriers[1:] if bench.barriers_per_call(arm)==2 else [])
        require(all(e.get("args", {}).get("stream") is not None for e in ordered) and
                len({e["args"]["stream"] for e in ordered}) == 1 and all(
                a["ts"] + a["dur"] <= b["ts"] + .01 for a, b in zip(ordered, ordered[1:])),
                "Wrong projection stream/barrier ordering")
        clones = [e for e in cpu_ops[tid] if e["name"] == "aten::clone" and
                  call["ts"] <= e["ts"] and e["ts"] + e["dur"] <= call["ts"] + call["dur"] + .01]
        require(len(clones) == bench.clone_per_call(arm), "Wrong per-projection output clone count")
        within_call = [e for e in cpu_ops[tid] if call['ts'] <= e['ts'] and
                       e['ts'] + e['dur'] <= call['ts'] + call['dur'] + .01]
        # GemmRSFunction always reshapes the input and returned output. Their
        # nested views remain valid; CR2 removes only the native output slice
        # and the standalone native view, not the unchanged autograd adapter.
        reshapes = [e for e in within_call if e['name'] == 'aten::reshape']
        views = [e for e in within_call if e['name'] == 'aten::slice' or
                 (e['name'] == 'aten::view' and not any(
                     p['ts'] <= e['ts'] and e['ts'] + e['dur'] <= p['ts'] + p['dur'] + .01
                     for p in reshapes))]
        adapter_views = [e for e in within_call if e['name'] == 'aten::view' and e not in views]
        require(not views, 'CR2 direct output executed legacy slice/view')
        launch = bench.validate_launch(gemms[0], decoders[0], arm, role, tuning)
        records[scope["name"].split(".")[1]].append(dict(launch=launch,cpu_us=call['dur'], gemm_us=gemms[0]["dur"],
                                                       decode_us=decoders[0]["dur"], output_slice_view_count=len(views),
                                                       adapter_view_count=len(adapter_views),
                                                       publication_barrier_us=barriers[0]['dur'],
                                                       device_path_us=ordered[-1]['ts']+ordered[-1]['dur']-ordered[0]['ts']))
    require(len(used) == sum(map(len, scopes.values())) and set(records) == set(bench.ROLES) and
            all(len(records[role]) == 128 for role in bench.ROLES),
            "Profile must cover every projection in 32 layers and four microbatches")
    return {role: dict(launches=[r["launch"] for r in rows],projection_calls=len(rows), barrier_calls=bench.barriers_per_call(arm) * len(rows),
                      output_clones=bench.clone_per_call(arm)*len(rows),
                      projection_cpu_median_us=statistics.median(r['cpu_us'] for r in rows),
                      output_slice_view_count=sum(r['output_slice_view_count'] for r in rows),
                      adapter_view_count=sum(r['adapter_view_count'] for r in rows),
                      publication_barrier_median_us=statistics.median(r['publication_barrier_us'] for r in rows),
                      device_path_median_us=statistics.median(r['device_path_us'] for r in rows),
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
                profiles[f"{job['arm']}/rank{rank}"] = trace_projection_records(json.load(handle)["traceEvents"], job["arm"], tuning=plan["tuning"])
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
    p.add_argument("--trials", type=int, choices=(1,2), default=1)
    p.add_argument("--blocks", type=int, default=8)
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

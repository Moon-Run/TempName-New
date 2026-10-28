#!/usr/bin/env python3
"""Independent A/decoder/lifecycle/combined validation; operator timings are not full steps.

No old calibration hardware identity is reused. A fresh experiment records the
actual eight H100 devices, frozen libraries, inputs, source and all rank results.
Preparation and the default run are CPU-only; --execute is required for workers.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PROTOCOL = "h100-arrival-op4-independent-operator-v1"
ARMS = ("A", "D", "L", "F")
ROLES = {"attn": [2048, 4096, 512], "mlp": [2048, 4096, 2048]}
SOURCE_PROTOCOL = "h100-arrival-op4"
COMPARISONS = (("F", "A"), ("L", "A"), ("D", "A"), ("F", "D"), ("F", "L"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def schedule(blocks):
    require(type(blocks) is int and blocks >= 4 and blocks % 4 == 0,
            "Use a positive multiple of four complete blocks")
    # Opposite rotations form pairs; each pair plus its reverse balances positions.
    rotations = (0, 2, 1, 3)
    first = [list(ARMS[rotations[b % 4]:] + ARMS[:rotations[b % 4]])
             for b in range(blocks // 2)]
    orders = first + [row[::-1] for row in first[::-1]]
    return [dict(block=b, arm=arm, index=b * len(ARMS) + i,
                 directory=f"block{b:02d}-{arm}")
            for b, order in enumerate(orders) for i, arm in enumerate(order)]


def libraries(build):
    result = {}
    for arm in ARMS:
        root = (build / arm).resolve()
        require((root / "python/flux").is_dir(), f"Missing Flux package: {root}")
        paths = {}
        for name in ("libflux_cuda.so", "libflux_cuda_ths_op.so"):
            choices = (root / name, root / "python/flux/lib" / name, root / "build/lib" / name)
            path = next((p.resolve() for p in choices if p.is_file()), None)
            require(path is not None, f"Missing compiled {arm}/{name}")
            paths[str(path)] = sha(path)
        bindings = sorted((root / "python").glob("flux_ths_pybind*.so"))
        require(bindings, f"Missing Python extension: {root}")
        paths.update({str(p.resolve()): sha(p) for p in bindings})
        result[arm] = dict(root=str(root), files=paths)
    return result


def build_identity(build, builder_root=None):
    """Only the registered, executable source transformation can be measured."""
    if builder_root is not None:
        sys.path.insert(1, str(builder_root))
    import arrival_op4_build as builder
    record = builder.verify(build, built=True)
    manifest = read(build / "manifest.json")
    require(not manifest.get("source_only") and not manifest.get("diagnostic_only")
            and not record.get("baseline_source_only"), "Diagnostic/source-only builds cannot be measured")
    source_files = {str((build / arm / name).resolve()): digest
                    for arm, variant in manifest["variants"].items()
                    for name, digest in variant["source_sha256"].items()}
    return dict(builder_root=str(Path(builder.__file__).resolve().parent),
                manifest_sha256=sha(build / "manifest.json"),
                op4_manifest_sha256=sha(build / builder.MANIFEST),
                source_files=source_files, builder_inputs=record["inputs"])


def prepare(args):
    # Source preparation is valid before compilation; runtime hashes are frozen
    # only once the requested A/D/L/F libraries actually exist.
    root, build = args.root.resolve(), args.build.resolve()
    provenance = build_identity(build)
    identities = libraries(build)
    require(args.warmup >= 1 and args.iterations >= 2 and args.reuse >= 2,
            "Positive warmup, at least two timed calls and reuse cycles required")
    require(len(args.seeds) >= 2 and len(set(args.seeds)) == len(args.seeds),
            "At least two distinct correctness seeds required")
    root.mkdir(parents=True, exist_ok=False)
    (root / "scripts").mkdir()
    (root / "results").mkdir()
    shutil.copy2(__file__, root / "scripts/bench.py")
    source = (REPO / "tools/a800/phase5/e2e/taco_support.py").read_text()
    before = source[:source.index("\nclass TacoForward:")]
    before = "\n".join(line for line in before.splitlines()
                       if not line.startswith("from routing import")) + "\n"
    reference = before + source[source.index("\n@torch.no_grad()\ndef check_taco_output"):]
    (root / "scripts/codec_reference.py").write_text(reference)
    shutil.copy2(REPO / "tools/a800/phase4/reference.py", root / "scripts/reference.py")
    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                kind="operator_microbenchmark", build=str(build), libraries=identities,
                build_identity=provenance,
                roles=ROLES, arms=list(ARMS), schedule=schedule(args.blocks),
                warmup=args.warmup, iterations=args.iterations, seeds=args.seeds,
                reuse=args.reuse, retained_output_calls=6,
                slow_rank_stress={"ranks": [0, 7], "cycles": 2000000,
                                  "scope": "First seed of correctness epochs 0 and 1 only; never timed."},
                communication={"NCCL_NVLS_ENABLE": "0", "NCCL_DEBUG": "WARN",
                               "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"},
                shared_environment={"CUDA_DEVICE_MAX_CONNECTIONS": "8",
                                    "FLUX_GEMM_RS_GEMM_ONLY": "1"},
                source_files={str(p.relative_to(root)): sha(p)
                              for p in (root / "scripts").iterdir()},
                target_percent=1.0, target_validated=False,
                limitation="A new same-host A/D/L/F operator experiment, not historical paired data or full training steps.")
    write(root / "plan.json", plan)
    print(root)


def verify(root, offline=False):
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    require(plan["protocol"] == PROTOCOL and plan["arms"] == list(ARMS), "Wrong op4 plan")
    require(plan["roles"] == ROLES, "Role shapes changed")
    for name, digest in plan["source_files"].items():
        require(sha(root / name) == digest, f"Frozen source changed: {name}")
    require(plan["schedule"] == schedule(len(plan["schedule"]) // len(ARMS)),
            "Incomplete or changed balanced schedule")
    require(plan["retained_output_calls"] >= 6, "Insufficient asynchronous workspace reuse")
    if not offline:
        require(sha(Path(__file__)) == plan["source_files"]["scripts/bench.py"],
                "Operator controller differs from its frozen worker")
        require(build_identity(Path(plan["build"]), plan["build_identity"]["builder_root"])
                == plan["build_identity"], "Frozen build or frontend changed")
        require(libraries(Path(plan["build"])) == plan["libraries"], "Frozen libraries changed")
    return plan


def environment(root, plan, arm, python, cuda_home, profile=False):
    library = Path(plan["libraries"][arm]["root"])
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("NCCL_", "TORCH_NCCL_", "H100_OP3_", "H100_OP4_")):
            env.pop(key)
    env.update(plan["communication"])
    env.update(plan["shared_environment"])
    env.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1",
               PYTHONPATH=f"{root}/scripts:{library}/python",
               H100_OP4_ARM=arm,
               H100_OP4_PROTOCOL="h100-arrival-op4",
               H100_OP4_BENCH_EXECUTE="1")
    for key in ("FLUX_FORCE_NVLINK", "FLUX_TUNE_CONFIG_FILE", "PYTHONOPTIMIZE"):
        env.pop(key, None)
    env["PATH"] = str(python.parent) + ":" + env.get("PATH", "")
    ld = [str(library / "python/flux/lib"), str(library), str(library / "build/lib")]
    if cuda_home:
        env.update(CUDA_HOME=str(cuda_home))
        env["PATH"] = str(cuda_home / "bin") + ":" + env["PATH"]
        ld.append(str(cuda_home / "lib64"))
    env["LD_LIBRARY_PATH"] = ":".join(ld) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def run_worker(command, cwd, env, log, timeout):
    """Own the torchrun process group, including ranks, and reap it on failure."""
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
        return code
    except BaseException:
        # start_new_session makes this child's PID its private process-group ID.
        # Never enumerate or signal unrelated users' jobs or existing allocations.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # torchrun may exit before a rank; reap remaining members of OUR group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        raise


def run(args):
    root = args.root.resolve()
    plan = verify(root)
    jobs = ([dict(arm=arm, block=-1, index=i, directory=f"profile-{arm}") for i, arm in enumerate(ARMS)]
            if args.profile else plan["schedule"])
    python = args.python.expanduser().absolute()  # Keep a venv launcher symlink intact.
    prefix = [str(python), "-m", "torch.distributed.run",
              "--standalone", "--nproc_per_node=8", str(root / "scripts/bench.py"),
              "worker", str(root)]
    if not args.execute:
        print(json.dumps(dict(dry_run=True, jobs=jobs, command=prefix,
                              target_validated=False), indent=2))
        return
    marker = root / ("profile-started.json" if args.profile else "timing-started.json")
    write(marker, dict(run_id=plan["run_id"], jobs=jobs, status="started"))
    for job in jobs:
        verify(root)
        dest = root / "results" / job["directory"]
        dest.mkdir(exist_ok=False)
        command = prefix + ["--job", json.dumps(job, sort_keys=True)]
        if args.profile:
            command.append("--profile")
        env = environment(root, plan, job["arm"], python, args.cuda_home, args.profile)
        write(dest / "command.json", dict(command=command, arm=job["arm"],
                                         op4_environment={k: v for k, v in env.items() if k.startswith("H100_OP4_")}))
        print("START", job["directory"], flush=True)
        try:
            with (dest / "process.log").open("x") as log:
                run_worker(command, root, env, log, args.worker_timeout)
        except BaseException as error:
            write(dest / "failed.json", dict(status="failed", retry=False,
                exception=type(error).__name__, message=str(error), job=job))
            raise
    if not args.profile:
        report(root)
    else:
        write(root / "profile-summary.json", profile_summary(root))


def correctness_delay(plan, epoch, seed):
    pressure = plan["slow_rank_stress"]
    if epoch < len(pressure["ranks"]) and seed == plan["seeds"][0]:
        return pressure["ranks"][epoch], pressure["cycles"]
    return -1, 0


def owned_forward(op, inputs, weights):
    """The measured API always returns storage retained across subsequent calls."""
    output = op.forward(inputs, weights)
    return output if getattr(op, "owns_output", False) else output.clone()


def worker(args):
    require(os.environ.get("H100_OP4_BENCH_EXECUTE") == "1", "Use run --execute")
    root = args.root.resolve()
    plan = verify(root)
    job = json.loads(args.job)
    require(job in ([dict(arm=a, block=-1, index=i, directory=f"profile-{a}") for i, a in enumerate(ARMS)]
                   if args.profile else plan["schedule"]), "Unregistered worker job")
    expected_environment = dict(plan["shared_environment"], **plan["communication"],
        H100_OP4_ARM=job["arm"], H100_OP4_PROTOCOL=SOURCE_PROTOCOL)
    require(all(os.environ.get(k) == v for k, v in expected_environment.items()),
            "Worker arm/protocol/shared environment differs from frozen plan")
    import torch
    import torch.distributed as dist
    from codec_reference import check_taco_output
    rank, local, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    require(world == 8 and rank == local and torch.cuda.device_count() == 8, "One complete local TP8 group required")
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    devices = []
    for index in range(world):
        p = torch.cuda.get_device_properties(index)
        require("H100" in p.name and (p.major, p.minor) == (9, 0) and p.multi_processor_count == 132,
                "This benchmark requires eight full 132-SM H100s; partition names do not identify hardware")
        devices.append(dict(name=p.name, uuid=str(p.uuid), sm_count=p.multi_processor_count))
    require(all(torch.cuda.can_device_access_peer(i, j) for i in range(world) for j in range(world) if i != j),
            "All local GPU pairs must support peer access")
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    group = dist.new_group(list(range(world)))
    import flux
    from flux.gemm_rs_taco import GemmRSTaco
    runtime = Path(plan["libraries"][job["arm"]]["root"])
    require(Path(flux.__file__).resolve().is_relative_to(runtime), "Wrong Flux Python package loaded")
    # /proc maps catches a loader choosing a different SONAME through a preexisting
    # LD_LIBRARY_PATH even if flux.__file__ itself has the expected package root.
    maps = Path("/proc/self/maps").read_text()
    for path, digest in plan["libraries"][job["arm"]]["files"].items():
        require(sha(path) == digest and str(Path(path).resolve()) in maps,
                f"Frozen library is not the loaded library: {path}")
    dest = root / "results" / job["directory"]
    result = dict(protocol=PROTOCOL, run_id=plan["run_id"], plan_sha256=sha(root / "plan.json"),
                  job=job, rank=rank, hardware=dict(host=os.uname().nodename, devices=devices),
                  software=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                nccl=list(torch.cuda.nccl.version()), python=sys.version),
                  profiling=args.profile, sampling=False, roles={}, completed=False,
                  execution_environment=expected_environment,
                  target_validated=False)

    def tensor_hash(value):
        return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    with torch.no_grad():
        for role_index, (role, shape) in enumerate(plan["roles"].items()):
            m, n, k = shape
            op = GemmRSTaco(group, m, n, k, placement="fused")
            owns_output = bool(getattr(op, "owns_output", False))
            require(owns_output == (job["arm"] in ("L", "F")), "Wrong output ownership")
            slots = getattr(op, "slots", [op])
            packet_pointers = [slot.peers[rank].data_ptr() for slot in slots]
            require(len(slots) == (2 if owns_output else 1) and
                    len(set(packet_pointers)) == len(slots), "Wrong workspace multiplicity/alias")
            checks = []
            for epoch in range(plan["reuse"]):
                for seed in plan["seeds"]:
                    # New input AND weight every epoch, same schedule across all
                    # arms. This catches stale packet contents across workspace reuse.
                    actual_seed = seed + epoch * 100000 + role_index * 10000 + rank
                    torch.manual_seed(actual_seed)
                    inputs = (torch.randn(m, k, device="cuda") * .02).to(torch.bfloat16)
                    weights = (torch.randn(n, k, device="cuda") * .02).to(torch.bfloat16)
                    delayed_rank, delay_cycles = correctness_delay(plan, epoch, seed)
                    if rank == delayed_rank:
                        torch.cuda._sleep(delay_cycles)
                    actual = owned_forward(op, inputs, weights)
                    correctness = check_taco_output(inputs @ weights.t(), actual, group, mask=None)
                    repeated = owned_forward(op, inputs, weights)
                    require(torch.equal(actual, repeated), "Repeated output differs; possible publication/reuse race")
                    checks.append(dict(epoch=epoch, seed=seed,
                                       delayed_rank=delayed_rank, delay_cycles=delay_cycles,
                                       inputs_sha256=tensor_hash(inputs), weights_sha256=tensor_hash(weights),
                                       output_sha256=tensor_hash(actual), correctness=correctness))
            # No reference collective, hash, or synchronize between these calls.
            # Six generations wrap both slots repeatedly and keep every output live.
            retained = []
            for generation in range(plan["retained_output_calls"]):
                torch.manual_seed(4711 + generation * 100000 + role_index * 10000 + rank)
                burst_input = (torch.randn(m, k, device="cuda") * .02).to(torch.bfloat16)
                burst_weight = (torch.randn(n, k, device="cuda") * .02).to(torch.bfloat16)
                if rank == (0 if generation % 2 == 0 else 7):
                    torch.cuda._sleep(plan["slow_rank_stress"]["cycles"])
                retained.append((burst_input, burst_weight, owned_forward(op, burst_input, burst_weight)))
            torch.cuda.synchronize()
            require(len({item[2].data_ptr() for item in retained}) == len(retained),
                    "Independent retained outputs alias across workspace generations")
            burst_checks = []
            for generation, (burst_input, burst_weight, burst_output) in enumerate(retained):
                check = check_taco_output(burst_input @ burst_weight.t(), burst_output, group, mask=None)
                require(check["passed"], "Unsynchronized reuse burst failed codec validation")
                burst_checks.append(dict(generation=generation,
                    inputs_sha256=tensor_hash(burst_input), weights_sha256=tensor_hash(burst_weight),
                    output_sha256=tensor_hash(burst_output), correctness=check))
            del retained, burst_input, burst_weight, burst_output
            for _ in range(plan["warmup"]):
                output = owned_forward(op, inputs, weights)
            dist.barrier(group=group)
            torch.cuda.synchronize()
            if args.profile:
                diagnostics = []
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    for invocation in range(4):
                        with torch.profiler.record_function(f"op4.{role}.{invocation}"):
                            output = owned_forward(op, inputs, weights)
                        torch.cuda.synchronize()
                trace = dest / f"rank{rank}-{role}-profile.json"
                profiler.export_chrome_trace(str(trace))
                with trace.open("rb") as src, gzip.open(str(trace) + ".gz", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                trace.unlink()
                timing = None
            else:
                diagnostics = []
                started = time.perf_counter()
                for _ in range(plan["iterations"]):
                    output = owned_forward(op, inputs, weights)
                torch.cuda.synchronize()
                timing = dict(wall_us=(time.perf_counter() - started) * 1e6 / plan["iterations"],
                              iterations=plan["iterations"], warmup=plan["warmup"],
                              includes="forward with independent output ownership and final synchronization")
            require(tensor_hash(output) == checks[-1]["output_sha256"], "Final output changed")
            result["roles"][role] = dict(shape=shape, checks=checks, timing=timing,
                diagnostics=diagnostics, retained_output_checks=burst_checks,
                owns_output=owns_output, workspace_count=len(slots), packet_pointers=packet_pointers)
            dist.barrier(group=group)
            del op, slots, inputs, weights, actual, repeated, output
    result["completed"] = True
    write(dest / f"rank{rank}.json", result)
    dist.destroy_process_group()


def paired(candidate, baseline):
    require(len(candidate) == len(baseline) and len(candidate) >= 2, "At least two paired blocks required")
    logs = [math.log(x / y) for x, y in zip(candidate, baseline)]
    rng = random.Random(20261020)
    samples = sorted(100 * (1 - math.exp(statistics.mean(rng.choices(logs, k=len(logs)))))
                     for _ in range(10000))
    return dict(percent=100 * (1 - math.exp(statistics.mean(logs))),
                ci95=[samples[249], samples[9749]], paired_blocks=len(logs))


def savings_estimate(samples):
    """Paired micro estimate for diagnostics only; never a training prerequisite."""
    blocks = sorted({row["block"] for row in samples})
    paired_rows = {}
    for block in blocks:
        rows = [row for row in samples if row["block"] == block]
        require(len(rows) == len(ARMS) and {row["arm"] for row in rows} == set(ARMS),
                "Micro estimate requires complete unique A/D/L/F blocks")
        paired_rows[block] = {row["arm"]: row["rankmax_wall_us"] for row in rows}
    require(len(blocks) >= 2, "Micro estimate requires at least two paired blocks")
    for rows in paired_rows.values():
        require(all(set(values) == set(ROLES) and
                    all(math.isfinite(value) and value > 0 for value in values.values())
                    for values in rows.values()), "Invalid micro estimate role timing")
    role_savings = {
        role: statistics.mean(rows["A"][role] - rows["F"][role] for rows in paired_rows.values())
        for role in ROLES}
    role_reductions = {
        role: 100 * (1 - math.exp(statistics.mean(
            math.log(rows["F"][role] / rows["A"][role]) for rows in paired_rows.values())))
        for role in ROLES}
    savings = [128 * sum(rows["A"][role] - rows["F"][role] for role in ROLES) / 1000
               for rows in paired_rows.values()]
    rng = random.Random(20261020)
    boot = sorted(statistics.mean(rng.choices(savings, k=len(savings))) for _ in range(10000))
    mean_ms, ci95 = statistics.mean(savings), [boot[249], boot[9749]]
    return dict(mean_ms=mean_ms, ci95_ms=ci95,
                paired_blocks=len(blocks), block_savings_ms=savings, role_mean_saving_us=role_savings,
                role_paired_reduction_percent=role_reductions,
                role_calls_per_optimizer_step=128,
                formula="128 * ((A_attn_us - F_attn_us) + (A_mlp_us - F_mlp_us)) / 1000",
                limitation="Diagnostic estimate only, not measured optimizer-step savings, "
                           "a training prerequisite or 1% acceptance.")


def report(root, save=True, offline=False):
    root = Path(root).resolve()
    plan = verify(root, offline)
    identity, expected_checks, samples = None, {}, []
    for job in plan["schedule"]:
        rows = [read(root / "results" / job["directory"] / f"rank{rank}.json") for rank in range(8)]
        for rank, row in enumerate(rows):
            require(row["completed"] and row["rank"] == rank and row["job"] == job
                    and row["protocol"] == PROTOCOL and row["run_id"] == plan["run_id"]
                    and row["plan_sha256"] == sha(root / "plan.json") and not row["profiling"],
                    "Incomplete or mismatched rank evidence")
            expected_environment = dict(plan["shared_environment"], **plan["communication"],
                H100_OP4_ARM=job["arm"], H100_OP4_PROTOCOL=SOURCE_PROTOCOL)
            require(row["execution_environment"] == expected_environment,
                    "Timed worker environment differs from plan")
            current = dict(hardware=row["hardware"], software=row["software"])
            identity = current if identity is None else identity
            require(identity == current, "Mixed hardware or software across A/D/L/F")
            for role, record in row["roles"].items():
                require(role in ROLES and record["shape"] == ROLES[role] and
                        len(record["checks"]) == len(plan["seeds"]) * plan["reuse"],
                        "Missing role or reuse correctness checks")
                require(all(c["correctness"]["passed"] for c in record["checks"]), "Codec correctness failed")
                owned = job["arm"] in ("L", "F")
                require(record["owns_output"] is owned and record["workspace_count"] == (2 if owned else 1)
                        and len(set(record["packet_pointers"])) == record["workspace_count"],
                        "Wrong output ownership/workspace evidence")
                burst = record["retained_output_checks"]
                require([c["generation"] for c in burst] == list(range(plan["retained_output_calls"]))
                        and all(c["correctness"]["passed"] for c in burst), "Missing retained-output validation")
                burst_key = (rank, role, "retained")
                burst_hashes = [{k: c[k] for k in
                    ("generation", "inputs_sha256", "weights_sha256", "output_sha256")} for c in burst]
                expected_checks.setdefault(burst_key, burst_hashes)
                require(expected_checks[burst_key] == burst_hashes, "A/D/L/F retained output differs bitwise")
                require([(c["epoch"], c["seed"]) for c in record["checks"]] ==
                        [(epoch, seed) for epoch in range(plan["reuse"]) for seed in plan["seeds"]],
                        "Missing or duplicated changing-input reuse epoch")
                require(all((c["delayed_rank"], c["delay_cycles"]) ==
                            correctness_delay(plan, c["epoch"], c["seed"]) for c in record["checks"]),
                        "Cross-rank delay stress differs from frozen correctness plan")
                hashes = [{k: c[k] for k in ("epoch", "seed", "delayed_rank", "delay_cycles",
                                             "inputs_sha256", "weights_sha256", "output_sha256")}
                          for c in record["checks"]]
                key = (rank, role)
                if key not in expected_checks:
                    expected_checks[key] = hashes
                require(hashes == expected_checks[key], "A/D/L/F input or output differs bitwise")
                timing = record["timing"]
                require(timing["iterations"] == plan["iterations"] and timing["warmup"] == plan["warmup"]
                        and math.isfinite(timing["wall_us"]) and timing["wall_us"] > 0, "Invalid timing")
        require(all(set(row["roles"]) == set(ROLES) for row in rows), "Missing worker role")
        samples.append(dict(job, rankmax_wall_us={role: max(r["roles"][role]["timing"]["wall_us"] for r in rows)
                                                 for role in ROLES}))
    roles = {}
    for role in ROLES:
        values = {arm: [r["rankmax_wall_us"][role] for r in sorted(samples, key=lambda x: x["block"])
                        if r["arm"] == arm] for arm in ARMS}
        roles[role] = dict(mean_us={arm: statistics.mean(v) for arm, v in values.items()},
                           comparisons={c + "/" + b: paired(values[c], values[b])
                                        for c, b in COMPARISONS})
    result = dict(protocol=PROTOCOL, run_id=plan["run_id"], completed=True,
                  correctness_passed=True, bitwise_equal_to_A=True, measurement_identity=identity,
                  slow_rank_stress=plan["slow_rank_stress"],
                  roles=roles, all_samples=samples, micro_savings_estimate=savings_estimate(samples),
                  target_percent=1.0, target_validated=False,
                  acceptance_status="full_step_measurement_required",
                  bootstrap=dict(unit="complete fresh-process A/D/L/F block", samples=10000, seed=20261020),
                  limitation="Operator microbenchmark only. No optimizer steps were measured; no 1% training speedup is claimed.",
                  evidence_files={str(p.relative_to(root)): sha(p) for p in (root / "results").glob("block*/rank*.json")})
    if save:
        path = root / "report.json"
        if path.exists():
            require(read(path) == result, "Saved operator report differs from raw evidence")
        else:
            write(path, result)
        print(json.dumps(dict(roles=result["roles"], micro_savings_estimate=result["micro_savings_estimate"]), indent=2))
    return result


def trace_path(events, arm, calls=4):
    """Check the executed serial path; these durations never count as speedup."""
    require(arm in ARMS and calls > 0, "Unknown execution path")
    owned = arm in ("L", "F")
    kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    gemms = [e for e in kernels if "flux_bf16" in e["name"] and "reducescatter" in e["name"]]
    decoder_name = "taco_op3_v2_decode" if arm in ("D", "F") else "taco_decode_ring_kernel"
    decoders = [e for e in kernels if decoder_name+"(" in e["name"]]
    barriers = [e for e in kernels if "CudaIpcBarrierAllKernel" in e["name"]]
    clones = [e for e in events if e.get("cat") == "cpu_op" and e.get("name") == "aten::clone"]
    require(len(gemms) == len(decoders) == calls and len(barriers) == calls * (1 if owned else 2),
            "Wrong GEMM/decoder/barrier counts")
    require(len(clones) == (0 if owned else calls), "Wrong output clone count")
    require(not any(any(x in e["name"] for x in ("try_decode", "tail_decode", "op3_v2_stop"))
                    for e in kernels), "Arrival protocol unexpectedly executed")
    for i, (gemm, decoder) in enumerate(zip(gemms, decoders)):
        group = barriers[i:i+1] if owned else barriers[i*2:i*2+2]
        sequence = [gemm, group[0], decoder] + ([] if owned else [group[1]])
        require(all(e.get("args", {}).get("stream") is not None for e in sequence) and
                len({e["args"]["stream"] for e in sequence}) == 1 and
                all(a["ts"] + a["dur"] <= b["ts"] + .01 for a, b in zip(sequence, sequence[1:])),
                "Publication/reduction/reuse order differs from the registered path")
    return dict(calls=calls, gemm_count=len(gemms), decoder_count=len(decoders),
                barrier_count=len(barriers), clone_count=len(clones), decoder=decoder_name,
                execution_path_validated=True)


def profile_summary(root, offline=False):
    root = Path(root).resolve()
    plan = verify(root, offline)
    profiles, evidence = {}, {}
    for index, arm in enumerate(ARMS):
        for rank in range(8):
            folder = root / "results" / f"profile-{arm}"
            row_path = folder / f"rank{rank}.json"
            row = read(row_path)
            require(row["completed"] and row["rank"] == rank and row["profiling"] and
                    row["protocol"] == PROTOCOL and row["run_id"] == plan["run_id"] and
                    row["plan_sha256"] == sha(root / "plan.json") and
                    row["job"] == dict(arm=arm, block=-1, index=index, directory=f"profile-{arm}"),
                    "Mismatched profile rank evidence")
            expected = dict(plan["shared_environment"], **plan["communication"],
                            H100_OP4_ARM=arm, H100_OP4_PROTOCOL=SOURCE_PROTOCOL)
            require(row["execution_environment"] == expected, "Profile environment differs")
            evidence[str(row_path.relative_to(root))] = sha(row_path)
            for role in ROLES:
                path = folder / f"rank{rank}-{role}-profile.json.gz"
                with gzip.open(path, "rt") as handle:
                    result = trace_path(json.load(handle)["traceEvents"], arm)
                profiles[f"{arm}/rank{rank}/{role}"] = dict(result,
                    trace=str(path.relative_to(root)), sha256=sha(path))
                evidence[str(path.relative_to(root))] = sha(path)
    return dict(protocol=PROTOCOL, run_id=plan["run_id"], completed=True, profiles=profiles,
                execution_path_validated=True, target_validated=False, evidence_files=evidence,
                limitation="Separate profiles verify the path and cannot establish unprofiled training speedup.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[817, 919, 1231])
    p.add_argument("--reuse", type=int, default=4)
    p = sub.add_parser("run")
    p.add_argument("root", type=Path)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--cuda-home", type=Path)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--worker-timeout", type=int, default=1800)
    p = sub.add_parser("worker")
    p.add_argument("root", type=Path)
    p.add_argument("--job", required=True)
    p.add_argument("--profile", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("root", type=Path)
    p.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.action == "report":
        report(args.root, offline=args.offline)
    else:
        globals()[args.action](args)


if __name__ == "__main__":
    main()

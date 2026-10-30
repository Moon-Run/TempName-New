#!/usr/bin/env python3
"""Independent A/serial/overlap validation; operator timings are not full steps.

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
PROTOCOL = "h100-arrival-op3-independent-operator-v1"
ARMS = ("A", "S", "O")
ROLES = {"attn": [2048, 4096, 512], "mlp": [2048, 4096, 2048]}


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
    require(blocks >= 2 and blocks % 2 == 0, "Use an even number of complete blocks, at least two")
    rng = random.Random(20261020)
    first = []
    for block in range(blocks // 2):
        order = list(ARMS)
        rng.shuffle(order)
        first.append(order)
    orders = first + [row[::-1] for row in first[::-1]]
    return [dict(block=b, arm=arm, index=b * 3 + i,
                 directory=f"block{b:02d}-{arm}")
            for b, order in enumerate(orders) for i, arm in enumerate(order)]


def libraries(build):
    result = {}
    for arm in ("A", "O"):
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
    import arrival_op3_build as builder
    record = builder.verify(build, built=True)
    manifest = read(build / "manifest.json")
    require(not manifest.get("source_only") and not manifest.get("diagnostic_only")
            and not record.get("baseline_source_only"), "Diagnostic/source-only builds cannot be measured")
    source_files = {str((build / arm / name).resolve()): digest
                    for arm, variant in manifest["variants"].items()
                    for name, digest in variant["source_sha256"].items()}
    return dict(builder_root=str(Path(builder.__file__).resolve().parent),
                manifest_sha256=sha(build / "manifest.json"),
                op3_manifest_sha256=sha(build / builder.MANIFEST),
                source_files=source_files, builder_inputs=record["inputs"])


def prepare(args):
    # Source preparation is valid before compilation; runtime hashes are frozen
    # only once the requested A/O libraries actually exist.
    root, build = args.root.resolve(), args.build.resolve()
    provenance = build_identity(build)
    identities = libraries(build)
    require(args.warmup >= 1 and args.iterations >= 2 and args.reuse >= 2,
            "Positive warmup, at least two timed calls and reuse cycles required")
    require(len(args.seeds) >= 2 and len(set(args.seeds)) == len(args.seeds),
            "At least two distinct correctness seeds required")
    require(1 <= args.rounds <= 4, "Overlap rounds must be between one and four")
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
                reuse=args.reuse, overlap_rounds=args.rounds,
                slow_rank_stress={"ranks": [0, 7], "cycles": 2000000,
                                  "scope": "First seed of correctness epochs 0 and 1 only; never timed."},
                communication={"NCCL_NVLS_ENABLE": "0", "NCCL_DEBUG": "WARN",
                               "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"},
                shared_environment={"CUDA_DEVICE_MAX_CONNECTIONS": "8",
                                    "FLUX_GEMM_RS_GEMM_ONLY": "1"},
                source_files={str(p.relative_to(root)): sha(p)
                              for p in (root / "scripts").iterdir()},
                target_percent=1.0, target_validated=False,
                limitation="A new same-host A/S/O operator experiment, not historical paired data or full training steps.")
    write(root / "plan.json", plan)
    print(root)


def verify(root):
    plan = read(root / "plan.json")
    require(plan["protocol"] == PROTOCOL and plan["arms"] == list(ARMS), "Wrong op3 plan")
    require(plan["roles"] == ROLES, "Role shapes changed")
    for name, digest in plan["source_files"].items():
        require(sha(root / name) == digest, f"Frozen source changed: {name}")
    require(build_identity(Path(plan["build"]), plan["build_identity"]["builder_root"])
            == plan["build_identity"], "Frozen build or frontend changed")
    require(libraries(Path(plan["build"])) == plan["libraries"], "Frozen libraries changed")
    return plan


def environment(root, plan, arm, python, cuda_home, profile=False):
    library = Path(plan["libraries"]["A" if arm == "A" else "O"]["root"])
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("NCCL_", "TORCH_NCCL_", "H100_OP3_")):
            env.pop(key)
    env.update(plan["communication"])
    env.update(plan["shared_environment"])
    env.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1",
               PYTHONPATH=f"{root}/scripts:{library}/python",
               H100_OP3_MODE="serial" if arm == "S" else "overlap",
               H100_OP3_ROUNDS=str(plan["overlap_rounds"]),
               H100_OP3_DIAGNOSTICS="1" if profile and arm != "A" else "0",
               H100_OP3_BENCH_EXECUTE="1")
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
    prefix = [str(args.python.resolve()), "-m", "torch.distributed.run",
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
        env = environment(root, plan, job["arm"], args.python.resolve(), args.cuda_home, args.profile)
        write(dest / "command.json", dict(command=command, arm=job["arm"],
                                         op3_environment={k: v for k, v in env.items() if k.startswith("H100_OP3_")}))
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


def worker(args):
    require(os.environ.get("H100_OP3_BENCH_EXECUTE") == "1", "Use run --execute")
    root = args.root.resolve()
    plan = verify(root)
    job = json.loads(args.job)
    require(job in ([dict(arm=a, block=-1, index=i, directory=f"profile-{a}") for i, a in enumerate(ARMS)]
                   if args.profile else plan["schedule"]), "Unregistered worker job")
    expected_environment = dict(plan["shared_environment"], **plan["communication"],
        H100_OP3_MODE="serial" if job["arm"] == "S" else "overlap",
        H100_OP3_ROUNDS=str(plan["overlap_rounds"]),
        H100_OP3_DIAGNOSTICS="1" if args.profile and job["arm"] != "A" else "0")
    require(all(os.environ.get(k) == v for k, v in expected_environment.items()),
            "Worker mode/rounds/diagnostics/shared environment differs from frozen plan")
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
    runtime = Path(plan["libraries"]["A" if job["arm"] == "A" else "O"]["root"])
    require(Path(flux.__file__).resolve().is_relative_to(runtime), "Wrong Flux Python package loaded")
    # /proc maps catches a loader choosing a different SONAME through a preexisting
    # LD_LIBRARY_PATH even if flux.__file__ itself has the expected package root.
    maps = Path("/proc/self/maps").read_text()
    for path, digest in plan["libraries"]["A" if job["arm"] == "A" else "O"]["files"].items():
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
            checks = []
            for epoch in range(plan["reuse"]):
                for seed in plan["seeds"]:
                    # New input AND weight every epoch, same schedule across all
                    # arms. This catches stale ready flags across workspace reuse.
                    actual_seed = seed + epoch * 100000 + role_index * 10000 + rank
                    torch.manual_seed(actual_seed)
                    inputs = (torch.randn(m, k, device="cuda") * .02).to(torch.bfloat16)
                    weights = (torch.randn(n, k, device="cuda") * .02).to(torch.bfloat16)
                    delayed_rank, delay_cycles = correctness_delay(plan, epoch, seed)
                    if rank == delayed_rank:
                        torch.cuda._sleep(delay_cycles)
                    actual = op.forward(inputs, weights).clone()
                    correctness = check_taco_output(inputs @ weights.t(), actual, group, mask=None)
                    repeated = op.forward(inputs, weights).clone()
                    require(torch.equal(actual, repeated), "Repeated output differs; possible publication/reuse race")
                    checks.append(dict(epoch=epoch, seed=seed,
                                       delayed_rank=delayed_rank, delay_cycles=delay_cycles,
                                       inputs_sha256=tensor_hash(inputs), weights_sha256=tensor_hash(weights),
                                       output_sha256=tensor_hash(actual), correctness=correctness))
            for _ in range(plan["warmup"]):
                output = op.forward(inputs, weights).clone()
            dist.barrier(group=group)
            torch.cuda.synchronize()
            if args.profile:
                diagnostics = []
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    for invocation in range(4):
                        with torch.profiler.record_function(f"op3.{role}.{invocation}"):
                            output = op.forward(inputs, weights).clone()
                        torch.cuda.synchronize()
                        if job["arm"] != "A":
                            require(callable(getattr(op, "op3_diagnostics", None)),
                                    "O library must expose op.op3_diagnostics() for useful-consumption validation")
                            diagnostic = op.op3_diagnostics()
                            require(isinstance(diagnostic, dict), "Wrong op3 diagnostics type")
                            diagnostics.append(dict(invocation=invocation, **diagnostic))
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
                    output = op.forward(inputs, weights).clone()
                torch.cuda.synchronize()
                timing = dict(wall_us=(time.perf_counter() - started) * 1e6 / plan["iterations"],
                              iterations=plan["iterations"], warmup=plan["warmup"],
                              includes="forward().clone() and final synchronization")
            require(tensor_hash(output) == checks[-1]["output_sha256"], "Final output changed")
            result["roles"][role] = dict(shape=shape, checks=checks, timing=timing,
                                        diagnostics=diagnostics)
            dist.barrier(group=group)
            del op, inputs, weights, actual, repeated, output
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


def report(root):
    root = Path(root).resolve()
    plan = verify(root)
    identity, expected_checks, samples = None, {}, []
    for job in plan["schedule"]:
        rows = [read(root / "results" / job["directory"] / f"rank{rank}.json") for rank in range(8)]
        for rank, row in enumerate(rows):
            require(row["completed"] and row["rank"] == rank and row["job"] == job
                    and row["protocol"] == PROTOCOL and row["run_id"] == plan["run_id"]
                    and row["plan_sha256"] == sha(root / "plan.json") and not row["profiling"],
                    "Incomplete or mismatched rank evidence")
            expected_environment = dict(plan["shared_environment"], **plan["communication"],
                H100_OP3_MODE="serial" if job["arm"] == "S" else "overlap",
                H100_OP3_ROUNDS=str(plan["overlap_rounds"]), H100_OP3_DIAGNOSTICS="0")
            require(row["execution_environment"] == expected_environment,
                    "Timed worker environment differs from plan")
            current = dict(hardware=row["hardware"], software=row["software"])
            identity = current if identity is None else identity
            require(identity == current, "Mixed hardware or software across A/S/O")
            for role, record in row["roles"].items():
                require(role in ROLES and record["shape"] == ROLES[role] and
                        len(record["checks"]) == len(plan["seeds"]) * plan["reuse"],
                        "Missing role or reuse correctness checks")
                require(all(c["correctness"]["passed"] for c in record["checks"]), "Codec correctness failed")
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
                require(hashes == expected_checks[key], "A/S/O input or output differs bitwise")
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
                                        for c, b in (("O", "A"), ("S", "A"), ("O", "S"))})
    result = dict(protocol=PROTOCOL, run_id=plan["run_id"], completed=True,
                  correctness_passed=True, bitwise_equal_to_A=True, measurement_identity=identity,
                  slow_rank_stress=plan["slow_rank_stress"],
                  roles=roles, all_samples=samples, target_percent=1.0, target_validated=False,
                  acceptance_status="full_step_measurement_required",
                  bootstrap=dict(unit="complete fresh-process A/S/O block", samples=10000, seed=20261020),
                  limitation="Operator microbenchmark only. No optimizer steps were measured; no 1% training speedup is claimed.",
                  evidence_files={str(p.relative_to(root)): sha(p) for p in (root / "results").glob("block*/rank*.json")})
    write(root / "report.json", result)
    print(json.dumps(result["roles"], indent=2))
    return result


def profile_summary(root):
    plan = verify(root)
    result, useful_calls, overlap_calls, matched_profiles = {}, 0, 0, 0
    for arm in ARMS:
        for rank in range(8):
            row = read(root / "results" / f"profile-{arm}" / f"rank{rank}.json")
            require(row["completed"] and row["rank"] == rank and row["profiling"] and
                    row["run_id"] == plan["run_id"] and row["plan_sha256"] == sha(root / "plan.json"),
                    "Incomplete or mismatched profile worker")
            for role in ROLES:
                path = root / "results" / f"profile-{arm}" / f"rank{rank}-{role}-profile.json.gz"
                with gzip.open(path, "rt") as handle:
                    events = json.load(handle)["traceEvents"]
                kernels = [e for e in events if e.get("cat") == "kernel"]
                gemms = [e for e in kernels if "flux_bf16" in e["name"] and "reducescatter" in e["name"]]
                decoders = [e for e in kernels if "taco_decode" in e["name"] or
                            "taco_op3_try_decode" in e["name"] or "taco_op3_tail_decode" in e["name"]]
                overlaps = [dict(gemm=g["name"], decoder=d["name"], gemm_stream=g["args"].get("stream"),
                                 decoder_stream=d["args"].get("stream"),
                                 overlap_us=min(g["ts"]+g["dur"], d["ts"]+d["dur"])-max(g["ts"], d["ts"]))
                            for g in gemms for d in decoders
                            if max(g["ts"], d["ts"]) < min(g["ts"]+g["dur"], d["ts"]+d["dur"])
                            and g["args"].get("stream") != d["args"].get("stream")]
                require(gemms and decoders, "Profile lacks the expected GEMM or decoder")
                diagnostics = row["roles"][role]["diagnostics"]
                useful = []
                if arm != "A":
                    require(len(diagnostics) == 4 and len({d["epoch"] for d in diagnostics}) == 4,
                            "Missing profile epochs or stale diagnostics")
                    for d in diagnostics:
                        require(d["mode"] == ("serial" if arm == "S" else "overlap") and
                                d["rounds"] == plan["overlap_rounds"], "Profile mode/rounds changed")
                        actual = (d["try_fragments"] > 0 and d["first_decode_complete_ns"] > 0 and
                                  d["first_decode_complete_ns"] < d["last_local_publish_ns"])
                        require(bool(d["useful_overlap"]) == actual, "Useful-overlap flag contradicts timestamps")
                        useful.append(actual)
                if arm == "O":
                    useful_calls += sum(useful)
                    overlap_calls += len(overlaps)
                    matched_profiles += int(any(useful) and bool(overlaps))
                result[f"{arm}/rank{rank}/{role}"] = dict(trace=str(path), sha256=sha(path),
                    gemm_count=len(gemms), decoder_count=len(decoders), distinct_stream_overlap=overlaps,
                    diagnostics=diagnostics, useful_early_calls=sum(useful))
    return dict(protocol=PROTOCOL, profiles=result, target_validated=False,
                useful_early_calls=useful_calls, different_stream_overlap_pairs=overlap_calls,
                useful_and_overlapped_rank_role_profiles=matched_profiles,
                useful_early_consumption_validated=matched_profiles > 0,
                limitation="Separate diagnostic profiles establish useful early consumption only when payload decode completes before local production ends and kernels overlap on different streams. They do not estimate unprofiled speedup.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[817, 919, 1231])
    p.add_argument("--reuse", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2)
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
    args = parser.parse_args()
    if args.action == "report":
        report(args.root)
    else:
        globals()[args.action](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Budget-matched scattered/closed masks and per-source/merged decoding; operator timings are not full steps.

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
PROTOCOL = "h100-reduction-closure-v1-operator"
ARMS = ("SP", "CP", "SM", "CM")
ROLES = {"attn": [2048, 4096, 512], "mlp": [2048, 4096, 2048]}
SOURCE_PROTOCOL = "h100-reduction-closure-v1"
try:
    import closure_plan
    import closure_reference
except ModuleNotFoundError:
    import arrival_op4_v2_plan as closure_plan
    import arrival_op4_v2_reference as closure_reference

COMPARISONS = (("CM", "SM"), ("CP", "SP"), ("CM", "CP"), ("SM", "SP"), ("CM", "SP"))


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
    require(type(blocks) is int and blocks >= 16 and blocks % 16 == 0,
            "Use a positive multiple of sixteen complete blocks")
    # Opposite rotations form pairs; each pair plus its reverse balances positions.
    rotations = (0, 2, 1, 3)
    first = [list(ARMS[rotations[b % 4]:] + ARMS[:rotations[b % 4]])
             for b in range(blocks // 2)]
    orders = first + [row[::-1] for row in first[::-1]]
    return [dict(block=b, phase=b % 8, arm=arm, index=b * len(ARMS) + i,
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
    import arrival_op4_v2_build as builder
    record = builder.verify(build, built=True)
    manifest = read(build / "manifest.json")
    require(not manifest.get("source_only") and not manifest.get("diagnostic_only")
            and not record.get("baseline_source_only"), "Diagnostic/source-only builds cannot be measured")
    source_files = {str((build / arm / name).resolve()): digest
                    for arm, variant in manifest["variants"].items()
                    for name, digest in variant["source_sha256"].items()}
    return dict(builder_root=str(Path(builder.__file__).resolve().parent),
                manifest_sha256=sha(build / "manifest.json"),
                closure_manifest_sha256=sha(build / builder.MANIFEST),
                source_files=source_files, builder_inputs=record["inputs"])


def prepare(args):
    # Source preparation is valid before compilation; runtime hashes are frozen
    # only once the requested SP/CP/SM/CM libraries actually exist.
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
    shutil.copy2(Path(closure_plan.__file__), root / "scripts/closure_plan.py")
    shutil.copy2(Path(closure_reference.__file__), root / "scripts/closure_reference.py")
    shutil.copy2(REPO / "tools/a800/phase4/reference.py", root / "scripts/reference.py")
    plan = dict(protocol=PROTOCOL, run_id=uuid.uuid4().hex,
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                kind="operator_microbenchmark", build=str(build), libraries=identities,
                quantization_protocol=closure_plan.protocol_record(),
                same_mask_sample_l2_budget=.005,
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
                limitation="A new same-host SP/CP/SM/CM operator experiment, not historical paired data or full training steps.")
    write(root / "plan.json", plan)
    print(root)


def verify(root, offline=False):
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    require(plan["protocol"] == PROTOCOL and plan["arms"] == list(ARMS), "Wrong op4 plan")
    require(plan["roles"] == ROLES, "Role shapes changed")
    require(plan["quantization_protocol"] == closure_plan.protocol_record() and
            plan["same_mask_sample_l2_budget"] == .005, "Quantization protocol or tolerance changed")
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


def environment(root, plan, arm, python, cuda_home, profile=False, phase=0):
    library = Path(plan["libraries"][arm]["root"])
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("NCCL_", "TORCH_NCCL_", "H100_OP3_", "H100_OP4_", "H100_CLOSURE_")):
            env.pop(key)
    env.update(plan["communication"])
    env.update(plan["shared_environment"])
    env.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1",
               PYTHONPATH=f"{root}/scripts:{library}/python",
               H100_CLOSURE_ARM=arm, H100_CLOSURE_PHASE=str(phase),
               H100_CLOSURE_PROTOCOL="h100-reduction-closure-v1",
               H100_CLOSURE_BENCH_EXECUTE="1")
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
    jobs = ([dict(arm=arm, phase=0, block=-1, index=i, directory=f"profile-{arm}") for i, arm in enumerate(ARMS)]
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
        env = environment(root, plan, job["arm"], python, args.cuda_home, args.profile, phase=job["phase"])
        write(dest / "command.json", dict(command=command, arm=job["arm"],
                                         closure_environment={k: v for k, v in env.items() if k.startswith("H100_CLOSURE_")}))
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
    require(os.environ.get("H100_CLOSURE_BENCH_EXECUTE") == "1", "Use run --execute")
    root = args.root.resolve()
    plan = verify(root)
    job = json.loads(args.job)
    require(job in ([dict(arm=a, phase=0, block=-1, index=i, directory=f"profile-{a}") for i, a in enumerate(ARMS)]
                   if args.profile else plan["schedule"]), "Unregistered worker job")
    expected_environment = dict(plan["shared_environment"], **plan["communication"],
        H100_CLOSURE_ARM=job["arm"], H100_CLOSURE_PHASE=str(job["phase"]), H100_CLOSURE_PROTOCOL=SOURCE_PROTOCOL)
    require(all(os.environ.get(k) == v for k, v in expected_environment.items()),
            "Worker arm/protocol/shared environment differs from frozen plan")
    import torch
    import torch.distributed as dist
    from closure_reference import check_taco_output, check_actual_packets
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
            require(not getattr(op, "owns_output", False) and not hasattr(op, "slots"),
                    "All closure arms must retain the same single workspace and clone")
            state = op.closure_state()
            validate_state(state, job["arm"], job["phase"])
            packet_pointers = [op.peers[rank].data_ptr()]
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
                    packet_check = check_actual_packets(op, inputs @ weights.t(), actual, group)
                    repeated = owned_forward(op, inputs, weights)
                    require(torch.equal(actual, repeated), "Repeated output differs; possible publication/reuse race")
                    checks.append(dict(epoch=epoch, seed=seed,
                                       delayed_rank=delayed_rank, delay_cycles=delay_cycles,
                                       inputs_sha256=tensor_hash(inputs), weights_sha256=tensor_hash(weights),
                                       output_sha256=tensor_hash(actual), correctness=correctness, packet_check=packet_check))
            # No reference collective, hash, or synchronize between these calls.
            # Six generations reuse the same workspace and keep every output live.
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
                        with torch.profiler.record_function(f"closure.{role}.{invocation}"):
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
                owns_output=False, workspace_count=1, packet_pointers=packet_pointers, closure_state=state)
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


def validate_state(state, arm, phase):
    expected = dict(protocol=SOURCE_PROTOCOL, arm=arm, phase=phase,
        mask_sha256=closure_plan.mask_sha256(arm, phase),
        mask_kind='closed' if arm[0]=='C' else 'dispersed',
        reduction='merged' if arm[1]=='M' else 'per_source',
        inverse_h128_per_rank=1024 if arm=='CM' else 7168,
        selected_per_source=56, selected_per_link=8, wire_bytes_per_rank=13819904)
    require(state == expected, 'Runtime mask/reduction/budget differs from the frozen protocol')
    return expected


def savings_estimate(samples):
    blocks = sorted({r['block'] for r in samples})
    differences = []
    for block in blocks:
        rows = [r for r in samples if r['block']==block]
        require(len(rows)==4 and {r['arm'] for r in rows}==set(ARMS), 'Incomplete four-arm block')
        times = {r['arm']:r['rankmax_wall_us'] for r in rows}
        require(all(set(t)==set(ROLES) and all(math.isfinite(v) and v>0 for v in t.values())
                    for t in times.values()), 'Invalid role timing')
        differences.append(128*sum(times['SM'][role]-times['CM'][role] for role in ROLES)/1000)
    require(len(blocks)>=2, 'Need complete paired blocks')
    rng=random.Random(20261021)
    bootstrap=sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(10000))
    return dict(mean_ms=statistics.mean(differences),ci95_ms=[bootstrap[249],bootstrap[9749]],
        paired_blocks=len(blocks),block_savings_ms=differences,role_calls_per_optimizer_step=128,
        formula='128 * sum_roles(SM_us - CM_us) / 1000',
        limitation='Operator extrapolation only; never a training prerequisite or measured step saving.')


def report(root, save=True, offline=False):
    root=Path(root).resolve()
    plan=verify(root, offline)
    identity, samples, evidence = None, [], {}
    inputs_seen, numerical_pairs, bursts_seen = {}, {}, {}
    for job in plan['schedule']:
        rows=[read(root/'results'/job['directory']/f'rank{rank}.json') for rank in range(8)]
        for rank,row in enumerate(rows):
            require(row['completed'] and row['rank']==rank and row['job']==job and
                row['protocol']==PROTOCOL and row['run_id']==plan['run_id'] and
                row['plan_sha256']==sha(root/'plan.json') and not row['profiling'] and not row['sampling'],
                'Incomplete or mismatched operator evidence')
            expected=dict(plan['shared_environment'],**plan['communication'],H100_CLOSURE_ARM=job['arm'],
                          H100_CLOSURE_PHASE=str(job['phase']),H100_CLOSURE_PROTOCOL=SOURCE_PROTOCOL)
            require(row['execution_environment']==expected,'Operator runtime environment differs')
            current=dict(hardware=row['hardware'],software=row['software'])
            identity=current if identity is None else identity
            require(current==identity,'Mixed operator hardware/software')
            require(set(row['roles'])==set(ROLES),'Missing operator role')
            for role,record in row['roles'].items():
                validate_state(record['closure_state'],job['arm'],job['phase'])
                require(record['owns_output'] is False and record['workspace_count']==1 and
                        len(record['packet_pointers'])==1 and record['packet_pointers'][0]>0,
                        'All arms must keep the same workspace/output lifecycle')
                checks=record['checks']
                require(record['shape']==ROLES[role] and [(c['epoch'],c['seed']) for c in checks]==
                    [(e,s) for e in range(plan['reuse']) for s in plan['seeds']], 'Missing changing-input checks')
                for check in checks:
                    correctness=check['correctness']; packet=check['packet_check']
                    require(correctness['passed'] and correctness['budget_vs_bf16']==.05 and
                        correctness['budget_vs_codec']==.01 and
                        math.isfinite(correctness['relative_l2']) and correctness['relative_l2']<=.05 and
                        math.isfinite(correctness['protocol_relative_l2']) and correctness['protocol_relative_l2']<=.01,
                        'Independent codec correctness failed')
                    require(packet['passed'] and packet['budget']==.005 and
                        math.isfinite(packet['relative_l2']) and packet['relative_l2']<=.005 and
                        packet['sample_pattern']=='multiplicative-512-v1' and packet['sample_elements']==512 and
                        len(packet['sample'])==512 and all(math.isfinite(v) for v in packet['sample']),
                        'Actual-packet oracle failed or missing')
                    require((check['delayed_rank'],check['delay_cycles'])==
                        correctness_delay(plan,check['epoch'],check['seed']),'Delay stress differs')
                    key=(job['phase'],rank,role,check['epoch'],check['seed'])
                    fingerprint=(check['inputs_sha256'],check['weights_sha256'])
                    inputs_seen.setdefault(key,fingerprint)
                    require(inputs_seen[key]==fingerprint,'Inputs differ across masks/reduction arms')
                    pairkey=(job['block'],rank,role,check['epoch'],check['seed'],job['arm'][0])
                    numerical_pairs.setdefault(pairkey,{})[job['arm'][1]]=packet
                burst=record['retained_output_checks']
                require([c['generation'] for c in burst]==list(range(plan['retained_output_calls'])) and
                        all(c['correctness']['passed'] for c in burst),'Missing retained-output correctness')
                # Determinism within one arm/phase is required; different masks need not agree.
                key=(job['arm'],job['phase'],rank,role)
                hashes=[(c['inputs_sha256'],c['weights_sha256'],c['output_sha256']) for c in burst]
                bursts_seen.setdefault(key,hashes)
                require(bursts_seen[key]==hashes,'Same arm/phase retained output changed')
                t=record['timing']
                require(t['iterations']==plan['iterations'] and t['warmup']==plan['warmup'] and
                        math.isfinite(t['wall_us']) and t['wall_us']>0,'Invalid operator timing')
            path=root/'results'/job['directory']/f'rank{rank}.json'
            evidence[str(path.relative_to(root))]=sha(path)
        samples.append(dict(job,rankmax_wall_us={role:max(r['roles'][role]['timing']['wall_us'] for r in rows)
                                                for role in ROLES}))
    max_pair_error=0.0
    for pair in numerical_pairs.values():
        require(set(pair)=={'P','M'},'Missing same-mask precision control')
        require(pair['P']['actual_packet_sha256']==pair['M']['actual_packet_sha256'],
                'P/M changed encoded payloads under the same mask')
        error=closure_reference.paired_sample_error(pair['P']['sample'],pair['M']['sample'])
        require(error<=plan['same_mask_sample_l2_budget'],'Same-mask P/M sample error exceeds frozen budget')
        max_pair_error=max(max_pair_error,error)
    roles={}
    for role in ROLES:
        times={arm:[r['rankmax_wall_us'][role] for r in sorted(samples,key=lambda r:r['block']) if r['arm']==arm]
               for arm in ARMS}
        roles[role]=dict(mean_us={a:statistics.mean(t) for a,t in times.items()},
                        comparisons={a+'/'+b:paired(times[a],times[b]) for a,b in COMPARISONS})
    result=dict(protocol=PROTOCOL,run_id=plan['run_id'],completed=True,correctness_passed=True,
        precision_control_passed=True,budget_matched=True,same_mask_packet_identical=True,
        same_mask_max_sample_relative_l2=max_pair_error,measurement_identity=identity,
        quantization_protocol=plan['quantization_protocol'],roles=roles,all_samples=samples,
        micro_savings_estimate=savings_estimate(samples),target_percent=1.0,target_validated=False,
        acceptance_status='full_step_measurement_required',evidence_files=evidence,
        bootstrap=dict(unit='complete fresh-process four-arm same-phase block',samples=10000,seed=20261020),
        limitation='Different masks intentionally need not be bitwise identical; no optimizer step was measured.')
    if save:
        path=root/'report.json'
        if path.exists(): require(read(path)==result,'Saved operator report differs from raw evidence')
        else: write(path,result)
        print(json.dumps(dict(roles=roles,micro_savings_estimate=result['micro_savings_estimate']),indent=2))
    return result


def trace_path(events, arm, calls=4):
    require(arm in ARMS and calls>0,'Unknown closure path')
    kernels=sorted((e for e in events if e.get('cat')=='kernel'),key=lambda e:e['ts'])
    gemms=[e for e in kernels if 'flux_bf16' in e['name'] and 'reducescatter' in e['name']]
    decoder='taco_closure_decode_merged' if arm[1]=='M' else 'taco_closure_decode_per_source'
    decoders=[e for e in kernels if decoder+'(' in e['name']]
    barriers=[e for e in kernels if 'CudaIpcBarrierAllKernel' in e['name']]
    clones=[e for e in events if e.get('cat')=='cpu_op' and e.get('name')=='aten::clone']
    require(len(gemms)==len(decoders)==calls and len(barriers)==2*calls and len(clones)==calls,
            'Wrong common GEMM/decoder/two-barrier/clone path')
    require(not any(any(x in e['name'] for x in ('try_decode','tail_decode','op3_v2_stop','taco_decode_ring_kernel'))
                    for e in kernels),'Legacy or arrival decoder unexpectedly executed')
    for i,(gemm,decode) in enumerate(zip(gemms,decoders)):
        order=[gemm,barriers[2*i],decode,barriers[2*i+1]]
        require(all(e.get('args',{}).get('stream') is not None for e in order) and
                len({e['args']['stream'] for e in order})==1 and all(
                a['ts']+a['dur']<=b['ts']+.01 for a,b in zip(order,order[1:])), 'Wrong stream/ordering')
    return dict(calls=calls,gemm_count=len(gemms),decoder_count=len(decoders),barrier_count=len(barriers),
                clone_count=len(clones),decoder=decoder,execution_path_validated=True,
                gemm_median_us=statistics.median(e['dur'] for e in gemms),
                decoder_median_us=statistics.median(e['dur'] for e in decoders),
                predicted_inverse_h128_per_rank=1024 if arm=='CM' else 7168,
                inverse_count_is_analytic=True)


def profile_summary(root, offline=False):
    root=Path(root).resolve(); plan=verify(root,offline)
    timing_identity=read(root/'report.json')['measurement_identity']
    profiles,evidence={},{}
    for index,arm in enumerate(ARMS):
        for rank in range(8):
            folder=root/'results'/f'profile-{arm}'; row_path=folder/f'rank{rank}.json'; row=read(row_path)
            job=dict(arm=arm,phase=0,block=-1,index=index,directory=f'profile-{arm}')
            require(row['completed'] and row['profiling'] and row['rank']==rank and row['job']==job and
                    row['protocol']==PROTOCOL and row['run_id']==plan['run_id'] and
                    row['plan_sha256']==sha(root/'plan.json'),'Mismatched profile rank')
            expected=dict(plan['communication'],**plan['shared_environment'],H100_CLOSURE_ARM=arm,
                          H100_CLOSURE_PHASE='0',H100_CLOSURE_PROTOCOL=SOURCE_PROTOCOL)
            require(row['execution_environment']==expected,'Profile environment differs')
            require(dict(hardware=row['hardware'],software=row['software'])==timing_identity,
                    'Profile hardware/software differs from timed operator evidence')
            evidence[str(row_path.relative_to(root))]=sha(row_path)
            for role in ROLES:
                validate_state(row['roles'][role]['closure_state'],arm,0)
                path=folder/f'rank{rank}-{role}-profile.json.gz'
                with gzip.open(path,'rt') as handle: summary=trace_path(json.load(handle)['traceEvents'],arm)
                profiles[f'{arm}/rank{rank}/{role}']=dict(summary,trace=str(path.relative_to(root)),sha256=sha(path))
                evidence[str(path.relative_to(root))]=sha(path)
    return dict(protocol=PROTOCOL,run_id=plan['run_id'],completed=True,profiles=profiles,
        execution_path_validated=True,target_validated=False,evidence_files=evidence,
        limitation='Separate diagnostic durations are not unprofiled training savings; inverse counts are analytic.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--blocks", type=int, default=16)
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

#!/usr/bin/env python3
"""MA5-R tuned Attention and original CR2 MLP versus unchanged CR2."""
import argparse
import csv
import ctypes
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE_PROTOCOL = "h100-ma5-role-split-v1"
PROTOCOL = SOURCE_PROTOCOL + "-operator"
ARMS = ("T", "CR2")
ROLES = {"attn": [2048, 4096, 512], "mlp": [2048, 4096, 2048]}
COMPARISONS = (("T", "CR2"),)
LABELS = {"T": "Role split: tuned Attention decoder, original CR2 MLP and producer",
          "CR2": "Unchanged CR2 cached submission baseline"}
C_POLICY = 'interleaved_arrival_selective_attn_mlp'
SELECTION_FINGERPRINT = 'af3ae293f080dd4688834dca3c25f3ee8190eb0b621e191805e680defc164b3b'

def verify_selection(selection):
    require(hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()==SELECTION_FINGERPRINT, 'Frozen C selection changed')



def family(arm):
    require(arm in ARMS,'Unknown arm')
    return 'selective'


def barriers_per_call(arm):
    require(arm in ARMS, 'Unknown arm')
    return 1


def clone_per_call(arm):
    require(arm in ARMS, 'Unknown arm')
    return 0


CASES = ("zero", "small", "outlier", "random")
BOOTSTRAP_SEED = 20261021


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


def save_unchanged(path, value):
    if Path(path).exists():
        require(read(path) == value, "Saved evidence differs from current raw evidence: " + str(path))
    else:
        write(path, value)


def schedule(blocks=8):
    require(type(blocks) is int and blocks >= 8 and blocks % 8 == 0,
            "Use a positive multiple of eight complete T/CR2 blocks")
    first = [list(ARMS[b % len(ARMS):] + ARMS[:b % len(ARMS)]) for b in range(blocks // 2)]
    orders = first + [row[::-1] for row in first[::-1]]
    return [dict(block=b, arm=arm, index=b * len(ARMS) + i,
                 directory=f"block{b:02d}-{arm}")
            for b, order in enumerate(orders) for i, arm in enumerate(order)]


def stage_jobs(stage):
    require(stage in ("preflight", "profile"), "Unknown standalone worker stage")
    return [dict(arm=arm, block=-1, index=i, directory=f"{stage}-{arm}")
            for i, arm in enumerate(ARMS)]


def libraries(build):
    result = {}
    for arm in ARMS:
        root = (Path(build) / arm).resolve()
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
        mapping = root / "check_mapping"
        require(mapping.is_file(), f"Missing compiled mapping probe: {mapping}")
        result[arm] = dict(root=str(root), files=paths,
                           mapping=dict(path=str(mapping), sha256=sha(mapping)))
    return result


def build_identity(build, builder_root=None):
    if builder_root is not None:
        sys.path.insert(1, str(builder_root))
    import ma5r_build as builder
    build = Path(build).resolve()
    record = builder.verify(build, built=True)
    manifest = read(build / "manifest.json")
    require(builder.PROTOCOL == SOURCE_PROTOCOL and record["protocol"] == SOURCE_PROTOCOL,
            "Wrong source protocol")
    require(not manifest.get("source_only") and not manifest.get("diagnostic_only")
            and not record.get("baseline_source_only"), "Diagnostic/source-only builds cannot be measured")
    source_files = {str((build / arm / name).resolve()): digest
                    for arm, variant in manifest["variants"].items()
                    for name, digest in variant["source_sha256"].items()}
    return dict(builder_root=str(Path(builder.__file__).resolve().parent),
                manifest_sha256=sha(build / "manifest.json"),
                ma5r_manifest_sha256=sha(build / builder.MANIFEST),
                source_files=source_files, builder_inputs=record["inputs"], tuning=record["tuning"])


BASELINE_TUNING = dict(decoder_threads=128, decoder_blocks=2048, gemm_sms=-1, gemm_tile_k=32)


def validate_tuning(tuning):
    require(isinstance(tuning, dict) and set(tuning) == {'attn', 'mlp'}, 'Missing role tuning')
    for role, value in tuning.items():
        require(isinstance(value, dict) and set(value) == set(BASELINE_TUNING), 'Incomplete role tuning')
        require(all(type(v) is int for v in value.values()), 'Noninteger tuning')
        require(value['decoder_threads'] in (64,128,256,512), 'Invalid decoder threads')
        require(1 <= value['decoder_blocks'] <= 8192 // (value['decoder_threads']//32), 'Invalid decoder blocks')
        require(value['gemm_sms'] == -1 and value['gemm_tile_k'] == 32, 'MA5-R must preserve GEMM')
    require(tuning['mlp'] == BASELINE_TUNING, 'MA5-R must preserve CR2 MLP')
    return tuning


def expected_tuning(plan, arm, role):
    require(arm in ARMS and role in ROLES, 'Unknown MA5-R arm/role')
    return dict(validate_tuning(plan['tuning'])[role]) if arm == 'T' else dict(BASELINE_TUNING)


def native_tuning(op, arm, plan, role):
    expected = expected_tuning(plan, arm, role)
    states = op.cr2_submission_state()
    require(len(states) == 2 and all(s['bound_k'] == ROLES[role][2] for s in states),
            'Native decoder role differs from the immutable instance K')
    if arm == 'T':
        lib = op.slots[0].lib
        version = lib.taco_ma5r_protocol_version
        version.argtypes = []; version.restype = ctypes.c_int
        require(version() == 1, 'Wrong MA5-R protocol')
        for key, value in expected.items():
            fn = getattr(lib, 'taco_ma5r_' + key)
            fn.argtypes = [ctypes.c_int]; fn.restype = ctypes.c_int
            require(fn(ROLES[role][2]) == value, 'Native MA5-R role tuning mismatch: ' + key)
    return expected


def validate_launch(gemm, decoder, arm, role, tuning):
    require(tuning is not None, 'Profile requires frozen launch tuning')
    expected = expected_tuning({'tuning': tuning}, arm, role)
    args = decoder.get('args', {})
    grid, block = args.get('grid'), args.get('block')
    require(grid == [expected['decoder_blocks'],1,1] and block == [expected['decoder_threads'],1,1],
            'Decoder launch grid/block differs from registered tuning')
    require('_128x128x' + str(expected['gemm_tile_k']) + '_gemmstreamk_' in gemm.get('name',''),
            'Executed GEMM K tile differs from registered tuning')
    ga = gemm.get('args', {})
    for name in ('grid','block'):
        require(isinstance(ga.get(name),list) and len(ga[name]) == 3 and
                all(type(v) is int and v > 0 for v in ga[name]), 'Missing actual GEMM launch dimensions')
    require(ga['block'] == [128,1,1], 'GEMM thread block changed')
    return dict(gemm_tile=[128,128,expected['gemm_tile_k']],decoder_grid=grid,decoder_block=block,gemm_grid=ga['grid'],gemm_block=ga['block'],
                configured_gemm_sms=expected['gemm_sms'])


def prepare(args):
    root, build = args.root.resolve(), args.build.resolve()
    provenance, identities = build_identity(build), libraries(build)
    selection = read(build/"selection-plan.json")
    jobs = schedule(args.blocks)
    require(args.warmup >= 1 and args.iterations >= 2 and args.reuse >= 2,
            "Positive warmup, at least two timed calls and reuse cycles required")
    require(len(args.seeds) >= 2 and len(set(args.seeds)) == len(args.seeds)
            and all(type(s) is int and 0 <= s < 10000 for s in args.seeds),
            "At least two distinct nonnegative correctness seeds below 10000 required")
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
                build_identity=provenance, tuning=provenance["tuning"], selection=selection, roles=ROLES, arms=list(ARMS), arm_labels=LABELS, schedule=jobs,
                warmup=args.warmup, iterations=args.iterations, seeds=args.seeds,
                cases=list(CASES), reuse=args.reuse, retained_output_calls=6,
                instance_isolation_calls=6,
                slow_rank_stress=dict(ranks=[0, 7], cycles=2000000,
                    scope="First random seed in correctness epochs 0 and 1, and retained-output burst; never timed."),
                communication={"NCCL_NVLS_ENABLE": "0", "NCCL_DEBUG": "WARN",
                               "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"},
                shared_environment={"CUDA_DEVICE_MAX_CONNECTIONS": "1",
                                    "FLUX_GEMM_RS_GEMM_ONLY": "1"},
                source_files={str(p.relative_to(root)): sha(p) for p in (root / "scripts").iterdir()},
                target_validated=False, historical_timings_used=False,
                limitation="Same-host complete operator API including required output ownership, not optimizer steps.")
    write(root / "plan.json", plan)
    print(root)


def verify(root, offline=False):
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    require(plan["protocol"] == PROTOCOL and plan["arms"] == list(ARMS)
            and set(plan["libraries"]) == set(ARMS), "Wrong MA5-R tile/block plan or libraries")
    require(plan["arm_labels"] == LABELS, "MA5-R tile/block control and candidate labels changed")
    require(plan["roles"] == ROLES and plan["cases"] == list(CASES), "Shapes or correctness cases changed")
    require(plan["schedule"] == schedule(len(plan["schedule"]) // len(ARMS)),
            "Incomplete or changed balanced schedule")
    require(plan["reuse"] >= 2 and plan["retained_output_calls"] >= 6 and
            plan["instance_isolation_calls"] == 6,
            "Insufficient workspace reuse validation")
    for name, digest in plan["source_files"].items():
        require(sha(root / name) == digest, f"Frozen source changed: {name}")
    verify_selection(plan['selection'])
    validate_tuning(plan['tuning'])
    require(plan['tuning'] == plan['build_identity']['tuning'], 'Tuning differs from frozen build')
    if not offline:
        require(sha(__file__) == plan["source_files"]["scripts/bench.py"],
                "Operator controller differs from its frozen worker")
        require(build_identity(Path(plan["build"]), plan["build_identity"]["builder_root"])
                == plan["build_identity"], "Frozen build or frontend changed")
        require(libraries(Path(plan["build"])) == plan["libraries"], "Frozen libraries changed")
    return plan


def execution_environment(plan, arm):
    require(arm in ARMS, "Unknown MA5-R tile/block arm")
    return dict(plan["communication"], **plan["shared_environment"],
                H100_MA5R_PROTOCOL=SOURCE_PROTOCOL, H100_MA5R_ARM=arm)


def environment(root, plan, arm, python, cuda_home, profile=False):
    library = Path(plan["libraries"][arm]["root"])
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("NCCL_", "TORCH_NCCL_", "H100_OP3_", "H100_OP4_", "H100_CLOSURE_", "H100_MA1_", "H100_MA2_", "H100_MA2C_", "H100_MA5R_", "H100_MA5_", "H100_CR2_", "H100_MA2R_")):
            env.pop(key)
    env.update(execution_environment(plan, arm))
    env.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", PYTHONPATH=f"{root}/scripts:{library}/python",
               H100_MA5R_BENCH_EXECUTE="1")
    for key in ("FLUX_FORCE_NVLINK", "FLUX_TUNE_CONFIG_FILE", "PYTHONOPTIMIZE"):
        env.pop(key, None)
    env["PATH"] = str(python.parent) + ":" + env.get("PATH", "")
    ld = [str(library / "python/flux/lib"), str(library), str(library / "build/lib")]
    if cuda_home:
        env["CUDA_HOME"] = str(cuda_home)
        env["PATH"] = str(cuda_home / "bin") + ":" + env["PATH"]
        ld.append(str(cuda_home / "lib64"))
    env["LD_LIBRARY_PATH"] = ":".join(ld) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def run_worker(command, cwd, env, log, timeout):
    """Own and reap only the new process group, including its torchrun ranks."""
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
        return code
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        raise


def verify_mapping(rows):
    require(len(rows) == 4096 and {r["rank"] for r in rows} == set(range(8)), "Incomplete GPU mapping")
    for rank in range(8):
        part = sorted((r for r in rows if r["rank"] == rank), key=lambda r: r["index"])
        require([r["index"] for r in part] == list(range(512)), "Duplicate or missing mapping index")
        require(all(0 <= r["m"] < 16 and 0 <= r["n"] < 32 and
                    r["destination"] == r["m"] // 2 and r["sm_occupancy"] > 0 for r in part),
                "Invalid mapped coordinate, destination or exact-kernel occupancy")
        require([r["m"] * 32 + r["n"] for r in part] ==
                [((i % 8 + rank) % 8) * 64 + i // 8 for i in range(512)],
                "Actual GPU mapping differs from the common interleaved order")


def mapping_evidence(root, plan):
    evidence, reference = {}, {}
    for arm in ARMS:
        for role in ROLES:
            path=Path(root)/'results'/f'mapping-{arm}-{role}.csv'
            with path.open() as handle: rows=[{k:int(v) for k,v in row.items()} for row in csv.DictReader(handle)]
            require(len(rows)==4096,'Incomplete mapping')
            for rank in range(8):
                rr=sorted((r for r in rows if r['rank']==rank),key=lambda r:r['index'])
                expected=(plan['selection']['variants'][C_POLICY]['roles'][role]['entry']['coords'][rank]
                          if family(arm)=='selective' else [((i%8+rank)%8)*64+i//8 for i in range(512)])
                require([r['index'] for r in rr]==list(range(512)) and
                        [r['m']*32+r['n'] for r in rr]==expected and
                        all(r['destination']==r['m']//2 and r['sm_occupancy']>0 for r in rr),
                        'Actual mapping differs from registered arm')
            # Occupancy may differ under tuning; exact coordinates and positive occupancy are checked above.
            evidence[str(path.relative_to(root))]=sha(path)
    return evidence


def run_mappings(root, plan, python, cuda_home, timeout):
    for arm in ARMS:
        library = plan["libraries"][arm]
        flux = next(path for path in library["files"] if Path(path).name == "libflux_cuda.so")
        for role, (m, n, k) in ROLES.items():
            command = [library["mapping"]["path"], str(m), str(n), str(k * 8), "8", flux]
            path = root / "results" / f"mapping-{arm}-{role}.csv"
            with path.open("x") as handle:
                run_worker(command, root, environment(root, plan, arm, python, cuda_home), handle, timeout)
    mapping_evidence(root, plan)


def run(args):
    root = args.root.resolve()
    plan = verify(root)
    jobs = stage_jobs("profile") if args.profile else stage_jobs("preflight") + plan["schedule"]
    python = args.python.expanduser().absolute()  # Preserve a virtualenv launcher symlink.
    prefix = [str(python), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
              str(root / "scripts/bench.py"), "worker", str(root)]
    if not args.execute:
        print(json.dumps(dict(dry_run=True, jobs=jobs, command=prefix,
                              mapping_required=True, target_validated=False), indent=2))
        return
    marker = root / ("profile-started.json" if args.profile else "timing-started.json")
    write(marker, dict(run_id=plan["run_id"], jobs=jobs, status="started"))
    if args.profile:
        require(report(root, save=False)["completed"], "Complete unprofiled operator run required first")
    else:
        run_mappings(root, plan, python, args.cuda_home, args.worker_timeout)
    for job in jobs:
        verify(root)
        is_preflight = job["directory"].startswith("preflight-")
        if not args.profile and not is_preflight and job["index"] == 0:
            preflight(root)
        dest = root / "results" / job["directory"]
        dest.mkdir(exist_ok=False)
        command = prefix + ["--job", json.dumps(job, sort_keys=True)]
        if args.profile:
            command.append("--profile")
        if is_preflight:
            command.append("--preflight")
        env = environment(root, plan, job["arm"], python, args.cuda_home, args.profile)
        write(dest / "command.json", dict(command=command, job=job,
                                          execution_environment=execution_environment(plan, job["arm"])))
        print("START", job["directory"], flush=True)
        try:
            with (dest / "process.log").open("x") as log:
                run_worker(command, root, env, log, args.worker_timeout)
        except BaseException as error:
            write(dest / "failed.json", dict(status="failed", retry=False,
                exception=type(error).__name__, message=str(error), job=job))
            raise
    if args.profile:
        save_unchanged(root / "profile-summary.json", profile_summary(root))
    else:
        report(root)


def correctness_delay(plan, epoch, seed, case="random"):
    stress = plan["slow_rank_stress"]
    if case == "random" and epoch < len(stress["ranks"]) and seed == plan["seeds"][0]:
        return stress["ranks"][epoch], stress["cycles"]
    return -1, 0


def check_indices(plan):
    return [(epoch, seed, case) for epoch in range(plan["reuse"])
            for seed in plan["seeds"] for case in CASES]


def make_inputs(torch, shape, seed, case):
    m, n, k = shape
    torch.manual_seed(seed)
    inputs = torch.randn(m, k, device="cuda") * .02
    weights = torch.randn(n, k, device="cuda") * .02
    if case == "zero":
        inputs.zero_()
    elif case == "small":
        inputs.mul_(1e-4)
    elif case == "outlier":
        inputs[:, ::128] *= 16
        weights[::128, ::128] *= 16
    else:
        require(case == "random", "Unknown correctness distribution")
    return inputs.to(torch.bfloat16), weights.to(torch.bfloat16)


def complete_forward(op, inputs, weights):
    """Both CR2-derived arms return independent storage."""
    output = op.forward(inputs, weights)
    return output if getattr(op, "owns_configuration", False) else output.clone()


def require_independent_outputs(outputs):
    require(len({output.data_ptr() for output in outputs}) == len(outputs),
            "Retained independent outputs alias")


def make_operator(plan, role, group, arm):
    from flux.gemm_rs_taco import GemmRSTaco
    if family(arm)=='selective':
        from flux.gemm_rs_taco import GemmRSTacoDoubleBuffered
        entry=plan['selection']['variants'][C_POLICY]['roles'][role]['entry']
        return GemmRSTacoDoubleBuffered(group,*ROLES[role],placement='fused',
            selected=entry['mask'],quantization_mode='targeted')
    return GemmRSTaco(group,*ROLES[role],placement='fused')


def packet_pointers(op):
    return [s.peers[s.rank].data_ptr() for s in getattr(op,'slots',[op])]


def reference_mask(op):
    return getattr(op,'selection_cpu',None)


def runtime_state(op, arm, plan, role, rank):
    slots=getattr(op,'slots',[op]); selective=family(arm)=='selective'
    require(len(slots)==(2 if selective else 1),'Wrong workspace lifecycle')
    require(bool(getattr(op,'owns_configuration',False)) is True,'Wrong native ownership')
    ptrs=packet_pointers(op); require(len(set(ptrs))==len(ptrs),'Workspace alias')
    lib=slots[0].lib
    names=('taco_cr2_protocol_version','taco_ma2c_protocol_version','taco_instance_config_supported')
    for name in names:
        fn=getattr(lib,name);fn.argtypes=[];fn.restype=ctypes.c_int
        require(fn()==1,'Wrong native mechanism ABI: '+name)
    mask_hash=None
    if selective:
        expected=plan['selection']['variants'][C_POLICY]['roles'][role]['mask_sha256']
        require(all(slot.selection_sha256==expected for slot in slots),'Wrong C mask')
        mask_hash=expected
    state=dict(arm=arm,workspace_count=len(slots),output_clone=bool(clone_per_call(arm)),
        owns_configuration=bool(getattr(op,'owns_configuration',False)),placement=op.placement,
        mask_sha256=mask_hash,wire_bytes_per_rank=op.wire_bytes_per_rank,protocol_version=1,
        tuning=native_tuning(op, arm, plan, role))
    validate_runtime_state(state,arm,plan,role,rank)
    return state


def validate_runtime_state(state, arm, plan, role, rank):
    selective=family(arm)=='selective'
    record=plan['selection']['variants'][C_POLICY]['roles'][role]
    wire=14680064-15360*sum(record['entry']['mask'][rank]) if selective else 7798784
    require(state==dict(arm=arm,workspace_count=2 if selective else 1,
        output_clone=bool(clone_per_call(arm)),owns_configuration=True,placement='fused',
        mask_sha256=record['mask_sha256'] if selective else None,wire_bytes_per_rank=wire,
        protocol_version=1,tuning=expected_tuning(plan,arm,role)),'Runtime lifecycle, mask, wire bytes or tuning changed')


def submission_state(op, arm):
    require(arm in ARMS, 'Unknown submission arm')
    return op.cr2_submission_state()


def validate_submission(states, arm, role, pointers, *, minimum_calls=0):
    require(arm in ARMS, 'Unknown submission arm')
    require(len(states) == len(pointers) == 2, 'CR2 requires two submission slots')
    for slot, (state, pointer) in enumerate(zip(states, pointers)):
        require(set(state) == {'protocol_version','initializations','successful_calls','bound_k',
                               'local_packet_ptr','failed','slot'}, 'Incomplete CR2 submission counters')
        count = state['successful_calls']
        require(type(count) is int and count >= minimum_calls and
                state['protocol_version'] == 1 and state['failed'] is False and
                state['slot'] == slot and
                state['local_packet_ptr'] == pointer and
                state['initializations'] == int(count > 0) and
                state['bound_k'] == ROLES[role][2],
                'CR2 cached submission path was not used or was rebound')
    require(abs(states[0]['successful_calls'] - states[1]['successful_calls']) <= 1,
            'CR2 workspace submission counters are not balanced')


def worker(args):
    require(os.environ.get("H100_MA5R_BENCH_EXECUTE") == "1", "Use run --execute")
    require(not (args.profile and args.preflight), "Worker stage must be unique")
    root, job = args.root.resolve(), json.loads(args.job)
    plan = verify(root)
    allowed = stage_jobs("profile" if args.profile else "preflight") if (args.profile or args.preflight) else plan["schedule"]
    require(job in allowed, "Unregistered worker job")
    expected = execution_environment(plan, job["arm"])
    require(all(os.environ.get(k) == v for k, v in expected.items()), "Worker environment differs from frozen plan")
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
                "This benchmark requires eight full 132-SM H100s")
        devices.append(dict(name=p.name, uuid=str(p.uuid), sm_count=p.multi_processor_count))
    require(len({p["uuid"] for p in devices}) == 8, "Expected eight distinct physical GPU UUIDs")
    require(all(torch.cuda.can_device_access_peer(i, j) for i in range(world) for j in range(world) if i != j),
            "All local GPU pairs must support peer access")
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    group = dist.new_group(list(range(world)))
    import flux
    from flux.gemm_rs_taco import GemmRSTaco
    library = plan["libraries"][job["arm"]]
    require(Path(flux.__file__).resolve().is_relative_to(Path(library["root"])), "Wrong Flux Python package loaded")
    mapped = {str(Path(line.split()[-1]).resolve()) for line in Path("/proc/self/maps").read_text().splitlines()
              if line.split() and line.split()[-1].startswith("/")}
    for path, digest in library["files"].items():
        require(sha(path) == digest and path in mapped, f"Frozen library is not loaded: {path}")
    dest = root / "results" / job["directory"]
    result = dict(protocol=PROTOCOL, run_id=plan["run_id"], plan_sha256=sha(root / "plan.json"),
                  job=job, rank=rank, hardware=dict(host=os.uname().nodename, devices=devices),
                  software=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                nccl=list(torch.cuda.nccl.version()), python=sys.version),
                  profiling=args.profile, preflight=args.preflight, sampling=False,
                  loaded_libraries=library["files"], roles={}, completed=False,
                  execution_environment=expected, target_validated=False)

    def tensor_hash(value):
        return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    with torch.no_grad():
        for role_index, (role, shape) in enumerate(ROLES.items()):
            op = make_operator(plan,role,group,job["arm"])
            state, checks = runtime_state(op, job["arm"],plan,role,rank), []
            submission_before = submission_state(op, job['arm'])
            validate_submission(submission_before, job['arm'], role, packet_pointers(op))
            require(all(s['successful_calls'] == 0 for s in submission_before), 'New CR2 operator already used')
            for epoch, seed, case in check_indices(plan):
                actual_seed = seed + epoch * 100000 + role_index * 10000 + rank
                inputs, weights = make_inputs(torch, shape, actual_seed, case)
                delayed_rank, delay_cycles = correctness_delay(plan, epoch, seed, case)
                if rank == delayed_rank:
                    torch.cuda._sleep(delay_cycles)
                actual = complete_forward(op, inputs, weights)
                correctness = check_taco_output(inputs @ weights.t(), actual, group, mask=reference_mask(op))
                repeated = complete_forward(op, inputs, weights)
                require(torch.equal(actual, repeated), "Repeated output differs; possible workspace reuse race")
                checks.append(dict(epoch=epoch, seed=seed, case=case,
                                   delayed_rank=delayed_rank, delay_cycles=delay_cycles,
                                   inputs_sha256=tensor_hash(inputs), weights_sha256=tensor_hash(weights),
                                   output_sha256=tensor_hash(actual), correctness=correctness))
            # Keep six generations live without reference collectives or host syncs.
            retained = []
            for generation in range(plan["retained_output_calls"]):
                burst_input, burst_weight = make_inputs(torch, shape,
                    4711 + generation * 100000 + role_index * 10000 + rank, "random")
                if rank == (0 if generation % 2 == 0 else 7):
                    torch.cuda._sleep(plan["slow_rank_stress"]["cycles"])
                retained.append((burst_input, burst_weight, complete_forward(op, burst_input, burst_weight)))
            torch.cuda.synchronize()
            require_independent_outputs([item[2] for item in retained])
            burst_checks = []
            for generation, (burst_input, burst_weight, burst_output) in enumerate(retained):
                correctness = check_taco_output(burst_input @ burst_weight.t(), burst_output, group, mask=reference_mask(op))
                burst_checks.append(dict(generation=generation, correctness=correctness,
                    inputs_sha256=tensor_hash(burst_input), weights_sha256=tensor_hash(burst_weight),
                    output_sha256=tensor_hash(burst_output)))
            del retained, burst_input, burst_weight, burst_output
            isolation_checks = []
            if args.preflight:
                # Two live instances with different K must restore their own
                # configuration; no reference collectives occur inside the burst.
                other_role = next(name for name in ROLES if name != role)
                other = make_operator(plan,other_role,group,job["arm"])
                runtime_state(other, job["arm"],plan,other_role,rank)
                require(not (set(packet_pointers(op)) & set(packet_pointers(other))),
                        "Two instances share their communication workspace")
                isolated = []
                for generation in range(plan["instance_isolation_calls"]):
                    current_role = role if generation % 2 == 0 else other_role
                    current_op = op if generation % 2 == 0 else other
                    x, w = make_inputs(torch, ROLES[current_role],
                        8711 + generation * 100000 + role_index * 10000 + rank, "random")
                    isolated.append((current_role, x, w, complete_forward(current_op, x, w)))
                torch.cuda.synchronize()
                require_independent_outputs([item[3] for item in isolated])
                for generation, (current_role, x, w, y) in enumerate(isolated):
                    correctness = check_taco_output(x @ w.t(), y, group, mask=reference_mask(op if current_role==role else other))
                    isolation_checks.append(dict(generation=generation, role=current_role,
                        correctness=correctness, inputs_sha256=tensor_hash(x),
                        weights_sha256=tensor_hash(w), output_sha256=tensor_hash(y)))
                dist.barrier(group=group)
                del isolated, other, current_op, x, w, y
            timing = None
            if not args.preflight:
                for _ in range(plan["warmup"]):
                    output = complete_forward(op, inputs, weights)
                dist.barrier(group=group)
                torch.cuda.synchronize()
                if args.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
                        for invocation in range(4):
                            with torch.profiler.record_function(f"ma5r.{role}.{invocation}"):
                                output = complete_forward(op, inputs, weights)
                            torch.cuda.synchronize()
                    trace = dest / f"rank{rank}-{role}-profile.json"
                    profiler.export_chrome_trace(str(trace))
                    with trace.open("rb") as src, gzip.open(str(trace) + ".gz", "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    trace.unlink()
                else:
                    started = time.perf_counter()
                    for _ in range(plan["iterations"]):
                        output = complete_forward(op, inputs, weights)
                    torch.cuda.synchronize()
                    timing = dict(wall_us=(time.perf_counter() - started) * 1e6 / plan["iterations"],
                                  iterations=plan["iterations"], warmup=plan["warmup"],
                                  includes="owned-output forward and final synchronization")
                require(tensor_hash(output) == checks[-1]["output_sha256"], "Final output changed")
                del output
            result["roles"][role] = dict(shape=shape, checks=checks, timing=timing,
                                        retained_output_checks=burst_checks, runtime_state=state,
                                        submission_before=submission_before,
                                        submission_after=submission_state(op, job['arm']),
                                        packet_pointers=packet_pointers(op),
                                        retained_storage_independent=True,
                                        instance_workspaces_distinct=bool(args.preflight),
                                        instance_isolation_checks=isolation_checks)
            dist.barrier(group=group)
            del op, inputs, weights, actual, repeated
    result["completed"] = True
    write(dest / f"rank{rank}.json", result)
    dist.destroy_process_group()


def paired(candidate, baseline):
    require(len(candidate) == len(baseline) and len(candidate) >= 2, "At least two paired blocks required")
    require(all(math.isfinite(x) and x > 0 for x in [*candidate, *baseline]), "Invalid paired timing")
    logs = [math.log(x / y) for x, y in zip(candidate, baseline)]
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(100 * (1 - math.exp(statistics.mean(rng.choices(logs, k=len(logs)))))
                     for _ in range(10000))
    return dict(percent=100 * (1 - math.exp(statistics.mean(logs))),
                ci95=[samples[249], samples[9749]], paired_blocks=len(logs))


def validate_correctness(check):
    require(check["passed"] and check["budget_vs_bf16"] == .05 and check["budget_vs_codec"] == .01
            and math.isfinite(check["relative_l2"]) and 0 <= check["relative_l2"] <= .05
            and math.isfinite(check["protocol_relative_l2"]) and 0 <= check["protocol_relative_l2"] <= .01
            and math.isfinite(check["max_abs"]) and check["max_abs"] >= 0,
            "Independent codec correctness failed")


def validate_rows(root, plan, jobs, stage, reference=None):
    identity, fingerprints = (None, {}) if reference is None else reference
    evidence, samples, isolation_fingerprints = {}, [], {}
    for job in jobs:
        rows = []
        for rank in range(8):
            path = root / "results" / job["directory"] / f"rank{rank}.json"
            row = read(path)
            require(row["completed"] and row["rank"] == rank and row["job"] == job and
                    row["protocol"] == PROTOCOL and row["run_id"] == plan["run_id"] and
                    row["plan_sha256"] == sha(root / "plan.json") and not row["sampling"] and
                    row["profiling"] == (stage == "profile") and row["preflight"] == (stage == "preflight"),
                    "Incomplete or mismatched rank evidence")
            require(row["execution_environment"] == execution_environment(plan, job["arm"]),
                    "Worker environment differs from plan")
            require(row["loaded_libraries"] == plan["libraries"][job["arm"]]["files"], "Wrong loaded runtime evidence")
            current = dict(hardware=row["hardware"], software=row["software"])
            identity = current if identity is None else identity
            require(identity == current, "Mixed hardware or software across T/CR2")
            require(set(row["roles"]) == set(ROLES), "Missing worker role")
            for role, record in row["roles"].items():
                require(record["shape"] == ROLES[role], "Wrong measured role shape")
                validate_runtime_state(record["runtime_state"], job["arm"],plan,role,rank)
                pointers = record['packet_pointers']
                require(len(pointers) == record['runtime_state']['workspace_count'] and
                        all(type(p) is int and p > 0 for p in pointers) and len(set(pointers)) == len(pointers),
                        'Invalid operator workspace pointers')
                validate_submission(record['submission_before'], job['arm'], role, pointers)
                validate_submission(record['submission_after'], job['arm'], role, pointers, minimum_calls=1)
                expected_calls = 2 * len(check_indices(plan)) + plan['retained_output_calls']
                expected_calls += plan['instance_isolation_calls'] // 2 if stage == 'preflight' else plan['warmup'] + (4 if stage == 'profile' else plan['iterations'])
                if job['arm'] in ARMS:
                    require(sum(s['successful_calls'] for s in record['submission_before']) == 0 and
                            sum(s['successful_calls'] for s in record['submission_after']) == expected_calls,
                            'CR2 actual cached submissions differ from executed calls')
                checks = record["checks"]
                require([(c["epoch"], c["seed"], c["case"]) for c in checks] == check_indices(plan),
                        "Missing changing-input distribution or reuse epoch")
                for check in checks:
                    validate_correctness(check["correctness"])
                    require((check["delayed_rank"], check["delay_cycles"]) == correctness_delay(
                        plan, check["epoch"], check["seed"], check["case"]), "Delay stress differs")
                burst = record["retained_output_checks"]
                require(record["retained_storage_independent"] is True,
                        "Retained independent output storage was not validated")
                require([c["generation"] for c in burst] == list(range(plan["retained_output_calls"])),
                        "Missing retained-output reuse check")
                for check in burst:
                    validate_correctness(check["correctness"])
                isolated = record["instance_isolation_checks"]
                if stage == "preflight":
                    require(record["instance_workspaces_distinct"] is True,
                            "Distinct instance workspaces were not validated")
                    other_role = next(name for name in ROLES if name != role)
                    require([(c["generation"], c["role"]) for c in isolated] ==
                            [(i, role if i % 2 == 0 else other_role)
                             for i in range(plan["instance_isolation_calls"])],
                            "Missing interleaved instance configuration check")
                    for check in isolated:
                        validate_correctness(check["correctness"])
                    isolation_hashes = [[c[k] for k in ("inputs_sha256", "weights_sha256", "output_sha256")]
                                        for c in isolated]
                    require(all(re.fullmatch("[0-9a-f]{64}", v)
                                for row_hashes in isolation_hashes for v in row_hashes),
                            "Invalid instance isolation fingerprint")
                    isolation_key = f"{family(job['arm'])}/rank{rank}/{role}"
                    isolation_fingerprints.setdefault(isolation_key, isolation_hashes)
                    require(isolation_hashes == isolation_fingerprints[isolation_key],
                            "T/CR2 interleaved instance output differs bitwise")
                else:
                    require(isolated == [] and record["instance_workspaces_distinct"] is False,
                            "Instance isolation belongs to untimed preflight")
                hashes = [[c[k] for k in ("inputs_sha256", "weights_sha256", "output_sha256")]
                          for c in checks + burst]
                require(all(re.fullmatch("[0-9a-f]{64}", v) for row_hashes in hashes for v in row_hashes),
                        "Invalid tensor fingerprint")
                input_key=f"input/rank{rank}/{role}"
                input_hashes=[row[:2] for row in hashes]
                fingerprints.setdefault(input_key,input_hashes)
                require(fingerprints[input_key]==input_hashes,'Cross-arm inputs or weights differ')
                key = f"{family(job['arm'])}/rank{rank}/{role}"
                fingerprints.setdefault(key, hashes)
                require(hashes == fingerprints[key], "T/CR2 input, weight or output differs bitwise")
                timing = record["timing"]
                if stage == "timing":
                    require(timing["iterations"] == plan["iterations"] and timing["warmup"] == plan["warmup"]
                            and math.isfinite(timing["wall_us"]) and timing["wall_us"] > 0 and
                            timing["includes"] == "owned-output forward and final synchronization", "Invalid complete operator timing")
                else:
                    require(timing is None, "Profile/preflight evidence cannot contain timing samples")
            evidence[str(path.relative_to(root))] = sha(path)
            rows.append(row)
        if stage == "timing":
            samples.append(dict(job, rankmax_wall_us={role: max(r["roles"][role]["timing"]["wall_us"] for r in rows)
                                                     for role in ROLES}))
    return identity, fingerprints, evidence, samples


def preflight(root, save=True, offline=False):
    root = Path(root).resolve()
    plan = verify(root, offline=offline)
    identity, hashes, evidence, _ = validate_rows(root, plan, stage_jobs("preflight"), "preflight")
    evidence.update(mapping_evidence(root, plan))
    result = dict(protocol=PROTOCOL, run_id=plan["run_id"], plan_sha256=sha(root / "plan.json"),
                  completed=True, correctness_passed=True, bitwise_equal_within_pairs=True, equivalence_pairs=[["T","CR2"]],
                  actual_mapping_validated=True, instance_isolation_validated=True,
                  cases=list(CASES), measurement_identity=identity,
                  tensor_fingerprints=hashes, evidence_files=evidence, target_validated=False)
    if save:
        save_unchanged(root / "preflight.json", result)
    return result


def report(root, save=True, offline=False):
    root = Path(root).resolve()
    plan = verify(root, offline=offline)
    gate = preflight(root, save=False, offline=offline)
    require(read(root / "preflight.json") == gate, "Pre-timing correctness gate differs from current evidence")
    identity, _, evidence, samples = validate_rows(root, plan, plan["schedule"], "timing",
        (gate["measurement_identity"], gate["tensor_fingerprints"]))
    evidence.update(gate["evidence_files"])
    roles = {}
    for role in ROLES:
        values = {arm: [r["rankmax_wall_us"][role] for r in sorted(samples, key=lambda x: x["block"])
                        if r["arm"] == arm] for arm in ARMS}
        roles[role] = dict(mean_us={arm: statistics.mean(v) for arm, v in values.items()},
                           comparisons={c + "/" + b: paired(values[c], values[b]) for c, b in COMPARISONS})
    result = dict(protocol=PROTOCOL, run_id=plan["run_id"], completed=True, correctness_passed=True,
                  bitwise_equal_within_pairs=True, equivalence_pairs=[["T","CR2"]], actual_mapping_validated=True,
                  measurement_identity=identity, arm_labels=LABELS, tuning=plan["tuning"], roles=roles, all_samples=samples,
                  target_validated=False, historical_timings_used=False,
                  acceptance_status="full_step_measurement_required", evidence_files=evidence,
                  bootstrap=dict(unit="complete fresh-process T/CR2 block; per-window maximum across eight ranks",
                                 samples=10000, seed=BOOTSTRAP_SEED),
                  limitation="All positive, negative and uncertain operator effects are retained; no optimizer-step speedup is claimed.")
    if save:
        save_unchanged(root / "report.json", result)
        print(json.dumps(dict(roles=roles, target_validated=False), indent=2))
    return result


def decoder_matches(name, arm, role):
    require(arm in ARMS, 'Unknown MA5-R arm')
    require(role in ROLES, 'Unknown decoder role')
    kernel = 'taco_ma5r_attention_decode_kernel' if arm == 'T' and role == 'attn' else 'taco_ma2c_decode_kernel'
    return bool(re.search(r'\b' + kernel + r'\s*\(|_Z\d+' + kernel, name))


def trace_path(events, arm, role, calls=4, tuning=None):
    """Bind each GPU path to its CPU call by runtime-launch correlation IDs."""
    require(arm in ARMS and role in ROLES and calls > 0, "Unknown MA5-R tile/block profile path")
    device = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy")]
    kernels = [e for e in device if e.get("cat") == "kernel"]
    require(not any(any(x in e.get("name", "") for x in ("try_decode", "tail_decode", "op3_v2", "closure_decode"))
                    for e in kernels), "Legacy or arrival decoder unexpectedly executed")
    all_gemms = [e for e in kernels if "flux_bf16" in e.get("name", "") and "reducescatter" in e["name"]]
    all_decoders = [e for e in kernels if "decode" in e.get("name", "") and "taco" in e["name"]]
    all_barriers = [e for e in kernels if "CudaIpcBarrierAllKernel" in e.get("name", "")]
    require(len(all_gemms) == len(all_decoders) == calls and len(all_barriers) == barriers_per_call(arm) * calls and
            all(decoder_matches(e["name"], arm, role) for e in all_decoders), "Wrong GEMM/decoder/two-barrier path or arm")
    launches = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver") and
                e.get("args", {}).get("correlation") is not None]
    matched, claimed = [], set()

    def contained(child, parent):
        return child.get("pid") == parent.get("pid") and child.get("tid") == parent.get("tid") and \
            child["ts"] >= parent["ts"] and child["ts"] + child.get("dur", 0) <= parent["ts"] + parent["dur"] + .01

    for invocation in range(calls):
        # Kineto may mirror record_function spans as gpu_user_annotation.
        # Launch containment uses the CPU pid/tid and its host-clock interval.
        markers = [e for e in events if e.get("cat") == "user_annotation" and
                   e.get("name") == f"ma5r.{role}.{invocation}" and e.get("ph") == "X"]
        require(len(markers) == 1, "Missing or duplicated profiled invocation marker")
        marker = markers[0]
        call_launches = [e for e in launches if contained(e, marker)]
        correlations = {e["args"]["correlation"] for e in call_launches}
        path = [e for e in device if e.get("args", {}).get("correlation") in correlations]
        gemms = [e for e in path if e in all_gemms]
        decoders = [e for e in path if e in all_decoders]
        barriers = sorted((e for e in path if e in all_barriers), key=lambda e: e["ts"])
        require(len(gemms) == len(decoders) == 1 and len(barriers) == barriers_per_call(arm),
                "Profile launch correlations do not identify one complete invocation")
        clones = [e for e in events if e.get("name") == "aten::clone" and e.get("cat") == "cpu_op" and contained(e, marker)]
        require(len(clones) == clone_per_call(arm), "Profile invocation has wrong output clone count")
        views = [e for e in events if e.get('cat') == 'cpu_op' and
                 e.get('name') in ('aten::slice','aten::view') and contained(e, marker)]
        require(not views, 'CR2 direct output executed legacy slice/view')
        clone_correlations = {e["args"]["correlation"] for e in call_launches
                              if clones and contained(e, clones[0])}
        clone_work = [e for e in path if e.get("args", {}).get("correlation") in clone_correlations]
        require(bool(clone_work) == bool(clone_per_call(arm)), "Output clone lacks correlated device work")
        order = [gemms[0], barriers[0], decoders[0]] + (barriers[1:] if barriers_per_call(arm)==2 else []) + sorted(clone_work, key=lambda e: e["ts"])
        require(all(e.get("args", {}).get("stream") is not None and math.isfinite(e.get("dur", -1)) and e["dur"] >= 0 for e in order)
                and len({e["args"]["stream"] for e in order}) == 1 and
                all(a["ts"] + a["dur"] <= b["ts"] + .01 for a, b in zip(order, order[1:])),
                "GEMM/publication/decode/reuse/clone stream ordering changed")
        identifiers = {id(e) for e in order}
        require(not (identifiers & claimed), "A profile kernel was attributed to multiple invocations")
        claimed.update(identifiers)
        launch = validate_launch(gemms[0], decoders[0], arm, role, tuning)
        matched.append(dict(launch=launch, invocation=invocation, gemm_us=gemms[0]["dur"], decode_us=decoders[0]["dur"],
                            output_slice_view_count=len(views),
                            cpu_submission_us=marker["dur"],
                            publication_barrier_us=barriers[0]["dur"], reuse_barrier_us=barriers[1]["dur"] if len(barriers)==2 else 0.0,
                            device_path_us=order[-1]["ts"] + order[-1]["dur"] - order[0]["ts"],
                            stream=order[0]["args"]["stream"], decoder=decoders[0]["name"]))
    require({id(e) for e in all_gemms + all_decoders + all_barriers} <= claimed,
            "Unattributed GEMM, decoder or barrier in profile")
    return dict(calls=calls, gemm_count=len(all_gemms), decoder_count=len(all_decoders),
                barrier_count=len(all_barriers), clone_count=calls * clone_per_call(arm), execution_path_validated=True,
                launch_correlations_validated=True, matched_calls=matched,
                cpu_submission_median_us=statistics.median(c["cpu_submission_us"] for c in matched),
                publication_barrier_median_us=statistics.median(c["publication_barrier_us"] for c in matched),
                reuse_barrier_median_us=statistics.median(c["reuse_barrier_us"] for c in matched),
                gemm_median_us=statistics.median(c["gemm_us"] for c in matched),
                decoder_median_us=statistics.median(c["decode_us"] for c in matched))


def profile_summary(root, offline=False):
    root = Path(root).resolve()
    plan = verify(root, offline=offline)
    timed = report(root, save=False, offline=offline)
    require(read(root / "report.json") == timed, "Saved operator evidence differs from profile source run")
    gate = preflight(root, save=False, offline=offline)
    _, _, evidence, _ = validate_rows(root, plan, stage_jobs("profile"), "profile",
        (timed["measurement_identity"], gate["tensor_fingerprints"]))
    profiles = {}
    for arm in ARMS:
        for rank in range(8):
            for role in ROLES:
                path = root / "results" / f"profile-{arm}" / f"rank{rank}-{role}-profile.json.gz"
                with gzip.open(path, "rt") as handle:
                    summary = trace_path(json.load(handle)["traceEvents"], arm, role, tuning=plan["tuning"])
                profiles[f"{arm}/rank{rank}/{role}"] = dict(summary, trace=str(path.relative_to(root)), sha256=sha(path))
                evidence[str(path.relative_to(root))] = sha(path)
    return dict(protocol=PROTOCOL, run_id=plan["run_id"], completed=True, arm_labels=LABELS, profiles=profiles,
                execution_path_validated=True, target_validated=False, evidence_files=evidence,
                limitation="Separately profiled decoder durations validate execution paths, not unprofiled throughput or machine-instruction overlap.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[817, 919])
    p.add_argument("--reuse", type=int, default=2)
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
    p.add_argument("--preflight", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("root", type=Path)
    p.add_argument("--offline", action="store_true")
    p = sub.add_parser("profile-report")
    p.add_argument("root", type=Path)
    p.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.action == "report":
        report(args.root, offline=args.offline)
    elif args.action == "profile-report":
        summary = profile_summary(args.root, offline=args.offline)
        save_unchanged(args.root / "profile-summary.json", summary)
        print(json.dumps(summary, indent=2))
    else:
        globals()[args.action](args)


if __name__ == "__main__":
    main()

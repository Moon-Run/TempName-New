"""Eight-GPU NCCL initialization probe; deliberately never imports Flux.

Run through torch.distributed.run with --nproc_per_node=8 and --execute.
Use a fresh process for each NCCL_NVLS_ENABLE setting; see the gxn75 guide.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import sys
import traceback


def emit(event, **fields):
    print(json.dumps(dict(event=event, rank=os.environ.get("RANK"),
                          pid=os.getpid(), host=socket.gethostname(), **fields)), flush=True)


def loaded_libraries():
    try:
        paths = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and any(name in fields[5] for name in
                                       ("libnccl", "libcuda", "libnvidia", "libflux")):
                paths.add(fields[5])
        return sorted(paths)
    except OSError as exc:
        return [str(exc)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run the GPU probe explicitly")
    args = parser.parse_args()
    if not args.execute:
        parser.error("GPU probe requires --execute; use --help for usage")
    try:
        rank, local, world, local_world = (int(os.environ[key]) for key in
                                         ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"))
    except (KeyError, ValueError):
        parser.error("Launch using torch.distributed.run --standalone --nproc_per_node=8")
    if world != 8 or local_world != 8 or rank != local or not 0 <= rank < 8:
        parser.error("Exactly eight local ranks on one node are required")

    # Fail all CPU-only invocation checks before importing torch or querying CUDA.
    import torch
    import torch.distributed as dist

    env_keys = {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_HOME",
                "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONNOUSERSITE",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS"}
    emit("environment", python=sys.executable, torch=torch.__version__,
         torch_cuda_build=torch.version.cuda, nccl=dist.is_nccl_available(),
         variables={key: value for key, value in sorted(os.environ.items())
                    if key in env_keys or key.startswith(("NCCL_", "TORCH_NCCL_"))})
    try:
        if torch.cuda.device_count() != 8:
            raise RuntimeError("Expose exactly eight GPUs using CUDA_VISIBLE_DEVICES")
        torch.cuda.set_device(local)
        props = torch.cuda.get_device_properties(local)
        emit("device", local_rank=local, name=props.name, uuid=str(props.uuid),
             nccl_version=torch.cuda.nccl.version(), libraries=loaded_libraries())
        timeout = datetime.timedelta(seconds=60)
        # Match the calibration's lazy initialization: no eager device_id or barrier.
        dist.init_process_group("nccl", timeout=timeout)
        group = dist.new_group(list(range(world)), backend="nccl", timeout=timeout)
        send = torch.full((64,), rank, dtype=torch.uint8, device="cuda")
        recv = torch.empty(world * 64, dtype=torch.uint8, device="cuda")
        emit("before_all_gather")
        dist.all_gather_into_tensor(recv, send, group=group)
        expected = torch.arange(world, dtype=torch.uint8).repeat_interleave(64)
        if not torch.equal(recv.cpu(), expected):
            raise RuntimeError("all_gather produced incorrect rank bytes")
        emit("all_gather_pass")
        value = torch.tensor([rank + 1.0], device="cuda", dtype=torch.float32)
        emit("before_all_reduce")
        dist.all_reduce(value, group=group)
        total = value.item()
        if total != 36.0:
            raise RuntimeError(f"all_reduce expected 36, got {total}")
        emit("all_reduce_pass", total=total, libraries=loaded_libraries())
        dist.destroy_process_group(group)
        dist.destroy_process_group()
        emit("PASS")
    except Exception:
        emit("FAIL", libraries=loaded_libraries())
        traceback.print_exc()
        # Do not add a collective or mask the original error with failed-group cleanup.
        # torchrun terminates sibling workers; the documented outer timeout bounds exit.
        raise SystemExit(1)


if __name__ == "__main__":
    main()

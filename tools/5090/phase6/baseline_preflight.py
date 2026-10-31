"""Run in a disposable process: never retain CUDA contexts in the supervisor."""
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def version(value):
    match = re.match(r'(\d+)\.(\d+)', str(value))
    require(match is not None, 'Cannot identify CUDA version: ' + str(value))
    return tuple(map(int, match.groups()))


def toolkit(root):
    root = Path(root).resolve()
    require((root/'bin/nvcc').is_file(), 'CUDA Toolkit missing: ' + str(root/'bin/nvcc'))
    text = subprocess.check_output([str(root/'bin/nvcc'), '--version'], text=True, timeout=30)
    match = re.search(r'\brelease (\d+\.\d+)', text)
    require(match is not None and version(match[1]) >= (12, 8), 'SM120 requires CUDA Toolkit >= 12.8')
    require((root/'include/cuda_runtime.h').is_file() and (root/'lib64/libcudart.so').is_file(),
            'CUDA headers/runtime missing: ' + str(root))
    return dict(root=str(root), version=match[1], nvcc=text.strip())


def validate_devices(devices, peers):
    require(len(devices) == 8, 'Expose exactly eight RTX 5090 GPUs (TP8/DP1)')
    require(len({d['uuid'] for d in devices}) == 8, 'Eight distinct GPU UUIDs required')
    require(all('RTX 5090' in d['name'] and d['capability'] == [12, 0] and d['sm_count'] == 170
                for d in devices), 'Expected eight RTX 5090 SM120 / 170-SM devices; no MIG or other SKU')
    require(len(peers) == 8 and all(len(row) == 8 for row in peers), 'Incomplete peer-access matrix')
    require(all(type(value) is bool for row in peers for value in row), 'Peer matrix must contain booleans')
    # Record physical capabilities, including all-False off-diagonal matrices.
    # This protocol communicates local packets through NCCL and never consumes
    # remote CUDA pointers. The distributed collective probe is the runtime gate.
    return [(i, j) for i in range(8) for j in range(8) if i != j and not peers[i][j]]


def memory_state(torch, device):
    """Observe driver-wide usage as well as this probe's allocator usage.

    Explicit logical CUDA indices respect CUDA_VISIBLE_DEVICES, unlike joining
    an nvidia-smi physical index to a torchrun rank. Driver usage includes other
    processes and non-Torch allocations; it is not a process attribution.
    """
    free, total = map(int, torch.cuda.mem_get_info(device))
    require(0 <= free <= total and total > 0, 'Invalid CUDA memory observation')
    return dict(free_memory_bytes=free, cuda_total_memory_bytes=total,
                used_memory_bytes=total-free,
                torch_allocated_bytes=int(torch.cuda.memory_allocated(device)),
                torch_reserved_bytes=int(torch.cuda.memory_reserved(device)))


def probe(cuda_home, build_cuda_home, *, evidence=None):
    # Preserve observations if a later Apex, topology, or inventory check fails.
    evidence = {} if evidence is None else evidence
    build_toolkit, runtime_toolkit = toolkit(build_cuda_home), toolkit(cuda_home)
    evidence.update(build_toolkit=build_toolkit, runtime_toolkit=runtime_toolkit)
    import torch
    require(torch.version.cuda is not None and version(torch.version.cuda) >= (12, 8),
            'Use a CUDA >= 12.8 PyTorch build with SM120 support; the H100 cu124 environment is unsuitable')
    require(torch.cuda.is_available(), 'CUDA is unavailable in the selected Python environment')
    require(torch.cuda.device_count() == 8, 'CUDA_VISIBLE_DEVICES must expose exactly eight RTX 5090 GPUs')
    devices = []
    evidence.update(devices=devices,
                    memory_observation_stage='after_cuda_context_initialization_before_apex_probes',
                    memory_observation_scope='device-wide point-in-time usage; not an idle or capacity guarantee')
    for i in range(8):
        prop = torch.cuda.get_device_properties(i)
        devices.append(dict(name=prop.name, uuid=str(prop.uuid), capability=[prop.major, prop.minor],
                            sm_count=prop.multi_processor_count, total_memory_bytes=prop.total_memory,
                            cuda_device=i, **memory_state(torch, i)))
    print(json.dumps(dict(memory_observation_stage=evidence['memory_observation_stage'],
                          devices=devices), ensure_ascii=False), flush=True)
    peers = [[i == j or torch.cuda.can_device_access_peer(i, j) for j in range(8)] for i in range(8)]
    unavailable_peers = validate_devices(devices, peers)
    # The shared step uses Apex Adam and layer normalization. Import checks alone
    # would miss an extension compiled only for H100, so launch both on SM120.
    for name in ('numpy', 'einops', 'regex', 'psutil', 'yaml', 'tqdm', 'six', 'packaging',
                 'pybind11', 'apex_C', 'amp_C', 'fused_layer_norm_cuda'):
        importlib.import_module(name)
    from apex.optimizers import FusedAdam
    from apex.normalization import FusedLayerNorm
    for i in range(8):
        with torch.cuda.device(i):
            matrix = torch.ones((64, 64), device='cuda', dtype=torch.bfloat16)
            result = matrix @ matrix
            require(bool((result == 64).all()), 'SM120 BF16 GEMM probe failed')
            layer = FusedLayerNorm(64).cuda().to(torch.bfloat16)
            value = torch.randn(4, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            layer(value).float().square().mean().backward()
            optimizer = FusedAdam(layer.parameters(), lr=1e-4)
            optimizer.step()
            torch.cuda.synchronize()
            require(bool(torch.isfinite(value.grad).all()), 'Apex SM120 layernorm/Adam probe failed')
            del matrix, result, layer, value, optimizer
            torch.cuda.empty_cache()
    try:
        topology = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True, stderr=subprocess.STDOUT, timeout=30)
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,memory.used',
                                             '--format=csv'], text=True, timeout=30)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError('nvidia-smi topology/inventory capture failed') from exc
    evidence.update(completed=False, environment_probe_passed=True, collective_probe_passed=False,
                host=socket.gethostname(), devices=devices, peer_access=peers,
                unavailable_peer_links=unavailable_peers, peer_access_required=False,
                topology=topology, nvidia_smi=inventory, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                build_toolkit=build_toolkit, runtime_toolkit=runtime_toolkit,
                environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                 nccl=list(torch.cuda.nccl.version()), python=sys.executable,
                                 python_version=sys.version,
                                 cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI)),
                bf16_and_apex_probe_passed=True,
                transport='rtx5090-nccl-staged-v2',
                limitation='Environment/Apex probes alone do not validate distributed communication or native model kernels')
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cuda-home', type=Path, required=True)
    parser.add_argument('--build-cuda-home', type=Path, required=True)
    args = parser.parse_args(argv)
    result = dict(completed=False, environment_probe_passed=False, collective_probe_passed=False)
    try:
        result = probe(args.cuda_home, args.build_cuda_home, evidence=result)
    except BaseException as exc:
        result.update(completed=False, error=f'{type(exc).__name__}: {exc}')
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps(result, indent=2, ensure_ascii=False), file=sys.stderr, flush=True)
        raise
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

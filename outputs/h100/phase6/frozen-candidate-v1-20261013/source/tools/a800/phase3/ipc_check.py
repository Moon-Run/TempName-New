"""Exercise imported IPC tensors using local GPU reads/writes, including ring aliases."""
import argparse
import ctypes
import datetime
import gc
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--reverse', action='store_true')
parser.add_argument('--probe', action='store_true')
args = parser.parse_args()
assert os.environ.get('SLURM_JOB_ID'), 'Run through the ordinary GPU queue.'
import torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
world = int(os.environ['WORLD_SIZE'])
device = world - 1 - rank if args.reverse else rank
torch.cuda.set_device(device)
import flux

dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=90), device_id=torch.device('cuda', device))
pg = dist.group.WORLD
flux.init_flux_shm(pg)

class PointerAttributes(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('device', ctypes.c_int),
                ('device_pointer', ctypes.c_void_p), ('host_pointer', ctypes.c_void_p)]

runtime = ctypes.CDLL('/data/apps/cuda/12.8/lib64/libcudart.so')
runtime.cudaPointerGetAttributes.argtypes = [ctypes.POINTER(PointerAttributes), ctypes.c_void_p]
runtime.cudaPointerGetAttributes.restype = ctypes.c_int
checks = []

def sync():
    torch.cuda.synchronize(device)
    dist.barrier(device_ids=[device])

def check_buffers(ring, iteration):
    tensors = flux.create_tensor_list([world, 128], torch.int32, pg, ring, True)
    sync()
    if args.probe:
        del tensors
        sync()
        return
    neighbors = {(rank - 1) % world, rank, (rank + 1) % world}
    attrs = []
    for peer, tensor in enumerate(tensors):
        assert tensor.device == torch.device('cuda', device), (peer, tensor.device, device)
        attr = PointerAttributes()
        assert runtime.cudaPointerGetAttributes(ctypes.byref(attr), tensor.data_ptr()) == 0
        attrs.append(dict(peer=peer, pointer_device=attr.device, tensor_device=tensor.device.index))
        assert torch.count_nonzero(tensor).item() == 0, (ring, iteration, peer, 'initial zero')
    del tensor
    sync()  # Every rank must finish checking zero before any owner writes its pattern.
    tensors[rank].fill_(100 + rank)
    sync()
    for peer, tensor in enumerate(tensors):
        owner = peer if not ring or peer in neighbors else rank
        assert torch.all(tensor.clone() == 100 + owner).item(), (ring, peer, 'peer read')
    del tensor
    sync()
    # Each source writes a separate row on each mapped destination.
    for peer, tensor in enumerate(tensors):
        if not ring or peer in neighbors:
            tensor[rank].fill_(1000 + rank)
    del tensor
    sync()
    actual = tensors[rank].cpu()
    for source in range(world):
        expected = 1000 + source if not ring or source in neighbors else 100 + rank
        assert torch.all(actual[source] == expected).item(), (ring, source, 'peer write')
    sync()
    del tensors
    gc.collect()
    sync()
    checks.append(dict(ring=ring, iteration=iteration, pointer_attributes=attrs, passed=True))

try:
    if args.probe:
        check_buffers(False, 0)
    else:
        for iteration in range(2):
            for ring in (False, True):
                check_buffers(ring, iteration)
        report = dict(rank=rank, world=world, device=device, reverse=args.reverse,
                      gpu_uuid=str(torch.cuda.get_device_properties(device).uuid), checks=checks)
        out = Path(os.environ['RESULT_DIR']) / f'ipc-tp{world}-reverse{int(args.reverse)}-rank{rank}.json'
        out.write_text(json.dumps(report, indent=2) + '\n')
    print('IPC CHECK PASS', rank, world, device, args.reverse, args.probe, flush=True)
finally:
    dist.destroy_process_group()

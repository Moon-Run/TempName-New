"""GPU gates, called only by explicitly launched H100 workers."""
import os
import socket


def check_topology(world):
    import torch
    if world != int(os.environ['H100_TP']) or torch.cuda.device_count() != world:
        raise RuntimeError('Expose exactly TP devices on one node; DP/subgroups are unsupported')
    devices=[]
    for i in range(world):
        p=torch.cuda.get_device_properties(i)
        if 'H100' not in p.name or (p.major,p.minor)!=(9,0):
            raise RuntimeError(f'Expected H100 SM90, got {p.name}')
        if p.multi_processor_count != int(os.environ['H100_SM_COUNT']):
            raise RuntimeError('Device SM count differs from build/config; MIG is unsupported')
        devices.append(dict(name=p.name, uuid=str(p.uuid), sm_count=p.multi_processor_count, capability=[p.major,p.minor]))
    if not all(torch.cuda.can_device_access_peer(i,j) for i in range(world) for j in range(world) if i!=j):
        raise RuntimeError('This IPC port requires peer access between every local pair')
    return dict(host=socket.gethostname(), devices=devices, local_rank=int(os.environ['LOCAL_RANK']))

"""Disposable eight-rank CUDA/NCCL correctness probe for staged transport v2.

The controller starts this file with torchrun. Peer access is deliberately
disabled in NCCL; no CUDA IPC pointer or native Flux library is used here.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

PROTOCOL = 'rtx5090-nccl-staged-collectives-v2'
ITERATIONS = 3
COLLECTIVES = ('all_gather', 'all_reduce', 'reduce_scatter', 'all_to_all_bytes')
REQUIRED_ENV = {'NCCL_P2P_DISABLE': '1', 'NCCL_SHM_DISABLE': '0', 'NCCL_NVLS_ENABLE': '0'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def packet_bytes(source, destination):
    return 0 if source == destination else 128 + 32*((source+2*destination) % 5)


def validate_records(rows, run_id, hardware, communication):
    require(len(rows) == 8 and {row['rank'] for row in rows} == set(range(8)),
            'All eight NCCL probe rank records are required')
    rows = sorted(rows, key=lambda row: row['rank'])
    for rank, row in enumerate(rows):
        require(row['protocol'] == PROTOCOL and row['run_id'] == run_id and row['completed'] is True
                and row['rank'] == row['local_rank'] == rank and row['world'] == 8,
                'Incomplete or foreign NCCL probe record')
        require(row['host'] == hardware['host'] and row['gpu_uuid'] == hardware['devices'][rank]['uuid']
                and row['environment'] == {key: hardware['environment'][key]
                                          for key in ('python', 'torch', 'cuda', 'nccl')},
                'NCCL probe hardware/software differs from environment probe')
        require(row['communication'] == communication and
                all(row['communication']['environment'].get(key) == value for key, value in REQUIRED_ENV.items()),
                'NCCL probe must use the frozen non-P2P transport configuration')
        require(row['iterations'] == ITERATIONS and row['non_default_stream'] is True
                and row['async_work_waits'] == ITERATIONS*len(COLLECTIVES)
                and row['collectives'] == dict.fromkeys(COLLECTIVES, True),
                'Missing NCCL correctness or stream-order validation')
        require(row['input_split_sizes'] == [packet_bytes(rank, destination) for destination in range(8)]
                and row['output_split_sizes'] == [packet_bytes(source, rank) for source in range(8)],
                'NCCL variable packet routing or zero self-split changed')
    return rows


def worker(out, run_id, communication):
    require(os.environ.get('RTX5090_TRANSPORT_PROBE_EXECUTE') == '1',
            'NCCL GPU probe requires explicit controller execution')
    require(all(os.environ.get(key) == value for key, value in REQUIRED_ENV.items()),
            'NCCL staged probe requires P2P disabled and SHM allowed')
    rank, local_rank, world = (int(os.environ[key]) for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    require(world == 8 and rank == local_rank, 'NCCL staged probe requires one node with eight ranks')
    record = dict(protocol=PROTOCOL, run_id=run_id, rank=rank, local_rank=local_rank,
                  world=world, completed=False)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    import torch
    import torch.distributed as dist
    try:
        # Match the campaign exactly, including configured NCCL files and NICs.
        actual = {key: value for key, value in os.environ.items()
                  if key.startswith(('NCCL_', 'TORCH_NCCL_')) or key == 'GLOO_SOCKET_IFNAME'}
        require(actual == communication['environment'], 'NCCL probe environment changed')
        actual_configs = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          if Path(name).is_file() else None for name in communication['config_files']}
        require(actual_configs == communication['config_files'], 'NCCL configuration files changed before probe')
        torch.cuda.set_device(local_rank)
        require(torch.cuda.device_count() == 8, 'Exactly eight CUDA devices are required')
        prop = torch.cuda.get_device_properties(local_rank)
        require('RTX 5090' in prop.name and (prop.major, prop.minor) == (12, 0)
                and prop.multi_processor_count == 170, 'Wrong GPU in NCCL probe')
        dist.init_process_group('nccl', timeout=timedelta(seconds=120))
        stream = torch.cuda.Stream()
        inputs = [packet_bytes(rank, destination) for destination in range(world)]
        outputs = [packet_bytes(source, rank) for source in range(world)]
        waits = 0

        def wait(work):
            nonlocal waits
            require(work is not None, 'Missing asynchronous NCCL Work')
            work.wait()  # Establish the NCCL completion dependency on this stream.
            waits += 1

        def packet(source, destination, epoch):
            count = packet_bytes(source, destination)
            return ((torch.arange(count, device='cuda', dtype=torch.int32)
                     + source*31 + destination*7 + epoch*17) % 251).to(torch.uint8)

        with torch.cuda.stream(stream):
            # Reuse receive storage on a non-default stream. A missing Work.wait
            # or premature producer overwrite can therefore corrupt a later epoch.
            received_packets = torch.empty(sum(outputs), device='cuda', dtype=torch.uint8)
            for epoch in range(ITERATIONS):
                local = torch.full((32,), rank, device='cuda', dtype=torch.bfloat16)
                gathered = torch.empty(world*32, device='cuda', dtype=torch.bfloat16)
                wait(dist.all_gather_into_tensor(gathered, local, async_op=True))
                expected = torch.arange(world, device='cuda', dtype=torch.bfloat16).repeat_interleave(32)
                require(torch.equal(gathered, expected), 'NCCL BF16 AllGather mismatch')

                reduced = torch.full((32,), rank+1, device='cuda', dtype=torch.bfloat16)
                wait(dist.all_reduce(reduced, async_op=True))
                require(torch.equal(reduced, torch.full_like(reduced, 36)), 'NCCL BF16 AllReduce mismatch')

                scatter_input = torch.cat([torch.full((32,), rank+1+destination, device='cuda',
                                                     dtype=torch.bfloat16) for destination in range(world)])
                scattered = torch.empty(32, device='cuda', dtype=torch.bfloat16)
                wait(dist.reduce_scatter_tensor(scattered, scatter_input, async_op=True))
                require(torch.equal(scattered, torch.full_like(scattered, 36+8*rank)),
                        'NCCL BF16 ReduceScatter mismatch')

                outgoing = torch.cat([packet(rank, destination, epoch) for destination in range(world)])
                wait(dist.all_to_all_single(received_packets, outgoing, output_split_sizes=outputs,
                                             input_split_sizes=inputs, async_op=True))
                expected_packets = torch.cat([packet(source, rank, epoch) for source in range(world)])
                require(torch.equal(received_packets, expected_packets), 'NCCL byte AllToAll routing mismatch')
            stream.synchronize()
        dist.barrier(device_ids=[local_rank])
        torch.cuda.synchronize()
        record.update(completed=True, host=socket.gethostname(), gpu_uuid=str(prop.uuid),
                      environment=dict(python=sys.executable, torch=str(torch.__version__),
                                       cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version())),
                      communication=communication, iterations=ITERATIONS,
                      non_default_stream=True, async_work_waits=waits,
                      collectives=dict.fromkeys(COLLECTIVES, True),
                      input_split_sizes=inputs, output_split_sizes=outputs)
    except BaseException as exc:
        record.update(error=f'{type(exc).__name__}: {exc}')
        (out/f'rank{rank}.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    (out/f'rank{rank}.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--communication', type=Path, required=True)
    args = parser.parse_args(argv)
    worker(args.out, args.run_id, json.loads(args.communication.read_text()))


if __name__ == '__main__':
    main()

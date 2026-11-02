#!/usr/bin/env python3
"""Disposable OP5 transport probe; target-sized BF16 AllGather before training."""
import argparse
from datetime import timedelta
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
import uuid

sys.dont_write_bytecode = True
from rtx5090_loader import HERE, REPO
import op3
import op3_protocol as p
from op5_evidence import OVERRIDES, copy_evidence

PROTOCOL = 'rtx5090-op5-shm-ce-probe-v1'
SETTINGS = dict(input_elements=1048576, dtype='bfloat16', world=8,
                correctness_iterations=3, warmup=20, timed_calls=100, profile_calls=10)
DEBUG = dict(NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,SHM')
require = p.require


def transport_log(text):
    links = re.findall(r'Channel\s+(\d+)\s*:\s*(\d+)\[(\d+)\]\s*->\s*(\d+)\[(\d+)\]\s+via\s+(\S+)', text)
    require(links, 'No actual NCCL transport connections in probe log')
    edges = set()
    for channel, src, src_gpu, dst, dst_gpu, transport in links:
        channel, src, dst = map(int, (channel, src, dst))
        require(0 <= src < 8 and 0 <= dst < 8 and src != dst
                and src_gpu == str(src) and dst_gpu == str(dst)
                and transport == 'SHM/direct/CE', 'Unexpected OP5 transport connection: '+str(links))
        edges.add((channel, src, dst))
    require(all({channel for channel, src, _ in edges if src == rank} >= set(range(16))
                and {channel for channel, _, dst in edges if dst == rank} >= set(range(16))
                for rank in range(8)), 'Missing 16-channel receive-CE connections on a probe rank')
    return dict(transport='SHM/direct/CE', ranks=8, connections=[list(e) for e in sorted(edges)])


def evidence(root, *, hardware=None, communication=None):
    root = Path(root)
    own_hardware = p.read(root/'environment-preflight.json')
    require(own_hardware['completed'] is True
            and own_hardware['environment']['nccl'] == [2, 27, 3],
            'OP5 probe requires a complete NCCL 2.27.3 environment')
    op3.base.collective_evidence(root, own_hardware)
    hardware = own_hardware if hardware is None else hardware
    communication = own_hardware['communication'] if communication is None else communication
    check_training_identity(root, hardware, communication)
    command = p.read(root/'command.json')
    env = dict(communication['environment'], **DEBUG)
    require(command['protocol'] == PROTOCOL and command['settings'] == SETTINGS
            and command['communication_environment'] == env, 'Probe launch settings changed')
    rows, profiles = [], []
    files = {name: p.sha(root/name) for name in ('environment-preflight.json', 'preflight-communication.json',
                                               'command.json', 'process.log')}
    for rank in range(8):
        row = p.read(root/f'rank{rank}.json')
        require(row['protocol'] == PROTOCOL and row['run_id'] == command['run_id']
                and row['rank'] == rank and row['completed'] is True and row['settings'] == SETTINGS
                and row['gpu_uuid'] == hardware['devices'][rank]['uuid'] and row['host'] == hardware['host']
                and row['environment'] == {k: hardware['environment'][k] for k in ('torch', 'cuda', 'nccl', 'python')}
                and row['communication_environment'] == env and row['config_files'] == communication['config_files']
                and row['bitwise_correct'] is True and row['non_default_stream'] is True
                and row['iterations'] == SETTINGS['correctness_iterations']
                and row['profile_complete'] is True, 'Missing/foreign OP5 transport rank evidence')
        require(all(math.isfinite(row[k]) and row[k] > 0 for k in ('gpu_us', 'wall_us')), 'Invalid probe timing')
        name = f'rank{rank}-profile.json.gz'
        with gzip.open(root/name, 'rt') as stream:
            profiles.append(copy_evidence(json.load(stream)['traceEvents'], rank, SETTINGS['profile_calls']))
        files[name] = p.sha(root/name)
        files[f'rank{rank}.json'] = p.sha(root/f'rank{rank}.json')
        rows.append(row)
    return dict(protocol=PROTOCOL, run_id=command['run_id'], completed=True, settings=SETTINGS,
        actual_transport=transport_log((root/'process.log').read_text()), profiles=profiles,
        gpu_us=max(row['gpu_us'] for row in rows), wall_us=max(row['wall_us'] for row in rows),
        bitwise_correct=True, evidence_files=files, full_step_measured=False,
        diagnostic_debug_only=DEBUG, training_debug='WARN')


def check_training_identity(root, hardware, communication):
    probe = p.read(Path(root)/'environment-preflight.json')
    keys = ('uuid', 'name', 'capability', 'sm_count', 'cuda_device')
    require(probe['host'] == hardware['host'] and probe['environment'] == hardware['environment']
            and probe['cuda_visible_devices'] == hardware['cuda_visible_devices']
            and [{k: d[k] for k in keys} for d in probe['devices']]
                == [{k: d[k] for k in keys} for d in hardware['devices']],
            'Probe and training hardware/software differ')
    require(communication == probe['communication'] == hardware['communication']
            and all(communication['environment'].get(k) == v for k, v in OVERRIDES.items()),
            'Probe and training transport differ')


def controller(args):
    from tp8_dp2_train import run_process
    import ablation4_runtime as comm
    require(os.environ.get('RTX5090_OP5_PROBE_EXECUTE') == '1', 'OP5 probe needs controller authorization')
    lock_path = REPO/'logs/5090/.baseline.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.out.mkdir(parents=True, exist_ok=False)
        hardware = op3.base.hardware_preflight(args)
        require(hardware['environment']['nccl'] == [2, 27, 3], 'OP5 transport is pinned to NCCL 2.27.3')
        communication = hardware['communication']
        require(all(communication['environment'].get(k) == v for k, v in OVERRIDES.items()), 'Wrong OP5 probe configuration')
        run_id = uuid.uuid4().hex
        env = comm.apply(dict(os.environ), communication)
        for key in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONOPTIMIZE'):
            env.pop(key, None)
        env.update(DEBUG, CUDA_DEVICE_MAX_CONNECTIONS='1', PYTHONNOUSERSITE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
            CUDA_VISIBLE_DEVICES=hardware['cuda_visible_devices'], CUDA_HOME=str(args.cuda_home))
        env['PATH'] = f'{args.python.parent}:{args.cuda_home}/bin:'+env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = str(args.cuda_home/'lib64')+':'+env.get('LD_LIBRARY_PATH', '')
        command = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                   '--max-restarts=0', HERE/'op5_probe.py', '--worker', '--out', args.out, '--run-id', run_id]
        p.write(args.out/'command.json', dict(protocol=PROTOCOL, run_id=run_id, settings=SETTINGS,
            communication_environment=dict(communication['environment'], **DEBUG), command=list(map(str, command))))
        op3.base.gpu_idle()
        run_process(command, REPO, env, args.out/'process.log', 600, op3.base.Peer())
        result = evidence(args.out)
        p.write(args.out/'report.json', result)
        print(json.dumps({k: result[k] for k in ('completed', 'gpu_us', 'wall_us', 'bitwise_correct')}), flush=True)


def worker(args):
    require(os.environ.get('RTX5090_OP5_PROBE_EXECUTE') == '1', 'OP5 worker requires controller execution')
    import torch
    import torch.distributed as dist
    import ablation4_runtime as comm
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    require(world == 8 and rank == local, 'OP5 probe requires single-node TP8')
    hardware = p.read(args.out/'environment-preflight.json')
    expected = dict(hardware['communication']['environment'], **DEBUG)
    actual = {k: v for k, v in os.environ.items() if comm.relevant(k)}
    require(actual == expected and comm.config_files() == hardware['communication']['config_files'], 'Probe communication changed')
    row = dict(protocol=PROTOCOL, run_id=args.run_id, rank=rank, completed=False, settings=SETTINGS)
    try:
        torch.cuda.set_device(local)
        prop = torch.cuda.get_device_properties(local)
        require(str(prop.uuid) == hardware['devices'][rank]['uuid'], 'Probe GPU changed')
        dist.init_process_group('nccl', timeout=timedelta(seconds=120))
        n = SETTINGS['input_elements']
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            index = torch.arange(n, device='cuda', dtype=torch.int32)
            source = torch.empty(n, device='cuda', dtype=torch.bfloat16)
            destination = torch.empty(8*n, device='cuda', dtype=torch.bfloat16)
            def bits(source_rank, epoch):
                return ((index*17+source_rank*101+epoch*31) % 0x7f00).to(torch.int16)
            def gather():
                work = dist.all_gather_into_tensor(destination, source, async_op=True)
                require(work is not None, 'Missing asynchronous AllGather work')
                work.wait()
            for epoch in range(SETTINGS['correctness_iterations']):
                source.view(torch.int16).copy_(bits(rank, epoch))
                if epoch == 1 and rank == 7:
                    time.sleep(.01)
                gather()
                for peer in range(8):
                    require(torch.equal(destination[peer*n:(peer+1)*n].view(torch.int16), bits(peer, epoch)),
                            f'BF16 AllGather corrupted rank {peer}, iteration {epoch}')
            for _ in range(SETTINGS['warmup']):
                gather()
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            start.record(stream)
            for _ in range(SETTINGS['timed_calls']):
                gather()
            end.record(stream)
            end.synchronize()
            wall_us = (time.perf_counter()-wall)*1e6/SETTINGS['timed_calls']
            gpu_us = start.elapsed_time(end)*1000/SETTINGS['timed_calls']
            dist.barrier()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA]) as profiler:
                for _ in range(SETTINGS['profile_calls']):
                    gather()
                torch.cuda.synchronize()
        raw = args.out/f'rank{rank}-profile.json'
        profiler.export_chrome_trace(str(raw))
        with gzip.open(args.out/f'rank{rank}-profile.json.gz', 'wb') as dest:
            dest.write(raw.read_bytes())
        raw.unlink()
        row.update(completed=True, gpu_uuid=str(prop.uuid), host=socket.gethostname(),
            environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                             nccl=list(torch.cuda.nccl.version()), python=str(Path(sys.executable).absolute())),
            communication_environment=actual, config_files=comm.config_files(),
            bitwise_correct=True, non_default_stream=True, iterations=SETTINGS['correctness_iterations'],
            gpu_us=gpu_us, wall_us=wall_us, profile_complete=True)
    except BaseException as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        p.write(args.out/f'rank{rank}.json', row)
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--cuda-home', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--build-cuda-home', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        controller(args)


if __name__ == '__main__':
    main()

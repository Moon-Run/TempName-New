#!/usr/bin/env python3
"""Two-node TP8/DP2 communication preflight, separate from any training benchmark.

Start once per container, with node-rank 0 on gxn74 and 1 on gxn75. Preview by
default; --execute tests local TP AllGather/ReduceScatter and eight cross-node
DP AllReduce groups. No Megatron import, model, optimizer or Flux build is used.
NCCL is explicitly restricted to IB; actual channel logs are checked afterwards.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import signal
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = 'h100-tp8-dp2-link-check-v1'
TP, DP, WORLD = 8, 2, 16
DEFAULT_HCA = 'mlx5_1:1,mlx5_4:1,mlx5_6:1,mlx5_8:1'
SMALL_ELEMENTS, LARGE_ELEMENTS, ITERATIONS = 1024, 4 * 1024 * 1024, 5


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups():
    return [list(range(TP)), list(range(TP, WORLD))], [[i, i+TP] for i in range(TP)]


def parse_hca(value):
    entries = value.removeprefix('=').split(',')
    require(entries and len(entries) == len(set(entries)) and
            all(re.fullmatch(r'[A-Za-z0-9_.-]+:[1-9][0-9]*', s) for s in entries),
            'HCA must be a comma-separated list of exact device:port names')
    return entries


def settings(args):
    return dict(NCCL_NET='IB', NCCL_IB_DISABLE='0', NCCL_IB_HCA='='+','.join(parse_hca(args.hca)),
                NCCL_SOCKET_IFNAME='='+args.socket_ifname, GLOO_SOCKET_IFNAME=args.socket_ifname,
                NCCL_SOCKET_FAMILY='AF_INET', NCCL_NVLS_ENABLE='0', NCCL_DEBUG='INFO',
                NCCL_DEBUG_SUBSYS='INIT,NET', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
                NCCL_DEBUG_FILE=str(args.out/'nccl.%h.%p.log'))


def command(args, script):
    return [str(args.python), '-m', 'torch.distributed.run', '--nnodes=2', '--nproc-per-node=8',
            '--node-rank='+str(args.node_rank), '--master-addr='+args.master_addr,
            '--master-port='+str(args.master_port), '--rdzv-backend=static',
            '--rdzv-conf=timeout='+str(args.rendezvous_timeout), '--max-restarts=0',
            str(script), '--worker', '--node-rank', str(args.node_rank),
            '--run-id', args.run_id, '--out', str(args.out), '--repo-root', str(args.repo_root),
            '--timeout', str(args.timeout), '--rendezvous-timeout', str(args.rendezvous_timeout),
            '--process-timeout', str(args.process_timeout), '--hca', args.hca, '--socket-ifname', args.socket_ifname]


def ensure_memlock():
    before = list(resource.getrlimit(resource.RLIMIT_MEMLOCK))
    if before[0] != resource.RLIM_INFINITY:
        try:
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except (ValueError, OSError) as error:
            raise RuntimeError('Cannot raise memlock for this process. Run ulimit -l unlimited or '
                               'configure the container memlock limit; '+str(error)) from error
    after = list(resource.getrlimit(resource.RLIMIT_MEMLOCK))
    require(after[0] == resource.RLIM_INFINITY, 'Unlimited memlock required by this preflight')
    return dict(before=before, after=after)


def local_requirements(args):
    require(args.python.is_file() and os.access(args.python, os.X_OK), 'Training Python is missing or not executable')
    require(Path('/sys/class/net', args.socket_ifname).exists(), 'Bootstrap network interface is missing')
    hcas = []
    for item in parse_hca(args.hca):
        device, port = item.split(':')
        path = Path('/sys/class/infiniband')/device/'ports'/port
        state = (path/'state').read_text().strip()
        layer = (path/'link_layer').read_text().strip()
        require(state.startswith('4:') and layer == 'InfiniBand', 'Requested IB port is not ACTIVE: '+item)
        hcas.append(dict(port=item, state=state, link_layer=layer, rate=(path/'rate').read_text().strip()))
    require(any(Path('/dev/infiniband').glob('uverbs*')), 'No RDMA userspace device is visible in this container')
    memlock = ensure_memlock()
    check = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name',
                            '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
    require(check.returncode == 0, 'Cannot inspect current GPU processes: '+check.stderr)
    require(not check.stdout.strip(), 'GPU compute processes are already present; retry when both nodes are idle:\n'+check.stdout)
    return dict(hcas=hcas, memlock_before=memlock['before'], memlock_after=memlock['after'], gpu_compute_processes=[])


def validate_metadata(records):
    require(len(records) == WORLD and {r['rank'] for r in records} == set(range(WORLD)),
            'Require all 16 distinct global ranks')
    ordered = sorted(records, key=lambda r: r['rank'])
    shared = ('protocol', 'run_id', 'worker_sha256', 'torch', 'cuda', 'nccl', 'hca', 'socket_ifname')
    require(all(all(r[k] == ordered[0][k] for k in shared) for r in ordered),
            'Nodes disagree on worker, run ID, software or requested network configuration')
    require(ordered[0]['protocol'] == PROTOCOL and ordered[0]['torch'].split('+')[0] == '2.6.0'
            and ordered[0]['cuda'] == '12.4' and ordered[0]['nccl'] == [2, 21, 5],
            'This preflight requires the probed PyTorch/CUDA/NCCL versions')
    hosts = [{r['host'] for r in ordered[node*TP:(node+1)*TP]} for node in range(DP)]
    require(all(len(h) == 1 for h in hosts) and hosts[0].isdisjoint(hosts[1]),
            'Ranks must occupy two different hosts, eight consecutive ranks per host')
    require(len({r['gpu_uuid'] for r in ordered}) == WORLD, 'GPU UUIDs are duplicated across ranks')
    for row in ordered:
        rank = row['rank']
        require(row['node_rank'] == rank//TP and row['local_rank'] == rank % TP
                and row['tp_members'] == list(range(rank//TP*TP, (rank//TP+1)*TP))
                and row['dp_members'] == [rank % TP, rank % TP+TP], 'Wrong TP/DP rank mapping')
        require(row['device_count'] == TP and 'H100' in row['gpu_name'] and row['sm_count'] == 132
                and row['capability'] == [9, 0] and row['local_peer_access'], 'Require eight full peer-accessible H100s per host')
    return ordered


def transport_evidence(log_text):
    ib = [line for line in log_text.splitlines() if re.search(r'\bvia NET/IB(?:/|\s|$)', line)]
    sockets = [line for line in log_text.splitlines() if re.search(r'\bvia NET/Socket(?:/|\s|$)', line)]
    return dict(ib_channels_observed=bool(ib), socket_channels_observed=bool(sockets),
                gdrdma_channels_observed=any('GDRDMA' in line for line in ib),
                ib_channel_examples=ib[:4], socket_channel_examples=sockets[:4])


def validate_results(records):
    records = validate_metadata(records)
    expected = {'tp_allgather', 'tp_reduce_scatter', 'dp_allreduce_small', 'dp_allreduce_large'}
    for row in records:
        require(row.get('completed') is True and set(row.get('checks', {})) == expected and
                all(value is True for value in row['checks'].values()), 'Missing or failed collective checks')
        require(row.get('large_elements') == LARGE_ELEMENTS and row.get('large_iterations') == ITERATIONS,
                'Incomplete cross-node large-message check')
    return records


def run_worker(args):
    require(os.environ.get('H100_TP8_DP2_LINK_EXECUTE') == '1', 'Use the launcher with --execute')
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    require(int(os.environ['WORLD_SIZE']) == WORLD and int(os.environ['LOCAL_WORLD_SIZE']) == TP,
            'Launch exactly two nodes with eight workers each')
    require(rank//TP == args.node_rank and rank % TP == local_rank, 'Global/local rank mapping changed')
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    require(torch.cuda.device_count() == TP, 'Expose exactly eight local GPUs')
    require(all(os.environ.get(key) == value for key, value in settings(args).items()),
            'Communication environment differs from the requested preflight')
    props = torch.cuda.get_device_properties(local_rank)
    tp_groups, dp_groups = groups()
    row = dict(protocol=PROTOCOL, run_id=args.run_id, worker_sha256=sha(__file__),
        rank=rank, node_rank=args.node_rank, local_rank=local_rank, host=socket.gethostname(), pid=os.getpid(),
        python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version()),
        device_count=torch.cuda.device_count(), gpu_uuid=str(props.uuid), gpu_name=props.name,
        sm_count=props.multi_processor_count, capability=[props.major, props.minor],
        local_peer_access=all(torch.cuda.can_device_access_peer(local_rank, j) for j in range(TP) if j != local_rank),
        tp_members=tp_groups[args.node_rank], dp_members=dp_groups[local_rank],
        hca=args.hca, socket_ifname=args.socket_ifname, memlock=list(resource.getrlimit(resource.RLIMIT_MEMLOCK)),
        large_elements=LARGE_ELEMENTS, large_iterations=ITERATIONS, completed=False, checks={})
    try:
        dist.init_process_group('gloo', timeout=timedelta(seconds=args.timeout))
        records = [None]*WORLD
        dist.all_gather_object(records, row)
        validate_metadata(records)
        communicators = []
        tp_group = dp_group = None
        for kind, layouts in (('tp', tp_groups), ('dp', dp_groups)):
            for index, members in enumerate(layouts):
                group = dist.new_group(members, backend='nccl', timeout=timedelta(seconds=args.timeout),
                                       group_desc=f'tp8-dp2-link-{kind}{index}')
                if rank in members:
                    communicators.append(group)
                    if kind == 'tp':
                        tp_group = group
                    else:
                        dp_group = group
        dist.barrier()
        value = torch.full((SMALL_ELEMENTS,), float(local_rank+1), device='cuda')
        gathered = torch.empty((TP*SMALL_ELEMENTS,), device='cuda')
        dist.all_gather_into_tensor(gathered, value, group=tp_group)
        expected = torch.arange(1, TP+1, device='cuda', dtype=torch.float32).repeat_interleave(SMALL_ELEMENTS)
        row['checks']['tp_allgather'] = bool(torch.equal(gathered, expected))
        require(row['checks']['tp_allgather'], 'Local TP AllGather values differ')
        input_ = expected*(local_rank+1)
        output = torch.empty_like(value)
        dist.reduce_scatter_tensor(output, input_, group=tp_group)
        row['checks']['tp_reduce_scatter'] = bool(torch.all(output == sum(range(1, TP+1))*(local_rank+1)).item())
        require(row['checks']['tp_reduce_scatter'], 'Local TP ReduceScatter values differ')
        dist.barrier()
        value.fill_(rank+1)
        dist.all_reduce(value, group=dp_group)
        total = sum(member+1 for member in row['dp_members'])
        row['checks']['dp_allreduce_small'] = bool(torch.all(value == total).item())
        require(row['checks']['dp_allreduce_small'], 'Cross-node DP small AllReduce values differ')
        value = torch.empty((LARGE_ELEMENTS,), dtype=torch.float32, device='cuda')
        for iteration in range(ITERATIONS):
            value.fill_(rank+1+iteration)
            dist.all_reduce(value, group=dp_group)
            require(bool(torch.all(value == total+DP*iteration).item()),
                    'Cross-node DP large AllReduce values differ at iteration '+str(iteration))
        row['checks']['dp_allreduce_large'] = True
        torch.cuda.synchronize()
        row['completed'] = True
        dist.all_gather_object(records, row)
        records = validate_results(records)
        write(args.out/f'rank{rank}.json', row)
        if local_rank == 0:
            write(args.out/'all-ranks.json', records)
            print('TP8_DP2_LINK_COLLECTIVES_PASSED', socket.gethostname(), flush=True)
        dist.barrier()
        for group in reversed(communicators):
            dist.destroy_process_group(group)
        dist.destroy_process_group()
    except BaseException as error:
        row.update(completed=False, error=type(error).__name__+': '+str(error))
        write(args.out/f'rank{rank}-failed.json', row)
        raise


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    state = dict(protocol=PROTOCOL, run_id=args.run_id, node_rank=args.node_rank,
                 status='preparing', training_started=False, collectives_passed=False, ib_transport_validated=False)
    write(args.out/'state.json', state)
    try:
        prerequisites = local_requirements(args)
        write(args.out/'local-preflight.json', prerequisites)
        frozen = args.out/'link-worker.py'
        shutil.copy2(__file__, frozen)
        frozen_sha = sha(frozen)
        cmd = command(args, frozen)
        network = settings(args)
        env = dict(os.environ, **network, H100_TP8_DP2_LINK_EXECUTE='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONNOUSERSITE='1')
        env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
        metadata = subprocess.run(['git', '-C', str(args.repo_root), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, timeout=10)
        write(args.out/'launch.json', dict(command=cmd, network=network, repo_commit=metadata.stdout.strip(),
              worker_sha256=frozen_sha, parameters={key: str(value) if isinstance(value, Path) else value
                                                    for key, value in vars(args).items()}))
        state['status'] = 'running'
        write(args.out/'state.json', state)
        print('Started communication preflight; launch the other node now. Log: '+str(args.out/'process.log'), flush=True)
        with (args.out/'process.log').open('x') as log:
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                result = process.wait(timeout=args.process_timeout)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
        require(result == 0, f'torchrun exited {result}; inspect process.log and nccl.*.log')
        records = validate_results(json.loads((args.out/'all-ranks.json').read_text()))
        require(sha(frozen) == frozen_sha and all(row['worker_sha256'] == frozen_sha
                and row['run_id'] == args.run_id and row['hca'] == args.hca
                and row['socket_ifname'] == args.socket_ifname for row in records),
                'Worker results differ from this launch source or configuration')
        state['collectives_passed'] = True
        write(args.out/'state.json', state)
        evidence = []
        for rank in range(args.node_rank*TP, (args.node_rank+1)*TP):
            row = json.loads((args.out/f'rank{rank}.json').read_text())
            require(row == records[rank], 'Local rank report differs from the globally exchanged record')
            files = list(args.out.glob(f'nccl.*.{row["pid"]}.log'))
            require(len(files) == 1, 'Missing unique NCCL log for rank '+str(rank))
            evidence.append(dict(rank=rank, log=str(files[0]), sha256=sha(files[0]),
                                 **transport_evidence(files[0].read_text(errors='replace'))))
        state.update(ib_transport_validated=all(
            item['ib_channels_observed'] and not item['socket_channels_observed'] for item in evidence))
        write(args.out/'transport.json', dict(ranks=evidence, ib_transport_validated=state['ib_transport_validated'],
            note='IB channel evidence does not prove peak bandwidth or GPUDirect RDMA on every channel.'))
        require(state['ib_transport_validated'], 'Collectives passed but actual IB channel evidence is incomplete; inspect transport.json')
        state.update(status='completed', stage='done', expected_global_ranks=WORLD,
                     global_collectives_validated=True, local_rank_logs_validated=TP)
        write(args.out/'state.json', state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
        print('本节点日志验证通过；两台 state.json 都 completed 后，才完成双节点通信预检。', flush=True)
        return state
    except BaseException as error:
        state.update(status='failed', error=type(error).__name__+': '+str(error))
        write(args.out/'state.json', state)
        raise


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--python', type=Path, help='Training Python; required for --execute')
    parser.add_argument('--master-addr', default='10.18.133.74')
    parser.add_argument('--master-port', type=int, default=29574)
    parser.add_argument('--socket-ifname', default='bond0')
    parser.add_argument('--hca', default=DEFAULT_HCA)
    parser.add_argument('--run-id', default='tp8-dp2-link-20261028')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--timeout', type=int, default=90, help='Distributed collective timeout in seconds')
    parser.add_argument('--rendezvous-timeout', type=int, default=180)
    parser.add_argument('--process-timeout', type=int, default=420)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--repo-root', type=Path, default=REPO, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require(1 <= args.master_port <= 65535 and args.timeout > 0 and args.rendezvous_timeout > 0
            and args.process_timeout > args.rendezvous_timeout+args.timeout, 'Invalid port or timeouts')
    require(re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', args.run_id) and args.run_id not in ('.', '..'), 'Invalid run ID')
    require(re.fullmatch(r'[A-Za-z0-9_.:-]{1,80}', args.socket_ifname), 'Invalid bootstrap interface')
    args.hca = ','.join(parse_hca(args.hca))
    require(not args.execute or args.python is not None, 'Provide --python with --execute')
    args.python = (args.python or Path(sys.executable)).expanduser().absolute()
    args.repo_root = args.repo_root.expanduser().resolve()
    if args.out is None:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')
        args.out = args.repo_root/'logs/h100/phase6'/f'{args.run_id}-node{args.node_rank}-{stamp}'
    args.out = args.out.expanduser().resolve()
    return args


def main(argv=None):
    args = arguments(argv)
    if args.worker:
        return run_worker(args)
    print('Output:', args.out, flush=True)
    if not args.execute:
        print('PLAN ONLY: no output directory, GPU queries, resource-limit changes or workers.')
        print(json.dumps(settings(args), indent=2))
        print(shlex.join(command(args, Path(__file__).resolve())))
        print('Two local TP8 groups; eight cross-node DP2 groups. No training or performance verdict.')
        return
    return execute(args)


if __name__ == '__main__':
    main()

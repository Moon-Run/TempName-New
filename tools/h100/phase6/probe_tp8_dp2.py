#!/usr/bin/env python3
"""Collect local container facts for the MA11/basic-quant/Megatron-CUDA TP8 DP2 test.

Run inside BOTH training containers. This does not launch training, allocate
test tensors, start a listener, configure networking, or run collectives.
Visible RDMA devices and local peer access do not establish cross-node connectivity.
Only explicit repository paths are hashed; there is no repository-wide status scan.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import ipaddress
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
PREFIX = 'TP8_DP2_TORCH_JSON='
SOURCE_NAMES = (
    'ma11_adapter.py', 'ma11_sources.py', 'ma11_softmax.cuh', 'ma11_fullstep.py',
    'ma8_adapter.py', 'ma9_adapter.py', 'ma9_zero_count.cuh', 'ma2_cr2_sources.py',
    'basic_quant_codec.py', 'basic_quant_sources.py', 'basic_quant_receiver.cuh',
    'baseline_codec.py', 'baseline_codec.cu', 'baseline_sources.py',
    'probe_tp8_dp2.py', 'probe_tp8_dp2.sh',
)
ENV_NAMES = (
    'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDA_PATH', 'CONDA_PREFIX', 'VIRTUAL_ENV',
    'MA11_PYTHON', 'H100_RECOVERY_PYTHON', 'H100_RECOVERY_BUILD_CUDA',
    'TP8_DP2_PYTHON', 'CUDA_DEVICE_MAX_CONNECTIONS', 'NCCL_SOCKET_IFNAME',
    'GLOO_SOCKET_IFNAME', 'NCCL_IB_HCA', 'NCCL_IB_DISABLE', 'NCCL_NET',
    'NCCL_IB_GID_INDEX', 'NCCL_NVLS_ENABLE', 'NCCL_P2P_DISABLE',
)
TORCH_PROBE = r'''
import json, sys
result = {'python': sys.executable, 'python_version': sys.version.split()[0]}
try:
    import torch
    result.update(torch=torch.__version__, torch_path=torch.__file__,
                  cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
    try:
        result['nccl'] = torch.cuda.nccl.version()
    except Exception as e:
        result['nccl_error'] = str(e)
    result['devices'] = []
    if result['cuda_available']:
        count = torch.cuda.device_count()
        for i in range(count):
            p = torch.cuda.get_device_properties(i)
            result['devices'].append(dict(index=i, name=p.name, uuid=str(getattr(p, 'uuid', '')),
                sm_count=p.multi_processor_count, capability=[p.major, p.minor],
                memory_gib=p.total_memory/2**30))
        result['local_peer_access'] = [[True if i == j else torch.cuda.can_device_access_peer(i,j)
                                       for j in range(count)] for i in range(count)]
    result['import_ok'] = True
except Exception as e:
    result.update(import_ok=False, error=type(e).__name__+': '+str(e))
print('TP8_DP2_TORCH_JSON='+json.dumps(result), flush=True)
'''


def run_command(argv, timeout=10, env=None):
    """Bounded subprocess; also remove its children if a diagnostic times out."""
    argv = list(map(str, argv))
    record = dict(command=argv)
    try:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, errors='replace', start_new_session=True, env=env)
    except OSError as error:
        return dict(record, status='unavailable', error=str(error), returncode=None)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        status = 'ok' if process.returncode == 0 else 'error'
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        status = 'timeout'
    limit = 128 * 1024
    return dict(record, status=status, returncode=process.returncode,
                stdout=stdout[:limit], stderr=stderr[:limit],
                truncated=len(stdout) > limit or len(stderr) > limit)


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def recovery_values(repo):
    """Read only literal environment paths; never execute/source the recovery file."""
    text = read_text(repo/'outputs/h100/container-recovery/env.sh') or ''
    result = {}
    for line in text.splitlines():
        match = re.match(r'\s*(?:export\s+)?(H100_RECOVERY_PYTHON|H100_RECOVERY_BUILD_CUDA|CUDA_HOME)=(.*)', line)
        if not match:
            continue
        key, raw = match.groups()
        if '$' in raw or '`' in raw:
            continue
        try:
            value = shlex.split(raw, comments=True)
        except ValueError:
            continue
        if len(value) == 1:
            result[key] = value[0]
    return result


def python_candidates(explicit, recovery):
    if explicit:
        paths = [str(explicit)]
    else:
        paths = [os.environ.get(name) for name in
                 ('TP8_DP2_PYTHON', 'MA11_PYTHON', 'H100_RECOVERY_PYTHON')]
        paths += [recovery.get('H100_RECOVERY_PYTHON')]
        paths += [str(Path(os.environ[name])/'bin/python') for name in
                  ('CONDA_PREFIX', 'VIRTUAL_ENV') if os.environ.get(name)]
        paths += ['/workspace/conda-envs/flux-h100-ablation3-20261017/bin/python']
        # Known container environment directories only, two levels deep.
        for root in ('/workspace/conda-envs', '/opt/conda/envs'):
            found = sorted(Path(root).glob('*/bin/python'), key=lambda p: ('flux' not in str(p), str(p)))
            paths += [str(p) for p in found[:8]]
        paths += [sys.executable, shutil.which('python'), shutil.which('python3'), '/opt/conda/bin/python']
    result = []
    for value in paths:
        if value:
            # Do not resolve symlinks: a virtualenv's Python path selects its environment.
            path = str(Path(value).expanduser().absolute())
            if path not in result:
                result.append(path)
    return result


def local_gpu_checks(info):
    devices = info.get('devices', [])
    peers = info.get('local_peer_access', [])
    return dict(
        pinned_torch_260=info.get('torch', '').split('+')[0] == '2.6.0',
        cuda_runtime_124=info.get('cuda_runtime') == '12.4',
        eight_full_h100=(len(devices) == 8 and all('H100' in p.get('name', '') and
            p.get('capability') == [9, 0] and p.get('sm_count') == 132 for p in devices)),
        all_local_peer_pairs=(len(peers) == 8 and all(len(row) == 8 and all(row) for row in peers)),
    )


def probe_python(candidates, timeout):
    results, selected = [], None
    env = dict(os.environ, PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTORCH_NVML_BASED_CUDA_CHECK='1')
    attempted = 0
    for path in candidates:
        if not Path(path).is_file() or not os.access(path, os.X_OK):
            results.append(dict(path=path, status='missing'))
            continue
        if attempted >= 4:
            results.append(dict(path=path, status='not_probed_limit'))
            continue
        attempted += 1
        result = dict(path=path, **run_command([path, '-I', '-c', TORCH_PROBE], timeout, env))
        for line in result.get('stdout', '').splitlines():
            if line.startswith(PREFIX):
                try:
                    result['info'] = json.loads(line[len(PREFIX):])
                except json.JSONDecodeError:
                    result['parse_error'] = True
        info = result.get('info', {})
        result['checks'] = local_gpu_checks(info)
        results.append(result)
        if result['status'] == 'ok' and info.get('import_ok'):
            if selected is None or (not selected['info'].get('cuda_available') and info.get('cuda_available')):
                selected = result
            if all(result['checks'].values()):
                selected = result
                break
    return dict(candidates=candidates, probes=results, selected=selected)


def network_addresses(command):
    addresses = []
    try:
        interfaces = json.loads(command.get('stdout', ''))
        for interface in interfaces:
            for addr in interface.get('addr_info', []):
                value = addr.get('local', '')
                ip = ipaddress.ip_address(value)
                if not ip.is_loopback and not ip.is_link_local and not ip.is_unspecified:
                    addresses.append(dict(interface=interface['ifname'], address=value,
                        family=addr.get('family'), prefixlen=addr.get('prefixlen'),
                        state=interface.get('operstate'), scope=addr.get('scope')))
    except (ValueError, TypeError, KeyError):
        pass
    return addresses


def system_ipv4_addresses():
    """Linux ioctl fallback for minimal containers without iproute2. Sends no traffic."""
    import fcntl
    import struct
    result = []
    try:
        interfaces = socket.if_nameindex()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            for _, name in interfaces:
                request = struct.pack('256s', name.encode()[:15])
                try:
                    address = socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, request)[20:24])
                    mask = socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x891b, request)[20:24])
                    ip = ipaddress.ip_address(address)
                    if ip.is_loopback or ip.is_link_local or ip.is_unspecified:
                        continue
                    result.append(dict(interface=name, address=address, family='inet', scope='unknown',
                        prefixlen=ipaddress.IPv4Network('0.0.0.0/'+mask).prefixlen,
                        state=read_text(Path('/sys/class/net')/name/'operstate'), source='local_ioctl'))
                except (OSError, ValueError):
                    continue
    except OSError:
        pass
    return result


def rdma_devices(sys_root=Path('/sys/class/infiniband'), dev_root=Path('/dev/infiniband')):
    devices = []
    for path in sorted(sys_root.glob('*')):
        ports = [{**{key: read_text(port/key) for key in ('state', 'phys_state', 'rate', 'link_layer')},
                  'port': port.name} for port in sorted((path/'ports').glob('*'))]
        devices.append(dict(name=path.name, ports=ports,
                            netdevs=[p.name for p in sorted((path/'device/net').glob('*'))]))
    nodes = [dict(path=str(path), readable=os.access(path, os.R_OK), writable=os.access(path, os.W_OK))
             for path in sorted(dev_root.glob('*'))]
    return dict(sysfs_devices=devices, device_nodes=nodes,
                note='Device visibility/access bits only; RDMA and cross-node NCCL have not been tested.')


def command_plan(repo, megatron, recovery):
    commands = {
        'gpu_inventory': ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version', '--format=csv,noheader,nounits'],
        'gpu_instances': ['nvidia-smi', '-L'],
        'gpu_processes': ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory', '--format=csv,noheader,nounits'],
        'gpu_topology': ['nvidia-smi', 'topo', '-m'],
        'ip_addresses': ['ip', '-j', 'address', 'show'],
        'ip_routes': ['ip', '-j', 'route', 'show'],
        'hostname_addresses': ['hostname', '-I'],
        'ibstat': ['ibstat'], 'ib_devices': ['ibv_devices'],
        'ib_netdevs': ['ibdev2netdev'], 'rdma_links': ['rdma', 'link', 'show'],
        'resolve_gxn74': ['getent', 'ahostsv4', 'gxn74'],
        'resolve_gxn75': ['getent', 'ahostsv4', 'gxn75'],
        'repo_commit': ['git', '-C', str(repo), 'rev-parse', 'HEAD'],
        'megatron_commit': ['git', '-C', str(megatron), 'rev-parse', 'HEAD'],
        'cutlass_commit': ['git', '-C', str(repo/'3rdparty/cutlass'), 'rev-parse', 'HEAD'],
        'nccl_source_commit': ['git', '-C', str(repo/'3rdparty/nccl'), 'rev-parse', 'HEAD'],
        'repo_mount': ['findmnt', '-J', '-T', str(repo), '-o', 'TARGET,SOURCE,FSTYPE'],
    }
    compilers = [os.environ.get('H100_RECOVERY_BUILD_CUDA'), recovery.get('H100_RECOVERY_BUILD_CUDA'),
                 '/usr/local/cuda-12.8', os.environ.get('CUDA_HOME'), recovery.get('CUDA_HOME')]
    existing = list(dict.fromkeys(str(Path(p)/'bin/nvcc') for p in compilers if p))
    for i, nvcc in enumerate(existing):
        if Path(nvcc).is_file():
            commands['nvcc_'+str(i)] = [nvcc, '--version']
    if not any(k.startswith('nvcc_') for k in commands):
        commands['nvcc_path'] = ['nvcc', '--version']
    return commands


def summary(record):
    chosen = record['python'].get('selected') or {}
    info = chosen.get('info', {})
    commands = record['commands']
    lines = [f"TP8 DP2 环境探测：{record['label']}（容器 hostname={record['hostname']}）",
             f"仓库：{record['repo']}", f"Megatron：{record['megatron']}",
             '仓库 commit：'+commands['repo_commit'].get('stdout', '').strip(),
             'Megatron commit：'+commands['megatron_commit'].get('stdout', '').strip(),
             '训练 Python：'+chosen.get('path', '已跳过 Torch 探测' if record['python'].get('skipped')
                                         else '未找到完成 Torch/CUDA 探测的环境；查看下方诊断'),
             f"PyTorch={info.get('torch', '未知')} CUDA={info.get('cuda_runtime', '未知')} NCCL={info.get('nccl', '未知')}",
             '本地 GPU 检查：'+json.dumps(chosen.get('checks', {}), ensure_ascii=False),
             '当前 GPU：\n'+commands['gpu_inventory'].get('stdout', commands['gpu_inventory'].get('error', '未知')).strip(),
             '当前 GPU 进程（快照）：\n'+(commands['gpu_processes'].get('stdout', '').strip() or
                 ('未列出计算进程' if commands['gpu_processes']['status'] == 'ok' else '查询失败；查看 report.json')),
             '候选网卡 IP（尚未验证互通）：']
    lines += [f"  {v['interface']}: {v['address']}/{v.get('prefixlen')} [{v.get('state')}]"
              for v in record['network_addresses']]
    if not record['network_addresses']:
        lines += ['  ip 工具未返回可用地址；hostname -I：'+commands['hostname_addresses'].get('stdout', '').strip()]
    lines += ['CUDA 编译器：'+json.dumps([{ 'path': value['command'][0], 'status': value['status'],
               'release': re.findall(r'release ([0-9.]+)', value.get('stdout', ''))}
              for key, value in commands.items() if key.startswith('nvcc_')], ensure_ascii=False)]
    if not chosen and not record['python'].get('skipped'):
        lines += ['Python 诊断：'+json.dumps([dict(path=p['path'], status=p['status'],
                   error=p.get('info', {}).get('error') or p.get('error') or p.get('stderr', '')[:500])
                  for p in record['python'].get('probes', [])], ensure_ascii=False)]
    lines += ['RDMA 网卡：'+json.dumps(record['rdma']['sysfs_devices'], ensure_ascii=False),
              '容器 RDMA 设备：'+json.dumps(record['rdma']['device_nodes'], ensure_ascii=False),
              'memlock soft/hard：'+str(record['memlock_bytes']),
              '本次未进行双节点连接或 NCCL 通信测试，未启动训练。',
              '正式比较范围：MA11、Flux＋基础量化 v2、Megatron＋CUDA量化；TP8 DP2，global batch=4，micro batch=1。',
              '请返回两台的 summary.txt；详细信息保存在 report.json。']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--label', default=socket.gethostname(), help='Physical node label, e.g. gxn74 or gxn75')
    parser.add_argument('--out', type=Path, help='Fresh output directory; defaults to outputs/h100/phase6')
    parser.add_argument('--python', type=Path, help='Explicit training Python; otherwise discover known environment paths')
    parser.add_argument('--megatron', type=Path, default=REPO.parent/'Megatron-LM')
    parser.add_argument('--command-timeout', type=float, default=10)
    parser.add_argument('--torch-timeout', type=float, default=45)
    parser.add_argument('--skip-torch', action='store_true', help='Only system diagnostics; no Torch import or CUDA runtime query')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', args.label) or args.label in ('.', '..'):
        parser.error('--label must be a short hostname-like identifier')
    if not (0 < args.command_timeout <= 60 and 0 < args.torch_timeout <= 120):
        parser.error('command timeout must be in (0,60]; torch timeout in (0,120]')
    stamp = datetime.now(timezone(timedelta(hours=8)))
    out = args.out or REPO/'outputs/h100/phase6'/('tp8-dp2-probe-'+args.label+'-'+stamp.strftime('%Y%m%d-%H%M%S-%f'))
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    print('环境探测输出：'+str(out), flush=True)
    recovery = recovery_values(REPO)
    commands = command_plan(REPO, args.megatron.expanduser().resolve(), recovery)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {key: pool.submit(run_command, cmd, args.command_timeout) for key, cmd in commands.items()}
        results = {key: future.result() for key, future in futures.items()}
    candidates = python_candidates(args.python, recovery)
    print('本地系统信息已采集；正在检查训练 Python。', flush=True)
    python = (dict(candidates=candidates, probes=[], selected=None, skipped=True) if args.skip_torch
              else probe_python(candidates, args.torch_timeout))
    sources = {name: hashlib.sha256((REPO/'tools/h100/phase6'/name).read_bytes()).hexdigest()
               for name in SOURCE_NAMES if (REPO/'tools/h100/phase6'/name).is_file()}
    record = dict(protocol='h100-tp8-dp2-local-environment-probe-v1', date=stamp.isoformat(),
        label=args.label, hostname=socket.gethostname(), repo=str(REPO),
        megatron=str(args.megatron.expanduser().resolve()), probe_python=sys.executable,
        container_markers={str(p): p.exists() for p in (Path('/.dockerenv'), Path('/run/.containerenv'))},
        container_network_mode='unknown; confirm host/bridge mode separately',
        environment={key: os.environ[key] for key in ENV_NAMES if key in os.environ},
        recovery_paths=recovery, commands=results, python=python,
        network_addresses=network_addresses(results['ip_addresses']) or system_ipv4_addresses(), rdma=rdma_devices(),
        memlock_bytes=list(resource.getrlimit(resource.RLIMIT_MEMLOCK)), source_sha256=sources,
        cross_node_connectivity_tested=False, nccl_collectives_tested=False,
        training_started=False, output=str(out))
    (out/'report.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    text = summary(record)
    (out/'summary.txt').write_text(text)
    print(text, end='', flush=True)
    return out


if __name__ == '__main__':
    main()

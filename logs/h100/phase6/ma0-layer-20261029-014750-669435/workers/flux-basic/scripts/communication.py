"""Freeze and verify NCCL settings without importing Torch or touching GPUs."""
import hashlib
import os
from pathlib import Path


def relevant(key):
    return key.startswith(('NCCL_', 'TORCH_NCCL_')) or key == 'GLOO_SOCKET_IFNAME'


def config_files():
    paths = [Path('/etc/nccl.conf'), Path.home()/'.nccl.conf']
    if os.environ.get('NCCL_CONF_FILE'):
        paths.append(Path(os.environ['NCCL_CONF_FILE']))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None for p in paths}


def capture(nvls='0'):
    if nvls not in ('0', 'auto'):
        raise ValueError('NVLS must be 0 or auto')
    env = {k: v for k, v in os.environ.items() if relevant(k)}
    for k in ('NCCL_NVLS_ENABLE', 'NCCL_DEBUG_FILE', 'NCCL_DEBUG_SUBSYS'):
        env.pop(k, None)
    if nvls == '0':
        env['NCCL_NVLS_ENABLE'] = '0'
    env.update(NCCL_DEBUG='WARN')
    env.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    return dict(environment=env, config_files=config_files(), nvls_request=nvls)


def validate(record):
    env = record['environment']
    if (record['nvls_request'] not in ('0', 'auto') or
            env.get('NCCL_NVLS_ENABLE') != ('0' if record['nvls_request'] == '0' else None) or
            env.get('NCCL_DEBUG') != 'WARN' or 'NCCL_DEBUG_FILE' in env or 'NCCL_DEBUG_SUBSYS' in env or
            not all(relevant(k) and isinstance(v, str) for k, v in env.items())):
        raise ValueError('Frozen communication settings changed')
    return record


def apply(env, record):
    validate(record)
    result = {k: v for k, v in env.items() if not relevant(k)}
    result.update(record['environment'])
    return result


def verify(record):
    validate(record)
    actual = {k: v for k, v in os.environ.items() if relevant(k)}
    if actual != record['environment'] or config_files() != record['config_files']:
        raise RuntimeError('Communication environment/config changed since prepare')
    return dict(environment=actual, config_files=config_files(), nvls_request=record['nvls_request'])

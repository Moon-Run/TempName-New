"""CPU-only configuration and topology validation for the reusable E2E runner."""
import argparse
import json
import math
from pathlib import Path

POLICIES = ['native', 'original', 'remote_first', 'interleaved']

def validate(c):
    for key in ['nodes', 'gpus_per_node', 'tp', 'layers', 'hidden', 'ffn', 'heads',
                'sequence', 'micro_batch', 'global_batch', 'warmup_steps', 'timed_steps', 'blocks', 'vocab']:
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if c['tp'] < 2:
        raise ValueError('sequence parallel runner requires TP >= 2')
    world = c['nodes'] * c['gpus_per_node']
    if world % c['tp']:
        raise ValueError('world_size must be divisible by TP (PP=CP=1)')
    c['world_size'] = world
    c['dp'] = world // c['tp']
    local = c['gpus_per_node']
    if local % c['tp'] and c['tp'] % local:
        raise ValueError('TP groups must fit inside a node or span whole nodes')
    c['tp_nodes'] = max(1, c['tp'] // local)
    c['local_tp'] = min(c['tp'], local)
    if c['hidden'] % c['heads'] or c['heads'] % c['tp'] or c['ffn'] % c['tp']:
        raise ValueError('hidden/heads and heads/TP and FFN/TP must be integral')
    if c['sequence'] % c['tp'] or c['global_batch'] % (c['micro_batch'] * c['dp']):
        raise ValueError('sequence/TP and global_batch/(micro_batch*DP) must be integral')
    if not c['policies'] or len(set(c['policies'])) != len(c['policies']) or any(p not in POLICIES for p in c['policies']):
        raise ValueError('policies must be a nonempty unique subset of known policies')
    if any(p != 'native' for p in c['policies']):
        if c['local_tp'] not in (2, 4, 8):
            raise ValueError('frozen Flux mapping supports local TP=2/4/8')
        # Existing overlays read physical LOCAL_RANK; sub-node TP needs an additional audit.
        if c['tp'] < local:
            raise ValueError('Flux currently requires TP >= gpus_per_node; native allows sub-node TP')
        if c['sequence'] * c['micro_batch'] % (128 * c['tp']):
            raise ValueError('Flux requires M divisible by 128*TP for frozen tile mappings')
        if c['tp_nodes'] > 1 and c['flux_transport'] != 'hierarchical':
            raise ValueError('cross-node Flux TP requires --flux-transport hierarchical')
    if c['flux_transport'] not in ('local', 'hierarchical'):
        raise ValueError('unknown Flux transport')
    if any(not math.isfinite(c[k]) or c[k] < 0 for k in ('atol', 'rtol')):
        raise ValueError('numerical tolerances must be nonnegative')
    c['microsteps'] = c['global_batch'] // (c['micro_batch'] * c['dp'])
    c['tokens_per_step'] = c['global_batch'] * c['sequence']
    c['orders'] = [c['policies'][i % len(c['policies']):] + c['policies'][:i % len(c['policies'])]
                   for i in range(c['blocks'])]
    if c['blocks'] % len(c['policies']):
        raise ValueError('blocks must be a multiple of policy count for position balance')
    return c

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='JSON defaults; explicit CLI options override')
    for key, default in dict(nodes=2, gpus_per_node=4, tp=4, layers=32, hidden=4096,
                             ffn=16384, heads=32, sequence=2048, micro_batch=1, global_batch=4,
                             warmup_steps=10, timed_steps=20, blocks=4, seed=1234,
                             token_seed=4321, vocab=32768).items():
        p.add_argument('--'+key.replace('_', '-'), type=int, default=default)
    p.add_argument('--policies', nargs='+', choices=POLICIES, default=POLICIES)
    p.add_argument('--flux-transport', choices=['local', 'hierarchical'], default='local')
    p.add_argument('--atol', type=float, default=0.02)
    p.add_argument('--rtol', type=float, default=0.02)
    p.add_argument('--stage', choices=['preflight', 'timing'], default='preflight')
    p.add_argument('--preflight', type=Path)
    p.add_argument('--submit', action='store_true', help='submit; default only prints resolved plan')
    p.add_argument('--time', default='01:00:00')
    p.add_argument('--cpus-per-task', type=int, default=16)
    p.add_argument('--python', type=Path)
    return p

def resolve(argv=None):
    p = parser()
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        defaults = json.loads(preliminary.config.read_text())
        # Result config.json is reusable; topology/accounting is always recomputed.
        for key in ('world_size', 'dp', 'tp_nodes', 'local_tp', 'microsteps', 'tokens_per_step', 'orders'):
            defaults.pop(key, None)
        allowed = {a.dest for a in p._actions} - {'submit', 'config', 'help'}
        if set(defaults) - allowed:
            raise ValueError(f'unknown config keys: {set(defaults)-allowed}')
        p.set_defaults(**defaults)
    a = p.parse_args(argv)
    c = {k: v for k, v in vars(a).items() if k not in
         ('config', 'submit', 'stage', 'preflight', 'time', 'cpus_per_task', 'python')}
    return a, validate(c)

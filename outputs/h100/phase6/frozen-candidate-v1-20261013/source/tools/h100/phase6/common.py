"""CPU-only configuration and provenance for the single-node H100 port."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MEGATRON_COMMIT = '3ea68ad6042cc1204386ae9364358f7c4de1bc37'
BASES = ('remote_first', 'interleaved', 'interleaved_remote', 'interleaved_remote_group')
CONTROLS = ('native', 'original', 'native_taco', 'taco_fused')
PROTOCOL = 'h100-single-node-phase6-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(text, old, new):
    require(text.count(old) == 1, f'Source contract changed: {old[:100]!r}')
    return text.replace(old, new)


def config(path):
    c = read(path)
    for key in ('tp', 'sm_count', 'layers', 'hidden', 'ffn', 'heads', 'sequence',
                'micro_batch', 'global_batch', 'warmup_steps', 'timed_steps', 'window'):
        require(type(c.get(key)) is int and c[key] > 0, f'Invalid integer: {key}')
    require(c['tp'] in (4, 8), 'This entrypoint supports single-node TP4/TP8, DP=PP=CP=1')
    require(c['sm_count'] in (114, 132), 'Specify H100 physical SM count: 114 or 132')
    require(c['global_batch'] % c['micro_batch'] == 0, 'Global batch must divide into microbatches')
    require(c['hidden'] % c['heads'] == 0 and c['heads'] % c['tp'] == 0, 'Invalid attention heads')
    m, n, k = shape(c)
    require(m % (128*c['tp']) == 0 and n % 128 == 0, 'Require M%(128*TP)=0 and N%128=0')
    require(k % (32*c['tp']) == n % (32*c['tp']) == 0, 'Both local GEMM K dimensions must be multiples of 32')
    require(m//128*(n//128) <= 2048, 'Sampler supports at most 2048 physical tiles')
    require((m//128*(n//128)//c['tp']) % 4 == 0, 'Sampler requires a multiple of four tiles per destination')
    require(math.isfinite(c['fraction']) and 0 < c['fraction'] <= 1, 'Invalid selection fraction')
    require(c['bases'] and len(c['bases']) == len(set(c['bases'])) and set(c['bases']) <= set(BASES), 'Invalid base orders')
    require(0 < c['target_percent'] < 100, 'Invalid acceptance target')
    for key in ('seed','token_seed','vocab'):
        require(type(c.get(key)) is int and c[key] >= 0, f'Invalid integer: {key}')
    require(c['vocab']>0, 'Vocabulary must be positive')
    return c


def shape(c):
    return [c['sequence']*c['micro_batch'], c['hidden'], c['ffn']]


def policy(base):
    return ('remote' if base == 'remote_first' else base) + '_arrival_selective'


def policies(c):
    return list(CONTROLS) + [policy(b) for b in c['bases']]


def orders(c):
    p = policies(c)
    first = [p[i:] + p[:i] for i in range(len(p))]
    return [first, [list(reversed(row)) for row in reversed(first)]]


def verify_files(root, hashes):
    for name, digest in hashes.items():
        p = Path(root) / name
        require(p.is_file() and sha(p) == digest, f'Missing or changed input: {p}')


def manifest(path, kind=None):
    p = Path(path).resolve()
    data = read(p / 'manifest.json')
    require(data['protocol'] == PROTOCOL and data['arch'] == 'sm90', 'Not an H100 Phase6 build')
    require(data['status'] == 'built', 'Build has not finished; prepared sources are not runnable')
    require(not data.get('diagnostic_only', False), 'Compile-check scratch builds cannot be used for measurements; prepare a fresh build')
    if kind:
        require(data['kind'] == kind, f'Expected {kind} build')
    verify_files(p, data['files'])
    return data


def validate_entry(entry, c):
    m, n, k = shape(c)
    world, tiles = c['tp'], m//128*(n//128)
    require(entry['shape'] == [m, n, k] and entry['K_local'] == k//world, 'Plan shape mismatch')
    require(entry['tiles'] == tiles, 'Plan tile count mismatch')
    for field in ('maps', 'coords', 'base_coords', 'mask'):
        require(len(entry[field]) == world and all(len(row) == tiles for row in entry[field]), f'Invalid {field} dimensions')
    per = tiles//world
    for rank in range(world):
        indices, coords, base = (entry[f][rank] for f in ('maps', 'coords', 'base_coords'))
        require(sorted(indices) == sorted(coords) == sorted(base) == list(range(tiles)), 'Non-bijective tile plan')
        require(coords == [base[i] for i in indices], 'Plan coordinates disagree with permutation')
        slots = {dst: [i for i, tile in enumerate(base) if tile//per == dst] for dst in range(world)}
        position = {i: (dst, j//c['window']) for dst, ids in slots.items() for j, i in enumerate(ids)}
        require(all(position[i] == position[j] for i, j in enumerate(indices)), 'Permutation crosses destination/window')
        mask = entry['mask'][rank]
        require(all(type(x) is int and x in (0, 1) for x in mask), 'Non-binary selection mask')
        require(not any(mask[rank*per:(rank+1)*per]), 'Local contribution cannot be quantized')
        require(sum(mask) <= int((tiles-per)*c['fraction']), 'Selection exceeds per-source budget')


def plan(path, c):
    p = read(path)
    require(p.get('protocol') == PROTOCOL and p.get('arch') == 'sm90', 'A800 plans cannot be used for H100')
    require(p.get('gpu_calibrated') is True and not p.get('test_only', False), 'A measured H100 calibration is required')
    require(p['config'] == c and p['world'] == c['tp'], 'Calibration/config mismatch')
    require(set(p['policies']) == set(c['bases']), 'Missing base calibration')
    verify_files('/', p['inputs'])
    for base in c['bases']:
        require(len(p['policies'][base]) == 1, 'One exact MLP shape per experiment')
        validate_entry(p['policies'][base][0], c)
    return p

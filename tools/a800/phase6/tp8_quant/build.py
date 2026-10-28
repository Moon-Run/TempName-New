"""Build isolated single-node TP8 MLP tables with the existing TACO runtime."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compiler = load('tp8_compiler', HERE.parents[1]/'phase4/build.py')
phase5 = load('tp8_base', HERE.parents[1]/'phase5/build.py')
sha = compiler.sha


def tables(entries, device=False):
    qualifier = 'static __device__ __constant__' if device else 'static const'
    field = 'maps' if device else 'coords'
    prefix = 'phase6_device' if device else 'phase6_host'
    text = '#pragma once\n'
    assert sum(2*8*s['tiles'] for s in entries) <= 64*1024
    for i, s in enumerate(entries):
        assert len(s[field]) == 8 and all(len(row) == s['tiles'] for row in s[field])
        values = [v for row in s[field] for v in row]
        assert all(0 <= v < 65536 for v in values)
        text += f'{qualifier} unsigned short {prefix}_{i}[] = {{' + ','.join(map(str, values)) + '};\n'
    text += ('__device__ __forceinline__ ' if device else 'inline ') + f'int {prefix}_lookup(int tm,int tn,int k,int rank,int idx) {{\n'
    for i, s in enumerate(entries):
        text += f"if(tm=={s['M']//128} && tn=={(s['N']+127)//128} && k=={s['K_local']} && rank>=0 && rank<8 && idx>=0 && idx<{s['tiles']})return {prefix}_{i}[rank*{s['tiles']}+idx];\n"
    return text + 'return -1;\n}\n'


def patch(old, base, entries):
    text = phase5.patch_swizzle(old, base)
    gate = 'phase5_mlp = local_world_size == 4 && problem_size_.m() == 2048 &&\n                   problem_size_.n() == 2048 && problem_size_.k() == 2048;'
    assert text.count(gate) == 1
    predicate = ' || '.join(f"(problem_size_.m()=={s['M']} && problem_size_.n()=={s['N']} && problem_size_.k()=={s['K_local']})" for s in entries)
    text = text.replace(gate, f'phase5_mlp = local_world_size == 8 && ({predicate});\n      phase6_local_k = problem_size_.k();')
    text = text.replace('  bool phase5_mlp = false;', '  bool phase5_mlp = false;\n  int phase6_local_k = 0;')
    text = text.replace('#pragma once', '#pragma once\n#ifdef __CUDACC__\n' + tables(entries, True) + '#else\ninline int phase6_device_lookup(int,int,int,int,int){return -1;}\n#endif', 1)
    anchor = '    auto coord = ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);'
    assert text.count(anchor) == 1
    return text.replace(anchor, '''    if (phase5_mlp) {
      int mapped = phase6_device_lookup(ThreadblockSwizzleStreamK::tiled_shape().m(),
          ThreadblockSwizzleStreamK::tiled_shape().n(), phase6_local_k, local_rank, tile_idx);
      if (mapped >= 0) tile_idx = mapped;
    }
''' + anchor)


def main(plan_path, out):
    plan_path, out = plan_path.resolve(), out.resolve()
    plan = json.loads(plan_path.read_text())
    assert plan['schema'] == 2 and plan['world'] == 8
    assert set(plan['policies']) == {'remote_first', 'interleaved'}
    for p, h in plan['inputs'].items():
        assert sha(Path(p)) == h, p
    out.mkdir(parents=True, exist_ok=False)
    # Change the rank domain, leaving the four coordinate fields per tile intact.
    checker = (HERE.parent/'check_mapping.cu').read_text()
    for old, new in [('tp!=4', 'tp!=8'), ('"LOCAL_WORLD_SIZE","4"', '"LOCAL_WORLD_SIZE","8"'),
                     ('rank<4', 'rank<8'), ('k/4', 'k/tp'), ('tm/4', 'tm/tp')]:
        assert old in checker
        checker = checker.replace(old, new)
    (out/'check_mapping.cu').write_text(checker)
    for base, entries in plan['policies'].items():
        assert entries and all(s['M'] % 1024 == 0 and s['N'] % 128 == 0 and
                               s['K_local'] % 32 == 0 and s['tiles'] == (s['M']//128)*(s['N']//128) and
                               s['tiles'] <= 2048 and s['shape'] == [s['M'],s['N'],8*s['K_local']]
                               for s in entries), 'Unsupported calibrated TP8 MLP shape'
        name = 'remote_arrival' if base == 'remote_first' else 'interleaved_arrival'
        folder = out/name
        compiler.main(folder, 'fused', lambda old: patch(old, base, entries), name)
        (folder/'taco/overlay/phase6_host.hpp').write_text(tables(entries))
        shutil.copy2(plan_path, folder/'joint-plan.json')
        manifest = json.loads((folder/'manifest.json').read_text())
        cmd = next(c['args'] for c in manifest['commands'] if c['args'][0].endswith('/nvcc') and '-c' in c['args'])
        cmd = cmd[:cmd.index('-c')] + [str(out/'check_mapping.cu'), '-o', str(folder/'check_mapping')]
        subprocess.run(cmd, check=True)
        manifest['commands'].append(dict(args=cmd, cwd=str(REPO)))
        manifest.update(arrival_plan_sha256=sha(plan_path), arrival_base=base, world=8)
        for p in (plan_path, Path(__file__), HERE.parent/'check_mapping.cu', out/'check_mapping.cu'):
            manifest['inputs'][str(p)] = sha(p)
        for p in (folder/'check_mapping', folder/'joint-plan.json', folder/'taco/overlay/phase6_host.hpp'):
            manifest['files'][str(p.relative_to(folder))] = sha(p)
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('TP8 QUANT BUILD COMPLETE', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('plan', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    main(a.plan, a.out)

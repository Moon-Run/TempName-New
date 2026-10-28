"""Build two MLP-only swizzles with the phase4 warp codec and mixed wire protocol."""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def patch_swizzle(old, policy):
    assert policy in ('remote_first', 'interleaved')
    field = '  int local_rank;'
    anchor = '      local_world_size = atoi(local_world_size_str);'
    body = '''    int m = (coord.m() + tiled_m / local_world_size * (local_rank)) % tiled_m;
    coord.m() = m;'''
    assert all(old.count(x) == 1 for x in (field, anchor, body))
    text = old.replace(field, '  bool phase5_mlp = false;\n'+field)
    text = text.replace(anchor, anchor+'''
      phase5_mlp = local_world_size == 4 && problem_size_.m() == 2048 &&
                   problem_size_.n() == 2048 && problem_size_.k() == 2048;''')
    if policy == 'remote_first':
        replacement = '''    int m = (coord.m() + tiled_m / local_world_size *
             (local_rank + int(phase5_mlp))) % tiled_m;
    coord.m() = m;'''
    else:
        replacement = '''    if (phase5_mlp) {
      int tiled_n = ThreadblockSwizzleStreamK::tiled_shape().n();
      int destination = (tile_idx % local_world_size + local_rank) % local_world_size;
      int within_partition = tile_idx / local_world_size;
      coord.m() = destination * (tiled_m / local_world_size) + within_partition / tiled_n;
      coord.n() = within_partition % tiled_n;
    } else {
      int m = (coord.m() + tiled_m / local_world_size * local_rank) % tiled_m;
      coord.m() = m;
    }'''
    return text.replace(body, replacement)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_arrival(plan_path):
    plan=json.loads(plan_path.read_text())
    assert plan['schema_version']==2 and plan['world_size']==4 and plan['score_mode']=='tail'
    for p,h in plan['inputs'].items(): assert sha(Path(p))==h,p
    shapes={}
    for policy,base in (('remote_arrival','remote_first'),('interleaved_arrival','interleaved')):
        entry=plan['policies'][policy]
        assert entry['base']==base
        shape=next(s for s in entry['shapes'] if (s['M'],s['N'],s['K_global'])==(2048,2048,8192))
        assert len(shape['maps'])==len(shape['coords'])==4
        for mapping,coords in zip(shape['maps'],shape['coords']):
            assert sorted(mapping)==sorted(coords)==list(range(256))
        assert not shape['identity_fallback']
        shapes[policy]=shape
    return plan,shapes


def arrival_device_header(shape):
    flat=[v for row in shape['maps'] for v in row]
    return ('#ifdef __CUDACC__\nstatic __device__ __constant__ unsigned short phase5_arrival_indices[1024] = {'+
            ','.join(map(str,flat))+'};\n'
            '__device__ __forceinline__ int phase5_arrival_index(int rank,int idx) { return phase5_arrival_indices[rank*256+idx]; }\n'
            '#else\ninline int phase5_arrival_index(int,int) { return -1; }\n#endif\n')


def patch_arrival(old, policy, shape):
    base={'remote_arrival':'remote_first','interleaved_arrival':'interleaved'}[policy]
    text=patch_swizzle(old,base)
    text=text.replace('#pragma once','#pragma once\n'+arrival_device_header(shape),1)
    anchor='    auto coord = ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);'
    assert text.count(anchor)==1
    return text.replace(anchor,'''    // local_rank remains the physical source, never the rank+1 mapping offset.
    if (phase5_mlp && tile_idx >= 0 && tile_idx < 256)
      tile_idx = phase5_arrival_index(local_rank, tile_idx);
'''+anchor,1)


def arrival_host_header(shape):
    text='#pragma once\n'
    for field,suffix in (('maps','index'),('coords','coord')):
        flat=[v for row in shape[field] for v in row]
        text += 'static const unsigned short phase5_host_'+suffix+'[1024] = {'+','.join(map(str,flat))+'};\n'
        text += f'inline int flux_arrival_host_{suffix}(int tm,int tn,int k,int tp,int rank,int idx) {{\n'
        text += f'if(tm==16 && tn==16 && k==2048 && tp==4 && rank>=0 && rank<4 && idx>=0 && idx<256) return phase5_host_{suffix}[rank*256+idx];\nreturn -1;\n}}\n'
    return text


def main(out, arrival_plan=None):
    spec = importlib.util.spec_from_file_location('phase4_build', HERE.parent/'phase4/build.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if arrival_plan:
        arrival_plan=arrival_plan.resolve()
        plan,shapes=load_arrival(arrival_plan)
    policies=('remote_arrival','interleaved_arrival') if arrival_plan else ('remote_first','interleaved')
    for policy in policies:
        folder = out/policy
        transform=(lambda old: patch_arrival(old,policy,shapes[policy])) if arrival_plan else (lambda old: patch_swizzle(old,policy))
        base.main(folder, 'fused', transform, policy)
        manifest = json.loads((folder/'manifest.json').read_text())
        checker=HERE.parent/'e2e/scale/check_mapping.cu'
        if arrival_plan:
            host=folder/'taco/overlay/arrival_host.hpp'
            host.write_text(arrival_host_header(shapes[policy]))
            shutil.copy2(arrival_plan,folder/'arrival-plan.json')
            checker=HERE.parent/'arrival/check_mapping.cu'
            manifest['arrival_plan_sha256']=sha(arrival_plan)
            manifest['arrival_base']=plan['policies'][policy]['base']
            manifest['arrival_changed_per_rank']=shapes[policy]['changed_per_rank']
            manifest['inputs'][str(arrival_plan)]=sha(arrival_plan)
            for p in (host,folder/'arrival-plan.json'):
                manifest['files'][str(p.relative_to(folder))]=sha(p)
        old = (base.PARENT/'original/overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp').read_text()
        new = (folder/'taco/overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp').read_text()
        (folder/'swizzle.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True))))
        # Use the exact build's overlay for the GPU bijection/expected-coordinate audit.
        command = next(c['args'] for c in manifest['commands'] if c['args'][0].endswith('/nvcc') and '-c' in c['args'])
        command = command[:command.index('-c')]+[str(checker), '-o', str(folder/'check_mapping')]
        subprocess.run(command, check=True)
        manifest['commands'].append(dict(args=command, cwd=str(ROOT)))
        manifest['inputs'][str(HERE/'build.py')] = sha(HERE/'build.py')
        manifest['inputs'][str(checker)] = sha(checker)
        for p in (folder/'swizzle.patch', folder/'check_mapping'):
            manifest['files'][str(p.relative_to(folder))] = sha(p)
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arrival-plan',type=Path)
    args=parser.parse_args()
    main(args.out,args.arrival_plan)

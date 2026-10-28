"""Build the exact TP4 model's MLP-only rank+1 mapping in a fresh directory.

The two module shapes are distinct and checked by the model harness. The base-policy selection
is made in the host constructor. The arrival variant additionally reads its
frozen table for the MLP shape; attention never matches that table.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_swizzle(old):
    anchor = '      local_world_size = atoi(local_world_size_str);'
    assert old.count(anchor) == 1
    selection = '''
      // Frozen l12-h2048-s2048-tp4: attention local K=512, MLP local K=2048.
      // This field is a mapping offset, not a communication rank.
      if (local_world_size != 4 || problem_size_.m() != 2048 ||
          problem_size_.n() != 2048 ||
          (problem_size_.k() != 512 && problem_size_.k() != 2048))
        throw std::runtime_error("Unsupported MLP-only model shape/TP");
      if (problem_size_.k() == 2048) local_rank += 1;'''
    return old.replace('#pragma once', '#pragma once\n#include <stdexcept>', 1).replace(anchor, anchor + selection)


def arrival_header(shape, device=False, coord=False):
    data = shape['coords' if coord else 'maps']
    flat = [x for row in data for x in row]
    assert len(data) == 4 and all(len(row) == 256 for row in data)
    assert all(0 <= x < 256 for x in flat)
    name = 'flux_arrival_index' if device else 'flux_arrival_host_coord' if coord else 'flux_arrival_host_index'
    array = 'device_arrival_indices' if device else 'host_arrival_coords' if coord else 'host_arrival_indices'
    qualifier = 'static __device__ __constant__' if device else 'static const'
    prefix = '__device__ __forceinline__' if device else 'inline'
    return (f'{qualifier} unsigned short {array}[1024] = {{'+','.join(map(str, flat))+'};\n'
            + f'{prefix} int {name}(int tm,int tn,int k,int tp,int rank,int idx) {{\n'
            + f'if(tm==16 && tn==16 && k==2048 && tp==4 && rank>=0 && rank<4 && idx>=0 && idx<256) return {array}[rank*256+idx];\nreturn -1;\n}}\n')


def patch_arrival(old):
    text = patch_swizzle(old)
    text = text.replace('#pragma once', '#pragma once\n#include "arrival_table.hpp"', 1)
    text = text.replace('  int local_rank;', '  int arrival_source_rank;\n  int arrival_local_k;\n  int local_rank;', 1)
    anchor = '      local_world_size = atoi(local_world_size_str);'
    text = text.replace(anchor, anchor+'\n      arrival_source_rank = local_rank;\n      arrival_local_k = problem_size_.k();', 1)
    anchor = '    auto coord = ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);'
    assert text.count(anchor) == 1
    return text.replace(anchor, '''    int mapped = flux_arrival_index(ThreadblockSwizzleStreamK::tiled_shape().m(),
        ThreadblockSwizzleStreamK::tiled_shape().n(), arrival_local_k,
        local_world_size, arrival_source_rank, tile_idx);
    if (mapped >= 0) tile_idx = mapped;
'''+anchor, 1)


def main(repo, out, check_source, plan_path):
    repo, out = repo.resolve(), out.resolve()
    source = repo/'outputs/a800/gemm_rs_validation/source'
    parent = repo/'outputs/a800/phase3/three-way-build'
    original = json.loads((parent/'manifest.json').read_text())
    for name, digest in original['files'].items():
        assert sha(parent/name) == digest, name
    out.mkdir()
    (out/'mapping').mkdir()
    plan = json.loads(plan_path.read_text())
    assert plan['schema_version'] == 2 and plan['world_size'] == 4 and plan['score_mode'] == 'tail'
    for name, digest in plan['inputs'].items():
        assert sha(Path(name)) == digest, name
    shape = next(s for s in plan['policies']['remote_arrival']['shapes']
                 if (s['M'], s['N'], s['K_global']) == (2048, 2048, 8192))
    assert plan['policies']['remote_arrival']['base'] == 'remote_first'
    assert not shape['identity_fallback']
    manifest = dict(scope='TP4 M=N=2048: attention original; MLP rank+1 or rank+1+tail',
                    plan_sha256=sha(plan_path), score_mode='tail',
                    parent_manifest_sha256=sha(parent/'manifest.json'), files={}, policies={}, commands=[])
    nvcc = '/data/apps/cuda/12.8/bin/nvcc'
    flags = ['-std=c++17','-O3','-DNDEBUG','-rdc=true','--expt-extended-lambda',
             '--expt-relaxed-constexpr','-Xcompiler=-fPIC','-gencode=arch=compute_80,code=[sm_80,compute_80]']

    def run(command, cwd=repo):
        command = list(map(str, command))
        manifest['commands'].append(dict(args=command, cwd=str(cwd)))
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=cwd, check=True)

    for policy in ('original', 'mlp_remote', 'mlp_remote_arrival'):
        dest = out/policy
        shutil.copytree(parent/'original', dest, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.o', 'check_mapping'))
        overlay = dest/'overlay'
        header = overlay/'gemm_rs/tile_scheduler/threadblock_swizzle.hpp'
        (overlay/'arrival_host.hpp').write_text(
            'inline int flux_arrival_host_index(int,int,int,int,int,int){return -1;}\n'
            'inline int flux_arrival_host_coord(int,int,int,int,int,int){return -1;}\n')
        inc = [overlay, source/'src', source/'include', source/'build',
               source/'3rdparty/cutlass/include', source/'3rdparty/cutlass/tools/util/include',
               source/'3rdparty/cutlass/tools/library/include', source/'3rdparty/cutlass/tools/profiler/include']
        if policy != 'original':
            old = header.read_text()
            new = patch_swizzle(old) if policy == 'mlp_remote' else patch_arrival(old)
            if policy == 'mlp_remote_arrival':
                (overlay/'arrival_table.hpp').write_text('#pragma once\n'+arrival_header(shape, device=True))
                (overlay/'arrival_host.hpp').write_text(arrival_header(shape)+arrival_header(shape, coord=True))
            header.write_text(new)
            (dest/'swizzle.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True))))
            reg = next((source/'build/src/gemm_rs/registers').glob('flux_bf16_bf16_void_bf16*_rcr_*_intranode.cu'))
            work = source/'build/src/cuda'
            obj, dlink, lib = dest/'registration.o', dest/'device_link.o', dest/'libflux_cuda.so'
            # Never let the linker's output follow a copied parent-library symlink.
            if lib.is_symlink():
                lib.unlink()
            run([nvcc, *flags, *['-I'+str(p) for p in inc], '-c', reg, '-o', obj])
            objects = [obj if 'registers/flux_bf16_bf16_void' in p else (work/p).resolve()
                       for p in shlex.split((work/'CMakeFiles/flux_cuda.dir/deviceObjects1.rsp').read_text())]
            run([nvcc, *flags, '-shared', '-dlink', *objects, '-o', dlink,
                 '-L/data/apps/cuda/12.8/lib64', '-lcudart_static', '-lcudadevrt', '-ldl', '-lrt', '-lpthread'])
            link = shlex.split((work/'CMakeFiles/flux_cuda.dir/link.txt').read_text())
            for i, arg in enumerate(link):
                if arg.endswith('.o'):
                    link[i] = str(obj if 'registers/flux_bf16_bf16_void' in arg else
                                  dlink if arg.endswith('cmake_device_link.o') else (work/arg).resolve())
            link[link.index('-o')+1] = str(lib)
            run(link, work)
        for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
            target = dest/'python/flux/lib'/name
            target.unlink()
            target.symlink_to(dest/name)
        run([nvcc, *flags, *['-I'+str(p) for p in inc], check_source,
             '-o', out/'mapping'/f'check_mapping-{policy}'])
        manifest['policies'][policy] = dict(library_sha256=sha(dest/'libflux_cuda.so'),
                                           swizzle_sha256=sha(header))
    for name, digest in original['files'].items():
        assert sha(parent/name) == digest, ('parent asset changed', name)
    manifest['files'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    files = {str(p): sha(p) for p in (out/'mapping').glob('check_mapping-*')}
    (out/'mapping/manifest.json').write_text(json.dumps(dict(files=files), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo', 'out', 'check_source', 'plan_path'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    main(args.repo, args.out, args.check_source, args.plan_path)

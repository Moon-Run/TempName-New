"""One shared dual-shape candidate library and a separate BF16 calibration sampler."""
import os
from pathlib import Path
import shutil
import subprocess

import build
import candidate
from ablation_plan import FREEZE
from common import HERE, REPO, read, replace_once, require, sha, verify_files, write
from scope_protocol import PROTOCOL, SHAPES
from scope_plan import bundle, model_config, validate

SWIZZLE = 'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp'


def tables(entries, device=False):
    prefix = 'phase6_device' if device else 'phase6_host'
    qualifier = '__device__ __forceinline__' if device else 'inline'
    if entries is None:
        return (f'{qualifier} int {prefix}_lookup(int tm,int tn,int k,int rank,int i) {{\n'
                ' if(tm!=16 || tn!=32 || (k!=512 && k!=2048) || rank<0 || rank>=8 || i<0 || i>=512) return -1;\n'
                ' return ((i%8+rank)%8)*64+i/8;\n}\n')
    declarations = []
    for role, entry in entries.items():
        declarations.append(build.table(entry, 8, device).replace(prefix, prefix+'_'+role))
    require(sum(len(e['maps'])*len(e['maps'][0])*2 for e in entries.values()) <= 65536, 'Constant tables exceed 64KiB')
    declarations.append(f'{qualifier} int {prefix}_lookup(int tm,int tn,int k,int rank,int i) {{\n'
                        f' if(k==512) return {prefix}_attn_lookup(tm,tn,k,rank,i);\n'
                        f' if(k==2048) return {prefix}_mlp_lookup(tm,tn,k,rank,i);\n return -1;\n}}\n')
    return '\n'.join(declarations)


def swizzle(source, entries=None):
    source = replace_once(source, '  int local_rank;', '  int local_rank;\n  int scope_k = 0;\n  bool scope_shape = false;')
    anchor = '      local_world_size = atoi(local_world_size_str);'
    source = replace_once(source, anchor, anchor+'''
      scope_k = problem_size_.k();
      scope_shape = local_world_size==8 && problem_size_.m()==2048 && problem_size_.n()==4096 &&
                    (scope_k==512 || scope_k==2048);''')
    source = replace_once(source, '''    int m = (coord.m() + tiled_m / local_world_size * (local_rank)) % tiled_m;
    coord.m() = m;''', '''    if(scope_shape) {
      int dst=(tile_idx%8+local_rank)%8;
      int within=tile_idx/8;
      coord.m()=dst*2+within/32;
      coord.n()=within%32;
    } else {
      coord.m()=(coord.m()+tiled_m/local_world_size*local_rank)%tiled_m;
    }''')
    if entries:
        source = replace_once(source, '#pragma once', '#pragma once\n#ifdef __CUDACC__\n'+tables(entries, True)+'\n#endif')
        anchor = '    auto coord = ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);'
        source = replace_once(source, anchor,
                              '    if(scope_shape && tile_idx>=0 && tile_idx<512)\n'
                              '      tile_idx=phase6_device_lookup(16,32,scope_k,local_rank,tile_idx);\n'+anchor)
    return source


def prepare(out, attn=None, mlp=None, source_only=False, sampler=False, roles=('attn',)):
    require(roles and len(set(roles)) == len(roles) and set(roles) <= set(SHAPES), 'Invalid calibration roles')
    plans = None if sampler else bundle(attn, mlp, source_only)
    freeze, old, measured = candidate.sources()
    source = old/'interleaved_arrival_selective'
    hashes = measured['variants']['interleaved_arrival_selective']['source_sha256']
    name = 'scope-sampler' if sampler else 'scope-candidate'
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    dest = out/name
    for relative in hashes:
        target = dest/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/relative, target)
    (dest/'3rdparty').mkdir(exist_ok=True)
    (dest/'3rdparty/cutlass').symlink_to(REPO/'3rdparty/cutlass', target_is_directory=True)
    verify_files(FREEZE/'source', {SWIZZLE: freeze['source_files'][SWIZZLE]['sha256']})
    entries = None if sampler else {r: p['entry'] for r, p in plans['roles'].items()}
    (dest/SWIZZLE).write_text(swizzle((FREEZE/'source'/SWIZZLE).read_text(), entries))
    (dest/'phase6_host.hpp').write_text('#pragma once\n'+tables(entries))
    if sampler:
        cm = dest/'CMakeLists.txt'
        cm.write_text(replace_once(cm.read_text(), 'add_compile_definitions(FLUX_TACO_BASELINE)\n', ''))
        cc = dest/'src/gemm_rs/ths_op/gemm_reduce_scatter.cc'
        cc.write_text(replace_once(cc.read_text(), '    if (owned_taco.enabled) {\n      TORCH_CHECK(nnodes',
                                  '    if (true) {\n      TORCH_CHECK(nnodes'))
        ep = dest/'src/gemm_rs/epilogue_evt.hpp'; ep.write_text(build.sampler_epilogue(ep.read_text()))
        for filename in ('sampler.h', 'sampler.cu'):
            shutil.copy2(REPO/'tools/a800/arrival8'/filename, dest/'src/gemm_rs'/filename)
        cm = dest/'src/gemm_rs/CMakeLists.txt'
        cm.write_text(replace_once(cm.read_text(), 'set(CU_FILES\n', 'set(CU_FILES\n  sampler.cu\n'))
        ld = dest/'src/cuda/version.ld'; ld.write_text(replace_once(ld.read_text(), 'global:', 'global:\n    mech_*;'))
    else:
        codec = dest/'python/flux/gemm_rs_taco.py'; codec.write_text(candidate.codec_source(codec.read_text()))
        write(out/'plans.json', plans)
    write(out/'config.json', model_config())
    identity = dict(protocol=PROTOCOL, source_only=source_only, sampling=sampler, roles=list(roles) if sampler else list(SHAPES),
                    status='pending_attention_calibration' if source_only else 'prepared')
    write(out/'identity.json', identity)
    names = list(hashes)+(['src/gemm_rs/sampler.h', 'src/gemm_rs/sampler.cu'] if sampler else [])
    after = {p: sha(dest/p) for p in names}
    changed = {p: dict(before=hashes.get(p), after=after[p]) for p in after if hashes.get(p) != after[p]}
    inputs = [HERE/p for p in ('scope_build.py', 'scope_plan.py', 'scope_protocol.py', 'build.py', 'candidate.py')]
    if sampler:
        inputs += [REPO/'tools/a800/arrival8'/p for p in ('instrument.py', 'sampler.h', 'sampler.cu')]
    info = dict(protocol=PROTOCOL, arch='sm90', kind='scope_sampler' if sampler else 'scope_candidate',
                status='prepared', config=model_config(), source_only=source_only, diagnostic_only=source_only,
                gpu_validated=False, calibration_roles=list(roles) if sampler else [],
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                parent_source_commit=measured['source_commit'], parent_manifest_sha256=sha(old/'manifest.json'),
                submodules=freeze['submodules'], inputs={str(p): sha(p) for p in inputs},
                variants={name: dict(role='sampler' if sampler else 'candidate', sampling=sampler, base='interleaved',
                                    python=f'{name}/python', source_sha256=after, source_changes=changed)},
                files={p: sha(out/p) for p in (['config.json', 'identity.json'] if sampler else ['config.json', 'identity.json', 'plans.json'])})
    write(out/'manifest.json', info)
    return out


def verify(root, built=False, kind=None):
    root = Path(root).resolve(); m = read(root/'manifest.json')
    require(m['protocol'] == PROTOCOL and m['arch'] == 'sm90' and m['config'] == model_config(), 'Wrong scope build/model')
    if kind:
        require(m['kind'] == kind, 'Wrong sampler/candidate build')
    verify_files(root, m['files'])
    identity = read(root/'identity.json')
    require(identity['source_only'] == m['source_only'] == m['diagnostic_only'], 'Build rehearsal flag changed')
    if built:
        require(m['status'] == 'built' and not m['source_only'], 'Unbuilt/source-only library cannot execute')
    require(len(m['variants']) == 1, 'Scope policies must share one build')
    name, v = next(iter(m['variants'].items()))
    require(v['sampling'] == identity['sampling'] == (m['kind'] == 'scope_sampler') and
            v['role'] == ('sampler' if v['sampling'] else 'candidate'), 'Sampler role changed')
    verify_files(root/name, v['source_sha256'])
    if not v['sampling']:
        plans = read(root/'plans.json')
        require(plans['source_only'] == m['source_only'] and set(plans['roles']) == set(SHAPES), 'Plan mode/roles changed')
        for role, record in plans['roles'].items():
            validate(record, role, allow_test=m['source_only'])
    if built:
        for path in ('libflux_cuda.so', 'libflux_cuda_ths_op.so', 'check_mapping'):
            require((root/name/path).is_file(), 'Missing compiled runtime artifact: '+path)
    return m


def compile_check(root, out, cuda_home, cxx='g++'):
    root, out, cuda_home = map(lambda p: Path(p).resolve(), (root, out, cuda_home))
    m = verify(root); out.mkdir(parents=True, exist_ok=False)
    name = next(iter(m['variants'])); src = root/name
    entries = None if m['kind'] == 'scope_sampler' else {r: p['entry'] for r, p in read(root/'plans.json')['roles'].items()}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = dict(completed=False, gpu_used=False, gpu_validated=False, diagnostic_only=True, commands=[])
    def run(cmd):
        cmd = list(map(str, cmd)); result['commands'].append(cmd)
        with (out/'compile.log').open('a') as log:
            log.write(repr(cmd)+'\n'); log.flush()
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        for role, shape in SHAPES.items():
            coords = entries[role]['coords'] if entries else [[((i % 8+r) % 8)*64+i//8 for i in range(512)] for r in range(8)]
            mask = entries[role]['mask'] if entries else [[0]*512 for _ in range(8)]
            code = '''#include "phase6_host.hpp"
#include "gemm_rs/taco_runtime.h"
int main(){
 const int coords[]={COORDS}; const unsigned char mask[]={MASK};
 FluxTacoConfig c; c.enabled=1;c.world=8;c.m=2048;c.n=4096;c.selected=mask;
 for(int r=0;r<8;++r)for(int i=0;i<512;++i){
  if(phase6_host_lookup(16,32,K,r,i)!=coords[r*512+i])return 1;
  for(int d=0;d<8;++d)if(taco_selected(c,r,d,i)!=(r!=d && bool(mask[r*512+i])))return 2;
 }
 if(phase6_host_lookup(16,32,1024,0,0)!=-1)return 3;
 if(phase6_host_lookup(8,32,K,0,0)!=-1)return 4;
}
'''.replace('COORDS', ','.join(str(x) for row in coords for x in row)).replace('MASK', ','.join(str(x) for row in mask for x in row)).replace(',K,', ','+str(shape[2]//8)+',')
            host = out/f'{role}.cc'; host.write_text(code)
            exe = out/f'{role}-host'
            run([cxx, '-std=c++17', '-O2', '-I'+str(src), '-I'+str(src/'src'), '-I'+str(cuda_home/'include'), host, '-o', exe]); run([exe])
        inc = [src, src/'src', src/'include', src/'3rdparty/cutlass/include', src/'3rdparty/cutlass/tools/util/include']
        run([cuda_home/'bin/nvcc', '-std=c++17', '-O2', '--expt-relaxed-constexpr', '--expt-extended-lambda',
             '-gencode=arch=compute_90,code=sm_90', *['-I'+str(p) for p in inc], '-c', src/'check_mapping.cu', '-o', out/'mapping.o'])
        runtime = 'sampler.cu' if m['kind'] == 'scope_sampler' else 'taco_runtime.cu'
        run([cuda_home/'bin/nvcc', '-std=c++17', '-O2', '-gencode=arch=compute_90,code=sm_90',
             '-I'+str(src/'src'), '-c', src/'src/gemm_rs'/runtime, '-o', out/'runtime.o'])
        result.update(completed=True, roles=list(SHAPES), mapping_sha256=sha(out/'mapping.o'), runtime_sha256=sha(out/'runtime.o'))
    finally:
        write(out/'validation.json', result)
    return result

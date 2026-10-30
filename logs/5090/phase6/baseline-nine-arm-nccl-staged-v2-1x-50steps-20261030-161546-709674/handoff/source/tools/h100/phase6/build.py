"""Prepare isolated H100 sources, then cross-compile without querying/using GPUs."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from common import (HERE, REPO, PROTOCOL, config, load, manifest, plan, policy,
                    read, replace_once, require, sha, shape, verify_files, write)


def table(entry, world, device):
    field = 'maps' if device else 'coords'
    prefix = 'phase6_device' if device else 'phase6_host'
    qualifier = 'static __device__ __constant__' if device else 'static const'
    values = [v for row in entry[field] for v in row]
    require(len(values)*2 <= 64*1024 and all(0 <= v < 65536 for v in values), 'Constant table overflow')
    m, n, k = entry['M'], entry['N'], entry['K_local']
    return (f'{qualifier} unsigned short {prefix}_data[] = {{' + ','.join(map(str, values)) + '};\n' +
            ('__device__ __forceinline__ ' if device else 'inline ') +
            f'int {prefix}_lookup(int tm,int tn,int k,int rank,int i) {{\n'
            f' if(tm=={m//128} && tn=={n//128} && k=={k} && rank>=0 && rank<{world} && i>=0 && i<{entry["tiles"]})'
            f' return {prefix}_data[rank*{entry["tiles"]}+i];\n return -1;\n}}\n')


def swizzle(source, c, base, entry=None):
    """Exact-shape MLP-only base order and optional destination-window permutation."""
    if base == 'original' and entry is None:
        return source
    m, n, k = shape(c)
    world = c['tp']
    source = replace_once(source, '  int local_rank;', '  int local_rank;\n  bool h100_mlp = false;')
    anchor = '      local_world_size = atoi(local_world_size_str);'
    source = replace_once(source, anchor, anchor +
        f'\n      h100_mlp = local_world_size=={world} && problem_size_.m()=={m} && problem_size_.n()=={n} && problem_size_.k()=={k//world};')
    old = '''    int m = (coord.m() + tiled_m / local_world_size * (local_rank)) % tiled_m;
    coord.m() = m;'''
    if base == 'remote_first':
        new = '''    coord.m() = (coord.m() + tiled_m / local_world_size *
        (local_rank + int(h100_mlp))) % tiled_m;'''
    elif base.startswith('interleaved'):
        change = {'interleaved': '', 'interleaved_remote': 'if(slot<2) slot ^= 1;',
                  'interleaved_remote_group': 'slot=(slot+1)%local_world_size;'}[base]
        new = f'''    if(h100_mlp) {{
      int slot=tile_idx%local_world_size; {change}
      int dst=(slot+local_rank)%local_world_size;
      int within=tile_idx/local_world_size;
      int tn=ThreadblockSwizzleStreamK::tiled_shape().n();
      coord.m()=dst*(tiled_m/local_world_size)+within/tn;
      coord.n()=within%tn;
    }} else {{
      coord.m()=(coord.m()+tiled_m/local_world_size*local_rank)%tiled_m;
    }}'''
    else:
        require(base == 'original', 'Unknown base order')
        new = old
    source = replace_once(source, old, new)
    if entry:
        declarations = '#ifdef __CUDACC__\n' + table(entry, world, True) + '\n#endif\n'
        source = replace_once(source, '#pragma once', '#pragma once\n' + declarations)
        anchor = '    auto coord = ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);'
        source = replace_once(source, anchor,
            f'    if(h100_mlp && tile_idx>=0 && tile_idx<{entry["tiles"]})\n'
            f'      tile_idx=phase6_device_lookup({m//128},{n//128},{k//world},local_rank,tile_idx);\n' + anchor)
    return source


def wrapper(source, force_v2=False, legacy_fused=False):
    # Local algorithm selection, not a global lie about device capability.
    # Instance identity is captured before init_output_buffer, so buffer, dispatch,
    # padding, reduction and publication barrier all use the SAME algorithm.
    source = source.replace('get_arch()', 'phase6_arch()')
    source = replace_once(source, 'SMCoreEnum sm_core = get_sm_core();',
                          'SMCoreEnum sm_core = arch == _Sm80{} ? _A100{}() : get_sm_core();')
    require(not (force_v2 and legacy_fused), 'Sampler and legacy fused roles are distinct')
    choose = 'true' if force_v2 else 'h100_taco' if legacy_fused else 'owned_taco.enabled'
    # The legacy frontend still configures/resets TACO on every forward. Only
    # the algorithm identity must survive reset, so attention stays on V3 and
    # MLP buffer construction, dispatch and publication all use V2.
    identity = '\n  const bool h100_taco = taco_config().enabled;' if legacy_fused else ''
    helper = f'''class GemmRS::GemmRSImpl {{{identity}
  ArchEnum phase6_arch() const {{
    TORCH_CHECK(get_arch() == _Sm90{{}}, "H100 Phase6 requires physical SM90");
    if ({choose}) {{
      TORCH_CHECK(nnodes == 1, "H100 Phase6 V2 port is single-node only");
      return _Sm80{{}}();  // V2 algorithm key; code is compiled for SM90.
    }}
    return get_arch();  // Unquantized attention retains upstream Hopper V3.
  }}
'''
    source = replace_once(source, 'class GemmRS::GemmRSImpl {', helper)
    return source.replace('requires single-node SM80 BF16 ring reduction',
                          'requires the single-node H100 Phase6 V2 BF16 ring path')


def legacy_fused_python(source):
    """Set the V2 instance identity at construction; preserve legacy forwards."""
    old = '''        self.op = cpp_mod.GemmRS(group, 1, m, n, torch.bfloat16, torch.bfloat16,
                                transpose_weight=False, fuse_reduction=False, ring_reduction=True)'''
    new = '''        # H100 needs the MLP's V2 buffer layout before the first forward.
        # Configure only while constructing it; subsequent attention is V3.
        code = self.lib.taco_configure(self.pointers, self.rank, self.world, m, n)
        if code:
            raise RuntimeError(f'TACO construction rejected: CUDA error {code}')
        try:
            self.op = cpp_mod.GemmRS(group, 1, m, n, torch.bfloat16, torch.bfloat16,
                                    transpose_weight=False, fuse_reduction=False, ring_reduction=True)
        finally:
            self.lib.taco_reset()'''
    return replace_once(source, old, new)


def install_fused_v2(dest):
    """Install the pinned A800 baseline, never an ignored experiment directory."""
    root = HERE/'fused_v2'
    provenance = read(root/'provenance.json')
    verify_files(root, provenance['files'])
    for relative in provenance['files']:
        shutil.copy2(root/relative, dest/relative)
    edit(dest/'python/flux/gemm_rs_taco.py', legacy_fused_python)
    return provenance['implementation']


def edit(path, transform):
    path.write_text(transform(path.read_text()))


def sampler_epilogue(source):
    """Apply the audited fragment sampler around the current TACO helper methods.

    Sampler builds do not define FLUX_TACO_BASELINE. Temporarily move those
    disabled methods out of the old insertion point; never change their bodies.
    """
    start=source.index('\n#ifdef FLUX_TACO_BASELINE\n#ifdef FLUX_TACO_SEPARATE\n    CUTLASS_DEVICE void\n    taco_stage_step')
    anchor='\n    ////\n    CUTLASS_DEVICE void\n    end_epilogue()'
    end=source.index(anchor,start)
    helper=source[start:end]
    source=source[:start]+source[end:]
    source=replace_once(source,'  static\n#ifndef FLUX_TACO_BASELINE\n  constexpr\n#endif\n  Params\n  to_underlying_arguments',
                        '  static constexpr Params\n  to_underlying_arguments')
    instrument=load('h100_instrument','tools/a800/arrival8/instrument.py')
    source=instrument.instrument_epilogue(source)
    return replace_once(source,anchor,helper+anchor)


def remove_unused_pcie_initializer(source):
    # This TU-local pointer has no readers anywhere in the maintained source.
    # Its eager cudaMalloc would initialize device 0 on dlopen, before rank
    # binding in an external caller, and prevents CPU-only ABI/import checks.
    old='''namespace {
bytedance::flux::SegmentInfo *segments_global_device = []() {
  void *ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, sizeof(bytedance::flux::SegmentInfo) * 1000));
  // bytedance::flux::SegmentInfo *ptr_device;
  return (bytedance::flux::SegmentInfo *)ptr;
}();
}  // namespace'''
    return replace_once(source,old,'// H100 single-node build: no unused process-global CUDA allocation.')


def snapshot(dest):
    ignore = shutil.ignore_patterns('.git', '__pycache__', '*.so', '*.so.*', '*.o', '*.a', '*.pyc', 'build', '*.egg-info')
    for name in ('src', 'include', 'cmake', 'python'):
        shutil.copytree(REPO/name, dest/name, ignore=ignore)
    for name in ('CMakeLists.txt', 'setup.py', 'LICENSE', 'README.md'):
        shutil.copy2(REPO/name, dest/name)
    (dest/'3rdparty').mkdir()
    require((REPO/'3rdparty/cutlass/include/cutlass/cutlass.h').is_file(), 'Run git submodule update --init --recursive first')
    require((REPO/'3rdparty/nccl/src/nccl.h.in').is_file(), 'Initialize the NCCL submodule before preparing a build')
    (dest/'3rdparty/cutlass').symlink_to(REPO/'3rdparty/cutlass', target_is_directory=True)


def prepare(out, config_path, plan_path=None, only_variants=None):
    c = config(config_path)
    measured = plan(plan_path, c) if plan_path else None
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', c)
    shutil.copy2(REPO/'tools/a800/phase4/TACO_LICENSE.txt',out/'TACO_LICENSE.txt')
    kind = 'candidates' if measured else 'baselines'
    variants = {policy(b): ('candidate', b) for b in c['bases']} if measured else {
        'original': ('hopper', 'original'), 'taco_fused': ('fused', 'original'),
        **{'sampler_'+b: ('sampler', b) for b in c['bases']}}
    if only_variants is not None:
        require(only_variants and len(set(only_variants)) == len(only_variants) and
                set(only_variants) <= set(variants), 'Unknown or duplicate build variants')
        variants = {name: variants[name] for name in only_variants}
    info = dict(protocol=PROTOCOL, arch='sm90', kind=kind, config=c, status='prepared',
                gpu_validated=False, variants={}, files={}, inputs={},
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                algorithm='MLP Phase6 uses SM90-compiled GemmV2; original reference and non-quantized attention use native Hopper GemmV3')
    if measured:
        shutil.copy2(plan_path, out/'selection-plan.json')
        info['plan_sha256'] = sha(plan_path)
    for name, (role, base) in variants.items():
        dest = out/name
        snapshot(dest)
        implementation = install_fused_v2(dest) if role == 'fused' else None
        edit(dest/'src/gemm_rs/tile_scheduler/threadblock_swizzle_pcie.hpp',remove_unused_pcie_initializer)
        # CMake is already invoked explicitly by this builder. Do not let legacy
        # setup_requires download a second CMake into setuptools' .eggs cache.
        edit(dest/'setup.py', lambda s: replace_once(s,
            'setup_requires=["torch", "cmake", "packaging"],',
            'setup_requires=["torch", "packaging"],'))
        edit(dest/'src/cuda/version.ld', lambda s: s.replace('global:', 'global:\n    h100_v2_occupancy;'))
        entry = measured['policies'][base][0] if measured else None
        info['variants'][name] = dict(role=role, base=base, sampling=role=='sampler', python=f'{name}/python')
        if implementation:
            info['variants'][name]['implementation'] = implementation
        # Only compile RS and its common IPC/helper bindings, not unrelated MoE/AG operators.
        p = dest/'src/CMakeLists.txt'
        edit(p, lambda s: '\n'.join(line for line in s.split('\n') if not re_other_operator(line)))
        gen = dest/'src/generator/CMakeLists.txt'
        edit(gen, lambda s: '\n'.join(line for line in s.split('\n') if not line.startswith('add_executable(') or line.startswith('add_executable(gen_gemm_rs ')))
        profile = '_H800' if c['sm_count'] == 132 else '_H100PCIE'
        (dest/'src/generator/gen_gemm_rs.cc').write_text((HERE/'generator.cc').read_text().replace('H100_PROFILE', profile))
        if c['sm_count'] == 114:
            edit(dest/'include/flux/flux.h', lambda s: replace_once(replace_once(replace_once(s,
                'H800 = 132,', 'H800 = 132, H100PCIE = 114,'),
                'using _H800 = cute::C<SMCoreEnum::H800>;',
                'using _H800 = cute::C<SMCoreEnum::H800>;\nusing _H100PCIE = cute::C<SMCoreEnum::H100PCIE>;'),
                'case SMCoreEnum::H800: return "H800";',
                'case SMCoreEnum::H800: return "H800";\n    case SMCoreEnum::H100PCIE: return "H100PCIE";'))
            edit(dest/'src/cuda/op_registry.cu', lambda s: replace_once(s,
                'case 132: sm_core = SMCoreEnum::H800; break;',
                'case 132: sm_core = SMCoreEnum::H800; break;\n    case 114: sm_core = SMCoreEnum::H100PCIE; break;'))
        cm = dest/'src/gemm_rs/CMakeLists.txt'
        edit(cm, lambda s: s.replace('"--archs=${CUDAARCHS}" "--sm_cores=${SM_CORES}"',
            f'"--archs=80;90" "--sm_cores=108;{c["sm_count"]}"').replace('file(GLOB TUNING_CONFIGS tuning_config/*.cu)', 'set(TUNING_CONFIGS)'))
        # Explicit SM90 cubins/PTX; no A800 object files or wheels are reused.
        extra = []
        if role in ('candidate', 'fused'):
            extra.append('taco_runtime.cu')
            edit(dest/'src/cuda/version.ld', lambda s: s.replace('global:', 'global:\n    taco_*;'))
            edit(dest/'src/gemm_rs/ths_op/gemm_reduce_scatter.cc',
                 lambda s: wrapper(s, legacy_fused=role=='fused'))
            edit(dest/'python/flux/gemm_rs_taco.py', lambda s: s.replace('(8, 0)', '(9, 0)').replace('supports SM80 only','requires the H100 SM90 port').replace('tools/a800/phase4/build.py','tools/h100/phase6/build.py'))
        elif role == 'sampler':
            extra.append('sampler.cu')
            for f in ('sampler.h','sampler.cu'):
                shutil.copy2(REPO/'tools/a800/arrival8'/f,dest/'src/gemm_rs'/f)
            edit(dest/'src/gemm_rs/epilogue_evt.hpp', sampler_epilogue)
            edit(dest/'src/gemm_rs/ths_op/gemm_reduce_scatter.cc', lambda s: wrapper(s, True))
            edit(dest/'src/cuda/version.ld', lambda s: s.replace('global:', 'global:\n    mech_*;'))
        if role != 'hopper':
            edit(dest/'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp', lambda s: swizzle(s,c,base,entry))
            (dest/'phase6_host.hpp').write_text('#pragma once\n' + (table(entry,c['tp'],False) if entry else 'inline int phase6_host_lookup(int,int,int,int,int){return -1;}\n'))
            shutil.copy2(HERE/'check_mapping.cu',dest/'check_mapping.cu')
        if extra:
            edit(cm, lambda s: replace_once(s, 'set(CU_FILES\n', 'set(CU_FILES\n  '+'\n  '.join(extra)+'\n'))
        # Package setup always declares nccl_static; build it once in this new output tree.
        # The CUDA/THS CMake targets need its public headers as well.
        edit(dest/'CMakeLists.txt', lambda s: replace_once(s, 'add_subdirectory(src)',
            'include_directories("$ENV{NCCL_ROOT}/include")\nadd_subdirectory(src)'))
        if role in ('candidate','fused'):
            # Add definitions before subdirectories are evaluated.
            edit(dest/'CMakeLists.txt', lambda s: replace_once(s, 'add_subdirectory(src)', 'add_compile_definitions(FLUX_TACO_BASELINE)\nadd_subdirectory(src)'))
        info['variants'][name]['source_sha256'] = {str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and not p.is_symlink()}
    # Record maintained sources and the submodule commits, without copying ignored builds.
    inputs = list(HERE.glob('*.py')) + list(HERE.glob('*.cc')) + list(HERE.glob('*.cu'))
    if not measured:
        frozen = HERE/'fused_v2'
        inputs += [frozen/'provenance.json'] + [frozen/p for p in read(frozen/'provenance.json')['files']]
    inputs += [REPO/'tools/a800/arrival8'/f for f in ('instrument.py','sampler.h','sampler.cu')]
    info['inputs'] = {str(p):sha(p) for p in inputs}
    info['submodules'] = {n:subprocess.check_output(['git','-C',str(REPO/'3rdparty'/n),'rev-parse','HEAD'],text=True).strip() for n in ('cutlass','nccl')}
    info['files']['config.json'] = sha(out/'config.json')
    info['files']['TACO_LICENSE.txt'] = sha(out/'TACO_LICENSE.txt')
    if measured:
        info['files']['selection-plan.json'] = sha(out/'selection-plan.json')
    write(out/'manifest.json',info)
    return out


def re_other_operator(line):
    line=line.strip()
    return line.startswith('add_subdirectory(') and line not in ('add_subdirectory(gemm_rs)', 'add_subdirectory(cuda)', 'add_subdirectory(ths_op)')


def compile_build(out, cuda_home, jobs=2, nccl_root=None):
    out, cuda_home = out.resolve(), cuda_home.resolve()
    info = read(out/'manifest.json')
    require(info['status'] in ('prepared','compile_failed'), 'Use a fresh prepared build')
    verify_files(out, info['files'])
    for name, variant in info['variants'].items():
        verify_files(out/name, variant['source_sha256'])
    verify_files('/', info['inputs'])
    for name, commit in info['submodules'].items():
        require(subprocess.check_output(['git','-C',str(REPO/'3rdparty'/name),'rev-parse','HEAD'],text=True).strip()==commit, f'{name} submodule changed')
    require((cuda_home/'bin/nvcc').is_file(), 'CUDA Toolkit is required; pass --cuda-home')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME=str(cuda_home), CUDACXX=str(cuda_home/'bin/nvcc'),
               PATH=f'{Path(sys.executable).parent}:{cuda_home}/bin:'+os.environ['PATH'],
               TORCH_CUDA_ARCH_LIST='9.0', FLUX_FORCE_BUILD='1', FLUX_SHM_USE_NVSHMEM='0',
               MAX_JOBS=str(jobs), CMAKE_BUILD_PARALLEL_LEVEL=str(jobs), PYTHONNOUSERSITE='1')
    info['status'] = 'compiling'; info['commands'] = []
    def run(args, cwd=out):
        args=list(map(str,args)); info['commands'].append(dict(args=args,cwd=str(cwd)))
        write(out/'manifest.json',info)
        with (out/'compile.log').open('a') as log:
            log.write('\nCOMMAND '+repr(args)+'\n');log.flush()
            subprocess.run(args,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        if nccl_root is None:
            nccl = out/'nccl-source'
            if not nccl.exists():
                shutil.copytree(REPO/'3rdparty/nccl',nccl,ignore=shutil.ignore_patterns('.git','build'))
            run(['make',f'-j{jobs}','src.build','NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90'],nccl)
            nccl_root=nccl/'build'
        nccl_root=nccl_root.resolve()
        require((nccl_root/'include/nccl.h').is_file() and (nccl_root/'lib/libnccl_static.a').is_file(), 'NCCL headers/static library missing')
        env['NCCL_ROOT']=str(nccl_root)
        info['nccl_files']={str(p):sha(p) for p in (nccl_root/'include/nccl.h',nccl_root/'lib/libnccl_static.a')}
        for name,variant in info['variants'].items():
            src=out/name; print('Compiling',name,flush=True)
            run(['cmake','-S',src,'-B',src/'build','-DCUDAARCHS=90',f'-DGPU_SM_CORES={info["config"]["sm_count"]}',
                 '-DENABLE_NVSHMEM=OFF','-DWITH_PROTOBUF=OFF','-DBUILD_TEST=OFF','-DCMAKE_BUILD_TYPE=Release',
                 f'-DPYTHON_EXECUTABLE={sys.executable}',f'-DPython_EXECUTABLE={sys.executable}',
                 f'-DCMAKE_CXX_FLAGS=-I{nccl_root}/include',f'-DCMAKE_CUDA_FLAGS=-I{nccl_root}/include',
                 f'-DCMAKE_INSTALL_PREFIX={src}/python/flux'])
            # Query the occupancy of this exact binary when checking mappings.
            # Instrumented/fused kernels can differ: never assume A800's two CTAs/SM.
            regs=list((src/'build/src/gemm_rs/registers').glob('*sm80*.cu'))
            require(len(regs)==1, 'Expected one frozen V2 registration')
            reg=regs[0]
            probe='''
extern "C" __attribute__((visibility("default"))) int h100_v2_occupancy() {
  using Device = decltype(GemmDevice_0().gemm_device());
  return Device::maximum_active_blocks();
}
'''
            generated = reg.read_text()
            if 'int h100_v2_occupancy()' not in generated:
                reg.write_text(generated+probe)
            run(['cmake','--build',src/'build','--parallel',str(jobs)])
            run(['cmake','--install',src/'build'])
            run([sys.executable,'setup.py','build_ext','--inplace'],src)
            for lib in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
                if not (src/lib).is_symlink():
                    (src/lib).symlink_to(Path('python/flux/lib')/lib)
            if variant['role']!='hopper':
                inc=[src,src/'src',src/'include',src/'build',src/'3rdparty/cutlass/include',src/'3rdparty/cutlass/tools/util/include']
                run([cuda_home/'bin/nvcc','-std=c++17','-O2','--expt-relaxed-constexpr','--expt-extended-lambda',
                     '-gencode=arch=compute_90,code=sm_90',*['-I'+str(p) for p in inc],src/'check_mapping.cu','-ldl','-o',src/'check_mapping'])
            for p in list((src/'python').rglob('*'))+[src/'libflux_cuda.so',src/'libflux_cuda_ths_op.so',src/'check_mapping']:
                if p.is_file() and '__pycache__' not in p.parts:
                    info['files'][str(p.relative_to(out))]=sha(p)
        info['status']='built'
        # Host checks/compilation never constitute GPU validation.
        info['gpu_validated']=False
    except BaseException:
        info['status']='compile_failed'
        raise
    finally:
        write(out/'manifest.json',info)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('out',type=Path)
    p.add_argument('--config',type=Path,default=HERE/'config.json');p.add_argument('--plan',type=Path)
    p=sub.add_parser('compile');p.add_argument('out',type=Path)
    p.add_argument('--cuda-home',type=Path,required=True);p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--nccl-root',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':print(prepare(args.out,args.config,args.plan))
    else:
        require(args.jobs>0,'jobs must be positive')
        compile_build(args.out,args.cuda_home,args.jobs,args.nccl_root)

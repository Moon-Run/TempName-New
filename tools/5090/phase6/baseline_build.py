"""Experimental SM120 NCCL staging port of the nine numerical variants.

Preparation and compilation never launch a GPU. The TACO V2 algorithm keeps
its SM80/A100 registry key; actual cubins target SM120 and runtime hardware is
checked independently. All producer/decoder pointers refer to local storage;
NCCL transfers compact packets. Historical masks remain H100-calibrated inputs.
"""
import copy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import rtx5090_loader  # Installs the H100 source directory before imports below.
import local_sources
import nccl_transport
import ablation6_build
import basic_quant_build
import ma11_build
import ma11_sources
from ma2_ablation_build import copy_registered
from native_exports import verify_native_exports
from common import REPO, MEGATRON_COMMIT, read, write, require, sha, verify_files, replace_once

HERE = Path(__file__).resolve().parent
PROTOCOL = 'rtx5090-nine-arm-native-nccl-staged-sm120-v2'
ARMS = ('ma11', 'ma89', 'ma8', 'ma9', 'cr3', 'r', 'c', 'a', 'flux-basic')
NATIVE_ARMS = ('ma11', 'ma89', 'cr3', 'r', 'c', 'a')
ALIASES = {'ma8': 'cr3', 'ma9': 'ma89'}
WRAPPER = 'src/gemm_rs/ths_op/gemm_reduce_scatter.cc'
TRANSPORT_INPUTS = ('local_sources.py', 'local_transport.cu', 'local_transport.h',
                    'local_layout.h', 'nccl_transport.py')


def origins(root):
    reference = Path(root)/'reference'
    locations = {'ma11': (reference, 'F4-cache-zcount-csoftmax'),
                 'ma89': (reference, 'F4-cache-zcount'),
                 'cr3': (reference/'reference', 'F4'),
                 'r': (reference/'reference/reference/reference', 'R'),
                 'c': (reference/'reference/reference/reference', 'C'),
                 'a': (reference/'reference/reference/reference', 'A')}
    return {arm: (directory/name, read(directory/'manifest.json')['variants'][name])
            for arm, (directory, name) in locations.items()}


def rs_generator(text):
    begin = text.index('  auto v3 = ')
    text = text[:begin] + '''  return main_template(options, {
      cute::make_tuple(build_gen_space(v2, v2hp),
        std::string("gemm_rs/gemm_v2_reduce_scatter.hpp"), std::string("GemmV2ReduceScatter"))});
}
'''
    return text.replace('H100 Phase6 experiment', 'RTX5090 frozen-algorithm experiment').replace(
        'compiled to SM90 machine code', 'compiled to SM120 machine code')


def port_sources(source, gemm_only=False, kind=None):
    source = Path(source)
    if gemm_only:
        return {'src/generator/gen_comm_none.cc': (HERE/'gemm_only_generator.cc').read_text()}
    require(kind in NATIVE_ARMS, 'Native arm identity is required for the staging transform')
    updates = local_sources.sources(source, kind)
    updates['src/generator/gen_gemm_rs.cc'] = rs_generator((source/'src/generator/gen_gemm_rs.cc').read_text())
    name = 'src/gemm_rs/CMakeLists.txt'
    updates[name] = replace_once(updates[name],
        '"--archs=80;90" "--sm_cores=108;132"', '"--archs=80" "--sm_cores=108"')
    name = 'python/flux/gemm_rs_taco.py'
    updates[name] = nccl_transport.frontend_source(kind)
    name = 'check_mapping.cu'
    updates[name] = replace_once((source/name).read_text(),
        'prop.major!=9 || prop.minor!=0', 'prop.major!=12 || prop.minor!=0 || prop.multiProcessorCount!=170')
    return updates


def _source_hashes(source):
    return {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*'))
            if p.is_file() and not p.is_symlink()}


def prepare(root):
    root = Path(root).resolve()
    require(not root.exists(), 'Use a fresh RTX5090 build directory')
    root.mkdir(parents=True)
    ma11_build.prepare(root/'reference')
    native = root/'selective'
    native.mkdir()
    variants = {}
    for arm, (source, item) in origins(root).items():
        dest = native/arm
        copy_registered(source, dest, item['source_sha256'])
        updates = port_sources(source, kind=arm)
        for name, content in updates.items(): (dest/name).write_text(content)
        variants[arm] = dict(python=arm+'/python', source_sha256=_source_hashes(dest),
                             h100_source_sha256=item['source_sha256'], ported_files=sorted(updates))
    for name in ('selection-plan.json', 'config.json', 'TACO_LICENSE.txt'):
        shutil.copy2(root/'reference'/name, native/name)
    config = read(native/'config.json')
    config['sm_count'] = 170
    write(native/'config.json', config)
    write(native/'manifest.json', dict(protocol=PROTOCOL, arch='sm120', status='prepared',
          config=config, variants=variants, gpu_validated=False,
          files={name: sha(native/name) for name in ('selection-plan.json', 'config.json', 'TACO_LICENSE.txt')}))
    ablation6_build.prepare(root/'flux-reference')
    reference = root/'flux-reference'/ablation6_build.VARIANT
    source = root/'flux'/ablation6_build.VARIANT
    item = read(root/'flux-reference/manifest.json')
    copy_registered(reference, source, item['source_sha256'])
    updates = port_sources(reference, gemm_only=True)
    for name, content in updates.items(): (source/name).write_text(content)
    write(root/'flux/manifest.json', dict(protocol=PROTOCOL, arch='sm120', status='prepared',
          config=config, variant=ablation6_build.VARIANT, gpu_validated=False,
          algorithm='Sm89/RTX5090 GemmV2 BF16-RCR 128x128x32 stage3, actual SM120',
          source_sha256=_source_hashes(source), files={}))
    basic_quant_build.prepare(root/'codec-basic')
    codec = read(root/'codec-basic/manifest.json')
    codec.update(arch='sm120', source_protocol=codec['protocol'], protocol=PROTOCOL)
    write(root/'codec-basic/manifest.json', codec)
    write(root/'baseline-build.json', dict(protocol=PROTOCOL, status='prepared', gpu_validated=False,
          experimental=True, recipe_sha256=sha(__file__), generator_sha256=sha(HERE/'gemm_only_generator.cc'),
          arms=list(ARMS), aliases=ALIASES, actual_arch='sm120',
          transport=nccl_transport.PROTOCOL, peer_memory_required=False, original_peer_overlap_preserved=False,
          transport_inputs={name: sha(HERE/name) for name in TRANSPORT_INPUTS},
          selection_origin='unchanged H100 calibration; no RTX5090 recalibration claimed',
          algorithm_key='Sm80/A100 V2 for TACO, Sm89/RTX5090 V2 for GemmOnly',
          submodules=read(root/'reference/manifest.json')['submodules']))
    verify(root)
    return root


def native_root(root, arm):
    require(arm in ARMS, 'Unknown RTX5090 arm: '+str(arm))
    root = Path(root)
    return root/'flux'/ablation6_build.VARIANT if arm == 'flux-basic' else root/'selective'/ALIASES.get(arm, arm)


def libraries(root, arm):
    source = native_root(root, arm)
    return {name: dict(path=str((source/name).resolve()), sha256=sha(source/name))
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so')}


def _expected_port_hashes(source, item, gemm_only=False, kind=None):
    return dict(item, **{name: hashlib.sha256(content.encode()).hexdigest()
                         for name, content in port_sources(source, gemm_only, kind).items()})


def verify(root, built=False):
    root = Path(root).resolve()
    record = read(root/'baseline-build.json')
    require(record['protocol'] == PROTOCOL and record['recipe_sha256'] == sha(__file__) and
            record['generator_sha256'] == sha(HERE/'gemm_only_generator.cc'), 'RTX5090 build recipe changed')
    require(tuple(record['arms']) == ARMS and record['aliases'] == ALIASES and
            record['actual_arch'] == 'sm120' and record['experimental'] is True and
            record['gpu_validated'] is False and record['transport'] == nccl_transport.PROTOCOL and
            record['peer_memory_required'] is False and record['original_peer_overlap_preserved'] is False,
            'Wrong experimental SM120 staging build identity')
    require(record['transport_inputs'] == {name: sha(HERE/name) for name in TRANSPORT_INPUTS},
            'Local staging implementation changed')
    ma11_build.verify(root/'reference')
    info = read(root/'selective/manifest.json')
    require(info['protocol'] == PROTOCOL and info['arch'] == 'sm120' and
            tuple(info['variants']) == NATIVE_ARMS, 'Wrong RTX5090 native variants')
    expected_config = read(root/'reference/config.json')
    expected_config['sm_count'] = 170
    require(info['config'] == read(root/'selective/config.json') == expected_config, 'RTX5090 model changed')
    verify_files(root/'selective', info['files'])
    require(sha(root/'selective/selection-plan.json') == sha(root/'reference/selection-plan.json'),
            'Frozen H100 arrival/quantization plan changed')
    for arm, (source, item) in origins(root).items():
        hashes = _expected_port_hashes(source, item['source_sha256'], kind=arm)
        require(info['variants'][arm]['source_sha256'] == hashes, 'Unregistered SM120 transform: '+arm)
        verify_files(native_root(root, arm), hashes)
    ablation6_build.verify(root/'flux-reference')
    flux = read(root/'flux/manifest.json')
    source = root/'flux-reference'/ablation6_build.VARIANT
    hashes = _expected_port_hashes(source, read(root/'flux-reference/manifest.json')['source_sha256'], True)
    require(flux['protocol'] == PROTOCOL and flux['arch'] == 'sm120' and
            flux['source_sha256'] == hashes and flux['config'] == expected_config, 'GemmOnly port changed')
    verify_files(native_root(root, 'flux-basic'), hashes)
    verify_files(root/'flux', flux['files'])
    codec = read(root/'codec-basic/manifest.json')
    require(codec['protocol'] == PROTOCOL and codec['source_protocol'] == basic_quant_build.PROTOCOL and
            codec['arch'] == 'sm120' and codec['source_identity'] == basic_quant_build.source_identity() and
            codec['source_only'] is False, 'Basic receiver v2 source identity changed')
    verify_files(root/'codec-basic', codec['files'])
    require((root/'codec-basic/upstream_kernels.cuh').read_text() == basic_quant_build.kernel_source(),
            'Pinned upstream TACO arithmetic changed')
    require(sha(root/'codec-basic/baseline_codec.cu') == sha(basic_quant_build.HERE/'baseline_codec.cu') and
            sha(root/'codec-basic/basic_quant_receiver.cuh') == sha(basic_quant_build.HERE/'basic_quant_receiver.cuh') and
            (root/'codec-basic/basic_quant.cu').read_text() ==
            '#include "baseline_codec.cu"\n#include "basic_quant_receiver.cuh"\n',
            'Basic receiver v2 arithmetic or translation unit changed')
    if built:
        require(record['status'] == info['status'] == flux['status'] == codec['status'] == 'built', 'Build incomplete')
        verify_files(root, record['files'])
        require(codec['library'] == dict(path=str(root/'codec-basic'/basic_quant_build.LIBRARY),
                                        sha256=sha(root/'codec-basic'/basic_quant_build.LIBRARY)), 'Codec binary changed')
        for arm in (*NATIVE_ARMS, 'flux-basic'):
            source = native_root(root, arm)
            require(len(list((source/'python').glob('flux_ths_pybind*.so'))) == 1, 'Missing native extension: '+arm)
            libraries(root, arm)
            if arm == 'flux-basic': continue
            require((source/'check_mapping').is_file(), 'Missing exact-kernel mapping probe: '+arm)
            required = ('h100_v2_occupancy',) + local_sources.NATIVE_EXPORTS
            if arm in ('ma11', 'ma89', 'cr3'): required += ('taco_ma4f_protocol_version', 'taco_ma4f_decode_reduce')
            if arm in ('ma11', 'ma89'): required += ma11_sources.ZERO_COUNT_EXPORTS
            if arm == 'ma11': required += ma11_sources.NATIVE_EXPORTS
            forbidden = (() if arm == 'ma11' else ma11_sources.NATIVE_EXPORTS)
            if arm not in ('ma11', 'ma89'): forbidden += ma11_sources.ZERO_COUNT_EXPORTS
            verify_native_exports(source/'libflux_cuda.so', required=required, forbidden=forbidden)
    return record


def environment_identity(cuda_home):
    import torch
    nvcc = Path(cuda_home)/'bin/nvcc'
    require(nvcc.is_file(), 'CUDA Toolkit nvcc is required')
    codes = subprocess.check_output([str(nvcc), '--list-gpu-code'], text=True)
    require('sm_120' in codes.split(), 'CUDA Toolkit must support sm_120 (CUDA >= 12.8)')
    runtime = tuple(int(x) for x in (torch.version.cuda or '0.0').split('.')[:2])
    # Unlike torch.cuda.get_arch_list(), this build-time query also works on a
    # CPU-only compilation host and does not initialize a CUDA context.
    arch_flags = torch._C._cuda_getArchFlags() if hasattr(torch._C, '_cuda_getArchFlags') else ''
    require(runtime >= (12, 8) and 'sm_120' in (arch_flags or '').split(),
            'Use a PyTorch CUDA >= 12.8 build containing SM120 kernels')
    return dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI), nvcc_sha256=sha(nvcc))


def compile_build(root, megatron, cuda_home, jobs=2):
    from export_tp8_dp2 import archive_snapshot
    from ablation_prepare import megatron_check
    root, megatron, cuda_home = map(lambda p: Path(p).resolve(), (root, megatron, cuda_home))
    record = verify(root)
    require(record['status'] == 'prepared' and type(jobs) is int and jobs > 0, 'Compile a fresh prepared RTX5090 build')
    megatron_check(megatron)
    for name, revision in record['submodules'].items():
        require(subprocess.check_output(['git', '-C', str(REPO/'3rdparty'/name), 'rev-parse', 'HEAD'], text=True).strip() == revision,
                'Dependency commit changed: '+name)
    record.update(status='building', environment=environment_identity(cuda_home), commands=[])
    write(root/'baseline-build.json', record)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME=str(cuda_home), CUDACXX=str(cuda_home/'bin/nvcc'),
               PATH=f'{Path(sys.executable).parent}:{cuda_home}/bin:'+os.environ['PATH'],
               TORCH_CUDA_ARCH_LIST='12.0', FLUX_FORCE_BUILD='1', FLUX_SHM_USE_NVSHMEM='0',
               MAX_JOBS=str(jobs), CMAKE_BUILD_PARALLEL_LEVEL=str(jobs), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1')
    env.pop('FLUX_FORCE_NVLINK', None)
    compiler_tmp = root/'compiler-tmp'
    compiler_tmp.mkdir()
    env['TMPDIR'] = str(compiler_tmp)

    def run(args, cwd=root):
        args = list(map(str, args))
        record['commands'].append(dict(args=args, cwd=str(cwd)))
        write(root/'baseline-build.json', record)
        with (root/'compile.log').open('a') as log:
            log.write('\nCOMMAND '+repr(args)+'\n'); log.flush()
            subprocess.run(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        record['megatron'] = archive_snapshot(megatron, root/'megatron', MEGATRON_COMMIT)
        run(['make', '-C', root/'megatron/megatron/core/datasets'])
        nccl = root/'nccl-source'
        shutil.copytree(REPO/'3rdparty/nccl', nccl, ignore=shutil.ignore_patterns('.git', 'build'))
        run(['make', f'-j{jobs}', 'src.build', 'NVCC_GENCODE=-gencode=arch=compute_120,code=sm_120'], nccl)
        nccl_root = nccl/'build'
        require((nccl_root/'lib/libnccl_static.a').is_file(), 'NCCL static build missing')
        env['NCCL_ROOT'] = str(nccl_root)
        for arm in (*NATIVE_ARMS, 'flux-basic'):
            source = native_root(root, arm)
            print('Compile RTX5090 native arm:', arm, flush=True)
            run(['cmake', '-S', source, '-B', source/'build', '-DCUDAARCHS=120', '-DGPU_SM_CORES=170',
                 '-DENABLE_NVSHMEM=OFF', '-DWITH_PROTOBUF=OFF', '-DBUILD_TEST=OFF', '-DCMAKE_BUILD_TYPE=Release',
                 f'-DPYTHON_EXECUTABLE={sys.executable}', f'-DPython_EXECUTABLE={sys.executable}',
                 f'-DCMAKE_CXX_FLAGS=-I{nccl_root}/include', f'-DCMAKE_CUDA_FLAGS=-I{nccl_root}/include',
                 f'-DCMAKE_INSTALL_PREFIX={source}/python/flux'])
            operator = 'comm_none' if arm == 'flux-basic' else 'gemm_rs'
            registrations = list((source/'build/src'/operator/'registers').glob('*.cu'))
            expected_arch = 'sm89' if arm == 'flux-basic' else 'sm80'
            require(len(registrations) == 1 and expected_arch in registrations[0].name.lower(),
                    'Wrong V2 algorithm registry for '+arm)
            if arm != 'flux-basic':
                with registrations[0].open('a') as handle:
                    handle.write('''
extern "C" __attribute__((visibility("default"))) int h100_v2_occupancy() {
  using Device = decltype(GemmDevice_0().gemm_device());
  return Device::maximum_active_blocks();
}
''')
            run(['cmake', '--build', source/'build', '--parallel', jobs])
            run(['cmake', '--install', source/'build'])
            run([sys.executable, 'setup.py', 'build_ext', '--inplace'], source)
            for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                (source/name).symlink_to(Path('python/flux/lib')/name)
            if arm != 'flux-basic':
                includes = [source, source/'src', source/'include', source/'build',
                            source/'3rdparty/cutlass/include', source/'3rdparty/cutlass/tools/util/include']
                run([cuda_home/'bin/nvcc', '-std=c++17', '-O2', '--expt-relaxed-constexpr', '--expt-extended-lambda',
                     '-gencode=arch=compute_120,code=sm_120', *['-I'+str(p) for p in includes],
                     source/'check_mapping.cu', '-ldl', '-o', source/'check_mapping'])
        codec_root = root/'codec-basic'
        cmd = [cuda_home/'bin/nvcc', '-std=c++17', '-O3', '-shared', '-Xcompiler=-fPIC', '-lineinfo',
               '-gencode=arch=compute_120,code=sm_120', '--cudart=shared',
               codec_root/'basic_quant.cu', '-o', codec_root/basic_quant_build.LIBRARY]
        if env.get('CUDAHOSTCXX') or env.get('CXX'): cmd += ['-ccbin', env.get('CUDAHOSTCXX') or env['CXX']]
        run(cmd)
        codec = read(codec_root/'manifest.json')
        codec.update(status='built', library=dict(path=str(codec_root/basic_quant_build.LIBRARY),
                     sha256=sha(codec_root/basic_quant_build.LIBRARY)))
        codec['files'][basic_quant_build.LIBRARY] = sha(codec_root/basic_quant_build.LIBRARY)
        write(codec_root/'manifest.json', codec)
        for directory in ('selective', 'flux'):
            manifest = root/directory/'manifest.json'
            item = read(manifest)
            item['status'] = 'built'
            for arm in NATIVE_ARMS if directory == 'selective' else ('flux-basic',):
                source = native_root(root, arm)
                for p in [*list((source/'python').rglob('*.so')), source/'libflux_cuda.so',
                          source/'libflux_cuda_ths_op.so', source/'check_mapping']:
                    if p.is_file(): item['files'][str(p.relative_to(root/directory))] = sha(p)
            write(manifest, item)
        files = {str(p.relative_to(root)): sha(p) for p in (root/'megatron').rglob('*') if p.is_file()}
        require(len(list((root/'megatron/megatron/core/datasets').glob('helpers*.so'))) == 1,
                'Missing Megatron dataset helper')
        for directory in ('selective', 'flux', 'codec-basic'):
            files[directory+'/manifest.json'] = sha(root/directory/'manifest.json')
            files.update({directory+'/'+name: digest for name, digest in read(root/directory/'manifest.json')['files'].items()})
        record.update(status='built', files=files)
        write(root/'baseline-build.json', record)
        return verify(root, True)
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(root/'baseline-build.json', record)
        raise

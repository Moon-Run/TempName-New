"""Assemble exact existing native variants; no new numerical kernels."""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys

import ablation6_build
import baseline_codec_build
import basic_quant_build
import build as flux_build
import ma11_build
import ma11_sources
from ma2_ablation_build import copy_registered
from native_exports import verify_native_exports
from common import HERE, REPO, MEGATRON_COMMIT, read, write, require, sha, verify_files
from baseline_new_protocol import PROTOCOL, SELECTIVE


def origins(root):
    reference = Path(root)/'reference'
    original = read(reference/'reference/manifest.json')
    composed = read(reference/'manifest.json')
    return {'ma11': (reference/'F4-cache-zcount-csoftmax', composed['variants']['F4-cache-zcount-csoftmax']),
            'ma89': (reference/'F4-cache-zcount', composed['variants']['F4-cache-zcount']),
            'cr3': (reference/'reference/F4', original['variants']['F4'])}


def prepare(root):
    root = Path(root).resolve()
    require(not root.exists(), 'Use a fresh five-arm build directory')
    root.mkdir(parents=True)
    ma11_build.prepare(root/'reference')
    reference = read(root/'reference/manifest.json')
    native = root/'selective'; native.mkdir()
    variants = {}
    for arm,(source,item) in origins(root).items():
        copy_registered(source,native/arm,item['source_sha256'])
        variants[arm] = copy.deepcopy(item)
        variants[arm]['python'] = arm+'/python'
    for name in ('selection-plan.json','config.json','TACO_LICENSE.txt'):
        shutil.copy2(root/'reference'/name,native/name)
    info = copy.deepcopy(reference)
    info.update(status='prepared',variants=variants,
                files={n:sha(native/n) for n in ('selection-plan.json','config.json','TACO_LICENSE.txt')})
    write(native/'manifest.json',info)
    ablation6_build.prepare(root/'flux')
    basic_quant_build.prepare(root/'codec-basic')
    baseline_codec_build.prepare(root/'codec-cuda')
    write(root/'baseline-build.json',dict(protocol=PROTOCOL,status='prepared',gpu_validated=False,
          recipe_sha256=sha(__file__)))
    verify(root)
    return root


def native_root(root, arm):
    root = Path(root)
    return root/'selective'/arm if arm in SELECTIVE else root/'flux'/ablation6_build.VARIANT


def libraries(root, arm):
    if arm == 'megatron-cuda': return {}
    src = native_root(root,arm)
    return {name:dict(path=str((src/name).resolve()),sha256=sha(src/name))
            for name in ('libflux_cuda.so','libflux_cuda_ths_op.so')}


def environment_identity(cuda_home):
    import torch
    require(torch.__version__=='2.6.0+cu124' and torch.version.cuda=='12.4' and
            torch._C._GLIBCXX_USE_CXX11_ABI is False, 'Use the configured PyTorch 2.6.0+cu124 ABI=0 environment')
    return dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
                nvcc_sha256=sha(Path(cuda_home)/'bin/nvcc'),
                gcc_sha256=sha('/usr/bin/gcc-11'),gxx_sha256=sha('/usr/bin/g++-11'))


def verify(root, built=False):
    root = Path(root).resolve(); record = read(root/'baseline-build.json')
    require(record['protocol']==PROTOCOL and record['recipe_sha256']==sha(__file__), 'Build recipe changed')
    ma11_build.verify(root/'reference')
    native = root/'selective'; info = read(native/'manifest.json')
    require(tuple(info['variants'])==SELECTIVE, 'Missing/extra native variants')
    require(info['config']==read(root/'reference/manifest.json')['config'], 'Native model changed')
    verify_files('/',info['inputs']); verify_files(native,info['files'])
    for arm,(source,item) in origins(root).items():
        require(info['variants'][arm]['source_sha256']==item['source_sha256'], 'Native arm relabelled: '+arm)
        verify_files(native/arm,item['source_sha256'])
    require(sha(native/'selection-plan.json')==sha(root/'reference/selection-plan.json'), 'Frozen selection changed')
    ablation6_build.verify(root/'flux',built)
    basic_quant_build.verify(root/'codec-basic',built)
    baseline_codec_build.verify(root/'codec-cuda',built)
    if built:
        require(record['status']=='built' and info['status']=='built', 'Build incomplete')
        verify_files(root,record['files'])
        for arm in SELECTIVE:
            src = native/arm
            require((src/'check_mapping').is_file() and
                    len(list((src/'python').glob('flux_ths_pybind*.so')))==1, 'Missing native extension: '+arm)
            required = ('taco_ma4f_protocol_version','taco_ma4f_decode_reduce')
            if arm != 'cr3': required += ma11_sources.ZERO_COUNT_EXPORTS
            if arm == 'ma11': required += ma11_sources.NATIVE_EXPORTS
            forbidden = (() if arm=='ma11' else ma11_sources.NATIVE_EXPORTS)
            if arm=='cr3': forbidden += ma11_sources.ZERO_COUNT_EXPORTS
            verify_native_exports(src/'libflux_cuda.so',required=required,forbidden=forbidden)
    return record


def compile_build(root, megatron, cuda_home, jobs=2):
    from export_tp8_dp2 import archive_snapshot
    from ablation_prepare import megatron_check
    root,megatron,cuda_home=map(lambda p:Path(p).resolve(),(root,megatron,cuda_home))
    record=verify(root)
    require(record['status']=='prepared' and jobs>0, 'Compile a fresh prepared build')
    megatron_check(megatron)
    record.update(status='building',environment=environment_identity(cuda_home))
    write(root/'baseline-build.json',record)
    saved=dict(os.environ)
    try:
        os.environ.update(CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',CUDAHOSTCXX='/usr/bin/g++-11',
                          DEBUG='0',MAKEFLAGS='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
        record['megatron']=archive_snapshot(megatron,root/'megatron',MEGATRON_COMMIT)
        sources={str(p.relative_to(root)):sha(p) for p in (root/'megatron').rglob('*') if p.is_file()}
        env=dict(os.environ,PATH=str(Path(sys.executable).parent)+':'+os.environ['PATH'],CUDA_VISIBLE_DEVICES='')
        with (root/'megatron-compile.log').open('x') as log:
            subprocess.run(['make','-C',str(root/'megatron/megatron/core/datasets')],env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        flux_build.compile_build(root/'selective',cuda_home,jobs)
        nccl=root/'selective/nccl-source/build'
        ablation6_build.compile_build(root/'flux',cuda_home,jobs,nccl)
        basic_quant_build.compile_build(root/'codec-basic',cuda_home)
        baseline_codec_build.compile_build(root/'codec-cuda',cuda_home)
        helpers=list((root/'megatron/megatron/core/datasets').glob('helpers*.so'))
        require(len(helpers)==1, 'Missing Megatron dataset helper')
        sources.update({str(p.relative_to(root)):sha(p) for p in helpers})
        # Include immutable native manifests as well as the exact produced bytes.
        for directory in ('selective','flux','codec-basic','codec-cuda'):
            manifest=root/directory/'manifest.json'
            sources[str(manifest.relative_to(root))]=sha(manifest)
            for name,h in read(manifest)['files'].items(): sources[directory+'/'+name]=h
        record.update(status='built',files=sources)
        write(root/'baseline-build.json',record)
        return verify(root,True)
    except BaseException as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}')
        write(root/'baseline-build.json',record)
        raise
    finally:
        os.environ.clear();os.environ.update(saved)

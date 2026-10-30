"""Frozen shared baseline/MAV2 CUDA codec; compile without a GPU."""
from pathlib import Path
import os
import shutil
import subprocess

import basic_quant_build as base
from common import HERE, read, require, sha, verify_files, write

PROTOCOL = 'mav2-three-arm-transform-codec-v2'
LIBRARY = 'libmav2_quant.so'
POLICIES = dict(basic=dict(representation='H8', quality_selection=False, mode=None),
                mav2=dict(representation='D8/H8', quality_selection=True, mode=-1),
                d8=dict(representation='D8', quality_selection=False, mode=1, fallback=None))
SELECTOR = dict(relative_l2=.025, max_abs_over_amax=.0625, group_size=128,
                packet_bytes_per_group=136, direct_tag=-1, fallback='H8',
                quantized_remote_fraction=1.0, scale_floor_hex='0x1p-149f')
TRANSLATION_UNIT = '#include "baseline_codec.cu"\n#include "basic_quant_receiver.cuh"\n#include "mav2_codec.cuh"\n'


def encoder_source():
    text = base.kernel_source()
    start = text.index('template<typename T, int GROUP_SIZE>')
    end = text.index('template<typename T, int GROUP_SIZE>', start + 1)
    text = text[start:end]
    require(text.count('fused_ash_compress_kernel') == 1, 'Upstream encoder name changed')
    text = text.replace('fused_ash_compress_kernel', 'mav2_encode_kernel')
    old = '__nv_fp8_interpretation_t fp8_type)'
    require(text.count(old) == 1, 'Upstream encoder signature changed')
    text = text.replace(old, '__nv_fp8_interpretation_t fp8_type, int mav2_mode)')
    anchor = '    float val_sq = val * val;'
    require(text.count(anchor) == 1, 'Upstream encoder boundary changed')
    prefix = '''    __shared__ float mav2_scratch[4];
    if (mav2::try_direct(val, reinterpret_cast<unsigned char*>(out_base),
        chunk_q_scales, chunk_ada_scales, idx, group_id, mav2_mode, mav2_scratch)) return;

'''
    return text.replace(anchor, prefix + anchor)


def source_identity():
    return base.source_identity() | {name: sha(HERE/name) for name in
        ('mav2_build.py', 'mav2_codec.cuh', 'mav2_codec.py', 'mav2_reference.py', 'mav2_validation.py')}


def prepare(out, source_only=False):
    root = Path(out).resolve(); root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(base.UPSTREAM, root/'upstream')
    for name in ('baseline_codec.cu', 'basic_quant_receiver.cuh', 'mav2_codec.cuh'):
        shutil.copy2(HERE/name, root/name)
    (root/'upstream_kernels.cuh').write_text(base.kernel_source())
    (root/'mav2_encoder.cuh').write_text(encoder_source())
    (root/'mav2_quant.cu').write_text(TRANSLATION_UNIT)
    write(root/'manifest.json', dict(protocol=PROTOCOL, status='prepared', arch='sm90',
        source_only=source_only, upstream_commit=base.UPSTREAM_COMMIT,
        source_identity=source_identity(), selector=SELECTOR, policies=POLICIES,
        files={str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()},
        library=None, gpu_validated=False))
    return root


def validate_snapshot(m, built=False):
    require(m['protocol'] == PROTOCOL and m['arch'] == 'sm90' and
            m['upstream_commit'] == base.UPSTREAM_COMMIT and m['selector'] == SELECTOR and
            m['policies'] == POLICIES and
            m['source_identity'] == source_identity(), 'MAV2 codec provenance changed')
    if built:
        require(m['status'] == 'built' and not m['source_only'] and
                set(m['library']) == {'path', 'sha256'}, 'MAV2 codec has not been built')
    return m


def verify(root, built=False):
    root = Path(root).resolve(); m = validate_snapshot(read(root/'manifest.json'), built)
    verify_files(root, m['files'])
    require((root/'upstream_kernels.cuh').read_text() == base.kernel_source() and
            (root/'mav2_encoder.cuh').read_text() == encoder_source() and
            (root/'mav2_quant.cu').read_text() == TRANSLATION_UNIT, 'MAV2 generated kernel changed')
    for name in ('baseline_codec.cu', 'basic_quant_receiver.cuh', 'mav2_codec.cuh'):
        require(sha(root/name) == sha(HERE/name), 'Codec source changed: '+name)
    if built:
        require(Path(m['library']['path']) == root/LIBRARY and sha(root/LIBRARY) == m['library']['sha256'],
                'MAV2 codec binary changed')
    return m


def compile_build(root, cuda_home):
    root, cuda_home = Path(root).resolve(), Path(cuda_home).resolve()
    m = verify(root)
    require(not m['source_only'] and m['status'] == 'prepared', 'Compile requires fresh executable build')
    nvcc = cuda_home/'bin/nvcc'; require(nvcc.is_file(), 'CUDA Toolkit missing')
    cmd = [str(nvcc), '-std=c++17', '-O3', '-shared', '-Xcompiler=-fPIC', '-lineinfo',
           '--ftz=false', '-gencode=arch=compute_90,code=sm_90', '--cudart=shared',
           str(root/'mav2_quant.cu'), '-o', str(root/LIBRARY)]
    host = os.environ.get('CUDAHOSTCXX') or os.environ.get('CXX')
    if host: cmd += ['-ccbin', host]
    m.update(status='building', command=cmd, cuda_home=str(cuda_home)); write(root/'manifest.json', m)
    try:
        with (root/'compile.log').open('x') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        m.update(status='built', library=dict(path=str(root/LIBRARY), sha256=sha(root/LIBRARY)),
                 compiler_sha256=sha(nvcc))
        m['files'][LIBRARY] = sha(root/LIBRARY); write(root/'manifest.json', m)
        return verify(root, built=True)
    except BaseException:
        m['status'] = 'failed'; write(root/'manifest.json', m); raise

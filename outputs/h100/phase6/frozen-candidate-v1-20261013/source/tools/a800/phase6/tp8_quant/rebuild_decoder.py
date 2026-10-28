"""Build isolated TP8 decoder variants without changing TP4/frozen baselines."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_runtime(text, default):
    anchor = 'static thread_local int decode_variant = -1;'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '#include "gemm_rs/taco_decode_tp8.cuh"\n' + anchor)
    assert text.count('variant > 13') == 1
    text = text.replace('variant > 13', 'variant > 19')
    anchor = '  int variant = decode_variant < 0 ?'
    assert text.count(anchor) == 1
    text = text.replace(anchor, f'''  // Only the selective TP8 model shape enters this isolated specialization.
  if (c.world == 8 && c.selected && c.m == 2048 && c.n == 2048) {{
    const int v = decode_variant < 0 ? {default} : decode_variant;
    auto* input = static_cast<const __nv_bfloat16*>(local);
    auto* result = static_cast<__nv_bfloat16*>(output);
    if (v == 14) taco_decode_ring8_kernel<<<1024, 128, 0, stream>>>(c, input, result);
    else if (v == 15) taco_decode_tile8_kernel<4, true><<<dim3(16,64),128,0,stream>>>(c,input,result);
    else if (v == 16) taco_decode_tile8_kernel<8, true><<<dim3(16,32),128,0,stream>>>(c,input,result);
    else if (v == 17) taco_decode_flat8_kernel<128><<<512,128,0,stream>>>(c,input,result);
    else if (v == 19 && c.quant_tile_count >= 0 && c.quant_tile_count <= 32 &&
             (!c.quant_tile_count || c.quant_tiles)) {{
      taco_decode_cooperative8_kernel<<<256+c.quant_tile_count*128,256,0,stream>>>(c,input,result);
      return int(cudaGetLastError());
    }}
    else if (v == 18 && c.quant_tile_count >= 0 && c.quant_tile_count <= 32 &&
             (!c.quant_tile_count || c.quant_tiles)) {{
      if (c.quant_tile_count)
        taco_materialize8_kernel<<<dim3(c.quant_tile_count*32,8),128,0,stream>>>(
            c, const_cast<__nv_bfloat16*>(input));
      taco_reduce_materialized8_kernel<<<256,256,0,stream>>>(c,input,result);
      return int(cudaGetLastError());
    }}
    if (v >= 14 && v <= 17) return int(cudaGetLastError());
  }}
''' + anchor)
    return text


def main(args):
    repo = Path(__file__).resolve().parents[4]
    base, out = args.base.resolve(), args.out.resolve()
    header = repo/'src/gemm_rs/taco_decode_tp8.cuh'
    out.mkdir(parents=True, exist_ok=False)
    for policy in ('remote_arrival', 'interleaved_arrival'):
        old, new = base/policy, out/policy
        manifest = json.loads((old/'manifest.json').read_text())
        assert manifest['world'] == 8
        for name, digest in manifest['files'].items():
            assert sha(old/name) == digest, name
        shutil.copytree(old, new, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        runtime = new/'taco/overlay/gemm_rs/taco_runtime.cu'
        original = runtime.read_text()
        if 'tp8_decoder_default' in manifest:
            # Re-freeze a screened GEMM geometry with the selected decoder.
            # Regenerate from its verified original runtime, avoiding stacked
            # patches while retaining the already-built wrapper and its ABI.
            source = repo/'src/gemm_rs/taco_runtime.cu'
            assert manifest['inputs'][str(source)] == sha(source)
            for name in ('taco_runtime.h', 'taco_codec.cuh', 'epilogue_evt.hpp'):
                assert sha(repo/'src/gemm_rs'/name) == sha(runtime.parent/name)
            original = source.read_text()
        runtime.write_text(patch_runtime(original, args.default))
        shutil.copy2(header, runtime.parent/header.name)
        for name in ('libflux_cuda.so', 'libflux_cuda_ths_op.so'):
            link = new/'taco/python/flux/lib'/name
            link.unlink()
            link.symlink_to(new/'taco'/name)
        commands = [dict(args=[v.replace(str(old), str(new)) for v in c['args']],
                         cwd=c['cwd'].replace(str(old), str(new)))
                    for c in manifest['commands']]
        for command in commands:
            argv = command['args']
            if '-o' in argv and Path(argv[argv.index('-o')+1]).name in (
                    'taco_runtime.o', 'device_link.o', 'libflux_cuda.so', 'libflux_cuda_ths_op.so'):
                print('BUILD', policy, Path(argv[argv.index('-o')+1]).name, flush=True)
                subprocess.run(argv, cwd=command['cwd'], check=True)
        manifest.update(commands=commands, tp8_decoder_default=args.default,
            decoder_parent_manifest=dict(path=str(old/'manifest.json'), sha256=sha(old/'manifest.json')))
        for p in (Path(__file__).resolve(), header):
            manifest['inputs'][str(p)] = sha(p)
        manifest['files'] = {str(p.relative_to(new)):sha(p) for p in new.rglob('*')
                            if p.is_file() and not p.is_symlink() and
                            p.name not in ('manifest.json', 'build-progress.json')}
        (new/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print('TP8 DECODER BUILD COMPLETE', policy, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('base', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--default', type=int, choices=[0,14,15,16,17,18,19], default=0)
    main(p.parse_args())

"""Compile TP=8-capable checkers against the unchanged three policy overlays."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'outputs/a800/gemm_rs_validation/source'
BUILD=ROOT/'outputs/a800/phase3/three-way-build'
OUT=ROOT/'outputs/a800/megatron-e2e/scale-mapping'
OUT.mkdir(parents=True,exist_ok=True)
source=Path(__file__).with_name('check_mapping.cu')
manifest=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),commands=[],files={})
for policy in ['original','remote_first','interleaved']:
    includes=[BUILD/policy/'overlay',BASE/'src',BASE/'include',BASE/'build',BASE/'3rdparty/cutlass/include',BASE/'3rdparty/cutlass/tools/util/include']
    command=['/data/apps/cuda/12.8/bin/nvcc','-std=c++17','-O3','--expt-relaxed-constexpr',
        '-gencode=arch=compute_80,code=[sm_80,compute_80]',*['-I'+str(p) for p in includes],str(source),'-o',str(OUT/f'check_mapping-{policy}')]
    subprocess.run(command,check=True);manifest['commands'].append(command)
    for p in [OUT/f'check_mapping-{policy}',BUILD/policy/'overlay/gemm_rs/tile_scheduler/threadblock_swizzle.hpp']:
        manifest['files'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Mapping checkers compiled serially; Flux libraries unchanged')

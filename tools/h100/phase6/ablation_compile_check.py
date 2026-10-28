"""Compile ablation tables/mapping CUDA and run only CPU table/mask checks.

This does not build runnable Flux libraries, query a GPU, load a driver, or run
the CUDA mapping executable. Full libraries use candidate.py compile on H100.
"""
import argparse
import os
from pathlib import Path
import subprocess

import candidate
from common import read, require, sha, write


def check(builds, out, cuda_home, cxx='g++'):
    info = candidate.verify(builds)
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    cuda_home = Path(cuda_home).resolve()
    require((cuda_home/'bin/nvcc').is_file(), 'CUDA Toolkit nvcc required')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = dict(gpu_used=False, gpu_validated=False, diagnostic_only=True, commands=[], variants={})
    def run(cmd):
        cmd = list(map(str, cmd)); result['commands'].append(cmd)
        with (out/'compile.log').open('a') as log:
            log.write(repr(cmd)+'\n'); log.flush()
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        for identity, v in info['variants'].items():
            if v['group'] in ('native', 'native_taco'):
                continue
            src = Path(builds).resolve()/v['build']/identity
            entry = read(Path(builds)/v['plan'])['entry']
            masks = ','.join(str(x) for row in entry['mask'] for x in row)
            coords = ','.join(str(x) for row in entry['coords'] for x in row)
            host = out/f'{identity}.cc'
            host.write_text('''#include <cstdio>
#include <set>
#include "phase6_host.hpp"
#include "gemm_rs/taco_runtime.h"
int main() {
  const unsigned char mask[] = {MASK};
  const int coords[] = {COORDS};
  FluxTacoConfig c; c.enabled=1; c.world=8; c.m=2048; c.n=4096; c.selected=mask;
  for(int src=0; src<8; ++src) {
    std::set<int> seen;
    for(int i=0; i<512; ++i) {
      int physical=phase6_host_lookup(16,32,2048,src,i);
      if(physical!=coords[src*512+i] || !seen.insert(physical).second) return 1;
      for(int dst=0; dst<8; ++dst) {
        bool expected=src!=dst && bool(mask[src*512+i]);
        if(taco_selected(c,src,dst,i)!=expected) return 2;
      }
    }
  }
  puts("CPU physical table and taco_selected mask checks passed; no CUDA calls");
}
'''.replace('MASK', masks).replace('COORDS', coords))
            executable = out/f'{identity}-host'
            run([cxx, '-std=c++17', '-O2', '-I'+str(src), '-I'+str(src/'src'),
                 '-I'+str(cuda_home/'include'), host, '-o', executable])
            run([executable])
            inc = [src, src/'src', src/'include', src/'3rdparty/cutlass/include',
                   src/'3rdparty/cutlass/tools/util/include']
            obj = out/f'{identity}-mapping.o'
            run([cuda_home/'bin/nvcc', '-std=c++17', '-O2', '--expt-relaxed-constexpr', '--expt-extended-lambda',
                 '-gencode=arch=compute_90,code=sm_90', *['-I'+str(p) for p in inc],
                 '-c', src/'check_mapping.cu', '-o', obj])
            result['variants'][identity] = dict(host_passed=True, mapping_compiled=True,
                                               plan_sha256=v['derived_plan_sha256'], object_sha256=sha(obj))
            print('CPU/compile', identity, flush=True)
        # Compute/codec implementation is identical across all ablation groups.
        runtime = out/'taco-runtime.o'
        run([cuda_home/'bin/nvcc', '-std=c++17', '-O2', '-gencode=arch=compute_90,code=sm_90',
             '-I'+str(src/'src'), '-c', src/'src/gemm_rs/taco_runtime.cu', '-o', runtime])
        result.update(completed=True, runtime_object_sha256=sha(runtime))
    finally:
        write(out/'validation.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--builds', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cuda-home', type=Path, required=True); p.add_argument('--cxx', default='g++')
    a = p.parse_args(); check(a.builds, a.out, a.cuda_home, a.cxx)

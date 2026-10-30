"""Compile/link the real MAV1 host TUs and load their pybind registration.

No GPU is queried or run. Only the existing GemmRS bodies/registry are replaced
by link fixtures; their headers, MAV1 native code, and MAV1 pybind are real.
This proves host API/ABI/linkage, NOT Flux integration or CUDA correctness.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

from common import HERE, REPO, require, sha, write
import ma4_f_build
import mav1_sources

FIXTURE = r'''
#include "flux/ths_op/ths_pybind.h"
#include "gemm_rs/ths_op/mav1_native.h"
namespace bytedance::flux::ths_op {
GemmRS::~GemmRS() {}
at::Tensor GemmRS::ma4_consumer_forward(at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, const ReduceScatterOptionWithOptional&) {
  TORCH_CHECK(false, "CPU linkage fixture cannot execute GemmRS");
}
std::vector<int64_t> GemmRS::ma4_consumer_state() const {
  TORCH_CHECK(false, "CPU linkage fixture cannot execute GemmRS");
}
ThsOpsInitRegistry& ThsOpsInitRegistry::instance() {
  static ThsOpsInitRegistry value; return value;
}
void ThsOpsInitRegistry::register_one(std::string name, OpInitFunc&& fn) {
  registry_.emplace(std::move(name), std::move(fn));
}
void ThsOpsInitRegistry::initialize_all(py::module& module) const {
  for (const auto& entry : registry_) entry.second(module);
}
}
PYBIND11_MODULE(mav1_native_check, module) {
  using namespace bytedance::flux::ths_op;
  // Same holder selected by the real candidate GemmRS registration.
  py::class_<TorchClassWrapper<GemmRS>,
      std::shared_ptr<TorchClassWrapper<GemmRS>>>(module, "GemmRS");
  py::class_<bytedance::flux::ReduceScatterOptionWithOptional>(module, "ReduceScatterOption");
  ThsOpsInitRegistry::instance().initialize_all(module);
}
'''


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cuda-home', type=Path, required=True)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args(argv)
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    out = args.out.resolve()
    require(not out.exists(), 'Native check requires a fresh output directory')
    out.mkdir(parents=True)
    reference = ma4_f_build.prepare(out / 'reference')
    updates = mav1_sources.sources(reference / 'F4')
    candidate = out / 'candidate'
    for name, source in updates.items():
        target = candidate / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    fixture = out / 'fixture.cc'
    fixture.write_text(FIXTURE)
    includes = [candidate / 'src', reference / 'F4/src', REPO / 'src', REPO / 'include',
                args.cuda_home / 'include', REPO / '3rdparty/cutlass/include',
                REPO / '3rdparty/cutlass/tools/util/include', Path(sysconfig.get_paths()['include']),
                *map(Path, include_paths())]
    flags = ['-std=c++17', '-O0', '-g0', '-fPIC',
             '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    flags += [item for path in includes for item in ('-I', str(path))]
    commands = []
    def run(name, command):
        commands.append(command)
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / (name + '.log')).write_text(result.stdout)
        require(result.returncode == 0, name + ' failed; see ' + str(out / (name + '.log')))
        return result.stdout

    objects = []
    for name, source in [('native', candidate / mav1_sources.NATIVE_SOURCE),
                         ('pybind', candidate / mav1_sources.PYBIND_SOURCE), ('fixture', fixture)]:
        obj = out / (name + '.o')
        run(name, [args.cxx, *flags, '-c', str(source), '-o', str(obj)])
        objects.append(str(obj))
    libraries = [*map(Path, library_paths()), args.cuda_home / 'lib64']
    module = out / ('mav1_native_check' + sysconfig.get_config_var('EXT_SUFFIX'))
    link = [args.cxx, '-shared', *objects, '-o', str(module)]
    link += [item for path in libraries for item in ('-L', str(path), '-Wl,-rpath,' + str(path))]
    link += ['-ltorch', '-ltorch_cpu', '-ltorch_python', '-lc10', '-ltorch_cuda', '-lc10_cuda', '-lcudart']
    run('link', link)
    symbols = run('symbols', ['nm', '-D', '--defined-only', str(module)])
    require('mav1_native_protocol_version' in symbols, 'Native protocol symbol is not exported')
    smoke = "import sys,torch;sys.path.insert(0,sys.argv[1]);import mav1_native_check as m;" \
            "assert m.mav1_native_protocol()==1;assert hasattr(m,'Mav1ColumnPlan');" \
            "assert hasattr(m,'Mav1RowPlan');assert torch.classes.flux_mav1.ColumnPlan;" \
            "assert torch.classes.flux_mav1.RowPlan;print('native protocol and typed classes loaded')"
    run('load', [sys.executable, '-c', smoke, str(out)])
    record = dict(passed=True, gpu_executed=False, full_flux_linked=False,
        validation='real_native_and_pybind_compile_link_load_with_GemmRS_body_link_fixtures',
        torch_version=torch.__version__, commands=commands,
        sources={str(HERE / name): sha(HERE / name) for name in
                 ('mav1_native.h', 'mav1_native.cc', 'mav1_pybind.cc', 'mav1_sources.py')},
        module_sha256=sha(module))
    write(out / 'validation.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

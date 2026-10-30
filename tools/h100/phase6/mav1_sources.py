"""Exact MA11 control; native host autograd plan only in the MAV1 candidate."""
from pathlib import Path
from common import HERE, replace_once, require
import ma11_sources as baseline

RUNTIME, HEADER = baseline.RUNTIME, baseline.HEADER
WRAPPER, DECODER = baseline.WRAPPER, baseline.DECODER
ZERO_COUNT_EXPORTS = baseline.ZERO_COUNT_EXPORTS
SOFTMAX_EXPORTS = baseline.NATIVE_EXPORTS
VARIANTS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-native-linear')
BASELINE_TUNING, DEFAULT_TUNING = baseline.BASELINE_TUNING, baseline.DEFAULT_TUNING
ROLES, ROLE_K = baseline.ROLES, baseline.ROLE_K
tuning_config = baseline.tuning_config
NATIVE_HEADER = 'src/gemm_rs/ths_op/mav1_native.h'
NATIVE_SOURCE = 'src/gemm_rs/ths_op/mav1_native.cc'
PYBIND_SOURCE = 'src/pybind/mav1_native.cc'
GEMM_PYBIND = 'src/pybind/gemm_rs.cc'
NATIVE_EXPORTS = ()  # typed pybind APIs, not libflux_cuda C exports


def sources(reference, arm=VARIANTS[1]):
    require(arm in VARIANTS, 'Unknown MAV1 source arm')
    reference = Path(reference)
    result = baseline.sources(reference, baseline.VARIANTS[1])
    if arm == VARIANTS[1]:
        result[NATIVE_HEADER] = (HERE / 'mav1_native.h').read_text()
        result[NATIVE_SOURCE] = (HERE / 'mav1_native.cc').read_text()
        result[PYBIND_SOURCE] = (HERE / 'mav1_pybind.cc').read_text()
        # C++ contexts own plans, plans own both original native slots. The
        # candidate holder change prevents Python wrapper destruction from
        # invalidating a backward node. The registered MA11 control is untouched.
        current = result.get(GEMM_PYBIND, (reference / GEMM_PYBIND).read_text())
        result[GEMM_PYBIND] = replace_once(current,
            'py::class_<GemmRSOpCls>(m, "GemmRS")',
            'py::class_<GemmRSOpCls, std::shared_ptr<GemmRSOpCls>>(m, "GemmRS")')
    return result


def kernel_name(role, tuning=None, arm=VARIANTS[0]):
    require(arm in VARIANTS, 'Unknown MAV1 arm')
    return baseline.kernel_name(role, tuning, baseline.VARIANTS[1])

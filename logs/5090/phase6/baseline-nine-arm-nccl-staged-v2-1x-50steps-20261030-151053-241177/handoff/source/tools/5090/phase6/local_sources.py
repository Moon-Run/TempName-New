"""Explicit local-memory transforms of the registered TACO source trees."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
NATIVE_EXPORTS = ('rtx5090_local_protocol_version', 'rtx5090_local_last_error',
                  'rtx5090_local_create', 'rtx5090_local_reconfigure',
                  'rtx5090_local_reconfiguration_count', 'rtx5090_local_workspace_bytes',
                  'rtx5090_local_produce',
                  'rtx5090_local_decode', 'rtx5090_local_state', 'rtx5090_local_destroy')
CONSUMER_ARMS = ('ma11', 'ma89', 'ma8', 'ma9', 'cr3')
SELECTIVE_ARMS = CONSUMER_ARMS + ('c',)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Frozen local transport transform anchor changed: ' + old)
    return text.replace(old, new, 1)


def sources(source, kind):
    """Apply to a frozen H100 variant before any legacy direct-peer port.

    A/C/R retain their original producer ordering and decoder. Only epilogue
    addresses change: destination-indexed local tx slots replace peer pointers.
    The new C ABI is independent of the legacy GemmRS/GroupBarrier constructor.
    """
    if kind not in (*CONSUMER_ARMS, 'a', 'r', 'c'):
        raise ValueError('Unknown local transport arm: ' + str(kind))
    source = Path(source)
    name = 'src/gemm_rs/epilogue_evt.hpp'
    epilogue = replace_once((source/name).read_text(),
        'unsigned char* slot = c.peers[dst_rank] + c.rank * taco_source_bytes(c);',
        'unsigned char* slot = c.peers[dst_rank];  // Local tx slot; NCCL moves it after GEMM.')
    epilogue = replace_once(epilogue,
        'int target_line = params.rank * M_lines_per_rank + cur_row % M_lines_per_rank;',
        'int target_line = cur_row % M_lines_per_rank;  // Destination-indexed local staging.')
    epilogue = replace_once(epilogue, 'Params params;', 'Params params{};')
    epilogue = replace_once(epilogue,
        'params.barrier_ptr_aux ? params.barrier_ptr_aux[dst_rank] : nullptr;',
        'params.has_barrier_ptr ? params.barrier_ptr_aux[dst_rank] : nullptr;')
    epilogue = epilogue.replace('The existing GemmRS group barrier publishes completion before DQ.',
        'The producer stream orders pack after GEMM; NCCL completion orders unpack and DQ.')
    cmake_name = 'src/gemm_rs/CMakeLists.txt'
    cmake = replace_once((source/cmake_name).read_text(), 'set(CU_FILES\n',
                         'set(CU_FILES\n  local_transport.cu\n')
    cmake = replace_once(cmake, '  file(GLOB THS_FILES ths_op/*.cc)',
        '  file(GLOB THS_FILES ths_op/*.cc)\n'
        '  # NCCL staging uses the local C ABI, never the legacy IPC constructor.\n'
        '  list(REMOVE_ITEM THS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/ths_op/gemm_reduce_scatter.cc")')
    version_name = 'src/cuda/version.ld'
    version = replace_once((source/version_name).read_text(), 'global:',
                           'global:\n    rtx5090_local_*;')
    native = '#define RTX5090_LOCAL_CONSUMER ' + str(int(kind in CONSUMER_ARMS)) + '\n'
    native += '#define RTX5090_LOCAL_SELECTIVE ' + str(int(kind in SELECTIVE_ARMS)) + '\n'
    native += (HERE/'local_transport.cu').read_text()
    runtime_name = 'src/gemm_rs/taco_runtime.cu'
    runtime = (source/runtime_name).read_text() + '''
// Private local-producer TLS binding; the arm's wire/decoder ABI is unchanged.
extern "C" void rtx5090_local_set_taco_config(FluxTacoConfig config) { active = config; }
'''
    pybind_name = 'src/pybind/gemm_rs.cc'
    pybind = (source/pybind_name).read_text()
    begin = pybind.index('    py::class_<GemmRSOpCls>(m, "GemmRS")')
    end = pybind.index('    py::class_<SegmentInfo>', begin)
    pybind = (pybind[:begin] +
        '    // Legacy GemmRS is deliberately unavailable in the no-P2P build.\n'
        '    // LocalTaco uses the separately exported rtx5090_local_* C ABI.\n\n' + pybind[end:])
    return {name: epilogue, cmake_name: cmake, version_name: version, runtime_name: runtime,
            pybind_name: pybind,
            'src/gemm_rs/local_transport.cu': native,
            'src/gemm_rs/local_layout.h': (HERE/'local_layout.h').read_text(),
            'src/gemm_rs/local_transport.h': (HERE/'local_transport.h').read_text()}

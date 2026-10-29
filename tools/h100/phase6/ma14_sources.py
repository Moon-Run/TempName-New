"""MA11 with a monotonic coarse rendezvous, preserving its dual-slot schedule.

No producer, decoder, FluxTacoConfig ABI, numerical kernel or packet changes.
The baseline is precisely the MA11 candidate. The candidate replaces one
binary IPC barrier with one monotonic publication kernel per projection.
"""
from pathlib import Path

from common import HERE, replace_once, require
import ma11_sources as baseline
import ma4_f_sources as f4

RUNTIME, HEADER = baseline.RUNTIME, baseline.HEADER
WRAPPER, DECODER = baseline.WRAPPER, baseline.DECODER
WRAPPER_HEADER, PYBIND, FRONTEND = f4.WRAPPER_HEADER, f4.PYBIND, f4.FRONTEND
ZERO_COUNT_HEADER = baseline.ZERO_COUNT_HEADER
ZERO_COUNT_INCLUDE = baseline.ZERO_COUNT_INCLUDE
ZERO_COUNT_EXPORTS = baseline.ZERO_COUNT_EXPORTS
SOFTMAX_EXPORTS = baseline.NATIVE_EXPORTS
VERSION_SCRIPT, CUDA_CMAKE = baseline.VERSION_SCRIPT, baseline.CUDA_CMAKE
EPOCH_HEADER = 'src/gemm_rs/ma14_epoch.cuh'
NATIVE_EXPORTS = ('taco_ma14_epoch_protocol', 'taco_ma14_epoch_barrier')
VARIANTS = ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-epoch')
BASELINE_TUNING = baseline.BASELINE_TUNING
ROLES, ROLE_K = baseline.ROLES, baseline.ROLE_K
DEFAULT_TUNING = baseline.DEFAULT_TUNING
tuning_config = baseline.tuning_config
STATE_KEYS = ('protocol_version', 'initializations', 'epoch', 'local_ready_ptr', 'failed')


def header(source):
    return source + r'''

// Separate launch ABI: do not increase the producer's FluxTacoConfig argument.
struct Ma14EpochConfig {
  unsigned long long* peers[8]{};
  unsigned long long epoch = 0;
  int rank = -1;
};
extern "C" int taco_ma14_epoch_protocol();
extern "C" int taco_ma14_epoch_barrier(Ma14EpochConfig, cudaStream_t);
'''


def wrapper(source):
    source = replace_once(source, '  bool cr2_failed = false;', '''  bool cr2_failed = false;
  std::vector<torch::Tensor> ma14_ready_buffers;
  int64_t ma14_initializations = 0;
  int64_t ma14_epoch = 0;''')
    anchor = '''      // Input and optimizer-updated weight STORAGE may change between calls.'''
    source = replace_once(source, anchor, '''      // Allocate and clear once per complete native double-buffer slot.
      // The all-rank initialization dependency must finish before publication:
      // a late initializer must never erase an already-published epoch.
      if (ma14_ready_buffers.empty()) {
        ma14_ready_buffers = flux_create_tensor_list({8}, c10::ScalarType::Long, this->group_.get());
        CUDA_CHECK(cudaMemsetAsync(ma14_ready_buffers[rank].data_ptr(), 0,
                                   8 * sizeof(int64_t), stream));
        group_barrier.barrier_all(stream);
        ++ma14_initializations;
      }
      TORCH_CHECK(ma14_epoch < INT64_MAX, "MA14 epoch exhaustion; reconstruct both slots");
''' + anchor)
    source = replace_once(source, '''      // Identical C publication rendezvous. Double buffering continues to guard
      // previous consumers before a later invocation reuses this source slot.
      group_barrier.barrier_all(stream);''', '''      // Retain the full publication rendezvous and dual-slot reuse proof.
      // Only the binary CAS/reset handshake changes to monotonic tickets.
      Ma14EpochConfig publication;
      publication.rank = rank;
      publication.epoch = static_cast<unsigned long long>(++ma14_epoch);
      for (int peer = 0; peer < 8; ++peer)
        publication.peers[peer] = static_cast<unsigned long long*>(ma14_ready_buffers[peer].data_ptr());
      CUDA_CHECK(static_cast<cudaError_t>(taco_ma14_epoch_barrier(publication, stream)));''')
    source = replace_once(source, '''  std::vector<int64_t> ma4_consumer_state() const {''', '''  std::vector<int64_t> ma14_epoch_state() const {
    std::lock_guard<std::recursive_mutex> lock(ma4_consumer_mutex);
    return {14, ma14_initializations, ma14_epoch,
            ma14_ready_buffers.empty() ? 0 : static_cast<int64_t>(
                reinterpret_cast<uintptr_t>(ma14_ready_buffers[rank].data_ptr())),
            cr2_failed ? 1 : 0};
  }

  std::vector<int64_t> ma4_consumer_state() const {''')
    source = replace_once(source, 'GemmRS::~GemmRS() { delete impl_; }', '''GemmRS::~GemmRS() { delete impl_; }
std::vector<int64_t> GemmRS::ma14_epoch_state() const {
  FLUX_CHECK(impl_ != nullptr);
  return impl_->ma14_epoch_state();
}''')
    return source


PAIR_GUARD = '''
class _Ma14EpochPair:
    """Enforce the two complete workspaces and one common ordered stream."""
    def __init__(self):
        self.stream = torch.cuda.current_stream().cuda_stream
        self.device = torch.cuda.current_device()
        self.slots = [None, None]
        self.next_slot = 0
        self.failed = False
        self.lock = threading.Lock()

    def register(self, slot, instance):
        if (type(slot) is not int or slot not in (0, 1) or self.slots[slot] is not None
                or instance.stream != self.stream or instance.device != self.device):
            raise ValueError('MA14 requires two distinct slots on one creation stream')
        # Store registration only; retaining the instance would create a
        # pair->slot->pair cycle and defer CUDA workspace destruction to GC.
        self.slots[slot] = True

    @contextlib.contextmanager
    def submission(self, slot):
        with self.lock:
            if self.failed:
                raise RuntimeError('MA14 cannot reuse a failed epoch pair')
            try:
                if (any(item is None for item in self.slots) or slot != self.next_slot
                        or torch.cuda.current_device() != self.device
                        or torch.cuda.current_stream().cuda_stream != self.stream):
                    raise RuntimeError('MA14 requires alternating complete slots on one fixed stream')
                yield
            except BaseException:
                self.failed = True
                raise
            self.next_slot ^= 1


'''


def frontend(source):
    source = replace_once(source, 'import ctypes\n', 'import ctypes\nimport contextlib\n')
    source = replace_once(source, 'class GemmRSTaco:\n', PAIR_GUARD + 'class GemmRSTaco:\n')
    source = replace_once(source,
        '                 _defer_reuse_barrier=False, quantization_mode=None):',
        '                 _defer_reuse_barrier=False, quantization_mode=None,\n'
        '                 _ma14_pair=None, _ma14_slot=None):')
    source = replace_once(source, "        if placement not in (None, 'fused', 'separate'):",
        "        if not isinstance(_ma14_pair, _Ma14EpochPair):\n"
        "            raise ValueError('MA14 requires GemmRSTacoDoubleBuffered ownership')\n"
        "        self._ma14_pair, self._ma14_slot = _ma14_pair, _ma14_slot\n"
        "        if placement not in (None, 'fused', 'separate'):")
    source = replace_once(source, '        self.m, self.n, self.k_local = m, n, k_local',
        '        self.m, self.n, self.k_local = m, n, k_local\n'
        '        self._ma14_pair.register(self._ma14_slot, self)')
    source = replace_once(source, '            self.lib.taco_ma4f_protocol_version.argtypes = []',
        '            self.lib.taco_ma14_epoch_protocol.argtypes = []\n'
        '            self.lib.taco_ma14_epoch_protocol.restype = ctypes.c_int\n'
        '            if self.lib.taco_ma14_epoch_protocol() != 14:\n'
        "                raise RuntimeError('Wrong MA14 epoch protocol')\n"
        '            self.lib.taco_ma4f_protocol_version.argtypes = []')
    source = replace_once(source,
        '                return self.op.forward(input, weight, reduce_scatter_option=self.option)',
        '                with self._ma14_pair.submission(self._ma14_slot):\n'
        '                    return self.op.forward(input, weight, reduce_scatter_option=self.option)')
    source = replace_once(source,
        '                return self.op.ma4_consumer_forward(input, weight, bias, residual, self.option)',
        '                with self._ma14_pair.submission(self._ma14_slot):\n'
        '                    return self.op.ma4_consumer_forward(input, weight, bias, residual, self.option)')
    source = replace_once(source,
        '        self.slots = [GemmRSTaco(*args, **kwargs, _defer_reuse_barrier=True) for _ in range(2)]',
        '        pair = _Ma14EpochPair()\n'
        '        self.slots = [GemmRSTaco(*args, **kwargs, _defer_reuse_barrier=True,\n'
        '                                  _ma14_pair=pair, _ma14_slot=slot) for slot in range(2)]')
    marker = '    def ma4_consumer_state(self):'
    require(source.count(marker) == 2, 'MA14 requires original two consumer state accessors')
    first = source.index(marker)
    source = source[:first] + '''    def ma14_epoch_state(self):
        keys = STATE_KEYS_LITERAL
        values = self.op.ma14_epoch_state()
        if len(values) != len(keys):
            raise RuntimeError('MA14 native epoch state ABI mismatch')
        state = dict(zip(keys, values))
        state['failed'] = bool(state['failed'] or self._failed or self._ma14_pair.failed)
        return state

'''.replace('STATE_KEYS_LITERAL', repr(STATE_KEYS)) + source[first:]
    second = source.index(marker, source.index('class GemmRSTacoDoubleBuffered:'))
    source = source[:second] + '''    def ma14_epoch_state(self):
        return [dict(slot=index, **slot.ma14_epoch_state())
                for index, slot in enumerate(self.slots)]

''' + source[second:]
    return source


def sources(reference, arm=VARIANTS[1]):
    require(arm in VARIANTS, 'Unknown MA14 source arm')
    reference = Path(reference)
    result = baseline.sources(reference, baseline.VARIANTS[1])
    if arm == VARIANTS[0]:
        return result
    for name, transform in ((HEADER, header), (WRAPPER, wrapper), (FRONTEND, frontend)):
        result[name] = transform(result.get(name, (reference / name).read_text()))
    result[WRAPPER_HEADER] = replace_once((reference / WRAPPER_HEADER).read_text(),
        '  std::vector<int64_t> ma4_consumer_state() const;',
        '  std::vector<int64_t> ma4_consumer_state() const;\n'
        '  std::vector<int64_t> ma14_epoch_state() const;')
    result[PYBIND] = replace_once((reference / PYBIND).read_text(),
        '        .def("ma4_consumer_state", &GemmRSOpCls::ma4_consumer_state)',
        '        .def("ma4_consumer_state", &GemmRSOpCls::ma4_consumer_state)\n'
        '        .def("ma14_epoch_state", &GemmRSOpCls::ma14_epoch_state)')
    result[EPOCH_HEADER] = (HERE / 'ma14_epoch.cuh').read_text()
    result[RUNTIME] += '\n#include "gemm_rs/ma14_epoch.cuh"\n'
    result[VERSION_SCRIPT] = replace_once(result[VERSION_SCRIPT], 'global:\n',
        'global:\n' + ''.join('    ' + name + ';\n' for name in NATIVE_EXPORTS))
    return result


def kernel_name(role, tuning=None, arm=VARIANTS[0]):
    require(arm in VARIANTS, 'Unknown MA14 arm')
    return baseline.kernel_name(role, tuning, baseline.VARIANTS[1])

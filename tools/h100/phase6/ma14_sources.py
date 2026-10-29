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
PAIR_HEADER = 'src/gemm_rs/ma14_pair.h'
NATIVE_PROTOCOL = 1402
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
    source = replace_once(source, '#include <mutex>',
        '#include <mutex>\n#include "gemm_rs/ma14_pair.h"')
    source = replace_once(source, '  bool cr2_failed = false;', '''  bool cr2_failed = false;
  std::vector<torch::Tensor> ma14_ready_buffers;
  Ma14EpochConfig ma14_publication;
  std::shared_ptr<taco_ma14::PairState> ma14_pair;
  int ma14_slot = -1;
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
        ma14_publication.rank = rank;
        for (int peer = 0; peer < 8; ++peer)
          ma14_publication.peers[peer] =
              static_cast<unsigned long long*>(ma14_ready_buffers[peer].data_ptr());
        ++ma14_initializations;
      }
      TORCH_CHECK(ma14_epoch < INT64_MAX, "MA14 epoch exhaustion; reconstruct both slots");
''' + anchor)
    source = replace_once(source, '''      // Identical C publication rendezvous. Double buffering continues to guard
      // previous consumers before a later invocation reuses this source slot.
      group_barrier.barrier_all(stream);''', '''      // Retain the full publication rendezvous and dual-slot reuse proof.
      // Only the binary CAS/reset handshake changes to monotonic tickets.
      ma14_publication.epoch = static_cast<unsigned long long>(++ma14_epoch);
      CUDA_CHECK(static_cast<cudaError_t>(taco_ma14_epoch_barrier(ma14_publication, stream)));''')
    source = replace_once(source, '''  std::vector<int64_t> ma4_consumer_state() const {''', '''  std::vector<int64_t> ma14_epoch_state() const {
    std::unique_lock<std::recursive_mutex> pair_lock;
    if (ma14_pair) pair_lock = std::unique_lock<std::recursive_mutex>(ma14_pair->mutex);
    std::lock_guard<std::recursive_mutex> lock(ma4_consumer_mutex);
    return {1402, ma14_initializations, ma14_epoch,
            ma14_ready_buffers.empty() ? 0 : static_cast<int64_t>(
                reinterpret_cast<uintptr_t>(ma14_ready_buffers[rank].data_ptr())),
            cr2_failed || (ma14_pair && ma14_pair->failed) ? 1 : 0};
  }

  // Called once after the frontend has constructed both complete workspaces.
  // No raw peer-instance pointer is retained: detachment poisons shared state.
  void ma14_bind_pair(GemmRSImpl& other) {
    TORCH_CHECK(this != &other, "MA14 cannot bind a slot to itself");
    std::scoped_lock lock(ma4_consumer_mutex, other.ma4_consumer_mutex);
    TORCH_CHECK(!ma14_pair && !other.ma14_pair && !cr2_failed && !other.cr2_failed &&
                    cr2_calls == 0 && other.cr2_calls == 0 &&
                    ma14_epoch == 0 && other.ma14_epoch == 0,
                "MA14 pair binding requires two unused, unbound, nonfailed slots");
    TORCH_CHECK(owned_taco.enabled && other.owned_taco.enabled &&
                    owned_taco.world == 8 && other.owned_taco.world == 8 &&
                    owned_taco.m == 2048 && other.owned_taco.m == 2048 &&
                    owned_taco.n == 4096 && other.owned_taco.n == 4096 &&
                    owned_taco.rank == other.owned_taco.rank && rank == other.rank &&
                    cr2_k == other.cr2_k && (cr2_k == 512 || cr2_k == 2048) &&
                    owned_taco.deferred_reuse_barrier && other.owned_taco.deferred_reuse_barrier &&
                    !owned_taco.allow_capture && !other.owned_taco.allow_capture &&
                    output_buffer.data_ptr() != other.output_buffer.data_ptr(),
                "MA14 requires matching exact-shape complete double workspaces");
    auto endpoint = [](const GemmRSImpl& op) {
      return taco_ma14::Endpoint{
          reinterpret_cast<uintptr_t>(static_cast<cudaStream_t>(op.taco_stream)),
          op.taco_stream.device_index(),
          reinterpret_cast<uintptr_t>(op.owned_taco.peers[op.rank])};
    };
    auto pair = std::make_shared<taco_ma14::PairState>(endpoint(*this), endpoint(other));
    ma14_pair = pair;
    other.ma14_pair = std::move(pair);
    ma14_slot = 0;
    other.ma14_slot = 1;
  }

  template <typename Submit>
  torch::Tensor ma14_submit(Submit&& submit) {
    if (!owned_taco.enabled) return submit();
    // Both public entrypoints enter here exactly once. Their existing native
    // stream/device/layout/capture validation runs inside the failure guard.
    try {
      taco_ma14::Submission submission(ma14_pair.get(), ma14_slot);
      auto output = submit();
      submission.complete();
      return output;
    } catch (...) {
      std::lock_guard<std::recursive_mutex> lock(ma4_consumer_mutex);
      cr2_failed = true;
      throw;
    }
  }

  void ma14_reject_auxiliary() {
    if (owned_taco.enabled) {
      if (ma14_pair) ma14_pair->poison();
      std::lock_guard<std::recursive_mutex> lock(ma4_consumer_mutex);
      cr2_failed = true;
      TORCH_CHECK(false, "MA14 owned slots support only paired forward/forward_consumer");
    }
  }

  std::vector<int64_t> ma4_consumer_state() const {''')
    source = replace_once(source, 'GemmRS::~GemmRS() { delete impl_; }', '''GemmRS::~GemmRS() { delete impl_; }
void GemmRS::ma14_bind_pair(GemmRS& other) {
  FLUX_CHECK(impl_ != nullptr && other.impl_ != nullptr);
  impl_->ma14_bind_pair(*other.impl_);
}
std::vector<int64_t> GemmRS::ma14_epoch_state() const {
  FLUX_CHECK(impl_ != nullptr);
  return impl_->ma14_epoch_state();
}''')
    source = replace_once(source, '  ~GemmRSImpl() {', '''  ~GemmRSImpl() {
    if (ma14_pair) ma14_pair->detach(ma14_slot);''')
    source = replace_once(source, '''  return impl_->ma4_consumer_forward(std::move(input), std::move(weight),
      std::move(bias), std::move(residual), opt);''', '''  return impl_->ma14_submit([&] {
    return impl_->ma4_consumer_forward(std::move(input), std::move(weight),
        std::move(bias), std::move(residual), opt);
  });''')
    source = replace_once(source, '''  return impl_->forward(
      std::move(input),''', '''  return impl_->ma14_submit([&] {
    return impl_->forward(
      std::move(input),''')
    source = replace_once(source, '''      reduce_scatter_option);
}
torch::Tensor
GemmRS::profiling(''', '''      reduce_scatter_option);
  });
}
torch::Tensor
GemmRS::profiling(''')
    # These public split/submission utilities do not obey the two-slot proof.
    # Constructors and CR2 keep their internal helpers; direct misuse fails.
    for call in ('impl_->zero_buffers();', 'return impl_->profiling(',
                 'impl_->forward_barrier(std::move(input), std::move(weight), std::move(bias));',
                 'return impl_->forward_reduce_scatter('):
        source = replace_once(source, '  ' + call,
                              '  impl_->ma14_reject_auxiliary();\n  ' + call)
    return source



def frontend(source):
    # Public call methods retain their exact MA11 Python submission path.
    # Native binding after construction enforces ownership for direct op calls
    # as well, without a per-call generator, stream wrapper or Python pair lock.
    source = replace_once(source, '            self.lib.taco_ma4f_protocol_version.argtypes = []',
        '            self.lib.taco_ma14_epoch_protocol.argtypes = []\n'
        '            self.lib.taco_ma14_epoch_protocol.restype = ctypes.c_int\n'
        '            if self.lib.taco_ma14_epoch_protocol() != 1402:\n'
        "                raise RuntimeError('Wrong MA14-v2 epoch protocol')\n"
        '            self.lib.taco_ma4f_protocol_version.argtypes = []')
    source = replace_once(source,
        '        self.slots = [GemmRSTaco(*args, **kwargs, _defer_reuse_barrier=True) for _ in range(2)]',
        '        self.slots = [GemmRSTaco(*args, **kwargs, _defer_reuse_barrier=True) for _ in range(2)]\n'
        '        self.slots[0].op.ma14_bind_pair(self.slots[1].op)')
    marker = '    def ma4_consumer_state(self):'
    require(source.count(marker) == 2, 'MA14 requires original two consumer state accessors')
    first = source.index(marker)
    methods = """    def ma14_epoch_state(self):
        keys = STATE_KEYS_LITERAL
        values = self.op.ma14_epoch_state()
        if len(values) != len(keys):
            raise RuntimeError('MA14 native epoch state ABI mismatch')
        state = dict(zip(keys, values))
        state['failed'] = bool(state['failed'] or self._failed)
        return state

""".replace('STATE_KEYS_LITERAL', repr(STATE_KEYS))
    source = source[:first] + methods + source[first:]
    second = source.index(marker, source.index('class GemmRSTacoDoubleBuffered:'))
    methods = """    def ma14_epoch_state(self):
        return [dict(slot=index, **slot.ma14_epoch_state())
                for index, slot in enumerate(self.slots)]

"""
    return source[:second] + methods + source[second:]


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
        '  std::vector<int64_t> ma14_epoch_state() const;\n'
        '  void ma14_bind_pair(GemmRS&);')
    result[PYBIND] = replace_once((reference / PYBIND).read_text(),
        '        .def("ma4_consumer_state", &GemmRSOpCls::ma4_consumer_state)',
        '        .def("ma4_consumer_state", &GemmRSOpCls::ma4_consumer_state)\n'
        '        .def("ma14_epoch_state", &GemmRSOpCls::ma14_epoch_state)\n'
        '        .def("ma14_bind_pair", [](GemmRSOpCls& self, GemmRSOpCls& other) {\n'
        '          self.ma14_bind_pair(other);\n'
        '        })')
    result[EPOCH_HEADER] = (HERE / 'ma14_epoch.cuh').read_text()
    result[PAIR_HEADER] = (HERE / 'ma14_pair.h').read_text()
    result[RUNTIME] += '\n#include "gemm_rs/ma14_epoch.cuh"\n'
    result[VERSION_SCRIPT] = replace_once(result[VERSION_SCRIPT], 'global:\n',
        'global:\n' + ''.join('    ' + name + ';\n' for name in NATIVE_EXPORTS))
    return result


def kernel_name(role, tuning=None, arm=VARIANTS[0]):
    require(arm in VARIANTS, 'Unknown MA14 arm')
    return baseline.kernel_name(role, tuning, baseline.VARIANTS[1])

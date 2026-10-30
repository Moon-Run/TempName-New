// MA14-v2 host-only ownership guard. No CUDA, Torch, callbacks or GPU work.
#pragma once
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace taco_ma14 {
struct Endpoint {
  uintptr_t stream;
  int device;
  uintptr_t packet;
};

struct PairState {
  const uintptr_t stream;
  const int device;
  mutable std::recursive_mutex mutex;
  int next_slot = 0;
  bool failed = false;
  bool active = false;
  bool attached[2] = {true, true};

  PairState(Endpoint first, Endpoint second) : stream(first.stream), device(first.device) {
    if (first.stream != second.stream || first.device != second.device ||
        !first.packet || !second.packet || first.packet == second.packet)
      throw std::runtime_error("MA14 requires distinct complete slots on one stream/device");
  }

  void detach(int slot) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (slot == 0 || slot == 1) attached[slot] = false;
    failed = true;
  }

  void poison() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    failed = true;
  }
};

// The pair lock is always acquired BEFORE an individual GemmRS instance's
// existing mutex. Recursive locking lets same-thread reentry fail explicitly
// instead of hanging; active forbids a nested submission from advancing slots.
class Submission {
  PairState* pair_;
  std::unique_lock<std::recursive_mutex> lock_;
  bool completed_ = false;

 public:
  Submission(PairState* pair, int slot) : pair_(pair) {
    if (!pair_) throw std::runtime_error("MA14 native slots must be bound before submission");
    lock_ = std::unique_lock<std::recursive_mutex>(pair_->mutex);
    if (pair_->failed || pair_->active || !pair_->attached[0] || !pair_->attached[1] ||
        slot != pair_->next_slot) {
      pair_->failed = true;
      throw std::runtime_error("MA14 requires a live, nonfailed, alternating native pair");
    }
    pair_->active = true;
  }

  Submission(const Submission&) = delete;
  Submission& operator=(const Submission&) = delete;

  void complete() {
    if (completed_) pair_->failed = true;
    if (pair_->failed || !pair_->attached[0] || !pair_->attached[1])
      throw std::runtime_error("MA14 cannot complete a failed or detached native pair");
    pair_->next_slot ^= 1;
    pair_->active = false;
    completed_ = true;
  }

  ~Submission() {
    if (!completed_) pair_->failed = true;
    pair_->active = false;
  }
};
}  // namespace taco_ma14

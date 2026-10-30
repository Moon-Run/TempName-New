// Copyright 2026. Licensed under the Apache License, Version 2.0.
// Fixed MA11 linear submission plans. No CUDA arithmetic is implemented here.
#pragma once
#include <torch/all.h>
#include <torch/custom_class.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include "gemm_rs/ths_op/gemm_reduce_scatter.h"

extern "C" int mav1_native_protocol_version();

namespace bytedance::flux::ths_op {

struct Mav1Plan : torch::CustomClassHolder {
  Mav1Plan(c10::intrusive_ptr<c10d::ProcessGroup>, int64_t, bool, int64_t);
  virtual ~Mav1Plan() = default;
  void check_stream() const;
  void check_tensor(const at::Tensor&) const;
  std::vector<int64_t> state() const;
  c10::intrusive_ptr<c10d::ProcessGroup> group;
  const int device;
  const cudaStream_t stream;
  const int64_t feature, plan_id;
  const bool profile;
  mutable std::recursive_mutex mutex;
  bool failed = false, active = false;
  int64_t forwards = 0, backwards = 0, next_slot = -1;
};

struct Mav1ColumnPlan final : Mav1Plan {
  Mav1ColumnPlan(c10::intrusive_ptr<c10d::ProcessGroup>, int64_t, bool, int64_t);
  at::Tensor apply(at::Tensor input, at::Tensor weight, std::optional<at::Tensor> bias);
};

struct Mav1RowPlan final : Mav1Plan {
  Mav1RowPlan(c10::intrusive_ptr<c10d::ProcessGroup>, std::shared_ptr<GemmRS>,
      std::shared_ptr<GemmRS>, ReduceScatterOptionWithOptional, int64_t, bool, int64_t);
  at::Tensor apply(at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor residual);
  void retain_storage(std::vector<at::Tensor>);
  // Declared before slots so packet/mask allocations outlive slot destructors.
  std::vector<at::Tensor> owned_storage;
  std::shared_ptr<GemmRS> slots[2];
  const ReduceScatterOptionWithOptional option;
};

}  // namespace bytedance::flux::ths_op

// Copyright 2026. Licensed under the Apache License, Version 2.0.
// The operation sequence mirrors the frozen Megatron/MA8 and MA4-F Functions.
#include "gemm_rs/ths_op/mav1_native.h"
#include <ATen/record_function.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/custom_function.h>
#include <optional>
#include <string>

extern "C" int mav1_native_protocol_version() { return 1; }

namespace bytedance::flux::ths_op {
namespace {
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

// All state is native and reference-counted. No Python object survives into a
// backward callback or destructor. Failure permanently poisons this plan.
struct Invocation {
  Mav1Plan& p;
  std::unique_lock<std::recursive_mutex> lock;
  bool success = false;
  explicit Invocation(Mav1Plan& plan) : p(plan), lock(plan.mutex) {
    TORCH_CHECK(!p.failed && !p.active, "MAV1 plan is failed or reentrant");
    p.active = true;
    try { p.check_stream(); } catch (...) { p.failed = true; p.active = false; throw; }
  }
  ~Invocation() { p.active = false; if (!success) p.failed = true; }
};

struct ProfileScope {
  std::string name;
  std::optional<at::RecordFunction> record;
  ProfileScope(const Mav1Plan& p, const char* role, const char* phase, int64_t call) {
    if (p.profile) {
      name = std::string("mav1.") + role + "." + phase + "." +
          std::to_string(p.plan_id) + "." + std::to_string(call);
      record.emplace(at::RecordScope::FUNCTION);
      record->before(name.c_str());
    }
  }
};

at::Tensor gather(Mav1Plan& p, at::Tensor input) {
  auto dimensions = input.sizes().vec();
  dimensions[0] *= 8;
  auto output = at::empty(dimensions, input.options());
  c10d::AllgatherOptions options;
  options.asyncOp = false;  // original all_gather_into_tensor(async_op=False)
  auto work = p.group->_allgather_base(output, input, options);
  TORCH_CHECK(work, "MAV1 missing AllGather Work");
  work->wait();
  return output;
}

struct Mav1ColumnFunction : torch::autograd::Function<Mav1ColumnFunction> {
  static at::Tensor forward(AutogradContext* ctx,
      c10::intrusive_ptr<Mav1ColumnPlan> p, at::Tensor input,
      at::Tensor weight, std::optional<at::Tensor> bias) {
    Invocation invocation(*p);
    const auto call = p->forwards + 1;
    ProfileScope scope(*p, "column", "forward", call);
    p->check_tensor(input); p->check_tensor(weight);
    TORCH_CHECK(input.sizes() == at::IntArrayRef({256, 1, 4096}) && input.is_contiguous(),
        "MAV1 column input must be contiguous [256,1,4096]");
    TORCH_CHECK(weight.sizes() == at::IntArrayRef({p->feature, 4096}) && weight.is_contiguous(),
        "MAV1 column weight layout changed");
    if (bias) {
      p->check_tensor(*bias);
      TORCH_CHECK(bias->sizes() == at::IntArrayRef({p->feature}) && bias->is_contiguous(),
          "MAV1 column bias layout changed");
    }
    auto total = gather(*p, input);  // independent MA8 allocation per context
    ctx->save_for_backward({input, weight, total});
    ctx->saved_data["plan"] = p;
    ctx->saved_data["call"] = call;
    ctx->saved_data["bias"] = bias.has_value();
    auto output = at::matmul(total, weight.t());
    if (bias) output = output + *bias;  // retain separate BF16 add
    ++p->forwards;
    invocation.success = true;
    return output;
  }

  static variable_list backward(AutogradContext* ctx, variable_list outputs) {
    auto p = ctx->saved_data["plan"].toCustomClass<Mav1ColumnPlan>();
    Invocation invocation(*p);
    ProfileScope scope(*p, "column", "backward", ctx->saved_data["call"].toInt());
    TORCH_CHECK(!at::GradMode::is_enabled(), "MAV1 does not support higher derivatives");
    TORCH_CHECK(outputs.size() == 1 && outputs[0].defined(), "MAV1 missing column gradient");
    auto saved = ctx->get_saved_variables();  // original SavedVariable version checks
    auto grad_output = outputs[0];
    p->check_tensor(grad_output);
    TORCH_CHECK(grad_output.sizes() == at::IntArrayRef({2048, 1, p->feature}),
        "MAV1 column gradient shape changed");
    auto grad_input = at::matmul(grad_output, saved[1]);
    // Exact prepare_input_tensors_for_wgrad_compute order: after dgrad GEMM.
    grad_output = grad_output.contiguous();
    auto total_input = saved[2].contiguous();
    grad_output = grad_output.view({2048, p->feature});
    total_input = total_input.view({2048, 4096});
    auto sub_grad_input = at::empty(saved[0].sizes(), saved[0].options().requires_grad(false));
    c10d::ReduceScatterOptions options;
    options.asyncOp = true;
    options.reduceOp = c10d::ReduceOp::SUM;
    auto work = p->group->_reduce_scatter_base(sub_grad_input, grad_input, options);
    TORCH_CHECK(work, "MAV1 missing ReduceScatter Work");
    auto grad_weight = at::matmul(grad_output.t(), total_input);
    at::Tensor grad_bias;
    if (ctx->saved_data["bias"].toBool()) grad_bias = grad_output.sum(0);
    work->wait();
    ++p->backwards;
    invocation.success = true;
    // Keep BF16 gradients and all original AccumulateGrad/DDP hooks.
    return {at::Tensor(), sub_grad_input, grad_weight, grad_bias};
  }
};

struct Mav1RowFunction : torch::autograd::Function<Mav1RowFunction> {
  static at::Tensor forward(AutogradContext* ctx, c10::intrusive_ptr<Mav1RowPlan> p,
      at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor residual) {
    Invocation invocation(*p);
    const auto call = p->forwards + 1;
    ProfileScope scope(*p, "row", "forward", call);
    TORCH_CHECK(!p->owned_storage.empty(), "MAV1 row packet/mask ownership is unbound");
    for (const auto& tensor : {input, weight, bias, residual}) p->check_tensor(tensor);
    TORCH_CHECK(input.sizes() == at::IntArrayRef({2048, 1, p->feature}) &&
        weight.sizes() == at::IntArrayRef({4096, p->feature}) && weight.is_contiguous() &&
        bias.sizes() == at::IntArrayRef({4096}) &&
        residual.sizes() == at::IntArrayRef({256, 1, 4096}) && residual.is_contiguous(),
        "MAV1 row shape/layout changed");
    ctx->save_for_backward({input, weight});
    ctx->saved_data["plan"] = p;
    ctx->saved_data["call"] = call;
    auto flat = input.reshape({2048, p->feature}).contiguous();
    auto output = p->slots[p->next_slot]->ma4_consumer_forward(
        flat, weight, bias, residual.reshape({256, 4096}), p->option);
    TORCH_CHECK(!output.is_view() && output.numel() == residual.numel(),
        "MAV1 requires independent MA4-F output");
    output.resize_(residual.sizes());  // same equal-numel metadata resize
    p->next_slot ^= 1;  // advance only after successful native submission
    ++p->forwards;
    invocation.success = true;
    return output;
  }

  static variable_list backward(AutogradContext* ctx, variable_list outputs) {
    auto p = ctx->saved_data["plan"].toCustomClass<Mav1RowPlan>();
    Invocation invocation(*p);
    ProfileScope scope(*p, "row", "backward", ctx->saved_data["call"].toInt());
    TORCH_CHECK(!at::GradMode::is_enabled(), "MAV1 does not support higher derivatives");
    TORCH_CHECK(outputs.size() == 1 && outputs[0].defined(), "MAV1 missing row gradient");
    auto saved = ctx->get_saved_variables();
    auto grad_output = outputs[0];
    p->check_tensor(grad_output);
    TORCH_CHECK(grad_output.sizes() == at::IntArrayRef({256, 1, 4096}),
        "MAV1 row gradient shape changed");
    auto grad_bias = grad_output.sum_to_size({4096});
    auto grad_residual = grad_output;  // preserve original alias semantics
    auto full_grad = gather(*p, grad_output.contiguous());
    auto grad_input = at::matmul(full_grad, saved[1]);
    auto grad_weight = at::matmul(full_grad.reshape({2048, 4096}).t(),
                                 saved[0].reshape({2048, p->feature}));
    ++p->backwards;
    invocation.success = true;
    return {at::Tensor(), grad_input, grad_weight, grad_bias, grad_residual};
  }
};
}  // namespace

Mav1Plan::Mav1Plan(c10::intrusive_ptr<c10d::ProcessGroup> pg, int64_t size,
    bool profiling, int64_t id)
    : group(std::move(pg)), device(c10::cuda::current_device()),
      stream(c10::cuda::getCurrentCUDAStream(device)), feature(size),
      plan_id(id), profile(profiling) {
  TORCH_CHECK(group && group->getSize() == 8 && group->getBackendName() == "nccl",
      "MAV1 requires the original TP8 NCCL ProcessGroup");
  TORCH_CHECK(plan_id >= 0, "MAV1 invalid plan id");
  check_stream();
}

void Mav1Plan::check_stream() const {
  TORCH_CHECK(c10::cuda::current_device() == device &&
      c10::cuda::getCurrentCUDAStream(device).stream() == stream,
      "MAV1 device/stream changed");
  cudaStreamCaptureStatus capture;
  const auto result = cudaStreamIsCapturing(stream, &capture);
  TORCH_CHECK(result == cudaSuccess && capture == cudaStreamCaptureStatusNone,
      "MAV1 forbids CUDA Graph capture");
}

void Mav1Plan::check_tensor(const at::Tensor& tensor) const {
  TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.get_device() == device &&
      tensor.scalar_type() == at::kBFloat16, "MAV1 requires local CUDA BF16 tensors");
}

std::vector<int64_t> Mav1Plan::state() const {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  return {1, forwards, backwards, failed ? 1 : 0, next_slot, feature};
}

Mav1ColumnPlan::Mav1ColumnPlan(c10::intrusive_ptr<c10d::ProcessGroup> pg,
    int64_t size, bool profiling, int64_t id) : Mav1Plan(std::move(pg), size, profiling, id) {
  TORCH_CHECK(size == 1536 || size == 2048, "MAV1 only supports QKV/FC1");
}

at::Tensor Mav1ColumnPlan::apply(at::Tensor input, at::Tensor weight,
    std::optional<at::Tensor> bias) {
  TORCH_CHECK(at::GradMode::is_enabled() && !c10::InferenceMode::is_enabled(),
      "MAV1 requires training autograd");
  return Mav1ColumnFunction::apply(
      c10::intrusive_ptr<Mav1ColumnPlan>::unsafe_reclaim_from_nonowning(this),
      std::move(input), std::move(weight), std::move(bias));
}

Mav1RowPlan::Mav1RowPlan(c10::intrusive_ptr<c10d::ProcessGroup> pg,
    std::shared_ptr<GemmRS> first, std::shared_ptr<GemmRS> second,
    ReduceScatterOptionWithOptional opt, int64_t k, bool profiling, int64_t id)
    : Mav1Plan(std::move(pg), k, profiling, id),
      slots{std::move(first), std::move(second)}, option(std::move(opt)) {
  TORCH_CHECK((k == 512 || k == 2048) && slots[0] && slots[1] && slots[0] != slots[1],
      "MAV1 requires distinct original row slots");
  for (const auto& slot : slots) {
    const auto state = slot->ma4_consumer_state();
    TORCH_CHECK(state.size() == 7 && state[0] == 1 && state[2] == 0 &&
        state[3] == 0 && state[4] == k && state[6] == 0,
        "MAV1 requires fresh original MA4-F slots");
  }
  next_slot = 0;
}

at::Tensor Mav1RowPlan::apply(at::Tensor input, at::Tensor weight,
    at::Tensor bias, at::Tensor residual) {
  TORCH_CHECK(at::GradMode::is_enabled() && !c10::InferenceMode::is_enabled(),
      "MAV1 requires training autograd");
  return Mav1RowFunction::apply(
      c10::intrusive_ptr<Mav1RowPlan>::unsafe_reclaim_from_nonowning(this),
      std::move(input), std::move(weight), std::move(bias), std::move(residual));
}

void Mav1RowPlan::retain_storage(std::vector<at::Tensor> buffers) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  TORCH_CHECK(!failed && !active && forwards == 0 && owned_storage.empty() && buffers.size() >= 18,
      "MAV1 backing storage must be bound once before use");
  for (auto& tensor : buffers) {
    TORCH_CHECK(tensor.defined() && tensor.is_cuda() && !tensor.requires_grad(),
        "MAV1 requires CUDA packet/mask storage without autograd");
    tensor = tensor.detach();
  }
  owned_storage = std::move(buffers);
}
}  // namespace bytedance::flux::ths_op

// Copyright 2026. Licensed under the Apache License, Version 2.0.
#include "flux/ths_op/ths_pybind.h"
#include "gemm_rs/ths_op/mav1_native.h"

namespace bytedance::flux::ths_op {
static int register_mav1 [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("mav1_native_linear", [](py::module& m) {
    // Register IValue types so the autograd context can own plans without any
    // Python reference or deprecated raw capsule. Pybind only constructs them.
    static auto column_type = torch::class_<Mav1ColumnPlan>("flux_mav1", "ColumnPlan");
    static auto row_type = torch::class_<Mav1RowPlan>("flux_mav1", "RowPlan");
    py::class_<Mav1ColumnPlan, c10::intrusive_ptr<Mav1ColumnPlan>>(m, "Mav1ColumnPlan")
        .def(py::init([](c10::intrusive_ptr<c10d::ProcessGroup> group,
             int64_t size, bool profile, int64_t id) {
          return c10::make_intrusive<Mav1ColumnPlan>(std::move(group), size, profile, id);
        }), py::arg("group"), py::arg("out_features"), py::arg("profile") = false,
            py::arg("plan_id") = 0)
        .def("apply", &Mav1ColumnPlan::apply, py::arg("input"), py::arg("weight"),
             py::arg("bias") = py::none(), py::call_guard<py::gil_scoped_release>())
        .def("state", &Mav1ColumnPlan::state);
    py::class_<Mav1RowPlan, c10::intrusive_ptr<Mav1RowPlan>>(m, "Mav1RowPlan")
        .def(py::init([](c10::intrusive_ptr<c10d::ProcessGroup> group,
             std::shared_ptr<TorchClassWrapper<GemmRS>> slot0,
             std::shared_ptr<TorchClassWrapper<GemmRS>> slot1,
             ReduceScatterOptionWithOptional option, int64_t k, bool profile, int64_t id) {
          return c10::make_intrusive<Mav1RowPlan>(std::move(group),
              std::static_pointer_cast<GemmRS>(slot0), std::static_pointer_cast<GemmRS>(slot1),
              std::move(option), k, profile, id);
        }), py::arg("group"), py::arg("slot0"), py::arg("slot1"), py::arg("option"),
            py::arg("k"), py::arg("profile") = false, py::arg("plan_id") = 0)
        .def("apply", &Mav1RowPlan::apply, py::arg("input"), py::arg("weight"),
             py::arg("bias"), py::arg("residual"), py::call_guard<py::gil_scoped_release>())
        .def("retain_storage", &Mav1RowPlan::retain_storage)
        .def("state", &Mav1RowPlan::state);
    m.def("mav1_native_protocol", &mav1_native_protocol_version);
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op

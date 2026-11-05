/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */
#include "grid_plan.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/script/printer/ir_docsifier.h>

namespace tvm {
namespace tl {
namespace tenstorrent {

GridTask::GridTask(int64_t task_id, int64_t x, int64_t y) {
  auto node = ffi::make_object<GridTaskNode>();
  node->task_id = task_id;
  node->x = x;
  node->y = y;
  data_ = std::move(node);
}

GridPlan::GridPlan(ffi::String operation, tirx::PrimFunc frontend,
                   DictAttrs module_attrs, tirx::Stmt body, tirx::Var core_x,
                   tirx::Var core_y, int64_t grid_x, int64_t grid_y,
                   ffi::Array<GridTask> tasks) {
  auto node = ffi::make_object<GridPlanNode>();
  node->operation = std::move(operation);
  node->frontend = std::move(frontend);
  node->module_attrs = std::move(module_attrs);
  node->body = std::move(body);
  node->core_x = std::move(core_x);
  node->core_y = std::move(core_y);
  node->grid_x = grid_x;
  node->grid_y = grid_y;
  node->tasks = std::move(tasks);
  data_ = std::move(node);
}

void GridTaskNode::RegisterReflection() {
  ffi::reflection::ObjectDef<GridTaskNode>()
      .def_ro("task_id", &GridTaskNode::task_id)
      .def_ro("x", &GridTaskNode::x)
      .def_ro("y", &GridTaskNode::y);
}

void GridPlanNode::RegisterReflection() {
  ffi::reflection::ObjectDef<GridPlanNode>()
      .def_ro("operation", &GridPlanNode::operation)
      .def_ro("frontend", &GridPlanNode::frontend)
      .def_ro("module_attrs", &GridPlanNode::module_attrs)
      .def_ro("body", &GridPlanNode::body)
      .def_ro("core_x", &GridPlanNode::core_x)
      .def_ro("core_y", &GridPlanNode::core_y)
      .def_ro("grid_x", &GridPlanNode::grid_x)
      .def_ro("grid_y", &GridPlanNode::grid_y)
      .def_ro("tasks", &GridPlanNode::tasks);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  GridTaskNode::RegisterReflection();
  GridPlanNode::RegisterReflection();
  using namespace script::printer;
  IRDocsifier::vtable().set_dispatch<GridPlan>(
      "", [](GridPlan plan, ffi::reflection::AccessPath, IRDocsifier d) {
        return d->AddMetadata(plan);
      });
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

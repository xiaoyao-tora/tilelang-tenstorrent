/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */
#ifndef TVM_TL_TENSTORRENT_IR_GRID_PLAN_H_
#define TVM_TL_TENSTORRENT_IR_GRID_PLAN_H_

#include "device_ir.h"
#include <tvm/ir/attrs.h>
#include <tvm/tirx/function.h>

namespace tvm {
namespace tl {
namespace tenstorrent {

constexpr const char *kGridPlanAttr = "tt.internal.grid_plan";

class GridTaskNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  int64_t task_id;
  int64_t x;
  int64_t y;
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.GridTask", GridTaskNode,
                                    ffi::Object);
};

class GridTask : public ffi::ObjectRef {
public:
  TVM_DLL GridTask(int64_t task_id, int64_t x, int64_t y);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GridTask, ffi::ObjectRef,
                                             GridTaskNode);
};

// All tasks share the immutable frontend and its metadata. The plan never owns
// the output module containing it, so retaining an instrument snapshot is safe.
class GridPlanNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  ffi::String operation;
  tirx::PrimFunc frontend;
  DictAttrs module_attrs;
  tirx::Stmt body;
  tirx::Var core_x;
  tirx::Var core_y;
  int64_t grid_x;
  int64_t grid_y;
  ffi::Array<GridTask> tasks;
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.GridPlan", GridPlanNode,
                                    ffi::Object);
};

class GridPlan : public ffi::ObjectRef {
public:
  TVM_DLL GridPlan(ffi::String operation, tirx::PrimFunc frontend,
                  DictAttrs module_attrs, tirx::Stmt body, tirx::Var core_x,
                  tirx::Var core_y, int64_t grid_x, int64_t grid_y,
                  ffi::Array<GridTask> tasks);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GridPlan, ffi::ObjectRef,
                                             GridPlanNode);
};

} // namespace tenstorrent
} // namespace tl
} // namespace tvm
#endif

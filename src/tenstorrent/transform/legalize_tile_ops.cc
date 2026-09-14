/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file tenstorrent/transform/legalize_tile_ops.cc
 * \brief Phase 1 capability gate for Tenstorrent tile operations.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>

namespace tvm {
namespace tl {
namespace tenstorrent {

namespace {

class TileOpGate : public tirx::StmtExprVisitor {
public:
  void VisitExpr_(const tirx::CallNode *op) final {
    if (const auto *op_node = op->op.as<OpNode>()) {
      const std::string name = op_node->name;
      if (name.rfind("tl.tileop.", 0) == 0) {
        TVM_FFI_THROW(NotImplementedError)
            << "[LegalizeTenstorrentTileOps] TileOp '" << name
            << "' is deferred to Phase 2; Phase 1 supports only the no-op "
               "Device TIR skeleton";
      }
    }
    tirx::StmtExprVisitor::VisitExpr_(op);
  }
};

tirx::PrimFunc CheckTileOps(tirx::PrimFunc func) {
  TileOpGate gate;
  gate(func->body);
  return func;
}

} // namespace

tvm::transform::Pass LegalizeTenstorrentTileOps() {
  auto pass_func = [](tirx::PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &context) {
    return CheckTileOps(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.LegalizeTenstorrentTileOps", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::GlobalDef().def("tl.tenstorrent.transform.LegalizeTenstorrentTileOps",
                        LegalizeTenstorrentTileOps);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

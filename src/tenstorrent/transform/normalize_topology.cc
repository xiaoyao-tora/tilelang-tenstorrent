/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file tenstorrent/transform/normalize_topology.cc
 * \brief Phase 1 capability gate for Tenstorrent topology operations.
 */

#include "../op/builtin.h"

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

constexpr const char *kForeachSrc = "tl.tt.foreach_src";
constexpr const char *kForeachDst = "tl.tt.foreach_dst";

class TopologyGate : public tirx::StmtExprVisitor {
public:
  void VisitStmt_(const tirx::ForNode *op) final {
    if (op->annotations.count(kForeachSrc) ||
        op->annotations.count(kForeachDst)) {
      TVM_FFI_THROW(NotImplementedError)
          << "[NormalizeTenstorrentTopology] PipeNet foreach regions are "
             "deferred beyond the Phase 1 no-Pipe subset";
    }
    tirx::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const tirx::CallNode *op) final {
    if (const auto *op_node = op->op.as<OpNode>()) {
      const std::string name = op_node->name;
      if (name.rfind("tl.tt.", 0) == 0) {
        TVM_FFI_THROW(NotImplementedError)
            << "[NormalizeTenstorrentTopology] topology operation '" << name
            << "' is deferred beyond the Phase 1 no-Pipe subset";
      }
    }
    tirx::StmtExprVisitor::VisitExpr_(op);
  }
};

tirx::PrimFunc CheckTopology(tirx::PrimFunc func) {
  TopologyGate gate;
  gate(func->body);
  return func;
}

} // namespace

tvm::transform::Pass NormalizeTenstorrentTopology() {
  auto pass_func = [](tirx::PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &context) {
    return CheckTopology(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.NormalizeTenstorrentTopology", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::GlobalDef().def("tl.tenstorrent.transform.NormalizeTenstorrentTopology",
                        NormalizeTenstorrentTopology);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

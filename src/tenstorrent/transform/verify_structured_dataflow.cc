/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*! \file tenstorrent/transform/verify_structured_dataflow.cc
 * \brief Prove definite initialization before returning capture-only IR.
 */

#include "../../op/copy.h"
#include "../../op/fill.h"
#include "../../op/operator.h"
#include "../ir/device_ir.h"
#include "attr.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm::tl::tenstorrent {
using namespace tirx;
namespace {

using Regions = std::unordered_map<Var, std::vector<BufferRegion>,
                                   ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

[[noreturn]] void Fail(const std::string &message) {
  TVM_FFI_THROW(ValueError) << "[VerifyTTStructuredDataflow] " << message;
}

// This verifier deliberately proves only normalized, static dataflow. A
// capability fallback is not permission to accept an unproven program.
class StructuredDataflowVerifier : public StmtExprVisitor {
public:
  static void Verify(const PrimFunc &func) {
    StructuredDataflowVerifier verifier;
    auto slot = func->GetAttr<ffi::String>(kKernelSlotAttr);
    verifier.fragment_slot_allowed_ =
        !slot.has_value() || slot.value() == "trisc";
    for (const auto &[parameter, buffer] : func->buffer_map) {
      verifier.known_.emplace(buffer->data, buffer);
      verifier.Write_(BufferRegion::FullRegion(buffer));
    }
    verifier(func->body);
  }

private:
  bool Covers_(const BufferRegion &written, const BufferRegion &read) {
    if (written->buffer->dtype != read->buffer->dtype ||
        written->region.size() != read->region.size())
      return false;
    for (size_t i = 0; i < read->region.size(); ++i) {
      const Range &a = written->region[i], &b = read->region[i];
      if (!analyzer_.CanProve(a->min <= b->min) ||
          !analyzer_.CanProve(b->min + b->extent <= a->min + a->extent))
        return false;
    }
    return true;
  }

  void CheckRegion_(const BufferRegion &region) {
    CheckFragmentSlot_(region);
    const Buffer &buffer = region->buffer;
    auto it = known_.find(buffer->data);
    if (it == known_.end() || !it->second.same_as(buffer))
      Fail("unknown or aliased Buffer identity '" + std::string(buffer->name) +
           "' cannot be verified for structured return");
    if (region->region.size() != buffer->shape.size())
      Fail("region rank disagrees with Buffer '" + std::string(buffer->name) +
           "'");
    for (size_t i = 0; i < region->region.size(); ++i) {
      const Range &range = region->region[i];
      VisitExpr(range->min);
      VisitExpr(range->extent);
      VisitExpr(buffer->shape[i]);
      if (!analyzer_.CanProve(range->min >= 0) ||
          !analyzer_.CanProve(range->extent > 0) ||
          !analyzer_.CanProve(range->min + range->extent <= buffer->shape[i]))
        Fail("out-of-bounds or unproven region in Buffer '" +
             std::string(buffer->name) + "'");
    }
  }

  void Read_(const BufferRegion &region) {
    CheckRegion_(region);
    auto found = initialized_.find(region->buffer->data);
    if (found != initialized_.end()) {
      for (const BufferRegion &written : found->second)
        if (Covers_(written, region))
          return;
    }
    Fail("read before definite initialization of Buffer '" +
         std::string(region->buffer->name) + "'");
  }

  void Write_(const BufferRegion &region) {
    CheckRegion_(region);
    initialized_[region->buffer->data].push_back(region);
  }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    if (!is_one(op->predicate) || !op->iter_values.empty())
      Fail("nontrivial block realization cannot be verified for structured "
           "return");
    VisitStmt(op->block);
  }

  void VisitStmt_(const SBlockNode *op) final {
    if (!op->match_buffers.empty())
      Fail("match-buffer aliases cannot be verified for structured return");
    for (const Buffer &buffer : op->alloc_buffers) {
      for (const PrimExpr &extent : buffer->shape)
        VisitExpr(extent);
      for (const PrimExpr &stride : buffer->strides)
        VisitExpr(stride);
      VisitExpr(buffer->elem_offset);
      if (known_.count(buffer->data))
        Fail("allocation aliases an existing Buffer");
      known_.emplace(buffer->data, buffer);
      initialized_.erase(buffer->data);
    }
    if (op->init.has_value())
      VisitStmt(op->init.value());
    if (op->annotations.count(kTTComputeKind)) {
      for (const BufferRegion &read : op->reads)
        Read_(read);
      for (const BufferRegion &write : op->writes)
        Write_(write);
    } else {
      VisitStmt(op->body);
    }
    for (const Buffer &buffer : op->alloc_buffers) {
      initialized_.erase(buffer->data);
      known_.erase(buffer->data);
    }
  }

  void VisitStmt_(const ForNode *op) final {
    // Loop bounds execute before the body and may read uninitialized buffers.
    VisitExpr(op->min);
    VisitExpr(op->extent);
    if (op->step.has_value())
      VisitExpr(op->step.value());
    if (!analyzer_.CanProve(op->extent > 0))
      Fail("loop must have a provably positive extent for structured dataflow");
    if (op->kind != ForKind::kSerial)
      Fail("unconsumed nonserial loop at structured boundary");
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    // A first-iteration read must be initialized before entering the loop.
    // Index-dependent writes cannot prove initialization after all iterations;
    // retain only loop-invariant regions at the exit.
    Regions before = initialized_;
    VisitStmt(op->body);
    for (auto &[data, regions] : initialized_) {
      std::vector<BufferRegion> invariant;
      for (const BufferRegion &region : regions) {
        bool varies = false;
        for (const Range &range : region->region) {
          PostOrderVisit(range->min, [&](const ObjectRef &node) {
            varies |= node.same_as(op->loop_var);
          });
          PostOrderVisit(range->extent, [&](const ObjectRef &node) {
            varies |= node.same_as(op->loop_var);
          });
        }
        if (!varies)
          invariant.push_back(region);
      }
      regions = std::move(invariant);
    }
    for (const auto &[data, regions] : before)
      for (const BufferRegion &region : regions)
        initialized_[data].push_back(region);
  }

  void VisitStmt_(const WhileNode *) final {
    // A possibly zero-trip body cannot establish definite initialization.
    // Keep the standalone verifier within the same static subset as the
    // frontend validator instead of inheriting one-pass visitor semantics.
    Fail("while loops cannot be verified for structured dataflow; only "
         "provably positive serial loops are supported");
  }

  void CheckFragmentSlot_(const BufferRegion &region) {
    if (region->buffer.scope() == "local.fragment" && !fragment_slot_allowed_)
      Fail("fragment cannot cross processor slots; materialize through an "
           "output DFB");
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    if (SideEffect(op->condition) > CallEffectKind::kReadState)
      Fail("effectful condition at structured boundary");
    VisitExpr(op->condition);
    if (analyzer_.CanProve(op->condition)) {
      VisitStmt(op->then_case);
      return;
    }
    if (analyzer_.CanProve(!op->condition)) {
      if (op->else_case.has_value())
        VisitStmt(op->else_case.value());
      return;
    }
    Regions before = initialized_;
    VisitStmt(op->then_case);
    Regions then_state = initialized_;
    initialized_ = before;
    if (op->else_case.has_value())
      VisitStmt(op->else_case.value());
    Regions common;
    for (const auto &[data, regions] : then_state) {
      auto other = initialized_.find(data);
      if (other == initialized_.end())
        continue;
      for (const BufferRegion &region : regions)
        for (const BufferRegion &candidate : other->second)
          if (Covers_(candidate, region)) {
            common[data].push_back(region);
            break;
          }
    }
    initialized_ = std::move(common);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (is_zero(op->value))
      return;
    const auto *node = op->value.as<CallNode>();
    const auto *operator_node = node ? node->op.as<OpNode>() : nullptr;
    if (!operator_node)
      Fail("unknown effect at structured boundary");
    const std::string name = operator_node->name;
    if (name != "tl.tileop.copy" && name != "tl.tileop.fill" &&
        name != "tl.tileop.gemm" && name != "tl.tileop.transpose" &&
        name != "tl.tileop.reduce")
      Fail("unverified operation '" + name + "' at structured boundary");
    Call call = ffi::GetRef<Call>(node);
    if (call->op.same_as(Copy::Get())) {
      Copy copy = Downcast<Copy>(ParseOperator(call));
      if (copy->src->dtype != copy->dst->dtype &&
          copy->src.scope() != "local.fragment")
        Fail("ordinary copy source and destination dtypes disagree");
    }
    if (call->op.same_as(Fill::Get())) {
      Fill fill = Downcast<Fill>(ParseOperator(call));
      if (SideEffect(fill->value) > CallEffectKind::kReadState)
        Fail("effectful fill expression at structured boundary");
      VisitExpr(fill->value);
    }
    AccessRegions accesses = ParseOperator(call)->GetAccessRegions();
    for (const BufferRegion &read : accesses.reads)
      Read_(read);
    for (const BufferRegion &write : accesses.writes)
      Write_(write);
  }

  void VisitExpr_(const CallNode *op) final {
    // Evaluate handles the supported TileOps explicitly. Other expression
    // contexts (Bind, Assert, attributes, bounds and indices) must not hide
    // an opaque operation that bypasses that whitelist.
    if (SideEffect(ffi::GetRef<Call>(op)) > CallEffectKind::kReadState)
      Fail("unverified effectful expression at structured boundary");
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    // The upstream visitor traverses indices but omits the load predicate.
    if (op->predicate.has_value())
      VisitExpr(op->predicate.value());
    ffi::Array<Range> ranges;
    for (const PrimExpr &index : op->indices)
      ranges.push_back(Range::FromMinExtent(index, 1));
    Read_(BufferRegion(op->buffer, ranges));
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Fail("uncaptured BufferStore at structured boundary");
  }

  arith::Analyzer analyzer_;
  bool fragment_slot_allowed_{true};
  Regions initialized_;
  std::unordered_map<Var, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      known_;
};

} // namespace

tvm::transform::Pass VerifyTTStructuredDataflow() {
  auto verify = [](PrimFunc func, const IRModule &,
                   const tvm::transform::PassContext &) {
    StructuredDataflowVerifier::Verify(func);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      verify, 0, "tl.tenstorrent.VerifyTTStructuredDataflow", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.tenstorrent.transform.VerifyTTStructuredDataflow",
      VerifyTTStructuredDataflow);
}

} // namespace tvm::tl::tenstorrent

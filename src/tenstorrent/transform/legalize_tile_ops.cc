/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file tenstorrent/transform/legalize_tile_ops.cc
 * \brief Consume structured elementwise operations supported by Device TIR.
 */

#include "../../op/copy.h"
#include "../../op/region.h"
#include "../ir/device_ir.h"
#include "../op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <utility>

namespace tvm {
namespace tl {
namespace tenstorrent {

using namespace tirx;

// The verifier owns the structured-operation contract. Device instruction
// selection must not reconstruct that contract from zero-coordinate stores.
tvm::transform::Pass VerifyTTComputeBlocks();

namespace {

bool IsElementwiseBlock(const SBlock &block) {
  auto kind = block->annotations.Get("tl.tt.compute_kind");
  return kind.has_value() && kind.value().as<ffi::String>().has_value() &&
         Downcast<ffi::String>(kind.value()) == "elementwise";
}

bool IsDeviceAdd(const SBlock &block,
                 const ffi::Array<TTBufferMetadata> &metadata) {
  if (!IsElementwiseBlock(block)) {
    return false;
  }
  const auto *store = block->body.as<BufferStoreNode>();
  const auto *add = store == nullptr ? nullptr : store->value.as<AddNode>();
  if (add == nullptr || !add->a.as<BufferLoadNode>() ||
      !add->b.as<BufferLoadNode>()) {
    return false;
  }
  const Buffer &output = store->buffer;
  const Buffer &lhs = add->a.as<BufferLoadNode>()->buffer;
  const Buffer &rhs = add->b.as<BufferLoadNode>()->buffer;
  if (lhs.same_as(rhs) || lhs.same_as(output) || rhs.same_as(output)) {
    return false;
  }
  // The implemented Device TIR transaction protocol currently requires two
  // blocks per DFB. Other capacities remain valid structured computations.
  for (const Buffer &buffer : {lhs, rhs, output}) {
    bool supported_capacity = false;
    for (const TTBufferMetadata &item : metadata) {
      if (item->buffer.same_as(buffer) && item->dfb_block_count.has_value() &&
          is_const_int(item->dfb_block_count.value(), 2)) {
        supported_capacity = true;
        break;
      }
    }
    if (!supported_capacity)
      return false;
  }
  const DataType dtype = output->dtype;
  if (dtype.lanes() != 1 ||
      !(dtype.is_bfloat16() || (dtype.is_float() && dtype.bits() == 32))) {
    return false;
  }
  auto maps = Downcast<ffi::Map<Var, ffi::Array<Integer>>>(
      block->annotations.at("tl.tt.access_maps"));
  for (const BufferRegion &region : block->reads) {
    if (region->buffer->dtype != dtype) {
      return false;
    }
  }
  for (const auto &entry : maps) {
    if (entry.second.size() != 2 || entry.second[0]->value != 0 ||
        entry.second[1]->value != 1) {
      return false;
    }
  }
  auto shape = Downcast<ffi::Array<PrimExpr>>(
      block->annotations.at("tl.tt.logical_domain"));
  return shape.size() == 2 && is_const_int(shape[0], 32) &&
         is_const_int(shape[1], 32);
}

// Preserve every structured operation in a function when any operation still
// needs future device lowering. This avoids returning a partly consumed DAG.
class ComputeCapabilityVisitor : public StmtExprVisitor {
public:
  explicit ComputeCapabilityVisitor(const PrimFunc &func)
      : metadata_(func->GetAttr<ffi::Array<TTBufferMetadata>>(
                          kBufferMetadataTableAttr)
                      .value_or(ffi::Array<TTBufferMetadata>{})) {}

  bool supported{true};

private:
  void VisitStmt_(const SBlockNode *op) final {
    SBlock block = ffi::GetRef<SBlock>(op);
    if (IsElementwiseBlock(block)) {
      supported =
          supported && ++compute_count_ == 1 && IsDeviceAdd(block, metadata_);
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  size_t compute_count_{0};
  ffi::Array<TTBufferMetadata> metadata_;
};

class StructuredComputeLowerer : public StmtExprMutator {
public:
  explicit StructuredComputeLowerer(bool consume_blocks)
      : consume_blocks_(consume_blocks) {}

private:
  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    const SBlock &block = op->block;
    if (!IsElementwiseBlock(block)) {
      return StmtExprMutator::VisitStmt_(op);
    }
    if (!consume_blocks_) {
      return ffi::GetRef<SBlockRealize>(op);
    }
    const auto *store = block->body.as<BufferStoreNode>();
    const auto *add = store->value.as<AddNode>();
    const Buffer &lhs = add->a.as<BufferLoadNode>()->buffer;
    const Buffer &rhs = add->b.as<BufferLoadNode>()->buffer;
    const Buffer &output = store->buffer;
    ffi::Map<ffi::String, ffi::ObjectRef> annotations = {
        {"tt.compute_dtype",
         StringImm(output->dtype.is_bfloat16() ? "bfloat16" : "float32")},
        {"tt.compute_tile_shape",
         Downcast<ffi::Array<PrimExpr>>(
             block->annotations.at("tl.tt.physical_tile_shape"))},
    };
    Span span = block->span.defined() ? block->span : op->span;
    Call call(DataType::Handle(), tile_add(),
              {BufferRegion::FullRegion(lhs)->ToPrimExpr(),
               BufferRegion::FullRegion(rhs)->ToPrimExpr(),
               BufferRegion::FullRegion(output)->ToPrimExpr()},
              std::move(annotations), span);
    return Evaluate(std::move(call), span);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      TVM_FFI_THROW(ValueError)
          << "[LegalizeTenstorrentTileOps] T.Parallel must first pass through "
             "CanonicalizeTTElementwise";
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    TVM_FFI_THROW(NotImplementedError)
        << "[LegalizeTenstorrentTileOps] BufferStore compute requires a "
           "structured T.Parallel or T.Tiles operation";
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (const auto *named = op->op.as<OpNode>()) {
      const std::string name = named->name;
      if (name.rfind("tl.tileop.", 0) == 0 && !op->op.same_as(Copy::Get()) &&
          !op->op.same_as(RegionOp::Get())) {
        TVM_FFI_THROW(NotImplementedError)
            << "[LegalizeTenstorrentTileOps] TileOp '" << name
            << "' has no structured Device TIR lowering";
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  bool consume_blocks_;
};

} // namespace

tvm::transform::Pass LegalizeTenstorrentTileOps() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &context) {
    ComputeCapabilityVisitor capability(func);
    capability(func->body);
    StructuredComputeLowerer lowerer(capability.supported);
    Stmt body = lowerer(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  };
  return tvm::transform::Sequential(
      {VerifyTTComputeBlocks(),
       tirx::transform::CreatePrimFuncPass(
           pass_func, 0, "tl.tenstorrent.LegalizeTenstorrentTileOps", {})},
      "tl.tenstorrent.LegalizeTenstorrentTileOps");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::GlobalDef().def("tl.tenstorrent.transform.LegalizeTenstorrentTileOps",
                        LegalizeTenstorrentTileOps);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

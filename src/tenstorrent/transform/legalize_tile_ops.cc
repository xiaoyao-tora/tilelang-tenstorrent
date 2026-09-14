/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file tenstorrent/transform/legalize_tile_ops.cc
 * \brief Legalize the initial Tenstorrent tiled elementwise operation set.
 */

#include "../../op/copy.h"
#include "../../op/utils.h"
#include "../ir/device_ir.h"
#include "../op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace tl {
namespace tenstorrent {

using namespace tirx;

namespace {

constexpr const char *kComputeDType = "tt.compute_dtype";
constexpr const char *kComputeTileShape = "tt.compute_tile_shape";
constexpr const char *kTransferKind = "tt.transfer_kind";

using BufferMetadataMap =
    std::unordered_map<Buffer, TTBufferMetadata, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;
using BufferSet =
    std::unordered_set<Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

[[noreturn]] void ThrowMalformed(const std::string &message) {
  TVM_FFI_THROW(ValueError)
      << "[LegalizeTenstorrentTileOps] malformed input: " << message;
}

[[noreturn]] void ThrowUnsupported(const std::string &message) {
  TVM_FFI_THROW(NotImplementedError)
      << "[LegalizeTenstorrentTileOps] unsupported operation: " << message;
}

void RequireStaticValue(const PrimExpr &value, int64_t expected,
                        const std::string &field) {
  const int64_t *integer = as_const_int(value);
  if (integer == nullptr || *integer != expected) {
    ThrowUnsupported(field + " must equal " + std::to_string(expected));
  }
}

bool IsSupportedDType(DataType dtype) {
  return dtype.lanes() == 1 &&
         (dtype.is_bfloat16() ||
          (dtype.is_float() && dtype.bits() == 32));
}

ffi::String DTypeName(DataType dtype) {
  if (dtype.is_bfloat16()) {
    return "bfloat16";
  }
  if (dtype.is_float() && dtype.bits() == 32 && dtype.lanes() == 1) {
    return "float32";
  }
  ThrowUnsupported("Add supports only scalar bfloat16 and float32 buffers");
}

bool IsSharedBuffer(const Buffer &buffer) {
  const std::string scope = buffer.scope();
  return scope == "shared" || scope == "shared.dyn";
}

bool IsGlobalBuffer(const Buffer &buffer) {
  const std::string scope = buffer.scope();
  return scope.empty() || scope == "global";
}

void RequireBufferShape(const Buffer &buffer, const std::string &role) {
  if (buffer->shape.size() != 2) {
    ThrowUnsupported(role + " buffer '" + std::string(buffer->name) +
                     "' must have rank two");
  }
  RequireStaticValue(buffer->shape[0], 32, role + " shape axis 0");
  RequireStaticValue(buffer->shape[1], 32, role + " shape axis 1");
  if (!IsSupportedDType(buffer->dtype)) {
    ThrowUnsupported(role + " buffer '" + std::string(buffer->name) +
                     "' must use bfloat16 or float32");
  }
}

void RequireFullRegion(const BufferRegion &region, const std::string &role) {
  const Buffer &buffer = region->buffer;
  RequireBufferShape(buffer, role);
  if (region->region.size() != 2) {
    ThrowMalformed(role + " region rank differs from its buffer rank");
  }
  for (size_t axis = 0; axis < 2; ++axis) {
    RequireStaticValue(region->region[axis]->min, 0,
                       role + " region minimum axis " +
                           std::to_string(axis));
    RequireStaticValue(region->region[axis]->extent, 32,
                       role + " region extent axis " +
                           std::to_string(axis));
  }
}

Call RequireCall(const Stmt &stmt, const std::string &role) {
  const auto *evaluate = stmt.as<EvaluateNode>();
  if (evaluate == nullptr) {
    ThrowUnsupported(role + " must be an Evaluate(Call)");
  }
  const auto *call = evaluate->value.as<CallNode>();
  if (call == nullptr) {
    ThrowUnsupported(role + " must be an Evaluate(Call)");
  }
  return ffi::GetRef<Call>(call);
}

struct CopyRecord {
  BufferRegion source;
  BufferRegion destination;
};

CopyRecord ParseCopy(const Stmt &stmt, const ffi::String &expected_kind,
                     const std::string &role) {
  Call call = RequireCall(stmt, role);
  if (!call->op.same_as(Copy::Get())) {
    ThrowUnsupported(role + " must be T.copy");
  }
  Copy copy = Downcast<Copy>(ParseOperator(call));
  if (!copy.defined()) {
    ThrowMalformed(role + " could not be parsed as T.copy");
  }
  auto transfer = call->annotations.Get(kTransferKind);
  if (!transfer.has_value()) {
    ThrowMalformed(role + " has no normalized 'tt.transfer_kind'");
  }
  const auto *kind = transfer.value().as<StringImmNode>();
  if (kind == nullptr || kind->value != expected_kind) {
    ThrowMalformed(role +
                   " has a transfer kind inconsistent with Add dataflow");
  }
  BufferRegion source(copy->src, copy->src_range);
  BufferRegion destination(copy->dst, copy->dst_range);
  RequireFullRegion(source, role + " source");
  RequireFullRegion(destination, role + " destination");
  return {std::move(source), std::move(destination)};
}

void RequireSameBuffer(const Buffer &actual, const Buffer &expected,
                       const std::string &message) {
  if (!actual.same_as(expected)) {
    ThrowUnsupported(message);
  }
}

void RequireIndex(const PrimExpr &index, const Var &expected,
                  const std::string &role) {
  if (!index.same_as(expected)) {
    ThrowUnsupported(role +
                     " must use the corresponding Parallel loop variable");
  }
}

void RequireParallelLoop(const For &loop, const std::string &role) {
  if (loop->kind != ForKind::kParallel) {
    ThrowUnsupported(role + " must be a T.Parallel loop");
  }
  RequireStaticValue(loop->min, 0, role + " minimum");
  RequireStaticValue(loop->extent, 32, role + " extent");
  if (!loop->HasTrivialStep() || loop->thread_binding.has_value() ||
      !loop->annotations.empty()) {
    ThrowUnsupported(
        role +
        " cannot carry a non-unit step, thread binding, or annotations");
  }
}

struct AddBuffers {
  Buffer lhs;
  Buffer rhs;
  Buffer output;
  Span span;
};

AddBuffers ParseScalarAdd(const Stmt &stmt) {
  const auto *outer_node = stmt.as<ForNode>();
  if (outer_node == nullptr) {
    ThrowUnsupported(
        "Add compute must start with a two-dimensional T.Parallel loop");
  }
  For outer = ffi::GetRef<For>(outer_node);
  RequireParallelLoop(outer, "Add outer Parallel loop");

  const auto *inner_node = outer->body.as<ForNode>();
  if (inner_node == nullptr) {
    ThrowUnsupported(
        "Add compute must contain exactly two nested Parallel loops");
  }
  For inner = ffi::GetRef<For>(inner_node);
  RequireParallelLoop(inner, "Add inner Parallel loop");

  const auto *store = inner->body.as<BufferStoreNode>();
  if (store == nullptr) {
    ThrowUnsupported("Add Parallel body must contain one BufferStore");
  }
  if (store->predicate.has_value()) {
    ThrowUnsupported("predicated Add stores are not implemented");
  }
  if (store->indices.size() != 2) {
    ThrowUnsupported("Add output access must have rank two");
  }
  RequireIndex(store->indices[0], outer->loop_var, "Add output axis 0");
  RequireIndex(store->indices[1], inner->loop_var, "Add output axis 1");

  const auto *add = store->value.as<AddNode>();
  if (add == nullptr) {
    ThrowUnsupported("Parallel compute must be a direct binary Add expression");
  }
  const auto *lhs_load = add->a.as<BufferLoadNode>();
  const auto *rhs_load = add->b.as<BufferLoadNode>();
  if (lhs_load == nullptr || rhs_load == nullptr) {
    ThrowUnsupported("Add operands must be direct BufferLoad expressions");
  }
  for (const auto *load : {lhs_load, rhs_load}) {
    if (load->predicate.has_value() || load->indices.size() != 2) {
      ThrowUnsupported(
          "Add input accesses must be unpredicated rank-two loads");
    }
    RequireIndex(load->indices[0], outer->loop_var, "Add input axis 0");
    RequireIndex(load->indices[1], inner->loop_var, "Add input axis 1");
  }

  AddBuffers result{lhs_load->buffer, rhs_load->buffer, store->buffer,
                    outer->span};
  if (!IsSharedBuffer(result.lhs)) {
    ThrowUnsupported("left Add input buffer must use shared memory");
  }
  if (!IsSharedBuffer(result.rhs)) {
    ThrowUnsupported("right Add input buffer must use shared memory");
  }
  if (!IsSharedBuffer(result.output)) {
    ThrowUnsupported("Add output buffer must use shared memory");
  }
  RequireBufferShape(result.lhs, "left Add input");
  RequireBufferShape(result.rhs, "right Add input");
  RequireBufferShape(result.output, "Add output");
  if (result.lhs->dtype != result.rhs->dtype ||
      result.lhs->dtype != result.output->dtype ||
      store->value.dtype() != result.output->dtype) {
    ThrowUnsupported(
        "Add inputs and output must have one identical dtype without promotion");
  }
  return result;
}

AddBuffers ParseCanonicalAdd(const Stmt &stmt) {
  Call call = RequireCall(stmt, "canonical Add");
  if (!call->op.same_as(tenstorrent::tile_add())) {
    ThrowUnsupported(
        "compute statement is not the canonical tl.tt.tile_add op");
  }
  if (call.dtype() != DataType::Handle() || call->args.size() != 3) {
    ThrowMalformed(
        "tl.tt.tile_add must return handle and have three region arguments");
  }
  AccessRegion lhs = NormalizeToAccessRegion(call->args[0], kAccessRead);
  AccessRegion rhs = NormalizeToAccessRegion(call->args[1], kAccessRead);
  AccessRegion output = NormalizeToAccessRegion(call->args[2], kAccessWrite);
  RequireFullRegion(lhs.region, "canonical left Add input");
  RequireFullRegion(rhs.region, "canonical right Add input");
  RequireFullRegion(output.region, "canonical Add output");

  if (call->annotations.size() != 2) {
    ThrowMalformed(
        "tl.tt.tile_add must carry exactly dtype and tile-shape annotations");
  }
  auto dtype_annotation = call->annotations.Get(kComputeDType);
  const auto *dtype = dtype_annotation.has_value()
                          ? dtype_annotation.value().as<StringImmNode>()
                          : nullptr;
  if (dtype == nullptr ||
      dtype->value != DTypeName(output.region->buffer->dtype)) {
    ThrowMalformed(
        "tl.tt.tile_add has missing or conflicting 'tt.compute_dtype'");
  }
  auto tile_annotation = call->annotations.Get(kComputeTileShape);
  if (!tile_annotation.has_value()) {
    ThrowMalformed("tl.tt.tile_add has no 'tt.compute_tile_shape'");
  }
  auto tile_shape = tile_annotation.value().as<ffi::Array<PrimExpr>>();
  if (!tile_shape.has_value() || tile_shape.value().size() != 2) {
    ThrowMalformed("tl.tt.tile_add has malformed 'tt.compute_tile_shape'");
  }
  RequireStaticValue(tile_shape.value()[0], 32,
                     "canonical Add tile axis 0");
  RequireStaticValue(tile_shape.value()[1], 32,
                     "canonical Add tile axis 1");

  AddBuffers result{lhs.region->buffer, rhs.region->buffer,
                    output.region->buffer, call->span};
  if (result.lhs->dtype != result.rhs->dtype ||
      result.lhs->dtype != result.output->dtype) {
    ThrowMalformed("tl.tt.tile_add region dtypes disagree");
  }
  return result;
}

BufferMetadataMap BuildMetadataMap(const PrimFunc &func) {
  auto table =
      func->GetAttr<ffi::Array<TTBufferMetadata>>(kBufferMetadataTableAttr);
  if (!table.has_value()) {
    ThrowMalformed("function-level tt.buffer_metadata_table is missing");
  }
  BufferMetadataMap result;
  for (const TTBufferMetadata &metadata : table.value()) {
    if (!metadata.defined() || !metadata->buffer.defined()) {
      ThrowMalformed("tt.buffer_metadata_table contains an undefined entry");
    }
    if (!result.emplace(metadata->buffer, metadata).second) {
      ThrowMalformed("tt.buffer_metadata_table repeats a Buffer identity");
    }
  }
  return result;
}

const TTBufferMetadata &RequireMetadata(const BufferMetadataMap &metadata,
                                        const Buffer &buffer,
                                        const std::string &role) {
  auto it = metadata.find(buffer);
  if (it == metadata.end()) {
    ThrowMalformed(role + " buffer '" + std::string(buffer->name) +
                   "' has no TTBufferMetadata");
  }
  return it->second;
}

void RequireShapeMetadata(const TTBufferMetadata &metadata,
                          const std::string &role) {
  if (metadata->tile_shape.size() != 2 ||
      metadata->tile_grid_shape.size() != 2) {
    ThrowMalformed(role + " has incomplete inferred tile metadata");
  }
  RequireStaticValue(metadata->tile_shape[0], 32, role + " tile axis 0");
  RequireStaticValue(metadata->tile_shape[1], 32, role + " tile axis 1");
  RequireStaticValue(metadata->tile_grid_shape[0], 1,
                     role + " tile-grid axis 0");
  RequireStaticValue(metadata->tile_grid_shape[1], 1,
                     role + " tile-grid axis 1");
  if (metadata->memory_layout != "interleaved" ||
      metadata->shard_spec.has_value()) {
    ThrowUnsupported(role + " must use an unsharded interleaved layout");
  }
  if (metadata->alias_of.has_value()) {
    ThrowUnsupported(role + " aliases buffer '" +
                     std::string(metadata->alias_of.value()) + "'");
  }
}

Buffer RequireParameterBuffer(const PrimFunc &func, size_t index) {
  if (index >= func->params.size()) {
    ThrowUnsupported("Add requires exactly three Tensor ABI parameters");
  }
  auto it = func->buffer_map.find(func->params[index]);
  if (it == func->buffer_map.end()) {
    ThrowMalformed("Add ABI parameter " + std::to_string(index) +
                   " has no Buffer binding");
  }
  return (*it).second;
}

void ValidateMetadata(const PrimFunc &func, const SBlock &block,
                      const BufferMetadataMap &metadata,
                      const CopyRecord &read_lhs, const CopyRecord &read_rhs,
                      const AddBuffers &add,
                      const CopyRecord &write_output) {
  if (func->params.size() != 3 || func->buffer_map.size() != 3) {
    ThrowUnsupported("Add requires exactly A, B, and C Tensor ABI parameters");
  }
  if (block->alloc_buffers.size() != 3) {
    ThrowUnsupported("Add requires exactly three shared DFB allocations");
  }
  if (metadata.size() != 6) {
    ThrowUnsupported(
        "Add requires exactly three Tensor and three DFB metadata entries");
  }

  const Buffer tensor_a = RequireParameterBuffer(func, 0);
  const Buffer tensor_b = RequireParameterBuffer(func, 1);
  const Buffer tensor_c = RequireParameterBuffer(func, 2);
  RequireSameBuffer(read_lhs.source->buffer, tensor_a,
                    "first Add read must originate from ABI Tensor A");
  RequireSameBuffer(read_rhs.source->buffer, tensor_b,
                    "second Add read must originate from ABI Tensor B");
  RequireSameBuffer(write_output.destination->buffer, tensor_c,
                    "Add write must target ABI Tensor C");
  RequireSameBuffer(read_lhs.destination->buffer, add.lhs,
                    "first Add copy must populate its left shared input");
  RequireSameBuffer(read_rhs.destination->buffer, add.rhs,
                    "second Add copy must populate its right shared input");
  RequireSameBuffer(write_output.source->buffer, add.output,
                    "Add output copy must consume its shared result");

  RequireSameBuffer(block->alloc_buffers[0], add.lhs,
                    "first shared allocation must be the left Add input");
  RequireSameBuffer(block->alloc_buffers[1], add.rhs,
                    "second shared allocation must be the right Add input");
  RequireSameBuffer(block->alloc_buffers[2], add.output,
                    "third shared allocation must be the Add output");

  BufferSet identities;
  VarSet data_identities;
  for (const Buffer &buffer :
       {tensor_a, tensor_b, tensor_c, add.lhs, add.rhs, add.output}) {
    if (!identities.insert(buffer).second ||
        !data_identities.insert(buffer->data).second) {
      ThrowUnsupported("Add Tensor and DFB buffers must not alias each other");
    }
    RequireBufferShape(buffer, "Add dataflow");
  }
  if (!IsGlobalBuffer(tensor_a) || !IsGlobalBuffer(tensor_b) ||
      !IsGlobalBuffer(tensor_c)) {
    ThrowMalformed("Add ABI Tensor buffers must use global memory");
  }

  for (size_t index = 0; index < 3; ++index) {
    const Buffer &buffer =
        index == 0 ? tensor_a : index == 1 ? tensor_b : tensor_c;
    const TTBufferMetadata &item =
        RequireMetadata(metadata, buffer, "Add Tensor metadata");
    if (item->kind != "tensor" || !item->global_arg_index.has_value() ||
        item->global_arg_index.value()->value !=
            static_cast<int64_t>(index)) {
      ThrowMalformed("Add Tensor metadata is not in stable ABI order");
    }
    RequireShapeMetadata(item, "Add Tensor metadata");
  }
  for (const Buffer &buffer : {add.lhs, add.rhs, add.output}) {
    const TTBufferMetadata &item =
        RequireMetadata(metadata, buffer, "Add DFB metadata");
    if (item->kind != "logical_dfb_candidate" ||
        !item->dfb_block_count.has_value()) {
      ThrowMalformed("Add shared buffers require logical DFB metadata");
    }
    RequireStaticValue(item->dfb_block_count.value(), 2,
                       "Add DFB block count");
    RequireShapeMetadata(item, "Add DFB metadata");
  }
}

Stmt MakeCanonicalAdd(const AddBuffers &add) {
  ffi::Map<ffi::String, ffi::ObjectRef> annotations = {
      {kComputeDType, StringImm(DTypeName(add.output->dtype))},
      {kComputeTileShape,
       ffi::Array<PrimExpr>{IntImm(DataType::Int(32), 32),
                            IntImm(DataType::Int(32), 32)}},
  };
  Call call(DataType::Handle(), tenstorrent::tile_add(),
            {BufferRegion::FullRegion(add.lhs)->ToPrimExpr(),
             BufferRegion::FullRegion(add.rhs)->ToPrimExpr(),
             BufferRegion::FullRegion(add.output)->ToPrimExpr()},
            std::move(annotations), add.span);
  return Evaluate(std::move(call), add.span);
}

bool IsCopyStmt(const Stmt &stmt) {
  const auto *evaluate = stmt.as<EvaluateNode>();
  const auto *call =
      evaluate == nullptr ? nullptr : evaluate->value.as<CallNode>();
  return call != nullptr && call->op.same_as(Copy::Get());
}

bool IsCanonicalAddStmt(const Stmt &stmt) {
  const auto *evaluate = stmt.as<EvaluateNode>();
  const auto *call =
      evaluate == nullptr ? nullptr : evaluate->value.as<CallNode>();
  return call != nullptr && call->op.same_as(tenstorrent::tile_add());
}

class AddLegalizer : public StmtExprMutator {
public:
  AddLegalizer(const PrimFunc &func, const BufferMetadataMap &metadata)
      : func_(func), metadata_(metadata) {}

  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    SBlock block = op->block;
    const auto *sequence = block->body.as<SeqStmtNode>();
    if (sequence != nullptr && sequence->seq.size() == 4 &&
        IsCopyStmt(sequence->seq[0]) && IsCopyStmt(sequence->seq[1]) &&
        IsCopyStmt(sequence->seq[3]) &&
        (sequence->seq[2].as<ForNode>() ||
         IsCanonicalAddStmt(sequence->seq[2]))) {
      if (!op->iter_values.empty() || !is_one(op->predicate) ||
          !block->iter_vars.empty() || !block->reads.empty() ||
          !block->writes.empty() || !block->match_buffers.empty() ||
          block->init.has_value()) {
        ThrowUnsupported("frozen Add must use one unconditional root SBlock");
      }
      CopyRecord read_lhs =
          ParseCopy(sequence->seq[0], "tensor_to_dfb", "left Add input copy");
      CopyRecord read_rhs = ParseCopy(sequence->seq[1], "tensor_to_dfb",
                                      "right Add input copy");
      const bool already_canonical = IsCanonicalAddStmt(sequence->seq[2]);
      AddBuffers add = already_canonical ? ParseCanonicalAdd(sequence->seq[2])
                                         : ParseScalarAdd(sequence->seq[2]);
      CopyRecord write_output = ParseCopy(
          sequence->seq[3], "dfb_to_tensor", "Add output copy");
      ValidateMetadata(func_, block, metadata_, read_lhs, read_rhs, add,
                       write_output);
      if (++add_count_ != 1) {
        ThrowUnsupported("one PrimFunc may contain only one Add compute region");
      }
      if (already_canonical) {
        return ffi::GetRef<Stmt>(op);
      }

      ffi::Array<Stmt> body = sequence->seq;
      body.Set(2, MakeCanonicalAdd(add));
      block.CopyOnWrite()->body = SeqStmt(std::move(body), sequence->span);
      SBlockRealize realize = ffi::GetRef<SBlockRealize>(op);
      realize.CopyOnWrite()->block = std::move(block);
      return realize;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      ThrowUnsupported(
          "T.Parallel is supported only by the frozen 32x32 Add pattern");
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    ThrowUnsupported(
        "BufferStore compute is supported only by the frozen 32x32 Add pattern");
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (const auto *op_node = op->op.as<OpNode>()) {
      const std::string name = op_node->name;
      if (name.rfind("tl.tileop.", 0) == 0) {
        ThrowUnsupported("TileOp '" + name +
                         "' is outside the frozen 32x32 Add pattern");
      }
      if (op->op.same_as(tenstorrent::tile_add())) {
        ThrowMalformed(
            "tl.tt.tile_add appears outside its canonical Add dataflow region");
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

private:
  PrimFunc func_;
  const BufferMetadataMap &metadata_;
  int add_count_{0};
};

PrimFunc LegalizeTileOps(PrimFunc func) {
  BufferMetadataMap metadata = BuildMetadataMap(func);
  AddLegalizer legalizer(func, metadata);
  Stmt body = legalizer(func->body);
  if (body.same_as(func->body)) {
    return func;
  }
  func.CopyOnWrite()->body = std::move(body);
  return func;
}

} // namespace

tvm::transform::Pass LegalizeTenstorrentTileOps() {
  auto pass_func = [](tirx::PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &context) {
    return LegalizeTileOps(std::move(func));
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

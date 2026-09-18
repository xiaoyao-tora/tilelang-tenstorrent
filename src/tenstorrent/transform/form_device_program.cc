/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file tenstorrent/transform/form_device_program.cc
 * \brief Atomically form Tenstorrent Device TIR programs.
 */

#include "../../op/copy.h"
#include "../../op/utils.h"
#include "../ir/device_ir.h"
#include "../op/builtin.h"
#include "verify_device_ir.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace tenstorrent {

using namespace tirx;

namespace {

constexpr int64_t kDeviceIRVersion = 1;
constexpr const char *kLogicalCoreAxis = "tt.logical_core_axis";
constexpr const char *kTransferKind = "tt.transfer_kind";
constexpr const char *kComputeDType = "tt.compute_dtype";
constexpr const char *kComputeTileShape = "tt.compute_tile_shape";

using BufferMetadataMap =
    std::unordered_map<Buffer, TTBufferMetadata, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;

struct Region2D {
  PrimExpr row_start;
  PrimExpr col_start;
  PrimExpr rows;
  PrimExpr cols;
};

struct AddDataflowPlan {
  std::array<TTBufferMetadata, 3> tensors;
  std::array<TTBufferMetadata, 3> dfbs;
  std::array<Region2D, 3> tensor_regions;
  Span compute_span;
};

[[noreturn]] void ThrowMalformed(const std::string &message) {
  TVM_FFI_THROW(ValueError)
      << "[FormTenstorrentDeviceProgram] malformed input: " << message;
}

[[noreturn]] void ThrowUnsupported(const std::string &message) {
  TVM_FFI_THROW(NotImplementedError)
      << "[FormTenstorrentDeviceProgram] unsupported program: " << message;
}

int64_t RequirePositiveStaticInteger(const PrimExpr &value,
                                     const std::string &field) {
  const int64_t *integer = as_const_int(value);
  if (integer == nullptr) {
    ThrowUnsupported(field + " must be a compile-time integer");
  }
  if (*integer <= 0) {
    ThrowMalformed(field + " must be positive");
  }
  return *integer;
}

Span RequireSourceSpan(const Span &preferred, const Span &fallback,
                       const std::string &owner) {
  if (preferred.defined()) {
    return preferred;
  }
  if (fallback.defined()) {
    return fallback;
  }
  ThrowMalformed(owner + " has no source span");
}

Stmt StripLogicalCoreLoops(const PrimFunc &func,
                           const ffi::Array<PrimExpr> &launch_grid) {
  Stmt current = func->body;
  for (size_t index = 0; index < 2; ++index) {
    const auto *loop_node = current.as<ForNode>();
    if (loop_node == nullptr || loop_node->kind != ForKind::kSerial) {
      ThrowMalformed("normalized launch must contain serial logical Core x/y "
                     "loops");
    }
    For loop = ffi::GetRef<For>(loop_node);
    ffi::Optional<ffi::Any> axis_value =
        loop->annotations.Get(kLogicalCoreAxis);
    std::optional<ffi::String> axis = axis_value.has_value()
                                          ? axis_value.value().as<ffi::String>()
                                          : std::optional<ffi::String>();
    const ffi::String expected(index == 0 ? "x" : "y");
    if (!axis.has_value() || axis.value() != expected) {
      ThrowMalformed("normalized logical Core axes must be ordered x then y");
    }
    const int64_t *minimum = as_const_int(loop->min);
    if (minimum == nullptr || *minimum != 0) {
      ThrowMalformed("normalized logical Core loops must start at zero");
    }
    if (!ffi::StructuralEqual()(loop->extent, launch_grid[index])) {
      ThrowMalformed("logical Core loop extent disagrees with tt.launch_grid");
    }
    current = loop->body;
  }
  return current;
}

void RequireNoOpBody(const Stmt &stmt) {
  if (const auto *evaluate = stmt.as<EvaluateNode>()) {
    const int64_t *value = as_const_int(evaluate->value);
    if (value == nullptr || *value != 0) {
      ThrowUnsupported("Evaluate contains a non-no-op expression");
    }
    return;
  }
  if (const auto *sequence = stmt.as<SeqStmtNode>()) {
    for (const Stmt &child : sequence->seq) {
      RequireNoOpBody(child);
    }
    return;
  }
  if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
    if (!realize->iter_values.empty() || !is_one(realize->predicate)) {
      ThrowUnsupported("non-trivial SBlock realization is deferred beyond "
                       "the Phase 1 skeleton");
    }
    const SBlock &block = realize->block;
    if (!block->iter_vars.empty() || !block->reads.empty() ||
        !block->writes.empty() || !block->alloc_buffers.empty() ||
        !block->match_buffers.empty() || block->init.has_value()) {
      ThrowUnsupported("SBlock dataflow, allocation, or initialization is "
                       "deferred beyond the Phase 1 skeleton");
    }
    RequireNoOpBody(block->body);
    return;
  }
  ThrowUnsupported(std::string("statement '") + stmt->GetTypeKey() +
                   "' requires compute or dataflow planning");
}

class TileAddCounter : public StmtExprVisitor {
public:
  static size_t Count(const Stmt &stmt) {
    TileAddCounter counter;
    counter(stmt);
    return counter.count_;
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tenstorrent::tile_add())) {
      ++count_;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

private:
  size_t count_{0};
};

BufferMetadataMap
IndexBufferMetadata(const ffi::Array<TTBufferMetadata> &buffer_table) {
  BufferMetadataMap by_buffer;
  for (const TTBufferMetadata &metadata : buffer_table) {
    if (!metadata.defined() || !metadata->buffer.defined()) {
      ThrowMalformed("tt.buffer_metadata_table contains an undefined entry");
    }
    if (!by_buffer.emplace(metadata->buffer, metadata).second) {
      ThrowMalformed("tt.buffer_metadata_table contains duplicate Buffer "
                     "identity");
    }
  }
  return by_buffer;
}

const TTBufferMetadata &
RequireMetadata(const BufferMetadataMap &metadata_by_buffer,
                const Buffer &buffer, const std::string &owner) {
  auto it = metadata_by_buffer.find(buffer);
  if (it == metadata_by_buffer.end()) {
    ThrowMalformed(owner + " references Buffer '" + std::string(buffer->name) +
                   "' missing from tt.buffer_metadata_table");
  }
  return it->second;
}

int64_t RequireStaticInteger(const PrimExpr &value, const std::string &field) {
  const int64_t *integer = as_const_int(value);
  if (integer == nullptr) {
    ThrowUnsupported(field + " must be a compile-time integer");
  }
  return *integer;
}

Region2D RequireFullRegion(const BufferRegion &region,
                           const std::string &owner) {
  if (region->region.size() != 2 || region->buffer->shape.size() != 2) {
    ThrowUnsupported(owner + " must be a rank-2 region");
  }
  for (size_t axis = 0; axis < 2; ++axis) {
    const int64_t start =
        RequireStaticInteger(region->region[axis]->min,
                             owner + " start axis " + std::to_string(axis));
    const int64_t extent =
        RequireStaticInteger(region->region[axis]->extent,
                             owner + " extent axis " + std::to_string(axis));
    const int64_t shape = RequireStaticInteger(region->buffer->shape[axis],
                                               owner + " buffer shape axis " +
                                                   std::to_string(axis));
    if (start != 0 || extent != shape) {
      ThrowUnsupported(owner + " must cover the complete Buffer; partial Add "
                               "tiles are deferred beyond Phase 2");
    }
  }
  return {region->region[0]->min, region->region[1]->min,
          region->region[0]->extent, region->region[1]->extent};
}

void RequireMatchingRegions(const BufferRegion &lhs, const BufferRegion &rhs,
                            const std::string &owner) {
  if (lhs->region.size() != rhs->region.size()) {
    ThrowMalformed(owner + " region ranks disagree");
  }
  for (size_t axis = 0; axis < lhs->region.size(); ++axis) {
    if (!ffi::StructuralEqual()(lhs->region[axis], rhs->region[axis])) {
      ThrowMalformed(owner + " regions disagree on axis " +
                     std::to_string(axis));
    }
  }
}

Call RequireCall(const Stmt &stmt, const Op &expected,
                 const std::string &owner) {
  const auto *evaluate = stmt.as<EvaluateNode>();
  if (evaluate == nullptr) {
    ThrowMalformed(owner + " must be an Evaluate(Call) statement");
  }
  const auto *call_node = evaluate->value.as<CallNode>();
  if (call_node == nullptr || !call_node->op.same_as(expected)) {
    ThrowMalformed(owner + " has an unexpected operation");
  }
  return ffi::GetRef<Call>(call_node);
}

void RequireTransferKind(const Call &call, const ffi::String &expected,
                         const std::string &owner) {
  ffi::Optional<ffi::ObjectRef> value = call->annotations.Get(kTransferKind);
  if (!value.has_value()) {
    ThrowMalformed(owner + " has no tt.transfer_kind annotation");
  }
  const auto *kind = value.value().as<StringImmNode>();
  if (kind == nullptr || kind->value != expected) {
    ThrowMalformed(owner + " has the wrong tt.transfer_kind annotation");
  }
}

void ValidateTileAddAnnotations(const Call &call, const DataType &dtype) {
  ffi::Optional<ffi::ObjectRef> dtype_value =
      call->annotations.Get(kComputeDType);
  ffi::Optional<ffi::ObjectRef> tile_shape_value =
      call->annotations.Get(kComputeTileShape);
  if (!dtype_value.has_value() || !tile_shape_value.has_value()) {
    ThrowMalformed("canonical tl.tt.tile_add is missing compute annotations");
  }
  const auto *dtype_string = dtype_value.value().as<StringImmNode>();
  const ffi::String expected_dtype =
      dtype == DataType::BFloat(16) ? "bfloat16" : "float32";
  if (dtype_string == nullptr || dtype_string->value != expected_dtype) {
    ThrowMalformed("canonical tl.tt.tile_add compute dtype disagrees with its "
                   "operands");
  }
  auto tile_shape = tile_shape_value.value().as<ffi::Array<PrimExpr>>();
  if (!tile_shape.has_value() || tile_shape.value().size() != 2 ||
      RequireStaticInteger(tile_shape.value()[0], "Add tile rows") != 32 ||
      RequireStaticInteger(tile_shape.value()[1], "Add tile cols") != 32) {
    ThrowMalformed("canonical tl.tt.tile_add tile shape must be [32, 32]");
  }
}

std::vector<Stmt> RequireSequence(const Stmt &stmt) {
  if (const auto *sequence = stmt.as<SeqStmtNode>()) {
    return {sequence->seq.begin(), sequence->seq.end()};
  }
  return {stmt};
}

AddDataflowPlan
AnalyzeAddDataflow(const PrimFunc &frontend, const Stmt &kernel_body,
                   const ffi::Array<TTBufferMetadata> &buffer_table) {
  if (buffer_table.size() != 6 || frontend->params.size() != 3) {
    ThrowUnsupported("Phase 2 Add requires three Tensor parameters and three "
                     "shared DFB buffers");
  }
  const auto *realize_node = kernel_body.as<SBlockRealizeNode>();
  if (realize_node == nullptr || !realize_node->iter_values.empty() ||
      !is_one(realize_node->predicate)) {
    ThrowMalformed("Phase 2 Add must be enclosed by one canonical root SBlock");
  }
  SBlock block = realize_node->block;
  if (!block->iter_vars.empty() || !block->reads.empty() ||
      !block->writes.empty() || !block->match_buffers.empty() ||
      block->init.has_value()) {
    ThrowMalformed("Phase 2 Add root SBlock has non-canonical dataflow fields");
  }
  if (block->alloc_buffers.size() != 3) {
    ThrowUnsupported("Phase 2 Add requires exactly three shared DFB buffers");
  }

  std::vector<Stmt> statements = RequireSequence(block->body);
  if (statements.size() != 4) {
    ThrowUnsupported("Phase 2 Add requires exactly two input copies, one Add, "
                     "and one output copy");
  }

  Call input_a_call = RequireCall(statements[0], Copy::Get(), "input A copy");
  Call input_b_call = RequireCall(statements[1], Copy::Get(), "input B copy");
  Call add_call =
      RequireCall(statements[2], tenstorrent::tile_add(), "canonical Add");
  Call output_call = RequireCall(statements[3], Copy::Get(), "output C copy");
  RequireTransferKind(input_a_call, "tensor_to_dfb", "input A copy");
  RequireTransferKind(input_b_call, "tensor_to_dfb", "input B copy");
  RequireTransferKind(output_call, "dfb_to_tensor", "output C copy");

  Copy input_a = Downcast<Copy>(ParseOperator(input_a_call));
  Copy input_b = Downcast<Copy>(ParseOperator(input_b_call));
  Copy output = Downcast<Copy>(ParseOperator(output_call));
  if (add_call->args.size() != 3) {
    ThrowMalformed(
        "canonical tl.tt.tile_add must have three BufferRegion args");
  }
  BufferRegion add_a =
      NormalizeToAccessRegion(add_call->args[0], kAccessRead).region;
  BufferRegion add_b =
      NormalizeToAccessRegion(add_call->args[1], kAccessRead).region;
  BufferRegion add_c =
      NormalizeToAccessRegion(add_call->args[2], kAccessWrite).region;

  BufferRegion input_a_source(input_a->src, input_a->src_range);
  BufferRegion input_a_destination(input_a->dst, input_a->dst_range);
  BufferRegion input_b_source(input_b->src, input_b->src_range);
  BufferRegion input_b_destination(input_b->dst, input_b->dst_range);
  BufferRegion output_source(output->src, output->src_range);
  BufferRegion output_destination(output->dst, output->dst_range);

  if (!input_a_destination->buffer.same_as(add_a->buffer) ||
      !input_b_destination->buffer.same_as(add_b->buffer) ||
      !output_source->buffer.same_as(add_c->buffer)) {
    ThrowMalformed("copy and canonical Add Buffer identities do not form "
                   "Tensor->DFB->Add->DFB->Tensor dataflow");
  }
  if (!block->alloc_buffers[0].same_as(add_a->buffer) ||
      !block->alloc_buffers[1].same_as(add_b->buffer) ||
      !block->alloc_buffers[2].same_as(add_c->buffer)) {
    ThrowMalformed("canonical Add operands do not follow shared allocation "
                   "lexical order");
  }
  RequireMatchingRegions(input_a_destination, add_a, "input A DFB");
  RequireMatchingRegions(input_b_destination, add_b, "input B DFB");
  RequireMatchingRegions(output_source, add_c, "output C DFB");

  AddDataflowPlan plan;
  plan.tensor_regions[0] = RequireFullRegion(input_a_source, "input A Tensor");
  plan.tensor_regions[1] = RequireFullRegion(input_b_source, "input B Tensor");
  plan.tensor_regions[2] =
      RequireFullRegion(output_destination, "output C Tensor");
  RequireFullRegion(add_a, "input A DFB");
  RequireFullRegion(add_b, "input B DFB");
  RequireFullRegion(add_c, "output C DFB");

  BufferMetadataMap metadata_by_buffer = IndexBufferMetadata(buffer_table);
  plan.tensors = {
      RequireMetadata(metadata_by_buffer, input_a_source->buffer, "input A"),
      RequireMetadata(metadata_by_buffer, input_b_source->buffer, "input B"),
      RequireMetadata(metadata_by_buffer, output_destination->buffer,
                      "output C")};
  plan.dfbs = {
      RequireMetadata(metadata_by_buffer, add_a->buffer, "input A DFB"),
      RequireMetadata(metadata_by_buffer, add_b->buffer, "input B DFB"),
      RequireMetadata(metadata_by_buffer, add_c->buffer, "output C DFB")};

  for (size_t index = 0; index < 3; ++index) {
    const TTBufferMetadata &tensor = plan.tensors[index];
    const TTBufferMetadata &dfb = plan.dfbs[index];
    if (tensor->kind != "tensor" || !tensor->global_arg_index.has_value() ||
        tensor->global_arg_index.value()->value !=
            static_cast<int64_t>(index)) {
      ThrowMalformed("Add Tensor operands must follow ABI order A=0, B=1, C=2");
    }
    if (dfb->kind != "logical_dfb_candidate" ||
        dfb->buffer_id != "buffer." + std::to_string(index)) {
      ThrowMalformed("Add DFB operands must follow stable shared allocation "
                     "order a=0, b=1, c=2");
    }
    if (dfb->buffer->dtype != tensor->buffer->dtype ||
        !ffi::StructuralEqual()(dfb->buffer->shape, tensor->buffer->shape)) {
      ThrowMalformed("Add Tensor and DFB dtype/shape disagree at operand " +
                     std::to_string(index));
    }
    if (dfb->tile_grid_shape.size() != 2 ||
        RequireStaticInteger(dfb->tile_grid_shape[0], "DFB tile-grid rows") !=
            1 ||
        RequireStaticInteger(dfb->tile_grid_shape[1], "DFB tile-grid cols") !=
            1) {
      ThrowUnsupported("Phase 2 Add supports exactly one 32x32 tile per DFB");
    }
    if (!dfb->dfb_block_count.has_value()) {
      ThrowMalformed("Add DFB metadata has no block count");
    }
  }
  if (plan.tensors[0]->buffer->dtype != plan.tensors[1]->buffer->dtype ||
      plan.tensors[0]->buffer->dtype != plan.tensors[2]->buffer->dtype) {
    ThrowMalformed("Add Tensor operand dtypes disagree");
  }
  ValidateTileAddAnnotations(add_call, plan.tensors[0]->buffer->dtype);
  plan.compute_span = RequireSourceSpan(add_call->span, frontend->span,
                                        "canonical Add operation");
  return plan;
}

ffi::Array<TensorDescriptor> BuildTensorTable(
    const PrimFunc &func, const ffi::Array<TTBufferMetadata> &buffer_table,
    const ffi::Array<ffi::String> &effects, bool allow_dfb_candidates) {
  if (effects.size() != func->params.size()) {
    ThrowMalformed("Tensor effect plan does not cover every ABI parameter");
  }
  ffi::Array<TensorDescriptor> tensors;
  size_t expected_tensor_index = 0;
  for (const TTBufferMetadata &metadata : buffer_table) {
    if (!metadata.defined() || !metadata->buffer.defined()) {
      ThrowMalformed("tt.buffer_metadata_table contains an undefined entry");
    }
    if (metadata->kind != "tensor") {
      if (metadata->kind == "compute_fragment" && allow_dfb_candidates)
        continue;
      if (metadata->kind == "logical_dfb_candidate") {
        if (allow_dfb_candidates) {
          continue;
        }
        ThrowUnsupported("logical DFB buffer '" +
                         std::string(metadata->buffer_id) +
                         "' requires Phase 2 transaction planning");
      }
      ThrowUnsupported("buffer '" + std::string(metadata->buffer_id) +
                       "' of kind '" + std::string(metadata->kind) +
                       "' is outside the Phase 1 skeleton");
    }
    if (!metadata->global_arg_index.has_value()) {
      ThrowMalformed("Tensor metadata '" + std::string(metadata->buffer_id) +
                     "' has no global_arg_index");
    }
    const int64_t global_arg_index = metadata->global_arg_index.value()->value;
    if (global_arg_index != static_cast<int64_t>(expected_tensor_index) ||
        expected_tensor_index >= func->params.size()) {
      ThrowMalformed("Tensor metadata is not in stable ABI parameter order");
    }
    const Buffer &buffer = metadata->buffer;
    Span source_span = RequireSourceSpan(
        metadata->source_span, func->span,
        "Tensor metadata '" + std::string(metadata->buffer_id) + "'");
    tensors.push_back(TensorDescriptor(
        global_arg_index, buffer->shape, buffer->dtype, buffer->strides,
        metadata->tile_shape, metadata->tile_grid_shape, "dram",
        metadata->memory_layout, metadata->shard_spec,
        effects[expected_tensor_index], global_arg_index,
        std::move(source_span)));
    ++expected_tensor_index;
  }
  if (expected_tensor_index != func->params.size()) {
    ThrowMalformed("tt.buffer_metadata_table does not cover every Tensor ABI "
                   "parameter");
  }
  return tensors;
}

ffi::Array<DFBDescriptor> BuildAddDFBTable(const AddDataflowPlan &plan,
                                           const CoreDomain &domain,
                                           const PrimFunc &frontend) {
  ffi::Array<DFBDescriptor> result;
  for (size_t index = 0; index < plan.dfbs.size(); ++index) {
    const TTBufferMetadata &metadata = plan.dfbs[index];
    Span source_span = RequireSourceSpan(
        metadata->source_span, frontend->span,
        "DFB metadata '" + std::string(metadata->buffer_id) + "'");
    ffi::String producer = index < 2 ? "ncrisc" : "trisc";
    ffi::String consumer = index < 2 ? "trisc" : "ncrisc";
    result.push_back(DFBDescriptor(
        static_cast<int64_t>(index), metadata->buffer_id,
        metadata->buffer->dtype, metadata->tile_shape,
        metadata->tile_grid_shape, metadata->dfb_block_count.value(),
        TensorBacking(static_cast<int64_t>(index), Integer(0)), producer,
        domain, consumer, domain, Integer(1), std::move(source_span)));
  }
  return result;
}

Stmt MakeDeviceCall(const Op &op, ffi::Array<PrimExpr> args, const Span &span) {
  return Evaluate(Call(DataType::Void(), op, std::move(args), {}, span), span);
}

Stmt BuildAddTRISCBody(const AddDataflowPlan &plan) {
  const Span &span = plan.compute_span;
  return SeqStmt(
      {MakeDeviceCall(tenstorrent::dfb_reserve(), {Integer(2), Integer(1)},
                      span),
       MakeDeviceCall(tenstorrent::dfb_wait(), {Integer(0), Integer(1)}, span),
       MakeDeviceCall(tenstorrent::dfb_wait(), {Integer(1), Integer(1)}, span),
       MakeDeviceCall(tenstorrent::dfb_add(),
                      {Integer(0), Integer(1), Integer(2), Integer(1)}, span)},
      span);
}

ffi::Array<PrimExpr> MakeTransferArgs(int64_t tensor_index, int64_t dfb_id,
                                      const Region2D &region,
                                      bool tensor_is_source) {
  ffi::Array<PrimExpr> args;
  if (tensor_is_source) {
    args.push_back(Integer(tensor_index));
    args.push_back(Integer(dfb_id));
  } else {
    args.push_back(Integer(dfb_id));
    args.push_back(Integer(tensor_index));
  }
  args.push_back(region.row_start);
  args.push_back(region.col_start);
  args.push_back(region.rows);
  args.push_back(region.cols);
  return args;
}

Stmt BuildAddNCRISCBody(const AddDataflowPlan &plan) {
  const Span &span = plan.compute_span;
  return SeqStmt(
      {MakeDeviceCall(tenstorrent::dfb_reserve(), {Integer(0), Integer(1)},
                      span),
       MakeDeviceCall(tenstorrent::tensor_to_dfb(),
                      MakeTransferArgs(0, 0, plan.tensor_regions[0], true),
                      span),
       MakeDeviceCall(tenstorrent::dfb_reserve(), {Integer(1), Integer(1)},
                      span),
       MakeDeviceCall(tenstorrent::tensor_to_dfb(),
                      MakeTransferArgs(1, 1, plan.tensor_regions[1], true),
                      span),
       MakeDeviceCall(tenstorrent::dfb_wait(), {Integer(2), Integer(1)}, span),
       MakeDeviceCall(tenstorrent::dfb_to_tensor(),
                      MakeTransferArgs(2, 2, plan.tensor_regions[2], false),
                      span)},
      span);
}

// Phase 4 assigns an immutable resource to every write.  This deliberately
// avoids capacity reuse, cyclic schedules, and cross-Core communication.
struct ResourceVersion {
  TTBufferMetadata metadata;
  ffi::String producer;
  ffi::String consumer;
  ffi::Optional<TensorBacking> backing;
  int64_t pool{-1};
  int64_t iteration{-1};
};

class DeviceExpressionRewriter : public StmtExprMutator {
public:
  explicit DeviceExpressionRewriter(
      const std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash,
                               ffi::ObjectPtrEqual> &inputs)
      : inputs_(inputs) {}
  DeviceExpressionRewriter(
      const std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash,
                               ffi::ObjectPtrEqual> &inputs,
      const std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash,
                               ffi::ObjectPtrEqual> &values)
      : inputs_(inputs), values_(&values) {}
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    if (values_) {
      auto value = values_->find(op->buffer);
      if (value != values_->end())
        return Call(op->dtype, tenstorrent::compute_value_load(),
                    {Integer(value->second)}, {}, op->span);
    }
    auto found = inputs_.find(op->buffer);
    if (found == inputs_.end()) {
      ThrowMalformed("compute expression reads an undeclared input");
    }
    return Call(op->dtype, tenstorrent::dfb_load(), {Integer(found->second)},
                {}, op->span);
  }

private:
  const std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash,
                           ffi::ObjectPtrEqual> &inputs_;
  const std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash,
                           ffi::ObjectPtrEqual> *values_{nullptr};
};

// TVM's generic expression mutator substitutes PrimExpr call annotations, but
// reconstructs a changed Call without its span. Keep operation provenance when
// specializing a static iteration's scalar expressions.
class StaticIterationSubstituter : public StmtExprMutator {
public:
  StaticIterationSubstituter(Var variable, PrimExpr value)
      : variable_(std::move(variable)), value_(std::move(value)) {}

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var variable = ffi::GetRef<Var>(op);
    if (variable_.same_as(variable))
      return value_;
    auto found = aliases_.find(variable);
    return found == aliases_.end() ? variable : found->second;
  }

  PrimExpr VisitExpr(const PrimExpr &expr) final {
    PrimExpr result = StmtExprMutator::VisitExpr(expr);
    return result.dtype().is_handle() ||
                   SideEffect(result) > CallEffectKind::kPure
               ? result
               : analyzer_.Simplify(result);
  }

  Stmt VisitStmt_(const BindNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    if (value.as<IntImmNode>() || value.as<FloatImmNode>()) {
      aliases_[op->var] = value;
      return Evaluate(Integer(0), op->span);
    }
    return Bind(op->var, value, op->span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call.same_as(ffi::GetRef<Call>(op)))
      call.CopyOnWrite()->span = op->span;
    return call;
  }

private:
  Var variable_;
  PrimExpr value_;
  arith::Analyzer analyzer_;
  std::unordered_map<Var, PrimExpr, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      aliases_;
};

struct PipeEndpoint {
  int64_t transfer_id;
  int64_t occurrence;
  ffi::Array<Integer> record;
  int64_t x, y, dfb_id;
  bool source;
  Span span;
};
using TransferKeys =
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>, int64_t>;

struct AccumulatorLifetime {
  BufferRegion region;
  DataType input_dtype{DataType::Void()};
  DataType output_dtype{DataType::Void()};
  int64_t full_k_tiles{0};
  int64_t output_id{-1};
  std::vector<int64_t> inputs;
  Span span;
};

class GeneralDataflowPlanner {
public:
  ffi::Array<AccumulatorDescriptor> Accumulators() const {
    ffi::Array<AccumulatorDescriptor> result;
    for (size_t id = 0; id < accumulators_.size(); ++id) {
      const auto &state = accumulators_[id];
      if ((!compute_value_mode_ && state.output_id < 0) ||
          state.full_k_tiles <= 0)
        ThrowMalformed(
            "accumulator requires updates and one final materialization");
      if (compute_value_mode_ || live_.count(state.output_id))
        result.push_back(AccumulatorDescriptor(
            id, state.region, state.input_dtype, state.region->buffer->dtype,
            state.output_dtype, state.full_k_tiles,
            RequireSourceSpan(state.span, frontend_->span, "accumulator")));
    }
    return result;
  }

  GeneralDataflowPlanner(const PrimFunc &frontend,
                         const ffi::Array<TTBufferMetadata> &metadata,
                         TransferKeys *transfers = nullptr, int64_t x = 0,
                         int64_t y = 0)
      : transfers_(transfers), core_x_(x), core_y_(y), frontend_(frontend),
        metadata_(IndexBufferMetadata(metadata)),
        tensor_reads_(frontend->params.size(), false),
        tensor_writes_(frontend->params.size(), false) {
    bool bf16_gemm = false, fp32_compute = false;
    PostOrderVisit(frontend->body, [&](const ffi::ObjectRef &object) {
      auto call = object.as<Call>();
      if (!call.has_value() || !call.value()->op.same_as(tile_compute()))
        return;
      if (call.value()->args.empty())
        ThrowMalformed("tile_compute has no output region");
      BufferRegion compute_output =
          NormalizeToAccessRegion(call.value()->args[0], kAccessWrite).region;
      fp32_compute |= compute_output->buffer->dtype == DataType::Float(32);
      auto kind = call.value()->annotations.Get("tt.compute_kind");
      if (kind.has_value()) {
        auto name = Downcast<StringImm>(kind.value())->value;
        if (name == "gemm_update") {
          BufferRegion output =
              NormalizeToAccessRegion(call.value()->args[0], kAccessWrite)
                  .region;
          bf16_gemm |= output->buffer->dtype == DataType::BFloat(16);
        }
        auto expression = call.value()->annotations.Get("tt.expression");
        if (expression.has_value())
          PostOrderVisit(Downcast<PrimExpr>(expression.value()),
                         [&](const ffi::ObjectRef &item) {
                           if (auto value = item.as<PrimExpr>())
                             fp32_compute |=
                                 value.value().dtype() == DataType::Float(32);
                         });
        if (name == "accumulator_init" || name == "gemm_update" ||
            name == "accumulator_materialize")
          return;
      }
      for (const PrimExpr &arg : call.value()->args) {
        BufferRegion region = NormalizeToAccessRegion(arg, kAccessRead).region;
        compute_value_mode_ |= region->buffer.scope() == "local.fragment";
      }
    });
    precision_regions_ = bf16_gemm && fp32_compute;
    compute_value_mode_ |= precision_regions_;
  }

  bool HasComputeValues() const { return compute_value_mode_; }
  bool HasPrecisionRegions() const { return precision_regions_; }
  ffi::Array<ComputeValueDescriptor> ComputeValues() const { return values_; }
  void FinalizeComputeValues() {
    if (!compute_value_mode_ || IsPipeline())
      return;
    AddReleases(&compute_, "trisc");
    AddReleases(&transfer_, "ncrisc");
  }

  const std::vector<PipeEndpoint> &Endpoints() const { return endpoints_; }
  int64_t ResourceCount() const { return resources_.size(); }
  int64_t AccumulatorCount() const { return accumulators_.size(); }
  bool HasProducerForwarding() const { return has_producer_forwarding_; }
  Stmt PipeBody() const { return Body(pipe_); }

  void FinalizeMulticore() {
    for (size_t id = 0; id < resources_.size(); ++id) {
      if (live_.count(id) && resources_[id].consumer.empty()) {
        resources_[id].consumer = "ncrisc";
        Wait(&transfer_, id, frontend_->span);
      }
    }
    if (IsPipeline()) {
      FinalizePipeline();
      return;
    }
    AddReleases(&compute_, "trisc");
    AddReleases(&transfer_, "ncrisc");
    AddReleases(&pipe_, "brisc");
  }

  // Phase 5 materializes bounded windows. A value remains an immutable
  // generation, while the same lexical write site shares one capacity pool.
  void PlanPipeline(const For &loop) {
    if (precision_regions_)
      ThrowUnsupported(
          "precision regions require statically serialized transactions");
    if (IsPipeline())
      ThrowUnsupported("one bounded T.Pipelined loop per Core is supported");
    pipeline_resource_begin_ = resources_.size();
    for (const auto &[key, value] : loop->annotations) {
      if (key != "num_stages" && key != "tt.pipeline_wait_policy")
        ThrowUnsupported(
            "T.Pipelined manual stage/order/group schedules are unsupported");
    }
    if (!loop->annotations.count("num_stages") ||
        loop->kind != ForKind::kSerial || loop->thread_binding.has_value() ||
        (loop->step.has_value() && !is_one(loop->step.value()))) {
      ThrowUnsupported("T.Pipelined requires num_stages only and unit step; "
                       "manual stage/order/group schedules are unsupported");
    }
    auto policy = loop->annotations.Get("tt.pipeline_wait_policy");
    if (policy.has_value()) {
      if (auto string = policy.value().as<ffi::String>()) {
        pipeline_wait_policy_ = string.value();
      } else if (auto literal = policy.value().as<StringImm>()) {
        pipeline_wait_policy_ = literal.value()->value;
      } else {
        ThrowMalformed("tt.pipeline_wait_policy must be a string");
      }
      if (pipeline_wait_policy_ != "conservative" &&
          pipeline_wait_policy_ != "delayed")
        ThrowMalformed(
            "tt.pipeline_wait_policy must be conservative or delayed");
    }
    pipeline_stages_ = RequirePositiveStaticInteger(
        Downcast<PrimExpr>(loop->annotations.at("num_stages")),
        "T.Pipelined num_stages");
    pipeline_extent_ =
        RequirePositiveStaticInteger(loop->extent, "T.Pipelined extent");
    int64_t minimum = RequireStaticInteger(loop->min, "T.Pipelined minimum");
    if (pipeline_stages_ > 32 || pipeline_extent_ > 1024) {
      ThrowUnsupported(
          "T.Pipelined requires num_stages <= 32 and extent <= 1024");
    }
    pipeline_depth_ = std::min(pipeline_stages_, pipeline_extent_);
    PostOrderVisit(loop->body, [&](const ffi::ObjectRef &object) {
      if (object.as<IfThenElseNode>()) {
        ThrowUnsupported("T.Pipelined conditional write sites are unsupported; "
                         "each iteration must have the same transaction sites");
      }
      if (const auto *nested = object.as<ForNode>()) {
        if (!nested->annotations.empty()) {
          ThrowUnsupported(
              "nested T.Pipelined or annotated loops are unsupported");
        }
      }
    });
    size_t writes_per_iteration = 0;
    for (int64_t iteration = 0; iteration < pipeline_extent_; ++iteration) {
      current_.clear();
      pipeline_value_begin_ = values_.size();
      size_t begin = resources_.size();
      StaticIterationSubstituter substitute(
          loop->loop_var, IntImm(loop->loop_var.dtype(), minimum + iteration));
      Plan(substitute(loop->body));
      size_t count = resources_.size() - begin;
      if (iteration == 0) {
        writes_per_iteration = count;
      } else if (count != writes_per_iteration) {
        ThrowUnsupported("T.Pipelined requires identical transaction sites in "
                         "every iteration");
      }
      for (size_t index = begin; index < resources_.size(); ++index) {
        resources_[index].pool = index - begin;
        resources_[index].iteration = iteration;
      }
    }
    for (size_t tensor = 0; tensor < tensor_reads_.size(); ++tensor) {
      if (tensor_reads_[tensor] && tensor_writes_[tensor]) {
        ThrowUnsupported("T.Pipelined prefetch cannot reorder an inout Tensor; "
                         "use a serial loop for cross-iteration Tensor state");
      }
    }
    pipeline_value_begin_ = 0;
  }

  bool IsPipeline() const { return pipeline_extent_ != 0; }
  int64_t PipelineStages() const { return pipeline_stages_; }
  int64_t PipelineExtent() const { return pipeline_extent_; }
  ffi::String PipelineWaitPolicy() const { return pipeline_wait_policy_; }

  ffi::Map<ffi::String, Integer> StorageGroups() const {
    ffi::Map<ffi::String, Integer> groups;
    for (size_t id = 0; id < resources_.size(); ++id) {
      if (live_.count(id)) {
        groups.Set(std::to_string(id),
                   Integer(resources_[id].pool < 0 ? resources_.size() + id
                                                   : resources_[id].pool));
      }
    }
    return groups;
  }

  ffi::Map<ffi::String, ffi::Array<Integer>> PipelineRelations() const {
    ffi::Map<ffi::String, ffi::Array<Integer>> relations;
    for (size_t id = 0; id < resources_.size(); ++id) {
      if (live_.count(id)) {
        int64_t iteration = resources_[id].iteration;
        relations.Set(
            std::to_string(id),
            {Integer(iteration),
             Integer(iteration < 0 ? 0 : iteration % pipeline_depth_)});
      }
    }
    return relations;
  }

  void FinalizePipeline() {
    if (!IsPipeline())
      return;
    ffi::Array<Stmt> scheduled;
    auto append_singleton = [&](const Stmt &statement) {
      const CallNode *call = statement.as<EvaluateNode>()->value.as<CallNode>();
      if (call->op.same_as(dfb_copy_wait()))
        return;
      scheduled.push_back(statement);
      if (call->op.same_as(tensor_to_dfb_nd()) ||
          call->op.same_as(dfb_to_tensor_nd())) {
        size_t arg = call->op.same_as(tensor_to_dfb_nd()) ? 1 : 0;
        scheduled.push_back(MakeDeviceCall(
            dfb_copy_wait(), {call->args[arg], Integer(1)}, statement->span));
      }
    };
    auto resource_id = [](const CallNode *call) {
      bool second = call->op.same_as(tensor_to_dfb_nd()) ||
                    call->op.same_as(dfb_pipe_recv()) ||
                    call->op.same_as(dfb_pipe_send()) ||
                    call->op.same_as(dfb_pipe_wait());
      return *as_const_int(call->args[second ? 1 : 0]);
    };
    for (const Stmt &statement : transfer_) {
      int64_t id =
          resource_id(statement.as<EvaluateNode>()->value.as<CallNode>());
      if (resources_[id].iteration < 0 && id < pipeline_resource_begin_)
        append_singleton(statement);
    }
    // Delayed policy issues the entire input window before copy completion;
    // conservative policy completes each copy immediately. Output waits stay
    // after all input publications, avoiding a cross-slot cycle.
    for (int64_t begin = 0; begin < pipeline_extent_;
         begin += pipeline_depth_) {
      ffi::Array<Stmt> completions, outputs;
      std::unordered_set<int64_t> forwarding_completions;
      for (const Stmt &statement : transfer_) {
        const CallNode *call =
            statement.as<EvaluateNode>()->value.as<CallNode>();
        bool input_copy = call->op.same_as(tenstorrent::tensor_to_dfb_nd());
        if (call->op.same_as(dfb_copy_wait()))
          continue;
        bool pipe = call->op.same_as(dfb_pipe_recv()) ||
                    call->op.same_as(dfb_pipe_send()) ||
                    call->op.same_as(dfb_pipe_wait());
        int64_t id = *as_const_int(call->args[input_copy || pipe ? 1 : 0]);
        const ResourceVersion &resource = resources_[id];
        if (resource.iteration < begin ||
            resource.iteration >= begin + pipeline_depth_)
          continue;
        if (resource.producer == "ncrisc") {
          // A forwarded panel must complete its Tensor copy before transport.
          // Complete only that dependency here; other input copies retain the
          // delayed window, and future K panels can overlap local computation.
          if (call->op.same_as(dfb_pipe_send()) &&
              forwarding_completions.insert(id).second) {
            for (const Stmt &completion : completions) {
              const auto *pending =
                  completion.as<EvaluateNode>()->value.as<CallNode>();
              if (*as_const_int(pending->args[0]) == id)
                scheduled.push_back(completion);
            }
          }
          scheduled.push_back(statement);
          if (input_copy) {
            Stmt completion =
                MakeDeviceCall(tenstorrent::dfb_copy_wait(),
                               {Integer(id), Integer(1)}, statement->span);
            if (pipeline_wait_policy_ == "conservative")
              scheduled.push_back(completion);
            else
              completions.push_back(completion);
          }
        } else {
          outputs.push_back(statement);
          if (call->op.same_as(tenstorrent::dfb_to_tensor_nd()))
            outputs.push_back(MakeDeviceCall(tenstorrent::dfb_copy_wait(),
                                             {Integer(id), Integer(1)},
                                             statement->span));
        }
      }
      for (const Stmt &statement : completions) {
        const auto *completion =
            statement.as<EvaluateNode>()->value.as<CallNode>();
        if (!forwarding_completions.count(*as_const_int(completion->args[0])))
          scheduled.push_back(statement);
      }
      for (const Stmt &statement : outputs)
        scheduled.push_back(statement);
    }
    for (const Stmt &statement : transfer_) {
      int64_t id =
          resource_id(statement.as<EvaluateNode>()->value.as<CallNode>());
      if (resources_[id].iteration < 0 && id >= pipeline_resource_begin_)
        append_singleton(statement);
    }
    transfer_ = std::move(scheduled);
    AddReleases(&compute_, "trisc");
    AddReleases(&transfer_, "ncrisc");
    if (transfers_)
      AddReleases(&pipe_, "brisc");
  }

  void EliminateDeadWrites() {
    if (compute_value_mode_) {
      EliminateDeadValueWrites();
      return;
    }
    std::unordered_set<int64_t> live;
    std::unordered_map<int64_t, std::vector<int64_t>> inputs;
    for (const Stmt &statement : compute_) {
      const auto *call = statement.as<EvaluateNode>()->value.as<CallNode>();
      if (call->op.same_as(tenstorrent::dfb_compute())) {
        int64_t output = *as_const_int(call->args[0]);
        for (size_t i = 1; i < call->args.size(); ++i)
          inputs[output].push_back(*as_const_int(call->args[i]));
      }
    }
    for (const auto &state : accumulators_) {
      if (state.output_id < 0)
        ThrowMalformed("accumulator has no final materialization");
      inputs[state.output_id] = state.inputs;
    }
    std::function<void(int64_t)> mark = [&](int64_t id) {
      if (!live.insert(id).second)
        return;
      for (int64_t input : inputs[id])
        mark(input);
    };
    for (const Stmt &statement : transfer_) {
      const auto *call = statement.as<EvaluateNode>()->value.as<CallNode>();
      if (call->op.same_as(tenstorrent::dfb_to_tensor_nd()))
        mark(*as_const_int(call->args[0]));
    }
    for (const PipeEndpoint &endpoint : endpoints_)
      mark(endpoint.dfb_id);
    auto filter = [&](const ffi::Array<Stmt> &body) {
      ffi::Array<Stmt> result;
      for (const Stmt &statement : body) {
        const auto *call = statement.as<EvaluateNode>()->value.as<CallNode>();
        if (call->op.same_as(tenstorrent::accumulator_init()) ||
            call->op.same_as(tenstorrent::gemm_update()) ||
            call->op.same_as(tenstorrent::accumulator_materialize())) {
          size_t arg = call->op.same_as(tenstorrent::gemm_update()) ? 2 : 0;
          int64_t id = *as_const_int(call->args[arg]);
          if (live.count(accumulators_[id].output_id))
            result.push_back(statement);
          continue;
        }
        size_t arg = call->op.same_as(tenstorrent::tensor_to_dfb_nd()) ||
                             call->op.same_as(tenstorrent::dfb_pipe_send()) ||
                             call->op.same_as(tenstorrent::dfb_pipe_recv()) ||
                             call->op.same_as(tenstorrent::dfb_pipe_wait())
                         ? 1
                         : 0;
        if (live.count(*as_const_int(call->args[arg])))
          result.push_back(statement);
      }
      return result;
    };
    compute_ = filter(compute_);
    transfer_ = filter(transfer_);
    live_ = std::move(live);
    std::fill(tensor_reads_.begin(), tensor_reads_.end(), false);
    for (const Stmt &statement : transfer_) {
      const auto *call = statement.as<EvaluateNode>()->value.as<CallNode>();
      if (call->op.same_as(tenstorrent::tensor_to_dfb_nd()))
        tensor_reads_[*as_const_int(call->args[0])] = true;
    }
  }

  void Plan(const Stmt &stmt) {
    if (!precision_regions_) {
      PlanImpl(stmt);
      return;
    }
    // Use the same static specialization as planning, including loop-carried
    // reads and constant branches. Do not preserve dead fragment generations
    // merely because they were the most recent definition of their buffer.
    std::vector<Call> calls;
    collected_calls_ = &calls;
    PlanImpl(stmt);
    collected_calls_ = nullptr;
    PrepareFragmentAccesses(calls);
    for (call_index_ = 0; call_index_ < calls.size(); ++call_index_)
      PlanCall(calls[call_index_]);
    fragment_accesses_.clear();
    unknown_accesses_.clear();
  }

  void PlanImpl(const Stmt &stmt) {
    if (++statement_count_ > 65536) {
      ThrowUnsupported(
          "static control-flow expansion exceeds 65536 statements");
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : seq->seq)
        PlanImpl(child);
      return;
    }
    if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
      if (!realize->iter_values.empty() || !is_one(realize->predicate) ||
          !realize->block->iter_vars.empty() ||
          realize->block->init.has_value() ||
          !realize->block->match_buffers.empty()) {
        ThrowUnsupported(
            "non-canonical block realization or match-buffer alias");
      }
      if (realize->block->annotations.count("tt.compute_kind")) {
        ThrowMalformed(
            "unconsumed compute block; run LegalizeTenstorrentTileOps");
      }
      PlanImpl(realize->block->body);
      return;
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      if (loop->annotations.count("num_stages")) {
        PlanPipeline(ffi::GetRef<For>(loop));
        return;
      }
      if (loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled) {
        ThrowUnsupported(
            "control flow must use a static serial or unrolled loop");
      }
      if (!loop->annotations.empty() ||
          (loop->step.has_value() && !is_one(loop->step.value()))) {
        ThrowUnsupported("serial computation requires unit step and no "
                         "pipeline/scheduling annotations");
      }
      int64_t minimum = RequireStaticInteger(loop->min, "serial loop minimum");
      int64_t extent =
          RequirePositiveStaticInteger(loop->extent, "serial loop extent");
      if (extent > 1024)
        ThrowUnsupported("serial loop extent exceeds 1024");
      for (int64_t i = 0; i < extent; ++i) {
        StaticIterationSubstituter substitute(
            loop->loop_var, IntImm(loop->loop_var.dtype(), minimum + i));
        PlanImpl(substitute(loop->body));
      }
      return;
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      arith::Analyzer analyzer;
      PrimExpr condition = analyzer.Simplify(branch->condition);
      if (!as_const_int(condition)) {
        ThrowUnsupported(
            "data-dependent control flow requires a proven balanced "
            "cross-slot transaction schedule");
      }
      if (is_one(condition))
        PlanImpl(branch->then_case);
      else if (branch->else_case.has_value())
        PlanImpl(branch->else_case.value());
      return;
    }
    const auto *evaluate = stmt.as<EvaluateNode>();
    if (!evaluate)
      ThrowUnsupported(std::string("unconsumed statement ") +
                       stmt->GetTypeKey());
    if (is_zero(evaluate->value))
      return;
    const auto *node = evaluate->value.as<CallNode>();
    if (!node)
      ThrowUnsupported("non-call Evaluate in compute dataflow");
    Call call = ffi::GetRef<Call>(node);
    if (collected_calls_) {
      collected_calls_->push_back(call);
      return;
    }
    PlanCall(call);
  }

  void PlanCall(Call call) {
    if ((IsPipeline() || transfers_) && !call->span.defined())
      call.CopyOnWrite()->span =
          RequireSourceSpan(call->span, frontend_->span, "pipeline operation");
    if (transfers_ && (call->op.same_as(tenstorrent::pipe_send()) ||
                       call->op.same_as(tenstorrent::pipe_recv()))) {
      PlanPipe(call);
    } else if (call->op.same_as(Copy::Get())) {
      PlanCopy(Downcast<Copy>(ParseOperator(call)), call);
    } else if (call->op.same_as(tenstorrent::tile_add())) {
      BufferRegion a =
          NormalizeToAccessRegion(call->args[0], kAccessRead).region;
      BufferRegion b =
          NormalizeToAccessRegion(call->args[1], kAccessRead).region;
      BufferRegion c =
          NormalizeToAccessRegion(call->args[2], kAccessWrite).region;
      ffi::Array<PrimExpr> zero(a->buffer->shape.size(), Integer(0));
      ffi::Array<Integer> identity;
      for (size_t i = 0; i < zero.size(); ++i)
        identity.push_back(Integer(i));
      auto attrs = call->annotations;
      attrs.Set("tt.compute_kind", StringImm("elementwise"));
      attrs.Set("tt.logical_domain", c->buffer->shape);
      attrs.Set("tt.access_maps",
                ffi::Array<ffi::Array<Integer>>{identity, identity});
      attrs.Set("tt.input_shapes", ffi::Array<ffi::Array<PrimExpr>>{
                                       a->buffer->shape, b->buffer->shape});
      attrs.Set("tt.expression",
                Add(BufferLoad(a->buffer, zero), BufferLoad(b->buffer, zero)));
      PlanCompute(Call(DataType::Void(), tenstorrent::tile_compute(),
                       {call->args[2], call->args[0], call->args[1]}, attrs,
                       call->span));
    } else if (call->op.same_as(tenstorrent::tile_compute())) {
      PlanCompute(call);
    } else {
      ThrowUnsupported(
          "unconsumed operation; expected legalized tile_compute or Copy");
    }
  }

  ffi::Array<ffi::String> Effects() const {
    ffi::Array<ffi::String> effects;
    for (int access : TensorAccesses()) {
      effects.push_back(access == 3   ? "inout"
                        : access == 2 ? "output"
                                      : "input");
    }
    return effects;
  }

  ffi::Array<Integer> UsedTensorIndices() const {
    ffi::Array<Integer> indices;
    auto accesses = TensorAccesses();
    for (size_t index = 0; index < accesses.size(); ++index)
      if (accesses[index])
        indices.push_back(Integer(index));
    return indices;
  }

  ffi::Array<DFBDescriptor> Descriptors(const CoreDomain &domain) const {
    ffi::Array<DFBDescriptor> result;
    for (size_t i = 0; i < resources_.size(); ++i) {
      if (!live_.count(i))
        continue;
      const ResourceVersion &resource = resources_[i];
      if (resource.consumer.empty()) {
        ThrowUnsupported(
            "DFB generation " + std::to_string(i) +
            " has no consumer (dead writes must be removed before formation)");
      }
      const TTBufferMetadata &metadata = resource.metadata;
      result.push_back(DFBDescriptor(
          i, metadata->buffer_id + ".v" + std::to_string(i),
          metadata->buffer->dtype, metadata->tile_shape,
          metadata->tile_grid_shape, Capacity(resource), resource.backing,
          resource.producer, domain, resource.consumer, domain, Integer(1),
          RequireSourceSpan(metadata->source_span, frontend_->span,
                            "DFB generation")));
    }
    return result;
  }

  Stmt ComputeBody() const { return Body(compute_); }
  Stmt TransferBody() const { return Body(transfer_); }

private:
  struct FragmentAccess {
    size_t index;
    bool read;
  };

  void PrepareFragmentAccesses(const std::vector<Call> &calls) {
    arith::Analyzer analyzer;
    for (size_t index = 0; index < calls.size(); ++index) {
      const Call &call = calls[index];
      std::unordered_map<Var, bool, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
          accesses;
      auto access = [&](const BufferRegion &region, bool read) {
        const Buffer &buffer = region->buffer;
        if (buffer.scope() != "local.fragment")
          return;
        // A partial write cannot kill the untouched portion of a value.
        if (!read) {
          if (region->region.size() != buffer->shape.size()) {
            read = true;
          } else {
            for (size_t axis = 0; axis < buffer->shape.size(); ++axis)
              read |= !analyzer.CanProveEqual(region->region[axis]->min,
                                              Integer(0)) ||
                      !analyzer.CanProveEqual(region->region[axis]->extent,
                                              buffer->shape[axis]);
          }
        }
        accesses[buffer->data] |= read;
      };
      if (call->op.same_as(tile_compute())) {
        auto kind_attr = call->annotations.Get("tt.compute_kind");
        ffi::String kind = kind_attr.has_value()
                               ? Downcast<StringImm>(kind_attr.value())->value
                               : ffi::String("");
        for (size_t arg = 0; arg < call->args.size(); ++arg) {
          bool read = kind == "accumulator_materialize" ? arg == 0
                      : kind == "gemm_update"           ? true
                                                        : arg != 0;
          access(NormalizeToAccessRegion(call->args[arg],
                                         read ? kAccessRead : kAccessWrite)
                     .region,
                 read);
        }
        auto expression = call->annotations.Get("tt.expression");
        if (expression.has_value())
          PostOrderVisit(Downcast<PrimExpr>(expression.value()),
                         [&](const ffi::ObjectRef &object) {
                           if (const auto *load = object.as<BufferLoadNode>())
                             if (load->buffer.scope() == "local.fragment")
                               accesses[load->buffer->data] = true;
                         });
      } else if (call->op.same_as(Copy::Get())) {
        Copy copy = Downcast<Copy>(ParseOperator(call));
        access(BufferRegion(copy->src, copy->src_range), true);
        access(BufferRegion(copy->dst, copy->dst_range), false);
      } else if (call->op.same_as(tile_add())) {
        for (size_t arg = 0; arg < 3; ++arg) {
          bool read = arg != 2;
          access(NormalizeToAccessRegion(call->args[arg],
                                         read ? kAccessRead : kAccessWrite)
                     .region,
                 read);
        }
      } else if (call->op.same_as(pipe_send()) ||
                 call->op.same_as(pipe_recv())) {
        // Pipe transactions touch only their explicit shared DFB operand;
        // they do not keep unrelated compute-local fragments alive.
        bool send = call->op.same_as(pipe_send());
        access(NormalizeToAccessRegion(call->args[send ? 0 : 1],
                                       send ? kAccessRead : kAccessWrite)
                   .region,
               send);
      } else {
        // Unknown operations may observe any live fragment. Keep their values
        // until that operation has executed rather than infer missing effects.
        unknown_accesses_.push_back(index);
      }
      for (const auto &[buffer, read] : accesses)
        fragment_accesses_[buffer].push_back({index, read});
    }
  }

  bool FragmentIsLive(const Var &buffer) const {
    auto unknown = std::lower_bound(unknown_accesses_.begin(),
                                    unknown_accesses_.end(), call_index_);
    auto found = fragment_accesses_.find(buffer);
    if (found == fragment_accesses_.end())
      return unknown != unknown_accesses_.end();
    const auto &accesses = found->second;
    auto next =
        std::lower_bound(accesses.begin(), accesses.end(), call_index_,
                         [](const FragmentAccess &access, size_t index) {
                           return access.index < index;
                         });
    return (unknown != unknown_accesses_.end() &&
            (next == accesses.end() || *unknown <= next->index)) ||
           (next != accesses.end() && next->read);
  }

  PrimExpr Capacity(const ResourceVersion &resource) const {
    PrimExpr count = resource.metadata->dfb_block_count.value();
    if (!IsPipeline() || resource.iteration < 0)
      return count;
    if (resource.metadata->block_count_origin == "explicit") {
      if (RequirePositiveStaticInteger(count, "DFB block_count") <
          pipeline_depth_)
        ThrowMalformed("DFB capacity insufficient for buffer '" +
                       std::string(resource.metadata->buffer_id) + "' pool " +
                       std::to_string(resource.pool) +
                       ": block_count=" + std::to_string(*as_const_int(count)) +
                       ", required=" + std::to_string(pipeline_depth_) +
                       " (bounded pipeline window)");
      return count;
    }
    return Integer(pipeline_depth_);
  }

  void AddReleases(ffi::Array<Stmt> *body, const ffi::String &slot) {
    std::unordered_map<int64_t, size_t> last_use;
    // Propagate final uses backwards through the immutable value DAG. Storing
    // every transitive borrowed DFB set would grow quadratically for online
    // reductions whose result depends on all preceding serial iterations.
    std::unordered_map<int64_t, std::vector<int64_t>> borrowed_sources;
    std::unordered_map<int64_t, std::vector<int64_t>> value_parents;
    std::unordered_map<int64_t, size_t> last_value_use;
    if (compute_value_mode_) {
      for (const Stmt &statement : compute_) {
        const CallNode *call =
            statement.as<EvaluateNode>()->value.as<CallNode>();
        if (!call->op.same_as(compute_value()) &&
            !call->op.same_as(compute_value_gemm()))
          continue;
        int64_t output = *as_const_int(call->args[0]);
        auto &sources = borrowed_sources[output];
        auto &parents = value_parents[output];
        if (call->op.same_as(compute_value_gemm())) {
          sources.push_back(*as_const_int(call->args[1]));
          sources.push_back(*as_const_int(call->args[2]));
          int64_t old = *as_const_int(call->args[3]);
          if (old >= 0)
            parents.push_back(old);
        } else {
          for (size_t arg = 1; arg < call->args.size(); ++arg)
            sources.push_back(*as_const_int(call->args[arg]));
          auto inputs = call->annotations.Get("tt.value_inputs");
          if (inputs.has_value())
            for (const Integer &input :
                 Downcast<ffi::Array<Integer>>(inputs.value()))
              parents.push_back(input->value);
        }
      }
    }
    for (size_t index = 0; index < body->size(); ++index) {
      const auto *call =
          (*body)[index].as<EvaluateNode>()->value.as<CallNode>();
      if (compute_value_mode_) {
        auto use_value = [&](int64_t value) { last_value_use[value] = index; };
        if (call->op.same_as(compute_value()) ||
            call->op.same_as(compute_value_gemm()))
          use_value(*as_const_int(call->args[0]));
        if (call->op.same_as(compute_value_store()))
          use_value(*as_const_int(call->args[0]));
        auto value_inputs = call->annotations.Get("tt.value_inputs");
        if (value_inputs.has_value())
          for (const Integer &input :
               Downcast<ffi::Array<Integer>>(value_inputs.value()))
            use_value(input->value);
      }
      if (call->op.same_as(tenstorrent::dfb_compute()) ||
          call->op.same_as(tenstorrent::compute_value())) {
        for (size_t arg = 1; arg < call->args.size(); ++arg)
          last_use[*as_const_int(call->args[arg])] = index;
      } else if (call->op.same_as(tenstorrent::compute_value_gemm())) {
        last_use[*as_const_int(call->args[1])] = index;
        last_use[*as_const_int(call->args[2])] = index;
      } else if (compute_value_mode_ &&
                 call->op.same_as(tenstorrent::dfb_to_tensor_nd())) {
        last_use[*as_const_int(call->args[0])] = index;
      } else if (call->op.same_as(tenstorrent::gemm_update())) {
        last_use[*as_const_int(call->args[0])] = index;
        last_use[*as_const_int(call->args[1])] = index;
      } else if (call->op.same_as(tenstorrent::dfb_pipe_wait())) {
        int64_t id = *as_const_int(call->args[1]);
        if (resources_[id].consumer == slot)
          last_use[id] = index;
      } else if (call->op.same_as(tenstorrent::dfb_wait()) && transfers_) {
        int64_t id = *as_const_int(call->args[0]);
        if (resources_[id].consumer == slot)
          last_use[id] = index;
      } else if (call->op.same_as(tenstorrent::dfb_copy_wait())) {
        int64_t id = *as_const_int(call->args[0]);
        if (resources_[id].consumer == slot)
          last_use[id] = index;
      }
    }
    // Value IDs are assigned in definition order, so every parent precedes
    // its children. One reverse pass is sufficient, including branched DAGs.
    for (int64_t value = static_cast<int64_t>(values_.size()) - 1; value >= 0;
         --value) {
      auto found = last_value_use.find(value);
      if (found == last_value_use.end())
        continue;
      size_t use = found->second;
      for (int64_t source : borrowed_sources[value])
        if (resources_[source].consumer == slot)
          last_use[source] = std::max(last_use[source], use);
      for (int64_t parent : value_parents[value])
        last_value_use[parent] = std::max(last_value_use[parent], use);
    }
    std::vector<std::vector<int64_t>> releases(body->size());
    // Resource order makes releases deterministic, independent of hash order.
    for (size_t id = 0; id < resources_.size(); ++id) {
      auto found = last_use.find(id);
      if (found != last_use.end())
        releases[found->second].push_back(id);
    }
    ffi::Array<Stmt> result;
    for (size_t index = 0; index < body->size(); ++index) {
      const Stmt &statement = (*body)[index];
      result.push_back(statement);
      for (int64_t id : releases[index])
        result.push_back(MakeDeviceCall(tenstorrent::dfb_release(),
                                        {Integer(id), Integer(1)},
                                        statement->span));
    }
    *body = std::move(result);
  }

  std::vector<int> TensorAccesses() const {
    std::vector<int> accesses(frontend_->params.size(), 0);
    for (const Stmt &statement : transfer_) {
      const auto *call = statement.as<EvaluateNode>()->value.as<CallNode>();
      if (call->op.same_as(tenstorrent::tensor_to_dfb_nd()))
        accesses[*as_const_int(call->args[0])] |= 1;
      if (call->op.same_as(tenstorrent::dfb_to_tensor_nd()))
        accesses[*as_const_int(call->args[1])] |= 2;
    }
    return accesses;
  }

  Stmt Body(const ffi::Array<Stmt> &statements) const {
    if (statements.empty())
      return Evaluate(Integer(0), frontend_->span);
    if (statements.size() == 1)
      return statements[0];
    return SeqStmt(statements, frontend_->span);
  }

  void FullRegion(const BufferRegion &region) const {
    if (region->region.size() != region->buffer->shape.size())
      ThrowMalformed("region rank disagrees with buffer rank");
    for (size_t i = 0; i < region->region.size(); ++i) {
      arith::Analyzer analyzer;
      if (!analyzer.CanProveEqual(region->region[i]->min, Integer(0)) ||
          !analyzer.CanProveEqual(region->region[i]->extent,
                                  region->buffer->shape[i]))
        ThrowUnsupported("Phase 4 dataflow requires complete Buffer regions; "
                         "partial alias/view is unsupported");
    }
  }

  int64_t Read(const Buffer &buffer, const ffi::String &consumer) {
    auto found = current_.find(buffer->data);
    if (found == current_.end())
      ThrowMalformed("read-before-write of DFB buffer '" +
                     std::string(buffer->name) + "'");
    ResourceVersion &resource = resources_[found->second];
    if (resource.metadata->buffer->dtype != buffer->dtype ||
        !ffi::StructuralEqual()(resource.metadata->buffer->shape,
                                buffer->shape))
      ThrowUnsupported("alias must preserve full Buffer shape and dtype");
    if (!resource.consumer.empty() && resource.consumer != consumer)
      ThrowUnsupported("DFB generation has consumers in multiple slots");
    resource.consumer = consumer;
    return found->second;
  }

  int64_t Write(const Buffer &buffer, const ffi::String &producer,
                ffi::Optional<TensorBacking> backing = std::nullopt,
                bool publish_current = true) {
    const TTBufferMetadata &metadata =
        RequireMetadata(metadata_, buffer, "DFB write");
    if (metadata->kind != "logical_dfb_candidate" ||
        !metadata->dfb_block_count.has_value())
      ThrowUnsupported(
          "compute and dataflow destinations must be shared DFB buffers");
    RequirePositiveStaticInteger(metadata->dfb_block_count.value(),
                                 "DFB block count");
    int64_t id = resources_.size();
    resources_.push_back({metadata, producer, "", backing});
    if (publish_current)
      current_[buffer->data] = id;
    return id;
  }

  void Wait(ffi::Array<Stmt> *body, int64_t id, const Span &span) {
    body->push_back(MakeDeviceCall(tenstorrent::dfb_wait(),
                                   {Integer(id), Integer(1)}, span));
  }
  void Reserve(ffi::Array<Stmt> *body, int64_t id, const Span &span) {
    body->push_back(MakeDeviceCall(tenstorrent::dfb_reserve(),
                                   {Integer(id), Integer(1)}, span));
  }

  int64_t ReadValue(const Buffer &buffer) const {
    auto found = current_values_.find(buffer->data);
    if (found == current_values_.end())
      ThrowMalformed("fragment read-before-definition for buffer '" +
                     std::string(buffer->name) + "'");
    if (IsPipeline() && found->second < pipeline_value_begin_)
      ThrowUnsupported("T.Pipelined fragment '" + std::string(buffer->name) +
                       "' carries a value across iterations; bounded prefetch "
                       "requires independently initialized fragment values");
    const Buffer &defined = values_[found->second]->buffer;
    if (defined->dtype != buffer->dtype ||
        !ffi::StructuralEqual()(defined->shape, buffer->shape))
      ThrowMalformed("fragment alias must preserve shape and dtype");
    return found->second;
  }

  int64_t DefineValue(const Buffer &buffer, int64_t accumulator,
                      const Span &span) {
    auto found = current_values_.find(buffer->data);
    int64_t previous = found == current_values_.end() ? -1 : found->second;
    int64_t version = previous < 0 ? 0 : values_[previous]->version + 1;
    int64_t id = values_.size();
    values_.push_back(ComputeValueDescriptor(
        id, buffer, version, previous, accumulator,
        RequireSourceSpan(span, frontend_->span, "compute value")));
    value_roots_.push_back({});
    if (accumulator >= 0)
      value_roots_.back().insert(accumulator);
    current_values_[buffer->data] = id;
    return id;
  }

  static ffi::Array<Integer> IdentityMap(size_t rank) {
    ffi::Array<Integer> map;
    for (size_t axis = 0; axis < rank; ++axis)
      map.push_back(Integer(axis));
    return map;
  }

  int64_t
  MakeValueStorage(const Buffer &buffer, const Span &span,
                   ffi::Optional<TensorBacking> backing = std::nullopt) {
    Buffer storage = decl_buffer(buffer->shape, buffer->dtype,
                                 buffer->name + "_materialized", "shared",
                                 std::nullopt, span);
    ffi::Array<PrimExpr> grid;
    for (const PrimExpr &extent : buffer->shape)
      grid.push_back(Integer((RequirePositiveStaticInteger(
                                  extent, "compute materialization extent") +
                              31) /
                             32));
    int64_t id = resources_.size();
    TTBufferMetadata metadata(
        "compute.materialization." + std::to_string(id), storage,
        "logical_dfb_candidate", std::nullopt, {Integer(32), Integer(32)}, grid,
        "interleaved", std::nullopt, Integer(1), std::nullopt, std::nullopt,
        "inferred", "inferred", "inferred", "inferred", span);
    resources_.push_back({metadata, "trisc", "", backing});
    return id;
  }

  int64_t MaterializeValue(int64_t value, const Span &span) {
    auto found = materialized_values_.find(value);
    if (found != materialized_values_.end())
      return found->second;
    int64_t id = MakeValueStorage(values_[value]->buffer, span);
    resources_[id].consumer = "trisc";
    Reserve(&compute_, id, span);
    compute_.push_back(MakeDeviceCall(tenstorrent::compute_value_store(),
                                      {Integer(value), Integer(id)}, span));
    materialized_values_[value] = id;
    dfb_roots_[id] = value_roots_[value];
    return id;
  }

  int64_t ReadDFBOperand(const Buffer &buffer, const Span &span) {
    if (buffer.scope() == "local.fragment")
      return MaterializeValue(ReadValue(buffer), span);
    return Read(buffer, "trisc");
  }

  void PlanValueAccumulator(const Call &call, const ffi::String &kind) {
    BufferRegion region =
        NormalizeToAccessRegion(call->args[0], kAccessReadWrite).region;
    FullRegion(region);
    const Buffer &buffer = region->buffer;
    auto found = accumulator_ids_.find(buffer);
    if (kind == "accumulator_init") {
      int64_t accumulator = accumulators_.size();
      accumulator_ids_[buffer] = accumulator;
      AccumulatorLifetime state;
      state.region = region;
      state.span = call->span;
      // The frontend proves the final storage dtype across all epilogue uses.
      auto requirements =
          frontend_->GetAttr<ffi::Array<ffi::Map<ffi::String, ffi::ObjectRef>>>(
              "tt.gemm_accumulator_requirements");
      if (requirements.has_value()) {
        for (const auto &requirement : requirements.value()) {
          if (Downcast<Buffer>(requirement.at("accumulator"))
                  ->data.same_as(buffer->data)) {
            auto dtype = requirement.Get("output_dtype");
            if (dtype.has_value())
              state.output_dtype =
                  Downcast<StringImm>(dtype.value())->value == "float32"
                      ? DataType::Float(32)
                      : DataType::BFloat(16);
          }
        }
      }
      if (state.output_dtype == DataType::Void())
        state.output_dtype = buffer->dtype;
      accumulators_.push_back(std::move(state));
      int64_t value = DefineValue(buffer, accumulator, call->span);
      ffi::Map<ffi::String, ffi::ObjectRef> attrs{
          {"tt.compute_kind", StringImm("fill")},
          {"tt.compute_dtype",
           StringImm(buffer->dtype == DataType::Float(32) ? "float32"
                                                          : "bfloat16")},
          {"tt.compute_tile_shape",
           ffi::Array<PrimExpr>{Integer(32), Integer(32)}},
          {"tt.logical_domain", buffer->shape},
          {"tt.expression", make_zero(buffer->dtype)},
          {"tt.access_maps", ffi::Array<ffi::Array<Integer>>()},
          {"tt.input_shapes", ffi::Array<ffi::Array<PrimExpr>>()},
          {"tt.value_inputs", ffi::Array<Integer>()},
          {"tt.value_access_maps", ffi::Array<ffi::Array<Integer>>()},
          {"tt.value_input_shapes", ffi::Array<ffi::Array<PrimExpr>>()}};
      compute_.push_back(Evaluate(Call(DataType::Void(), compute_value(),
                                       {Integer(value)}, attrs, call->span),
                                  call->span));
      return;
    }
    if (found == accumulator_ids_.end())
      ThrowMalformed("accumulator update requires a dominating initialization");
    int64_t accumulator = found->second;
    if (kind == "accumulator_materialize") {
      BufferRegion output =
          NormalizeToAccessRegion(call->args[1], kAccessWrite).region;
      ffi::Array<PrimExpr> zeros(buffer->shape.size(), Integer(0));
      ffi::Map<ffi::String, ffi::ObjectRef> attrs{
          {"tt.compute_kind", StringImm("copy")},
          {"tt.compute_dtype",
           StringImm(output->buffer->dtype == DataType::Float(32)
                         ? "float32"
                         : "bfloat16")},
          {"tt.compute_tile_shape",
           ffi::Array<PrimExpr>{Integer(32), Integer(32)}},
          {"tt.expression",
           cast(output->buffer->dtype, BufferLoad(buffer, zeros))}};
      PlanValueCompute(Call(DataType::Void(), tile_compute(),
                            {output->ToPrimExpr(), region->ToPrimExpr()}, attrs,
                            call->span));
      return;
    }
    BufferRegion lhs =
        NormalizeToAccessRegion(call->args[1], kAccessRead).region;
    BufferRegion rhs =
        NormalizeToAccessRegion(call->args[2], kAccessRead).region;
    FullRegion(lhs);
    FullRegion(rhs);
    int64_t a = ReadDFBOperand(lhs->buffer, call->span);
    int64_t b = ReadDFBOperand(rhs->buffer, call->span);
    int64_t ta =
        Downcast<Integer>(call->annotations.at("tt.transpose_a"))->value;
    int64_t tb =
        Downcast<Integer>(call->annotations.at("tt.transpose_b"))->value;
    int64_t old = ReadValue(buffer);
    int64_t value = DefineValue(buffer, accumulator, call->span);
    value_roots_[value] = value_roots_[old];
    value_roots_[value].insert(accumulator);
    auto &state = accumulators_[accumulator];
    state.input_dtype = lhs->buffer->dtype;
    state.full_k_tiles +=
        RequirePositiveStaticInteger(lhs->buffer->shape[ta ? 0 : 1], "GEMM K") /
        32;
    state.inputs.push_back(a);
    state.inputs.push_back(b);
    Wait(&compute_, a, call->span);
    Wait(&compute_, b, call->span);
    compute_.push_back(MakeDeviceCall(compute_value_gemm(),
                                      {Integer(value), Integer(a), Integer(b),
                                       Integer(old), Integer(ta), Integer(tb)},
                                      call->span));
  }

  void EnsurePrecision(const Call &call, const ffi::String &kind) {
    if (!precision_regions_)
      return;
    if (IsPipeline())
      ThrowUnsupported(
          "precision regions require statically serialized transactions");
    int mode = precision_mode_ < 0 ? 0 : precision_mode_;
    BufferRegion output =
        NormalizeToAccessRegion(call->args[0], kAccessWrite).region;
    bool fp32 = output->buffer->dtype == DataType::Float(32);
    auto expression = call->annotations.Get("tt.expression");
    if (expression.has_value())
      PostOrderVisit(Downcast<PrimExpr>(expression.value()),
                     [&](const ffi::ObjectRef &item) {
                       if (auto value = item.as<PrimExpr>())
                         fp32 |= value.value().dtype() == DataType::Float(32);
                     });
    if (kind == "accumulator_init" || kind == "gemm_update")
      mode = output->buffer->dtype == DataType::Float(32) ? 1 : 0;
    else if (fp32)
      mode = 1;
    if (mode == precision_mode_)
      return;
    std::vector<int64_t> crossing;
    if (precision_mode_ >= 0) {
      for (const auto &[buffer, value] : current_values_)
        if (FragmentIsLive(buffer))
          crossing.push_back(value);
      std::sort(crossing.begin(), crossing.end());
    }
    std::vector<int64_t> storage;
    for (int64_t value : crossing)
      storage.push_back(MaterializeValue(value, call->span));
    compute_.push_back(
        MakeDeviceCall(compute_precision(), {Integer(mode)}, call->span));
    precision_mode_ = mode;
    // A new precision region may not refer to an earlier compute SSA value.
    // Re-enter through exact-dtype DFB snapshots with new immutable versions.
    for (size_t i = 0; i < crossing.size(); ++i) {
      int64_t old = crossing[i], dfb = storage[i];
      const Buffer buffer = values_[old]->buffer;
      int64_t value =
          DefineValue(buffer, values_[old]->accumulator_id, call->span);
      value_roots_[value] = value_roots_[old];
      ffi::Map<ffi::String, ffi::ObjectRef> attrs{
          {"tt.compute_kind", StringImm("elementwise")},
          {"tt.compute_dtype",
           StringImm(buffer->dtype == DataType::Float(32) ? "float32"
                                                          : "bfloat16")},
          {"tt.compute_tile_shape",
           ffi::Array<PrimExpr>{Integer(32), Integer(32)}},
          {"tt.logical_domain", buffer->shape},
          {"tt.expression",
           Call(buffer->dtype, dfb_load(), {Integer(dfb)}, {}, call->span)},
          {"tt.access_maps",
           ffi::Array<ffi::Array<Integer>>{IdentityMap(buffer->shape.size())}},
          {"tt.input_shapes", ffi::Array<ffi::Array<PrimExpr>>{buffer->shape}},
          {"tt.value_inputs", ffi::Array<Integer>()},
          {"tt.value_access_maps", ffi::Array<ffi::Array<Integer>>()},
          {"tt.value_input_shapes", ffi::Array<ffi::Array<PrimExpr>>()}};
      Wait(&compute_, dfb, call->span);
      compute_.push_back(
          Evaluate(Call(DataType::Void(), compute_value(),
                        {Integer(value), Integer(dfb)}, attrs, call->span),
                   call->span));
    }
  }

  void PlanValueCompute(const Call &call) {
    ffi::String kind =
        Downcast<StringImm>(call->annotations.at("tt.compute_kind"))->value;
    EnsurePrecision(call, kind);
    if (kind == "accumulator_init" || kind == "gemm_update" ||
        kind == "accumulator_materialize") {
      PlanValueAccumulator(call, kind);
      return;
    }
    BufferRegion output =
        NormalizeToAccessRegion(call->args[0], kAccessWrite).region;
    const auto &output_metadata =
        RequireMetadata(metadata_, output->buffer, "compute output");
    bool global = output_metadata->kind == "tensor";
    if (!global)
      FullRegion(output);
    if (global && kind != "copy" && kind != "typecast")
      ThrowMalformed(
          "only an explicit compute copy may materialize to a Tensor");
    ffi::Array<PrimExpr> domain;
    for (const Range &range : output->region)
      domain.push_back(range->extent);
    // Tensor coordinates retain the public ABI, while computation uses the
    // local operand shape after dropping only proven leading unit axes.
    if (global && call->args.size() == 2) {
      BufferRegion input =
          NormalizeToAccessRegion(call->args[1], kAccessRead).region;
      if (domain.size() > input->region.size()) {
        size_t offset = domain.size() - input->region.size();
        arith::Analyzer analyzer;
        for (size_t axis = 0; axis < offset; ++axis)
          if (!analyzer.CanProveEqual(domain[axis], Integer(1)))
            ThrowUnsupported("Tensor export may only omit leading unit axes");
        ffi::Array<PrimExpr> local_domain;
        for (size_t axis = offset; axis < domain.size(); ++axis) {
          if (!analyzer.CanProveEqual(domain[axis],
                                      input->region[axis - offset]->extent))
            ThrowUnsupported("Tensor export requires matching local extents");
          local_domain.push_back(domain[axis]);
        }
        domain = std::move(local_domain);
      }
    }
    ffi::Array<PrimExpr> dfb_args;
    ffi::Array<Integer> value_inputs;
    ffi::Array<ffi::Array<Integer>> dfb_maps, value_maps;
    ffi::Array<ffi::Array<PrimExpr>> dfb_shapes, value_shapes;
    std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
        dfb_inputs, local_inputs;
    auto original_maps = call->annotations.Get("tt.access_maps");
    ffi::Array<ffi::Array<Integer>> maps =
        original_maps.has_value()
            ? Downcast<ffi::Array<ffi::Array<Integer>>>(original_maps.value())
            : ffi::Array<ffi::Array<Integer>>();
    bool dfb_only = kind == "gemm" || kind == "transpose" || kind == "reduce";
    std::unordered_set<int64_t> roots;
    for (size_t index = 1; index < call->args.size(); ++index) {
      BufferRegion input =
          NormalizeToAccessRegion(call->args[index], kAccessRead).region;
      FullRegion(input);
      ffi::Array<Integer> map = index <= maps.size()
                                    ? maps[index - 1]
                                    : IdentityMap(input->buffer->shape.size());
      if (input->buffer.scope() == "local.fragment" && !dfb_only) {
        int64_t value = ReadValue(input->buffer);
        local_inputs.emplace(input->buffer, value);
        value_inputs.push_back(Integer(value));
        value_maps.push_back(map);
        value_shapes.push_back(input->buffer->shape);
        roots.insert(value_roots_[value].begin(), value_roots_[value].end());
      } else {
        int64_t id = ReadDFBOperand(input->buffer, call->span);
        dfb_inputs.emplace(input->buffer, id);
        dfb_args.push_back(Integer(id));
        dfb_maps.push_back(map);
        dfb_shapes.push_back(input->buffer->shape);
        const auto &input_roots = dfb_roots_[id];
        roots.insert(input_roots.begin(), input_roots.end());
        Wait(&compute_, id, call->span);
      }
    }
    // Def-use edges are authoritative for derived values with several roots.
    // An in-place accumulator update still retains its own accumulator
    // identity.
    int64_t provenance = roots.size() == 1 ? *roots.begin() : -1;
    auto own_accumulator = accumulator_ids_.find(output->buffer);
    if (own_accumulator != accumulator_ids_.end())
      provenance = own_accumulator->second;
    auto attrs = call->annotations;
    if (kind == "copy")
      attrs.Set("tt.compute_kind", StringImm("elementwise"));
    attrs.Set("tt.logical_domain", domain);
    attrs.Set("tt.access_maps", dfb_maps);
    attrs.Set("tt.input_shapes", dfb_shapes);
    attrs.Set("tt.value_inputs", value_inputs);
    attrs.Set("tt.value_access_maps", value_maps);
    attrs.Set("tt.value_input_shapes", value_shapes);
    auto expression = attrs.Get("tt.expression");
    if (expression.has_value()) {
      DeviceExpressionRewriter rewriter(dfb_inputs, local_inputs);
      attrs.Set("tt.expression",
                rewriter(Downcast<PrimExpr>(expression.value())));
    }
    if (output->buffer.scope() == "local.fragment") {
      int64_t value = DefineValue(output->buffer, provenance, call->span);
      value_roots_[value] = roots;
      if (own_accumulator != accumulator_ids_.end())
        value_roots_[value].insert(own_accumulator->second);
      if (dfb_only) {
        // DFB-only operators retain their established operand/result contract.
        // Re-enter the value domain through an exact-dtype identity load; the
        // borrowed DFB stays live through every transitive value consumer.
        int64_t storage = MakeValueStorage(output->buffer, call->span);
        dfb_roots_[storage] = roots;
        resources_[storage].consumer = "trisc";
        Reserve(&compute_, storage, call->span);
        ffi::Array<PrimExpr> operation_args{Integer(storage)};
        for (const PrimExpr &arg : dfb_args)
          operation_args.push_back(arg);
        compute_.push_back(Evaluate(Call(DataType::Void(), dfb_compute(),
                                         operation_args, attrs, call->span),
                                    call->span));
        Wait(&compute_, storage, call->span);
        ffi::Map<ffi::String, ffi::ObjectRef> load_attrs{
            {"tt.compute_kind", StringImm("elementwise")},
            {"tt.compute_dtype", attrs.at("tt.compute_dtype")},
            {"tt.compute_tile_shape", attrs.at("tt.compute_tile_shape")},
            {"tt.logical_domain", domain},
            {"tt.expression", Call(output->buffer->dtype, dfb_load(),
                                   {Integer(storage)}, {}, call->span)},
            {"tt.access_maps",
             ffi::Array<ffi::Array<Integer>>{IdentityMap(domain.size())}},
            {"tt.input_shapes", ffi::Array<ffi::Array<PrimExpr>>{domain}},
            {"tt.value_inputs", ffi::Array<Integer>()},
            {"tt.value_access_maps", ffi::Array<ffi::Array<Integer>>()},
            {"tt.value_input_shapes", ffi::Array<ffi::Array<PrimExpr>>()}};
        compute_.push_back(Evaluate(Call(DataType::Void(), compute_value(),
                                         {Integer(value), Integer(storage)},
                                         load_attrs, call->span),
                                    call->span));
        return;
      }
      ffi::Array<PrimExpr> args{Integer(value)};
      for (const PrimExpr &arg : dfb_args)
        args.push_back(arg);
      compute_.push_back(Evaluate(
          Call(DataType::Void(), compute_value(), args, attrs, call->span),
          call->span));
      return;
    }
    int64_t output_id;
    if (global) {
      Buffer storage_shape =
          decl_buffer(domain, output->buffer->dtype,
                      output->buffer->name + "_export", "local.fragment");
      int64_t tensor = output_metadata->global_arg_index.value()->value;
      tensor_writes_[tensor] = true;
      output_id = MakeValueStorage(storage_shape, call->span,
                                   TensorBacking(tensor, Integer(0)));
      resources_[output_id].consumer = "ncrisc";
      Wait(&transfer_, output_id, call->span);
      ffi::Array<PrimExpr> args{Integer(output_id), Integer(tensor)};
      for (const Range &range : output->region) {
        args.push_back(range->min);
        args.push_back(range->extent);
      }
      transfer_.push_back(MakeDeviceCall(dfb_to_tensor_nd(), args, call->span));
      if (transfers_)
        transfer_.push_back(MakeDeviceCall(
            dfb_copy_wait(), {Integer(output_id), Integer(1)}, call->span));
    } else {
      output_id = Write(output->buffer, "trisc");
    }
    dfb_roots_[output_id] = roots;
    Reserve(&compute_, output_id, call->span);
    ffi::Array<PrimExpr> args{Integer(output_id)};
    for (const PrimExpr &arg : dfb_args)
      args.push_back(arg);
    compute_.push_back(
        Evaluate(Call(DataType::Void(), dfb_compute(), args, attrs, call->span),
                 call->span));
  }

  void EliminateDeadValueWrites() {
    std::unordered_map<int64_t, std::vector<int64_t>> inputs;
    std::unordered_set<int64_t> live;
    // Compute values are immutable definitions. Retain their predecessor chains
    // and all input lifetimes even when a value only feeds another fragment.
    for (const Stmt &stmt : compute_) {
      const CallNode *call = stmt.as<EvaluateNode>()->value.as<CallNode>();
      if (call->op.same_as(dfb_compute())) {
        int64_t out = *as_const_int(call->args[0]);
        for (size_t i = 1; i < call->args.size(); ++i)
          inputs[out].push_back(*as_const_int(call->args[i]));
      }
    }
    std::function<void(int64_t)> mark = [&](int64_t id) {
      if (!live.insert(id).second)
        return;
      for (int64_t input : inputs[id])
        mark(input);
    };
    for (const Stmt &stmt : compute_) {
      const CallNode *call = stmt.as<EvaluateNode>()->value.as<CallNode>();
      if (call->op.same_as(compute_value())) {
        for (size_t i = 1; i < call->args.size(); ++i)
          mark(*as_const_int(call->args[i]));
      } else if (call->op.same_as(compute_value_gemm())) {
        mark(*as_const_int(call->args[1]));
        mark(*as_const_int(call->args[2]));
      }
    }
    for (const Stmt &stmt : transfer_) {
      const CallNode *call = stmt.as<EvaluateNode>()->value.as<CallNode>();
      if (call->op.same_as(dfb_to_tensor_nd()))
        mark(*as_const_int(call->args[0]));
    }
    for (const PipeEndpoint &endpoint : endpoints_)
      mark(endpoint.dfb_id);
    auto filter = [&](const ffi::Array<Stmt> &body) {
      ffi::Array<Stmt> result;
      for (const Stmt &stmt : body) {
        const CallNode *call = stmt.as<EvaluateNode>()->value.as<CallNode>();
        if (call->op.same_as(compute_value()) ||
            call->op.same_as(compute_value_gemm()) ||
            call->op.same_as(compute_precision())) {
          result.push_back(stmt);
          continue;
        }
        size_t arg = call->op.same_as(tensor_to_dfb_nd()) ||
                             call->op.same_as(compute_value_store()) ||
                             call->op.same_as(dfb_pipe_send()) ||
                             call->op.same_as(dfb_pipe_recv()) ||
                             call->op.same_as(dfb_pipe_wait())
                         ? 1
                         : 0;
        if (live.count(*as_const_int(call->args[arg])))
          result.push_back(stmt);
      }
      return result;
    };
    compute_ = filter(compute_);
    transfer_ = filter(transfer_);
    live_ = std::move(live);
  }

  void PlanCompute(const Call &call) {
    if (compute_value_mode_) {
      PlanValueCompute(call);
      return;
    }
    auto kind = call->annotations.Get("tt.compute_kind");
    if (kind.has_value()) {
      auto name = Downcast<StringImm>(kind.value())->value;
      if (name == "accumulator_init" || name == "gemm_update" ||
          name == "accumulator_materialize") {
        PlanAccumulator(call, name);
        return;
      }
    }
    if (call->args.empty())
      ThrowMalformed("tile_compute has no output region");
    BufferRegion output =
        NormalizeToAccessRegion(call->args[0], kAccessWrite).region;
    FullRegion(output);
    ffi::Array<PrimExpr> args;
    std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
        inputs;
    for (size_t i = 1; i < call->args.size(); ++i) {
      BufferRegion input =
          NormalizeToAccessRegion(call->args[i], kAccessRead).region;
      FullRegion(input);
      int64_t id = Read(input->buffer, "trisc");
      inputs.emplace(input->buffer, id);
      args.push_back(Integer(id));
      Wait(&compute_, id, call->span);
    }
    int64_t output_id = Write(output->buffer, "trisc");
    Reserve(&compute_, output_id, call->span);
    ffi::Array<PrimExpr> device_args{Integer(output_id)};
    for (const PrimExpr &arg : args)
      device_args.push_back(arg);
    auto annotations = call->annotations;
    auto expression = annotations.Get("tt.expression");
    if (expression.has_value()) {
      DeviceExpressionRewriter rewriter(inputs);
      annotations.Set("tt.expression",
                      rewriter(Downcast<PrimExpr>(expression.value())));
    }
    compute_.push_back(
        Evaluate(Call(DataType::Void(), tenstorrent::dfb_compute(), device_args,
                      annotations, call->span),
                 call->span));
  }

  void PlanAccumulator(const Call &call, const ffi::String &kind) {
    BufferRegion region =
        NormalizeToAccessRegion(call->args[0], kAccessReadWrite).region;
    FullRegion(region);
    const Buffer &buffer = region->buffer;
    auto found = accumulator_ids_.find(buffer);
    if (kind == "accumulator_init") {
      if (found != accumulator_ids_.end())
        ThrowMalformed("accumulator initialization must be unique");
      if (RequireMetadata(metadata_, buffer, "accumulator")->kind !=
          "compute_fragment")
        ThrowMalformed("accumulator must reference compute_fragment metadata");
      int64_t id = accumulators_.size();
      accumulator_ids_.emplace(buffer, id);
      AccumulatorLifetime state;
      state.region = region;
      state.span = call->span;
      accumulators_.push_back(std::move(state));
      compute_.push_back(MakeDeviceCall(tenstorrent::accumulator_init(),
                                        {Integer(id)}, call->span));
      return;
    }
    if (found == accumulator_ids_.end())
      ThrowMalformed(
          "accumulator update/materialization requires initialization");
    int64_t id = found->second;
    auto &state = accumulators_[id];
    if (state.output_id >= 0)
      ThrowMalformed("accumulator access after final materialization");
    if (kind == "gemm_update") {
      BufferRegion lhs =
          NormalizeToAccessRegion(call->args[1], kAccessRead).region;
      BufferRegion rhs =
          NormalizeToAccessRegion(call->args[2], kAccessRead).region;
      FullRegion(lhs);
      FullRegion(rhs);
      int64_t a = Read(lhs->buffer, "trisc"), b = Read(rhs->buffer, "trisc");
      int64_t transpose_a =
          Downcast<Integer>(call->annotations.at("tt.transpose_a"))->value;
      int64_t transpose_b =
          Downcast<Integer>(call->annotations.at("tt.transpose_b"))->value;
      state.input_dtype = lhs->buffer->dtype;
      state.full_k_tiles +=
          RequirePositiveStaticInteger(lhs->buffer->shape[transpose_a ? 0 : 1],
                                       "GEMM K") /
          32;
      state.inputs.push_back(a);
      state.inputs.push_back(b);
      Wait(&compute_, a, call->span);
      Wait(&compute_, b, call->span);
      compute_.push_back(
          MakeDeviceCall(tenstorrent::gemm_update(),
                         {Integer(a), Integer(b), Integer(id),
                          Integer(transpose_a), Integer(transpose_b)},
                         call->span));
      return;
    }
    if (state.full_k_tiles <= 0)
      ThrowMalformed("accumulator cannot materialize before GEMM updates");
    BufferRegion output =
        NormalizeToAccessRegion(call->args[1], kAccessWrite).region;
    const auto &metadata =
        RequireMetadata(metadata_, output->buffer, "accumulator output");
    int64_t output_id;
    if (metadata->kind == "logical_dfb_candidate") {
      FullRegion(output);
      output_id = Write(output->buffer, "trisc");
    } else if (metadata->kind == "tensor") {
      // Materialization allocates output storage only once, after complete K.
      Buffer storage = decl_buffer(buffer->shape, output->buffer->dtype,
                                   buffer->name + "_materialized", "shared",
                                   std::nullopt, call->span);
      ICHECK(!storage->data.same_as(buffer->data));
      const auto &fragment = RequireMetadata(metadata_, buffer, "accumulator");
      TTBufferMetadata output_metadata(
          fragment->buffer_id + ".materialized", storage,
          "logical_dfb_candidate", std::nullopt, fragment->tile_shape,
          fragment->tile_grid_shape, "interleaved", std::nullopt, Integer(1),
          std::nullopt, std::nullopt, "inferred", "inferred", "inferred",
          "inferred", call->span);
      int64_t tensor = metadata->global_arg_index.value()->value;
      output_id = resources_.size();
      resources_.push_back({output_metadata, "trisc", "ncrisc",
                            TensorBacking(tensor, Integer(0))});
      tensor_writes_[tensor] = true;
      Wait(&transfer_, output_id, call->span);
      ffi::Array<PrimExpr> args{Integer(output_id), Integer(tensor)};
      for (const Range &range : output->region) {
        args.push_back(range->min);
        args.push_back(range->extent);
      }
      transfer_.push_back(
          MakeDeviceCall(tenstorrent::dfb_to_tensor_nd(), args, call->span));
      if (transfers_)
        transfer_.push_back(MakeDeviceCall(tenstorrent::dfb_copy_wait(),
                                           {Integer(output_id), Integer(1)},
                                           call->span));
    } else {
      ThrowMalformed("accumulator must materialize to shared DFB or Tensor");
    }
    state.output_id = output_id;
    state.output_dtype = output->buffer->dtype;
    Reserve(&compute_, output_id, call->span);
    compute_.push_back(MakeDeviceCall(tenstorrent::accumulator_materialize(),
                                      {Integer(id), Integer(output_id)},
                                      call->span));
  }

  void PlanCopy(const Copy &copy, const Call &call) {
    BufferRegion source(copy->src, copy->src_range);
    BufferRegion destination(copy->dst, copy->dst_range);
    bool has_accumulator =
        frontend_->attrs->dict.count("tt.gemm_accumulator_requirements");
    bool rank_mapped_tensor =
        source->region.size() != destination->region.size() &&
        (copy->src.scope() == "global" || copy->dst.scope() == "global");
    if ((!transfers_ && !has_accumulator && !IsPipeline() &&
         !rank_mapped_tensor) ||
        copy->src.scope() == "shared" || copy->src.scope() == "shared.dyn")
      FullRegion(source);
    if ((!transfers_ && !has_accumulator && !IsPipeline() &&
         !rank_mapped_tensor) ||
        copy->dst.scope() == "shared" || copy->dst.scope() == "shared.dyn")
      FullRegion(destination);
    const TTBufferMetadata &src =
        RequireMetadata(metadata_, copy->src, "copy source");
    const TTBufferMetadata &dst =
        RequireMetadata(metadata_, copy->dst, "copy destination");
    if (copy->src->dtype != copy->dst->dtype)
      ThrowUnsupported("copy requires matching dtypes");
    arith::Analyzer analyzer;
    size_t source_offset = 0, destination_offset = 0;
    if (source->region.size() > destination->region.size() &&
        src->kind == "tensor")
      source_offset = source->region.size() - destination->region.size();
    else if (destination->region.size() > source->region.size() &&
             dst->kind == "tensor")
      destination_offset = destination->region.size() - source->region.size();
    if (source->region.size() - source_offset !=
        destination->region.size() - destination_offset)
      ThrowUnsupported(
          "copy rank difference requires leading Tensor unit axes");
    for (size_t axis = 0; axis < source_offset; ++axis)
      if (!analyzer.CanProveEqual(source->region[axis]->extent, Integer(1)))
        ThrowUnsupported("copy may only omit leading Tensor unit axes");
    for (size_t axis = 0; axis < destination_offset; ++axis)
      if (!analyzer.CanProveEqual(destination->region[axis]->extent,
                                  Integer(1)))
        ThrowUnsupported("copy may only omit leading Tensor unit axes");
    for (size_t axis = source_offset; axis < source->region.size(); ++axis)
      if (!analyzer.CanProveEqual(
              source->region[axis]->extent,
              destination->region[axis - source_offset + destination_offset]
                  ->extent))
        ThrowUnsupported("copy requires matching trailing region extents");
    if (src->kind == "tensor" && dst->kind == "logical_dfb_candidate") {
      int64_t tensor = src->global_arg_index.value()->value;
      tensor_reads_[tensor] = true;
      int64_t id =
          Write(copy->dst, "ncrisc", TensorBacking(tensor, Integer(0)));
      Reserve(&transfer_, id, call->span);
      ffi::Array<PrimExpr> args{Integer(tensor), Integer(id)};
      for (const Range &range : source->region) {
        args.push_back(range->min);
        args.push_back(range->extent);
      }
      transfer_.push_back(
          MakeDeviceCall(tenstorrent::tensor_to_dfb_nd(), args, call->span));
      if (transfers_)
        transfer_.push_back(MakeDeviceCall(tenstorrent::dfb_copy_wait(),
                                           {Integer(id), Integer(1)},
                                           call->span));
      return;
    }
    if (src->kind == "logical_dfb_candidate" && dst->kind == "tensor") {
      EnsurePrecision(Call(DataType::Void(), tile_compute(),
                           {source->ToPrimExpr()}, {}, call->span),
                      "copy");
      int64_t tensor = dst->global_arg_index.value()->value;
      tensor_writes_[tensor] = true;
      // A separate immutable export snapshot keeps every resource single-slot
      // consumer, even when the original value is used by later computation.
      int64_t input = Read(copy->src, "trisc");
      int64_t output =
          Write(copy->src, "trisc", TensorBacking(tensor, Integer(0)), false);
      resources_[output].consumer = "ncrisc";
      Wait(&compute_, input, call->span);
      Reserve(&compute_, output, call->span);
      ffi::Array<Integer> identity;
      for (size_t axis = 0; axis < copy->src->shape.size(); ++axis)
        identity.push_back(Integer(axis));
      ffi::Map<ffi::String, ffi::ObjectRef> annotations{
          {"tt.compute_kind", StringImm("copy")},
          {"tt.compute_dtype",
           StringImm(copy->src->dtype == DataType::BFloat(16) ? "bfloat16"
                                                              : "float32")},
          {"tt.compute_tile_shape", src->tile_shape},
          {"tt.logical_domain", copy->src->shape},
          {"tt.input_shapes",
           ffi::Array<ffi::Array<PrimExpr>>{copy->src->shape}},
          {"tt.access_maps", ffi::Array<ffi::Array<Integer>>{identity}}};
      compute_.push_back(Evaluate(
          Call(DataType::Void(), tenstorrent::dfb_compute(),
               {Integer(output), Integer(input)}, annotations, call->span),
          call->span));
      Wait(&transfer_, output, call->span);
      ffi::Array<PrimExpr> args{Integer(output), Integer(tensor)};
      for (const Range &range : destination->region) {
        args.push_back(range->min);
        args.push_back(range->extent);
      }
      transfer_.push_back(
          MakeDeviceCall(tenstorrent::dfb_to_tensor_nd(), args, call->span));
      if (transfers_)
        transfer_.push_back(MakeDeviceCall(tenstorrent::dfb_copy_wait(),
                                           {Integer(output), Integer(1)},
                                           call->span));
      return;
    }
    ThrowUnsupported("Copy must connect Tensor and DFB; shared Copy requires "
                     "compute legalization");
  }

  void PlanPipe(const Call &call) {
    auto descriptor = call->annotations.Get("tt.pipe_record");
    if (!descriptor.has_value())
      ThrowMalformed("Pipe operation lacks normalized record metadata");
    auto record = Downcast<ffi::Array<Integer>>(descriptor.value());
    bool send = call->op.same_as(tenstorrent::pipe_send());
    BufferRegion region =
        NormalizeToAccessRegion(call->args[send ? 0 : 1],
                                send ? kAccessRead : kAccessWrite)
            .region;
    FullRegion(region);
    int64_t id;
    bool forward = false;
    if (send) {
      int64_t input = Read(region->buffer, "trisc");
      const ResourceVersion &resource = resources_[input];
      forward =
          frontend_->attrs->dict.count("tt.gemm_accumulator_requirements") &&
          resource.producer == "ncrisc" && resource.backing.has_value();
      if (forward) {
        has_producer_forwarding_ = true;
        // Forward the completed Tensor load before publishing the same panel
        // to local computation. The verifier's publication frontier includes
        // every send completion; no TRISC copy or second payload is needed.
        id = input;
      } else {
        EnsurePrecision(Call(DataType::Void(), tile_compute(),
                             {region->ToPrimExpr()}, {}, call->span),
                        "copy");
        id = Write(region->buffer, "trisc", std::nullopt, false);
        resources_[id].consumer = "brisc";
        Wait(&compute_, input, call->span);
        Reserve(&compute_, id, call->span);
        const auto &metadata = resources_[id].metadata;
        ffi::Array<Integer> identity;
        for (size_t axis = 0; axis < region->buffer->shape.size(); ++axis)
          identity.push_back(Integer(axis));
        ffi::Map<ffi::String, ffi::ObjectRef> annotations{
            {"tt.compute_kind", StringImm("copy")},
            {"tt.compute_dtype",
             StringImm(region->buffer->dtype == DataType::BFloat(16)
                           ? "bfloat16"
                           : "float32")},
            {"tt.compute_tile_shape", metadata->tile_shape},
            {"tt.logical_domain", region->buffer->shape},
            {"tt.input_shapes",
             ffi::Array<ffi::Array<PrimExpr>>{region->buffer->shape}},
            {"tt.access_maps", ffi::Array<ffi::Array<Integer>>{identity}}};
        compute_.push_back(Evaluate(
            Call(DataType::Void(), tenstorrent::dfb_compute(),
                 {Integer(id), Integer(input)}, annotations, call->span),
            call->span));
        Wait(&pipe_, id, call->span);
      }
    } else {
      id = Write(region->buffer, "ncrisc");
      Reserve(&transfer_, id, call->span);
    }
    // Each endpoint's lexical occurrence names the same K-stage transaction.
    // Separate source/receiver counters also handle self-delivery.
    auto occurrence_key =
        std::make_tuple(record[0]->value, record[1]->value, send);
    int64_t occurrence = occurrences_[occurrence_key]++;
    int64_t bx = send ? record[5]->value : core_x_;
    int64_t by = send ? record[6]->value : core_y_;
    int64_t ex = send ? record[7]->value : core_x_ + 1;
    int64_t ey = send ? record[8]->value : core_y_ + 1;
    for (int64_t x = bx; x < ex; ++x) {
      for (int64_t y = by; y < ey; ++y) {
        auto key = std::make_tuple(record[0]->value, record[1]->value,
                                   occurrence, x, y);
        auto found = transfers_->find(key);
        int64_t transfer =
            found == transfers_->end() ? transfers_->size() : found->second;
        (*transfers_)[key] = transfer;
        endpoints_.push_back(
            {transfer, occurrence, record, x, y, id, send, call->span});
        auto *body = send && !forward ? &pipe_ : &transfer_;
        body->push_back(MakeDeviceCall(
            send ? tenstorrent::dfb_pipe_send() : tenstorrent::dfb_pipe_recv(),
            {Integer(transfer), Integer(id), Integer(1)}, call->span));
        body->push_back(MakeDeviceCall(
            tenstorrent::dfb_pipe_wait(),
            {Integer(transfer), Integer(id), Integer(1)}, call->span));
      }
    }
  }

  std::vector<Call> *collected_calls_{nullptr};
  size_t call_index_{0};
  std::unordered_map<Var, std::vector<FragmentAccess>, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      fragment_accesses_;
  std::vector<size_t> unknown_accesses_;
  bool compute_value_mode_{false};
  bool precision_regions_{false};
  int precision_mode_{-1};
  ffi::Array<ComputeValueDescriptor> values_;
  std::vector<std::unordered_set<int64_t>> value_roots_;
  std::unordered_map<int64_t, std::unordered_set<int64_t>> dfb_roots_;
  std::unordered_map<Var, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      current_values_;
  std::unordered_map<int64_t, int64_t> materialized_values_;
  std::vector<AccumulatorLifetime> accumulators_;
  std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      accumulator_ids_;
  std::map<std::tuple<int64_t, int64_t, bool>, int64_t> occurrences_;
  bool has_producer_forwarding_{false};
  TransferKeys *transfers_;
  int64_t core_x_, core_y_;
  std::vector<PipeEndpoint> endpoints_;
  ffi::Array<Stmt> pipe_;
  PrimFunc frontend_;
  BufferMetadataMap metadata_;
  std::unordered_map<Var, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      current_;
  std::vector<ResourceVersion> resources_;
  std::vector<bool> tensor_reads_, tensor_writes_;
  ffi::Array<Stmt> compute_, transfer_;
  size_t statement_count_{0};
  std::unordered_set<int64_t> live_;
  ffi::String pipeline_wait_policy_{"delayed"};
  int64_t pipeline_stages_{0};
  int64_t pipeline_extent_{0};
  int64_t pipeline_depth_{0};
  int64_t pipeline_value_begin_{0};
  int64_t pipeline_resource_begin_{0};
};

bool HasGeneralCompute(const Stmt &body) {
  bool found = false;
  PostOrderVisit(body, [&](const ffi::ObjectRef &object) {
    if (const auto *call = object.as<CallNode>())
      found |= call->op.same_as(tenstorrent::tile_compute());
  });
  return found;
}

ffi::Optional<For> FindPipelineLoop(const Stmt &body) {
  Stmt candidate = body;
  while (const auto *realize = candidate.as<SBlockRealizeNode>()) {
    if (!realize->iter_values.empty() || !is_one(realize->predicate) ||
        !realize->block->iter_vars.empty() ||
        realize->block->init.has_value() ||
        !realize->block->match_buffers.empty())
      return std::nullopt;
    candidate = realize->block->body;
  }
  auto loop = candidate.as<For>();
  if (loop.has_value() && loop.value()->annotations.count("num_stages"))
    return loop.value();
  return std::nullopt;
}

// A complete iteration can retain its structured loop without constructing
// loop-carried DFB state. Tensor inout remains ordered by the NCRISC stream.
ffi::Optional<For> FindIndependentLoop(const Stmt &body) {
  Stmt candidate = body;
  while (const auto *realize = candidate.as<SBlockRealizeNode>()) {
    if (!realize->iter_values.empty() || !is_one(realize->predicate) ||
        !realize->block->iter_vars.empty() ||
        realize->block->init.has_value() ||
        !realize->block->match_buffers.empty()) {
      return std::nullopt;
    }
    candidate = realize->block->body;
  }
  auto loop = candidate.as<For>();
  if (!loop.has_value() || loop.value()->kind != ForKind::kSerial ||
      !loop.value()->annotations.empty() ||
      (loop.value()->step.has_value() && !is_one(loop.value()->step.value()))) {
    return std::nullopt;
  }
  For value = loop.value();
  if (!as_const_int(value->min) || !as_const_int(value->extent) ||
      *as_const_int(value->extent) <= 0 ||
      *as_const_int(value->extent) > 1024 ||
      UsesVar(value->body, [&](const VarNode *var) {
        return ffi::GetRef<Var>(var).same_as(value->loop_var);
      })) {
    return std::nullopt;
  }
  return value;
}

Stmt WrapIndependentLoop(Stmt body, const For &loop) {
  if (const auto *evaluate = body.as<EvaluateNode>()) {
    if (is_zero(evaluate->value))
      return body;
  }
  // Each function owns its binder, even though the body does not use it.
  return For(Var("tt_iteration", loop->loop_var.dtype()), loop->min,
             loop->extent, ForKind::kSerial, std::move(body), std::nullopt, {},
             std::nullopt, loop->span);
}

PrimFunc MakeSlotFunction(const PrimFunc &frontend, const Target &target,
                          const CoreDomain &domain,
                          const ffi::String &operation, const ffi::String &slot,
                          const ffi::String &thread_kind,
                          ffi::Optional<Integer> noc_index,
                          const ffi::Array<Integer> &tensor_arg_indices,
                          const ffi::String &logical_kind,
                          const ffi::String &logical_role, Stmt body) {
  ffi::Array<Var> params;
  ffi::Map<Var, Buffer> buffer_map;
  for (const Integer &tensor_index : tensor_arg_indices) {
    if (tensor_index->value < 0 ||
        tensor_index->value >= static_cast<int64_t>(frontend->params.size())) {
      ThrowMalformed("slot Tensor ABI index is out of range");
    }
    const Var &parameter = frontend->params[tensor_index->value];
    ffi::Optional<Buffer> buffer = frontend->buffer_map.Get(parameter);
    if (!buffer.has_value()) {
      ThrowMalformed("slot Tensor ABI parameter has no Buffer");
    }
    params.push_back(parameter);
    buffer_map.Set(parameter, buffer.value());
  }

  const ffi::String symbol = operation + "_" + slot;
  LogicalKernel logical_kernel("kernel." + slot, logical_kind, logical_role,
                               frontend->span);
  ffi::Map<ffi::String, ffi::Any> attrs = {
      {tvm::attr::kGlobalSymbol, symbol},
      {tvm::attr::kCallingConv,
       Integer(static_cast<int64_t>(CallingConv::kDeviceKernelLaunch))},
      {tvm::attr::kTarget, target},
      {kKernelSlotAttr, slot},
      {kKernelThreadAttr, thread_kind},
      {kLogicalKernelAttr, logical_kernel},
      {kTensorArgIndicesAttr, tensor_arg_indices},
      {kCoreDomainAttr, domain},
  };
  if (noc_index.has_value()) {
    attrs.Set(kNocIndexAttr, noc_index.value());
  }
  return PrimFunc(std::move(params), std::move(body), frontend->ret_type,
                  std::move(buffer_map), DictAttrs(std::move(attrs)),
                  frontend->span);
}

// Every Core uses the established single-Core value planner. Only identity
// remapping and explicit Pipe transfer matching cross this boundary.
class DeviceIDRemapper : public StmtExprMutator {
public:
  DeviceIDRemapper(int64_t offset, int64_t accumulator_offset,
                   int64_t value_offset)
      : offset_(offset), accumulator_offset_(accumulator_offset),
        value_offset_(value_offset) {}
  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    auto args = call->args;
    auto shift = [&](size_t index) {
      args.Set(index, Integer(*as_const_int(args[index]) + offset_));
    };
    auto shift_value = [&](size_t index) {
      int64_t value = *as_const_int(args[index]);
      if (value >= 0)
        args.Set(index, Integer(value + value_offset_));
    };
    if (call->op.same_as(compute_value())) {
      shift_value(0);
      for (size_t index = 1; index < args.size(); ++index)
        shift(index);
    } else if (call->op.same_as(compute_value_gemm())) {
      shift_value(0);
      shift(1);
      shift(2);
      shift_value(3);
    } else if (call->op.same_as(compute_value_store())) {
      shift_value(0);
      shift(1);
    } else if (call->op.same_as(compute_value_load())) {
      shift_value(0);
    } else if (call->op.same_as(tenstorrent::accumulator_init()) ||
               call->op.same_as(tenstorrent::accumulator_materialize())) {
      args.Set(0, Integer(*as_const_int(args[0]) + accumulator_offset_));
      if (call->op.same_as(tenstorrent::accumulator_materialize()))
        shift(1);
    } else if (call->op.same_as(tenstorrent::gemm_update())) {
      shift(0);
      shift(1);
      args.Set(2, Integer(*as_const_int(args[2]) + accumulator_offset_));
    } else if (call->op.same_as(tenstorrent::dfb_compute())) {
      for (size_t index = 0; index < args.size(); ++index)
        shift(index);
    } else if (call->op.same_as(tenstorrent::tensor_to_dfb_nd()) ||
               call->op.same_as(tenstorrent::dfb_pipe_send()) ||
               call->op.same_as(tenstorrent::dfb_pipe_recv()) ||
               call->op.same_as(tenstorrent::dfb_pipe_wait())) {
      shift(1);
    } else if (call->op.same_as(tenstorrent::dfb_to_tensor_nd()) ||
               call->op.same_as(tenstorrent::dfb_reserve()) ||
               call->op.same_as(tenstorrent::dfb_wait()) ||
               call->op.same_as(tenstorrent::dfb_release()) ||
               call->op.same_as(tenstorrent::dfb_copy_wait()) ||
               call->op.same_as(tenstorrent::dfb_load())) {
      shift(0);
    }
    auto attrs = call->annotations;
    auto inputs = attrs.Get("tt.value_inputs");
    if (inputs.has_value()) {
      ffi::Array<Integer> remapped;
      for (const Integer &value : Downcast<ffi::Array<Integer>>(inputs.value()))
        remapped.push_back(Integer(value->value + value_offset_));
      attrs.Set("tt.value_inputs", remapped);
    }
    return Call(call->dtype, call->op, args, attrs, op->span);
  }

private:
  int64_t offset_;
  int64_t accumulator_offset_;
  int64_t value_offset_;
};

IRModule FormMulticoreProgram(const IRModule &input, const PrimFunc &frontend,
                              const Target &target,
                              const ffi::String &operation,
                              const ffi::String &arch, int64_t gx, int64_t gy,
                              const Stmt &body,
                              const ffi::Array<TTBufferMetadata> &metadata) {
  ffi::Array<Stmt> cores;
  if (const auto *seq = body.as<SeqStmtNode>())
    cores = seq->seq;
  else
    cores.push_back(body);
  if (cores.size() != static_cast<size_t>(gx * gy))
    ThrowMalformed("normalized topology does not cover the Core grid");
  ffi::Map<GlobalVar, BaseFunc> functions;
  ffi::Array<ffi::String> order;
  ffi::Array<DFBDescriptor> dfbs;
  ffi::Array<PipeDescriptor> pipes;
  ffi::Array<PipeTransferDescriptor> transfers;
  ffi::Array<AccumulatorDescriptor> accumulators;
  ffi::Array<ComputeValueDescriptor> compute_values;
  std::vector<int> effects(frontend->params.size(), 0);
  TransferKeys transfer_keys;
  std::vector<PipeEndpoint> endpoints;
  int64_t offset = 0, accumulator_offset = 0, value_offset = 0;
  int64_t pipeline_stages = 0, pipeline_extent = 0, group_offset = 0;
  ffi::String pipeline_wait_policy;
  ffi::Map<ffi::String, Integer> storage_groups;
  ffi::Map<ffi::String, ffi::Array<Integer>> pipeline_relations;
  bool producer_forwarding = false;
  bool precision_regions = false;
  for (size_t index = 0; index < cores.size(); ++index) {
    int64_t x = index / gy, y = index % gy;
    const auto *realize = cores[index].as<SBlockRealizeNode>();
    if (!realize || !realize->block->annotations.count("tt.core_x") ||
        Downcast<Integer>(realize->block->annotations.at("tt.core_x"))->value !=
            x ||
        Downcast<Integer>(realize->block->annotations.at("tt.core_y"))->value !=
            y)
      ThrowMalformed("normalized topology Core order is not lexicographic x/y");
    GeneralDataflowPlanner planner(frontend, metadata, &transfer_keys, x, y);
    planner.Plan(realize->block->body);
    planner.EliminateDeadWrites();
    planner.FinalizeMulticore();
    if (planner.IsPipeline()) {
      if (pipeline_extent &&
          (pipeline_extent != planner.PipelineExtent() ||
           pipeline_stages != planner.PipelineStages() ||
           pipeline_wait_policy != planner.PipelineWaitPolicy()))
        ThrowUnsupported("all Core pipelines require identical static extent, "
                         "num_stages and wait policy");
      pipeline_extent = planner.PipelineExtent();
      pipeline_stages = planner.PipelineStages();
      pipeline_wait_policy = planner.PipelineWaitPolicy();
      for (const auto &[key, group] : planner.StorageGroups())
        storage_groups.Set(
            std::to_string(std::stoll(std::string(key)) + offset),
            Integer(group->value + group_offset));
      for (const auto &[key, relation] : planner.PipelineRelations())
        pipeline_relations.Set(
            std::to_string(std::stoll(std::string(key)) + offset), relation);
    } else if (pipeline_extent || !storage_groups.empty()) {
      ThrowUnsupported("all active Core bodies must retain their pipeline");
    }
    producer_forwarding |= planner.HasProducerForwarding();
    precision_regions |= planner.HasPrecisionRegions();
    CoreDomain domain(CoreCoord(x, y), CoreCoord(x + 1, y + 1));
    for (const DFBDescriptor &dfb : planner.Descriptors(domain)) {
      dfbs.push_back(DFBDescriptor(
          dfb->dfb_id + offset,
          dfb->source_buffer_identity + ".core" + std::to_string(x) + "_" +
              std::to_string(y),
          dfb->element_dtype, dfb->tile_shape, dfb->block_shape_in_tiles,
          dfb->block_count, dfb->tensor_backing, dfb->producer_slot, domain,
          dfb->consumer_slot, domain, dfb->transaction_count_or_loop_relation,
          dfb->source_span));
    }
    std::unordered_map<Var, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
        core_buffers;
    auto isolate_buffer = [&](const Buffer &buffer) -> Buffer {
      auto found = core_buffers.find(buffer->data);
      if (found != core_buffers.end())
        return found->second;
      Buffer isolated = buffer;
      isolated.CopyOnWrite()->data = buffer->data.copy_with_suffix(
          "_core" + std::to_string(x) + "_" + std::to_string(y));
      core_buffers.emplace(buffer->data, isolated);
      return isolated;
    };
    for (const AccumulatorDescriptor &accumulator : planner.Accumulators()) {
      accumulators.push_back(AccumulatorDescriptor(
          accumulator->accumulator_id + accumulator_offset,
          BufferRegion(isolate_buffer(accumulator->accumulator_region->buffer),
                       accumulator->accumulator_region->region),
          accumulator->input_dtype, accumulator->accumulation_dtype,
          accumulator->output_dtype, accumulator->full_k_tiles,
          accumulator->source_span));
    }
    for (const ComputeValueDescriptor &value : planner.ComputeValues()) {
      compute_values.push_back(ComputeValueDescriptor(
          value->value_id + value_offset, isolate_buffer(value->buffer),
          value->version,
          value->previous_value_id < 0
              ? -1
              : value->previous_value_id + value_offset,
          value->accumulator_id < 0
              ? -1
              : value->accumulator_id + accumulator_offset,
          value->source_span));
    }
    for (PipeEndpoint endpoint : planner.Endpoints()) {
      endpoint.dfb_id += offset;
      endpoints.push_back(std::move(endpoint));
    }
    auto core_effects = planner.Effects();
    for (const Integer &tensor : planner.UsedTensorIndices()) {
      const auto &effect = core_effects[tensor->value];
      effects[tensor->value] |= effect == "inout"    ? 3
                                : effect == "output" ? 2
                                                     : 1;
    }
    DeviceIDRemapper remap(offset, accumulator_offset, value_offset);
    ffi::String core_operation =
        operation + "_x" + std::to_string(x) + "_y" + std::to_string(y);
    for (const ffi::String &slot :
         ffi::Array<ffi::String>{"trisc", "ncrisc", "brisc"}) {
      bool compute = slot == "trisc";
      Stmt slot_body = compute            ? planner.ComputeBody()
                       : slot == "ncrisc" ? planner.TransferBody()
                                          : planner.PipeBody();
      PrimFunc func = MakeSlotFunction(
          frontend, target, domain, core_operation, slot,
          compute ? "compute" : "datamovement",
          compute ? ffi::Optional<Integer>(std::nullopt)
                  : ffi::Optional<Integer>(Integer(slot == "ncrisc" ? 0 : 1)),
          slot == "ncrisc" ? planner.UsedTensorIndices()
                           : ffi::Array<Integer>(),
          compute ? "compute" : "datamovement",
          compute            ? "general"
          : slot == "ncrisc" ? "tensor_io"
                             : "pipe_source",
          remap(slot_body));
      ffi::Array<Var> isolated_params;
      ffi::Map<Var, Buffer> isolated_buffers;
      for (const Var &parameter : func->params) {
        Var isolated_parameter = parameter.copy_with_suffix("");
        Buffer buffer = func->buffer_map.at(parameter);
        buffer.CopyOnWrite()->data = buffer->data.copy_with_suffix("");
        isolated_params.push_back(isolated_parameter);
        isolated_buffers.Set(isolated_parameter, buffer);
      }
      func.CopyOnWrite()->params = isolated_params;
      func.CopyOnWrite()->buffer_map = isolated_buffers;
      ffi::String symbol = core_operation + "_" + slot;
      func = WithAttr(func, kLogicalKernelAttr,
                      LogicalKernel("kernel." + symbol,
                                    compute ? "compute" : "datamovement",
                                    compute            ? "general"
                                    : slot == "ncrisc" ? "tensor_io"
                                                       : "pipe_source",
                                    frontend->span));
      functions.Set(GlobalVar(symbol), func);
      order.push_back(symbol);
    }
    offset += planner.ResourceCount();
    accumulator_offset += planner.AccumulatorCount();
    value_offset += planner.ComputeValues().size();
    group_offset += 2 * planner.ResourceCount();
  }
  std::map<int64_t, PipeEndpoint> senders, receivers;
  std::map<std::pair<int64_t, int64_t>, int64_t> record_payloads;
  std::map<std::pair<int64_t, int64_t>, PipeDescriptor> ordered_pipes;
  for (const PipeEndpoint &endpoint : endpoints) {
    auto &side = endpoint.source ? senders : receivers;
    if (!side.emplace(endpoint.transfer_id, endpoint).second)
      ThrowMalformed(
          "Pipe record has duplicate endpoint for one transaction occurrence");
    if (endpoint.source) {
      const auto &r = endpoint.record;
      auto key = std::make_pair(r[0]->value, r[1]->value);
      auto previous = record_payloads.find(key);
      if (previous == record_payloads.end()) {
        record_payloads.emplace(key, endpoint.dfb_id);
        ordered_pipes.emplace(
            key, PipeDescriptor(r[0]->value, r[1]->value,
                                CoreCoord(r[3]->value, r[4]->value),
                                CoreCoord(r[5]->value, r[6]->value),
                                CoreCoord(r[7]->value, r[8]->value),
                                r[2]->value ? "collective" : "point_to_point",
                                endpoint.dfb_id, endpoint.span));
      } else if (endpoint.occurrence == 0 &&
                 previous->second != endpoint.dfb_id) {
        ThrowMalformed(
            "Pipe record has more than one source payload per occurrence");
      }
    }
  }
  auto original_records =
      frontend->GetAttr<ffi::Array<ffi::Array<Integer>>>("tt.topology_records");
  if (!original_records.has_value())
    ThrowMalformed("normalized topology original record registry is missing");
  std::unordered_set<int64_t> active_nets;
  for (const auto &[key, payload] : record_payloads)
    active_nets.insert(key.first);
  for (const auto &record : original_records.value()) {
    if (active_nets.count(record[0]->value) &&
        !record_payloads.count({record[0]->value, record[1]->value}))
      ThrowMalformed("active PipeNet must retain one transaction for every "
                     "original record");
  }
  for (const auto &[key, pipe] : ordered_pipes)
    pipes.push_back(pipe);
  if (senders.size() != receivers.size())
    ThrowMalformed(
        "Pipe producer/consumer transaction count mismatch (" +
        std::to_string(senders.size()) + " send occurrences, " +
        std::to_string(receivers.size()) +
        " receive occurrences); each repeated transaction occurrence "
        "requires a matching peer");
  for (const auto &[id, sender] : senders) {
    auto receiver = receivers.find(id);
    if (receiver == receivers.end())
      ThrowMalformed("Pipe send has no matching receive");
    const auto &r = sender.record;
    if (!ffi::StructuralEqual()(r, receiver->second.record))
      ThrowMalformed("Pipe endpoint frozen descriptors disagree");
    transfers.push_back(PipeTransferDescriptor(
        id, r[0]->value, r[1]->value, sender.occurrence,
        CoreCoord(r[3]->value, r[4]->value), CoreCoord(sender.x, sender.y),
        sender.dfb_id, receiver->second.dfb_id, 1, sender.span));
  }
  ffi::Array<ffi::String> tensor_effects;
  for (int effect : effects)
    tensor_effects.push_back(effect == 3   ? "inout"
                             : effect == 2 ? "output"
                                           : "input");
  auto tensors = BuildTensorTable(frontend, metadata, tensor_effects, true);
  bool repeated_transfers = std::any_of(
      endpoints.begin(), endpoints.end(),
      [](const PipeEndpoint &endpoint) { return endpoint.occurrence != 0; });
  ffi::Map<ffi::String, ffi::Any> attrs{
      {kDeviceIRVersionAttr,
       Integer(precision_regions         ? 8
               : !compute_values.empty() ? 7
               : !accumulators.empty() || repeated_transfers ||
                       producer_forwarding
                   ? 6
                   : 4)},
      {kTargetArchAttr, arch},
      {kLaunchGridAttr, CoreCoord(gx, gy)},
      {kOperationIdentityAttr, OperationIdentity(operation, frontend->span)},
      {kTensorTableAttr, tensors},
      {kDFBTableAttr, dfbs},
      {kPipeTableAttr, pipes},
      {kPipeTransferTableAttr, transfers},
      {kKernelOrderAttr, order}};
  if (!accumulators.empty())
    attrs.Set(kAccumulatorTableAttr, accumulators);
  if (!compute_values.empty())
    attrs.Set(kComputeValueTableAttr, compute_values);
  if (pipeline_extent) {
    attrs.Set("tt.dfb_storage_groups", storage_groups);
    attrs.Set("tt.pipeline_relations", pipeline_relations);
    attrs.Set("tt.pipeline_stages", Integer(pipeline_stages));
    attrs.Set("tt.pipeline_extent", Integer(pipeline_extent));
    attrs.Set("tt.pipeline_wait_policy", pipeline_wait_policy);
  }
  auto capacity = input->GetAttr<Integer>("tt.l1_capacity_bytes");
  if (capacity.has_value())
    attrs.Set("tt.l1_capacity_bytes", capacity.value());
  return IRModule(functions, input->source_map, DictAttrs(attrs),
                  input->global_infos);
}

IRModule FormProgram(const IRModule &input) {
  if (input->GetAttr<Integer>(kDeviceIRVersionAttr).has_value()) {
    return VerifyTenstorrentDeviceIR()(input);
  }

  ffi::Optional<GlobalVar> frontend_global;
  ffi::Optional<PrimFunc> frontend_func;
  for (const auto &[global_var, base_func] : input->functions) {
    ffi::Optional<PrimFunc> func = base_func.as<PrimFunc>();
    if (!func.has_value()) {
      ThrowUnsupported("non-PrimFunc global '" +
                       std::string(global_var->name_hint) + "'");
    }
    if (frontend_func.has_value()) {
      ThrowUnsupported(
          "multiple frontend PrimFuncs; the current backend forms one "
          "operation at a time");
    }
    frontend_global = global_var;
    frontend_func = func.value();
  }
  if (!frontend_func.has_value()) {
    ThrowMalformed("IRModule contains no frontend PrimFunc");
  }

  const PrimFunc &frontend = frontend_func.value();
  ffi::Optional<Target> target = frontend->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.has_value() || target.value()->kind->name != "tenstorrent") {
    ThrowMalformed("frontend PrimFunc has no bound Tenstorrent target");
  }
  ffi::Optional<ffi::String> target_arch =
      target.value()->GetAttr<ffi::String>("arch");
  if (!target_arch.has_value()) {
    ThrowMalformed("bound Tenstorrent target has no architecture");
  }
  if (target_arch.value() != "wormhole_b0" &&
      target_arch.value() != "blackhole") {
    ThrowUnsupported("target architecture '" +
                     std::string(target_arch.value()) + "'");
  }

  ffi::Optional<ffi::Array<PrimExpr>> launch_grid =
      frontend->GetAttr<ffi::Array<PrimExpr>>(kLaunchGridAttr);
  if (!launch_grid.has_value() || launch_grid.value().size() != 2) {
    ThrowMalformed("function-level tt.launch_grid is missing or malformed");
  }
  const int64_t grid_x =
      RequirePositiveStaticInteger(launch_grid.value()[0], "launch grid x");
  const int64_t grid_y =
      RequirePositiveStaticInteger(launch_grid.value()[1], "launch grid y");

  ffi::Optional<ffi::Array<TTBufferMetadata>> buffer_table =
      frontend->GetAttr<ffi::Array<TTBufferMetadata>>(kBufferMetadataTableAttr);
  if (!buffer_table.has_value()) {
    ThrowMalformed("function-level tt.buffer_metadata_table is missing");
  }

  Stmt kernel_body = StripLogicalCoreLoops(frontend, launch_grid.value());
  if (frontend->GetAttr<Integer>("tt.topology_normalized").has_value())
    return FormMulticoreProgram(
        input, frontend, target.value(), frontend_global.value()->name_hint,
        target_arch.value(), grid_x, grid_y, kernel_body, buffer_table.value());
  if (grid_x != 1 || grid_y != 1)
    ThrowMalformed("multi-Core input requires NormalizeTenstorrentTopology");
  const size_t tile_add_count = TileAddCounter::Count(kernel_body);
  bool legacy_add = tile_add_count == 1 && frontend->params.size() == 3 &&
                    buffer_table.value().size() == 6;
  if (const auto *realize = kernel_body.as<SBlockRealizeNode>()) {
    auto statements = RequireSequence(realize->block->body);
    legacy_add &= statements.size() == 4;
    if (statements.size() == 4) {
      const auto *evaluate = statements[2].as<EvaluateNode>();
      const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
      legacy_add &= call && call->op.same_as(tenstorrent::tile_add());
    }
  } else {
    legacy_add = false;
  }
  const bool general_compute = HasGeneralCompute(kernel_body) ||
                               FindPipelineLoop(kernel_body).has_value() ||
                               (tile_add_count > 0 && !legacy_add);
  if (!general_compute && tile_add_count > 1) {
    ThrowUnsupported("Phase 2 supports exactly one canonical Add operation");
  }

  const ffi::String operation = frontend_global.value()->name_hint;
  Span source_span =
      RequireSourceSpan(frontend->span, Span(), "frontend PrimFunc");
  CoreCoord launch(grid_x, grid_y);
  CoreDomain domain(CoreCoord(0, 0), CoreCoord(grid_x, grid_y));
  ffi::Array<TensorDescriptor> tensors;
  ffi::Array<DFBDescriptor> dfbs;
  ffi::Array<AccumulatorDescriptor> accumulators;
  ffi::Array<ComputeValueDescriptor> compute_values;
  ffi::Map<GlobalVar, BaseFunc> functions;
  PrimFunc trisc;
  PrimFunc ncrisc;
  PrimFunc brisc;
  ffi::Map<ffi::String, Integer> storage_groups;
  ffi::Map<ffi::String, ffi::Array<Integer>> pipeline_relations;
  int64_t pipeline_stages = 0, pipeline_extent = 0;
  bool precision_regions = false;
  ffi::String pipeline_wait_policy;
  Stmt idle = Evaluate(IntImm(DataType::Int(32), 0), frontend->span);
  if (general_compute) {
    GeneralDataflowPlanner planner(frontend, buffer_table.value());
    ffi::Optional<For> independent_loop =
        planner.HasComputeValues() ? ffi::Optional<For>(std::nullopt)
                                   : FindIndependentLoop(kernel_body);
    ffi::Optional<For> pipeline_loop = FindPipelineLoop(kernel_body);
    if (pipeline_loop.has_value()) {
      planner.PlanPipeline(pipeline_loop.value());
    } else {
      planner.Plan(independent_loop.has_value() ? independent_loop.value()->body
                                                : kernel_body);
    }
    planner.EliminateDeadWrites();
    planner.FinalizePipeline();
    planner.FinalizeComputeValues();
    if (planner.IsPipeline()) {
      pipeline_stages = planner.PipelineStages();
      pipeline_extent = planner.PipelineExtent();
      pipeline_wait_policy = planner.PipelineWaitPolicy();
      storage_groups = planner.StorageGroups();
      pipeline_relations = planner.PipelineRelations();
    }
    tensors = BuildTensorTable(frontend, buffer_table.value(),
                               planner.Effects(), true);
    dfbs = planner.Descriptors(domain);
    accumulators = planner.Accumulators();
    compute_values = planner.ComputeValues();
    precision_regions = planner.HasPrecisionRegions();
    Stmt compute_body = planner.ComputeBody();
    Stmt transfer_body = planner.TransferBody();
    if (independent_loop.has_value()) {
      const For &loop = independent_loop.value();
      ffi::Array<DFBDescriptor> repeated;
      for (const DFBDescriptor &dfb : dfbs) {
        repeated.push_back(DFBDescriptor(
            dfb->dfb_id, dfb->source_buffer_identity, dfb->element_dtype,
            dfb->tile_shape, dfb->block_shape_in_tiles, dfb->block_count,
            dfb->tensor_backing, dfb->producer_slot, dfb->producer_domain,
            dfb->consumer_slot, dfb->consumer_domain, loop->extent,
            dfb->source_span));
      }
      dfbs = std::move(repeated);
      compute_body = WrapIndependentLoop(std::move(compute_body), loop);
      transfer_body = WrapIndependentLoop(std::move(transfer_body), loop);
    }
    ffi::Array<Integer> tensor_indices = planner.UsedTensorIndices();
    trisc = MakeSlotFunction(frontend, target.value(), domain, operation,
                             "trisc", "compute", std::nullopt, {}, "compute",
                             "general", std::move(compute_body));
    ncrisc =
        MakeSlotFunction(frontend, target.value(), domain, operation, "ncrisc",
                         "datamovement", Integer(0), tensor_indices,
                         "datamovement", "tensor_io", std::move(transfer_body));
  } else if (tile_add_count == 0) {
    ffi::Array<ffi::String> effects;
    ffi::Array<Integer> all_tensor_indices;
    for (size_t index = 0; index < frontend->params.size(); ++index) {
      effects.push_back("input");
      all_tensor_indices.push_back(Integer(index));
    }
    tensors = BuildTensorTable(frontend, buffer_table.value(), effects,
                               /*allow_dfb_candidates=*/false);
    RequireNoOpBody(kernel_body);
    trisc = MakeSlotFunction(frontend, target.value(), domain, operation,
                             "trisc", "compute", /*noc_index=*/std::nullopt,
                             all_tensor_indices, "compute", "phase1_skeleton",
                             idle);
    ncrisc =
        MakeSlotFunction(frontend, target.value(), domain, operation, "ncrisc",
                         "datamovement", Integer(0), {}, "idle", "idle", idle);
  } else {
    AddDataflowPlan plan =
        AnalyzeAddDataflow(frontend, kernel_body, buffer_table.value());
    tensors = BuildTensorTable(frontend, buffer_table.value(),
                               {"input", "input", "output"},
                               /*allow_dfb_candidates=*/true);
    dfbs = BuildAddDFBTable(plan, domain, frontend);
    trisc = MakeSlotFunction(frontend, target.value(), domain, operation,
                             "trisc", "compute", /*noc_index=*/std::nullopt, {},
                             "compute", "add", BuildAddTRISCBody(plan));
    ncrisc = MakeSlotFunction(
        frontend, target.value(), domain, operation, "ncrisc", "datamovement",
        Integer(0), {Integer(0), Integer(1), Integer(2)}, "datamovement",
        "tensor_io", BuildAddNCRISCBody(plan));
  }
  brisc =
      MakeSlotFunction(frontend, target.value(), domain, operation, "brisc",
                       "datamovement", Integer(1), {}, "idle", "idle", idle);
  functions.Set(GlobalVar(operation + "_trisc"), std::move(trisc));
  functions.Set(GlobalVar(operation + "_ncrisc"), std::move(ncrisc));
  functions.Set(GlobalVar(operation + "_brisc"), std::move(brisc));

  ffi::Map<ffi::String, ffi::Any> attrs = {
      {kDeviceIRVersionAttr, Integer(precision_regions         ? 8
                                     : !compute_values.empty() ? 7
                                     : !accumulators.empty()   ? 5
                                     : pipeline_extent         ? 3
                                     : general_compute ? 2
                                                       : kDeviceIRVersion)},
      {kTargetArchAttr, target_arch.value()},
      {kLaunchGridAttr, launch},
      {kOperationIdentityAttr,
       OperationIdentity(operation, std::move(source_span))},
      {kTensorTableAttr, tensors},
      {kDFBTableAttr, dfbs},
      {kPipeTableAttr, ffi::Array<PipeDescriptor>()},
      {kKernelOrderAttr, ffi::Array<ffi::String>({"trisc", "ncrisc", "brisc"})},
  };
  if (!accumulators.empty())
    attrs.Set(kAccumulatorTableAttr, accumulators);
  if (!compute_values.empty())
    attrs.Set(kComputeValueTableAttr, compute_values);
  if (pipeline_extent) {
    attrs.Set("tt.dfb_storage_groups", storage_groups);
    attrs.Set("tt.pipeline_relations", pipeline_relations);
    attrs.Set("tt.pipeline_stages", Integer(pipeline_stages));
    attrs.Set("tt.pipeline_extent", Integer(pipeline_extent));
    attrs.Set("tt.pipeline_wait_policy", pipeline_wait_policy);
  }
  auto l1_capacity = input->GetAttr<Integer>("tt.l1_capacity_bytes");
  if (l1_capacity.has_value())
    attrs.Set("tt.l1_capacity_bytes", l1_capacity.value());
  return IRModule(std::move(functions), input->source_map,
                  DictAttrs(std::move(attrs)), input->global_infos);
}

} // namespace

tvm::transform::Pass FormTenstorrentDeviceProgram() {
  auto pass_func = [](IRModule mod,
                      const tvm::transform::PassContext &context) {
    return FormProgram(mod);
  };
  return tvm::transform::CreateModulePass(
      pass_func, 0, "tl.tenstorrent.FormTenstorrentDeviceProgram", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::GlobalDef().def("tl.tenstorrent.transform.FormTenstorrentDeviceProgram",
                        FormTenstorrentDeviceProgram);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

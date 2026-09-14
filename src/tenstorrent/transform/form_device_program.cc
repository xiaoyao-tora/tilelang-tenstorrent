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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
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
      << "[FormTenstorrentDeviceProgram] unsupported program: "
      << message;
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
    const int64_t start = RequireStaticInteger(
        region->region[axis]->min,
        owner + " start axis " + std::to_string(axis));
    const int64_t extent = RequireStaticInteger(
        region->region[axis]->extent,
        owner + " extent axis " + std::to_string(axis));
    const int64_t shape = RequireStaticInteger(
        region->buffer->shape[axis],
        owner + " buffer shape axis " + std::to_string(axis));
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
    ThrowMalformed("canonical tl.tt.tile_add must have three BufferRegion args");
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
  plan.tensor_regions[2] = RequireFullRegion(output_destination, "output C Tensor");
  RequireFullRegion(add_a, "input A DFB");
  RequireFullRegion(add_b, "input B DFB");
  RequireFullRegion(add_c, "output C DFB");

  BufferMetadataMap metadata_by_buffer = IndexBufferMetadata(buffer_table);
  plan.tensors = {
      RequireMetadata(metadata_by_buffer, input_a_source->buffer, "input A"),
      RequireMetadata(metadata_by_buffer, input_b_source->buffer, "input B"),
      RequireMetadata(metadata_by_buffer, output_destination->buffer, "output C")};
  plan.dfbs = {
      RequireMetadata(metadata_by_buffer, add_a->buffer, "input A DFB"),
      RequireMetadata(metadata_by_buffer, add_b->buffer, "input B DFB"),
      RequireMetadata(metadata_by_buffer, add_c->buffer, "output C DFB")};

  for (size_t index = 0; index < 3; ++index) {
    const TTBufferMetadata &tensor = plan.tensors[index];
    const TTBufferMetadata &dfb = plan.dfbs[index];
    if (tensor->kind != "tensor" || !tensor->global_arg_index.has_value() ||
        tensor->global_arg_index.value()->value != static_cast<int64_t>(index)) {
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

ffi::Array<TensorDescriptor>
BuildTensorTable(const PrimFunc &func,
                 const ffi::Array<TTBufferMetadata> &buffer_table,
                 const ffi::Array<ffi::String> &effects,
                 bool allow_dfb_candidates) {
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
        effects[expected_tensor_index],
        global_arg_index, std::move(source_span)));
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

Stmt MakeDeviceCall(const Op &op, ffi::Array<PrimExpr> args,
                    const Span &span) {
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
                      MakeTransferArgs(0, 0, plan.tensor_regions[0], true), span),
       MakeDeviceCall(tenstorrent::dfb_reserve(), {Integer(1), Integer(1)},
                      span),
       MakeDeviceCall(tenstorrent::tensor_to_dfb(),
                      MakeTransferArgs(1, 1, plan.tensor_regions[1], true), span),
       MakeDeviceCall(tenstorrent::dfb_wait(), {Integer(2), Integer(1)}, span),
       MakeDeviceCall(tenstorrent::dfb_to_tensor(),
                      MakeTransferArgs(2, 2, plan.tensor_regions[2], false),
                      span)},
      span);
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
      ThrowUnsupported("multiple frontend PrimFuncs; the current backend forms one "
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
  if (grid_x != 1 || grid_y != 1) {
    ThrowUnsupported("multi-Core program formation is deferred beyond Phase 2; "
                     "expected launch grid 1x1");
  }
  ffi::Optional<ffi::Array<TTBufferMetadata>> buffer_table =
      frontend->GetAttr<ffi::Array<TTBufferMetadata>>(kBufferMetadataTableAttr);
  if (!buffer_table.has_value()) {
    ThrowMalformed("function-level tt.buffer_metadata_table is missing");
  }

  Stmt kernel_body = StripLogicalCoreLoops(frontend, launch_grid.value());
  const size_t tile_add_count = TileAddCounter::Count(kernel_body);
  if (tile_add_count > 1) {
    ThrowUnsupported("Phase 2 supports exactly one canonical Add operation");
  }

  const ffi::String operation = frontend_global.value()->name_hint;
  Span source_span =
      RequireSourceSpan(frontend->span, Span(), "frontend PrimFunc");
  CoreCoord launch(grid_x, grid_y);
  CoreDomain domain(CoreCoord(0, 0), CoreCoord(grid_x, grid_y));
  ffi::Array<TensorDescriptor> tensors;
  ffi::Array<DFBDescriptor> dfbs;
  ffi::Map<GlobalVar, BaseFunc> functions;
  PrimFunc trisc;
  PrimFunc ncrisc;
  PrimFunc brisc;
  Stmt idle = Evaluate(IntImm(DataType::Int(32), 0), frontend->span);
  if (tile_add_count == 0) {
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
    ncrisc = MakeSlotFunction(frontend, target.value(), domain, operation,
                              "ncrisc", "datamovement", Integer(0), {}, "idle",
                              "idle", idle);
  } else {
    AddDataflowPlan plan =
        AnalyzeAddDataflow(frontend, kernel_body, buffer_table.value());
    tensors = BuildTensorTable(frontend, buffer_table.value(),
                               {"input", "input", "output"},
                               /*allow_dfb_candidates=*/true);
    dfbs = BuildAddDFBTable(plan, domain, frontend);
    trisc = MakeSlotFunction(frontend, target.value(), domain, operation,
                             "trisc", "compute", /*noc_index=*/std::nullopt,
                             {}, "compute", "add", BuildAddTRISCBody(plan));
    ncrisc = MakeSlotFunction(
        frontend, target.value(), domain, operation, "ncrisc", "datamovement",
        Integer(0), {Integer(0), Integer(1), Integer(2)}, "datamovement",
        "tensor_io", BuildAddNCRISCBody(plan));
  }
  brisc = MakeSlotFunction(frontend, target.value(), domain, operation, "brisc",
                           "datamovement", Integer(1), {}, "idle", "idle",
                           idle);
  functions.Set(GlobalVar(operation + "_trisc"), std::move(trisc));
  functions.Set(GlobalVar(operation + "_ncrisc"), std::move(ncrisc));
  functions.Set(GlobalVar(operation + "_brisc"), std::move(brisc));

  ffi::Map<ffi::String, ffi::Any> attrs = {
      {kDeviceIRVersionAttr, Integer(kDeviceIRVersion)},
      {kTargetArchAttr, target_arch.value()},
      {kLaunchGridAttr, launch},
      {kOperationIdentityAttr,
       OperationIdentity(operation, std::move(source_span))},
      {kTensorTableAttr, tensors},
      {kDFBTableAttr, dfbs},
      {kPipeTableAttr, ffi::Array<PipeDescriptor>()},
      {kKernelOrderAttr, ffi::Array<ffi::String>({"trisc", "ncrisc", "brisc"})},
  };
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

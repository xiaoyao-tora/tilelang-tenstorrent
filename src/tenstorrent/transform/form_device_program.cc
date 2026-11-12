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
#include "device_metadata_interner.h"
#include "device_statement_compaction.h"
#include "device_table_compaction.h"
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
#include <cctype>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <set>
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

constexpr const char *kLogicalCoreAxis = "tt.logical_core_axis";

using BufferMetadataMap =
    std::unordered_map<Buffer, TTBufferMetadata, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;

// Transport payload buffers are declared once for the whole program so every
// Core role reuses one DFB identity and the storage table stays parameterized.
using TransportBufferCache =
    std::map<std::pair<int64_t, int64_t>, std::pair<Buffer, TTBufferMetadata>>;

[[noreturn]] void ThrowMalformed(const std::string &message) {
  TVM_FFI_THROW(ValueError)
      << "[FormTenstorrentDeviceProgram] malformed input: " << message;
}

[[noreturn]] void ThrowUnsupported(const std::string &message) {
  TVM_FFI_THROW(NotImplementedError)
      << "[FormTenstorrentDeviceProgram] unsupported program: " << message;
}

DataType RequireDTypeAnnotation(const Call &call, const char *key) {
  auto value = call->annotations.Get(key);
  if (!value.has_value() || !value.value().as<StringImmNode>())
    ThrowMalformed(std::string("accumulator operation requires '") + key +
                   "' string annotation");
  const ffi::String name = Downcast<StringImm>(value.value())->value;
  if (name == "bfloat16")
    return DataType::BFloat(16);
  if (name == "float32")
    return DataType::Float(32);
  ThrowMalformed(std::string("accumulator operation has unsupported '") + key +
                 "' dtype '" + std::string(name) + "'");
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
  return Span();
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

Stmt MakeDeviceCall(const Op &op, ffi::Array<PrimExpr> args, const Span &span) {
  return Evaluate(Call(DataType::Void(), op, std::move(args), {}, span), span);
}

Stmt MakeCommunicationDeviceCall(const Op &op, ffi::Array<PrimExpr> args,
                                 int64_t communication_id, const Span &span) {
  ffi::Map<ffi::String, ffi::ObjectRef> annotations{
      {"tt.internal_communication_id", Integer(communication_id)}};
  return Evaluate(
      Call(DataType::Void(), op, std::move(args), annotations, span), span);
}

bool IsPointPipeOperation(const CallNode *call) {
  return call->op.same_as(tenstorrent::dfb_pipe_send()) ||
         call->op.same_as(tenstorrent::dfb_pipe_recv()) ||
         call->op.same_as(tenstorrent::dfb_pipe_wait());
}

bool IsRecordPipeOperation(const CallNode *call) {
  return call->op.same_as(tenstorrent::dfb_pipe_send_record()) ||
         call->op.same_as(tenstorrent::dfb_pipe_recv_record()) ||
         call->op.same_as(tenstorrent::dfb_pipe_wait_record());
}

size_t PipeStreamOperand(const CallNode *call) {
  if (IsRecordPipeOperation(call))
    return 2;
  if (IsPointPipeOperation(call))
    return 1;
  ThrowMalformed("operation is not a Device Pipe operation");
}

// Assign an immutable generation to every write. Logical storage reuse is
// planned only after the complete cross-slot transaction schedule is known.
struct ResourceVersion {
  TTBufferMetadata metadata;
  ffi::String producer;
  ffi::String consumer;
  ffi::Optional<TensorBacking> backing;
  ffi::String storage_identity;
  int64_t generation{0};
  int64_t pool{-1};
  int64_t iteration{-1};
  int64_t transaction_count{1};
  int64_t loop_depth{0};
  bool loop_result{false};
  bool pipeline_repeated{false};
};

struct LoopCarriedValuePlan {
  int64_t state_id;
  int64_t entry_value;
  Buffer buffer;
  Span span;
};

struct StructuredSlotScope {
  ffi::Array<Stmt> compute;
  ffi::Array<Stmt> transfer;
  ffi::Array<Stmt> pipe;
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
class StaticValueSubstituter : public StmtExprMutator {
public:
  StaticValueSubstituter(Var variable, PrimExpr value)
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

class CoreProofEvaluator : public StmtExprMutator {
public:
  CoreProofEvaluator(int64_t x, int64_t y) : x_(x), y_(y) {}

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tenstorrent::core_x()))
      return make_const(op->dtype, x_);
    if (op->op.same_as(tenstorrent::core_y()))
      return make_const(op->dtype, y_);
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call.same_as(ffi::GetRef<Call>(op)))
      call.CopyOnWrite()->span = op->span;
    return call;
  }

  PrimExpr VisitExpr_(const CastNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    if (const auto *integer = value.as<IntImmNode>())
      return make_const(op->dtype, integer->value, op->span);
    return Cast(op->dtype, value, op->span);
  }

  PrimExpr VisitExpr(const PrimExpr &expr) final {
    PrimExpr result = StmtExprMutator::VisitExpr(expr);
    return result.dtype().is_handle() ||
                   SideEffect(result) > CallEffectKind::kPure
               ? result
               : analyzer_.Simplify(result);
  }

private:
  int64_t x_;
  int64_t y_;
  arith::Analyzer analyzer_;
};

class DFBIdentityRewriter : public StmtExprMutator {
public:
  DFBIdentityRewriter(int64_t source, int64_t destination)
      : source_(source), destination_(destination) {}

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    std::vector<size_t> positions;
    if (call->op.same_as(tenstorrent::dfb_compute())) {
      positions.resize(call->args.size());
      std::iota(positions.begin(), positions.end(), 0);
    } else if (call->op.same_as(tenstorrent::compute_value())) {
      positions.resize(call->args.size() > 0 ? call->args.size() - 1 : 0);
      std::iota(positions.begin(), positions.end(), 1);
    } else if (call->op.same_as(tenstorrent::dfb_reserve()) ||
               call->op.same_as(tenstorrent::dfb_wait()) ||
               call->op.same_as(tenstorrent::dfb_copy_wait()) ||
               call->op.same_as(tenstorrent::dfb_release()) ||
               call->op.same_as(tenstorrent::dfb_load()) ||
               call->op.same_as(tenstorrent::dfb_to_tensor_nd())) {
      positions = {0};
    } else if (call->op.same_as(tenstorrent::tensor_to_dfb_nd())) {
      positions = {1};
    } else if (IsPointPipeOperation(call.get()) ||
               IsRecordPipeOperation(call.get())) {
      positions = {PipeStreamOperand(call.get())};
    } else if (call->op.same_as(tenstorrent::compute_value_gemm())) {
      positions = {1, 2};
    } else if (call->op.same_as(tenstorrent::gemm_update()) ||
               call->op.same_as(tenstorrent::dfb_local_copy()) ||
               call->op.same_as(tenstorrent::dfb_gather_copy())) {
      positions = {0, 1};
    } else if (call->op.same_as(tenstorrent::compute_value_store()) ||
               call->op.same_as(tenstorrent::accumulator_materialize())) {
      positions = {1};
    } else if (call->op.same_as(tenstorrent::dfb_collective_reduce())) {
      positions = {0, 1, 2};
    }
    auto args = call->args;
    bool changed = false;
    for (size_t position : positions) {
      if (position >= args.size())
        continue;
      const int64_t *value = as_const_int(args[position]);
      if (value && *value == source_) {
        args.Set(position, Integer(destination_));
        changed = true;
      }
    }
    if (!changed)
      return call;
    return Call(call->dtype, call->op, args, call->annotations, call->span);
  }

  int64_t source_;
  int64_t destination_;
};

struct PipeEndpoint {
  int64_t transfer_id;
  int64_t occurrence;
  ffi::Array<Integer> record;
  int64_t x, y, dfb_id;
  bool source;
  bool producer_forwarding;
  int64_t transaction_count{1};
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
                         int64_t y = 0,
                         TransportBufferCache *transport_buffers = nullptr)
      : transfers_(transfers), core_x_(x), core_y_(y), frontend_(frontend),
        metadata_(IndexBufferMetadata(metadata)),
        transport_buffers_(transport_buffers),
        tensor_reads_(frontend->params.size(), false),
        tensor_writes_(frontend->params.size(), false) {
    auto grid = frontend_->GetAttr<ffi::Array<PrimExpr>>(kLaunchGridAttr);
    if (!grid.has_value() || grid.value().size() != 2)
      ThrowMalformed("planner requires normalized launch grid");
    grid_x_ = RequirePositiveStaticInteger(grid.value()[0], "launch grid x");
    grid_y_ = RequirePositiveStaticInteger(grid.value()[1], "launch grid y");
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
            name == "accumulator_materialize") {
          has_accumulator_compute_ = true;
          return;
        }
      }
      for (const PrimExpr &arg : call.value()->args) {
        BufferRegion region = NormalizeToAccessRegion(arg, kAccessRead).region;
        compute_value_mode_ |= region->buffer.scope() == "local.fragment";
      }
    });
    // Current shared-grid Device IR has one Kernel-level compute requirement.
    // Preserve compute values for mixed expression analysis, but never create
    // legacy v8 region-scoped precision modes inside one shared TRISC.
    precision_regions_ = false;
    compute_value_mode_ |= bf16_gemm && fp32_compute;
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
  ffi::String ResourceIdentity(int64_t id) const {
    if (id < 0 || id >= static_cast<int64_t>(resources_.size()))
      ThrowMalformed("resource identity index is out of range");
    const ResourceVersion &resource = resources_[id];
    return resource.metadata->buffer_id + ".generation" +
           std::to_string(resource.generation) + "." + resource.producer + "." +
           resource.consumer;
  }
  ffi::String ResourceDebugName(int64_t id) const {
    if (id < 0 || id >= static_cast<int64_t>(resources_.size()))
      ThrowMalformed("resource debug-name index is out of range");
    return resources_[id].metadata->buffer->name;
  }
  std::string ResourceStorageIdentity(int64_t id) const {
    if (id < 0 || id >= static_cast<int64_t>(resources_.size()))
      ThrowMalformed("resource storage identity index is out of range");
    const ResourceVersion &resource = resources_[id];
    const ffi::String &identity = resource.storage_identity.empty()
                                      ? resource.metadata->buffer_id
                                      : resource.storage_identity;
    return std::string(identity) + "|" + std::string(resource.producer) + "|" +
           std::string(resource.consumer);
  }
  bool ResourceIsLive(int64_t id) const { return live_.count(id) != 0; }
  int64_t AccumulatorCount() const { return accumulators_.size(); }
  bool HasProducerForwarding() const { return has_producer_forwarding_; }
  Stmt PipeBody() const { return Body(pipe_); }

  void FinalizeMulticore() {
    for (size_t id = 0; id < resources_.size(); ++id) {
      if (live_.count(id) && resources_[id].consumer.empty()) {
        resources_[id].consumer = "ncrisc";
        Wait(TransferStatements(), id, frontend_->span);
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

  // Preserve the source pipeline loop. The loop body owns one lexical stream
  // per transaction site; transaction epochs, rather than iteration-specific
  // resource IDs, carry the repeated schedule.
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
    RequireStaticInteger(loop->min, "T.Pipelined minimum");
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
    current_.clear();
    pipeline_value_begin_ = values_.size();
    PlanStructuredLoop(loop, pipeline_extent_);
    for (size_t index = pipeline_resource_begin_; index < resources_.size();
         ++index) {
      resources_[index].pool = index - pipeline_resource_begin_;
      resources_[index].pipeline_repeated = true;
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
    VisitDeviceCalls(Body(compute_), [&](const Call &typed) {
      const CallNode *call = typed.get();
      if (call->op.same_as(tenstorrent::dfb_compute())) {
        int64_t output = *as_const_int(call->args[0]);
        for (size_t i = 1; i < call->args.size(); ++i)
          inputs[output].push_back(*as_const_int(call->args[i]));
      } else if (call->op.same_as(tenstorrent::dfb_local_copy())) {
        inputs[*as_const_int(call->args[1])].push_back(
            *as_const_int(call->args[0]));
      } else if (call->op.same_as(tenstorrent::dfb_gather_copy())) {
        inputs[*as_const_int(call->args[1])].push_back(
            *as_const_int(call->args[0]));
      } else if (call->op.same_as(tenstorrent::dfb_collective_reduce())) {
        int64_t output = *as_const_int(call->args[2]);
        inputs[output].push_back(*as_const_int(call->args[0]));
        inputs[output].push_back(*as_const_int(call->args[1]));
      }
    });
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
    VisitDeviceCalls(Body(transfer_), [&](const Call &typed) {
      const CallNode *call = typed.get();
      if (call->op.same_as(tenstorrent::dfb_to_tensor_nd()))
        mark(*as_const_int(call->args[0]));
    });
    for (const PipeEndpoint &endpoint : endpoints_)
      mark(endpoint.dfb_id);
    for (int64_t id : observable_resources_)
      mark(id);
    auto keep = [&](const Call &typed) {
      const CallNode *call = typed.get();
      if (call->op.same_as(tenstorrent::accumulator_init()) ||
          call->op.same_as(tenstorrent::gemm_update()) ||
          call->op.same_as(tenstorrent::accumulator_materialize())) {
        size_t arg = call->op.same_as(tenstorrent::gemm_update()) ? 2 : 0;
        int64_t id = *as_const_int(call->args[arg]);
        return live.count(accumulators_[id].output_id) != 0;
      }
      size_t arg = call->op.same_as(tenstorrent::tensor_to_dfb_nd()) ? 1
                   : IsPointPipeOperation(call) || IsRecordPipeOperation(call)
                       ? PipeStreamOperand(call)
                       : 0;
      return live.count(*as_const_int(call->args[arg])) != 0;
    };
    compute_ = FilterDeviceCalls(compute_, keep);
    transfer_ = FilterDeviceCalls(transfer_, keep);
    live_ = std::move(live);
    std::fill(tensor_reads_.begin(), tensor_reads_.end(), false);
    for (const ffi::Array<Stmt> *body : {&transfer_, &pipe_})
      VisitDeviceCalls(Body(*body), [&](const Call &typed) {
        const CallNode *call = typed.get();
        if (call->op.same_as(tenstorrent::tensor_to_dfb_nd()))
          tensor_reads_[*as_const_int(call->args[0])] = true;
      });
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
          "structured control-flow planning exceeds 65536 statements");
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
      RequireStaticInteger(loop->min, "serial loop minimum");
      int64_t extent =
          RequirePositiveStaticInteger(loop->extent, "serial loop extent");
      if (extent > 1024)
        ThrowUnsupported("serial loop extent exceeds 1024");
      PlanStructuredLoop(ffi::GetRef<For>(loop), extent);
      return;
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      PrimExpr condition = EvaluateForCurrentCore(branch->condition);
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
    } else if (transfers_ && call->op.same_as(tenstorrent::transfer())) {
      PlanHighLevelTransfer(call);
    } else if (transfers_ && call->op.same_as(tenstorrent::all_gather())) {
      PlanAllGather(call);
    } else if (transfers_ && call->op.same_as(tenstorrent::all_reduce())) {
      PlanAllReduce(call);
    } else if (call->op.same_as(Copy::Get())) {
      PlanCopy(Downcast<Copy>(ParseOperator(call)), call);
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

  ffi::Array<Integer> UsedTensorIndices(const ffi::String &slot) const {
    ffi::Array<Integer> indices;
    auto accesses = TensorAccesses(slot);
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
          resource.producer, domain, resource.consumer, domain,
          Integer(resource.transaction_count),
          RequireSourceSpan(metadata->source_span, frontend_->span,
                            "DFB generation")));
    }
    return result;
  }

  Stmt ComputeBody() const { return Body(compute_); }
  Stmt TransferBody() const { return Body(transfer_); }

private:
  PrimExpr EvaluateForCurrentCore(const PrimExpr &value) const {
    return CoreProofEvaluator(core_x_, core_y_).VisitExpr(value);
  }

  int64_t RequireCurrentCoreInteger(const PrimExpr &value,
                                    const std::string &field) const {
    return RequireStaticInteger(EvaluateForCurrentCore(value), field);
  }

  ffi::Array<Stmt> *ComputeStatements() {
    return statement_scopes_.empty() ? &compute_
                                     : &statement_scopes_.back().compute;
  }

  ffi::Array<Stmt> *TransferStatements() {
    return statement_scopes_.empty() ? &transfer_
                                     : &statement_scopes_.back().transfer;
  }

  ffi::Array<Stmt> *PipeStatements() {
    return statement_scopes_.empty() ? &pipe_ : &statement_scopes_.back().pipe;
  }

  void AppendStructuredLoop(const For &source, StructuredSlotScope scope) {
    auto wrap = [&](ffi::Array<Stmt> body) -> Stmt {
      if (body.empty())
        return Evaluate(Integer(0), source->span);
      Stmt sequence = Body(body);
      Var variable = source->loop_var.copy_with_suffix("");
      sequence = Substitute(sequence, {{source->loop_var, variable}});
      return For(variable, source->min, source->extent, source->kind,
                 std::move(sequence), std::nullopt, {}, source->step,
                 source->span);
    };
    if (!scope.compute.empty())
      ComputeStatements()->push_back(wrap(std::move(scope.compute)));
    if (!scope.transfer.empty())
      TransferStatements()->push_back(wrap(std::move(scope.transfer)));
    if (!scope.pipe.empty())
      PipeStatements()->push_back(wrap(std::move(scope.pipe)));
  }

  static bool CallReadsDFB(const CallNode *call, int64_t resource) {
    auto matches = [&](size_t index) {
      return index < call->args.size() && as_const_int(call->args[index]) &&
             *as_const_int(call->args[index]) == resource;
    };
    if (call->op.same_as(tenstorrent::dfb_compute()) ||
        call->op.same_as(tenstorrent::compute_value())) {
      for (size_t index = 1; index < call->args.size(); ++index)
        if (matches(index))
          return true;
    } else if (call->op.same_as(tenstorrent::compute_value_gemm())) {
      return matches(1) || matches(2);
    } else if (call->op.same_as(tenstorrent::gemm_update()) ||
               call->op.same_as(tenstorrent::dfb_local_copy()) ||
               call->op.same_as(tenstorrent::dfb_gather_copy())) {
      return matches(0) || matches(1) || matches(2);
    } else if (call->op.same_as(tenstorrent::dfb_collective_reduce())) {
      return matches(0) || matches(1);
    } else if (call->op.same_as(tenstorrent::dfb_wait()) ||
               call->op.same_as(tenstorrent::dfb_copy_wait()) ||
               call->op.same_as(tenstorrent::dfb_to_tensor_nd())) {
      return matches(0);
    } else if (IsPointPipeOperation(call) || IsRecordPipeOperation(call)) {
      return matches(PipeStreamOperand(call));
    }
    return false;
  }

  static bool ScopeReadsDFB(const StructuredSlotScope &scope,
                            int64_t resource) {
    bool found = false;
    auto inspect = [&](const ffi::Array<Stmt> &body) {
      if (body.empty())
        return;
      Stmt root = body.size() == 1 ? body[0] : Stmt(SeqStmt(body));
      VisitDeviceCalls(root, [&](const Call &call) {
        found |= CallReadsDFB(call.get(), resource);
      });
    };
    inspect(scope.compute);
    inspect(scope.transfer);
    inspect(scope.pipe);
    return found;
  }

  StructuredSlotScope RewriteScopeDFB(StructuredSlotScope scope, int64_t source,
                                      int64_t destination) {
    auto rewrite = [&](ffi::Array<Stmt> body) {
      if (body.empty())
        return body;
      Stmt root = body.size() == 1 ? body[0] : Stmt(SeqStmt(body));
      Stmt rewritten = DFBIdentityRewriter(source, destination)(root);
      if (auto sequence = rewritten.as<SeqStmt>())
        return sequence.value()->seq;
      return ffi::Array<Stmt>{rewritten};
    };
    scope.compute = rewrite(std::move(scope.compute));
    scope.transfer = rewrite(std::move(scope.transfer));
    scope.pipe = rewrite(std::move(scope.pipe));
    return scope;
  }

  void HoistLoopInvariantWaits(StructuredSlotScope *scope,
                               size_t loop_resource_begin,
                               const std::set<int64_t> &recurrences) {
    auto hoist = [&](ffi::Array<Stmt> *body, ffi::Array<Stmt> *outer) {
      std::set<int64_t> resources;
      if (!body->empty()) {
        Stmt root = body->size() == 1 ? (*body)[0] : Stmt(SeqStmt(*body));
        VisitDeviceCalls(root, [&](const Call &call) {
          if (!call->op.same_as(tenstorrent::dfb_wait()) ||
              call->args.size() != 1)
            return;
          const int64_t *resource = as_const_int(call->args[0]);
          if (resource && *resource >= 0 &&
              *resource < static_cast<int64_t>(loop_resource_begin) &&
              !recurrences.count(*resource))
            resources.insert(*resource);
        });
      }
      if (resources.empty())
        return;
      *body = FilterDeviceCalls(*body, [&](const Call &call) {
        if (!call->op.same_as(tenstorrent::dfb_wait()) ||
            call->args.size() != 1)
          return true;
        const int64_t *resource = as_const_int(call->args[0]);
        return !resource || !resources.count(*resource);
      });
      for (int64_t resource : resources)
        Wait(outer, resource, resources_[resource].metadata->source_span);
    };
    hoist(&scope->compute, ComputeStatements());
    hoist(&scope->transfer, TransferStatements());
    hoist(&scope->pipe, PipeStatements());
  }

  void PlanStructuredLoop(const For &loop, int64_t extent) {
    const size_t resource_begin = resources_.size();
    const size_t endpoint_begin = endpoints_.size();
    const size_t value_begin = values_.size();
    auto entry_resources = current_;
    std::vector<int64_t> accumulator_tiles;
    accumulator_tiles.reserve(accumulators_.size());
    for (const AccumulatorLifetime &state : accumulators_)
      accumulator_tiles.push_back(state.full_k_tiles);

    std::vector<LoopCarriedValuePlan> loop_states;
    if (compute_value_mode_)
      loop_states = FindSymbolicLoopCarriedValuePlans(loop);
    if (IsPipeline() && !loop_states.empty())
      ThrowUnsupported(
          "T.Pipelined compute values must be independently initialized");
    for (const LoopCarriedValuePlan &state : loop_states)
      ComputeStatements()->push_back(MakeDeviceCall(
          compute_value_state_init(),
          {Integer(state.state_id), Integer(state.entry_value)}, state.span));

    bool previous_materialization = stable_loop_materializations_;
    stable_loop_materializations_ = true;
    ++structured_loop_depth_;
    statement_scopes_.push_back({});
    for (const LoopCarriedValuePlan &state : loop_states)
      DefineLoopStateInput(state);
    PlanImpl(loop->body);
    for (const LoopCarriedValuePlan &state : loop_states)
      YieldLoopState(state);
    StructuredSlotScope scope = std::move(statement_scopes_.back());
    statement_scopes_.pop_back();
    const size_t loop_resource_end = resources_.size();
    std::set<int64_t> recurrence_resources;

    for (const auto &[buffer, entry] : entry_resources) {
      auto result = current_.find(buffer);
      if (result == current_.end() || result->second == entry ||
          !ScopeReadsDFB(scope, entry))
        continue;
      int64_t output = result->second;
      if (entry < 0 || output < 0 ||
          entry >= static_cast<int64_t>(resources_.size()) ||
          output >= static_cast<int64_t>(resources_.size()))
        ThrowMalformed("structured DFB recurrence has an invalid resource");
      if (resources_[output].consumer.empty())
        resources_[output].consumer = resources_[entry].consumer;
      if (resources_[entry].metadata->buffer->dtype !=
              resources_[output].metadata->buffer->dtype ||
          !ffi::StructuralEqual()(resources_[entry].metadata->buffer->shape,
                                  resources_[output].metadata->buffer->shape))
        ThrowUnsupported("loop-carried DFB '" +
                         std::string(resources_[entry].metadata->buffer->name) +
                         "' changes shape or dtype");

      int64_t predecessor = entry;
      if (resources_[entry].producer != resources_[output].producer ||
          resources_[entry].consumer != resources_[output].consumer) {
        const ResourceVersion &next = resources_[output];
        ResourceVersion seed{next.metadata, next.producer, next.consumer,
                             std::nullopt};
        seed.generation = generations_[next.metadata->buffer->data]++;
        seed.storage_identity = next.storage_identity;
        seed.loop_depth = structured_loop_depth_;
        predecessor = resources_.size();
        resources_.push_back(std::move(seed));
        dfb_roots_[predecessor] = dfb_roots_[entry];

        Wait(ComputeStatements(), entry, loop->span);
        Reserve(ComputeStatements(), predecessor, loop->span);
        ffi::Map<ffi::String, ffi::ObjectRef> annotations{
            {"tt.compute_kind", StringImm("copy")},
            {"tt.compute_dtype",
             StringImm(resources_[predecessor].metadata->buffer->dtype ==
                               DataType::BFloat(16)
                           ? "bfloat16"
                           : "float32")},
            {"tt.compute_tile_shape",
             resources_[predecessor].metadata->tile_shape},
            {"tt.logical_domain",
             resources_[predecessor].metadata->buffer->shape},
            {"tt.input_shapes",
             ffi::Array<ffi::Array<PrimExpr>>{
                 resources_[entry].metadata->buffer->shape}}};
        ComputeStatements()->push_back(
            Evaluate(Call(DataType::Void(), tenstorrent::dfb_compute(),
                          {Integer(predecessor), Integer(entry)}, annotations,
                          loop->span),
                     loop->span));
        scope = RewriteScopeDFB(std::move(scope), entry, predecessor);
        for (size_t value = value_begin; value < value_dfb_sources_.size();
             ++value) {
          if (value_dfb_sources_[value].erase(entry))
            value_dfb_sources_[value].insert(predecessor);
        }
        for (AccumulatorLifetime &accumulator : accumulators_)
          for (int64_t &input : accumulator.inputs)
            if (input == entry)
              input = predecessor;
      }
      resources_[predecessor].loop_depth =
          std::max(resources_[predecessor].loop_depth, structured_loop_depth_);
      resources_[output].loop_result = true;
      recurrence_resources.insert(predecessor);
    }
    HoistLoopInvariantWaits(&scope, resource_begin, recurrence_resources);
    if (extent == 1) {
      for (const auto &[_, output] : current_) {
        if (output < static_cast<int64_t>(resource_begin) ||
            output >= static_cast<int64_t>(loop_resource_end) ||
            ScopeReadsDFB(scope, output))
          continue;
        ResourceVersion &result = resources_[output];
        if (result.consumer.empty())
          result.consumer = result.producer;
        result.loop_result = true;
      }
    }

    for (size_t id = resource_begin; id < loop_resource_end; ++id) {
      ResourceVersion &resource = resources_[id];
      if (resource.transaction_count >
          std::numeric_limits<int64_t>::max() / extent)
        ThrowUnsupported("structured loop DFB transaction count overflows");
      resource.transaction_count *= extent;
      resource.loop_depth =
          std::max(resource.loop_depth, structured_loop_depth_);
    }
    for (size_t index = endpoint_begin; index < endpoints_.size(); ++index) {
      PipeEndpoint &endpoint = endpoints_[index];
      if (endpoint.transaction_count >
          std::numeric_limits<int64_t>::max() / extent)
        ThrowUnsupported("structured loop transport count overflows");
      endpoint.transaction_count *= extent;
    }
    for (size_t id = 0; id < accumulator_tiles.size(); ++id) {
      int64_t before = accumulator_tiles[id];
      int64_t delta = accumulators_[id].full_k_tiles - before;
      if (delta > 0) {
        if (delta > (std::numeric_limits<int64_t>::max() - before) / extent)
          ThrowUnsupported("structured loop accumulator K extent overflows");
        accumulators_[id].full_k_tiles = before + delta * extent;
      }
    }

    --structured_loop_depth_;
    stable_loop_materializations_ = previous_materialization;
    AppendStructuredLoop(loop, std::move(scope));
    for (const LoopCarriedValuePlan &state : loop_states)
      DefineLoopStateResult(state);
  }

  std::vector<LoopCarriedValuePlan>
  FindSymbolicLoopCarriedValuePlans(const For &loop) {
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> reads;
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> writes;
    auto record = [&](const PrimExpr &value, bool read, bool write) {
      BufferRegion region =
          NormalizeToAccessRegion(value, read && write ? kAccessReadWrite
                                         : read        ? kAccessRead
                                                       : kAccessWrite)
              .region;
      if (region->buffer.scope() != "local.fragment")
        return;
      if (read)
        reads.insert(region->buffer->data);
      if (write)
        writes.insert(region->buffer->data);
    };
    PostOrderVisit(loop->body, [&](const ffi::ObjectRef &object) {
      auto call = object.as<Call>();
      if (!call.has_value() || !call.value()->op.same_as(tile_compute()) ||
          call.value()->args.empty())
        return;
      auto kind = call.value()->annotations.Get("tt.compute_kind");
      ffi::String name = kind.has_value()
                             ? Downcast<StringImm>(kind.value())->value
                             : ffi::String("");
      if (name == "accumulator_init") {
        record(call.value()->args[0], false, true);
        return;
      }
      if (name == "gemm_update") {
        record(call.value()->args[0], true, true);
        for (size_t index = 1; index < call.value()->args.size(); ++index)
          record(call.value()->args[index], true, false);
        return;
      }
      if (name == "accumulator_materialize") {
        record(call.value()->args[0], true, false);
        if (call.value()->args.size() > 1)
          record(call.value()->args[1], false, true);
        return;
      }
      record(call.value()->args[0], false, true);
      for (size_t index = 1; index < call.value()->args.size(); ++index)
        record(call.value()->args[index], true, false);
    });

    std::vector<LoopCarriedValuePlan> result;
    for (const auto &[buffer_var, entry_value] : current_values_) {
      if (!reads.count(buffer_var) || !writes.count(buffer_var))
        continue;
      if (entry_value < 0 ||
          entry_value >= static_cast<int64_t>(values_.size()))
        ThrowMalformed("loop-carried compute value is missing its descriptor");
      const ComputeValueDescriptor &entry = values_[entry_value];
      result.push_back({next_loop_state_id_++, entry_value, entry->buffer,
                        entry->source_span});
    }
    std::sort(result.begin(), result.end(),
              [](const auto &left, const auto &right) {
                return left.entry_value < right.entry_value;
              });
    return result;
  }

  static void
  VisitDeviceCalls(const Stmt &stmt,
                   const std::function<void(const Call &)> &visitor) {
    if (const auto *sequence = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : sequence->seq)
        VisitDeviceCalls(child, visitor);
      return;
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      VisitDeviceCalls(loop->body, visitor);
      return;
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      VisitDeviceCalls(branch->then_case, visitor);
      if (branch->else_case.has_value())
        VisitDeviceCalls(branch->else_case.value(), visitor);
      return;
    }
    if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
      VisitDeviceCalls(realize->block->body, visitor);
      return;
    }
    const auto *evaluate = stmt.as<EvaluateNode>();
    const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
    if (call)
      visitor(ffi::GetRef<Call>(call));
  }

  static Stmt FilterDeviceCalls(const Stmt &stmt,
                                const std::function<bool(const Call &)> &keep) {
    if (const auto *sequence = stmt.as<SeqStmtNode>()) {
      ffi::Array<Stmt> result;
      for (const Stmt &child : sequence->seq)
        result.push_back(FilterDeviceCalls(child, keep));
      return SeqStmt::Flatten(result);
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      return For(loop->loop_var, loop->min, loop->extent, loop->kind,
                 FilterDeviceCalls(loop->body, keep), loop->thread_binding,
                 loop->annotations, loop->step, loop->span);
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      ffi::Optional<Stmt> otherwise = std::nullopt;
      if (branch->else_case.has_value())
        otherwise = FilterDeviceCalls(branch->else_case.value(), keep);
      return IfThenElse(branch->condition,
                        FilterDeviceCalls(branch->then_case, keep), otherwise,
                        branch->span);
    }
    const auto *evaluate = stmt.as<EvaluateNode>();
    const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
    if (!call || keep(ffi::GetRef<Call>(call)))
      return stmt;
    return Evaluate(Integer(0), stmt->span);
  }

  static ffi::Array<Stmt>
  FilterDeviceCalls(const ffi::Array<Stmt> &body,
                    const std::function<bool(const Call &)> &keep) {
    Stmt source = body.empty()       ? Stmt(Evaluate(Integer(0)))
                  : body.size() == 1 ? body[0]
                                     : Stmt(SeqStmt(body));
    Stmt filtered = FilterDeviceCalls(source, keep);
    if (auto sequence = filtered.as<SeqStmt>())
      return sequence.value()->seq;
    return {filtered};
  }

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
    if (!IsPipeline() || !resource.pipeline_repeated)
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

  void AddStructuredReleases(ffi::Array<Stmt> *body, const ffi::String &slot) {
    std::map<int64_t, const StmtNode *> last_use;
    std::vector<const ForNode *> loop_stack;
    auto uses = [&](const CallNode *call) {
      std::vector<int64_t> result;
      auto append = [&](size_t index) {
        if (index >= call->args.size())
          ThrowMalformed("Device operation is missing a DFB operand");
        const int64_t *id = as_const_int(call->args[index]);
        if (!id)
          ThrowMalformed("pre-stream DFB operand must be static");
        result.push_back(*id);
      };
      if (call->op.same_as(tenstorrent::dfb_compute())) {
        for (size_t index = 1; index < call->args.size(); ++index)
          append(index);
      } else if (call->op.same_as(tenstorrent::compute_value()) ||
                 call->op.same_as(tenstorrent::compute_value_gemm())) {
        const int64_t *value = as_const_int(call->args[0]);
        if (!value || *value < 0 ||
            *value >= static_cast<int64_t>(value_dfb_sources_.size()))
          ThrowMalformed("compute value has an invalid DFB source relation");
        result.insert(result.end(), value_dfb_sources_[*value].begin(),
                      value_dfb_sources_[*value].end());
      } else if (call->op.same_as(tenstorrent::compute_value_store())) {
        const int64_t *value = as_const_int(call->args[0]);
        if (!value || *value < 0 ||
            *value >= static_cast<int64_t>(value_dfb_sources_.size()))
          ThrowMalformed("compute-value store has an invalid value identity");
        result.insert(result.end(), value_dfb_sources_[*value].begin(),
                      value_dfb_sources_[*value].end());
      } else if (call->op.same_as(tenstorrent::compute_value_state_init()) ||
                 call->op.same_as(tenstorrent::compute_value_state_yield())) {
        const int64_t *value = as_const_int(call->args[1]);
        if (value && *value >= 0 &&
            *value < static_cast<int64_t>(value_dfb_sources_.size()))
          result.insert(result.end(), value_dfb_sources_[*value].begin(),
                        value_dfb_sources_[*value].end());
      } else if (call->op.same_as(tenstorrent::compute_value_state_input()) ||
                 call->op.same_as(tenstorrent::compute_value_state_result())) {
        const int64_t *value = as_const_int(call->args[0]);
        if (value && *value >= 0 &&
            *value < static_cast<int64_t>(value_dfb_sources_.size()))
          result.insert(result.end(), value_dfb_sources_[*value].begin(),
                        value_dfb_sources_[*value].end());
      } else if (call->op.same_as(tenstorrent::gemm_update())) {
        append(0);
        append(1);
      } else if (call->op.same_as(tenstorrent::dfb_local_copy()) ||
                 call->op.same_as(tenstorrent::dfb_gather_copy())) {
        append(0);
      } else if (call->op.same_as(tenstorrent::dfb_collective_reduce())) {
        append(0);
        append(1);
      } else if (call->op.same_as(tenstorrent::dfb_pipe_wait()) ||
                 call->op.same_as(tenstorrent::dfb_pipe_wait_record())) {
        size_t index = PipeStreamOperand(call);
        const int64_t *id = as_const_int(call->args[index]);
        if (id && resources_[*id].consumer == slot)
          append(index);
      } else if (call->op.same_as(tenstorrent::dfb_wait()) && transfers_) {
        const int64_t *id = as_const_int(call->args[0]);
        if (id && resources_[*id].consumer == slot)
          append(0);
      } else if (call->op.same_as(tenstorrent::dfb_copy_wait())) {
        const int64_t *id = as_const_int(call->args[0]);
        if (id && resources_[*id].consumer == slot)
          append(0);
      }
      auto value_inputs = call->annotations.Get("tt.value_inputs");
      if (value_inputs.has_value()) {
        for (const PrimExpr &value_expr :
             Downcast<ffi::Array<PrimExpr>>(value_inputs.value())) {
          const int64_t *value = as_const_int(value_expr);
          if (!value || *value < 0 ||
              *value >= static_cast<int64_t>(value_dfb_sources_.size()))
            ThrowMalformed("Device compute has an invalid value input");
          result.insert(result.end(), value_dfb_sources_[*value].begin(),
                        value_dfb_sources_[*value].end());
        }
      }
      return result;
    };

    std::function<void(const Stmt &, int64_t)> scan = [&](const Stmt &stmt,
                                                          int64_t depth) {
      if (const auto *sequence = stmt.as<SeqStmtNode>()) {
        for (const Stmt &child : sequence->seq)
          scan(child, depth);
        return;
      }
      if (const auto *loop = stmt.as<ForNode>()) {
        loop_stack.push_back(loop);
        scan(loop->body, depth + 1);
        loop_stack.pop_back();
        return;
      }
      if (const auto *branch = stmt.as<IfThenElseNode>()) {
        scan(branch->then_case, depth);
        if (branch->else_case.has_value())
          scan(branch->else_case.value(), depth);
        return;
      }
      const auto *evaluate = stmt.as<EvaluateNode>();
      const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
      if (!call)
        return;
      for (int64_t id : uses(call)) {
        if (id < 0 || id >= static_cast<int64_t>(resources_.size()))
          ThrowMalformed("Device operation references an unknown DFB");
        int64_t owner_depth = resources_[id].loop_depth;
        if (owner_depth > depth && !resources_[id].loop_result)
          ThrowUnsupported("loop-local DFB '" +
                           std::string(resources_[id].metadata->buffer->name) +
                           "' (resource " + std::to_string(id) + ", depth " +
                           std::to_string(owner_depth) +
                           ") escapes to structured depth " +
                           std::to_string(depth));
        const StmtNode *target = stmt.get();
        if (owner_depth < depth) {
          if (owner_depth >= static_cast<int64_t>(loop_stack.size()))
            ThrowMalformed("structured DFB release has no enclosing loop");
          target = loop_stack[owner_depth];
        }
        last_use[id] = target;
      }
    };
    Stmt root = Body(*body);
    scan(root, 0);

    std::unordered_map<const StmtNode *, std::vector<int64_t>> releases;
    for (const auto &[id, target] : last_use)
      releases[target].push_back(id);
    std::function<Stmt(const Stmt &)> rewrite = [&](const Stmt &stmt) -> Stmt {
      Stmt result = stmt;
      if (const auto *sequence = stmt.as<SeqStmtNode>()) {
        ffi::Array<Stmt> children;
        for (const Stmt &child : sequence->seq)
          children.push_back(rewrite(child));
        result = SeqStmt::Flatten(children);
      } else if (const auto *loop = stmt.as<ForNode>()) {
        result = For(loop->loop_var, loop->min, loop->extent, loop->kind,
                     rewrite(loop->body), loop->thread_binding,
                     loop->annotations, loop->step, loop->span);
      } else if (const auto *branch = stmt.as<IfThenElseNode>()) {
        ffi::Optional<Stmt> otherwise = std::nullopt;
        if (branch->else_case.has_value())
          otherwise = rewrite(branch->else_case.value());
        result = IfThenElse(branch->condition, rewrite(branch->then_case),
                            otherwise, branch->span);
      }
      auto found = releases.find(stmt.get());
      if (found == releases.end())
        return result;
      ffi::Array<Stmt> sequence{result};
      for (int64_t id : found->second)
        sequence.push_back(MakeDeviceCall(tenstorrent::dfb_release(),
                                          {Integer(id)}, stmt->span));
      return SeqStmt::Flatten(sequence);
    };
    Stmt released = rewrite(root);
    if (auto sequence = released.as<SeqStmt>())
      *body = sequence.value()->seq;
    else
      *body = {released};
  }

  void AddReleases(ffi::Array<Stmt> *body, const ffi::String &slot) {
    AddStructuredReleases(body, slot);
  }

  std::vector<int> TensorAccesses(const ffi::Array<Stmt> &body) const {
    std::vector<int> accesses(frontend_->params.size(), 0);
    VisitDeviceCalls(Body(body), [&](const Call &typed) {
      const CallNode *call = typed.get();
      if (call->op.same_as(tenstorrent::tensor_to_dfb_nd()))
        accesses[*as_const_int(call->args[0])] |= 1;
      if (call->op.same_as(tenstorrent::dfb_to_tensor_nd()))
        accesses[*as_const_int(call->args[1])] |= 2;
    });
    return accesses;
  }

  std::vector<int> TensorAccesses(const ffi::String &slot) const {
    if (slot == "ncrisc")
      return TensorAccesses(transfer_);
    if (slot == "brisc")
      return TensorAccesses(pipe_);
    return std::vector<int>(frontend_->params.size(), 0);
  }

  std::vector<int> TensorAccesses() const {
    std::vector<int> accesses = TensorAccesses("ncrisc");
    std::vector<int> brisc = TensorAccesses("brisc");
    for (size_t index = 0; index < accesses.size(); ++index)
      accesses[index] |= brisc[index];
    return accesses;
  }

  void MoveTensorLoadToPipe(int64_t id) {
    if (!tensor_load_args_.count(id))
      return;
    ResourceVersion &resource = resources_.at(id);
    if (resource.producer != "ncrisc" || !resource.consumer.empty())
      ThrowUnsupported(
          "loopback source Tensor load is not an unconsumed NCRISC transaction");

    ffi::Array<Stmt> retained, moved;
    int stage = 0;
    for (const Stmt &statement : *TransferStatements()) {
      const auto *evaluate = statement.as<EvaluateNode>();
      const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
      bool match = false;
      if (call && call->op.same_as(tenstorrent::dfb_reserve()) &&
          call->args.size() == 1 && as_const_int(call->args[0]) &&
          *as_const_int(call->args[0]) == id) {
        if (stage != 0)
          ThrowMalformed("loopback Tensor load reserve is out of order");
        stage = 1;
        match = true;
      } else if (call && call->op.same_as(tenstorrent::tensor_to_dfb_nd()) &&
                 call->args.size() >= 2 && as_const_int(call->args[1]) &&
                 *as_const_int(call->args[1]) == id) {
        if (stage != 1)
          ThrowMalformed("loopback Tensor load issue is out of order");
        stage = 2;
        match = true;
      } else if (call && call->op.same_as(tenstorrent::dfb_copy_wait()) &&
                 call->args.size() == 1 && as_const_int(call->args[0]) &&
                 *as_const_int(call->args[0]) == id) {
        if (stage != 2)
          ThrowMalformed("loopback Tensor load completion is out of order");
        stage = 3;
        match = true;
      }
      if (match)
        moved.push_back(statement);
      else
        retained.push_back(statement);
    }
    if (stage != 3 || moved.size() != 3)
      ThrowUnsupported(
          "loopback source requires one complete Tensor load transaction");
    *TransferStatements() = std::move(retained);
    for (const Stmt &statement : moved)
      PipeStatements()->push_back(statement);
    resource.producer = "brisc";
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
    resources_.back().generation = generations_[buffer->data]++;
    if (publish_current)
      current_[buffer->data] = id;
    return id;
  }

  void Wait(ffi::Array<Stmt> *body, int64_t id, const Span &span) {
    body->push_back(
        MakeDeviceCall(tenstorrent::dfb_wait(), {Integer(id)}, span));
  }
  void Reserve(ffi::Array<Stmt> *body, int64_t id, const Span &span) {
    body->push_back(
        MakeDeviceCall(tenstorrent::dfb_reserve(), {Integer(id)}, span));
  }

  int64_t DefineLoopStateInput(const LoopCarriedValuePlan &state) {
    int64_t previous = ReadValue(state.buffer, state.span);
    int64_t value = DefineValue(state.buffer, values_[previous]->accumulator_id,
                                state.span);
    value_roots_[value] = value_roots_[previous];
    ComputeStatements()->push_back(
        MakeDeviceCall(compute_value_state_input(),
                       {Integer(value), Integer(state.state_id)}, state.span));
    return value;
  }

  void YieldLoopState(const LoopCarriedValuePlan &state) {
    int64_t value = ReadValue(state.buffer, state.span);
    ComputeStatements()->push_back(
        MakeDeviceCall(compute_value_state_yield(),
                       {Integer(state.state_id), Integer(value)}, state.span));
  }

  void DefineLoopStateResult(const LoopCarriedValuePlan &state) {
    int64_t previous = ReadValue(state.buffer, state.span);
    int64_t value = DefineValue(state.buffer, values_[previous]->accumulator_id,
                                state.span);
    value_roots_[value] = value_roots_[previous];
    ComputeStatements()->push_back(
        MakeDeviceCall(compute_value_state_result(),
                       {Integer(value), Integer(state.state_id)}, state.span));
  }

  int64_t ReadValue(const Buffer &buffer, const Span &span) {
    auto found = current_values_.find(buffer->data);
    if (found == current_values_.end())
      ThrowMalformed("fragment read-before-definition for buffer '" +
                     std::string(buffer->name) + "'");
    if (IsPipeline() && found->second < pipeline_value_begin_)
      ThrowUnsupported("T.Pipelined fragment '" + std::string(buffer->name) +
                       "' carries a value across iterations; bounded prefetch "
                       "requires independently initialized fragment values");
    int64_t value = found->second;
    const Buffer &defined = values_[value]->buffer;
    if (defined->dtype != buffer->dtype ||
        !ffi::StructuralEqual()(defined->shape, buffer->shape))
      ThrowMalformed("fragment alias must preserve shape and dtype");
    return value;
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
    value_dfb_sources_.push_back({});
    if (accumulator >= 0)
      value_roots_.back().insert(accumulator);
    current_values_[buffer->data] = id;
    return id;
  }

  int64_t
  MakeValueStorage(const Buffer &buffer, const Span &span,
                   ffi::Optional<TensorBacking> backing = std::nullopt) {
    int64_t id = resources_.size();
    int64_t site_id = id;
    ffi::String storage_identity;
    if (stable_loop_materializations_) {
      auto source = metadata_.find(buffer);
      Span fallback = source == metadata_.end() ? buffer->span
                                                : source->second->source_span;
      Span source_span =
          RequireSourceSpan(span, fallback, "compute materialization site");
      std::string site_key =
          source == metadata_.end()
              ? "synthetic." + std::string(buffer->name) + "." +
                    std::to_string(buffer->dtype.code()) + "." +
                    std::to_string(buffer->dtype.bits()) + "." +
                    std::to_string(buffer->dtype.lanes()) + "." +
                    std::to_string(buffer->shape.size())
              : std::string(source->second->buffer_id);
      site_key += "|";
      if (source_span->source_name.defined())
        site_key += std::string(source_span->source_name->name);
      site_key += ":" + std::to_string(source_span->line) + ":" +
                  std::to_string(source_span->column) + ":" +
                  std::to_string(source_span->end_line) + ":" +
                  std::to_string(source_span->end_column) + "|" +
                  (backing.has_value() ? "tensor" : "local");
      site_id =
          materialization_sites_
              .emplace(site_key,
                       static_cast<int64_t>(materialization_sites_.size()))
              .first->second;
      storage_identity =
          "compute.materialization.site." + std::to_string(site_id);
    }
    Buffer storage =
        decl_buffer(buffer->shape, buffer->dtype,
                    buffer->name + "_materialized_" + std::to_string(site_id),
                    "shared", std::nullopt, span);
    ffi::Array<PrimExpr> grid;
    for (const PrimExpr &extent : buffer->shape)
      grid.push_back(Integer((RequirePositiveStaticInteger(
                                  extent, "compute materialization extent") +
                              31) /
                             32));
    TTBufferMetadata metadata(
        "compute.materialization." + std::to_string(id), storage,
        "logical_dfb_candidate", std::nullopt, {Integer(32), Integer(32)}, grid,
        "interleaved", std::nullopt, Integer(1), std::nullopt, std::nullopt,
        "inferred", "inferred", "inferred", "inferred", span);
    resources_.push_back({metadata, "trisc", "", backing});
    resources_.back().storage_identity = storage_identity;
    resources_.back().generation = generations_[buffer->data]++;
    return id;
  }

  int64_t MaterializeValue(int64_t value, const Span &span) {
    auto found = materialized_values_.find(value);
    if (found != materialized_values_.end())
      return found->second;
    int64_t id = MakeValueStorage(values_[value]->buffer, span);
    resources_[id].consumer = "trisc";
    Reserve(ComputeStatements(), id, span);
    ComputeStatements()->push_back(
        MakeDeviceCall(tenstorrent::compute_value_store(),
                       {Integer(value), Integer(id)}, span));
    materialized_values_[value] = id;
    dfb_roots_[id] = value_roots_[value];
    return id;
  }

  int64_t ReadDFBOperand(const Buffer &buffer, const Span &span) {
    if (buffer.scope() == "local.fragment")
      return MaterializeValue(ReadValue(buffer, span), span);
    return Read(buffer, "trisc");
  }

  void MergeAccumulatorContract(const Call &call, const Buffer &buffer,
                                AccumulatorLifetime *state) {
    DataType input = RequireDTypeAnnotation(call, "tt.input_dtype");
    DataType accumulation = RequireDTypeAnnotation(call, "tt.accum_dtype");
    DataType output = RequireDTypeAnnotation(call, "tt.output_dtype");
    if (accumulation != buffer->dtype)
      ThrowMalformed(
          "accumulator operation accumulation dtype disagrees with its buffer");
    if (state->input_dtype != DataType::Void() && state->input_dtype != input)
      ThrowMalformed("accumulator operation input dtype contract changed");
    if (state->output_dtype != DataType::Void() &&
        state->output_dtype != output)
      ThrowMalformed("accumulator operation output dtype contract changed");
    state->input_dtype = input;
    state->output_dtype = output;
  }

  void PlanValueAccumulator(const Call &call, const ffi::String &kind) {
    BufferRegion region =
        NormalizeToAccessRegion(call->args[0], kAccessReadWrite).region;
    FullRegion(region);
    const Buffer &buffer = region->buffer;
    auto found = accumulator_ids_.find(buffer);
    if (kind == "accumulator_init") {
      if (call->args.size() != 1)
        ThrowMalformed("accumulator_init requires one accumulator region");
      int64_t accumulator = accumulators_.size();
      accumulator_ids_[buffer] = accumulator;
      AccumulatorLifetime state;
      state.region = region;
      state.span = call->span;
      MergeAccumulatorContract(call, buffer, &state);
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
          {"tt.input_shapes", ffi::Array<ffi::Array<PrimExpr>>()},
          {"tt.value_inputs", ffi::Array<Integer>()},
          {"tt.value_input_shapes", ffi::Array<ffi::Array<PrimExpr>>()}};
      ComputeStatements()->push_back(
          Evaluate(Call(DataType::Void(), compute_value(), {Integer(value)},
                        attrs, call->span),
                   call->span));
      return;
    }
    if (found == accumulator_ids_.end())
      ThrowMalformed("accumulator update requires a dominating initialization");
    int64_t accumulator = found->second;
    auto &state = accumulators_[accumulator];
    MergeAccumulatorContract(call, buffer, &state);
    if (kind == "accumulator_materialize") {
      if (call->args.size() != 2)
        ThrowMalformed(
            "accumulator_materialize requires accumulator and output regions");
      BufferRegion output =
          NormalizeToAccessRegion(call->args[1], kAccessWrite).region;
      if (output->buffer->dtype != state.output_dtype)
        ThrowMalformed(
            "accumulator materialization dtype disagrees with its contract");
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
    if (call->args.size() != 3)
      ThrowMalformed("gemm_update requires accumulator, lhs and rhs regions");
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
    int64_t old = ReadValue(buffer, call->span);
    int64_t value = DefineValue(buffer, accumulator, call->span);
    value_roots_[value] = value_roots_[old];
    value_roots_[value].insert(accumulator);
    value_dfb_sources_[value] = value_dfb_sources_[old];
    value_dfb_sources_[value].insert(a);
    value_dfb_sources_[value].insert(b);
    if (lhs->buffer->dtype != state.input_dtype ||
        rhs->buffer->dtype != state.input_dtype)
      ThrowMalformed("GEMM inputs disagree with accumulator dtype contract");
    state.full_k_tiles +=
        RequirePositiveStaticInteger(lhs->buffer->shape[ta ? 0 : 1], "GEMM K") /
        32;
    state.inputs.push_back(a);
    state.inputs.push_back(b);
    Wait(ComputeStatements(), a, call->span);
    Wait(ComputeStatements(), b, call->span);
    ComputeStatements()->push_back(
        MakeDeviceCall(compute_value_gemm(),
                       {Integer(value), Integer(a), Integer(b), Integer(old),
                        Integer(ta), Integer(tb)},
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
    ComputeStatements()->push_back(
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
      value_dfb_sources_[value].insert(dfb);
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
          {"tt.input_shapes", ffi::Array<ffi::Array<PrimExpr>>{buffer->shape}},
          {"tt.value_inputs", ffi::Array<Integer>()},
          {"tt.value_input_shapes", ffi::Array<ffi::Array<PrimExpr>>()}};
      Wait(ComputeStatements(), dfb, call->span);
      ComputeStatements()->push_back(
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
    ffi::Array<ffi::Array<PrimExpr>> dfb_shapes, value_shapes;
    std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
        dfb_inputs, local_inputs;
    bool dfb_only = kind == "gemm" || kind == "transpose" || kind == "reduce";
    std::unordered_set<int64_t> roots;
    std::unordered_set<int64_t> dfb_sources;
    for (size_t index = 1; index < call->args.size(); ++index) {
      BufferRegion input =
          NormalizeToAccessRegion(call->args[index], kAccessRead).region;
      if (dfb_only)
        FullRegion(input);
      ffi::Array<PrimExpr> input_shape;
      for (const Range &range : input->region)
        input_shape.push_back(range->extent);
      if (input->buffer.scope() == "local.fragment" && !dfb_only) {
        int64_t value = ReadValue(input->buffer, call->span);
        local_inputs.emplace(input->buffer, value);
        value_inputs.push_back(Integer(value));
        value_shapes.push_back(input_shape);
        roots.insert(value_roots_[value].begin(), value_roots_[value].end());
        dfb_sources.insert(value_dfb_sources_[value].begin(),
                           value_dfb_sources_[value].end());
      } else {
        int64_t id = ReadDFBOperand(input->buffer, call->span);
        dfb_inputs.emplace(input->buffer, id);
        dfb_args.push_back(Integer(id));
        dfb_shapes.push_back(input_shape);
        const auto &input_roots = dfb_roots_[id];
        roots.insert(input_roots.begin(), input_roots.end());
        dfb_sources.insert(id);
        Wait(ComputeStatements(), id, call->span);
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
    attrs.Set("tt.input_shapes", dfb_shapes);
    attrs.Set("tt.value_inputs", value_inputs);
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
        Reserve(ComputeStatements(), storage, call->span);
        ffi::Array<PrimExpr> operation_args{Integer(storage)};
        for (const PrimExpr &arg : dfb_args)
          operation_args.push_back(arg);
        ComputeStatements()->push_back(
            Evaluate(Call(DataType::Void(), dfb_compute(), operation_args,
                          attrs, call->span),
                     call->span));
        Wait(ComputeStatements(), storage, call->span);
        ffi::Map<ffi::String, ffi::ObjectRef> load_attrs{
            {"tt.compute_kind", StringImm("elementwise")},
            {"tt.compute_dtype", attrs.at("tt.compute_dtype")},
            {"tt.compute_tile_shape", attrs.at("tt.compute_tile_shape")},
            {"tt.logical_domain", domain},
            {"tt.expression", Call(output->buffer->dtype, dfb_load(),
                                   {Integer(storage)}, {}, call->span)},
            {"tt.input_shapes", ffi::Array<ffi::Array<PrimExpr>>{domain}},
            {"tt.value_inputs", ffi::Array<Integer>()},
            {"tt.value_input_shapes", ffi::Array<ffi::Array<PrimExpr>>()}};
        ComputeStatements()->push_back(Evaluate(
            Call(DataType::Void(), compute_value(),
                 {Integer(value), Integer(storage)}, load_attrs, call->span),
            call->span));
        materialized_values_[value] = storage;
        value_dfb_sources_[value].insert(storage);
        return;
      }
      ffi::Array<PrimExpr> args{Integer(value)};
      for (const PrimExpr &arg : dfb_args)
        args.push_back(arg);
      ComputeStatements()->push_back(Evaluate(
          Call(DataType::Void(), compute_value(), args, attrs, call->span),
          call->span));
      value_dfb_sources_[value] = std::move(dfb_sources);
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
      Wait(TransferStatements(), output_id, call->span);
      ffi::Array<PrimExpr> args{Integer(output_id), Integer(tensor)};
      for (const Range &range : output->region) {
        args.push_back(range->min);
        args.push_back(range->extent);
      }
      TransferStatements()->push_back(
          MakeDeviceCall(dfb_to_tensor_nd(), args, call->span));
      if (transfers_)
        TransferStatements()->push_back(
            MakeDeviceCall(dfb_copy_wait(), {Integer(output_id)}, call->span));
    } else {
      output_id = Write(output->buffer, "trisc");
    }
    dfb_roots_[output_id] = roots;
    Reserve(ComputeStatements(), output_id, call->span);
    ffi::Array<PrimExpr> args{Integer(output_id)};
    for (const PrimExpr &arg : dfb_args)
      args.push_back(arg);
    ComputeStatements()->push_back(
        Evaluate(Call(DataType::Void(), dfb_compute(), args, attrs, call->span),
                 call->span));
  }

  void EliminateDeadValueWrites() {
    std::unordered_map<int64_t, std::vector<int64_t>> inputs;
    std::unordered_set<int64_t> live;
    // Compute values are immutable definitions. Retain their predecessor chains
    // and all input lifetimes even when a value only feeds another fragment.
    VisitDeviceCalls(Body(compute_), [&](const Call &typed) {
      const CallNode *call = typed.get();
      if (call->op.same_as(dfb_compute())) {
        int64_t out = *as_const_int(call->args[0]);
        for (size_t i = 1; i < call->args.size(); ++i)
          inputs[out].push_back(*as_const_int(call->args[i]));
      }
    });
    std::function<void(int64_t)> mark = [&](int64_t id) {
      if (!live.insert(id).second)
        return;
      for (int64_t input : inputs[id])
        mark(input);
    };
    VisitDeviceCalls(Body(compute_), [&](const Call &typed) {
      const CallNode *call = typed.get();
      if (call->op.same_as(compute_value())) {
        for (size_t i = 1; i < call->args.size(); ++i)
          mark(*as_const_int(call->args[i]));
      } else if (call->op.same_as(compute_value_gemm())) {
        mark(*as_const_int(call->args[1]));
        mark(*as_const_int(call->args[2]));
      }
    });
    VisitDeviceCalls(Body(transfer_), [&](const Call &typed) {
      const CallNode *call = typed.get();
      if (call->op.same_as(dfb_to_tensor_nd()))
        mark(*as_const_int(call->args[0]));
    });
    for (const PipeEndpoint &endpoint : endpoints_)
      mark(endpoint.dfb_id);
    auto keep = [&](const Call &typed) {
      const CallNode *call = typed.get();
      if (call->op.same_as(compute_value()) ||
          call->op.same_as(compute_value_gemm()) ||
          call->op.same_as(compute_precision()) ||
          call->op.same_as(compute_value_state_init()) ||
          call->op.same_as(compute_value_state_input()) ||
          call->op.same_as(compute_value_state_yield()) ||
          call->op.same_as(compute_value_state_result())) {
        return true;
      }
      size_t arg = call->op.same_as(tensor_to_dfb_nd()) ||
                           call->op.same_as(compute_value_store())
                       ? 1
                   : IsPointPipeOperation(call) || IsRecordPipeOperation(call)
                       ? PipeStreamOperand(call)
                       : 0;
      return live.count(*as_const_int(call->args[arg])) != 0;
    };
    compute_ = FilterDeviceCalls(compute_, keep);
    transfer_ = FilterDeviceCalls(transfer_, keep);
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
      int64_t id = Read(input->buffer, "trisc");
      inputs.emplace(input->buffer, id);
      args.push_back(Integer(id));
      Wait(ComputeStatements(), id, call->span);
    }
    int64_t output_id = Write(output->buffer, "trisc");
    Reserve(ComputeStatements(), output_id, call->span);
    ffi::Array<PrimExpr> device_args{Integer(output_id)};
    for (const PrimExpr &arg : args)
      device_args.push_back(arg);
    auto annotations = call->annotations;
    auto expression = annotations.Get("tt.expression");
    if (expression.has_value()) {
      std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash,
                         ffi::ObjectPtrEqual>
          no_values;
      DeviceExpressionRewriter rewriter(inputs, no_values);
      annotations.Set("tt.expression",
                      rewriter(Downcast<PrimExpr>(expression.value())));
    }
    ComputeStatements()->push_back(
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
      if (call->args.size() != 1)
        ThrowMalformed("accumulator_init requires one accumulator region");
      if (found != accumulator_ids_.end() &&
          accumulators_[found->second].output_id < 0)
        ThrowMalformed(
            "accumulator cannot be reinitialized before materialization");
      if (RequireMetadata(metadata_, buffer, "accumulator")->kind !=
          "compute_fragment")
        ThrowMalformed("accumulator must reference compute_fragment metadata");
      int64_t id = accumulators_.size();
      accumulator_ids_[buffer] = id;
      AccumulatorLifetime state;
      state.region = region;
      state.span = call->span;
      MergeAccumulatorContract(call, buffer, &state);
      accumulators_.push_back(std::move(state));
      ComputeStatements()->push_back(MakeDeviceCall(
          tenstorrent::accumulator_init(), {Integer(id)}, call->span));
      return;
    }
    if (found == accumulator_ids_.end())
      ThrowMalformed(
          "accumulator update/materialization requires initialization");
    int64_t id = found->second;
    auto &state = accumulators_[id];
    MergeAccumulatorContract(call, buffer, &state);
    if (state.output_id >= 0)
      ThrowMalformed("accumulator access after final materialization");
    if (kind == "gemm_update") {
      if (call->args.size() != 3)
        ThrowMalformed("gemm_update requires accumulator, lhs and rhs regions");
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
      if (lhs->buffer->dtype != state.input_dtype ||
          rhs->buffer->dtype != state.input_dtype)
        ThrowMalformed("GEMM inputs disagree with accumulator dtype contract");
      state.full_k_tiles +=
          RequirePositiveStaticInteger(lhs->buffer->shape[transpose_a ? 0 : 1],
                                       "GEMM K") /
          32;
      state.inputs.push_back(a);
      state.inputs.push_back(b);
      Wait(ComputeStatements(), a, call->span);
      Wait(ComputeStatements(), b, call->span);
      ComputeStatements()->push_back(
          MakeDeviceCall(tenstorrent::gemm_update(),
                         {Integer(a), Integer(b), Integer(id),
                          Integer(transpose_a), Integer(transpose_b)},
                         call->span));
      return;
    }
    if (state.full_k_tiles <= 0)
      ThrowMalformed("accumulator cannot materialize before GEMM updates");
    if (call->args.size() != 2)
      ThrowMalformed(
          "accumulator_materialize requires accumulator and output regions");
    BufferRegion output =
        NormalizeToAccessRegion(call->args[1], kAccessWrite).region;
    if (output->buffer->dtype != state.output_dtype)
      ThrowMalformed(
          "accumulator materialization dtype disagrees with its contract");
    const auto &metadata =
        RequireMetadata(metadata_, output->buffer, "accumulator output");
    int64_t output_id;
    if (metadata->kind == "logical_dfb_candidate") {
      FullRegion(output);
      output_id = Write(output->buffer, "trisc");
    } else if (metadata->kind == "tensor") {
      // Materialization allocates output storage only once, after complete K.
      ffi::String materialized_id = ".materialized." + std::to_string(id);
      Buffer storage =
          decl_buffer(buffer->shape, output->buffer->dtype,
                      buffer->name + "_materialized_" + std::to_string(id),
                      "shared", std::nullopt, call->span);
      ICHECK(!storage->data.same_as(buffer->data));
      const auto &fragment = RequireMetadata(metadata_, buffer, "accumulator");
      TTBufferMetadata output_metadata(
          fragment->buffer_id + materialized_id, storage,
          "logical_dfb_candidate", std::nullopt, fragment->tile_shape,
          fragment->tile_grid_shape, "interleaved", std::nullopt, Integer(1),
          std::nullopt, std::nullopt, "inferred", "inferred", "inferred",
          "inferred", call->span);
      int64_t tensor = metadata->global_arg_index.value()->value;
      output_id = resources_.size();
      resources_.push_back({output_metadata, "trisc", "ncrisc",
                            TensorBacking(tensor, Integer(0))});
      resources_.back().generation = generations_[storage->data]++;
      tensor_writes_[tensor] = true;
      Wait(TransferStatements(), output_id, call->span);
      ffi::Array<PrimExpr> args{Integer(output_id), Integer(tensor)};
      for (const Range &range : output->region) {
        args.push_back(range->min);
        args.push_back(range->extent);
      }
      TransferStatements()->push_back(
          MakeDeviceCall(tenstorrent::dfb_to_tensor_nd(), args, call->span));
      if (transfers_)
        TransferStatements()->push_back(MakeDeviceCall(
            tenstorrent::dfb_copy_wait(), {Integer(output_id)}, call->span));
    } else {
      ThrowMalformed("accumulator must materialize to shared DFB or Tensor");
    }
    state.output_id = output_id;
    Reserve(ComputeStatements(), output_id, call->span);
    ComputeStatements()->push_back(
        MakeDeviceCall(tenstorrent::accumulator_materialize(),
                       {Integer(id), Integer(output_id)}, call->span));
  }

  void PlanCopy(const Copy &copy, const Call &call) {
    BufferRegion source(copy->src, copy->src_range);
    BufferRegion destination(copy->dst, copy->dst_range);
    bool rank_mapped_tensor =
        source->region.size() != destination->region.size() &&
        (copy->src.scope() == "global" || copy->dst.scope() == "global");
    if ((!transfers_ && !has_accumulator_compute_ && !IsPipeline() &&
         !rank_mapped_tensor) ||
        copy->src.scope() == "shared" || copy->src.scope() == "shared.dyn")
      FullRegion(source);
    if ((!transfers_ && !has_accumulator_compute_ && !IsPipeline() &&
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
      Reserve(TransferStatements(), id, call->span);
      ffi::Array<PrimExpr> args{Integer(tensor), Integer(id)};
      for (const Range &range : source->region) {
        args.push_back(range->min);
        args.push_back(range->extent);
      }
      // Keep the load shape so a pipe send can re-issue the same panel into its
      // transport buffer without sharing the compute DFB generation.
      tensor_load_args_[id] = args;
      TransferStatements()->push_back(
          MakeDeviceCall(tenstorrent::tensor_to_dfb_nd(), args, call->span));
      if (transfers_)
        TransferStatements()->push_back(MakeDeviceCall(
            tenstorrent::dfb_copy_wait(), {Integer(id)}, call->span));
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
      Wait(ComputeStatements(), input, call->span);
      Reserve(ComputeStatements(), output, call->span);
      ffi::Map<ffi::String, ffi::ObjectRef> annotations{
          {"tt.compute_kind", StringImm("copy")},
          {"tt.compute_dtype",
           StringImm(copy->src->dtype == DataType::BFloat(16) ? "bfloat16"
                                                              : "float32")},
          {"tt.compute_tile_shape", src->tile_shape},
          {"tt.logical_domain", copy->src->shape},
          {"tt.input_shapes",
           ffi::Array<ffi::Array<PrimExpr>>{copy->src->shape}}};
      ComputeStatements()->push_back(Evaluate(
          Call(DataType::Void(), tenstorrent::dfb_compute(),
               {Integer(output), Integer(input)}, annotations, call->span),
          call->span));
      Wait(TransferStatements(), output, call->span);
      ffi::Array<PrimExpr> args{Integer(output), Integer(tensor)};
      for (const Range &range : destination->region) {
        args.push_back(range->min);
        args.push_back(range->extent);
      }
      TransferStatements()->push_back(
          MakeDeviceCall(tenstorrent::dfb_to_tensor_nd(), args, call->span));
      if (transfers_)
        TransferStatements()->push_back(MakeDeviceCall(
            tenstorrent::dfb_copy_wait(), {Integer(output)}, call->span));
      return;
    }
    ThrowUnsupported("Copy must connect Tensor and DFB; shared Copy requires "
                     "compute legalization");
  }

  void PlanHighLevelTransfer(const Call &call) {
    if (call->args.size() != 10)
      ThrowMalformed("high-level transfer arity mismatch");
    int64_t sx =
        RequireCurrentCoreInteger(call->args[2], "transfer source x");
    int64_t sy =
        RequireCurrentCoreInteger(call->args[3], "transfer source y");
    int64_t bx =
        RequireCurrentCoreInteger(call->args[4], "transfer begin x");
    int64_t by =
        RequireCurrentCoreInteger(call->args[5], "transfer begin y");
    int64_t ex =
        RequireCurrentCoreInteger(call->args[6], "transfer end x");
    int64_t ey =
        RequireCurrentCoreInteger(call->args[7], "transfer end y");
    int64_t site = RequireStaticInteger(call->args[9], "transfer site_id");
    const auto *contract = call->args[8].as<StringImmNode>();
    if (!contract)
      ThrowMalformed("high-level transfer contract is not a string");
    bool source = core_x_ == sx && core_y_ == sy;
    bool destination =
        bx <= core_x_ && core_x_ < ex && by <= core_y_ && core_y_ < ey;
    if (!source && !destination)
      return;
    ffi::Array<Integer> record{Integer(site),
                               Integer(sy * grid_x_ + sx),
                               Integer(contract->value == "collective" ? 1 : 0),
                               Integer(sx),
                               Integer(sy),
                               Integer(bx),
                               Integer(by),
                               Integer(ex),
                               Integer(ey)};
    ffi::Map<ffi::String, ffi::ObjectRef> annotations{
        {"tt.pipe_record", record}};
    if (source) {
      Call endpoint(DataType::Void(), tenstorrent::pipe_send(),
                    {call->args[0], Integer(0)}, annotations, call->span);
      // When the source is also a destination, the existing source generation
      // is the private send staging transaction. The receive below creates a
      // new generation for the uniform compute stream.
      PlanPipe(endpoint, /*reuse_source_generation=*/destination);
    }
    if (destination) {
      Call endpoint(DataType::Void(), tenstorrent::pipe_recv(),
                    {Integer(0), call->args[1]}, annotations, call->span);
      PlanPipe(endpoint);
    }
  }

  int64_t GroupRank(int64_t x, int64_t y, int64_t bx, int64_t by,
                    int64_t ex) const {
    return (y - by) * (ex - bx) + (x - bx);
  }

  ffi::Array<Integer> DirectRecord(int64_t site, int64_t record_index,
                                   int64_t sx, int64_t sy, int64_t dx,
                                   int64_t dy) const {
    return {Integer(site), Integer(record_index), Integer(0),
            Integer(sx),   Integer(sy),           Integer(dx),
            Integer(dy),   Integer(dx + 1),       Integer(dy + 1)};
  }

  void PlanDirectEndpoint(bool send, const BufferRegion &region,
                          const ffi::Array<Integer> &record, const Span &span) {
    ffi::Map<ffi::String, ffi::ObjectRef> annotations{
        {"tt.pipe_record", record}};
    Call endpoint(DataType::Void(),
                  send ? tenstorrent::pipe_send() : tenstorrent::pipe_recv(),
                  send ? ffi::Array<PrimExpr>{region->ToPrimExpr(), Integer(0)}
                       : ffi::Array<PrimExpr>{Integer(0), region->ToPrimExpr()},
                  annotations, span);
    PlanPipe(endpoint);
  }

  Stmt MakeCollectiveLocalCall(const Op &op, ffi::Array<PrimExpr> args,
                               int64_t site, int64_t occurrence,
                               const ffi::String &kind,
                               const Span &span) const {
    ffi::Map<ffi::String, ffi::ObjectRef> annotations{
        {"tt.communication_site", Integer(site)},
        {"tt.communication_occurrence", Integer(occurrence)},
        {"tt.communication_local_kind", StringImm(kind)},
        {"tt.core_x", Integer(core_x_)},
        {"tt.core_y", Integer(core_y_)}};
    return Evaluate(
        Call(DataType::Void(), op, std::move(args), annotations, span), span);
  }

  Buffer MakeCollectiveTemporary(const BufferRegion &payload, int64_t site,
                                 int64_t occurrence, int64_t source_rank,
                                 const Span &span) {
    const Buffer &source = payload->buffer;
    Buffer storage = decl_buffer(
        source->shape, source->dtype,
        "tt_collective_" + std::to_string(site) + "_" +
            std::to_string(occurrence) + "_" + std::to_string(source_rank) +
            "_" + std::to_string(generated_collective_buffers_++),
        "shared", std::nullopt, span);
    ffi::Array<PrimExpr> tile_shape{Integer(32), Integer(32)};
    ffi::Array<PrimExpr> tile_grid;
    for (const PrimExpr &extent : source->shape)
      tile_grid.push_back(Integer(
          RequirePositiveStaticInteger(extent, "collective payload extent") /
          32));
    TTBufferMetadata metadata(
        "collective.temp." + std::to_string(generated_collective_buffers_),
        storage, "logical_dfb_candidate", std::nullopt, tile_shape, tile_grid,
        "interleaved", std::nullopt, Integer(1), std::nullopt, std::nullopt,
        "inferred", "inferred", "unset", "inferred", span);
    metadata_.emplace(storage, metadata);
    return storage;
  }

  // A dedicated transport buffer keeps the compute panel and the payload as
  // independent single-consumer generations in every Core role, so BRISC can
  // own the inter-core transfer without sharing a DFB with TRISC computation.
  Buffer MakeTransportBuffer(const Buffer &source, int64_t site,
                             int64_t occurrence, const Span &span) {
    auto key = std::make_pair(site, occurrence);
    TransportBufferCache &cache =
        transport_buffers_ ? *transport_buffers_ : local_transport_buffers_;
    auto found = cache.find(key);
    if (found != cache.end()) {
      // Another Core already declared this payload; reuse its DFB identity so
      // the storage table can stay parameterized across the launch grid.
      metadata_.emplace(found->second.first, found->second.second);
      return found->second.first;
    }
    const TTBufferMetadata &metadata =
        RequireMetadata(metadata_, source, "transport payload");
    const std::string name = "tt_transport_" + std::to_string(site) + "_" +
                             std::to_string(occurrence);
    Buffer storage = decl_buffer(source->shape, source->dtype, name, "shared",
                                 std::nullopt, span);
    TTBufferMetadata transport(
        "transport.temp." + name, storage, "logical_dfb_candidate",
        std::nullopt, metadata->tile_shape, metadata->tile_grid_shape,
        "interleaved", std::nullopt, metadata->dfb_block_count, std::nullopt,
        std::nullopt, "inferred", "inferred", "unset", "inferred", span);
    metadata_.emplace(storage, transport);
    cache.emplace(key, std::make_pair(storage, transport));
    return storage;
  }

  void PlanAllGather(const Call &call) {
    if (call->args.size() != 8)
      ThrowMalformed("all_gather arity mismatch");
    BufferRegion source =
        NormalizeToAccessRegion(call->args[0], kAccessRead).region;
    BufferRegion destination =
        NormalizeToAccessRegion(call->args[1], kAccessWrite).region;
    FullRegion(source);
    FullRegion(destination);
    int64_t bx =
        RequireCurrentCoreInteger(call->args[2], "all_gather begin x");
    int64_t by =
        RequireCurrentCoreInteger(call->args[3], "all_gather begin y");
    int64_t ex =
        RequireCurrentCoreInteger(call->args[4], "all_gather end x");
    int64_t ey =
        RequireCurrentCoreInteger(call->args[5], "all_gather end y");
    int64_t axis = RequireStaticInteger(call->args[6], "all_gather axis");
    int64_t site = RequireStaticInteger(call->args[7], "all_gather site_id");
    if (!(bx <= core_x_ && core_x_ < ex && by <= core_y_ && core_y_ < ey))
      return;
    int64_t occurrence = collective_occurrences_[site]++;
    int64_t group_size = (ex - bx) * (ey - by);
    int64_t local_rank = GroupRank(core_x_, core_y_, bx, by, ex);

    for (int64_t dy = by; dy < ey; ++dy)
      for (int64_t dx = bx; dx < ex; ++dx) {
        int64_t destination_rank = GroupRank(dx, dy, bx, by, ex);
        if (destination_rank == local_rank)
          continue;
        int64_t record_index = local_rank * group_size + destination_rank;
        PlanDirectEndpoint(
            true, source,
            DirectRecord(site, record_index, core_x_, core_y_, dx, dy),
            call->span);
      }

    int64_t output = Write(destination->buffer, "trisc");
    observable_resources_.insert(output);
    Reserve(ComputeStatements(), output, call->span);
    for (int64_t sy = by; sy < ey; ++sy)
      for (int64_t sx = bx; sx < ex; ++sx) {
        int64_t source_rank = GroupRank(sx, sy, bx, by, ex);
        int64_t input;
        if (source_rank == local_rank) {
          input = Read(source->buffer, "trisc");
        } else {
          Buffer temporary = MakeCollectiveTemporary(source, site, occurrence,
                                                     source_rank, call->span);
          BufferRegion temporary_region = BufferRegion::FullRegion(temporary);
          int64_t record_index = source_rank * group_size + local_rank;
          PlanDirectEndpoint(
              false, temporary_region,
              DirectRecord(site, record_index, sx, sy, core_x_, core_y_),
              call->span);
          input = Read(temporary, "trisc");
        }
        Wait(ComputeStatements(), input, call->span);
        ComputeStatements()->push_back(MakeCollectiveLocalCall(
            tenstorrent::dfb_gather_copy(),
            {Integer(input), Integer(output), Integer(axis),
             Integer(source_rank), Integer(group_size)},
            site, occurrence, "local_copy", call->span));
      }
  }

  void PlanAllReduce(const Call &call) {
    if (call->args.size() != 8)
      ThrowMalformed("all_reduce arity mismatch");
    BufferRegion source =
        NormalizeToAccessRegion(call->args[0], kAccessRead).region;
    BufferRegion destination =
        NormalizeToAccessRegion(call->args[1], kAccessWrite).region;
    FullRegion(source);
    FullRegion(destination);
    int64_t bx =
        RequireCurrentCoreInteger(call->args[2], "all_reduce begin x");
    int64_t by =
        RequireCurrentCoreInteger(call->args[3], "all_reduce begin y");
    int64_t ex =
        RequireCurrentCoreInteger(call->args[4], "all_reduce end x");
    int64_t ey =
        RequireCurrentCoreInteger(call->args[5], "all_reduce end y");
    const auto *reduction = call->args[6].as<StringImmNode>();
    if (!reduction)
      ThrowMalformed("all_reduce reduction operation is not a string");
    int64_t site = RequireStaticInteger(call->args[7], "all_reduce site_id");
    if (!(bx <= core_x_ && core_x_ < ex && by <= core_y_ && core_y_ < ey))
      return;
    int64_t occurrence = collective_occurrences_[site]++;
    int64_t group_size = (ex - bx) * (ey - by);
    int64_t local_rank = GroupRank(core_x_, core_y_, bx, by, ex);
    const int64_t root_x = bx, root_y = by;

    if (local_rank != 0) {
      PlanDirectEndpoint(
          true, source,
          DirectRecord(site, local_rank, core_x_, core_y_, root_x, root_y),
          call->span);
      PlanDirectEndpoint(false, destination,
                         DirectRecord(site, group_size + local_rank, root_x,
                                      root_y, core_x_, core_y_),
                         call->span);
      return;
    }

    int64_t output = Write(destination->buffer, "trisc");
    observable_resources_.insert(output);
    Reserve(ComputeStatements(), output, call->span);
    int64_t local = Read(source->buffer, "trisc");
    Wait(ComputeStatements(), local, call->span);
    ComputeStatements()->push_back(MakeCollectiveLocalCall(
        tenstorrent::dfb_local_copy(), {Integer(local), Integer(output)}, site,
        occurrence, "local_copy", call->span));
    for (int64_t sy = by; sy < ey; ++sy)
      for (int64_t sx = bx; sx < ex; ++sx) {
        int64_t source_rank = GroupRank(sx, sy, bx, by, ex);
        if (source_rank == 0)
          continue;
        Buffer temporary = MakeCollectiveTemporary(source, site, occurrence,
                                                   source_rank, call->span);
        BufferRegion temporary_region = BufferRegion::FullRegion(temporary);
        PlanDirectEndpoint(
            false, temporary_region,
            DirectRecord(site, source_rank, sx, sy, root_x, root_y),
            call->span);
        int64_t input = Read(temporary, "trisc");
        Wait(ComputeStatements(), input, call->span);
        ComputeStatements()->push_back(MakeCollectiveLocalCall(
            tenstorrent::dfb_collective_reduce(),
            {Integer(output), Integer(input), Integer(output), call->args[6]},
            site, occurrence, "local_reduce", call->span));
      }
    for (int64_t dy = by; dy < ey; ++dy)
      for (int64_t dx = bx; dx < ex; ++dx) {
        int64_t destination_rank = GroupRank(dx, dy, bx, by, ex);
        if (destination_rank == 0)
          continue;
        PlanDirectEndpoint(true, destination,
                           DirectRecord(site, group_size + destination_rank,
                                        root_x, root_y, dx, dy),
                           call->span);
      }
  }

  // NCRISC owns every payload move: a Tensor-backed panel is re-issued into
  // the transport buffer so the compute panel and the payload keep exactly one
  // consumer slot each, and any other payload is copied locally inside L1.
  int64_t FillTransportPayload(const Buffer &transport,
                               const BufferRegion &region, int64_t site,
                               int64_t occurrence, const Span &span) {
    auto current = current_.find(region->buffer->data);
    if (current == current_.end())
      ThrowMalformed("Pipe send payload has no producing definition");
    const ResourceVersion &source = resources_[current->second];
    auto loaded = tensor_load_args_.find(current->second);
    if (source.backing.has_value() && loaded != tensor_load_args_.end()) {
      // A Tensor-backed panel is local compute input and transport payload at
      // the same time. Re-issue the load into the transport buffer so both
      // generations keep exactly one consumer slot.
      int64_t id = Write(transport, "ncrisc", source.backing);
      resources_[id].consumer = "brisc";
      Reserve(TransferStatements(), id, span);
      const ffi::Array<PrimExpr> &original = loaded->second;
      ffi::Array<PrimExpr> args;
      args.push_back(original[0]);
      args.push_back(Integer(id));
      for (size_t index = 2; index < original.size(); ++index)
        args.push_back(original[index]);
      TransferStatements()->push_back(MakeDeviceCall(
          tenstorrent::tensor_to_dfb_nd(), std::move(args), span));
      TransferStatements()->push_back(
          MakeDeviceCall(tenstorrent::dfb_copy_wait(), {Integer(id)}, span));
      return id;
    }
    if (IsPipeline())
      ThrowUnsupported("pipelined transport payload requires a Tensor-backed "
                       "source panel");
    // The payload already lives in the slot that produced it, so stage it from
    // that same slot: a generation may only ever have one consumer slot, and
    // the local copy stays inside L1 (no DRAM traffic).
    const ffi::String slot = source.producer;
    ffi::Array<Stmt> *body =
        slot == "trisc" ? ComputeStatements() : TransferStatements();
    int64_t input = Read(region->buffer, slot);
    int64_t id = Write(transport, slot);
    resources_[id].consumer = "brisc";
    Reserve(body, id, span);
    Wait(body, input, span);
    if (body == ComputeStatements()) {
      // A compute-slot payload move is a plain local copy, not a collective
      // contribution: keep it out of the all_gather/all_reduce local-kind
      // accounting by expressing it as dfb_compute.
      const TTBufferMetadata &metadata = resources_[id].metadata;
      ffi::Map<ffi::String, ffi::ObjectRef> annotations{
          {"tt.compute_kind", StringImm("copy")},
          {"tt.compute_dtype",
           StringImm(transport->dtype == DataType::BFloat(16) ? "bfloat16"
                                                              : "float32")},
          {"tt.compute_tile_shape", metadata->tile_shape},
          {"tt.logical_domain", transport->shape},
          {"tt.input_shapes",
           ffi::Array<ffi::Array<PrimExpr>>{transport->shape}}};
      body->push_back(
          Evaluate(Call(DataType::Void(), tenstorrent::dfb_compute(),
                        {Integer(id), Integer(input)}, annotations, span),
                   span));
    } else {
      body->push_back(MakeDeviceCall(tenstorrent::dfb_local_copy(),
                                     {Integer(input), Integer(id)}, span));
    }
    return id;
  }

  void PlanPipe(const Call &call, bool reuse_source_generation = false) {
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
    // Each endpoint's lexical occurrence names the same K-stage transaction.
    // Separate source/receiver counters also handle self-delivery.
    auto occurrence_key =
        std::make_tuple(record[0]->value, record[1]->value, send);
    int64_t occurrence = occurrences_[occurrence_key]++;
    // Source-inclusive delivery reuses the current source generation as the
    // private BRISC send staging transaction; its destination path creates a
    // new NCRISC-produced generation for uniform compute. Other sends retain a
    // dedicated transport buffer so one generation never gains two consumers.
    int64_t id;
    if (send) {
      if (reuse_source_generation) {
        auto current = current_.find(region->buffer->data);
        if (current == current_.end())
          ThrowMalformed("loopback source has no current DFB generation");
        MoveTensorLoadToPipe(current->second);
        id = Read(region->buffer, "brisc");
      } else {
        Buffer transport = MakeTransportBuffer(
            region->buffer, record[0]->value, occurrence, call->span);
        // NCRISC owns every payload move and BRISC owns the inter-core
        // transfer.
        id = FillTransportPayload(transport, region, record[0]->value,
                                  occurrence, call->span);
      }
      Wait(PipeStatements(), id, call->span);
    } else {
      // Destination work stays on NCRISC, matching TT-Lang's if_dst affinity.
      // The received payload lands directly in the consuming generation, so
      // no local staging copy is introduced.
      id = Write(region->buffer, "ncrisc");
      Reserve(TransferStatements(), id, call->span);
    }
    int64_t bx = send ? record[5]->value : core_x_;
    int64_t by = send ? record[6]->value : core_y_;
    int64_t ex = send ? record[7]->value : core_x_ + 1;
    int64_t ey = send ? record[8]->value : core_y_ + 1;
    for (int64_t x = bx; x < ex; ++x) {
      for (int64_t y = by; y < ey; ++y) {
        auto key = std::make_tuple(record[0]->value, record[1]->value,
                                   occurrence, x, y);
        auto found = transfers_->find(key);
        int64_t site = record[0]->value, record_index = record[1]->value;
        if (site < 0 || site >= 128 || record_index < 0 ||
            record_index >= 1024 || occurrence < 0 || occurrence >= 1024)
          ThrowUnsupported("communication family index exceeds the supported "
                           "site/record/occurrence domain: site=" +
                           std::to_string(site) +
                           ", record=" + std::to_string(record_index) +
                           ", occurrence=" + std::to_string(occurrence));
        int64_t destination_rank = x * grid_y_ + y;
        int64_t grid_size = grid_x_ * grid_y_;
        __int128 identity =
            ((static_cast<__int128>(site) * 1024 + record_index) * 1024 +
             occurrence) *
                grid_size +
            destination_rank;
        if (identity < 0 || identity > std::numeric_limits<int64_t>::max())
          ThrowUnsupported("communication transfer identity exceeds int64");
        int64_t transfer = static_cast<int64_t>(identity);
        if (found != transfers_->end() && found->second != transfer)
          ThrowMalformed("communication transfer identity is not stable");
        (*transfers_)[key] = transfer;
        // BRISC owns source work and NCRISC owns destination work. Neither
        // endpoint forwards one generation to multiple consumer slots.
        endpoints_.push_back({transfer, occurrence, record, x, y, id, send,
                              false, 1, call->span});
      }
    }
    int64_t site = record[0]->value, record_index = record[1]->value;
    auto *body = send ? PipeStatements() : TransferStatements();
    body->push_back(MakeCommunicationDeviceCall(
        send ? tenstorrent::dfb_pipe_send_record()
             : tenstorrent::dfb_pipe_recv_record(),
        {Integer(site), Integer(record_index), Integer(id)}, site, call->span));
    body->push_back(MakeCommunicationDeviceCall(
        tenstorrent::dfb_pipe_wait_record(),
        {Integer(site), Integer(record_index), Integer(id)}, site, call->span));
  }

  std::vector<Call> *collected_calls_{nullptr};
  size_t call_index_{0};
  std::unordered_map<Var, std::vector<FragmentAccess>, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      fragment_accesses_;
  std::vector<size_t> unknown_accesses_;
  bool compute_value_mode_{false};
  bool has_accumulator_compute_{false};
  bool precision_regions_{false};
  int precision_mode_{-1};
  bool stable_loop_materializations_{false};
  ffi::Array<ComputeValueDescriptor> values_;
  std::vector<std::unordered_set<int64_t>> value_roots_;
  std::vector<std::unordered_set<int64_t>> value_dfb_sources_;
  std::unordered_map<int64_t, std::unordered_set<int64_t>> dfb_roots_;
  std::unordered_map<Var, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      current_values_;
  std::map<std::string, int64_t> materialization_sites_;
  std::unordered_map<int64_t, int64_t> materialized_values_;
  std::vector<AccumulatorLifetime> accumulators_;
  std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      accumulator_ids_;
  std::map<std::tuple<int64_t, int64_t, bool>, int64_t> occurrences_;
  std::unordered_map<int64_t, int64_t> collective_occurrences_;
  int64_t generated_collective_buffers_{0};
  std::unordered_set<int64_t> observable_resources_;
  bool has_producer_forwarding_{false};
  TransportBufferCache *transport_buffers_{nullptr};
  TransportBufferCache local_transport_buffers_;
  std::unordered_map<int64_t, ffi::Array<PrimExpr>> tensor_load_args_;
  int64_t next_loop_state_id_{0};
  TransferKeys *transfers_;
  int64_t core_x_, core_y_, grid_x_{1}, grid_y_{1};
  std::vector<PipeEndpoint> endpoints_;
  ffi::Array<Stmt> pipe_;
  std::vector<StructuredSlotScope> statement_scopes_;
  int64_t structured_loop_depth_{0};
  PrimFunc frontend_;
  BufferMetadataMap metadata_;
  std::unordered_map<Var, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      current_;
  std::unordered_map<Var, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      generations_;
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
std::vector<For> FindIndependentLoops(const Stmt &body) {
  Stmt candidate = body;
  while (const auto *realize = candidate.as<SBlockRealizeNode>()) {
    if (!realize->iter_values.empty() || !is_one(realize->predicate) ||
        !realize->block->iter_vars.empty() ||
        realize->block->init.has_value() ||
        !realize->block->match_buffers.empty()) {
      return {};
    }
    candidate = realize->block->body;
  }
  std::vector<For> loops;
  while (auto loop = candidate.as<For>()) {
    For value = loop.value();
    if (value->kind != ForKind::kSerial || !value->annotations.empty() ||
        (value->step.has_value() && !is_one(value->step.value())) ||
        !as_const_int(value->min) || !as_const_int(value->extent) ||
        *as_const_int(value->extent) <= 0 ||
        *as_const_int(value->extent) > 1024)
      break;
    loops.push_back(value);
    candidate = value->body;
    while (const auto *realize = candidate.as<SBlockRealizeNode>()) {
      if (!realize->iter_values.empty() || !is_one(realize->predicate) ||
          !realize->block->iter_vars.empty() ||
          realize->block->init.has_value() ||
          !realize->block->match_buffers.empty())
        return loops;
      candidate = realize->block->body;
    }
  }
  return loops;
}

Stmt WrapIndependentLoops(Stmt body, const std::vector<For> &loops) {
  if (const auto *evaluate = body.as<EvaluateNode>()) {
    if (is_zero(evaluate->value))
      return body;
  }
  for (auto it = loops.rbegin(); it != loops.rend(); ++it) {
    const For &loop = *it;
    Var variable = loop->loop_var.copy_with_suffix("");
    body = Substitute(body, {{loop->loop_var, variable}});
    body = For(variable, loop->min, loop->extent, ForKind::kSerial,
               std::move(body), std::nullopt, {}, loop->step, loop->span);
  }
  return body;
}

PrimExpr IndependentIterationCount(const std::vector<For> &loops) {
  PrimExpr count = Integer(1);
  for (const For &loop : loops)
    count = count * loop->extent;
  return count;
}

Stmt IndependentLoopBody(const std::vector<For> &loops, const Stmt &fallback) {
  return loops.empty() ? fallback : loops.back()->body;
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
  DeviceIDRemapper(int64_t offset,
                   const std::vector<int64_t> *resource_ids = nullptr,
                   bool annotate_streams = false)
      : offset_(offset), resource_ids_(resource_ids),
        annotate_streams_(annotate_streams) {}
  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    auto args = call->args;
    std::vector<int64_t> referenced_resources;
    auto shift = [&](size_t index) {
      int64_t id = *as_const_int(args[index]);
      if (resource_ids_) {
        if (id < 0 || id >= static_cast<int64_t>(resource_ids_->size()) ||
            resource_ids_->at(id) < 0)
          ThrowMalformed("Device operation references an uninterned DFB ID");
        id = resource_ids_->at(id);
      } else {
        id += offset_;
      }
      args.Set(index, IntImm(args[index].dtype(), id,
                             args[index]->span.defined() ? args[index]->span
                                                         : call->span));
      referenced_resources.push_back(id);
    };
    if (call->op.same_as(compute_value())) {
      for (size_t index = 1; index < args.size(); ++index)
        shift(index);
    } else if (call->op.same_as(compute_value_gemm())) {
      shift(1);
      shift(2);
    } else if (call->op.same_as(compute_value_store())) {
      shift(1);
    } else if (call->op.same_as(tenstorrent::accumulator_init()) ||
               call->op.same_as(tenstorrent::accumulator_materialize())) {
      if (call->op.same_as(tenstorrent::accumulator_materialize()))
        shift(1);
    } else if (call->op.same_as(tenstorrent::gemm_update())) {
      shift(0);
      shift(1);
    } else if (call->op.same_as(tenstorrent::dfb_compute())) {
      for (size_t index = 0; index < args.size(); ++index)
        shift(index);
    } else if (call->op.same_as(tenstorrent::tensor_to_dfb_nd()) ||
               call->op.same_as(tenstorrent::dfb_pipe_send()) ||
               call->op.same_as(tenstorrent::dfb_pipe_recv()) ||
               call->op.same_as(tenstorrent::dfb_pipe_wait())) {
      shift(1);
    } else if (call->op.same_as(tenstorrent::dfb_pipe_send_record()) ||
               call->op.same_as(tenstorrent::dfb_pipe_recv_record()) ||
               call->op.same_as(tenstorrent::dfb_pipe_wait_record())) {
      shift(2);
    } else if (call->op.same_as(tenstorrent::dfb_to_tensor_nd()) ||
               call->op.same_as(tenstorrent::dfb_reserve()) ||
               call->op.same_as(tenstorrent::dfb_wait()) ||
               call->op.same_as(tenstorrent::dfb_release()) ||
               call->op.same_as(tenstorrent::dfb_copy_wait()) ||
               call->op.same_as(tenstorrent::dfb_load())) {
      shift(0);
    }
    auto annotations = call->annotations;
    bool stream_lifecycle =
        call->op.same_as(tenstorrent::dfb_reserve()) ||
        call->op.same_as(tenstorrent::dfb_wait()) ||
        call->op.same_as(tenstorrent::dfb_copy_wait()) ||
        call->op.same_as(tenstorrent::dfb_release()) ||
        call->op.same_as(tenstorrent::tensor_to_dfb_nd()) ||
        call->op.same_as(tenstorrent::dfb_to_tensor_nd()) ||
        call->op.same_as(tenstorrent::dfb_pipe_send()) ||
        call->op.same_as(tenstorrent::dfb_pipe_recv()) ||
        call->op.same_as(tenstorrent::dfb_pipe_wait()) ||
        call->op.same_as(tenstorrent::dfb_pipe_send_record()) ||
        call->op.same_as(tenstorrent::dfb_pipe_recv_record()) ||
        call->op.same_as(tenstorrent::dfb_pipe_wait_record());
    if (annotate_streams_ && stream_lifecycle &&
        !referenced_resources.empty()) {
      ffi::Array<Integer> identity;
      for (int64_t resource : referenced_resources)
        identity.push_back(Integer(resource));
      annotations.Set("tt.stream_identity", identity);
    }
    return Call(call->dtype, call->op, args, annotations, op->span);
  }

private:
  int64_t offset_;
  const std::vector<int64_t> *resource_ids_{nullptr};
  bool annotate_streams_{false};
};

using PlannedPipeRecordKey = std::pair<int64_t, int64_t>;
using MaterializedPipeRecordIdentity = std::pair<int64_t, int64_t>;

class PipeRecordIdentityRemapper : public StmtExprMutator {
public:
  explicit PipeRecordIdentityRemapper(
      const std::map<PlannedPipeRecordKey, MaterializedPipeRecordIdentity>
          *identities)
      : identities_(identities) {}

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call->op.same_as(tenstorrent::dfb_pipe_send_record()) &&
        !call->op.same_as(tenstorrent::dfb_pipe_recv_record()) &&
        !call->op.same_as(tenstorrent::dfb_pipe_wait_record()))
      return call;
    if (call->args.size() != 3 || !as_const_int(call->args[0]) ||
        !as_const_int(call->args[1]))
      ThrowMalformed("pre-materialization Pipe record identity is not static");
    PlannedPipeRecordKey key{*as_const_int(call->args[0]),
                             *as_const_int(call->args[1])};
    auto found = identities_->find(key);
    if (found == identities_->end())
      ThrowMalformed("Device Pipe record operation has no materialized record");
    auto args = call->args;
    args.Set(0, Integer(found->second.first));
    args.Set(1, Integer(found->second.second));
    return Call(call->dtype, call->op, args, call->annotations, call->span);
  }

private:
  const std::map<PlannedPipeRecordKey, MaterializedPipeRecordIdentity>
      *identities_;
};

class DuplicateGenerationWaitRemover : public StmtExprMutator {
  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !call->op.same_as(tenstorrent::dfb_wait()))
      return StmtExprMutator::VisitStmt_(op);
    const int64_t *resource = as_const_int(call->args[0]);
    if (!resource)
      ThrowMalformed("pre-stream DFB wait must have a static generation ID");
    if (!waited_.insert(*resource).second)
      return Evaluate(Integer(0));
    return StmtExprMutator::VisitStmt_(op);
  }

  std::set<int64_t> waited_;
};

class TransientDeviceAnnotationRemover : public StmtExprMutator {
  class OperandIndexRewriter : public ExprMutator {
  public:
    OperandIndexRewriter(ffi::Array<PrimExpr> dfb_inputs,
                         ffi::Array<PrimExpr> value_inputs)
        : dfb_inputs_(std::move(dfb_inputs)),
          value_inputs_(std::move(value_inputs)) {}

  private:
    size_t Find(const PrimExpr &identity, const ffi::Array<PrimExpr> &inputs,
                const char *label) const {
      for (size_t index = 0; index < inputs.size(); ++index)
        if (ffi::StructuralEqual()(identity, inputs[index]))
          return index;
      ThrowMalformed(std::string("compute expression references undeclared ") +
                     label);
    }

    PrimExpr VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(tenstorrent::dfb_load())) {
        if (op->args.size() != 1 || !op->annotations.empty())
          ThrowMalformed("dfb_load requires one identity without annotations");
        return Call(op->dtype, tenstorrent::dfb_operand(),
                    {Integer(Find(op->args[0], dfb_inputs_, "DFB input"))}, {},
                    op->span);
      }
      if (op->op.same_as(tenstorrent::compute_value_load())) {
        if (op->args.size() != 1 || !op->annotations.empty())
          ThrowMalformed(
              "compute_value_load requires one identity without annotations");
        return Call(
            op->dtype, tenstorrent::compute_value_operand(),
            {Integer(Find(op->args[0], value_inputs_, "compute-value input"))},
            {}, op->span);
      }
      return ExprMutator::VisitExpr_(op);
    }

    ffi::Array<PrimExpr> dfb_inputs_;
    ffi::Array<PrimExpr> value_inputs_;
  };

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    bool derived_compute_value = call->op.same_as(compute_value());
    bool device_compute = call->op.same_as(dfb_compute());
    if (!call->annotations.count("tt.stream_identity") &&
        !call->annotations.count("tt.internal_communication_id") &&
        !derived_compute_value && !device_compute)
      return call;
    auto annotations = call->annotations;
    annotations.erase("tt.stream_identity");
    annotations.erase("tt.internal_communication_id");
    if (derived_compute_value || device_compute) {
      ffi::Array<PrimExpr> dfb_inputs;
      for (size_t index = 1; index < call->args.size(); ++index)
        dfb_inputs.push_back(call->args[index]);
      ffi::Array<PrimExpr> value_inputs;
      if (auto inputs = annotations.Get("tt.value_inputs"))
        value_inputs = Downcast<ffi::Array<PrimExpr>>(inputs.value());
      auto expression = annotations.Get("tt.expression");
      if (expression.has_value()) {
        auto typed = expression.value().as<PrimExpr>();
        if (!typed.has_value())
          ThrowMalformed("compute expression annotation is not PrimExpr");
        annotations.Set(
            "tt.expression",
            OperandIndexRewriter(dfb_inputs, value_inputs)(typed.value()));
      }

      if (device_compute) {
        auto kind = annotations.Get("tt.compute_kind");
        auto expression = annotations.Get("tt.expression");
        if (kind.has_value() && expression.has_value() &&
            Downcast<StringImm>(kind.value())->value == "typecast") {
          const auto *operand = expression.value().as<CallNode>();
          if (operand && operand->op.same_as(tenstorrent::dfb_operand()) &&
              operand->args.size() == 1 && is_zero(operand->args[0]) &&
              call->args.size() == 2 && value_inputs.empty()) {
            annotations.Set("tt.compute_kind", StringImm("copy"));
            annotations.erase("tt.expression");
          }
        }
        auto clear = annotations.Get("tt.clear");
        auto current_kind = annotations.Get("tt.compute_kind");
        bool accumulative =
            current_kind.has_value() &&
            (Downcast<StringImm>(current_kind.value())->value == "gemm" ||
             Downcast<StringImm>(current_kind.value())->value == "reduce");
        if (clear.has_value() && accumulative)
          annotations.erase("tt.clear");
        auto nan = annotations.Get("tt.nan_propagate");
        if (nan.has_value() && is_zero(Downcast<PrimExpr>(nan.value())))
          annotations.erase("tt.nan_propagate");
        auto accumulation = annotations.Get("tt.accum_dtype");
        auto output_dtype = annotations.Get("tt.compute_dtype");
        current_kind = annotations.Get("tt.compute_kind");
        if (accumulation.has_value() && output_dtype.has_value() &&
            current_kind.has_value() &&
            Downcast<StringImm>(current_kind.value())->value == "reduce" &&
            Downcast<StringImm>(accumulation.value())->value ==
                Downcast<StringImm>(output_dtype.value())->value)
          annotations.erase("tt.accum_dtype");
      }

      annotations.erase("tt.compute_dtype");
      annotations.erase("tt.compute_tile_shape");
      if (derived_compute_value)
        annotations.erase("tt.logical_domain");
      auto input_shapes = annotations.Get("tt.input_shapes")
                              .value_or(ffi::Array<ffi::Array<PrimExpr>>())
                              .as<ffi::Array<ffi::Array<PrimExpr>>>();
      if (input_shapes.has_value() && input_shapes.value().empty())
        annotations.erase("tt.input_shapes");
      auto retained_value_inputs = annotations.Get("tt.value_inputs")
                                       .value_or(ffi::Array<PrimExpr>())
                                       .as<ffi::Array<PrimExpr>>();
      if (retained_value_inputs.has_value() &&
          retained_value_inputs.value().empty())
        annotations.erase("tt.value_inputs");
      annotations.erase("tt.value_input_shapes");
    }
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }
};

class DeviceCallExpressionSimplifier : public StmtExprMutator {
  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    bool record = call->op.same_as(tenstorrent::dfb_pipe_send_record()) ||
                  call->op.same_as(tenstorrent::dfb_pipe_recv_record()) ||
                  call->op.same_as(tenstorrent::dfb_pipe_wait_record());
    bool tensor = call->op.same_as(tenstorrent::tensor_to_dfb_nd()) ||
                  call->op.same_as(tenstorrent::dfb_to_tensor_nd());
    if (!record && !tensor)
      return call;
    auto args = call->args;
    for (size_t index = tensor ? 2 : 0; index < args.size(); ++index)
      args.Set(index, analyzer_.Simplify(args[index]));
    auto annotations = call->annotations;
    if (record) {
      for (const auto &[key, object] : call->annotations) {
        if (auto expression = object.as<PrimExpr>()) {
          annotations.Set(key, analyzer_.Simplify(expression.value()));
        } else if (auto expressions = object.as<ffi::Array<PrimExpr>>()) {
          ffi::Array<PrimExpr> simplified;
          for (const PrimExpr &expression : expressions.value())
            simplified.push_back(analyzer_.Simplify(expression));
          annotations.Set(key, simplified);
        }
      }
    }
    return Call(call->dtype, call->op, args, annotations, call->span);
  }

  arith::Analyzer analyzer_;
};

enum class DeviceResourceKind : int {
  kTensor = 0,
  kDFBStream = 1,
  kPipeNet = 2,
  kAccumulator = 3,
};

using DeviceResourceKey = std::pair<DeviceResourceKind, int64_t>;

struct DeviceResourceNames {
  std::array<std::map<int64_t, std::string>, 4> by_kind;

  const std::string &Get(DeviceResourceKind kind, int64_t id) const {
    const auto &names = by_kind[static_cast<size_t>(kind)];
    auto found = names.find(id);
    if (found == names.end())
      ThrowMalformed("Device resource alias has no stable debug name");
    return found->second;
  }
};

std::string ResourceFallbackName(DeviceResourceKind kind, int64_t id);

std::string SanitizeResourceName(std::string name) {
  for (char &character : name)
    if (!std::isalnum(static_cast<unsigned char>(character)) &&
        character != '_')
      character = '_';
  if (name.empty() || std::isdigit(static_cast<unsigned char>(name.front())))
    name = "resource_" + name;
  return name;
}

std::map<int64_t, std::string>
BuildResourceNames(const std::map<int64_t, std::set<std::string>> &sources,
                   DeviceResourceKind kind, const std::string &suffix) {
  std::map<int64_t, std::string> result;
  std::set<std::string> used;
  for (const auto &[id, names] : sources) {
    std::string name = names.size() == 1
                           ? SanitizeResourceName(*names.begin()) + suffix
                           : ResourceFallbackName(kind, id);
    if (!used.insert(name).second) {
      name += "_" + std::to_string(id);
      used.insert(name);
    }
    result.emplace(id, std::move(name));
  }
  return result;
}

const Op &ResourceRefOp(DeviceResourceKind kind) {
  switch (kind) {
  case DeviceResourceKind::kTensor:
    return tenstorrent::tensor_ref();
  case DeviceResourceKind::kDFBStream:
    return tenstorrent::dfb_stream_ref();
  case DeviceResourceKind::kPipeNet:
    return tenstorrent::pipe_net_ref();
  case DeviceResourceKind::kAccumulator:
    return tenstorrent::accumulator_ref();
  }
  TVM_FFI_UNREACHABLE();
}

std::string ResourceFallbackName(DeviceResourceKind kind, int64_t id) {
  const char *prefix = kind == DeviceResourceKind::kTensor      ? "tensor"
                       : kind == DeviceResourceKind::kDFBStream ? "stream"
                       : kind == DeviceResourceKind::kPipeNet   ? "pipe_net"
                                                                : "accumulator";
  return std::string(prefix) + "_" + std::to_string(id);
}

std::vector<std::pair<size_t, DeviceResourceKind>>
ResourceOperands(const Call &call) {
  std::vector<std::pair<size_t, DeviceResourceKind>> result;
  auto add = [&](size_t index, DeviceResourceKind kind) {
    result.emplace_back(index, kind);
  };
  if (call->op.same_as(tenstorrent::dfb_compute())) {
    for (size_t index = 0; index < call->args.size(); ++index)
      add(index, DeviceResourceKind::kDFBStream);
  } else if (call->op.same_as(tenstorrent::compute_value())) {
    for (size_t index = 1; index < call->args.size(); ++index)
      add(index, DeviceResourceKind::kDFBStream);
  } else if (call->op.same_as(tenstorrent::compute_value_gemm())) {
    add(1, DeviceResourceKind::kDFBStream);
    add(2, DeviceResourceKind::kDFBStream);
  } else if (call->op.same_as(tenstorrent::compute_value_store())) {
    add(1, DeviceResourceKind::kDFBStream);
  } else if (call->op.same_as(tenstorrent::dfb_load()) ||
             call->op.same_as(tenstorrent::dfb_reserve()) ||
             call->op.same_as(tenstorrent::dfb_wait()) ||
             call->op.same_as(tenstorrent::dfb_copy_wait()) ||
             call->op.same_as(tenstorrent::dfb_release())) {
    add(0, DeviceResourceKind::kDFBStream);
  } else if (call->op.same_as(tenstorrent::tensor_to_dfb_nd())) {
    add(0, DeviceResourceKind::kTensor);
    add(1, DeviceResourceKind::kDFBStream);
  } else if (call->op.same_as(tenstorrent::dfb_to_tensor_nd())) {
    add(0, DeviceResourceKind::kDFBStream);
    add(1, DeviceResourceKind::kTensor);
  } else if (call->op.same_as(tenstorrent::dfb_pipe_send_record()) ||
             call->op.same_as(tenstorrent::dfb_pipe_recv_record()) ||
             call->op.same_as(tenstorrent::dfb_pipe_wait_record())) {
    add(0, DeviceResourceKind::kPipeNet);
    add(2, DeviceResourceKind::kDFBStream);
  } else if (call->op.same_as(tenstorrent::gemm_update())) {
    add(0, DeviceResourceKind::kDFBStream);
    add(1, DeviceResourceKind::kDFBStream);
    add(2, DeviceResourceKind::kAccumulator);
  } else if (call->op.same_as(tenstorrent::accumulator_init())) {
    add(0, DeviceResourceKind::kAccumulator);
  } else if (call->op.same_as(tenstorrent::accumulator_materialize())) {
    add(0, DeviceResourceKind::kAccumulator);
    add(1, DeviceResourceKind::kDFBStream);
  } else if (call->op.same_as(tenstorrent::dfb_local_copy()) ||
             call->op.same_as(tenstorrent::dfb_gather_copy())) {
    add(0, DeviceResourceKind::kDFBStream);
    add(1, DeviceResourceKind::kDFBStream);
  } else if (call->op.same_as(tenstorrent::dfb_collective_reduce())) {
    add(0, DeviceResourceKind::kDFBStream);
    add(1, DeviceResourceKind::kDFBStream);
    add(2, DeviceResourceKind::kDFBStream);
  }
  return result;
}

class DeviceResourceUseCollector : public StmtExprVisitor {
public:
  const std::map<DeviceResourceKey, size_t> &Counts() const { return counts_; }

private:
  void VisitExpr_(const CallNode *op) final {
    Call call = ffi::GetRef<Call>(op);
    for (const auto &[index, kind] : ResourceOperands(call)) {
      if (index >= call->args.size())
        ThrowMalformed("resource alias operand is missing");
      if (const int64_t *id = as_const_int(call->args[index]))
        ++counts_[{kind, *id}];
      else if (!call->args[index].dtype().is_int() ||
               call->args[index].dtype().lanes() != 1 ||
               SideEffect(call->args[index]) > CallEffectKind::kPure)
        ThrowMalformed(
            "parameterized resource ID must be a pure scalar integer");
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::map<DeviceResourceKey, size_t> counts_;
};

class DeviceResourceRefRewriter : public StmtExprMutator {
public:
  explicit DeviceResourceRefRewriter(
      const std::map<DeviceResourceKey, Var> *aliases)
      : aliases_(aliases) {}

private:
  PrimExpr ResourceRef(DeviceResourceKind kind, const PrimExpr &id,
                       const Span &span) const {
    if (const int64_t *constant = as_const_int(id)) {
      auto alias = aliases_->find({kind, *constant});
      if (alias != aliases_->end())
        return alias->second;
    }
    return Call(DataType::Handle(), ResourceRefOp(kind), {id}, {}, span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    auto args = call->args;
    for (const auto &[index, kind] : ResourceOperands(call)) {
      args.Set(index, ResourceRef(kind, args[index], args[index]->span));
    }
    return Call(call->dtype, call->op, args, call->annotations, op->span);
  }

  const std::map<DeviceResourceKey, Var> *aliases_;
};

class DeviceResourceAliasMaterializer {
public:
  explicit DeviceResourceAliasMaterializer(const DeviceResourceNames *names)
      : names_(names) {}

  Stmt operator()(const Stmt &input) const {
    return MaterializePrologue(input);
  }

private:
  Stmt MaterializePrologue(const Stmt &input) const {
    DeviceResourceUseCollector collector;
    collector(input);
    std::map<DeviceResourceKey, Var> aliases;
    std::set<std::string> used_names;
    for (const auto &[key, count] : collector.Counts()) {
      bool name = key.first != DeviceResourceKind::kDFBStream || count > 1;
      if (!name)
        continue;
      std::string candidate = names_->Get(key.first, key.second);
      if (!used_names.insert(candidate).second) {
        candidate += "_" + std::to_string(key.second);
        used_names.insert(candidate);
      }
      aliases.emplace(key, Var(candidate, DataType::Handle()));
    }
    DeviceResourceRefRewriter rewrite(&aliases);
    Stmt body = rewrite(input);
    if (!aliases.empty()) {
      ffi::Array<Stmt> sequence;
      for (const auto &[key, variable] : aliases)
        sequence.push_back(
            Bind(variable, Call(DataType::Handle(), ResourceRefOp(key.first),
                                {Integer(key.second)})));
      sequence.push_back(body);
      body = SeqStmt::Flatten(sequence);
    }
    return body;
  }

  const DeviceResourceNames *names_;
};

using StorageLocalKey = std::tuple<int64_t, int64_t, int64_t>;
using StorageCoreKey = std::pair<int64_t, int64_t>;

struct ScheduledDeviceCall {
  Call call;
  bool repeated;
};

std::vector<size_t> StorageDFBOperandPositions(const Call &call) {
  if (call->op.same_as(tenstorrent::dfb_compute())) {
    std::vector<size_t> result(call->args.size());
    std::iota(result.begin(), result.end(), 0);
    return result;
  }
  if (call->op.same_as(tenstorrent::compute_value())) {
    std::vector<size_t> result(call->args.size() > 0 ? call->args.size() - 1
                                                     : 0);
    std::iota(result.begin(), result.end(), 1);
    return result;
  }
  if (call->op.same_as(tenstorrent::dfb_reserve()) ||
      call->op.same_as(tenstorrent::dfb_wait()) ||
      call->op.same_as(tenstorrent::dfb_copy_wait()) ||
      call->op.same_as(tenstorrent::dfb_release()) ||
      call->op.same_as(tenstorrent::dfb_load()) ||
      call->op.same_as(tenstorrent::dfb_to_tensor_nd()))
    return {0};
  if (call->op.same_as(tenstorrent::tensor_to_dfb_nd()) ||
      call->op.same_as(tenstorrent::dfb_pipe_send()) ||
      call->op.same_as(tenstorrent::dfb_pipe_recv()) ||
      call->op.same_as(tenstorrent::dfb_pipe_wait()) ||
      call->op.same_as(tenstorrent::accumulator_materialize()) ||
      call->op.same_as(tenstorrent::compute_value_store()))
    return {1};
  if (call->op.same_as(tenstorrent::dfb_pipe_send_record()) ||
      call->op.same_as(tenstorrent::dfb_pipe_recv_record()) ||
      call->op.same_as(tenstorrent::dfb_pipe_wait_record()))
    return {3};
  if (call->op.same_as(tenstorrent::compute_value_gemm()))
    return {1, 2};
  if (call->op.same_as(tenstorrent::gemm_update()) ||
      call->op.same_as(tenstorrent::dfb_local_copy()) ||
      call->op.same_as(tenstorrent::dfb_gather_copy()))
    return {0, 1};
  if (call->op.same_as(tenstorrent::dfb_collective_reduce()))
    return {0, 1, 2};
  return {};
}

void CollectStorageScheduleCalls(const Stmt &stmt, bool repeated,
                                 std::vector<ScheduledDeviceCall> *calls) {
  if (const auto *sequence = stmt.as<SeqStmtNode>()) {
    for (const Stmt &child : sequence->seq)
      CollectStorageScheduleCalls(child, repeated, calls);
    return;
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    CollectStorageScheduleCalls(loop->body, true, calls);
    return;
  }
  if (const auto *branch = stmt.as<IfThenElseNode>()) {
    CollectStorageScheduleCalls(branch->then_case, repeated, calls);
    if (branch->else_case.has_value())
      CollectStorageScheduleCalls(branch->else_case.value(), repeated, calls);
    return;
  }
  if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
    CollectStorageScheduleCalls(realize->block->body, repeated, calls);
    return;
  }
  if (const auto *evaluate = stmt.as<EvaluateNode>())
    if (const auto *call = evaluate->value.as<CallNode>())
      calls->push_back({ffi::GetRef<Call>(call), repeated});
}

bool IsAcyclic(const std::vector<std::vector<int64_t>> &successors) {
  std::vector<int64_t> indegree(successors.size(), 0);
  for (const auto &edges : successors)
    for (int64_t successor : edges)
      ++indegree[successor];
  std::vector<int64_t> ready;
  for (size_t event = 0; event < indegree.size(); ++event)
    if (!indegree[event])
      ready.push_back(event);
  size_t visited = 0;
  while (!ready.empty()) {
    int64_t event = ready.back();
    ready.pop_back();
    ++visited;
    for (int64_t successor : successors[event])
      if (!--indegree[successor])
        ready.push_back(successor);
  }
  return visited == successors.size();
}

struct StorageSchedule {
  std::vector<std::vector<int64_t>> successors;
  std::map<StorageLocalKey, int64_t> reserves;
  std::map<StorageLocalKey, int64_t> producers;
  std::map<StorageLocalKey, int64_t> first_waits;
  std::map<StorageLocalKey, int64_t> releases;
  std::set<int64_t> repeated_resources;
};

StorageSchedule
BuildStorageSchedule(const std::array<std::vector<DeviceCoreBody>, 3> &bodies,
                     const ffi::Array<DFBDescriptor> &dfbs) {
  StorageSchedule result;
  std::map<int64_t, DFBDescriptor> descriptors;
  for (const DFBDescriptor &dfb : dfbs)
    descriptors.emplace(dfb->dfb_id, dfb);
  std::map<StorageLocalKey, std::pair<int64_t, bool>> copy_issues;
  std::map<StorageLocalKey, int64_t> copy_completions;
  std::map<StorageLocalKey, std::vector<int64_t>> waits;
  std::map<StorageLocalKey, std::vector<int64_t>> forwarding_completions;
  using TransferKey = std::tuple<int64_t, int64_t, int64_t, int64_t>;
  std::map<TransferKey, bool> sends;
  std::set<TransferKey> receives;
  using RecordTransferKey =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;
  std::map<RecordTransferKey, bool> record_sends;
  std::set<RecordTransferKey> record_receives;

  auto add_edge = [&](int64_t source, int64_t destination) {
    if (source != destination)
      result.successors[source].push_back(destination);
  };
  auto static_arg = [&](const Call &call, size_t index) {
    if (index >= call->args.size() || !as_const_int(call->args[index]))
      ThrowMalformed("storage planning requires static concrete DFB operands");
    return *as_const_int(call->args[index]);
  };

  for (size_t slot_index = 0; slot_index < bodies.size(); ++slot_index) {
    const ffi::String slot = slot_index == 0   ? ffi::String("trisc")
                             : slot_index == 1 ? ffi::String("ncrisc")
                                               : ffi::String("brisc");
    for (const DeviceCoreBody &core_body : bodies[slot_index]) {
      std::vector<ScheduledDeviceCall> calls;
      CollectStorageScheduleCalls(core_body.body, false, &calls);
      int64_t previous = -1;
      for (const ScheduledDeviceCall &scheduled : calls) {
        const Call &call = scheduled.call;
        int64_t event = result.successors.size();
        result.successors.emplace_back();
        if (previous >= 0)
          add_edge(previous, event);
        previous = event;
        for (size_t position : StorageDFBOperandPositions(call))
          if (position < call->args.size() &&
              as_const_int(call->args[position]) && scheduled.repeated)
            result.repeated_resources.insert(
                *as_const_int(call->args[position]));

        auto key = [&](int64_t resource) {
          return StorageLocalKey{core_body.x, core_body.y, resource};
        };
        if (call->op.same_as(tenstorrent::dfb_reserve())) {
          result.reserves.emplace(key(static_arg(call, 0)), event);
        } else if (call->op.same_as(tenstorrent::dfb_wait())) {
          waits[key(static_arg(call, 0))].push_back(event);
        } else if (call->op.same_as(tenstorrent::dfb_release())) {
          result.releases[key(static_arg(call, 0))] = event;
        } else if (call->op.same_as(tenstorrent::tensor_to_dfb_nd())) {
          copy_issues.emplace(key(static_arg(call, 1)),
                              std::make_pair(event, true));
        } else if (call->op.same_as(tenstorrent::dfb_to_tensor_nd())) {
          copy_issues.emplace(key(static_arg(call, 0)),
                              std::make_pair(event, false));
        } else if (call->op.same_as(tenstorrent::dfb_copy_wait())) {
          int64_t resource = static_arg(call, 0);
          copy_completions[key(resource)] = event;
          auto issue = copy_issues.find(key(resource));
          if (issue != copy_issues.end() && issue->second.second)
            result.producers[key(resource)] = event;
        } else if (call->op.same_as(tenstorrent::dfb_compute())) {
          result.producers[key(static_arg(call, 0))] = event;
        } else if (call->op.same_as(tenstorrent::accumulator_materialize()) ||
                   call->op.same_as(tenstorrent::compute_value_store())) {
          result.producers[key(static_arg(call, 1))] = event;
        } else if (call->op.same_as(tenstorrent::dfb_local_copy()) ||
                   call->op.same_as(tenstorrent::dfb_gather_copy())) {
          result.producers[key(static_arg(call, 1))] = event;
        } else if (call->op.same_as(tenstorrent::dfb_collective_reduce())) {
          result.producers[key(static_arg(call, 2))] = event;
        } else if (call->op.same_as(tenstorrent::dfb_pipe_send()) ||
                   call->op.same_as(tenstorrent::dfb_pipe_recv())) {
          int64_t transfer = static_arg(call, 0);
          int64_t resource = static_arg(call, 1);
          TransferKey transfer_key{core_body.x, core_body.y, transfer,
                                   resource};
          if (call->op.same_as(tenstorrent::dfb_pipe_send())) {
            const DFBDescriptor &descriptor = descriptors.at(resource);
            bool forwarding = slot == descriptor->producer_slot &&
                              descriptor->tensor_backing.has_value();
            sends[transfer_key] = forwarding;
          } else {
            receives.insert(transfer_key);
          }
        } else if (call->op.same_as(tenstorrent::dfb_pipe_wait())) {
          int64_t transfer = static_arg(call, 0);
          int64_t resource = static_arg(call, 1);
          TransferKey transfer_key{core_body.x, core_body.y, transfer,
                                   resource};
          auto sent = sends.find(transfer_key);
          if (sent != sends.end() && sent->second)
            forwarding_completions[key(resource)].push_back(event);
          else if (receives.count(transfer_key))
            result.producers[key(resource)] = event;
        } else if (call->op.same_as(tenstorrent::dfb_pipe_send_record()) ||
                   call->op.same_as(tenstorrent::dfb_pipe_recv_record())) {
          int64_t pipe_net = static_arg(call, 0);
          int64_t record = static_arg(call, 1);
          int64_t resource = static_arg(call, 2);
          RecordTransferKey transfer_key{core_body.x, core_body.y, pipe_net,
                                         record, resource};
          if (call->op.same_as(tenstorrent::dfb_pipe_send_record())) {
            const DFBDescriptor &descriptor = descriptors.at(resource);
            bool forwarding = slot == descriptor->producer_slot &&
                              descriptor->tensor_backing.has_value();
            record_sends[transfer_key] = forwarding;
          } else {
            record_receives.insert(transfer_key);
          }
        } else if (call->op.same_as(tenstorrent::dfb_pipe_wait_record())) {
          int64_t pipe_net = static_arg(call, 0);
          int64_t record = static_arg(call, 1);
          int64_t resource = static_arg(call, 2);
          RecordTransferKey transfer_key{core_body.x, core_body.y, pipe_net,
                                         record, resource};
          auto sent = record_sends.find(transfer_key);
          if (sent != record_sends.end() && sent->second)
            forwarding_completions[key(resource)].push_back(event);
          else if (record_receives.count(transfer_key))
            result.producers[key(resource)] = event;
        }
      }
    }
  }

  for (const auto &[key, events] : waits) {
    auto producer = result.producers.find(key);
    if (producer == result.producers.end())
      ThrowMalformed("storage planning found a wait without publication");
    result.first_waits.emplace(key, events.front());
    for (int64_t event : events)
      add_edge(producer->second, event);
  }
  for (const auto &[key, issue] : copy_issues) {
    auto completion = copy_completions.find(key);
    if (completion == copy_completions.end())
      ThrowMalformed("storage planning found a copy without completion");
    add_edge(issue.first, completion->second);
    if (!issue.second) {
      auto release = result.releases.find(key);
      if (release == result.releases.end())
        ThrowMalformed("storage planning found an export without release");
      add_edge(completion->second, release->second);
    }
  }
  for (const auto &[key, completions] : forwarding_completions) {
    auto release = result.releases.find(key);
    if (release == result.releases.end())
      ThrowMalformed("storage planning found forwarding without release");
    for (int64_t completion : completions)
      add_edge(completion, release->second);
  }
  if (!IsAcyclic(result.successors))
    ThrowMalformed("base transaction schedule contains a dependency cycle");
  return result;
}

std::map<int64_t, std::map<StorageCoreKey, int64_t>>
IndexStorageEvents(const std::map<StorageLocalKey, int64_t> &events) {
  std::map<int64_t, std::map<StorageCoreKey, int64_t>> result;
  for (const auto &[key, event] : events)
    result[std::get<2>(key)][{std::get<0>(key), std::get<1>(key)}] = event;
  return result;
}

ffi::Map<ffi::String, Integer> PlanNonPipelineStorageGroups(
    const ffi::Array<DFBDescriptor> &dfbs,
    const std::array<std::vector<DeviceCoreBody>, 3> &bodies,
    const std::map<int64_t, std::string> &storage_keys) {
  StorageSchedule schedule = BuildStorageSchedule(bodies, dfbs);
  auto reserves = IndexStorageEvents(schedule.reserves);
  auto producers = IndexStorageEvents(schedule.producers);
  auto waits = IndexStorageEvents(schedule.first_waits);
  auto releases = IndexStorageEvents(schedule.releases);
  std::map<int64_t, DFBDescriptor> descriptors;
  std::map<std::string, std::vector<int64_t>> compatible;
  for (const DFBDescriptor &dfb : dfbs) {
    descriptors.emplace(dfb->dfb_id, dfb);
    compatible[storage_keys.at(dfb->dfb_id)].push_back(dfb->dfb_id);
  }

  auto ordered = [&](const auto &events, int64_t before, int64_t after) {
    auto lhs = events.find(before), rhs = events.find(after);
    if (lhs == events.end() || rhs == events.end())
      return true;
    for (const auto &[core, event] : lhs->second) {
      auto other = rhs->second.find(core);
      if (other != rhs->second.end() && event >= other->second)
        return false;
    }
    return true;
  };
  auto fifo_ordered = [&](int64_t before, int64_t after) {
    return ordered(reserves, before, after) &&
           ordered(producers, before, after) && ordered(waits, before, after) &&
           ordered(releases, before, after);
  };

  struct Group {
    int64_t id;
    int64_t capacity;
    std::vector<int64_t> resources;
  };
  std::vector<Group> groups;
  ffi::Map<ffi::String, Integer> assignments;
  for (auto &[storage_key, resources] : compatible) {
    auto first_reserve = [&](int64_t resource) {
      auto found = reserves.find(resource);
      if (found == reserves.end() || found->second.empty())
        return std::numeric_limits<int64_t>::max();
      int64_t first = std::numeric_limits<int64_t>::max();
      for (const auto &[_, event] : found->second)
        first = std::min(first, event);
      return first;
    };
    std::sort(resources.begin(), resources.end(),
              [&](int64_t left, int64_t right) {
                return std::make_pair(first_reserve(left), left) <
                       std::make_pair(first_reserve(right), right);
              });
    std::vector<size_t> local_groups;
    for (int64_t resource : resources) {
      int64_t capacity = RequirePositiveStaticInteger(
          descriptors.at(resource)->block_count, "DFB storage capacity");
      bool assigned = false;
      bool repeated = schedule.repeated_resources.count(resource);
      for (size_t group_index : local_groups) {
        Group &group = groups[group_index];
        if (group.capacity != capacity)
          continue;
        bool same_repetition = std::all_of(
            group.resources.begin(), group.resources.end(), [&](int64_t value) {
              return schedule.repeated_resources.count(value) == repeated;
            });
        if (!same_repetition)
          continue;
        if (repeated) {
          bool closed = releases.count(resource) && reserves.count(resource);
          for (int64_t value : group.resources)
            closed &= releases.count(value) && reserves.count(value);
          if (!closed)
            continue;
        }
        bool valid_order = std::all_of(
            group.resources.begin(), group.resources.end(),
            [&](int64_t previous) { return fifo_ordered(previous, resource); });
        if (!valid_order)
          continue;
        std::vector<std::pair<int64_t, int64_t>> added_edges;
        if (group.resources.size() >= static_cast<size_t>(capacity)) {
          int64_t previous = group.resources[group.resources.size() - capacity];
          auto old_releases = releases.find(previous);
          auto next_reserves = reserves.find(resource);
          if (old_releases == releases.end() || next_reserves == reserves.end())
            continue;
          for (const auto &[core, release] : old_releases->second) {
            auto reserve = next_reserves->second.find(core);
            if (reserve != next_reserves->second.end()) {
              schedule.successors[release].push_back(reserve->second);
              added_edges.emplace_back(release, reserve->second);
            }
          }
          if (!IsAcyclic(schedule.successors)) {
            for (auto it = added_edges.rbegin(); it != added_edges.rend(); ++it)
              schedule.successors[it->first].pop_back();
            continue;
          }
        }
        group.resources.push_back(resource);
        assignments.Set(std::to_string(resource), Integer(group.id));
        assigned = true;
        break;
      }
      if (!assigned) {
        int64_t group_id = groups.size();
        groups.push_back({group_id, capacity, {resource}});
        local_groups.push_back(groups.size() - 1);
        assignments.Set(std::to_string(resource), Integer(group_id));
      }
    }
  }
  if (assignments.size() != dfbs.size())
    ThrowMalformed("storage-group planning did not cover every DFB generation");
  return assignments;
}

struct LogicalStreamPlan {
  ffi::Array<DFBStorageGroupDescriptor> descriptors;
  std::vector<int64_t> resource_stream_ids;
};

bool SameLogicalStreamContract(const DFBStorageGroupDescriptor &stream,
                               const DFBDescriptor &resource) {
  return stream->element_dtype == resource->element_dtype &&
         ffi::StructuralEqual()(stream->tile_shape, resource->tile_shape) &&
         ffi::StructuralEqual()(stream->block_shape_in_tiles,
                                resource->block_shape_in_tiles) &&
         ffi::StructuralEqual()(stream->block_count, resource->block_count) &&
         stream->producer_slot == resource->producer_slot &&
         ffi::StructuralEqual()(stream->producer_domain,
                                resource->producer_domain) &&
         stream->consumer_slot == resource->consumer_slot &&
         ffi::StructuralEqual()(stream->consumer_domain,
                                resource->consumer_domain);
}

LogicalStreamPlan
BuildLogicalStreamPlan(const ffi::Array<DFBDescriptor> &resources,
                       const ffi::Map<ffi::String, Integer> &raw_groups) {
  if (raw_groups.size() != resources.size())
    ThrowMalformed("logical stream plan must cover every DFB transaction");
  LogicalStreamPlan result;
  result.resource_stream_ids.assign(resources.size(), -1);
  std::map<int64_t, int64_t> canonical_groups;
  for (const DFBDescriptor &resource : resources) {
    int64_t id = resource->dfb_id;
    if (id < 0 || id >= static_cast<int64_t>(resources.size()) ||
        result.resource_stream_ids[id] >= 0)
      ThrowMalformed("DFB transaction IDs must be dense and unique");
    ffi::String key = std::to_string(id);
    if (!raw_groups.count(key))
      ThrowMalformed("logical stream plan is missing a DFB transaction");
    int64_t raw_group = raw_groups.at(key)->value;
    if (raw_group < 0)
      ThrowMalformed("logical stream ID must be nonnegative");
    auto [found, inserted] =
        canonical_groups.emplace(raw_group, canonical_groups.size());
    int64_t stream_id = found->second;
    result.resource_stream_ids[id] = stream_id;
    if (inserted) {
      result.descriptors.push_back(DFBStorageGroupDescriptor(
          stream_id, resource->element_dtype, resource->tile_shape,
          resource->block_shape_in_tiles, resource->block_count,
          resource->producer_slot, resource->producer_domain,
          resource->consumer_slot, resource->consumer_domain,
          resource->transaction_count_or_loop_relation, std::nullopt,
          resource->source_span));
    } else if (!SameLogicalStreamContract(result.descriptors[stream_id],
                                          resource)) {
      ThrowMalformed("logical stream transactions disagree on capacity, "
                     "layout, or slots");
    }
  }
  return result;
}

void FinalizeLogicalStreamTransactionCounts(
    LogicalStreamPlan *plan, const ffi::Array<DFBDescriptor> &resources,
    const std::array<std::vector<DeviceCoreBody>, 3> &shared_bodies) {
  std::map<std::tuple<int64_t, int64_t, int64_t>, int64_t> per_core;
  for (const auto &slot_bodies : shared_bodies) {
    for (const DeviceCoreBody &core_body : slot_bodies) {
      PostOrderVisit(core_body.body, [&](const ffi::ObjectRef &object) {
        const auto *call = object.as<CallNode>();
        if (!call || !call->op.same_as(tenstorrent::dfb_reserve()))
          return;
        const int64_t *resource_id = as_const_int(call->args[0]);
        if (!resource_id || *resource_id < 0 ||
            *resource_id >= static_cast<int64_t>(resources.size()))
          ThrowMalformed("pre-stream DFB reserve has an invalid generation ID");
        int64_t stream_id = plan->resource_stream_ids[*resource_id];
        int64_t transactions = RequirePositiveStaticInteger(
            resources[*resource_id]->transaction_count_or_loop_relation,
            "DFB transaction count");
        auto key = std::make_tuple(core_body.x, core_body.y, stream_id);
        if (per_core[key] > std::numeric_limits<int64_t>::max() - transactions)
          ThrowMalformed("logical stream transaction count overflows");
        per_core[key] += transactions;
      });
    }
  }
  std::vector<int64_t> maxima(plan->descriptors.size(), 0);
  for (const auto &[key, count] : per_core)
    maxima[std::get<2>(key)] = std::max(maxima[std::get<2>(key)], count);
  for (size_t stream_id = 0; stream_id < plan->descriptors.size();
       ++stream_id) {
    if (maxima[stream_id] <= 0)
      ThrowMalformed("logical stream has no producer reservation");
    const DFBStorageGroupDescriptor &stream = plan->descriptors[stream_id];
    plan->descriptors.Set(
        stream_id, DFBStorageGroupDescriptor(
                       stream->storage_group_id, stream->element_dtype,
                       stream->tile_shape, stream->block_shape_in_tiles,
                       stream->block_count, stream->producer_slot,
                       stream->producer_domain, stream->consumer_slot,
                       stream->consumer_domain, Integer(maxima[stream_id]),
                       stream->storage_backing, stream->source_span));
  }
}

IRModule FormMulticoreProgram(const IRModule &input, const PrimFunc &frontend,
                              const Target &target,
                              const ffi::String &operation,
                              const ffi::String &arch, int64_t gx, int64_t gy,
                              const Stmt &kernel_body, const Var &core_x_var,
                              const Var &core_y_var,
                              const ffi::Array<TTBufferMetadata> &metadata) {
  ffi::Map<GlobalVar, BaseFunc> functions;
  ffi::Array<ffi::String> order;
  std::array<std::vector<DeviceCoreBody>, 3> shared_bodies;
  std::array<std::set<int64_t>, 3> shared_tensor_indices;
  ffi::Array<DFBDescriptor> dfbs;
  ffi::Array<PipeDescriptor> pipes;
  ffi::Array<PipeTransferDescriptor> transfers;
  ffi::Array<AccumulatorDescriptor> accumulators;
  ffi::Array<ComputeValueDescriptor> compute_values;
  std::vector<int> effects(frontend->params.size(), 0);
  TransferKeys transfer_keys;
  std::vector<PipeEndpoint> endpoints;
  int64_t pipeline_stages = 0, pipeline_extent = 0;
  ffi::String pipeline_wait_policy;
  ffi::Map<ffi::String, Integer> storage_groups;
  ffi::Map<ffi::String, ffi::Array<Integer>> pipeline_relations;
  bool producer_forwarding = false;
  bool precision_regions = false;
  CoreDomain shared_domain(CoreCoord(0, 0), CoreCoord(gx, gy));
  std::map<std::string, int64_t> logical_resource_ids;
  std::map<int64_t, std::string> resource_storage_keys;
  std::map<int64_t, std::set<std::string>> resource_debug_names;
  StaticValueSubstituter bind_core_x(
      core_x_var, Call(core_x_var.dtype(), tenstorrent::core_x(), {}));
  Stmt symbolic_kernel_body = bind_core_x(kernel_body);
  StaticValueSubstituter bind_core_y(
      core_y_var, Call(core_y_var.dtype(), tenstorrent::core_y(), {}));
  symbolic_kernel_body = bind_core_y(symbolic_kernel_body);

  // Instantiate one proof Core at a time while retaining the same symbolic
  // body for Device emission. Core constants are used only by planner
  // decisions, never as the source of the committed scalar expressions.
  TransportBufferCache transport_buffers;
  for (int64_t index = 0; index < gx * gy; ++index) {
    int64_t x = index / gy, y = index % gy;
    GeneralDataflowPlanner planner(frontend, metadata, &transfer_keys, x, y,
                                   &transport_buffers);
    std::vector<For> independent_loops =
        FindIndependentLoops(symbolic_kernel_body);
    planner.Plan(
        IndependentLoopBody(independent_loops, symbolic_kernel_body));
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
    } else if (pipeline_extent) {
      ThrowUnsupported("all active Core bodies must retain their pipeline");
    }
    producer_forwarding |= planner.HasProducerForwarding();
    precision_regions |= planner.HasPrecisionRegions();
    CoreDomain domain(CoreCoord(x, y), CoreCoord(x + 1, y + 1));
    PrimExpr transaction_relation =
        IndependentIterationCount(independent_loops);
    std::vector<int64_t> resource_ids(planner.ResourceCount(), -1);
    for (const DFBDescriptor &dfb : planner.Descriptors(domain)) {
      int64_t local_id = dfb->dfb_id;
      std::string key = planner.ResourceIdentity(local_id);
      auto [found, inserted] =
          logical_resource_ids.emplace(key, logical_resource_ids.size());
      int64_t global_id = found->second;
      resource_ids[local_id] = global_id;
      resource_debug_names[global_id].insert(
          std::string(planner.ResourceDebugName(local_id)));
      std::string storage_key = planner.ResourceStorageIdentity(local_id);
      if (inserted) {
        resource_storage_keys.emplace(global_id, std::move(storage_key));
        dfbs.push_back(
            DFBDescriptor(global_id, key, dfb->element_dtype, dfb->tile_shape,
                          dfb->block_shape_in_tiles, dfb->block_count,
                          dfb->tensor_backing, dfb->producer_slot,
                          shared_domain, dfb->consumer_slot, shared_domain,
                          independent_loops.empty()
                              ? dfb->transaction_count_or_loop_relation
                              : dfb->transaction_count_or_loop_relation *
                                    transaction_relation,
                          dfb->source_span));
      } else if (resource_storage_keys.at(global_id) != storage_key) {
        ThrowMalformed("one logical DFB has inconsistent storage identity");
      }
    }
    if (planner.IsPipeline()) {
      for (const auto &[key, group] : planner.StorageGroups()) {
        int64_t local = std::stoll(std::string(key));
        if (local < 0 || local >= static_cast<int64_t>(resource_ids.size()) ||
            resource_ids[local] < 0)
          ThrowMalformed("pipeline storage group references dead DFB");
        ffi::String global_key = std::to_string(resource_ids[local]);
        if (storage_groups.count(global_key)) {
          if (storage_groups.at(global_key)->value != group->value)
            ThrowUnsupported(
                "Core-local storage groups disagree for one logical DFB");
        } else {
          storage_groups.Set(global_key, group);
        }
      }
      for (const auto &[key, relation] : planner.PipelineRelations()) {
        int64_t local = std::stoll(std::string(key));
        if (local < 0 || local >= static_cast<int64_t>(resource_ids.size()) ||
            resource_ids[local] < 0)
          ThrowMalformed("pipeline relation references dead DFB");
        ffi::String global_key = std::to_string(resource_ids[local]);
        if (pipeline_relations.count(global_key)) {
          if (!ffi::StructuralEqual()(pipeline_relations.at(global_key),
                                      relation))
            ThrowUnsupported(
                "Core-local pipeline relations disagree for one logical DFB");
        } else {
          pipeline_relations.Set(global_key, relation);
        }
      }
    }
    for (const AccumulatorDescriptor &accumulator : planner.Accumulators()) {
      accumulators.push_back(AccumulatorDescriptor(
          accumulator->accumulator_id, accumulator->accumulator_region,
          accumulator->input_dtype, accumulator->accumulation_dtype,
          accumulator->output_dtype, accumulator->full_k_tiles, domain,
          accumulator->source_span));
    }
    for (const ComputeValueDescriptor &value : planner.ComputeValues()) {
      compute_values.push_back(ComputeValueDescriptor(
          value->value_id, value->buffer, value->version,
          value->previous_value_id, value->accumulator_id, value->value_kind,
          value->logical_shape, value->compute_tile_shape, value->dtype,
          value->owner_slot, domain, value->materialization_refs,
          value->source_span));
    }
    for (PipeEndpoint endpoint : planner.Endpoints()) {
      if (endpoint.dfb_id < 0 ||
          endpoint.dfb_id >= static_cast<int64_t>(resource_ids.size()) ||
          resource_ids[endpoint.dfb_id] < 0)
        ThrowMalformed("communication endpoint references dead DFB");
      endpoint.dfb_id = resource_ids[endpoint.dfb_id];
      endpoints.push_back(std::move(endpoint));
    }
    auto core_effects = planner.Effects();
    std::set<int64_t> used_tensors;
    for (const char *slot : {"ncrisc", "brisc"})
      for (const Integer &tensor :
           planner.UsedTensorIndices(ffi::String(slot)))
        used_tensors.insert(tensor->value);
    for (int64_t tensor : used_tensors) {
      const auto &effect = core_effects[tensor];
      effects[tensor] |= effect == "inout"    ? 3
                         : effect == "output" ? 2
                                              : 1;
    }
    DeviceIDRemapper remap(0, &resource_ids);
    size_t slot_index = 0;
    for (const ffi::String &slot :
         ffi::Array<ffi::String>{"trisc", "ncrisc", "brisc"}) {
      bool compute = slot == "trisc";
      Stmt slot_body = compute            ? planner.ComputeBody()
                       : slot == "ncrisc" ? planner.TransferBody()
                                          : planner.PipeBody();
      slot_body = WrapIndependentLoops(std::move(slot_body), independent_loops);
      Stmt local = remap(slot_body);
      if (!(local.as<EvaluateNode>() &&
            is_zero(local.as<EvaluateNode>()->value))) {
        shared_bodies[slot_index].push_back({x, y, local});
      }
      for (const Integer &tensor : planner.UsedTensorIndices(slot))
        shared_tensor_indices[slot_index].insert(tensor->value);
      ++slot_index;
    }
  }
  if (!pipeline_extent)
    storage_groups = PlanNonPipelineStorageGroups(dfbs, shared_bodies,
                                                  resource_storage_keys);
  std::map<int64_t, PipeEndpoint> senders, receivers;
  std::map<PlannedPipeRecordKey, int64_t> record_payloads;
  std::map<PlannedPipeRecordKey, PipeEndpoint> ordered_record_sources;
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
        ordered_record_sources.emplace(key, endpoint);
      } else if (endpoint.occurrence == 0 &&
                 previous->second != endpoint.dfb_id) {
        ThrowMalformed(
            "Pipe record has more than one source payload per occurrence");
      }
    }
  }
  std::map<int64_t, int64_t> pipe_net_ids;
  std::map<int64_t, int64_t> next_record_indices;
  std::map<PlannedPipeRecordKey, MaterializedPipeRecordIdentity>
      materialized_record_ids;
  for (const auto &[key, endpoint] : ordered_record_sources) {
    int64_t communication_id = key.first;
    auto [pipe_net, inserted] =
        pipe_net_ids.emplace(communication_id, pipe_net_ids.size());
    int64_t pipe_net_id = pipe_net->second;
    int64_t record_index = next_record_indices[pipe_net_id]++;
    materialized_record_ids.emplace(
        key, MaterializedPipeRecordIdentity{pipe_net_id, record_index});
    const auto &r = endpoint.record;
    pipes.push_back(
        PipeDescriptor(pipe_net_id, communication_id, record_index,
                       CoreCoord(r[3]->value, r[4]->value),
                       CoreCoord(r[5]->value, r[6]->value),
                       CoreCoord(r[7]->value, r[8]->value),
                       r[2]->value ? "collective" : "point_to_point",
                       endpoint.dfb_id, endpoint.span));
  }
  PipeRecordIdentityRemapper remap_record_ids(&materialized_record_ids);
  for (auto &slot_bodies : shared_bodies)
    for (DeviceCoreBody &core_body : slot_bodies)
      core_body.body = remap_record_ids(core_body.body);
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
    auto identity = materialized_record_ids.find({r[0]->value, r[1]->value});
    if (identity == materialized_record_ids.end())
      ThrowMalformed("Pipe endpoint has no materialized record identity");
    transfers.push_back(PipeTransferDescriptor(
        id, identity->second.first, r[0]->value, identity->second.second,
        sender.occurrence, CoreCoord(r[3]->value, r[4]->value),
        CoreCoord(sender.x, sender.y), sender.dfb_id, receiver->second.dfb_id,
        sender.transaction_count, sender.span));
  }
  ffi::Array<CommunicationDescriptor> communications;
  auto normalized_topology =
      frontend->GetAttr<ffi::Array<NormalizedTopologyDescriptor>>(
          "tt.normalized_topology");
  if (normalized_topology.has_value()) {
    For logical_x = Downcast<For>(frontend->body);
    For logical_y = Downcast<For>(logical_x->body);
    StaticValueSubstituter bind_x(
        logical_x->loop_var,
        Call(logical_x->loop_var.dtype(), tenstorrent::core_x(), {}));
    StaticValueSubstituter bind_y(
        logical_y->loop_var,
        Call(logical_y->loop_var.dtype(), tenstorrent::core_y(), {}));
    auto parameterize = [&](const PrimExpr &expr) {
      return bind_y(bind_x(expr));
    };
    auto parameterize_array = [&](const ffi::Array<PrimExpr> &values) {
      ffi::Array<PrimExpr> result;
      for (const PrimExpr &value : values)
        result.push_back(parameterize(value));
      return result;
    };
    for (const NormalizedTopologyDescriptor &entry :
         normalized_topology.value()) {
      ffi::String algorithm =
          entry->kind == "all_gather" ? ffi::String("direct_all_to_all")
          : entry->kind == "all_reduce"
              ? ffi::String("root0_left_fold_broadcast")
          : entry->contract == "collective" ? ffi::String("direct_multicast")
                                            : ffi::String("direct_p2p");
      communications.push_back(CommunicationDescriptor(
          entry->site_id, entry->kind, algorithm, entry->source_region,
          entry->destination_region, parameterize_array(entry->source_coord),
          parameterize_array(entry->group_begin),
          parameterize_array(entry->group_end), parameterize(entry->parameter),
          entry->contract, entry->occurrence_extents, entry->source_span));
    }
  }
  ffi::Map<ffi::String, Integer> materialized_streams = storage_groups;
  if (pipeline_extent) {
    materialized_streams = {};
    for (const DFBDescriptor &dfb : dfbs)
      materialized_streams.Set(std::to_string(dfb->dfb_id),
                               Integer(dfb->dfb_id));
  }
  LogicalStreamPlan stream_plan =
      BuildLogicalStreamPlan(dfbs, materialized_streams);
  FinalizeLogicalStreamTransactionCounts(&stream_plan, dfbs, shared_bodies);
  auto stream_id = [&](int64_t resource) {
    if (resource < 0)
      return resource;
    if (resource >=
            static_cast<int64_t>(stream_plan.resource_stream_ids.size()) ||
        stream_plan.resource_stream_ids[resource] < 0)
      ThrowMalformed("Device metadata references an unplanned DFB transaction");
    return stream_plan.resource_stream_ids[resource];
  };
  DeviceIDRemapper remap_streams(0, &stream_plan.resource_stream_ids,
                                 /*annotate_streams=*/true);
  for (auto &slot_bodies : shared_bodies)
    for (DeviceCoreBody &core_body : slot_bodies) {
      core_body.body = DuplicateGenerationWaitRemover()(core_body.body);
      core_body.body = remap_streams(core_body.body);
    }

  ffi::Array<PipeDescriptor> stream_pipes;
  for (const PipeDescriptor &pipe : pipes)
    stream_pipes.push_back(PipeDescriptor(
        pipe->pipe_net_id, pipe->communication_id, pipe->record_index,
        pipe->src_coord, pipe->dst_begin, pipe->dst_end, pipe->contract,
        stream_id(pipe->payload_dfb_id), pipe->source_span));
  pipes = std::move(stream_pipes);

  ffi::Array<ComputeValueDescriptor> stream_values;
  for (const ComputeValueDescriptor &value : compute_values) {
    ffi::Array<Integer> materialization_refs;
    for (const Integer &resource : value->materialization_refs)
      materialization_refs.push_back(Integer(stream_id(resource->value)));
    stream_values.push_back(ComputeValueDescriptor(
        value->value_id, value->buffer, value->version,
        value->previous_value_id, value->accumulator_id, value->value_kind,
        value->logical_shape, value->compute_tile_shape, value->dtype,
        value->owner_slot, value->core_domain, materialization_refs,
        value->source_span));
  }
  compute_values = std::move(stream_values);

  DeviceResourceNames resource_names;
  std::map<int64_t, std::set<std::string>> tensor_name_sources;
  for (size_t index = 0; index < frontend->params.size(); ++index) {
    auto buffer = frontend->buffer_map.Get(frontend->params[index]);
    if (!buffer.has_value())
      ThrowMalformed("Tensor ABI parameter has no Buffer for resource naming");
    tensor_name_sources[index].insert(std::string(buffer.value()->name));
  }
  resource_names.by_kind[static_cast<size_t>(DeviceResourceKind::kTensor)] =
      BuildResourceNames(tensor_name_sources, DeviceResourceKind::kTensor,
                         "_tensor");

  std::map<int64_t, std::set<std::string>> stream_name_sources;
  for (size_t resource = 0; resource < stream_plan.resource_stream_ids.size();
       ++resource) {
    int64_t stream = stream_plan.resource_stream_ids[resource];
    auto names = resource_debug_names.find(resource);
    if (stream < 0 || names == resource_debug_names.end() ||
        names->second.empty())
      ThrowMalformed("logical stream has no stable resource debug source");
    stream_name_sources[stream].insert(names->second.begin(),
                                       names->second.end());
  }
  resource_names.by_kind[static_cast<size_t>(DeviceResourceKind::kDFBStream)] =
      BuildResourceNames(stream_name_sources, DeviceResourceKind::kDFBStream,
                         "_stream");

  std::map<int64_t, std::set<std::string>> pipe_name_sources;
  for (const PipeDescriptor &pipe : pipes) {
    if (pipe->communication_id < 0 ||
        pipe->communication_id >= static_cast<int64_t>(communications.size()))
      ThrowMalformed("PipeNet has no parent communication for resource naming");
    const CommunicationDescriptor &communication =
        communications[pipe->communication_id];
    pipe_name_sources[pipe->pipe_net_id].insert(
        std::string(communication->source_region->buffer->name));
  }
  resource_names.by_kind[static_cast<size_t>(DeviceResourceKind::kPipeNet)] =
      BuildResourceNames(pipe_name_sources, DeviceResourceKind::kPipeNet,
                         "_pipe");

  std::map<int64_t, std::set<std::string>> accumulator_name_sources;
  for (const AccumulatorDescriptor &accumulator : accumulators)
    accumulator_name_sources[accumulator->accumulator_id].insert(
        std::string(accumulator->accumulator_region->buffer->name));
  resource_names
      .by_kind[static_cast<size_t>(DeviceResourceKind::kAccumulator)] =
      BuildResourceNames(accumulator_name_sources,
                         DeviceResourceKind::kAccumulator, "_acc");

  const std::array<ffi::String, 3> slots = {"trisc", "ncrisc", "brisc"};
  for (size_t index = 0; index < slots.size(); ++index) {
    const ffi::String &slot = slots[index];
    bool compute = slot == "trisc";
    ffi::Array<Integer> tensor_indices;
    for (int64_t tensor : shared_tensor_indices[index])
      tensor_indices.push_back(Integer(tensor));
    Stmt slot_body = ParameterizeDeviceCoreStatements(shared_bodies[index], gx,
                                                      gy, slot, frontend->span);
    slot_body = DeviceCallExpressionSimplifier()(slot_body);
    slot_body = TransientDeviceAnnotationRemover()(slot_body);
    slot_body = DeviceResourceAliasMaterializer(&resource_names)(slot_body);
    PrimFunc func = MakeSlotFunction(
        frontend, target, shared_domain, operation, slot,
        compute ? "compute" : "datamovement",
        compute ? ffi::Optional<Integer>(std::nullopt)
                : ffi::Optional<Integer>(Integer(slot == "ncrisc" ? 0 : 1)),
        tensor_indices, compute ? "compute" : "datamovement",
        compute            ? "general"
        : slot == "ncrisc" ? "tensor_io"
                           : "transport",
        slot_body);
    ffi::String symbol = operation + "_" + slot;
    functions.Set(GlobalVar(symbol), func);
    order.push_back(symbol);
  }
  ffi::Array<ffi::String> tensor_effects;
  for (int effect : effects)
    tensor_effects.push_back(effect == 3   ? "inout"
                             : effect == 2 ? "output"
                                           : "input");
  auto tensors = BuildTensorTable(frontend, metadata, tensor_effects, true);
  ffi::Map<ffi::String, ffi::Any> attrs{
      {"tt.ir_stage", ffi::String("device")},
      {kTargetArchAttr, arch},
      {kLaunchGridAttr, CoreCoord(gx, gy)},
      {kOperationIdentityAttr, OperationIdentity(operation, frontend->span)},
      {kTensorTableAttr, tensors},
      {kDFBTableAttr, dfbs},
      {kDFBStorageGroupsAttr, stream_plan.descriptors},
      {"tt.stream_operands", Integer(1)},
      {"tt.communication_table", communications},
      {"tt.transport_table", pipes},
      {kKernelOrderAttr, order}};
  if (!accumulators.empty())
    attrs.Set(kAccumulatorTableAttr, accumulators);
  if (!compute_values.empty())
    attrs.Set(kComputeValueTableAttr, compute_values);
  if (!storage_groups.empty())
    attrs.Set(kInternalDFBStorageGroupsAttr, storage_groups);
  if (pipeline_extent) {
    attrs.Set(kInternalDFBTransactionRelationsAttr, pipeline_relations);
    attrs.Set("tt.pipeline_stages", Integer(pipeline_stages));
    attrs.Set("tt.pipeline_extent", Integer(pipeline_extent));
    attrs.Set("tt.pipeline_wait_policy", pipeline_wait_policy);
  }
  auto capacity = input->GetAttr<Integer>("tt.l1_capacity_bytes");
  if (capacity.has_value())
    attrs.Set("tt.l1_capacity_bytes", capacity.value());
  return ParameterizeCurrentDeviceTables(DeviceMetadataInterner::Rewrite(
      IRModule(functions, input->source_map, DictAttrs(attrs),
               input->global_infos)));
}

IRModule FormProgram(const IRModule &input) {
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
  if (!frontend->GetAttr<Integer>("tt.topology_normalized").has_value())
    ThrowMalformed(
        "input requires NormalizeTenstorrentTopology before formation");
  For xloop = Downcast<For>(frontend->body);
  For yloop = Downcast<For>(xloop->body);
  return FormMulticoreProgram(
      input, frontend, target.value(), frontend_global.value()->name_hint,
      target_arch.value(), grid_x, grid_y, kernel_body, xloop->loop_var,
      yloop->loop_var, buffer_table.value());
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

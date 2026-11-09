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
};

struct SerialIterationContext {
  size_t value_begin;
  size_t loop_resource_begin;
  size_t iteration_resource_begin;
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
  bool producer_forwarding;
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
        scheduled.push_back(MakeDeviceCall(dfb_copy_wait(), {call->args[arg]},
                                           statement->span));
      }
    };
    auto resource_id = [](const CallNode *call) {
      size_t operand =
          call->op.same_as(tensor_to_dfb_nd()) ? 1
          : IsPointPipeOperation(call) || IsRecordPipeOperation(call)
              ? PipeStreamOperand(call)
              : 0;
      return *as_const_int(call->args[operand]);
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
        bool pipe = IsPointPipeOperation(call) || IsRecordPipeOperation(call);
        size_t resource_operand = input_copy ? 1
                                  : pipe     ? PipeStreamOperand(call)
                                             : 0;
        int64_t id = *as_const_int(call->args[resource_operand]);
        const ResourceVersion &resource = resources_[id];
        if (resource.iteration < begin ||
            resource.iteration >= begin + pipeline_depth_)
          continue;
        if (resource.producer == "ncrisc") {
          // A forwarded panel must complete its Tensor copy before transport.
          // Complete only that dependency here; other input copies retain the
          // delayed window, and future K panels can overlap local computation.
          if ((call->op.same_as(dfb_pipe_send()) ||
               call->op.same_as(dfb_pipe_send_record())) &&
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
            Stmt completion = MakeDeviceCall(tenstorrent::dfb_copy_wait(),
                                             {Integer(id)}, statement->span);
            if (pipeline_wait_policy_ == "conservative")
              scheduled.push_back(completion);
            else
              completions.push_back(completion);
          }
        } else {
          outputs.push_back(statement);
          if (call->op.same_as(tenstorrent::dfb_to_tensor_nd()))
            outputs.push_back(MakeDeviceCall(tenstorrent::dfb_copy_wait(),
                                             {Integer(id)}, statement->span));
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
    for (int64_t id : observable_resources_)
      mark(id);
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
        size_t arg = call->op.same_as(tenstorrent::tensor_to_dfb_nd()) ? 1
                     : IsPointPipeOperation(call) || IsRecordPipeOperation(call)
                         ? PipeStreamOperand(call)
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
      bool previous_materialization = stable_loop_materializations_;
      stable_loop_materializations_ = true;
      size_t loop_resource_begin = resources_.size();
      for (int64_t i = 0; i < extent; ++i) {
        size_t iteration_resource_begin = resources_.size();
        if (i)
          serial_iterations_.push_back(
              {values_.size(), loop_resource_begin, iteration_resource_begin});
        StaticIterationSubstituter substitute(
            loop->loop_var, IntImm(loop->loop_var.dtype(), minimum + i));
        PlanImpl(substitute(loop->body));
        if (i)
          serial_iterations_.pop_back();
      }
      stable_loop_materializations_ = previous_materialization;
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
      } else if (call->op.same_as(tenstorrent::dfb_local_copy())) {
        last_use[*as_const_int(call->args[0])] = index;
      } else if (call->op.same_as(tenstorrent::dfb_gather_copy())) {
        last_use[*as_const_int(call->args[0])] = index;
      } else if (call->op.same_as(tenstorrent::dfb_collective_reduce())) {
        last_use[*as_const_int(call->args[0])] = index;
        last_use[*as_const_int(call->args[1])] = index;
      } else if (call->op.same_as(tenstorrent::dfb_pipe_wait()) ||
                 call->op.same_as(tenstorrent::dfb_pipe_wait_record())) {
        int64_t id = *as_const_int(call->args[PipeStreamOperand(call)]);
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
                                        {Integer(id)}, statement->span));
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

  int64_t ReadValue(const Buffer &buffer, const Span &span,
                    bool snapshot_loop_carry = true) {
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
    for (auto context = serial_iterations_.rbegin();
         snapshot_loop_carry && context != serial_iterations_.rend();
         ++context) {
      if (value >= static_cast<int64_t>(context->value_begin))
        continue;
      bool borrows_loop_resource = std::any_of(
          value_dfb_sources_[value].begin(), value_dfb_sources_[value].end(),
          [&](int64_t resource) {
            return resource >=
                       static_cast<int64_t>(context->loop_resource_begin) &&
                   resource <
                       static_cast<int64_t>(context->iteration_resource_begin);
          });
      if (borrows_loop_resource)
        return SnapshotLoopCarriedValue(value, span);
    }
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
    Reserve(&compute_, id, span);
    compute_.push_back(MakeDeviceCall(tenstorrent::compute_value_store(),
                                      {Integer(value), Integer(id)}, span));
    materialized_values_[value] = id;
    dfb_roots_[id] = value_roots_[value];
    return id;
  }

  int64_t SnapshotLoopCarriedValue(int64_t old, const Span &span) {
    int64_t storage = MaterializeValue(old, span);
    const Buffer buffer = values_[old]->buffer;
    int64_t value = DefineValue(buffer, values_[old]->accumulator_id, span);
    value_roots_[value] = value_roots_[old];
    value_dfb_sources_[value].insert(storage);
    ffi::Map<ffi::String, ffi::ObjectRef> attrs{
        {"tt.compute_kind", StringImm("elementwise")},
        {"tt.compute_dtype",
         StringImm(buffer->dtype == DataType::Float(32) ? "float32"
                                                        : "bfloat16")},
        {"tt.compute_tile_shape",
         ffi::Array<PrimExpr>{Integer(32), Integer(32)}},
        {"tt.logical_domain", buffer->shape},
        {"tt.expression",
         Call(buffer->dtype, dfb_load(), {Integer(storage)}, {}, span)},
        {"tt.input_shapes", ffi::Array<ffi::Array<PrimExpr>>{buffer->shape}},
        {"tt.value_inputs", ffi::Array<Integer>()},
        {"tt.value_input_shapes", ffi::Array<ffi::Array<PrimExpr>>()}};
    Wait(&compute_, storage, span);
    compute_.push_back(
        Evaluate(Call(DataType::Void(), compute_value(),
                      {Integer(value), Integer(storage)}, attrs, span),
                 span));
    return value;
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
      compute_.push_back(Evaluate(Call(DataType::Void(), compute_value(),
                                       {Integer(value)}, attrs, call->span),
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
    int64_t old = ReadValue(buffer, call->span, /*snapshot_loop_carry=*/false);
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
            {"tt.input_shapes", ffi::Array<ffi::Array<PrimExpr>>{domain}},
            {"tt.value_inputs", ffi::Array<Integer>()},
            {"tt.value_input_shapes", ffi::Array<ffi::Array<PrimExpr>>()}};
        compute_.push_back(Evaluate(Call(DataType::Void(), compute_value(),
                                         {Integer(value), Integer(storage)},
                                         load_attrs, call->span),
                                    call->span));
        materialized_values_[value] = storage;
        value_dfb_sources_[value].insert(storage);
        return;
      }
      ffi::Array<PrimExpr> args{Integer(value)};
      for (const PrimExpr &arg : dfb_args)
        args.push_back(arg);
      compute_.push_back(Evaluate(
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
      Wait(&transfer_, output_id, call->span);
      ffi::Array<PrimExpr> args{Integer(output_id), Integer(tensor)};
      for (const Range &range : output->region) {
        args.push_back(range->min);
        args.push_back(range->extent);
      }
      transfer_.push_back(MakeDeviceCall(dfb_to_tensor_nd(), args, call->span));
      if (transfers_)
        transfer_.push_back(
            MakeDeviceCall(dfb_copy_wait(), {Integer(output_id)}, call->span));
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
                             call->op.same_as(compute_value_store())
                         ? 1
                     : IsPointPipeOperation(call) || IsRecordPipeOperation(call)
                         ? PipeStreamOperand(call)
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
      std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash,
                         ffi::ObjectPtrEqual>
          no_values;
      DeviceExpressionRewriter rewriter(inputs, no_values);
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
      compute_.push_back(MakeDeviceCall(tenstorrent::accumulator_init(),
                                        {Integer(id)}, call->span));
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
                                           {Integer(output_id)}, call->span));
    } else {
      ThrowMalformed("accumulator must materialize to shared DFB or Tensor");
    }
    state.output_id = output_id;
    Reserve(&compute_, output_id, call->span);
    compute_.push_back(MakeDeviceCall(tenstorrent::accumulator_materialize(),
                                      {Integer(id), Integer(output_id)},
                                      call->span));
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
                                           {Integer(id)}, call->span));
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
      ffi::Map<ffi::String, ffi::ObjectRef> annotations{
          {"tt.compute_kind", StringImm("copy")},
          {"tt.compute_dtype",
           StringImm(copy->src->dtype == DataType::BFloat(16) ? "bfloat16"
                                                              : "float32")},
          {"tt.compute_tile_shape", src->tile_shape},
          {"tt.logical_domain", copy->src->shape},
          {"tt.input_shapes",
           ffi::Array<ffi::Array<PrimExpr>>{copy->src->shape}}};
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
                                           {Integer(output)}, call->span));
      return;
    }
    ThrowUnsupported("Copy must connect Tensor and DFB; shared Copy requires "
                     "compute legalization");
  }

  void PlanHighLevelTransfer(const Call &call) {
    if (call->args.size() != 10)
      ThrowMalformed("high-level transfer arity mismatch");
    int64_t sx = RequireStaticInteger(call->args[2], "transfer source x");
    int64_t sy = RequireStaticInteger(call->args[3], "transfer source y");
    int64_t bx = RequireStaticInteger(call->args[4], "transfer begin x");
    int64_t by = RequireStaticInteger(call->args[5], "transfer begin y");
    int64_t ex = RequireStaticInteger(call->args[6], "transfer end x");
    int64_t ey = RequireStaticInteger(call->args[7], "transfer end y");
    int64_t site = RequireStaticInteger(call->args[9], "transfer site_id");
    const auto *contract = call->args[8].as<StringImmNode>();
    if (!contract)
      ThrowMalformed("high-level transfer contract is not a string");
    bool source = core_x_ == sx && core_y_ == sy;
    bool destination =
        bx <= core_x_ && core_x_ < ex && by <= core_y_ && core_y_ < ey;
    if (!source && !destination)
      return;
    if (source && destination)
      ThrowMalformed("source-exclusive transfer includes the source Core");
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
    Call endpoint(DataType::Void(),
                  source ? tenstorrent::pipe_send() : tenstorrent::pipe_recv(),
                  source ? ffi::Array<PrimExpr>{call->args[0], Integer(0)}
                         : ffi::Array<PrimExpr>{Integer(0), call->args[1]},
                  annotations, call->span);
    PlanPipe(endpoint);
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

  void PlanAllGather(const Call &call) {
    if (call->args.size() != 8)
      ThrowMalformed("all_gather arity mismatch");
    BufferRegion source =
        NormalizeToAccessRegion(call->args[0], kAccessRead).region;
    BufferRegion destination =
        NormalizeToAccessRegion(call->args[1], kAccessWrite).region;
    FullRegion(source);
    FullRegion(destination);
    int64_t bx = RequireStaticInteger(call->args[2], "all_gather begin x");
    int64_t by = RequireStaticInteger(call->args[3], "all_gather begin y");
    int64_t ex = RequireStaticInteger(call->args[4], "all_gather end x");
    int64_t ey = RequireStaticInteger(call->args[5], "all_gather end y");
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
    Reserve(&compute_, output, call->span);
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
        Wait(&compute_, input, call->span);
        compute_.push_back(MakeCollectiveLocalCall(
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
    int64_t bx = RequireStaticInteger(call->args[2], "all_reduce begin x");
    int64_t by = RequireStaticInteger(call->args[3], "all_reduce begin y");
    int64_t ex = RequireStaticInteger(call->args[4], "all_reduce end x");
    int64_t ey = RequireStaticInteger(call->args[5], "all_reduce end y");
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
    Reserve(&compute_, output, call->span);
    int64_t local = Read(source->buffer, "trisc");
    Wait(&compute_, local, call->span);
    compute_.push_back(MakeCollectiveLocalCall(
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
        Wait(&compute_, input, call->span);
        compute_.push_back(MakeCollectiveLocalCall(
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
      forward = has_accumulator_compute_ && resource.producer == "ncrisc" &&
                resource.backing.has_value();
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
        ffi::Map<ffi::String, ffi::ObjectRef> annotations{
            {"tt.compute_kind", StringImm("copy")},
            {"tt.compute_dtype",
             StringImm(region->buffer->dtype == DataType::BFloat(16)
                           ? "bfloat16"
                           : "float32")},
            {"tt.compute_tile_shape", metadata->tile_shape},
            {"tt.logical_domain", region->buffer->shape},
            {"tt.input_shapes",
             ffi::Array<ffi::Array<PrimExpr>>{region->buffer->shape}}};
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
        endpoints_.push_back({transfer, occurrence, record, x, y, id, send,
                              send && forward, call->span});
      }
    }
    int64_t site = record[0]->value, record_index = record[1]->value;
    auto *body = send && !forward ? &pipe_ : &transfer_;
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
  std::vector<SerialIterationContext> serial_iterations_;
  std::unordered_map<int64_t, int64_t> materialized_values_;
  std::vector<AccumulatorLifetime> accumulators_;
  std::unordered_map<Buffer, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      accumulator_ids_;
  std::map<std::tuple<int64_t, int64_t, bool>, int64_t> occurrences_;
  std::unordered_map<int64_t, int64_t> collective_occurrences_;
  int64_t generated_collective_buffers_{0};
  std::unordered_set<int64_t> observable_resources_;
  bool has_producer_forwarding_{false};
  TransferKeys *transfers_;
  int64_t core_x_, core_y_, grid_x_{1}, grid_y_{1};
  std::vector<PipeEndpoint> endpoints_;
  ffi::Array<Stmt> pipe_;
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
  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    bool derived_compute_value = call->op.same_as(compute_value());
    if (!call->annotations.count("tt.stream_identity") &&
        !call->annotations.count("tt.internal_communication_id") &&
        !derived_compute_value)
      return call;
    auto annotations = call->annotations;
    annotations.erase("tt.stream_identity");
    annotations.erase("tt.internal_communication_id");
    if (derived_compute_value) {
      annotations.erase("tt.compute_dtype");
      annotations.erase("tt.compute_tile_shape");
      annotations.erase("tt.logical_domain");
    }
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }
};

class DeviceCallExpressionSimplifier : public StmtExprMutator {
  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call->op.same_as(tenstorrent::dfb_pipe_send_record()) &&
        !call->op.same_as(tenstorrent::dfb_pipe_recv_record()) &&
        !call->op.same_as(tenstorrent::dfb_pipe_wait_record()))
      return call;
    auto args = call->args;
    for (size_t index = 0; index < args.size(); ++index)
      args.Set(index, analyzer_.Simplify(args[index]));
    auto annotations = call->annotations;
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
    return Call(call->dtype, call->op, args, annotations, call->span);
  }

  arith::Analyzer analyzer_;
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
    std::sort(resources.begin(), resources.end());
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
  // Specialize and consume one proof instance at a time. Do not materialize a
  // gx*gy clone of the structured input before shared-body parameterization.
  for (int64_t index = 0; index < gx * gy; ++index) {
    int64_t x = index / gy, y = index % gy;
    StaticIterationSubstituter specialize_x(core_x_var, Integer(x));
    Stmt specialized = specialize_x(kernel_body);
    StaticIterationSubstituter specialize_y(core_y_var, Integer(y));
    specialized = specialize_y(specialized);
    GeneralDataflowPlanner planner(frontend, metadata, &transfer_keys, x, y);
    std::vector<For> independent_loops = FindIndependentLoops(specialized);
    planner.Plan(IndependentLoopBody(independent_loops, specialized));
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
      std::string storage_key = planner.ResourceStorageIdentity(local_id);
      if (inserted) {
        resource_storage_keys.emplace(global_id, std::move(storage_key));
        dfbs.push_back(DFBDescriptor(
            global_id, key, dfb->element_dtype, dfb->tile_shape,
            dfb->block_shape_in_tiles, dfb->block_count, dfb->tensor_backing,
            dfb->producer_slot, shared_domain, dfb->consumer_slot,
            shared_domain,
            independent_loops.empty() ? dfb->transaction_count_or_loop_relation
                                      : transaction_relation,
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
    for (const Integer &tensor : planner.UsedTensorIndices()) {
      const auto &effect = core_effects[tensor->value];
      effects[tensor->value] |= effect == "inout"    ? 3
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
      if (slot == "ncrisc")
        for (const Integer &tensor : planner.UsedTensorIndices())
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
        1, sender.span));
  }
  ffi::Array<CommunicationDescriptor> communications;
  auto normalized_topology =
      frontend->GetAttr<ffi::Array<NormalizedTopologyDescriptor>>(
          "tt.normalized_topology");
  if (normalized_topology.has_value()) {
    For logical_x = Downcast<For>(frontend->body);
    For logical_y = Downcast<For>(logical_x->body);
    StaticIterationSubstituter bind_x(
        logical_x->loop_var,
        Call(logical_x->loop_var.dtype(), tenstorrent::core_x(), {}));
    StaticIterationSubstituter bind_y(
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

/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file tenstorrent/transform/verify_device_ir.cc
 * \brief Read-only verifier for the Tenstorrent Device TIR v1/v2 schemas.
 */
#include "verify_device_ir.h"
#include "compact_device_program.h"
#include "device_table_compaction.h"
#include "infer_compute_requirements.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/type.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../ir/device_ir.h"
#include "../op/builtin.h"

namespace tvm {
namespace tl {
namespace tenstorrent {
namespace {

constexpr int64_t kDeviceIRVersion = 1;

[[noreturn]] void Fail(const std::string &message) {
  TVM_FFI_THROW(ValueError) << "[VerifyTenstorrentDeviceIR] " << message;
}

void Check(bool condition, const std::string &message) {
  if (!condition) {
    Fail(message);
  }
}

template <typename TObjectRef>
TObjectRef RequireModuleAttr(const IRModule &mod, const char *key) {
  ffi::Optional<TObjectRef> value = mod->GetAttr<TObjectRef>(key);
  Check(static_cast<bool>(value),
        std::string("missing required Module attr `") + key + "`");
  return value.value();
}

template <typename TObjectRef>
TObjectRef RequireFuncAttr(const tirx::PrimFunc &func,
                           const std::string &symbol, const char *key) {
  ffi::Optional<TObjectRef> value = func->GetAttr<TObjectRef>(key);
  Check(static_cast<bool>(value),
        "PrimFunc `" + symbol + "` is missing required attr `" + key + "`");
  return value.value();
}

bool IsSupportedSlot(const ffi::String &slot) {
  return slot == "trisc" || slot == "ncrisc" || slot == "brisc";
}

void VerifyCoord(const CoreCoord &coord, const CoreCoord &launch_grid,
                 const std::string &label) {
  Check(coord.defined(), label + " is undefined");
  Check(coord->x >= 0 && coord->y >= 0,
        label + " must contain non-negative coordinates");
  Check(coord->x < launch_grid->x && coord->y < launch_grid->y,
        label + " lies outside launch grid");
}

void VerifyDomain(const CoreDomain &domain, const CoreCoord &launch_grid,
                  const std::string &label) {
  Check(domain.defined(), label + " is undefined");
  Check(domain->begin.defined() && domain->end.defined(),
        label + " must define begin and end coordinates");
  Check(domain->begin->x >= 0 && domain->begin->y >= 0,
        label + " begin must be non-negative");
  Check(domain->begin->x < domain->end->x && domain->begin->y < domain->end->y,
        label + " must be a non-empty half-open rectangle");
  Check(domain->end->x <= launch_grid->x && domain->end->y <= launch_grid->y,
        label + " lies outside launch grid");
}

void VerifyPositiveShape(const ffi::Array<PrimExpr> &shape,
                         const std::string &label) {
  Check(!shape.empty(), label + " must not be empty");
  for (size_t index = 0; index < shape.size(); ++index) {
    Check(shape[index].defined(), label + " contains an undefined extent");
    if (const int64_t *extent = tirx::as_const_int(shape[index])) {
      Check(*extent > 0,
            label + " extent " + std::to_string(index) + " must be positive");
    }
  }
}

bool IsCanonicalNoOp(const tirx::Stmt &body) {
  const auto *evaluate = body.as<tirx::EvaluateNode>();
  if (evaluate == nullptr) {
    return false;
  }
  const int64_t *value = tirx::as_const_int(evaluate->value);
  return value != nullptr && *value == 0;
}

bool HasForbiddenPrefix(const std::string &op_name) {
  static constexpr const char *kForbiddenPrefixes[] = {
      "ttl.",     "ttkernel.", "emitc.",     "tl.cuda.",
      "tl.rocm.", "tl.metal.", "tl.webgpu.", "tir.cuda.",
      "tir.ptx.", "tir.rocm.", "tir.metal.",
  };
  return std::any_of(
      std::begin(kForbiddenPrefixes), std::end(kForbiddenPrefixes),
      [&](const char *prefix) { return op_name.rfind(prefix, 0) == 0; });
}

bool IsCurrentDeviceOp(const std::string &op_name) {
  static const std::unordered_set<std::string> allowed = {
      "tl.tt.core_x",
      "tl.tt.core_y",
      "tl.tt.dfb_compute",
      "tl.tt.dfb_load",
      "tl.tt.compute_broadcast",
      "tl.tt.compute_precision",
      "tl.tt.compute_value",
      "tl.tt.compute_value_load",
      "tl.tt.compute_value_store",
      "tl.tt.compute_value_gemm",
      "tl.tt.tensor_to_dfb_nd",
      "tl.tt.dfb_to_tensor_nd",
      "tl.tt.dfb_reserve",
      "tl.tt.dfb_wait",
      "tl.tt.dfb_copy_wait",
      "tl.tt.dfb_release",
      "tl.tt.dfb_pipe_send_record",
      "tl.tt.dfb_pipe_recv_record",
      "tl.tt.dfb_pipe_wait_record",
      "tl.tt.dfb_local_copy",
      "tl.tt.dfb_gather_copy",
      "tl.tt.dfb_collective_reduce",
      "tl.tt.accumulator_init",
      "tl.tt.gemm_update",
      "tl.tt.accumulator_materialize",
  };
  return allowed.count(op_name) != 0;
}

class CoreCoordinateSubstituter : public tirx::StmtExprMutator {
public:
  CoreCoordinateSubstituter(int64_t x, int64_t y) : x_(x), y_(y) {}

  PrimExpr VisitExpr_(const tirx::CallNode *op) final {
    if (op->op.same_as(tenstorrent::core_x()))
      return tirx::make_const(op->dtype, x_);
    if (op->op.same_as(tenstorrent::core_y()))
      return tirx::make_const(op->dtype, y_);
    tirx::Call call = Downcast<tirx::Call>(StmtExprMutator::VisitExpr_(op));
    if (!call.same_as(ffi::GetRef<tirx::Call>(op)))
      call.CopyOnWrite()->span = op->span;
    return call;
  }

  PrimExpr VisitExpr_(const tirx::CastNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    if (const auto *integer = value.as<IntImmNode>())
      return tirx::make_const(op->dtype, integer->value, op->span);
    return tirx::Cast(op->dtype, value, op->span);
  }

  PrimExpr VisitExpr(const PrimExpr &expr) final {
    PrimExpr result = StmtExprMutator::VisitExpr(expr);
    return result.dtype().is_handle() ||
                   tirx::SideEffect(result) > tirx::CallEffectKind::kPure
               ? result
               : analyzer_.Simplify(result);
  }

private:
  int64_t x_;
  int64_t y_;
  arith::Analyzer analyzer_;
};

using CurrentComputeKey = std::tuple<int64_t, int64_t, int64_t>;

void VerifyNoForbiddenOps(const tirx::PrimFunc &func, const std::string &symbol,
                          bool current = false) {
  tirx::PostOrderVisit(func->body, [&](const ffi::ObjectRef &object) {
    const auto *call = object.as<tirx::CallNode>();
    if (call == nullptr) {
      return;
    }
    const auto *op = call->op.as<OpNode>();
    if (op != nullptr) {
      const std::string op_name = op->name;
      Check(!HasForbiddenPrefix(op_name), "PrimFunc `" + symbol +
                                              "` contains forbidden op `" +
                                              op_name + "`");
      Check(op_name != "tl.tt.tile_add" && op_name != "tl.tt.tile_compute",
            "PrimFunc `" + symbol +
                "` retains intermediate op `tl.tt.tile_add`");
      if (current && (op_name.rfind("tl.tt.", 0) == 0 ||
                      SideEffect(ffi::GetRef<tirx::Call>(call)) >
                          tirx::CallEffectKind::kPure)) {
        Check(IsCurrentDeviceOp(op_name),
              "PrimFunc `" + symbol + "` contains non-current Device op `" +
                  op_name + "`");
      }
    }
  });
}

using TensorTable = std::unordered_map<int64_t, TensorDescriptor>;
using DFBTable = std::unordered_map<int64_t, DFBDescriptor>;

std::string SourceLocation(const Span &span) {
  if (!span.defined() || !span->source_name.defined())
    return "<unknown>";
  return std::string(span->source_name->name) + ":" +
         std::to_string(span->line) + ":" + std::to_string(span->column);
}

std::string OperationContext(const tirx::PrimFunc &function,
                             const tirx::Stmt &statement) {
  std::ostringstream context;
  context << "PrimFunc `"
          << function->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)
                 .value_or("<unknown>")
          << "` ";
  Span source = statement->span;
  const auto *evaluate = statement.as<tirx::EvaluateNode>();
  const auto *call = evaluate ? evaluate->value.as<tirx::CallNode>() : nullptr;
  if (call) {
    if (const auto *op = call->op.as<OpNode>())
      context << "op `" << op->name << "` ";
    context << "resource_ids=" << call->args << " ";
    if (call->span.defined())
      source = call->span;
  } else {
    context << "resource=statement ";
  }
  context << "source_span=" << SourceLocation(source) << ": ";
  return context.str();
}

std::string ResourceContext(const IRModule &mod, const DFBDescriptor &dfb) {
  std::string symbol = "<unknown>";
  for (const auto &[global, base] : mod->functions) {
    auto function = base.as<tirx::PrimFunc>();
    if (!function.has_value())
      continue;
    auto slot = function.value()->GetAttr<ffi::String>(kKernelSlotAttr);
    auto domain = function.value()->GetAttr<CoreDomain>(kCoreDomainAttr);
    if (slot.has_value() && domain.has_value() &&
        slot.value() == dfb->consumer_slot &&
        ffi::StructuralEqual()(domain.value(), dfb->consumer_domain)) {
      symbol = global->name_hint;
      break;
    }
  }
  return "PrimFunc `" + symbol + "` DFB " + std::to_string(dfb->dfb_id) +
         " source_span=" + SourceLocation(dfb->source_span) + ": ";
}

TensorTable VerifyTensorTable(const ffi::Array<TensorDescriptor> &tensors,
                              const CoreCoord &launch_grid,
                              bool general = false, bool require_span = true) {
  TensorTable table;
  for (const TensorDescriptor &tensor : tensors) {
    const std::string label =
        "Tensor descriptor " + std::to_string(tensor->global_arg_index);
    Check(tensor->global_arg_index >= 0,
          label + " has a negative global_arg_index");
    Check(table.emplace(tensor->global_arg_index, tensor).second,
          label + " duplicates a global_arg_index");
    VerifyPositiveShape(tensor->shape, label + " shape");
    VerifyPositiveShape(tensor->tile_shape, label + " tile_shape");
    VerifyPositiveShape(tensor->tile_grid_shape, label + " tile_grid_shape");
    Check(general ? tensor->tile_shape.size() == 2
                  : tensor->tile_shape.size() == tensor->shape.size(),
          label + " tile_shape rank does not match shape rank");
    Check(tensor->tile_grid_shape.size() == tensor->shape.size(),
          label + " tile_grid_shape rank does not match shape rank");
    Check(tensor->strides.empty() ||
              tensor->strides.size() == tensor->shape.size(),
          label + " strides rank does not match shape rank");
    Check(!tensor->dtype.is_void(), label + " has a void dtype");
    Check(!tensor->memory_space.empty(), label + " has no memory_space");
    Check(!tensor->memory_layout.empty(), label + " has no memory_layout");
    Check(tensor->effect == "input" || tensor->effect == "output" ||
              tensor->effect == "inout",
          label + " has invalid effect `" + std::string(tensor->effect) + "`");
    Check(tensor->alias_group >= 0, label + " has a negative alias_group");
    Check(!require_span || tensor->source_span.defined(),
          label + " has no source_span");
    if (tensor->shard_spec.defined()) {
      const ShardSpec &shard = tensor->shard_spec.value();
      VerifyDomain(shard->core_domain, launch_grid,
                   label + " shard core_domain");
      VerifyPositiveShape(shard->shard_shape, label + " shard_shape");
      Check(shard->shard_shape.size() == tensor->shape.size(),
            label + " shard_shape rank does not match shape rank");
      Check(shard->orientation == "row_major" ||
                shard->orientation == "column_major",
            label + " has invalid shard orientation");
    }
  }
  return table;
}

DFBTable VerifyDFBTable(const ffi::Array<DFBDescriptor> &dfbs,
                        const TensorTable &tensors,
                        const CoreCoord &launch_grid, bool general = false,
                        bool require_span = true) {
  DFBTable table;
  std::unordered_set<std::string> source_buffer_ids;
  for (const DFBDescriptor &dfb : dfbs) {
    const std::string label = "DFB " + std::to_string(dfb->dfb_id);
    Check(dfb->dfb_id >= 0, label + " has a negative dfb_id");
    Check(table.emplace(dfb->dfb_id, dfb).second, label + " duplicates dfb_id");
    Check(!dfb->source_buffer_identity.empty(),
          label + " has no source_buffer_identity");
    Check(source_buffer_ids.insert(dfb->source_buffer_identity).second,
          label + " duplicates source_buffer_identity `" +
              std::string(dfb->source_buffer_identity) + "`");
    Check(!dfb->element_dtype.is_void(), label + " has a void dtype");
    VerifyPositiveShape(dfb->tile_shape, label + " tile_shape");
    VerifyPositiveShape(dfb->block_shape_in_tiles,
                        label + " block_shape_in_tiles");
    Check(general ? dfb->tile_shape.size() == 2
                  : dfb->tile_shape.size() == dfb->block_shape_in_tiles.size(),
          label + " block_shape_in_tiles rank does not match tile_shape");
    Check(dfb->block_count.defined(), label + " has no block_count");
    if (const int64_t *count = tirx::as_const_int(dfb->block_count)) {
      Check(*count > 0, label + " block_count must be positive");
    }
    Check(dfb->transaction_count_or_loop_relation.defined(),
          label + " has no transaction_count_or_loop_relation");
    if (const int64_t *count =
            tirx::as_const_int(dfb->transaction_count_or_loop_relation)) {
      Check(*count > 0, label + " transaction count must be positive");
    }
    if (dfb->tensor_backing.defined()) {
      const TensorBacking &backing = dfb->tensor_backing.value();
      const int64_t tensor_index = backing->global_arg_index;
      Check(tensors.count(tensor_index) != 0,
            label + " references missing Tensor " +
                std::to_string(tensor_index));
      Check(backing->byte_offset.defined(),
            label + " Tensor backing has no byte_offset");
      if (const int64_t *byte_offset =
              tirx::as_const_int(backing->byte_offset)) {
        Check(*byte_offset >= 0,
              label + " Tensor backing byte_offset must be non-negative");
      }
    }
    Check(IsSupportedSlot(dfb->producer_slot),
          label + " has invalid producer_slot `" +
              std::string(dfb->producer_slot) + "`");
    Check(IsSupportedSlot(dfb->consumer_slot),
          label + " has invalid consumer_slot `" +
              std::string(dfb->consumer_slot) + "`");
    Check(general || dfb->producer_slot != dfb->consumer_slot,
          label + " must have distinct SPSC producer and consumer slots");
    VerifyDomain(dfb->producer_domain, launch_grid, label + " producer_domain");
    VerifyDomain(dfb->consumer_domain, launch_grid, label + " consumer_domain");
    Check(!require_span || dfb->source_span.defined(),
          label + " has no source_span");
  }
  return table;
}

void VerifyPipeTable(const ffi::Array<PipeDescriptor> &pipes,
                     const DFBTable &dfbs, const CoreCoord &launch_grid) {
  std::unordered_set<std::string> record_ids;
  std::unordered_map<int64_t, int64_t> next_record;
  std::unordered_map<int64_t, ffi::String> contracts;
  std::unordered_map<int64_t, int64_t> communications;
  int64_t expected_pipe_net = 0;
  for (const PipeDescriptor &pipe : pipes) {
    const std::string label = "PipeNet " + std::to_string(pipe->pipe_net_id) +
                              " record " + std::to_string(pipe->record_index);
    Check(pipe->pipe_net_id >= 0, label + " has a negative pipe_net_id");
    Check(pipe->communication_id >= 0,
          label + " has a negative communication_id");
    Check(pipe->record_index >= 0, label + " has a negative record_index");
    const std::string record_id = std::to_string(pipe->pipe_net_id) + ":" +
                                  std::to_string(pipe->record_index);
    Check(record_ids.insert(record_id).second,
          label + " duplicates a Pipe record identity");
    int64_t &expected = next_record[pipe->pipe_net_id];
    if (expected == 0)
      Check(pipe->pipe_net_id == expected_pipe_net++,
            label + " PipeNet IDs must be dense and deterministic");
    Check(pipe->record_index == expected,
          label + " is out of order; expected record_index " +
              std::to_string(expected));
    ++expected;
    auto [communication, new_pipe_net] =
        communications.emplace(pipe->pipe_net_id, pipe->communication_id);
    Check(new_pipe_net || communication->second == pipe->communication_id,
          label + " has inconsistent parent communication IDs");
    VerifyCoord(pipe->src_coord, launch_grid, label + " src_coord");
    VerifyDomain(CoreDomain(pipe->dst_begin, pipe->dst_end), launch_grid,
                 label + " destination range");
    Check(pipe->contract == "point_to_point" || pipe->contract == "collective",
          label + " has invalid contract `" + std::string(pipe->contract) +
              "`");
    auto [contract, inserted] =
        contracts.emplace(pipe->pipe_net_id, pipe->contract);
    Check(inserted || contract->second == pipe->contract,
          label +
              " mixes point-to-point and collective contracts in one PipeNet");
    if (pipe->contract == "point_to_point") {
      Check(pipe->dst_end->x == pipe->dst_begin->x + 1 &&
                pipe->dst_end->y == pipe->dst_begin->y + 1,
            label + " point_to_point destination must contain one core");
    }
    Check(dfbs.count(pipe->payload_dfb_id) != 0,
          label + " references missing DFB " +
              std::to_string(pipe->payload_dfb_id));
    Check(pipe->source_span.defined(), label + " has no source_span");
  }
}

struct Phase2AddPlan {
  DFBDescriptor input_a;
  DFBDescriptor input_b;
  DFBDescriptor output;
  int64_t rows;
  int64_t cols;
  int64_t tile_count;
};

int64_t RequireStaticInteger(const PrimExpr &expr, const std::string &label) {
  const int64_t *value = tirx::as_const_int(expr);
  Check(value != nullptr, label + " must be a static integer");
  return *value;
}

void VerifyUnitBlockShape(const DFBDescriptor &dfb) {
  const std::string label = "DFB " + std::to_string(dfb->dfb_id);
  Check(dfb->block_shape_in_tiles.size() == 2,
        label + " Phase 2 block_shape_in_tiles must have rank 2");
  Check(RequireStaticInteger(dfb->block_shape_in_tiles[0],
                             label + " block row extent") == 1 &&
            RequireStaticInteger(dfb->block_shape_in_tiles[1],
                                 label + " block column extent") == 1,
        label + " Phase 2 block_shape_in_tiles must be [1, 1]");
}

void VerifyDFBTensorContract(const DFBDescriptor &dfb,
                             const TensorDescriptor &tensor,
                             const ffi::String &producer_slot,
                             const ffi::String &consumer_slot,
                             const ffi::String &effect) {
  const std::string label = "DFB " + std::to_string(dfb->dfb_id);
  Check(dfb->producer_slot == producer_slot,
        label + " producer_slot must be `" + std::string(producer_slot) + "`");
  Check(dfb->consumer_slot == consumer_slot,
        label + " consumer_slot must be `" + std::string(consumer_slot) + "`");
  Check(tensor->effect == effect,
        "Tensor " + std::to_string(tensor->global_arg_index) +
            " effect must be `" + std::string(effect) + "`");
  Check(dfb->element_dtype == tensor->dtype,
        label + " dtype disagrees with its Tensor backing");
  Check(ffi::StructuralEqual()(dfb->tile_shape, tensor->tile_shape),
        label + " tile_shape disagrees with its Tensor backing");
  Check(ffi::StructuralEqual()(dfb->producer_domain, dfb->consumer_domain),
        label + " producer and consumer domains must match for Phase 2 SPSC");
  Check(RequireStaticInteger(dfb->tensor_backing.value()->byte_offset,
                             label + " Tensor backing byte_offset") == 0,
        label + " Phase 2 Tensor backing byte_offset must be zero");
  VerifyUnitBlockShape(dfb);
  Check(RequireStaticInteger(dfb->block_count, label + " block_count") == 2,
        label + " Phase 2 block_count must be exactly two");
  const int64_t transaction_count = RequireStaticInteger(
      dfb->transaction_count_or_loop_relation, label + " transaction count");
  Check(transaction_count == 1,
        label + " Phase 2 transaction count must be exactly one");
}

Phase2AddPlan VerifyPhase2AddDescriptors(
    const ffi::Array<DFBDescriptor> &dfbs, const TensorTable &tensors,
    const ffi::Array<PipeDescriptor> &pipes, const CoreCoord &launch_grid) {
  Check(launch_grid->x == 1 && launch_grid->y == 1,
        "Phase 2 Add requires a 1x1 launch grid");
  Check(tensors.size() == 3,
        "Phase 2 Add requires exactly three Tensor descriptors");
  Check(dfbs.size() == 3, "Phase 2 Add requires exactly three DFB descriptors");
  Check(pipes.empty(), "Phase 2 Add must not contain Pipe descriptors");

  std::unordered_map<int64_t, DFBDescriptor> by_tensor;
  for (const DFBDescriptor &dfb : dfbs) {
    const std::string label = "DFB " + std::to_string(dfb->dfb_id);
    Check(dfb->tensor_backing.defined(),
          label + " Phase 2 Add requires Tensor backing");
    const int64_t tensor_index = dfb->tensor_backing.value()->global_arg_index;
    Check(by_tensor.emplace(tensor_index, dfb).second,
          "Phase 2 Add DFBs must have unique Tensor backings");
  }

  for (int64_t tensor_index = 0; tensor_index < 3; ++tensor_index) {
    Check(tensors.count(tensor_index) != 0,
          "Phase 2 Add requires Tensor descriptors 0, 1, and 2");
    Check(by_tensor.count(tensor_index) != 0,
          "Phase 2 Add requires one DFB backed by Tensor " +
              std::to_string(tensor_index));
    const TensorDescriptor &tensor = tensors.at(tensor_index);
    const DFBDescriptor &dfb = by_tensor.at(tensor_index);
    Check(dfb->dfb_id == tensor_index,
          "Phase 2 Add DFB IDs must follow stable order 0, 1, 2");
    Check(dfb->source_buffer_identity ==
              "buffer." + std::to_string(tensor_index),
          "Phase 2 Add DFB source_buffer_identity must follow stable shared "
          "allocation order");
    Check(tensor->memory_space == "dram" &&
              tensor->memory_layout == "interleaved" &&
              !tensor->shard_spec.defined(),
          "Phase 2 Add Tensors must use unsharded interleaved DRAM layout");
    Check(tensor->alias_group == tensor_index,
          "Phase 2 Add Tensor alias_group must match global_arg_index");
  }

  const TensorDescriptor &tensor_a = tensors.at(0);
  const TensorDescriptor &tensor_b = tensors.at(1);
  const TensorDescriptor &tensor_c = tensors.at(2);
  Check(ffi::StructuralEqual()(tensor_a->shape, tensor_b->shape) &&
            ffi::StructuralEqual()(tensor_a->shape, tensor_c->shape),
        "Phase 2 Add Tensor shapes must match");
  Check(tensor_a->shape.size() == 2,
        "Phase 2 Add requires rank-2 Tensor descriptors");
  const int64_t rows =
      RequireStaticInteger(tensor_a->shape[0], "Phase 2 Add row extent");
  const int64_t cols =
      RequireStaticInteger(tensor_a->shape[1], "Phase 2 Add column extent");
  Check(tensor_a->dtype == tensor_b->dtype &&
            tensor_a->dtype == tensor_c->dtype,
        "Phase 2 Add Tensor dtypes must match");
  Check(tensor_a->dtype == DataType::BFloat(16) ||
            tensor_a->dtype == DataType::Float(32),
        "Phase 2 Add supports only bfloat16 and float32 Tensors");
  Check(ffi::StructuralEqual()(tensor_a->tile_shape, tensor_b->tile_shape) &&
            ffi::StructuralEqual()(tensor_a->tile_shape, tensor_c->tile_shape),
        "Phase 2 Add Tensor tile shapes must match");
  Check(tensor_a->tile_shape.size() == 2 &&
            RequireStaticInteger(tensor_a->tile_shape[0],
                                 "Phase 2 Add tile row extent") == 32 &&
            RequireStaticInteger(tensor_a->tile_shape[1],
                                 "Phase 2 Add tile column extent") == 32,
        "Phase 2 Add Tensor tile shape must be [32, 32]");

  DFBDescriptor input_a = by_tensor.at(0);
  DFBDescriptor input_b = by_tensor.at(1);
  DFBDescriptor output = by_tensor.at(2);
  VerifyDFBTensorContract(input_a, tensor_a, "ncrisc", "trisc", "input");
  VerifyDFBTensorContract(input_b, tensor_b, "ncrisc", "trisc", "input");
  VerifyDFBTensorContract(output, tensor_c, "trisc", "ncrisc", "output");

  Check(tensor_a->tile_grid_shape.size() == 2,
        "Phase 2 Add Tensor tile_grid_shape must have rank 2");
  Check(ffi::StructuralEqual()(tensor_a->tile_grid_shape,
                               tensor_b->tile_grid_shape) &&
            ffi::StructuralEqual()(tensor_a->tile_grid_shape,
                                   tensor_c->tile_grid_shape),
        "Phase 2 Add Tensor tile-grid shapes must match");
  const int64_t tile_rows = RequireStaticInteger(
      tensor_a->tile_grid_shape[0], "Phase 2 Add tile-grid row extent");
  const int64_t tile_cols = RequireStaticInteger(
      tensor_a->tile_grid_shape[1], "Phase 2 Add tile-grid column extent");
  Check(tile_rows > 0 && tile_cols > 0,
        "Phase 2 Add tile-grid extents must be positive");
  Check(tile_rows == 1 && tile_cols == 1,
        "Phase 2 Add Tensor tile-grid shape must be [1, 1]");
  return {
      std::move(input_a),   std::move(input_b), std::move(output), rows, cols,
      tile_rows * tile_cols};
}

enum class MarkerKind {
  kDFBReserve,
  kDFBWait,
  kTensorToDFB,
  kDFBToTensor,
  kDFBAdd,
};

struct Marker {
  MarkerKind kind;
  std::vector<int64_t> args;
};

const char *MarkerName(MarkerKind kind) {
  switch (kind) {
  case MarkerKind::kDFBReserve:
    return "tl.tt.dfb_reserve";
  case MarkerKind::kDFBWait:
    return "tl.tt.dfb_wait";
  case MarkerKind::kTensorToDFB:
    return "tl.tt.tensor_to_dfb";
  case MarkerKind::kDFBToTensor:
    return "tl.tt.dfb_to_tensor";
  case MarkerKind::kDFBAdd:
    return "tl.tt.dfb_add";
  }
  return "<unknown>";
}

MarkerKind ParseMarkerKind(const tirx::Call &call, const std::string &symbol) {
  if (call->op.same_as(dfb_reserve())) {
    return MarkerKind::kDFBReserve;
  }
  if (call->op.same_as(dfb_wait())) {
    return MarkerKind::kDFBWait;
  }
  if (call->op.same_as(tensor_to_dfb())) {
    return MarkerKind::kTensorToDFB;
  }
  if (call->op.same_as(dfb_to_tensor())) {
    return MarkerKind::kDFBToTensor;
  }
  if (call->op.same_as(dfb_add())) {
    return MarkerKind::kDFBAdd;
  }
  const auto *op = call->op.as<OpNode>();
  const std::string name = op == nullptr ? std::string(call->op->GetTypeKey())
                                         : std::string(op->name);
  Fail("Phase 2 PrimFunc `" + symbol + "` contains non-canonical op `" + name +
       "`");
}

size_t MarkerArity(MarkerKind kind) {
  switch (kind) {
  case MarkerKind::kDFBReserve:
  case MarkerKind::kDFBWait:
    return 1;
  case MarkerKind::kDFBAdd:
    return 4;
  case MarkerKind::kTensorToDFB:
  case MarkerKind::kDFBToTensor:
    return 6;
  }
  return 0;
}

std::vector<Marker> ParseMarkers(const tirx::Stmt &body,
                                 const std::string &symbol) {
  ffi::Array<tirx::Stmt> statements;
  if (const auto *sequence = body.as<tirx::SeqStmtNode>()) {
    statements = sequence->seq;
  } else {
    statements.push_back(body);
  }

  std::vector<Marker> markers;
  markers.reserve(statements.size());
  for (const tirx::Stmt &statement : statements) {
    const auto *evaluate = statement.as<tirx::EvaluateNode>();
    Check(evaluate != nullptr,
          "Phase 2 PrimFunc `" + symbol +
              "` body must be a flat sequence of marker Evaluate nodes");
    const auto *call_node = evaluate->value.as<tirx::CallNode>();
    Check(call_node != nullptr && call_node->dtype.is_void(),
          "Phase 2 PrimFunc `" + symbol +
              "` marker must be a void call_intrin");
    tirx::Call call = ffi::GetRef<tirx::Call>(call_node);
    MarkerKind kind = ParseMarkerKind(call, symbol);
    Check(call->args.size() == MarkerArity(kind),
          "marker `" + std::string(MarkerName(kind)) + "` in PrimFunc `" +
              symbol + "` has wrong arity");
    std::vector<int64_t> args;
    args.reserve(call->args.size());
    for (size_t index = 0; index < call->args.size(); ++index) {
      args.push_back(RequireStaticInteger(
          call->args[index], "marker `" + std::string(MarkerName(kind)) +
                                 "` argument " + std::to_string(index)));
    }
    markers.push_back({kind, std::move(args)});
  }
  return markers;
}

void VerifyMarkerSequence(const std::vector<Marker> &actual,
                          const std::vector<Marker> &expected,
                          const std::string &symbol) {
  Check(actual.size() == expected.size(),
        "Phase 2 PrimFunc `" + symbol + "` has " +
            std::to_string(actual.size()) + " markers; expected " +
            std::to_string(expected.size()));
  for (size_t index = 0; index < expected.size(); ++index) {
    Check(actual[index].kind == expected[index].kind,
          "Phase 2 PrimFunc `" + symbol + "` marker " + std::to_string(index) +
              " must be `" + MarkerName(expected[index].kind) + "`");
    Check(actual[index].args == expected[index].args,
          "Phase 2 PrimFunc `" + symbol + "` marker `" +
              MarkerName(expected[index].kind) + "` has wrong arguments");
  }
}

void VerifyPhase2Body(const tirx::PrimFunc &func, const std::string &symbol,
                      const ffi::String &slot,
                      const LogicalKernel &logical_kernel,
                      const Phase2AddPlan &plan) {
  const int64_t input_a_id = plan.input_a->dfb_id;
  const int64_t input_b_id = plan.input_b->dfb_id;
  const int64_t output_id = plan.output->dfb_id;
  if (slot == "trisc") {
    Check(
        logical_kernel->kind == "compute" && logical_kernel->role == "add",
        "Phase 2 trisc LogicalKernel must have kind `compute` and role `add`");
    VerifyMarkerSequence(
        ParseMarkers(func->body, symbol),
        {{MarkerKind::kDFBReserve, {output_id}},
         {MarkerKind::kDFBWait, {input_a_id}},
         {MarkerKind::kDFBWait, {input_b_id}},
         {MarkerKind::kDFBAdd,
          {input_a_id, input_b_id, output_id, plan.tile_count}}},
        symbol);
    return;
  }
  if (slot == "ncrisc") {
    Check(logical_kernel->kind == "datamovement" &&
              logical_kernel->role == "tensor_io",
          "Phase 2 ncrisc LogicalKernel must have kind `datamovement` and role "
          "`tensor_io`");
    VerifyMarkerSequence(ParseMarkers(func->body, symbol),
                         {{MarkerKind::kDFBReserve, {input_a_id}},
                          {MarkerKind::kTensorToDFB,
                           {0, input_a_id, 0, 0, plan.rows, plan.cols}},
                          {MarkerKind::kDFBReserve, {input_b_id}},
                          {MarkerKind::kTensorToDFB,
                           {1, input_b_id, 0, 0, plan.rows, plan.cols}},
                          {MarkerKind::kDFBWait, {output_id}},
                          {MarkerKind::kDFBToTensor,
                           {output_id, 2, 0, 0, plan.rows, plan.cols}}},
                         symbol);
    return;
  }
  Check(logical_kernel->kind == "idle" && logical_kernel->role == "idle",
        "Phase 2 brisc LogicalKernel must be canonical idle");
  Check(IsCanonicalNoOp(func->body),
        "Phase 2 brisc body must be the canonical Evaluate(0) no-op");
}

void VerifyFunctionABI(const tirx::PrimFunc &func, const std::string &symbol,
                       const TensorTable &tensors,
                       const ffi::Array<Integer> &tensor_arg_indices) {
  Check(func->params.size() == tensor_arg_indices.size(),
        "PrimFunc `" + symbol +
            "` parameter count does not match tt.tensor_arg_indices");
  int64_t previous_index = -1;
  for (size_t position = 0; position < tensor_arg_indices.size(); ++position) {
    const int64_t tensor_index = tensor_arg_indices[position]->value;
    Check(tensor_index > previous_index,
          "PrimFunc `" + symbol +
              "` tensor_arg_indices must be unique and strictly increasing");
    previous_index = tensor_index;
    auto tensor_it = tensors.find(tensor_index);
    Check(tensor_it != tensors.end(), "PrimFunc `" + symbol +
                                          "` references missing Tensor " +
                                          std::to_string(tensor_index));

    const tirx::Var &parameter = func->params[position];
    ffi::Optional<tirx::Buffer> buffer = func->buffer_map.Get(parameter);
    Check(buffer.defined(), "PrimFunc `" + symbol + "` Tensor parameter " +
                                std::to_string(position) +
                                " has no buffer_map entry");
    const TensorDescriptor &tensor = tensor_it->second;
    Check(buffer.value()->dtype == tensor->dtype,
          "PrimFunc `" + symbol + "` Tensor " + std::to_string(tensor_index) +
              " dtype disagrees with ABI");
    Check(ffi::StructuralEqual()(buffer.value()->shape, tensor->shape),
          "PrimFunc `" + symbol + "` Tensor " + std::to_string(tensor_index) +
              " shape disagrees with ABI");
  }

  ffi::Array<tirx::Var> undefined =
      tirx::UndefinedVars(func->body, func->params);
  Check(undefined.empty(),
        "PrimFunc `" + symbol +
            "` contains an undefined or cross-function Var reference");
  Check(tirx::VerifySSA(func), "PrimFunc `" + symbol + "` is not in SSA form");
}

void VerifyFunctions(const IRModule &mod, const ffi::String &target_arch,
                     const CoreCoord &launch_grid, const TensorTable &tensors,
                     const Phase2AddPlan *phase2_plan, bool general = false,
                     bool multicore = false, bool require_span = true,
                     bool current = false) {
  std::unordered_set<std::string> slots;
  std::unordered_set<std::string> symbols;
  std::unordered_set<std::string> kernel_ids;
  std::unordered_map<tirx::Var, std::string, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      var_owners;
  size_t prim_func_count = 0;

  for (const auto &[global_var, base_func] : mod->functions) {
    ffi::Optional<tirx::PrimFunc> optional_func =
        base_func.as<tirx::PrimFunc>();
    Check(optional_func.defined(), "Device IR contains non-PrimFunc global `" +
                                       std::string(global_var->name_hint) +
                                       "`");
    ++prim_func_count;
    tirx::PrimFunc func = optional_func.value();

    ffi::String symbol_attr = RequireFuncAttr<ffi::String>(
        func, global_var->name_hint, tvm::attr::kGlobalSymbol);
    const std::string symbol = symbol_attr;
    Check(!symbol.empty(), "PrimFunc has an empty global_symbol");
    Check(symbols.insert(symbol).second,
          "duplicate PrimFunc global_symbol `" + symbol + "`");

    ffi::String slot =
        RequireFuncAttr<ffi::String>(func, symbol, kKernelSlotAttr);
    Check(IsSupportedSlot(slot), "PrimFunc `" + symbol +
                                     "` has invalid slot `" +
                                     std::string(slot) + "`");
    std::string slot_identity = slot;
    if (multicore) {
      CoreDomain domain =
          RequireFuncAttr<CoreDomain>(func, symbol, kCoreDomainAttr);
      VerifyDomain(domain, launch_grid,
                   "PrimFunc `" + symbol + "` core_domain");
      Check(domain->end->x == domain->begin->x + 1 &&
                domain->end->y == domain->begin->y + 1,
            "Phase 6 slot core_domain must contain exactly one Core");
      slot_identity = std::to_string(domain->begin->x) + ":" +
                      std::to_string(domain->begin->y) + ":" +
                      std::string(slot);
    }
    Check(slots.insert(slot_identity).second,
          "operation contains duplicate Core/slot `" + slot_identity + "`");

    ffi::String thread =
        RequireFuncAttr<ffi::String>(func, symbol, kKernelThreadAttr);
    ffi::Optional<Integer> noc_index = func->GetAttr<Integer>(kNocIndexAttr);
    if (slot == "trisc") {
      Check(thread == "compute",
            "PrimFunc `" + symbol + "` trisc thread must be `compute`");
      Check(!noc_index.defined(),
            "PrimFunc `" + symbol + "` trisc must not define tt.noc_index");
    } else {
      Check(thread == "datamovement",
            "PrimFunc `" + symbol +
                "` data-movement slot must use thread `datamovement`");
      Check(noc_index.defined(),
            "PrimFunc `" + symbol + "` data-movement slot has no noc_index");
      const int64_t expected_noc = slot == "ncrisc" ? 0 : 1;
      Check(noc_index.value()->value == expected_noc,
            "PrimFunc `" + symbol + "` has wrong noc_index for slot `" +
                std::string(slot) + "`");
    }

    Integer calling_conv =
        RequireFuncAttr<Integer>(func, symbol, tvm::attr::kCallingConv);
    Check(calling_conv->value ==
              static_cast<int64_t>(CallingConv::kDeviceKernelLaunch),
          "PrimFunc `" + symbol + "` calling_conv is not DEVICE_KERNEL_LAUNCH");

    Target target = RequireFuncAttr<Target>(func, symbol, tvm::attr::kTarget);
    Check(target->kind->name == "tenstorrent",
          "PrimFunc `" + symbol + "` target kind is not tenstorrent");
    ffi::Optional<ffi::String> function_arch =
        target->GetAttr<ffi::String>("arch");
    Check(static_cast<bool>(function_arch) &&
              function_arch.value() == target_arch,
          "PrimFunc `" + symbol +
              "` target arch disagrees with Module tt.target_arch");

    LogicalKernel logical_kernel =
        RequireFuncAttr<LogicalKernel>(func, symbol, kLogicalKernelAttr);
    Check(!logical_kernel->kernel_id.empty(),
          "PrimFunc `" + symbol + "` has an empty logical kernel ID");
    Check(kernel_ids.insert(logical_kernel->kernel_id).second,
          "duplicate logical kernel ID `" +
              std::string(logical_kernel->kernel_id) + "`");
    Check(!logical_kernel->kind.empty(),
          "PrimFunc `" + symbol + "` has no logical kernel kind");
    Check(!logical_kernel->role.empty(),
          "PrimFunc `" + symbol + "` has no logical kernel role");
    Check(!require_span || logical_kernel->source_span.defined(),
          "PrimFunc `" + symbol + "` logical kernel has no source_span");

    CoreDomain core_domain =
        RequireFuncAttr<CoreDomain>(func, symbol, kCoreDomainAttr);
    VerifyDomain(core_domain, launch_grid,
                 "PrimFunc `" + symbol + "` core_domain");
    ffi::Array<Integer> tensor_arg_indices =
        RequireFuncAttr<ffi::Array<Integer>>(func, symbol,
                                             kTensorArgIndicesAttr);
    VerifyFunctionABI(func, symbol, tensors, tensor_arg_indices);
    if (phase2_plan != nullptr) {
      if (slot == "ncrisc") {
        Check(tensor_arg_indices.size() == 3 &&
                  tensor_arg_indices[0]->value == 0 &&
                  tensor_arg_indices[1]->value == 1 &&
                  tensor_arg_indices[2]->value == 2,
              "Phase 2 ncrisc must own Tensor ABI indices [0, 1, 2]");
      } else {
        Check(tensor_arg_indices.empty(),
              "Phase 2 " + std::string(slot) + " must not own Tensor ABI args");
      }
    }
    Check(IsVoidType(func->ret_type),
          "PrimFunc `" + symbol + "` must return void");
    VerifyNoForbiddenOps(func, symbol, current);
    if (phase2_plan != nullptr) {
      VerifyPhase2Body(func, symbol, slot, logical_kernel, *phase2_plan);
    } else if (!general) {
      Check(IsCanonicalNoOp(func->body),
            "Phase 1 PrimFunc `" + symbol +
                "` body must be the canonical Evaluate(0) no-op");
    }
    Check(!func->attrs->dict.count(kBufferMetadataTableAttr),
          "PrimFunc `" + symbol + "` retains Form-input-only attr `" +
              std::string(kBufferMetadataTableAttr) + "`");

    auto record_var = [&](const tirx::Var &var) {
      auto owner = var_owners.find(var);
      if (owner != var_owners.end() &&
          owner->second != (multicore ? symbol : std::string(slot))) {
        Fail("Var `" + std::string(var->name_hint) +
             "` is shared across Device IR slots `" + owner->second +
             "` and `" + std::string(slot) + "`");
      }
      var_owners.emplace(var, multicore ? symbol : std::string(slot));
    };
    for (const tirx::Var &param : func->params) {
      record_var(param);
    }
    tirx::PostOrderVisit(func->body, [&](const ffi::ObjectRef &object) {
      if (const auto *var = object.as<tirx::VarNode>()) {
        record_var(ffi::GetRef<tirx::Var>(var));
      }
    });
  }

  if (multicore) {
    Check(launch_grid->x <= 256 && launch_grid->y <= 256 &&
              launch_grid->x * launch_grid->y <= 256,
          "Phase 6 static launch grid is limited to 256 Cores");
    Check(prim_func_count ==
              static_cast<size_t>(3 * launch_grid->x * launch_grid->y),
          "Phase 6 operation must contain three slot PrimFuncs per Core");
    return;
  }
  Check(prim_func_count == 3,
        "operation must contain exactly three slot PrimFuncs");
  Check(slots.size() == 3 && slots.count("trisc") != 0 &&
            slots.count("ncrisc") != 0 && slots.count("brisc") != 0,
        "operation must contain exactly trisc, ncrisc, and brisc slots");
}

using namespace tirx;

ffi::String ComputeString(const Call &call, const char *key) {
  auto value = call->annotations.Get(key);
  Check(value.has_value(),
        std::string("dfb_compute missing annotation ") + key);
  const auto *text = value.value().as<StringImmNode>();
  Check(text != nullptr,
        std::string("dfb_compute annotation must be StringImm: ") + key);
  return text->value;
}

int64_t ComputeInteger(const Call &call, const char *key) {
  auto value = call->annotations.Get(key);
  Check(value.has_value(),
        std::string("dfb_compute missing annotation ") + key);
  auto expr = value.value().as<PrimExpr>();
  Check(expr.has_value(),
        std::string("dfb_compute invalid integer annotation ") + key);
  return RequireStaticInteger(expr.value(), key);
}

ffi::Array<PrimExpr> CheckedShape(const ffi::Any &value,
                                  const std::string &owner) {
  auto array = value.as<ffi::Array<ffi::Any>>();
  Check(array.has_value(), owner + " annotation must be an Array");
  ffi::Array<PrimExpr> shape;
  for (const ffi::Any &item : array.value()) {
    auto extent = item.as<PrimExpr>();
    Check(extent.has_value(), owner + " shape entries must be PrimExpr");
    shape.push_back(extent.value());
  }
  return shape;
}

ffi::Array<ffi::Array<PrimExpr>> CheckedShapes(const ffi::Any &value,
                                               const std::string &owner) {
  auto array = value.as<ffi::Array<ffi::Any>>();
  Check(array.has_value(), owner + " annotation must be an Array");
  ffi::Array<ffi::Array<PrimExpr>> result;
  for (const ffi::Any &item : array.value())
    result.push_back(CheckedShape(item, owner));
  return result;
}

ffi::Array<ffi::Array<Integer>> CheckedMaps(const ffi::Any &value) {
  auto arrays = CheckedShapes(value, "tt.access_maps");
  ffi::Array<ffi::Array<Integer>> maps;
  for (const auto &array : arrays) {
    ffi::Array<Integer> axes;
    for (const PrimExpr &axis : array) {
      auto integer = axis.as<IntImm>();
      Check(integer.has_value(), "tt.access_maps entries must be IntImm");
      axes.push_back(integer.value());
    }
    maps.push_back(axes);
  }
  return maps;
}

ffi::Array<PrimExpr> ComputeShape(const Call &call, const char *key) {
  auto value = call->annotations.Get(key);
  Check(value.has_value(),
        std::string("dfb_compute missing annotation ") + key);
  return CheckedShape(value.value(), key);
}

bool SupportedComputeDType(DataType dtype) {
  return dtype == DataType::BFloat(16) || dtype == DataType::Float(32);
}

void VerifyTileGrid(const ffi::Array<PrimExpr> &shape,
                    const ffi::Array<PrimExpr> &grid,
                    const std::string &owner) {
  Check(shape.size() == grid.size() && !shape.empty(),
        owner + " shape/grid rank mismatch");
  for (size_t i = 0; i < shape.size(); ++i) {
    int64_t extent = RequireStaticInteger(shape[i], owner + " shape extent");
    int64_t tile = i + 2 >= shape.size() ? 32 : 1;
    Check(extent > 0 && (extent == 1 || extent % tile == 0),
          owner + " unsupported logical extent");
    int64_t expected = (extent + tile - 1) / tile;
    Check(RequireStaticInteger(grid[i], owner + " tile-grid extent") ==
              expected,
          owner + " logical shape disagrees with tile-grid metadata");
  }
}

bool IsCompatibleLogicalInput(const ffi::Array<PrimExpr> &producer,
                              const ffi::Array<PrimExpr> &use) {
  if (producer.size() < use.size())
    return false;
  size_t leading = producer.size() - use.size();
  for (size_t axis = 0; axis < leading; ++axis)
    if (!is_one(producer[axis]))
      return false;
  for (size_t axis = 0; axis < use.size(); ++axis)
    if (!ffi::StructuralEqual()(producer[leading + axis], use[axis]) &&
        !is_one(use[axis]))
      return false;
  return true;
}

// Tensor coordinates retain their ABI rank. Local DFBs may omit only leading
// singleton slice axes, never interior axes or nontrivial batch dimensions.
ffi::Array<PrimExpr> LocalTransferShape(const ffi::Array<PrimExpr> &shape,
                                        const DFBDescriptor &dfb) {
  size_t rank = dfb->block_shape_in_tiles.size();
  Check(shape.size() >= rank, "Tensor transfer rank is smaller than DFB rank");
  size_t leading = shape.size() - rank;
  ffi::Array<PrimExpr> local;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    if (axis < leading)
      Check(RequireStaticInteger(shape[axis], "Tensor leading slice extent") ==
                1,
            "Tensor transfer may remove only leading singleton slice axes");
    else
      local.push_back(shape[axis]);
  }
  return local;
}

using BoundVars = std::unordered_set<const VarNode *>;

void VerifyGeneralCompute(const Call &call, const DFBTable &dfbs,
                          const BoundVars *bound_vars = nullptr,
                          bool allow_stream_inplace = false) {
  Check(!call->args.empty(), "dfb_compute must have an output DFB");
  std::vector<DFBDescriptor> operands;
  std::unordered_set<int64_t> input_ids;
  arith::Analyzer analyzer;
  for (size_t i = 0; i < call->args.size(); ++i) {
    int64_t id = RequireStaticInteger(analyzer.Simplify(call->args[i]),
                                      "dfb_compute resource ID");
    Check(dfbs.count(id), "dfb_compute references missing DFB");
    operands.push_back(dfbs.at(id));
    if (i)
      input_ids.insert(id);
  }
  Check(allow_stream_inplace || !input_ids.count(operands[0]->dfb_id),
        "dfb_compute output must be a fresh generation");
  const DFBDescriptor &output = operands[0];
  ffi::String dtype = ComputeString(call, "tt.compute_dtype");
  Check(dtype == (output->element_dtype == DataType::BFloat(16) ? "bfloat16"
                                                                : "float32"),
        "dfb_compute dtype annotation disagrees with output DFB");
  auto tile = ComputeShape(call, "tt.compute_tile_shape");
  Check(ffi::StructuralEqual()(tile, output->tile_shape),
        "dfb_compute tile shape mismatch");
  auto shape = ComputeShape(call, "tt.logical_domain");
  VerifyTileGrid(shape, output->block_shape_in_tiles, "dfb_compute output");
  auto maps_value = call->annotations.Get("tt.access_maps");
  ffi::Array<ffi::Array<Integer>> maps =
      maps_value.has_value() ? CheckedMaps(maps_value.value())
                             : ffi::Array<ffi::Array<Integer>>();
  if (maps_value.has_value())
    Check(maps.size() + 1 == operands.size(),
          "dfb_compute access maps must cover every input operand");
  ffi::String kind = ComputeString(call, "tt.compute_kind");
  auto input_shapes_value = call->annotations.Get("tt.input_shapes");
  Check(input_shapes_value.has_value(), "dfb_compute missing tt.input_shapes");
  auto input_shapes =
      CheckedShapes(input_shapes_value.value(), "tt.input_shapes");
  Check(input_shapes.size() + 1 == operands.size(),
        "dfb_compute input shape count mismatch");
  auto access_ids_attr = call->annotations.Get("tt.access_ids");
  if (access_ids_attr.has_value()) {
    auto access_ids = access_ids_attr.value().as<ffi::Array<Integer>>();
    Check(access_ids.has_value() && access_ids.value().size() == maps.size(),
          "dfb_compute access_ids must cover every input access");
    std::unordered_set<int64_t> unique_accesses;
    for (const Integer &access_id : access_ids.value())
      Check(access_id->value >= 0 &&
                unique_accesses.insert(access_id->value).second,
            "dfb_compute access_ids must be nonnegative and unique");
  }
  std::unordered_map<int64_t, ffi::Array<PrimExpr>> shape_by_input;
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    auto input_shape = input_shapes[i];
    VerifyTileGrid(input_shape, operands[i + 1]->block_shape_in_tiles,
                   "dfb_compute input");
    shape_by_input.emplace(operands[i + 1]->dfb_id, input_shape);
    if (!maps_value.has_value())
      continue;
    Check(maps[i].size() == input_shape.size(),
          "dfb_compute access map rank disagrees with input");
    if (kind == "elementwise" || kind == "typecast" || kind == "fill") {
      std::unordered_set<int64_t> mapped;
      for (size_t axis = 0; axis < maps[i].size(); ++axis) {
        int64_t output_axis = maps[i][axis]->value;
        Check(output_axis >= -1 &&
                  output_axis < static_cast<int64_t>(shape.size()),
              "dfb_compute access map axis is out of range");
        if (output_axis >= 0) {
          Check(mapped.insert(output_axis).second,
                "dfb_compute access map duplicates an output axis");
          Check(ffi::StructuralEqual()(input_shape[axis], shape[output_axis]),
                "dfb_compute mapped input extent disagrees with output");
        }
      }
    }
  }
  bool scalar = kind == "elementwise" || kind == "fill" || kind == "typecast";
  Check(scalar || kind == "copy" || kind == "transpose" || kind == "gemm" ||
            kind == "reduce",
        "dfb_compute has unknown compute kind");
  if (scalar) {
    auto expression_value = call->annotations.Get("tt.expression");
    Check(expression_value.has_value(), "scalar dfb_compute has no expression");
    auto expression = expression_value.value().as<PrimExpr>();
    Check(expression.has_value(), "scalar dfb_compute has invalid expression");
    Check(expression.value().dtype() == output->element_dtype,
          "dfb_compute expression dtype disagrees with output");
    std::unordered_set<int64_t> used;
    PostOrderVisit(expression.value(), [&](const ffi::ObjectRef &object) {
      const auto *variable = object.as<VarNode>();
      bool bound_variable =
          variable && bound_vars && bound_vars->count(variable);
      Check(!object.as<BufferLoadNode>() && (!variable || bound_variable),
            "Device expression retains BufferLoad or free scalar Var");
      if (auto expr = object.as<PrimExpr>()) {
        bool boolean_node =
            object.as<EQNode>() || object.as<NENode>() || object.as<LTNode>() ||
            object.as<LENode>() || object.as<GTNode>() || object.as<GENode>() ||
            object.as<AndNode>() || object.as<OrNode>() || object.as<NotNode>();
        Check(bound_variable || object.as<IntImmNode>() ||
                  SupportedComputeDType(expr.value().dtype()) ||
                  (boolean_node && expr.value().dtype() == DataType::Bool()),
              "Device expression has unsupported intermediate dtype");
        Check(bound_variable || object.as<IntImmNode>() ||
                  object.as<FloatImmNode>() || object.as<CastNode>() ||
                  object.as<CallNode>() || object.as<AddNode>() ||
                  object.as<SubNode>() || object.as<MulNode>() ||
                  object.as<DivNode>() || object.as<MinNode>() ||
                  object.as<MaxNode>() || object.as<EQNode>() ||
                  object.as<NENode>() || object.as<LTNode>() ||
                  object.as<LENode>() || object.as<GTNode>() ||
                  object.as<GENode>() || object.as<AndNode>() ||
                  object.as<OrNode>() || object.as<NotNode>() ||
                  object.as<SelectNode>(),
              "Device expression contains an unsupported expression node");
        if (const auto *select = object.as<SelectNode>())
          Check(select->condition.dtype() == DataType::Bool() &&
                    SupportedComputeDType(select->dtype) &&
                    select->true_value.dtype() == select->dtype &&
                    select->false_value.dtype() == select->dtype,
                "Device Select requires a scalar bool condition and matching "
                "BF16/FP32 branches");
        auto verify_comparison = [&](const auto *comparison) {
          if (comparison)
            Check(SupportedComputeDType(comparison->a.dtype()) &&
                      comparison->a.dtype() == comparison->b.dtype() &&
                      comparison->dtype == DataType::Bool(),
                  "Device comparison requires matching BF16/FP32 operands");
        };
        verify_comparison(object.as<EQNode>());
        verify_comparison(object.as<NENode>());
        verify_comparison(object.as<LTNode>());
        verify_comparison(object.as<LENode>());
        verify_comparison(object.as<GTNode>());
        verify_comparison(object.as<GENode>());
        if (const auto *cast = object.as<CastNode>()) {
          Check(cast->annotations.empty(),
                "Device Cast annotations are unsupported");
          Check(
              cast->value.dtype() != DataType::Bool(),
              "Device boolean values are predicates, not numeric cast inputs");
        }
      }
      if (const auto *node = object.as<CallNode>()) {
        Check(node->annotations.empty(),
              "Device scalar Call annotations are unsupported");
        const auto *operation = node->op.as<OpNode>();
        Check(operation != nullptr,
              "Device expression contains an external call");
        if (node->op.same_as(dfb_load())) {
          Check(node->args.size() == 1, "dfb_load must have one resource ID");
          int64_t id =
              RequireStaticInteger(node->args[0], "dfb_load resource ID");
          Check(input_ids.count(id),
                "dfb_load references an undeclared input DFB");
          Check(node->dtype == dfbs.at(id)->element_dtype,
                "dfb_load dtype mismatch");
          used.insert(id);
        } else if (node->op.same_as(compute_broadcast())) {
          Check(node->args.size() >= 2,
                "compute_broadcast requires one value and broadcast axes");
          const auto *load = node->args[0].as<CallNode>();
          Check(load && load->op.same_as(dfb_load()) && load->args.size() == 1,
                "compute_broadcast input must be a direct dfb_load");
          int64_t id = RequireStaticInteger(load->args[0],
                                            "broadcast input resource ID");
          Check(shape_by_input.count(id),
                "compute_broadcast references an undeclared input DFB");
          const auto &input_shape = shape_by_input.at(id);
          Check(input_shape.size() == shape.size(),
                "compute_broadcast rank disagrees with logical domain");
          std::unordered_set<int64_t> axes;
          for (size_t i = 1; i < node->args.size(); ++i) {
            int64_t axis =
                RequireStaticInteger(node->args[i], "compute_broadcast axis");
            Check(axis >= 0 && axis < static_cast<int64_t>(shape.size()) &&
                      axes.insert(axis).second,
                  "compute_broadcast axes must be unique and in range");
          }
          for (size_t axis = 0; axis < shape.size(); ++axis)
            Check(ffi::StructuralEqual()(input_shape[axis],
                                         axes.count(axis) ? PrimExpr(Integer(1))
                                                          : shape[axis]),
                  "compute_broadcast input shape disagrees with its axes");
          Check(node->dtype == load->dtype,
                "compute_broadcast must preserve its input dtype");
        } else {
          Check(
              node->args.size() == 1 && node->args[0].dtype() == node->dtype,
              "Device unary operation requires one operand of the same dtype");
          const std::string name = operation->name;
          Check(name == "tirx.exp" || name == "tirx.log" ||
                    name == "tirx.sqrt" || name == "tirx.tanh" ||
                    name == "tirx.rsqrt" || name == "tirx.floor" ||
                    name == "tirx.ceil" || name == "tirx.exp2" ||
                    name == "tirx.log2" || name == "tirx.sin" ||
                    name == "tirx.cos" || name == "tirx.fabs",
                "Device expression contains unsupported scalar operation " +
                    name);
        }
      }
    });
    Check(used == input_ids,
          "dfb_compute has unused declared expression inputs");
    Check(kind != "fill" || input_ids.empty(), "fill must not read DFB inputs");
  } else if (kind == "copy" || kind == "transpose") {
    Check(operands.size() == 2, "copy/transpose requires one input");
    Check(operands[1]->element_dtype == output->element_dtype,
          "copy/transpose dtype mismatch");
    auto expected = operands[1]->block_shape_in_tiles;
    if (kind == "transpose") {
      Check(expected.size() >= 2, "transpose requires rank at least two");
      auto axes_value = call->annotations.Get("tt.axes");
      Check(axes_value.has_value(), "transpose missing tt.axes");
      auto axes = CheckedShape(axes_value.value(), "tt.axes");
      Check(axes.size() == expected.size(), "transpose axes rank mismatch");
      for (size_t i = 0; i < expected.size(); ++i) {
        int64_t axis = i + 2 < expected.size()
                           ? i
                           : (i + 1 == expected.size() ? i - 1 : i + 1);
        Check(RequireStaticInteger(axes[i], "transpose axis") == axis,
              "transpose supports only last-two-axis permutation");
      }
      PrimExpr last = expected[expected.size() - 1];
      expected.Set(expected.size() - 1, expected[expected.size() - 2]);
      expected.Set(expected.size() - 2, last);
    }
    Check(ffi::StructuralEqual()(expected, output->block_shape_in_tiles),
          "copy/transpose input-output tile-grid mismatch");
  } else {
    int64_t clear = ComputeInteger(call, "tt.clear");
    Check(clear == 0 || clear == 1, "compute tt.clear must be Boolean");
    ffi::String accumulation = ComputeString(call, "tt.accum_dtype");
    Check(accumulation == "float32" ||
              (kind == "gemm" && accumulation == "bfloat16" &&
               dtype == "bfloat16") ||
              (kind == "reduce" &&
               ComputeString(call, "tt.reduce_kind") != "sum" &&
               accumulation == dtype),
          "unsupported accumulation dtype");
    size_t count = kind == "gemm" ? 3 : 2;
    Check(operands.size() == count + (clear ? 0 : 1),
          "GEMM/reduce operand count mismatch");
    if (!clear) {
      Check(operands.back()->element_dtype == output->element_dtype &&
                ffi::StructuralEqual()(operands.back()->block_shape_in_tiles,
                                       output->block_shape_in_tiles),
            "accumulation input must match output DFB");
    }
    if (kind == "gemm") {
      Check(accumulation == dtype,
            "GEMM accumulation dtype must match its materialized output");
      Check(ComputeString(call, "tt.input_dtype") ==
                    (operands[1]->element_dtype == DataType::BFloat(16)
                         ? "bfloat16"
                         : "float32") &&
                ComputeString(call, "tt.output_dtype") == dtype,
            "GEMM input/output dtype requirements disagree with DFBs");
      Check(dtype == "float32" ||
                operands[1]->element_dtype == DataType::BFloat(16),
            "BF16 GEMM accumulation requires BF16 inputs");
      Check(
          ComputeString(call, "tt.dest_precision_requirement") ==
              (accumulation == "float32" ? "bits32_required"
                                         : "bits16_required"),
          "GEMM hard precision requirement conflicts with accumulation dtype");
      if (accumulation == "bfloat16")
        Check(ComputeString(call, "tt.matmul_full_fp32") == "forbidden",
              "BF16 GEMM forbids matmul_full_fp32");
      Check(operands[1]->element_dtype == operands[2]->element_dtype,
            "GEMM input dtype mismatch");
      int64_t ta = ComputeInteger(call, "tt.transpose_a"),
              tb = ComputeInteger(call, "tt.transpose_b");
      Check((ta == 0 || ta == 1) && (tb == 0 || tb == 1),
            "GEMM transpose flags must be Boolean");
      auto a = operands[1]->block_shape_in_tiles,
           b = operands[2]->block_shape_in_tiles;
      auto c = output->block_shape_in_tiles;
      Check(a.size() >= 2 && a.size() == b.size() && a.size() == c.size(),
            "GEMM rank mismatch");
      size_t n = a.size();
      Check(ffi::StructuralEqual()(a[n - (ta ? 2 : 1)], b[n - (tb ? 1 : 2)]) &&
                ffi::StructuralEqual()(a[n - (ta ? 1 : 2)], c[n - 2]) &&
                ffi::StructuralEqual()(b[n - (tb ? 2 : 1)], c[n - 1]),
            "GEMM M/N/K tile-grid mismatch");
      for (size_t i = 0; i + 2 < n; ++i)
        Check(ffi::StructuralEqual()(a[i], b[i]) &&
                  ffi::StructuralEqual()(a[i], c[i]),
              "GEMM batch tile-grid mismatch");
    } else {
      int64_t axis = ComputeInteger(call, "tt.reduce_axis");
      auto input = input_shapes[0];
      Check(axis >= 0 && axis < static_cast<int64_t>(input.size()),
            "reduction axis out of range");
      ffi::String reduction = ComputeString(call, "tt.reduce_kind");
      Check(reduction == "sum" || reduction == "max" || reduction == "min",
            "unsupported reduction kind");
      int64_t nan = ComputeInteger(call, "tt.nan_propagate");
      Check(nan == 0 || nan == 1, "reduction nan propagation must be Boolean");
      ffi::Array<PrimExpr> expected;
      for (size_t i = 0; i < input.size(); ++i)
        if (static_cast<int64_t>(i) != axis)
          expected.push_back(input[i]);
        else if (shape.size() == input.size())
          expected.push_back(Integer(1));
      Check(ffi::StructuralEqual()(expected, shape),
            "reduction output logical shape mismatch");
    }
  }
}

// Reuse the expression/geometry verifier with temporary descriptor IDs. These
// descriptors are local proof objects, never storage allocations in Device IR.
using ComputeValueTable = std::map<int64_t, ComputeValueDescriptor>;

void VerifyValueExpression(const Call &call, const DFBTable &dfbs,
                           const ComputeValueTable &values,
                           const std::set<int64_t> &defined,
                           const BoundVars *bound_vars = nullptr,
                           int64_t core_x = -1, int64_t core_y = -1) {
  arith::Analyzer analyzer;
  auto static_integer = [&](const PrimExpr &value, const char *label) {
    PrimExpr bound = value;
    if (core_x >= 0 && core_y >= 0) {
      CoreCoordinateSubstituter bind_core(core_x, core_y);
      bound = bind_core.VisitExpr(bound);
    }
    return RequireStaticInteger(analyzer.Simplify(bound), label);
  };
  bool local_output = call->op.same_as(compute_value());
  ffi::String compute_kind = ComputeString(call, "tt.compute_kind");
  bool scalar_expression = compute_kind == "elementwise" ||
                           compute_kind == "fill" || compute_kind == "typecast";
  Check(!local_output || scalar_expression,
        "compute_value requires an elementwise, fill or typecast expression");
  Check(
      scalar_expression || !call->annotations.count("tt.expression"),
      "non-elementwise Device operation must not carry an ignored expression");
  Check(!call->args.empty(), "compute value missing output ID");
  // Proof descriptors cover this operation only. Materializing the entire
  // module's value table here makes verification quadratic in unrolled loops.
  DFBTable proof;
  std::set<int64_t> dfb_inputs;
  for (size_t i = local_output ? 1 : 0; i < call->args.size(); ++i) {
    int64_t id = static_integer(call->args[i], "compute DFB operand");
    Check(dfbs.count(id), "dfb_compute references missing DFB");
    proof.emplace(id, dfbs.at(id));
    if (i)
      dfb_inputs.insert(id);
  }
  std::set<int64_t> required_values;
  if (local_output)
    required_values.insert(static_integer(call->args[0], "compute output ID"));
  auto declared_inputs = call->annotations.Get("tt.value_inputs");
  if (declared_inputs.has_value()) {
    auto inputs = declared_inputs.value().as<ffi::Array<PrimExpr>>();
    Check(inputs.has_value(), "invalid tt.value_inputs");
    for (const PrimExpr &id : inputs.value())
      required_values.insert(static_integer(id, "compute value input ID"));
  }
  int64_t next = 0;
  for (const auto &[id, ignored] : proof) {
    Check(id < std::numeric_limits<int64_t>::max() -
                   static_cast<int64_t>(required_values.size()) - 2,
          "DFB ID overflows compute value verification");
    next = std::max(next, id + 1);
  }
  std::map<int64_t, int64_t> virtual_ids;
  CoreDomain domain(CoreCoord(0, 0), CoreCoord(1, 1));
  for (int64_t id : required_values) {
    Check(values.count(id), "compute value references missing descriptor");
    const auto &value = values.at(id);
    ffi::Array<PrimExpr> grid;
    for (const PrimExpr &extent : value->buffer->shape)
      grid.push_back(
          Integer((RequireStaticInteger(extent, "value extent") + 31) / 32));
    virtual_ids[id] = next;
    proof.emplace(next,
                  DFBDescriptor(next, "compute proof", value->buffer->dtype,
                                {Integer(32), Integer(32)}, grid, Integer(1),
                                std::nullopt, "trisc", domain, "trisc", domain,
                                Integer(1), value->source_span));
    ++next;
  }
  ffi::Array<PrimExpr> args = call->args;
  if (local_output) {
    int64_t id = static_integer(args[0], "compute output ID");
    Check(values.count(id), "compute value references missing descriptor");
    args.Set(0, Integer(virtual_ids.at(id)));
  }
  auto attrs = call->annotations;
  auto input_attr = attrs.Get("tt.value_inputs");
  auto maps_attr = attrs.Get("tt.value_access_maps");
  auto shapes_attr = attrs.Get("tt.value_input_shapes");
  if (!local_output && !input_attr.has_value() && !shapes_attr.has_value()) {
    VerifyGeneralCompute(call, dfbs, bound_vars);
    return;
  }
  Check(input_attr.has_value() && shapes_attr.has_value(),
        "v7 expression missing compute value input metadata");
  auto inputs = input_attr.value().as<ffi::Array<PrimExpr>>();
  Check(inputs.has_value(), "invalid tt.value_inputs");
  ffi::Array<ffi::Array<Integer>> maps =
      maps_attr.has_value() ? CheckedMaps(maps_attr.value())
                            : ffi::Array<ffi::Array<Integer>>();
  auto shapes = CheckedShapes(shapes_attr.value(), "tt.value_input_shapes");
  Check(inputs.value().size() == shapes.size() &&
            (!maps_attr.has_value() || maps.size() == shapes.size()),
        "compute value input metadata count mismatch");
  auto all_shapes =
      CheckedShapes(attrs.at("tt.input_shapes"), "tt.input_shapes");
  ffi::Array<ffi::Array<Integer>> all_maps;
  if (auto access_maps = attrs.Get("tt.access_maps"))
    all_maps = CheckedMaps(access_maps.value());
  std::set<int64_t> declared;
  for (size_t i = 0; i < inputs.value().size(); ++i) {
    int64_t id = static_integer(inputs.value()[i], "compute value input ID");
    Check(values.count(id) && defined.count(id),
          "compute value use is not dominated by its definition");
    declared.insert(id);
    VerifyTileGrid(shapes[i],
                   proof.at(virtual_ids.at(id))->block_shape_in_tiles,
                   "compute value input");
    args.push_back(Integer(virtual_ids.at(id)));
    if (maps_attr.has_value())
      all_maps.push_back(maps[i]);
    all_shapes.push_back(shapes[i]);
  }
  if (maps_attr.has_value())
    attrs.Set("tt.access_maps", all_maps);
  attrs.Set("tt.input_shapes", all_shapes);
  auto expression = attrs.Get("tt.expression");
  if (expression.has_value()) {
    class RewriteLoads : public ExprMutator {
    public:
      RewriteLoads(const ComputeValueTable &values,
                   const std::map<int64_t, int64_t> &ids,
                   const std::set<int64_t> &declared,
                   const std::set<int64_t> &dfb_inputs)
          : values_(values), ids_(ids), declared_(declared),
            dfb_inputs_(dfb_inputs) {}
      PrimExpr VisitExpr_(const CallNode *op) final {
        if (op->op.same_as(dfb_load())) {
          Check(op->args.size() == 1 &&
                    dfb_inputs_.count(RequireStaticInteger(
                        op->args[0], "dfb_load resource ID")),
                "dfb_load references an undeclared input DFB");
        }
        if (!op->op.same_as(compute_value_load()))
          return ExprMutator::VisitExpr_(op);
        Check(op->args.size() == 1 && op->annotations.empty(),
              "compute_value_load requires one ID and no annotations");
        int64_t id = RequireStaticInteger(op->args[0], "compute value load ID");
        Check(declared_.count(id), "undeclared compute value input");
        Check(op->dtype == values_.at(id)->buffer->dtype,
              "compute value load dtype mismatch");
        return Call(op->dtype, dfb_load(), {Integer(ids_.at(id))});
      }

    private:
      const ComputeValueTable &values_;
      const std::map<int64_t, int64_t> &ids_;
      const std::set<int64_t> &declared_;
      const std::set<int64_t> &dfb_inputs_;
    } rewrite(values, virtual_ids, declared, dfb_inputs);
    auto expr = expression.value().as<PrimExpr>();
    Check(expr.has_value(), "invalid compute value expression");
    attrs.Set("tt.expression", rewrite(expr.value()));
  }
  VerifyGeneralCompute(
      Call(call->dtype, dfb_compute(), args, attrs, call->span), proof,
      bound_vars);
}

class ComputeValueVerifier {
public:
  explicit ComputeValueVerifier(const IRModule &mod) {
    current_device_ =
        mod->GetAttr<ffi::String>("tt.ir_stage").value_or("") == "device";
    int64_t version =
        mod->GetAttr<Integer>(kDeviceIRVersionAttr).value_or(Integer(0))->value;
    precision_regions_ = version == 8;
    if (!current_device_ && version != 7 && version != 8)
      return;
    if (current_device_ && !mod->attrs->dict.count(kComputeValueTableAttr))
      return;
    enabled_ = true;
    if (current_device_ && mod->attrs->dict.count(kPipelineRelationsAttr)) {
      auto relations =
          RequireModuleAttr<ffi::Map<ffi::String, ffi::Array<Integer>>>(
              mod, kPipelineRelationsAttr);
      std::unordered_map<std::string, int64_t> canonical_ids;
      for (const DFBDescriptor &dfb :
           RequireModuleAttr<ffi::Array<DFBDescriptor>>(mod, kDFBTableAttr))
        canonical_ids.emplace(std::to_string(dfb->dfb_id), dfb->dfb_id);
      for (const auto &[key, relation] : relations) {
        Check(relation.size() == 2,
              "current Device pipeline relation must be [iteration, stage]");
        auto id = canonical_ids.find(std::string(key));
        Check(id != canonical_ids.end(),
              "current Device pipeline relation requires a canonical decimal "
              "DFB key");
        pipeline_iterations_.emplace(id->second, relation[0]->value);
      }
    }
    auto entries = RequireModuleAttr<ffi::Array<ComputeValueDescriptor>>(
        mod, kComputeValueTableAttr);
    Check(!entries.empty(), "v7 requires a nonempty compute value table");
    using ValueInstanceKey = std::tuple<const VarNode *, int64_t, int64_t>;
    std::map<ValueInstanceKey, ComputeValueDescriptor> previous;
    for (const ComputeValueDescriptor &entry : entries) {
      Check(entry.defined() && entry->value_id >= 0 &&
                entry->buffer.defined() &&
                (current_device_ || entry->source_span.defined()),
            "invalid compute value descriptor");
      Check(values_.emplace(entry->value_id, entry).second,
            "duplicate compute value ID");
      Check(entry->buffer.scope() == "local.fragment" &&
                SupportedComputeDType(entry->buffer->dtype) &&
                entry->buffer->shape.size() == 2,
            "compute value requires a rank-2 BF16/FP32 fragment");
      for (const PrimExpr &extent : entry->buffer->shape) {
        int64_t n = RequireStaticInteger(extent, "compute value shape");
        Check(n > 0 && (n == 1 || n % 32 == 0),
              "compute value shape must be positive and tile-aligned or "
              "singleton");
      }
      int64_t core_x = current_device_ ? entry->core_domain->begin->x : -1;
      int64_t core_y = current_device_ ? entry->core_domain->begin->y : -1;
      ValueInstanceKey instance{entry->buffer->data.get(), core_x, core_y};
      auto old = previous.find(instance);
      Check(old == previous.end()
                ? entry->version == 0 && entry->previous_value_id == -1
                : entry->buffer.same_as(old->second->buffer) &&
                      entry->version == old->second->version + 1 &&
                      entry->previous_value_id == old->second->value_id,
            "compute value Buffer/version predecessor chain is invalid");
      previous[instance] = entry;
      Check(entry->accumulator_id >= -1,
            "invalid compute value accumulator ID");
    }
    auto acc =
        mod->GetAttr<ffi::Array<AccumulatorDescriptor>>(kAccumulatorTableAttr);
    if (acc.has_value())
      for (const AccumulatorDescriptor &entry : acc.value()) {
        Check(entry.defined() && entry->accumulator_id >= 0 &&
                  accumulators_.emplace(entry->accumulator_id, entry).second,
              "invalid or duplicate v7 accumulator descriptor");
        Check(entry->accumulator_region.defined() &&
                  entry->accumulator_region->buffer.defined() &&
                  entry->accumulator_region->buffer.scope() ==
                      "local.fragment" &&
                  entry->accumulator_region->buffer->dtype ==
                      entry->accumulation_dtype &&
                  entry->full_k_tiles > 0 &&
                  (current_device_ || entry->source_span.defined()) &&
                  IsSupportedAccumulatorDTypeTriple(entry->input_dtype,
                                                    entry->accumulation_dtype,
                                                    entry->output_dtype),
              "invalid v7 accumulator precision/lifetime descriptor");
        const auto &region = entry->accumulator_region;
        Check(region->region.size() == 2 && region->buffer->shape.size() == 2,
              "v7 accumulator region must have rank two");
        for (size_t axis = 0; axis < 2; ++axis) {
          int64_t minimum = RequireStaticInteger(region->region[axis]->min,
                                                 "accumulator region minimum");
          int64_t extent = RequireStaticInteger(region->region[axis]->extent,
                                                "accumulator region extent");
          Check(
              minimum == 0 && extent > 0 && extent % 32 == 0 &&
                  extent == RequireStaticInteger(region->buffer->shape[axis],
                                                 "accumulator buffer extent"),
              "v7 accumulator region must cover its full tile-aligned Buffer");
        }
      }
    for (const auto &[id, entry] : values_)
      Check(entry->accumulator_id == -1 ||
                accumulators_.count(entry->accumulator_id),
            "compute value references missing accumulator descriptor");
    if (!current_device_) {
      // Legacy specialized functions bind ownership directly from their
      // concrete value definitions. Current shared functions bind it while
      // enumerating each family instance in EnterCoreInstance.
      for (const auto &[global, base] : mod->functions) {
        PrimFunc function = Downcast<PrimFunc>(base);
        PostOrderVisit(function->body, [&](const ffi::ObjectRef &object) {
          const auto *call = object.as<CallNode>();
          if (!call || !(call->op.same_as(compute_value()) ||
                         call->op.same_as(compute_value_gemm())))
            return;
          Check(!call->args.empty(), "compute value missing output ID");
          int64_t id = RequireStaticInteger(call->args[0], "compute value ID");
          Check(values_.count(id),
                "compute value definition missing descriptor");
          Check(owners_.emplace(id, function).second,
                "compute value definition missing or duplicated");
          Check(function->GetAttr<ffi::String>(kKernelSlotAttr).value_or("") ==
                    "trisc",
                "compute values cannot cross processor slots");
          int64_t accumulator = values_.at(id)->accumulator_id;
          if (accumulator >= 0) {
            auto [owner, inserted] =
                accumulator_owners_.emplace(accumulator, function);
            Check(inserted || owner->second.same_as(function),
                  "compute value accumulator crosses Core ownership");
          }
        });
      }
      for (const auto &[id, entry] : values_) {
        if (entry->previous_value_id >= 0 && owners_.count(id) &&
            owners_.count(entry->previous_value_id))
          Check(owners_.at(id).same_as(owners_.at(entry->previous_value_id)),
                "compute value predecessor crosses Core ownership");
      }
    }
  }
  void BindFunction(const PrimFunc &function) {
    function_ = function;
    precision_epoch_ = 0;
    bound_loop_vars_.clear();
    PostOrderVisit(function->body, [&](const ffi::ObjectRef &object) {
      if (const auto *loop = object.as<ForNode>())
        bound_loop_vars_.insert(loop->loop_var.get());
    });
  }
  void EnterCoreInstance(int64_t x, int64_t y) {
    precision_epoch_ = 0;
    stored_values_.clear();
    dfb_roots_.clear();
    active_dfb_transactions_.clear();
    released_dfb_transactions_.clear();
    current_core_x_ = x;
    current_core_y_ = y;
    for (const auto &[id, value] : values_) {
      const CoreDomain &domain = value->core_domain;
      if (x >= domain->begin->x && x < domain->end->x &&
          y >= domain->begin->y && y < domain->end->y)
        owners_[id] = function_;
    }
    for (const auto &[id, accumulator] : accumulators_) {
      const CoreDomain &domain = accumulator->core_domain;
      if (x >= domain->begin->x && x < domain->end->x &&
          y >= domain->begin->y && y < domain->end->y)
        accumulator_owners_[id] = function_;
    }
  }
  void Precision(const Call &call) {
    Check(
        precision_regions_ && call->args.size() == 1 &&
            call->annotations.empty() &&
            function_->GetAttr<ffi::String>(kKernelSlotAttr).value_or("") ==
                "trisc",
        "compute_precision requires v8 TRISC and one mode without annotations");
    int64_t mode =
        RequireStaticInteger(call->args[0], "compute precision mode");
    Check(mode == 0 || mode == 1, "compute precision mode must be 0 or 1");
    ++precision_epoch_;
  }
  void CheckOwner(int64_t id) const {
    Check(owners_.count(id) && owners_.at(id).same_as(function_),
          "compute value " + std::to_string(id) +
              " crosses Core or processor ownership");
    if (current_device_) {
      CoreDomain domain = values_.at(id)->core_domain;
      Check(current_core_x_ >= domain->begin->x &&
                current_core_x_ < domain->end->x &&
                current_core_y_ >= domain->begin->y &&
                current_core_y_ < domain->end->y,
            "compute value " + std::to_string(id) +
                " crosses Core or processor ownership");
    }
  }
  bool Enabled() const { return enabled_; }
  const ComputeValueTable &Values() const { return values_; }
  const ComputeValueDescriptor &Read(int64_t id) const {
    Check(values_.count(id) && defined_.count(id),
          "compute value use is not dominated by its definition");
    CheckOwner(id);
    Check(!precision_regions_ || value_epochs_.at(id) == precision_epoch_,
          "compute value crosses precision region without exact-dtype DFB "
          "snapshot");
    for (const DFBTransaction &source : source_transactions_.at(id))
      Check(!released_dfb_transactions_.count(source),
            "compute value reads DFB provenance after release");
    return values_.at(id);
  }
  void AcquireDFB(int64_t id, int64_t transaction) {
    active_dfb_transactions_[id] = transaction;
  }
  void ReleaseDFB(int64_t id, int64_t transaction) {
    released_dfb_transactions_.insert({id, transaction});
    auto active = active_dfb_transactions_.find(id);
    if (active != active_dfb_transactions_.end() &&
        active->second == transaction)
      active_dfb_transactions_.erase(active);
  }
  std::set<int64_t> Expression(const Call &call, const DFBTable &dfbs) {
    arith::Analyzer analyzer;
    auto static_integer = [&](const PrimExpr &value, const char *label) {
      CoreCoordinateSubstituter bind_core(current_core_x_, current_core_y_);
      return RequireStaticInteger(analyzer.Simplify(bind_core.VisitExpr(value)),
                                  label);
    };
    std::set<int64_t> sources;
    std::set<DFBTransaction> source_transactions;
    for (size_t i = 1; i < call->args.size(); ++i)
      sources.insert(static_integer(call->args[i], "expression DFB input"));
    auto value_inputs = call->annotations.Get("tt.value_inputs");
    if (value_inputs.has_value()) {
      auto ids = value_inputs.value().as<ffi::Array<PrimExpr>>();
      Check(ids.has_value(), "invalid tt.value_inputs");
      for (const PrimExpr &id : ids.value())
        CheckOwner(RequireStaticInteger(arith::Analyzer().Simplify(id),
                                        "compute value input ID"));
    }
    VerifyValueExpression(call, dfbs, values_, defined_, &bound_loop_vars_,
                          current_core_x_, current_core_y_);
    std::set<int64_t> roots;
    std::map<int64_t, int64_t> completed_k;
    auto inputs = call->annotations.Get("tt.value_inputs");
    if (inputs.has_value()) {
      for (const PrimExpr &id_expr :
           Downcast<ffi::Array<PrimExpr>>(inputs.value())) {
        int64_t id = RequireStaticInteger(arith::Analyzer().Simplify(id_expr),
                                          "compute value input ID");
        Read(id);
        const auto &borrowed = sources_.at(id);
        sources.insert(borrowed.begin(), borrowed.end());
        const auto &borrowed_transactions = source_transactions_.at(id);
        source_transactions.insert(borrowed_transactions.begin(),
                                   borrowed_transactions.end());
        for (const auto &[root, count] : value_k_.at(id)) {
          roots.insert(root);
          auto [it, inserted] = completed_k.emplace(root, count);
          if (!inserted)
            it->second = std::min(it->second, count);
          if (!call->op.same_as(compute_value()))
            Check(count == accumulators_.at(root)->full_k_tiles,
                  "GEMM output materialization precedes complete K reduction");
        }
      }
    }
    for (size_t i = 1; i < call->args.size(); ++i) {
      int64_t input = static_integer(call->args[i], "DFB provenance input");
      Check(active_dfb_transactions_.count(input),
            "compute value reads DFB without an active acquisition");
      source_transactions.insert({input, active_dfb_transactions_.at(input)});
      auto found = dfb_roots_.find(input);
      if (found != dfb_roots_.end()) {
        for (int64_t root : found->second) {
          roots.insert(root);
          completed_k.emplace(root, accumulators_.at(root)->full_k_tiles);
        }
      }
    }
    if (call->op.same_as(compute_value())) {
      int64_t id = static_integer(call->args[0], "compute value output");
      const auto &entry = values_.at(id);
      if (roots.empty() && entry->accumulator_id >= 0) {
        auto acc = accumulators_.at(entry->accumulator_id);
        auto expression = call->annotations.Get("tt.expression");
        Check(entry->buffer.same_as(acc->accumulator_region->buffer) &&
                  ComputeString(call, "tt.compute_kind") == "fill" &&
                  expression.has_value() &&
                  (is_zero(Downcast<PrimExpr>(expression.value())) ||
                   (expression.value().as<FloatImmNode>() &&
                    expression.value().as<FloatImmNode>()->value == 0)),
              "accumulator initial value must be a unique zero definition");
        Check(initialized_.insert(entry->accumulator_id).second,
              "accumulator has multiple initial definitions");
        completed_k[entry->accumulator_id] = 0;
      } else {
        bool owns_accumulator =
            entry->accumulator_id >= 0 && roots.count(entry->accumulator_id) &&
            entry->buffer.same_as(accumulators_.at(entry->accumulator_id)
                                      ->accumulator_region->buffer);
        Check(entry->accumulator_id == -1 || owns_accumulator ||
                  entry->accumulator_id ==
                      (roots.size() == 1 ? *roots.begin() : -1),
              "compute value accumulator provenance disagrees with inputs");
      }
      if (entry->previous_value_id >= 0) {
        auto previous = values_.at(entry->previous_value_id);
        bool previous_owner =
            previous->accumulator_id >= 0 &&
            previous->buffer.same_as(accumulators_.at(previous->accumulator_id)
                                         ->accumulator_region->buffer);
        if (previous_owner &&
            (entry->accumulator_id == -1 ||
             entry->accumulator_id == previous->accumulator_id)) {
          bool reads_previous = false;
          if (inputs.has_value())
            for (const PrimExpr &input :
                 Downcast<ffi::Array<PrimExpr>>(inputs.value()))
              reads_previous |=
                  RequireStaticInteger(arith::Analyzer().Simplify(input),
                                       "compute value input ID") ==
                  entry->previous_value_id;
          // Reentry is an identity load of the exact preceding version's
          // completed snapshot, never an untyped relabeling across DST modes.
          if (precision_regions_ && call->args.size() == 2) {
            int64_t source = static_integer(call->args[1], "reentry DFB");
            auto expression = call->annotations.Get("tt.expression");
            const auto *load = expression.has_value()
                                   ? expression.value().as<CallNode>()
                                   : nullptr;
            reads_previous |=
                stored_values_.count(source) &&
                stored_values_.at(source) == entry->previous_value_id && load &&
                load->op.same_as(dfb_load()) && load->args.size() == 1 &&
                static_integer(load->args[0], "reentry DFB") == source;
          }
          Check(
              reads_previous,
              "accumulator elementwise update must read its previous version");
        }
      }
      value_k_[id] = completed_k;
      sources_[id] = sources;
      source_transactions_[id] = source_transactions;
      Define(id);
    }
    if (!call->op.same_as(compute_value())) {
      int64_t output = static_integer(call->args[0], "DFB provenance output");
      auto occurrence = pipeline_iterations_.find(output);
      if (occurrence != pipeline_iterations_.end() && occurrence->second >= 0)
        for (int64_t source : sources) {
          auto input = pipeline_iterations_.find(source);
          Check(input == pipeline_iterations_.end() || input->second == -1 ||
                    input->second == occurrence->second,
                "compute value crosses pipeline iterations after release");
        }
      dfb_roots_[output] = roots;
    }
    return sources;
  }
  void Define(int64_t id) {
    CheckOwner(id);
    Check(!precision_regions_ || precision_epoch_ > 0,
          "v8 computation requires an explicit precision region");
    Check(values_.count(id) && defined_.insert(id).second,
          "compute value definition missing or duplicated");
    const auto &entry = values_.at(id);
    value_epochs_[id] = precision_epoch_;
    Check(entry->previous_value_id == -1 ||
              defined_.count(entry->previous_value_id),
          "compute value predecessor is not defined");
  }
  std::set<int64_t> Gemm(const Call &call, const DFBTable &dfbs) {
    Check(call->args.size() == 6 && call->annotations.empty(),
          "compute_value_gemm requires six IDs/flags without annotations");
    arith::Analyzer analyzer;
    auto integer = [&](size_t i) {
      return RequireStaticInteger(analyzer.Simplify(call->args[i]),
                                  "value GEMM operand");
    };
    int64_t out = integer(0), lhs = integer(1), rhs = integer(2),
            old = integer(3);
    Check(values_.count(out) && dfbs.count(lhs) && dfbs.count(rhs),
          "value GEMM references missing operand");
    auto value = values_.at(out);
    Check(accumulators_.count(value->accumulator_id),
          "value GEMM has no accumulator provenance");
    const auto &acc = accumulators_.at(value->accumulator_id);
    Check(value->buffer.same_as(acc->accumulator_region->buffer) &&
              value->buffer->dtype == acc->accumulation_dtype,
          "value GEMM accumulator identity or dtype mismatch");
    Check(old >= 0 && Read(old)->buffer.same_as(value->buffer) &&
              value->previous_value_id == old &&
              Read(old)->accumulator_id == value->accumulator_id,
          "value GEMM update must consume its previous accumulator version");
    auto a = dfbs.at(lhs), b = dfbs.at(rhs);
    Check(a->element_dtype == acc->input_dtype &&
              b->element_dtype == acc->input_dtype,
          "value GEMM input dtype mismatch");
    int64_t ta = integer(4), tb = integer(5);
    Check((ta == 0 || ta == 1) && (tb == 0 || tb == 1) &&
              a->block_shape_in_tiles.size() == 2 &&
              b->block_shape_in_tiles.size() == 2,
          "value GEMM transpose/rank mismatch");
    auto size = [&](const PrimExpr &e) {
      return RequireStaticInteger(e, "value GEMM shape");
    };
    int64_t k = size(a->block_shape_in_tiles[ta ? 0 : 1]);
    Check(k == size(b->block_shape_in_tiles[tb ? 1 : 0]) &&
              size(a->block_shape_in_tiles[ta ? 1 : 0]) * 32 ==
                  size(value->buffer->shape[0]) &&
              size(b->block_shape_in_tiles[tb ? 0 : 1]) * 32 ==
                  size(value->buffer->shape[1]),
          "value GEMM M/N/K mismatch");
    Check(k > 0 && k_tiles_[value->accumulator_id] <= acc->full_k_tiles - k,
          "value GEMM updates exceed full K");
    Check(value_k_.at(old).count(value->accumulator_id) &&
              value_k_.at(old).at(value->accumulator_id) ==
                  k_tiles_[value->accumulator_id],
          "value GEMM uses a stale K accumulator version");
    k_tiles_[value->accumulator_id] += k;
    value_k_[out] = value_k_.at(old);
    value_k_[out][value->accumulator_id] = k_tiles_[value->accumulator_id];
    sources_[out] = sources_.at(old);
    sources_[out].insert(lhs);
    sources_[out].insert(rhs);
    source_transactions_[out] = source_transactions_.at(old);
    for (int64_t input : {lhs, rhs}) {
      Check(active_dfb_transactions_.count(input),
            "value GEMM reads DFB without an active acquisition");
      source_transactions_[out].insert(
          {input, active_dfb_transactions_.at(input)});
    }
    Define(out);
    return sources_.at(out);
  }
  std::set<int64_t> Store(const Call &call, const DFBTable &dfbs) {
    Check(
        call->args.size() == 2 && call->annotations.empty(),
        "compute_value_store requires a value and DFB ID without annotations");
    arith::Analyzer analyzer;
    auto value = Read(RequireStaticInteger(analyzer.Simplify(call->args[0]),
                                           "stored compute value"));
    int64_t id = RequireStaticInteger(analyzer.Simplify(call->args[1]),
                                      "materialization DFB");
    Check(dfbs.count(id), "compute value store references missing DFB");
    Check(value->buffer->dtype == dfbs.at(id)->element_dtype,
          "compute value materialization must preserve dtype");
    VerifyTileGrid(value->buffer->shape, dfbs.at(id)->block_shape_in_tiles,
                   "compute value store");
    stored_values_[id] = value->value_id;
    for (const auto &[root, count] : value_k_.at(value->value_id)) {
      Check(count == accumulators_.at(root)->full_k_tiles,
            "GEMM materialization precedes complete K reduction");
      dfb_roots_[id].insert(root);
    }
    return sources_.at(value->value_id);
  }
  void Export(int64_t id, DataType dtype) const {
    auto found = dfb_roots_.find(id);
    if (found == dfb_roots_.end())
      return;
    for (int64_t root : found->second)
      Check(accumulators_.at(root)->output_dtype == dtype,
            "GEMM final output dtype disagrees with accumulator capability "
            "registry");
  }
  void Finish() const {
    if (!enabled_)
      return;
    Check(defined_.size() == values_.size(),
          "unused or undefined compute value descriptor");
    for (const auto &[id, acc] : accumulators_) {
      auto it = k_tiles_.find(id);
      Check(it != k_tiles_.end() && it->second == acc->full_k_tiles,
            "value GEMM lifetime does not contain complete full-K updates");
    }
  }

private:
  using DFBTransaction = std::pair<int64_t, int64_t>;

  bool enabled_{false};
  bool current_device_{false};
  int64_t current_core_x_{-1};
  int64_t current_core_y_{-1};
  bool precision_regions_{false};
  int64_t precision_epoch_{0};
  std::map<int64_t, int64_t> value_epochs_;
  std::map<int64_t, int64_t> stored_values_;
  PrimFunc function_;
  std::map<int64_t, PrimFunc> owners_;
  std::map<int64_t, PrimFunc> accumulator_owners_;
  ComputeValueTable values_;
  std::map<int64_t, AccumulatorDescriptor> accumulators_;
  std::map<int64_t, int64_t> k_tiles_;
  std::set<int64_t> defined_;
  std::set<int64_t> initialized_;
  std::map<int64_t, std::map<int64_t, int64_t>> value_k_;
  std::map<int64_t, std::set<int64_t>> sources_;
  std::map<int64_t, std::set<DFBTransaction>> source_transactions_;
  std::map<int64_t, std::set<int64_t>> dfb_roots_;
  std::map<int64_t, int64_t> active_dfb_transactions_;
  std::set<DFBTransaction> released_dfb_transactions_;
  std::map<int64_t, int64_t> pipeline_iterations_;
  BoundVars bound_loop_vars_;
};

// Schema v3 preserves immutable DFB generations while assigning generations
// from the same lexical write to a bounded storage pool.  These identities are
// intentionally independent: waiting publishes readiness, releasing returns
// storage, and each ordinal defines an iteration and logical window stage.
struct PipelineResources {
  bool enabled{false};
  int64_t depth{0};
  std::unordered_map<int64_t, int64_t> groups;
  std::unordered_map<int64_t, int64_t> ordinals;
  std::map<int64_t, std::vector<int64_t>> ordered;
  std::set<int64_t> singleton_groups;
};

int64_t CheckedProduct(int64_t left, int64_t right, const std::string &label) {
  Check(left > 0 && right > 0 &&
            left <= std::numeric_limits<int64_t>::max() / right,
        label + " overflows static byte/capacity arithmetic");
  return left * right;
}

void VerifyLegacyL1Budget(const IRModule &mod, const DFBTable &dfbs) {
  auto budget = mod->GetAttr<Integer>(kL1CapacityBytesAttr);
  if (!budget.has_value())
    return;
  Check(budget.value()->value > 0, "tt.l1_capacity_bytes must be positive");
  int64_t payload = 0;
  for (const auto &[id, dfb] : dfbs) {
    const std::string label = "DFB " + std::to_string(id) + " L1 payload";
    int64_t bytes = RequireStaticInteger(dfb->block_count, label);
    for (const PrimExpr &extent : dfb->tile_shape)
      bytes = CheckedProduct(bytes, RequireStaticInteger(extent, label), label);
    for (const PrimExpr &extent : dfb->block_shape_in_tiles)
      bytes = CheckedProduct(bytes, RequireStaticInteger(extent, label), label);
    bytes = CheckedProduct(bytes, dfb->element_dtype.bytes(), label);
    Check(payload <= std::numeric_limits<int64_t>::max() - bytes,
          "L1 payload byte sum overflows static arithmetic");
    payload += bytes;
  }
  Check(payload <= budget.value()->value,
        "L1 logical payload lower bound " + std::to_string(payload) +
            " bytes exceeds explicit tt.l1_capacity_bytes budget " +
            std::to_string(budget.value()->value) +
            "; physical allocation overhead is not included");
}

PipelineResources VerifyPipelineResources(const IRModule &mod,
                                          const DFBTable &dfbs) {
  PipelineResources result;
  result.enabled =
      RequireModuleAttr<Integer>(mod, kDeviceIRVersionAttr)->value == 3 ||
      mod->attrs->dict.count(kPipelineStagesAttr) ||
      mod->attrs->dict.count(kPipelineExtentAttr) ||
      mod->attrs->dict.count(kDFBStorageGroupsAttr) ||
      mod->attrs->dict.count(kPipelineRelationsAttr) ||
      mod->attrs->dict.count("tt.pipeline_wait_policy");
  if (!result.enabled)
    return result;
  Check(RequireModuleAttr<Integer>(mod, kDeviceIRVersionAttr)->value >= 3,
        "pipeline metadata requires Device IR schema v3 or newer");
  int64_t stages = RequireModuleAttr<Integer>(mod, kPipelineStagesAttr)->value;
  int64_t extent = RequireModuleAttr<Integer>(mod, kPipelineExtentAttr)->value;
  Check(stages > 0 && stages <= 32 && extent > 0 && extent <= 1024,
        "pipeline stages must be in [1,32] and extent in [1,1024]");
  result.depth = std::min(stages, extent);
  auto wait_policy = mod->GetAttr<ffi::String>("tt.pipeline_wait_policy");
  if (wait_policy.has_value())
    Check(wait_policy.value() == "conservative" ||
              wait_policy.value() == "delayed",
          "tt.pipeline_wait_policy must be conservative or delayed");
  auto groups = RequireModuleAttr<ffi::Map<ffi::String, Integer>>(
      mod, kDFBStorageGroupsAttr);
  auto relations =
      RequireModuleAttr<ffi::Map<ffi::String, ffi::Array<Integer>>>(
          mod, kPipelineRelationsAttr);
  Check(groups.size() == dfbs.size() && relations.size() == dfbs.size(),
        "pipeline resource metadata must cover exactly the DFB table");
  std::unordered_map<std::string, int64_t> keyed_dfbs;
  for (const auto &[id, dfb] : dfbs)
    keyed_dfbs.emplace(std::to_string(id), id);
  for (const auto &[key, group] : groups) {
    Check(keyed_dfbs.count(key),
          "pipeline storage group requires a canonical decimal DFB key");
    int64_t id = keyed_dfbs.at(key);
    Check(group->value >= 0,
          "pipeline storage group references invalid DFB/pool ID");
    Check(result.groups.emplace(id, group->value).second,
          "duplicate pipeline storage group DFB ID");
    if (!result.ordered.count(group->value))
      result.ordered.emplace(group->value, std::vector<int64_t>(extent, -1));
  }
  for (const auto &[key, relation] : relations) {
    Check(keyed_dfbs.count(key),
          "pipeline transaction relation requires a canonical decimal DFB key");
    int64_t id = keyed_dfbs.at(key);
    Check(relation.size() == 2,
          "pipeline transaction relation must be [iteration, stage]");
    int64_t ordinal = relation[0]->value;
    Check((ordinal == -1 && relation[1]->value == 0) ||
              (ordinal >= 0 && ordinal < extent &&
               relation[1]->value == ordinal % result.depth),
          "pipeline transaction relation has invalid iteration/stage");
    Check(result.ordinals.emplace(id, ordinal).second,
          "duplicate pipeline transaction relation DFB ID");
    int64_t group = result.groups.at(id);
    auto &ordered = result.ordered.at(group);
    Check(!result.singleton_groups.count(group),
          "pipeline singleton storage pool must contain exactly one resource");
    if (ordinal == -1) {
      Check(std::all_of(ordered.begin(), ordered.end(),
                        [](int64_t entry) { return entry == -1; }),
            "pipeline storage pool mixes singleton and iterative resources");
      result.singleton_groups.insert(group);
      ordered = {id};
    } else {
      Check(ordered[ordinal] == -1,
            "pipeline storage pool has duplicate iteration");
      ordered[ordinal] = id;
    }
  }
  int64_t payload_bytes = 0;
  std::map<std::pair<int64_t, int64_t>, int64_t> core_payload;
  for (const auto &[group, ids] : result.ordered) {
    const std::string label = "pipeline storage pool " + std::to_string(group);
    Check(
        std::all_of(ids.begin(), ids.end(), [](int64_t id) { return id >= 0; }),
        label + " must have exactly one generation per iteration");
    DFBDescriptor first = dfbs.at(ids.front());
    int64_t capacity =
        RequireStaticInteger(first->block_count, label + " block_count");
    int64_t required = result.singleton_groups.count(group) ? 1 : result.depth;
    Check(capacity >= required && capacity <= 32,
          label + " capacity insufficient for pipeline window: requires " +
              std::to_string(required) + ", has " + std::to_string(capacity));
    for (int64_t id : ids) {
      DFBDescriptor dfb = dfbs.at(id);
      Check(ffi::StructuralEqual()(first->block_count, dfb->block_count) &&
                first->element_dtype == dfb->element_dtype &&
                ffi::StructuralEqual()(first->tile_shape, dfb->tile_shape) &&
                ffi::StructuralEqual()(first->block_shape_in_tiles,
                                       dfb->block_shape_in_tiles) &&
                first->producer_slot == dfb->producer_slot &&
                first->consumer_slot == dfb->consumer_slot &&
                ffi::StructuralEqual()(first->producer_domain,
                                       dfb->producer_domain) &&
                ffi::StructuralEqual()(first->consumer_domain,
                                       dfb->consumer_domain) &&
                ffi::StructuralEqual()(first->tensor_backing,
                                       dfb->tensor_backing),
            label + " generations disagree on capacity, layout, or slots");
    }
    int64_t bytes = capacity;
    for (const PrimExpr &extent : first->tile_shape)
      bytes = CheckedProduct(bytes, RequireStaticInteger(extent, label), label);
    for (const PrimExpr &extent : first->block_shape_in_tiles)
      bytes = CheckedProduct(bytes, RequireStaticInteger(extent, label), label);
    bytes = CheckedProduct(bytes, first->element_dtype.bytes(), label);
    Check(payload_bytes <= std::numeric_limits<int64_t>::max() - bytes,
          "L1 payload byte sum overflows static arithmetic");
    payload_bytes += bytes;
    auto owner = std::make_pair(first->producer_domain->begin->x,
                                first->producer_domain->begin->y);
    Check(core_payload[owner] <= std::numeric_limits<int64_t>::max() - bytes,
          "Core L1 payload byte sum overflows static arithmetic");
    core_payload[owner] += bytes;
  }
  // This is logical payload storage, counted once per pool.  It does not
  // estimate alignment, TT-Lang scratch, firmware reservations, or free L1.
  auto budget = mod->GetAttr<Integer>(kL1CapacityBytesAttr);
  if (budget.has_value()) {
    Check(budget.value()->value > 0, "tt.l1_capacity_bytes must be positive");
    for (const auto &[core, bytes] : core_payload)
      Check(bytes <= budget.value()->value,
            "Core (" + std::to_string(core.first) + "," +
                std::to_string(core.second) +
                ") L1 logical payload lower bound " + std::to_string(bytes) +
                " bytes exceeds explicit tt.l1_capacity_bytes budget " +
                std::to_string(budget.value()->value) +
                "; physical allocation overhead is not included");
  }
  auto reported = mod->GetAttr<Integer>("tt.l1_payload_bytes");
  if (reported.has_value())
    Check(reported.value()->value == payload_bytes,
          "tt.l1_payload_bytes disagrees with static storage-pool payload");
  return result;
}

void VerifyGeneralProgram(const IRModule &mod, const TensorTable &tensors,
                          const DFBTable &dfbs) {
  PipelineResources pipeline = VerifyPipelineResources(mod, dfbs);
  ComputeValueVerifier values(mod);
  bool value_program = values.Enabled();
  bool accumulators =
      RequireModuleAttr<Integer>(mod, kDeviceIRVersionAttr)->value == 5;
  std::unordered_map<int64_t, AccumulatorDescriptor> accumulator_table;
  if (accumulators) {
    for (const auto &entry :
         RequireModuleAttr<ffi::Array<AccumulatorDescriptor>>(
             mod, kAccumulatorTableAttr))
      accumulator_table.emplace(entry->accumulator_id, entry);
  }
  // Verify an independent structured iteration by induction: every active
  // slot has the same loop boundary, each resource is defined and fully used
  // in one iteration, and no DFB value escapes that iteration. Tensor inout
  // dependencies remain ordered in the NCRISC stream.
  bool has_loop = false;
  for (const auto &[global, base] : mod->functions)
    has_loop |= Downcast<PrimFunc>(base)->body.as<ForNode>() != nullptr;
  if (has_loop) {
    Check(!accumulators && !value_program,
          "schema v5/v7 requires statically expanded serial K updates");
    Check(!pipeline.enabled,
          "pipeline Device IR must contain explicit scheduled operations");
    ffi::Optional<For> boundary;
    IRModule iteration = mod;
    iteration.CopyOnWrite();
    for (const auto &[global, base] : mod->functions) {
      PrimFunc func = Downcast<PrimFunc>(base);
      if (IsCanonicalNoOp(func->body))
        continue;
      auto loop_value = func->body.as<For>();
      Check(loop_value.has_value(),
            "structured iteration must enclose every active slot");
      For loop = loop_value.value();
      Check(loop->kind == ForKind::kSerial && loop->annotations.empty() &&
                !loop->thread_binding.has_value() &&
                (!loop->step.has_value() || is_one(loop->step.value())),
            "Device structured iteration requires an unannotated unit-step "
            "serial loop");
      int64_t extent =
          RequireStaticInteger(loop->extent, "structured loop extent");
      RequireStaticInteger(loop->min, "structured loop minimum");
      Check(extent > 0 && extent <= 1024,
            "structured loop extent must be in [1,1024]");
      Check(!UsesVar(loop->body,
                     [&](const VarNode *var) {
                       return ffi::GetRef<Var>(var).same_as(loop->loop_var);
                     }),
            "structured iteration body must be independent of its loop index");
      if (boundary.has_value()) {
        Check(
            ffi::StructuralEqual()(boundary.value()->min, loop->min) &&
                ffi::StructuralEqual()(boundary.value()->extent, loop->extent),
            "structured slot loop boundaries disagree");
      } else {
        boundary = loop;
      }
      func.CopyOnWrite()->body = loop->body;
      iteration->Update(global, func);
    }
    DFBTable local;
    for (const auto &[id, dfb] : dfbs) {
      Check(ffi::StructuralEqual()(dfb->transaction_count_or_loop_relation,
                                   boundary.value()->extent),
            "DFB transaction relation disagrees with structured loop extent");
      local.emplace(id, DFBDescriptor(dfb->dfb_id, dfb->source_buffer_identity,
                                      dfb->element_dtype, dfb->tile_shape,
                                      dfb->block_shape_in_tiles,
                                      dfb->block_count, dfb->tensor_backing,
                                      dfb->producer_slot, dfb->producer_domain,
                                      dfb->consumer_slot, dfb->consumer_domain,
                                      Integer(1), dfb->source_span));
    }
    VerifyGeneralProgram(iteration, tensors, local);
    return;
  }
  struct Event {
    Call call;
    std::string slot;
  };
  std::vector<Event> events;
  std::vector<std::vector<size_t>> dependencies;
  std::unordered_map<int64_t, size_t> producers;
  std::unordered_map<int64_t, size_t> reserves;
  std::unordered_map<int64_t, size_t> copy_issues;
  std::unordered_map<int64_t, size_t> copy_completions;
  std::unordered_map<int64_t, size_t> releases;
  std::unordered_map<int64_t, std::vector<size_t>> uses;
  std::unordered_map<int64_t, size_t> use_count;
  std::unordered_map<int64_t, std::string> consumers;
  std::unordered_map<int64_t, int> effects;
  std::unordered_map<int64_t, ffi::Array<PrimExpr>> logical_shapes;
  for (const auto &[id, tensor] : tensors) {
    Check(SupportedComputeDType(tensor->dtype),
          "Phase 4 Tensor dtype unsupported");
    Check(tensor->tile_shape.size() == 2 &&
              RequireStaticInteger(tensor->tile_shape[0], "Tensor tile rows") ==
                  32 &&
              RequireStaticInteger(tensor->tile_shape[1],
                                   "Tensor tile columns") == 32,
          "Phase 4 Tensor physical tile shape must be [32,32]");
    Check(!pipeline.enabled || tensor->effect != "inout",
          "pipeline does not support inout Tensor prefetch dependencies");
    Check(tensor->alias_group == id,
          "Phase 4 cross-parameter storage aliases are unsupported");
    if (!tensor->strides.empty()) {
      arith::Analyzer analyzer;
      PrimExpr stride = Integer(1);
      for (size_t axis = tensor->shape.size(); axis-- > 0;) {
        Check(analyzer.CanProveEqual(tensor->strides[axis], stride),
              "Phase 4 Tensor requires compact row-major strides");
        stride = stride * tensor->shape[axis];
      }
    }
    Check(tensor->memory_space == "dram" &&
              tensor->memory_layout == "interleaved",
          "Phase 4 Tensor layout unsupported");
    Check(!tensor->shard_spec.has_value(),
          "Phase 4 sharded Tensor unsupported");
    VerifyTileGrid(tensor->shape, tensor->tile_grid_shape,
                   "Tensor " + std::to_string(id));
  }
  for (const auto &[id, dfb] : dfbs) {
    Check(SupportedComputeDType(dfb->element_dtype),
          "Phase 4 DFB dtype unsupported");
    int64_t capacity =
        RequireStaticInteger(dfb->block_count, "DFB block count");
    Check(capacity >= 1 && capacity <= 32,
          "Phase 4 DFB block count must be in [1,32]");
    Check(dfb->tile_shape.size() == 2 &&
              RequireStaticInteger(dfb->tile_shape[0], "tile rows") == 32 &&
              RequireStaticInteger(dfb->tile_shape[1], "tile cols") == 32,
          "Phase 4 physical tile shape must be [32,32]");
    Check(RequireStaticInteger(dfb->transaction_count_or_loop_relation,
                               "transaction count") == 1,
          "immutable DFB generation has exactly one publication");
    Check(ffi::StructuralEqual()(dfb->producer_domain, dfb->consumer_domain),
          "Phase 4 resource domains must match");
    if (dfb->tensor_backing.has_value()) {
      const auto &backing = dfb->tensor_backing.value();
      const TensorDescriptor &tensor = tensors.at(backing->global_arg_index);
      Check(RequireStaticInteger(backing->byte_offset, "backing byte offset") ==
                0,
            "Phase 4 backing offset must be zero");
      Check(dfb->element_dtype == tensor->dtype &&
                (accumulators || value_program ||
                 dfb->block_shape_in_tiles.size() <
                     tensor->tile_grid_shape.size() ||
                 ffi::StructuralEqual()(dfb->block_shape_in_tiles,
                                        tensor->tile_grid_shape)),
            "DFB Tensor backing metadata mismatch");
    }
  }
  for (const auto &[global, base] : mod->functions) {
    PrimFunc function = Downcast<PrimFunc>(base);
    values.BindFunction(function);
    std::string slot = function->GetAttr<ffi::String>(kKernelSlotAttr).value();
    auto tensor_indices =
        function->GetAttr<ffi::Array<Integer>>(kTensorArgIndicesAttr).value();
    Check(slot == "ncrisc" || tensor_indices.empty(),
          "Phase 4 Tensor ABI belongs exclusively to ncrisc");
    if (slot == "brisc") {
      Check(IsCanonicalNoOp(function->body), "Phase 4 brisc must be idle");
      continue;
    }
    if (IsCanonicalNoOp(function->body))
      continue;
    ffi::Array<Stmt> statements;
    if (const auto *seq = function->body.as<SeqStmtNode>())
      statements = seq->seq;
    else
      statements.push_back(function->body);
    std::unordered_set<int64_t> waited;
    size_t previous = events.size();
    bool first = true;
    for (const Stmt &statement : statements) {
      try {
        const auto *evaluate = statement.as<EvaluateNode>();
        Check(
            evaluate != nullptr,
            "Phase 4 slot body must contain only scheduled Device operations");
        const auto *node = evaluate->value.as<CallNode>();
        Check(node && node->dtype.is_void(),
              "Phase 4 Device operation must be a void intrinsic");
        Call call = ffi::GetRef<Call>(node);
        size_t event = events.size();
        events.push_back({call, slot});
        dependencies.emplace_back();
        if (!first)
          dependencies[event].push_back(previous);
        first = false;
        previous = event;
        auto id_at = [&](size_t index) {
          Check(index < call->args.size(),
                "Device operation missing resource ID");
          int64_t id = RequireStaticInteger(call->args[index], "resource ID");
          Check(dfbs.count(id), "Device operation references missing DFB " +
                                    std::to_string(id));
          return id;
        };
        auto read = [&](int64_t id) {
          Check(waited.count(id),
                "DFB use must be preceded by dfb_wait in consumer slot");
          Check(dfbs.at(id)->consumer_slot == slot,
                "DFB consumer slot metadata mismatch");
          Check(!(pipeline.enabled || value_program) || !releases.count(id),
                "DFB use after release risks overwritten data");
          ++use_count[id];
          uses[id].push_back(event);
          consumers[id] = slot;
        };
        auto publish = [&](int64_t id) {
          Check(reserves.count(id),
                "DFB publication must be preceded by dfb_reserve");
          Check(dfbs.at(id)->producer_slot == slot,
                "DFB producer slot metadata mismatch");
          Check(producers.emplace(id, event).second,
                "DFB generation published more than once");
        };
        if (call->op.same_as(compute_precision())) {
          values.Precision(call);
        } else if (call->op.same_as(dfb_reserve()) ||
                   call->op.same_as(dfb_wait())) {
          Check(call->args.size() == 1,
                "immutable DFB reserve/wait requires one stream operand");
          int64_t id = id_at(0);
          if (call->op.same_as(dfb_reserve())) {
            Check(dfbs.at(id)->producer_slot == slot,
                  "DFB reserve in wrong slot");
            Check(reserves.emplace(id, event).second,
                  "DFB generation reserved more than once");
          } else {
            Check(dfbs.at(id)->consumer_slot == slot, "DFB wait in wrong slot");
            Check(!(pipeline.enabled || value_program) || !releases.count(id),
                  "DFB wait after release risks overwritten data");
            waited.insert(id);
          }
        } else if (pipeline.enabled && call->op.same_as(dfb_copy_wait())) {
          Check(slot == "ncrisc" && call->args.size() == 1,
                "dfb_copy_wait requires ncrisc and one stream operand");
          int64_t id = id_at(0);
          Check(copy_issues.count(id),
                "dfb_copy_wait must follow its copy issue");
          Check(copy_completions.emplace(id, event).second,
                "DFB copy completed more than once");
          if (events[copy_issues.at(id)].call->op.same_as(tensor_to_dfb_nd()))
            publish(id);
        } else if ((pipeline.enabled || value_program) &&
                   call->op.same_as(dfb_release())) {
          Check(call->args.size() == 1,
                "dfb_release requires one stream operand");
          int64_t id = id_at(0);
          Check(dfbs.at(id)->consumer_slot == slot,
                "DFB release in wrong consumer slot");
          Check(waited.count(id) && use_count[id] > 0,
                "DFB release must follow wait and consumption");
          Check(releases.emplace(id, event).second,
                "DFB generation released more than once");
          if (copy_issues.count(id) &&
              events[copy_issues.at(id)].call->op.same_as(dfb_to_tensor_nd()))
            Check(copy_completions.count(id),
                  "DFB output release before copy completion risks overwritten "
                  "data");
        } else if (accumulators &&
                   (call->op.same_as(accumulator_init()) ||
                    call->op.same_as(gemm_update()) ||
                    call->op.same_as(accumulator_materialize()))) {
          Check(slot == "trisc", "accumulator operation must execute in trisc");
          bool update = call->op.same_as(gemm_update());
          bool init = call->op.same_as(accumulator_init());
          Check(call->args.size() == (update ? 5
                                      : init ? 1
                                             : 2),
                "accumulator operation arity mismatch");
          int64_t accumulator_id = RequireStaticInteger(
              call->args[update ? 2 : 0], "accumulator ID");
          Check(accumulator_table.count(accumulator_id),
                "operation references missing accumulator");
          if (update) {
            read(id_at(0));
            read(id_at(1));
          } else if (!init) {
            ffi::Array<PrimExpr> shape;
            for (const auto &range : accumulator_table.at(accumulator_id)
                                         ->accumulator_region->region)
              shape.push_back(range->extent);
            logical_shapes[id_at(1)] = shape;
            publish(id_at(1));
          }
        } else if (value_program && (call->op.same_as(compute_value()) ||
                                     call->op.same_as(compute_value_gemm()) ||
                                     call->op.same_as(compute_value_store()))) {
          Check(slot == "trisc", "compute values cannot cross processor slots");
          if (call->op.same_as(compute_value())) {
            for (int64_t id : values.Expression(call, dfbs))
              read(id);
          } else if (call->op.same_as(compute_value_gemm())) {
            for (int64_t id : values.Gemm(call, dfbs))
              read(id);
          } else {
            for (int64_t id : values.Store(call, dfbs))
              read(id);
            auto source = values.Read(
                RequireStaticInteger(call->args[0], "stored value"));
            logical_shapes[id_at(1)] = source->buffer->shape;
            publish(id_at(1));
          }
        } else if (call->op.same_as(dfb_compute())) {
          Check(slot == "trisc", "dfb_compute must execute in trisc");
          if (value_program) {
            for (int64_t id : values.Expression(call, dfbs))
              read(id);
          } else
            VerifyGeneralCompute(call, dfbs);
          logical_shapes[id_at(0)] = ComputeShape(call, "tt.logical_domain");
          for (size_t i = 1; i < call->args.size(); ++i) {
            read(id_at(i));
            if (pipeline.enabled)
              Check(pipeline.ordinals.at(id_at(i)) == -1 ||
                        pipeline.ordinals.at(id_at(0)) == -1 ||
                        pipeline.ordinals.at(id_at(i)) ==
                            pipeline.ordinals.at(id_at(0)),
                    "dfb_compute transaction relation crosses pipeline "
                    "iterations");
          }
          publish(id_at(0));
        } else if (call->op.same_as(tensor_to_dfb_nd()) ||
                   call->op.same_as(dfb_to_tensor_nd())) {
          Check(slot == "ncrisc", "Tensor transfer must execute in ncrisc");
          bool input = call->op.same_as(tensor_to_dfb_nd());
          int64_t id = id_at(input ? 1 : 0);
          int64_t tensor_id =
              RequireStaticInteger(call->args[input ? 0 : 1], "Tensor ID");
          Check(tensors.count(tensor_id), "transfer references missing Tensor");
          auto tensor = tensors.at(tensor_id);
          Check(call->args.size() == 2 + 2 * tensor->shape.size(),
                "ND transfer rank/arity mismatch");
          ffi::Array<PrimExpr> transfer_shape;
          for (size_t axis = 0; axis < tensor->shape.size(); ++axis) {
            const PrimExpr &start = call->args[2 + 2 * axis];
            const PrimExpr &extent = call->args[3 + 2 * axis];
            if (accumulators || value_program ||
                tensor->shape.size() >
                    dfbs.at(id)->block_shape_in_tiles.size()) {
              arith::Analyzer analyzer;
              int64_t begin = RequireStaticInteger(start, "transfer start");
              int64_t size = RequireStaticInteger(extent, "transfer extent");
              int64_t tile = axis + 2 >= tensor->shape.size() ? 32 : 1;
              Check(
                  begin >= 0 && begin % tile == 0 && size > 0 &&
                      size % tile == 0 &&
                      analyzer.CanProve(start + extent <= tensor->shape[axis]),
                  "schema v5 Tensor slice must be tile-aligned and in bounds");
            } else {
              Check(RequireStaticInteger(start, "transfer start") == 0 &&
                        ffi::StructuralEqual()(extent, tensor->shape[axis]),
                    "ND transfer must cover full Tensor");
            }
            transfer_shape.push_back(extent);
          }
          transfer_shape = LocalTransferShape(transfer_shape, dfbs.at(id));
          VerifyTileGrid(transfer_shape, dfbs.at(id)->block_shape_in_tiles,
                         "Tensor transfer");
          auto dfb = dfbs.at(id);
          Check(dfb->tensor_backing.has_value() &&
                    dfb->tensor_backing.value()->global_arg_index == tensor_id,
                "transfer disagrees with DFB Tensor backing");
          effects[tensor_id] |= input ? 1 : 2;
          if (input) {
            logical_shapes[id] = transfer_shape;
            if (!pipeline.enabled)
              publish(id);
          } else
            read(id);
          if (pipeline.enabled) {
            Check(copy_issues.emplace(id, event).second,
                  "DFB generation has multiple copy issues");
            if (input)
              Check(reserves.count(id) && dfb->producer_slot == slot,
                    "DFB copy issue must follow reserve in producer slot");
          }
        } else {
          Fail("Phase 4 slot retains an unsupported Device operation");
        }
      } catch (const ffi::Error &error) {
        Fail(OperationContext(function, statement) + error.what());
      }
    }
  }
  values.Finish();
  for (const auto &[id, descriptor] : dfbs) {
    auto Check = [&, resource = descriptor](bool condition,
                                            const std::string &message) {
      if (!condition)
        Fail(ResourceContext(mod, resource) + message);
    };
    Check(producers.count(id) && reserves.count(id),
          "DFB generation is never published");
    Check(use_count[id] > 0, "DFB generation is never consumed");
    if (pipeline.enabled || value_program) {
      Check(releases.count(id), "DFB generation is missing release");
      if (copy_issues.count(id))
        Check(copy_completions.count(id),
              "DFB copy is missing completion wait");
      for (size_t use : uses[id])
        dependencies[releases.at(id)].push_back(use);
    }
  }
  if (pipeline.enabled) {
    for (const auto &[group, ids] : pipeline.ordered) {
      int64_t capacity = RequireStaticInteger(dfbs.at(ids.front())->block_count,
                                              "pipeline pool block_count");
      for (size_t ordinal = 0; ordinal < ids.size(); ++ordinal) {
        if (ordinal > 0) {
          Check(reserves.at(ids[ordinal - 1]) < reserves.at(ids[ordinal]),
                "pipeline pool reserve order disagrees with transaction "
                "relation");
          Check(producers.at(ids[ordinal - 1]) < producers.at(ids[ordinal]) &&
                    releases.at(ids[ordinal - 1]) < releases.at(ids[ordinal]) &&
                    uses.at(ids[ordinal - 1]).back() <
                        uses.at(ids[ordinal]).front(),
                "pipeline pool publication/consumption/release order disagrees "
                "with "
                "transaction relation");
        }
        if (ordinal >= static_cast<size_t>(capacity))
          dependencies[reserves.at(ids[ordinal])].push_back(
              releases.at(ids[ordinal - capacity]));
      }
    }
  }
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].call->op.same_as(dfb_wait())) {
      int64_t id = RequireStaticInteger(events[i].call->args[0], "wait ID");
      dependencies[i].push_back(producers.at(id));
    }
  }
  for (const Event &event : events) {
    const Call &call = event.call;
    if (call->op.same_as(dfb_compute()) || call->op.same_as(compute_value())) {
      auto shapes = Downcast<ffi::Array<ffi::Array<PrimExpr>>>(
          call->annotations.at("tt.input_shapes"));
      for (size_t i = 1; i < call->args.size(); ++i) {
        int64_t id = RequireStaticInteger(call->args[i], "input DFB ID");
        Check(
            logical_shapes.count(id) &&
                IsCompatibleLogicalInput(logical_shapes.at(id), shapes[i - 1]),
            "dfb_compute input logical shape disagrees with its producer");
      }
    } else if (accumulators && call->op.same_as(gemm_update())) {
      for (size_t operand = 0; operand < 2; ++operand) {
        int64_t id =
            RequireStaticInteger(call->args[operand], "GEMM input DFB ID");
        Check(logical_shapes.count(id) && logical_shapes.at(id).size() == 2,
              "GEMM update input has no rank-2 logical shape producer");
        for (size_t axis = 0; axis < 2; ++axis) {
          int64_t extent = RequireStaticInteger(logical_shapes.at(id)[axis],
                                                "GEMM input extent");
          Check(extent > 0 && extent % 32 == 0 &&
                    extent / 32 == RequireStaticInteger(
                                       dfbs.at(id)->block_shape_in_tiles[axis],
                                       "GEMM tile extent"),
                "GEMM update input logical shape disagrees with tile grid");
        }
      }
    } else if (call->op.same_as(dfb_to_tensor_nd())) {
      int64_t id = RequireStaticInteger(call->args[0], "export DFB ID");
      if (value_program)
        values.Export(id, dfbs.at(id)->element_dtype);
      ffi::Array<PrimExpr> shape;
      for (size_t axis = 3; axis < call->args.size(); axis += 2)
        shape.push_back(call->args[axis]);
      shape = LocalTransferShape(shape, dfbs.at(id));
      Check(logical_shapes.count(id) &&
                ffi::StructuralEqual()(logical_shapes.at(id), shape),
            "export logical shape disagrees with its producer");
    }
  }
  // Check the combined slot-order and wait-for graph: individually legal
  // streams can still deadlock when their cross-slot dependencies form a cycle.
  std::vector<int> state(events.size(), 0);
  std::function<void(size_t)> visit = [&](size_t event) {
    if (state[event] == 1) {
      std::string detail;
      if (pipeline.enabled) {
        const Call &call = events[event].call;
        size_t index = call->op.same_as(tensor_to_dfb_nd()) ? 1 : 0;
        detail = " involving pipeline capacity/release dependencies";
        if (index < call->args.size()) {
          const int64_t *id = as_const_int(call->args[index]);
          if (id && pipeline.groups.count(*id))
            detail += " for DFB " + std::to_string(*id) + " storage pool " +
                      std::to_string(pipeline.groups.at(*id));
        }
      }
      Fail("Device transaction dependency cycle (cross-slot deadlock)" +
           detail);
    }
    if (state[event] == 2)
      return;
    state[event] = 1;
    for (size_t dependency : dependencies[event])
      visit(dependency);
    state[event] = 2;
  };
  for (size_t i = 0; i < events.size(); ++i)
    visit(i);
  for (const auto &[id, tensor] : tensors) {
    ffi::String expected = effects[id] == 3   ? "inout"
                           : effects[id] == 2 ? "output"
                                              : "input";
    Check(!pipeline.enabled || effects[id] != 3,
          "pipeline does not support inout Tensor prefetch dependencies");
    Check(tensor->effect == expected,
          "Tensor effect disagrees with scheduled dataflow");
  }
  for (const auto &[global, base] : mod->functions) {
    PrimFunc function = Downcast<PrimFunc>(base);
    if (function->GetAttr<ffi::String>(kKernelSlotAttr).value() != "ncrisc")
      continue;
    std::unordered_set<int64_t> declared;
    for (const Integer &id :
         function->GetAttr<ffi::Array<Integer>>(kTensorArgIndicesAttr).value())
      declared.insert(id->value);
    std::unordered_set<int64_t> used;
    for (const auto &[id, access] : effects)
      if (access)
        used.insert(id);
    Check(declared == used,
          "NCRISC Tensor ABI must contain exactly the used Tensor arguments");
  }
}

// Schema v4 specializes every Core and keeps each DFB generation immutable.
// Edges describe logical completion/readiness, not a claim about a NoC runtime.
using CoreKey = std::pair<int64_t, int64_t>;

struct CurrentStoragePlan {
  std::map<int64_t, DFBStorageGroupDescriptor> groups;
  std::map<int64_t, int64_t> resource_groups;
  std::map<int64_t, int64_t> resource_ordinals;
  std::map<int64_t, int64_t> resource_pipeline_iterations;
  std::map<int64_t, std::map<int64_t, int64_t>> ordered_resources;
};

CurrentStoragePlan VerifyCurrentStoragePlan(const IRModule &mod,
                                            const TensorTable &tensors,
                                            const CoreCoord &launch_grid) {
  Check(!mod->attrs->dict.count(kInternalDFBStorageGroupsAttr) &&
            !mod->attrs->dict.count(kInternalDFBTransactionRelationsAttr) &&
            !mod->attrs->dict.count(kPipelineRelationsAttr),
        "current Device IR cannot persist concrete storage-group maps");
  bool stream_operands =
      mod->GetAttr<Integer>("tt.stream_operands").value_or(Integer(0))->value ==
      1;
  auto families =
      mod->GetAttr<ffi::Array<DeviceIRDescriptorFamily>>(kDFBFamiliesAttr);
  auto groups = RequireModuleAttr<ffi::Array<DFBStorageGroupDescriptor>>(
      mod, kDFBStorageGroupsAttr);
  auto relations = mod->GetAttr<ffi::Array<DFBStorageGroupRelation>>(
      kDFBStorageGroupRelationsAttr);
  if (stream_operands) {
    Check(!families.has_value() && !relations.has_value(),
          "stream-centric Device IR cannot persist DFB generation relations");
  } else {
    Check(families.has_value() && relations.has_value() &&
              relations.value().size() == families.value().size() &&
              groups.empty() == families.value().empty(),
          "current Device IR requires one storage relation per DFB family");
  }

  CurrentStoragePlan result;
  auto pipeline_stages = mod->GetAttr<Integer>(kPipelineStagesAttr);
  auto pipeline_extent = mod->GetAttr<Integer>(kPipelineExtentAttr);
  Check(pipeline_stages.has_value() == pipeline_extent.has_value(),
        "pipeline stages and extent must be provided together");
  int64_t pipeline_depth = pipeline_stages.has_value()
                               ? std::min(pipeline_stages.value()->value,
                                          pipeline_extent.value()->value)
                               : 0;
  for (size_t index = 0; index < groups.size(); ++index) {
    const DFBStorageGroupDescriptor &group = groups[index];
    Check(group.defined() &&
              group->storage_group_id == static_cast<int64_t>(index),
          "storage group IDs must be dense and deterministic");
    Check(!group->element_dtype.is_void(),
          "storage group has a void element dtype");
    VerifyPositiveShape(group->tile_shape, "storage group tile shape");
    VerifyPositiveShape(group->block_shape_in_tiles,
                        "storage group block shape");
    int64_t capacity =
        RequireStaticInteger(group->block_count, "storage group capacity");
    Check(capacity >= 1 && capacity <= 32,
          "storage group capacity must be in [1,32]");
    int64_t transactions =
        RequireStaticInteger(group->transaction_count_or_loop_relation,
                             "storage group transaction count");
    Check(transactions >= 1,
          "storage group transaction count must be positive");
    Check(IsSupportedSlot(group->producer_slot) &&
              IsSupportedSlot(group->consumer_slot),
          "storage group has an invalid producer or consumer slot");
    VerifyDomain(group->producer_domain, launch_grid,
                 "storage group producer domain");
    VerifyDomain(group->consumer_domain, launch_grid,
                 "storage group consumer domain");
    if (group->storage_backing.has_value()) {
      const TensorStorageBacking &backing = group->storage_backing.value();
      Check(tensors.count(backing->global_arg_index),
            "storage backing references a missing Tensor");
      int64_t offset = RequireStaticInteger(backing->byte_offset,
                                            "storage backing byte offset");
      int64_t size =
          RequireStaticInteger(backing->byte_size, "storage backing byte size");
      Check(offset >= 0 && size > 0,
            "storage backing byte range must be positive");
      int64_t expected = capacity;
      for (const PrimExpr &extent : group->tile_shape)
        expected = CheckedProduct(expected,
                                  RequireStaticInteger(extent, "tile extent"),
                                  "storage backing byte size");
      for (const PrimExpr &extent : group->block_shape_in_tiles)
        expected = CheckedProduct(expected,
                                  RequireStaticInteger(extent, "block extent"),
                                  "storage backing byte size");
      expected = CheckedProduct(expected, group->element_dtype.bytes(),
                                "storage backing byte size");
      Check(size == expected,
            "storage backing byte size must equal the complete DFB capacity");
    }
    result.groups.emplace(group->storage_group_id, group);
  }

  if (stream_operands)
    return result;

  for (size_t family_id = 0; family_id < families.value().size(); ++family_id) {
    const DeviceIRDescriptorFamily &family = families.value()[family_id];
    const DFBStorageGroupRelation &relation = relations.value()[family_id];
    Check(relation.defined() &&
              relation->dfb_family_id == static_cast<int64_t>(family_id) &&
              family.defined() && family->positions.defined() &&
              family->fields.count("dfb_id"),
          "storage relation does not match its DFB family");
    int64_t count = family->positions->count;
    std::vector<int64_t> ids =
        ExpandDeviceIntColumn(family->fields.at("dfb_id"), count);
    std::vector<int64_t> group_ids =
        ExpandDeviceIntColumn(relation->storage_group_ids, count);
    std::vector<int64_t> ordinals =
        ExpandDeviceIntColumn(relation->transaction_ordinals, count);
    std::vector<int64_t> pipeline_iterations =
        ExpandDeviceIntColumn(relation->pipeline_iterations, count);
    auto prototype = family->prototype.as<DFBDescriptor>();
    Check(prototype.has_value(), "DFB family has the wrong prototype type");
    for (int64_t row = 0; row < count; ++row) {
      int64_t id = ids[row], group_id = group_ids[row], ordinal = ordinals[row];
      int64_t pipeline_iteration = pipeline_iterations[row];
      Check(id >= 0 && result.groups.count(group_id) && ordinal >= 0,
            "storage relation contains an invalid ID or ordinal");
      Check(result.resource_groups.emplace(id, group_id).second &&
                result.resource_ordinals.emplace(id, ordinal).second &&
                result.resource_pipeline_iterations
                    .emplace(id, pipeline_iteration)
                    .second &&
                result.ordered_resources[group_id].emplace(ordinal, id).second,
            "storage relation has duplicate DFB identities or ordinals");
      Check((pipeline_depth == 0 && pipeline_iteration == -1) ||
                (pipeline_depth > 0 &&
                 (pipeline_iteration == -1 ||
                  (pipeline_iteration >= 0 &&
                   pipeline_iteration < pipeline_extent.value()->value &&
                   pipeline_iteration == ordinal))),
            "storage relation has an invalid pipeline iteration");
      const DFBStorageGroupDescriptor &group = result.groups.at(group_id);
      const DFBDescriptor &dfb = prototype.value();
      Check(group->element_dtype == dfb->element_dtype &&
                ffi::StructuralEqual()(group->tile_shape, dfb->tile_shape) &&
                ffi::StructuralEqual()(group->block_shape_in_tiles,
                                       dfb->block_shape_in_tiles) &&
                ffi::StructuralEqual()(group->block_count, dfb->block_count) &&
                group->producer_slot == dfb->producer_slot &&
                ffi::StructuralEqual()(group->producer_domain,
                                       dfb->producer_domain) &&
                group->consumer_slot == dfb->consumer_slot &&
                ffi::StructuralEqual()(group->consumer_domain,
                                       dfb->consumer_domain),
            "DFB family and storage group contracts disagree");
    }
  }
  Check(result.resource_groups.empty() == families.value().empty(),
        "current Device IR has incomplete storage-group resource relations");
  for (const auto &[group_id, resources] : result.ordered_resources) {
    Check(!resources.empty() && resources.begin()->first == 0,
          "storage-group transaction ordinals must start at zero");
    int64_t expected = 0;
    for (const auto &[ordinal, resource] : resources)
      Check(ordinal == expected++,
            "storage-group transaction ordinals must be dense");
    if (pipeline_depth > 0) {
      std::set<int64_t> iterations;
      for (const auto &[ordinal, resource] : resources)
        iterations.insert(result.resource_pipeline_iterations.at(resource));
      bool singleton = iterations.count(-1);
      Check((singleton && resources.size() == 1 && iterations.size() == 1) ||
                (!singleton &&
                 resources.size() ==
                     static_cast<size_t>(pipeline_extent.value()->value) &&
                 *iterations.begin() == 0 &&
                 *iterations.rbegin() == pipeline_extent.value()->value - 1),
            "DFB transaction count disagrees with pipeline group coverage");
      if (!singleton)
        Check(RequireStaticInteger(result.groups.at(group_id)->block_count,
                                   "pipeline storage-group capacity") >=
                  pipeline_depth,
              "pipeline storage-group capacity insufficient");
    }
  }
  return result;
}

void VerifyCurrentL1Budget(
    const IRModule &mod, const DFBTable &dfbs,
    const std::map<int64_t, std::set<CoreKey>> &resource_cores,
    const CurrentStoragePlan &storage) {
  if (mod->attrs->dict.count(kPipelineStagesAttr) ||
      mod->attrs->dict.count(kPipelineExtentAttr) ||
      mod->attrs->dict.count("tt.pipeline_wait_policy")) {
    int64_t stages =
        RequireModuleAttr<Integer>(mod, kPipelineStagesAttr)->value;
    int64_t extent =
        RequireModuleAttr<Integer>(mod, kPipelineExtentAttr)->value;
    Check(stages > 0 && stages <= 32 && extent > 0 && extent <= 1024,
          "current Device pipeline stages/extent are outside supported bounds");
    auto wait_policy =
        RequireModuleAttr<ffi::String>(mod, "tt.pipeline_wait_policy");
    Check(wait_policy == "conservative" || wait_policy == "delayed",
          "current Device pipeline wait policy is invalid");
  }

  std::map<CoreKey, int64_t> payload;
  int64_t total_payload = 0;
  for (const auto &[group_id, group] : storage.groups) {
    std::set<CoreKey> cores;
    bool stream_operands = mod->GetAttr<Integer>("tt.stream_operands")
                               .value_or(Integer(0))
                               ->value == 1;
    if (stream_operands) {
      Check(dfbs.count(group_id),
            "logical stream descriptor references a missing DFB");
      auto uses = resource_cores.find(group_id);
      Check(uses != resource_cores.end() && !uses->second.empty(),
            "current Device stream has no body use: " +
                std::to_string(group_id));
      cores.insert(uses->second.begin(), uses->second.end());
    } else {
      for (const auto &[ordinal, id] : storage.ordered_resources.at(group_id)) {
        Check(dfbs.count(id), "storage relation references a missing DFB");
        auto uses = resource_cores.find(id);
        Check(uses != resource_cores.end() && !uses->second.empty(),
              "current Device DFB has no body use: " + std::to_string(id));
        cores.insert(uses->second.begin(), uses->second.end());
      }
    }
    const std::string label =
        "current Device storage group " + std::to_string(group_id);
    int64_t bytes =
        RequireStaticInteger(group->block_count, label + " block_count");
    for (const PrimExpr &value : group->tile_shape)
      bytes = CheckedProduct(bytes, RequireStaticInteger(value, label), label);
    for (const PrimExpr &value : group->block_shape_in_tiles)
      bytes = CheckedProduct(bytes, RequireStaticInteger(value, label), label);
    bytes = CheckedProduct(bytes, group->element_dtype.bytes(), label);
    Check(total_payload <= std::numeric_limits<int64_t>::max() - bytes,
          "current Device L1 payload byte sum overflows");
    total_payload += bytes;
    for (const CoreKey &core : cores) {
      Check(payload[core] <= std::numeric_limits<int64_t>::max() - bytes,
            "current Device per-Core L1 payload byte sum overflows");
      payload[core] += bytes;
    }
  }
  auto budget = mod->GetAttr<Integer>(kL1CapacityBytesAttr);
  if (budget.has_value()) {
    Check(budget.value()->value > 0, "tt.l1_capacity_bytes must be positive");
    for (const auto &[core, bytes] : payload)
      Check(bytes <= budget.value()->value,
            "Core (" + std::to_string(core.first) + "," +
                std::to_string(core.second) +
                ") L1 logical payload lower bound " + std::to_string(bytes) +
                " bytes exceeds explicit tt.l1_capacity_bytes budget " +
                std::to_string(budget.value()->value) +
                "; physical allocation overhead is not included");
  }
  auto reported = mod->GetAttr<Integer>("tt.l1_payload_bytes");
  if (reported.has_value())
    Check(reported.value()->value == total_payload,
          "tt.l1_payload_bytes disagrees with current Device storage pools");
}

CoreKey CoreOf(const CoreDomain &domain) {
  Check(domain->end->x == domain->begin->x + 1 &&
            domain->end->y == domain->begin->y + 1,
        "Phase 6 DFB domain must contain exactly one Core");
  return {domain->begin->x, domain->begin->y};
}

CoreKey CoreOf(const CoreCoord &coord) { return {coord->x, coord->y}; }

void VerifyMulticoreProgram(const IRModule &mod, const TensorTable &tensors,
                            const DFBTable &dfbs, const CoreCoord &grid,
                            const ffi::Array<PipeDescriptor> &pipes) {
  PipelineResources pipeline = VerifyPipelineResources(mod, dfbs);
  const auto transfers = RequireModuleAttr<ffi::Array<PipeTransferDescriptor>>(
      mod, kPipeTransferTableAttr);
  std::unordered_map<int64_t, PipeTransferDescriptor> transfer_table;
  std::map<std::pair<int64_t, int64_t>, PipeDescriptor> records;
  for (const PipeDescriptor &pipe : pipes)
    records.emplace(std::make_pair(pipe->pipe_net_id, pipe->record_index),
                    pipe);
  const bool extended =
      RequireModuleAttr<Integer>(mod, kDeviceIRVersionAttr)->value >= 6 ||
      pipeline.enabled;
  ComputeValueVerifier values(mod);
  const bool value_program = values.Enabled();
  std::unordered_map<int64_t, AccumulatorDescriptor> accumulators;
  if (auto entries = mod->GetAttr<ffi::Array<AccumulatorDescriptor>>(
          kAccumulatorTableAttr))
    for (const auto &entry : entries.value())
      accumulators.emplace(entry->accumulator_id, entry);
  using OccurrenceKey = std::tuple<int64_t, int64_t, int64_t>;
  std::map<OccurrenceKey, std::map<CoreKey, int64_t>> destinations;
  std::map<OccurrenceKey, int64_t> sources;
  std::map<std::pair<int64_t, int64_t>, std::set<int64_t>> occurrences;
  std::unordered_map<int64_t, int64_t> incoming;
  std::unordered_map<int64_t, std::vector<int64_t>> outgoing;
  std::map<CoreKey, int64_t> payload;
  for (const auto &[id, dfb] : dfbs) {
    CoreKey core = CoreOf(dfb->producer_domain);
    Check(core == CoreOf(dfb->consumer_domain),
          "Phase 6 DFB producer/consumer owner Core mismatch");
    Check(SupportedComputeDType(dfb->element_dtype),
          "Phase 6 DFB dtype unsupported");
    Check(dfb->tile_shape.size() == 2 &&
              RequireStaticInteger(dfb->tile_shape[0], "tile rows") == 32 &&
              RequireStaticInteger(dfb->tile_shape[1], "tile columns") == 32,
          "Phase 6 physical tile shape must be [32,32]");
    int64_t capacity =
        RequireStaticInteger(dfb->block_count, "DFB block count");
    Check(capacity >= 1 && capacity <= 32,
          "Phase 6 DFB capacity must be in [1,32]");
    Check(RequireStaticInteger(dfb->transaction_count_or_loop_relation,
                               "DFB transaction count") == 1,
          "Phase 6 immutable DFB generation requires exactly one transaction");
    int64_t bytes = capacity;
    for (const PrimExpr &extent : dfb->tile_shape)
      bytes = CheckedProduct(bytes, RequireStaticInteger(extent, "tile extent"),
                             "Phase 6 L1 payload");
    for (const PrimExpr &extent : dfb->block_shape_in_tiles)
      bytes =
          CheckedProduct(bytes, RequireStaticInteger(extent, "block extent"),
                         "Phase 6 L1 payload");
    bytes =
        CheckedProduct(bytes, dfb->element_dtype.bytes(), "Phase 6 L1 payload");
    Check(payload[core] <= std::numeric_limits<int64_t>::max() - bytes,
          "Phase 6 L1 payload byte sum overflows");
    payload[core] += bytes;
    if (dfb->tensor_backing.has_value()) {
      const TensorBacking &backing = dfb->tensor_backing.value();
      Check(RequireStaticInteger(backing->byte_offset, "backing byte offset") ==
                0,
            "Phase 6 Tensor backing offset must be zero; slices belong to "
            "transfer ops");
      Check(dfb->element_dtype == tensors.at(backing->global_arg_index)->dtype,
            "Phase 6 DFB Tensor backing dtype mismatch");
    }
  }
  auto budget = mod->GetAttr<Integer>(kL1CapacityBytesAttr);
  if (budget.has_value() && !pipeline.enabled) {
    Check(budget.value()->value > 0, "tt.l1_capacity_bytes must be positive");
    for (const auto &[core, bytes] : payload)
      Check(bytes <= budget.value()->value,
            "Core (" + std::to_string(core.first) + "," +
                std::to_string(core.second) +
                ") L1 logical payload lower bound " + std::to_string(bytes) +
                " exceeds explicit tt.l1_capacity_bytes; "
                "physical allocation overhead is not included");
  }
  for (const auto &[id, tensor] : tensors) {
    Check(SupportedComputeDType(tensor->dtype),
          "Phase 6 Tensor dtype unsupported");
    Check(tensor->tile_shape.size() == 2 &&
              RequireStaticInteger(tensor->tile_shape[0], "Tensor tile rows") ==
                  32 &&
              RequireStaticInteger(tensor->tile_shape[1],
                                   "Tensor tile columns") == 32,
          "Phase 6 Tensor tile shape must be [32,32]");
    Check(tensor->alias_group == id && !tensor->shard_spec.has_value() &&
              tensor->memory_space == "dram" &&
              tensor->memory_layout == "interleaved",
          "Phase 6 requires unaliased interleaved DRAM Tensors");
    VerifyTileGrid(tensor->shape, tensor->tile_grid_shape, "Phase 6 Tensor");
    if (!tensor->strides.empty()) {
      arith::Analyzer analyzer;
      PrimExpr stride = Integer(1);
      for (size_t axis = tensor->shape.size(); axis-- > 0;) {
        Check(analyzer.CanProveEqual(tensor->strides[axis], stride),
              "Phase 6 Tensor requires compact row-major strides");
        stride = stride * tensor->shape[axis];
      }
    }
  }
  int64_t expected_transfer = 0;
  for (const PipeTransferDescriptor &transfer : transfers) {
    Check(transfer->transfer_id == expected_transfer++,
          "Pipe transfer IDs must follow stable contiguous table order");
    Check(transfer_table.emplace(transfer->transfer_id, transfer).second,
          "duplicate Pipe transfer identity");
    const auto record_key =
        std::make_pair(transfer->pipe_net_id, transfer->record_index);
    Check(records.count(record_key),
          "Pipe transfer references missing original record");
    const PipeDescriptor &record = records.at(record_key);
    Check(transfer->communication_id == record->communication_id,
          "Pipe transfer parent communication disagrees with original record");
    Check(transfer->occurrence >= 0 && (extended || transfer->occurrence == 0),
          "schema v4 supports one occurrence; schema v6 requires a nonnegative "
          "epoch");
    OccurrenceKey occurrence_key{transfer->pipe_net_id, transfer->record_index,
                                 transfer->occurrence};
    occurrences[record_key].insert(transfer->occurrence);
    Check(transfer->transaction_count == 1,
          "Pipe transfer must contain exactly one transaction");
    VerifyCoord(transfer->src_coord, grid, "Pipe transfer source Core");
    VerifyCoord(transfer->dst_coord, grid, "Pipe transfer destination Core");
    Check(CoreOf(transfer->src_coord) == CoreOf(record->src_coord),
          "Pipe transfer source Core disagrees with original record");
    CoreKey dest = CoreOf(transfer->dst_coord);
    Check(dest.first >= record->dst_begin->x &&
              dest.first < record->dst_end->x &&
              dest.second >= record->dst_begin->y &&
              dest.second < record->dst_end->y,
          "Pipe transfer destination Core is outside original record domain");
    Check(destinations[occurrence_key]
              .emplace(dest, transfer->transfer_id)
              .second,
          "Pipe record has a duplicate destination transaction");
    Check(dfbs.count(transfer->source_dfb_id) &&
              dfbs.count(transfer->destination_dfb_id),
          "Pipe transfer references missing source/destination DFB");
    Check(transfer->occurrence != 0 ||
              transfer->source_dfb_id == record->payload_dfb_id,
          "Pipe transfer source DFB disagrees with original record payload");
    auto [source_entry, new_source] =
        sources.emplace(occurrence_key, transfer->source_dfb_id);
    Check(new_source || source_entry->second == transfer->source_dfb_id,
          "Pipe record occurrence has inconsistent source payloads");
    DFBDescriptor source = dfbs.at(transfer->source_dfb_id);
    DFBDescriptor destination = dfbs.at(transfer->destination_dfb_id);
    if (pipeline.enabled)
      Check(pipeline.ordinals.at(transfer->source_dfb_id) ==
                pipeline.ordinals.at(transfer->destination_dfb_id),
            "Pipe transfer " + std::to_string(transfer->transfer_id) +
                " crosses pipeline iterations");
    Check(CoreOf(source->consumer_domain) == CoreOf(transfer->src_coord) &&
              CoreOf(destination->producer_domain) == dest,
          "Pipe transfer endpoint owner Core mismatch");
    const bool forwarding = extended && source->producer_slot == "ncrisc" &&
                            source->consumer_slot == "trisc" &&
                            source->tensor_backing.has_value();
    Check(
        (source->consumer_slot == "brisc" || forwarding) &&
            destination->producer_slot == "ncrisc",
        "Pipe source requires BRISC affinity or a forwarding NCRISC producer, "
        "and receiver requires NCRISC");
    Check(source->element_dtype == destination->element_dtype &&
              ffi::StructuralEqual()(source->tile_shape,
                                     destination->tile_shape) &&
              ffi::StructuralEqual()(source->block_shape_in_tiles,
                                     destination->block_shape_in_tiles),
          "Pipe transfer payload dtype/shape mismatch");
    Check(incoming.emplace(transfer->destination_dfb_id, transfer->transfer_id)
              .second,
          "Pipe destination DFB has multiple producers (overwrite risk)");
    outgoing[transfer->source_dfb_id].push_back(transfer->transfer_id);
    Check(transfer->source_span.defined(), "Pipe transfer has no source span");
  }
  for (const auto &[key, record] : records) {
    int64_t count = (record->dst_end->x - record->dst_begin->x) *
                    (record->dst_end->y - record->dst_begin->y);
    Check(!occurrences[key].empty(), "Pipe record has no transactions");
    int64_t expected_occurrence = 0;
    for (int64_t occurrence : occurrences[key]) {
      Check(occurrence == expected_occurrence++,
            "Pipe record occurrences must be contiguous from zero");
      OccurrenceKey occurrence_key{key.first, key.second, occurrence};
      Check(destinations[occurrence_key].size() == static_cast<size_t>(count),
            "Pipe record destination transactions are not closed (missing "
            "receiver)");
    }
  }

  struct Event {
    Call call;
    std::string slot;
    CoreKey core;
  };
  struct WriteRegion {
    int64_t tensor;
    CoreKey core;
    ffi::Array<PrimExpr> bounds;
  };
  std::vector<Event> events;
  std::vector<std::vector<size_t>> dependencies;
  std::vector<WriteRegion> writes;
  std::unordered_map<int64_t, size_t> reserves, producers, releases,
      copy_issues, copy_completions, sends, receives, send_completions,
      recv_completions;
  std::unordered_map<int64_t, std::vector<size_t>> uses;
  std::unordered_set<int64_t> discards;
  std::unordered_map<int64_t, ffi::Array<PrimExpr>> logical_shapes;
  std::unordered_map<int64_t, int> effects;
  std::map<std::pair<CoreKey, std::string>, std::string> function_names;
  for (const auto &[global, base] : mod->functions) {
    PrimFunc func = Downcast<PrimFunc>(base);
    values.BindFunction(func);
    std::string slot = func->GetAttr<ffi::String>(kKernelSlotAttr).value();
    CoreKey core = CoreOf(func->GetAttr<CoreDomain>(kCoreDomainAttr).value());
    function_names.emplace(std::make_pair(core, slot), global->name_hint);
    std::unordered_set<int64_t> declared, used_tensors, waited;
    std::unordered_map<int64_t, std::pair<int64_t, int64_t>> last_record;
    for (const Integer &id :
         func->GetAttr<ffi::Array<Integer>>(kTensorArgIndicesAttr).value())
      declared.insert(id->value);
    Check(slot == "ncrisc" || declared.empty(),
          "Phase 6 Tensor ABI belongs to NCRISC");
    ffi::Array<Stmt> statements;
    if (!IsCanonicalNoOp(func->body)) {
      if (const auto *seq = func->body.as<SeqStmtNode>())
        statements = seq->seq;
      else
        statements.push_back(func->body);
    }
    size_t previous = 0;
    bool first = true;
    for (const Stmt &statement : statements) {
      try {
        const auto *evaluate = statement.as<EvaluateNode>();
        Check(
            evaluate != nullptr,
            "Phase 6 slot body requires explicit scheduled Device operations");
        const auto *node = evaluate->value.as<CallNode>();
        Check(node && node->dtype.is_void(),
              "Phase 6 Device operation must be a void intrinsic");
        Call call = ffi::GetRef<Call>(node);
        size_t event = events.size();
        events.push_back({call, slot, core});
        dependencies.emplace_back();
        if (!first)
          dependencies[event].push_back(previous);
        first = false;
        previous = event;
        auto id_at = [&](size_t index) {
          Check(index < call->args.size(), "Device operation missing DFB ID");
          int64_t id = RequireStaticInteger(call->args[index], "DFB ID");
          Check(dfbs.count(id), "Device operation references missing DFB");
          Check(CoreOf(dfbs.at(id)->producer_domain) == core,
                "Device operation references a DFB owned by another Core");
          return id;
        };
        auto read = [&](int64_t id) {
          Check(dfbs.count(id) && CoreOf(dfbs.at(id)->consumer_domain) == core,
                "compute value borrows a DFB owned by another Core");
          Check(waited.count(id),
                "DFB use must be preceded by dfb_wait in consumer slot");
          Check(dfbs.at(id)->consumer_slot == slot,
                "DFB consumer slot metadata mismatch");
          Check(!releases.count(id),
                "DFB use after release risks overwritten data");
          uses[id].push_back(event);
        };
        auto publish = [&](int64_t id) {
          Check(reserves.count(id), "DFB publication must follow dfb_reserve");
          Check(dfbs.at(id)->producer_slot == slot,
                "DFB producer slot metadata mismatch");
          Check(producers.emplace(id, event).second,
                "DFB generation published more than once");
        };
        if (call->op.same_as(compute_precision())) {
          values.Precision(call);
        } else if (call->op.same_as(dfb_reserve()) ||
                   call->op.same_as(dfb_wait()) ||
                   call->op.same_as(dfb_release()) ||
                   call->op.same_as(dfb_copy_wait())) {
          Check(call->args.size() == 1,
                "Phase 6 DFB lifecycle operation requires one stream operand");
          int64_t id = id_at(0);
          if (call->op.same_as(dfb_reserve())) {
            Check(dfbs.at(id)->producer_slot == slot,
                  "DFB reserve in wrong producer slot");
            Check(reserves.emplace(id, event).second,
                  "DFB generation reserved more than once");
          } else if (call->op.same_as(dfb_wait())) {
            Check(dfbs.at(id)->consumer_slot == slot,
                  "DFB wait in wrong consumer slot");
            Check(!releases.count(id),
                  "DFB wait after release risks overwritten data");
            waited.insert(id);
          } else if (call->op.same_as(dfb_copy_wait())) {
            Check(slot == "ncrisc" && copy_issues.count(id),
                  "dfb_copy_wait must follow copy issue in NCRISC");
            Check(copy_completions.emplace(id, event).second,
                  "DFB copy completed more than once");
            if (events[copy_issues.at(id)].call->op.same_as(tensor_to_dfb_nd()))
              publish(id);
          } else {
            bool discard =
                incoming.count(id) && slot == "ncrisc" && uses[id].empty();
            Check(dfbs.at(id)->consumer_slot == slot && waited.count(id) &&
                      (!uses[id].empty() || discard),
                  "DFB release must follow wait and consumption in consumer "
                  "slot");
            if (discard) {
              Check(recv_completions.count(incoming.at(id)),
                    "Pipe discard must follow receive completion, wait, and "
                    "release");
              discards.insert(id);
            }
            Check(releases.emplace(id, event).second,
                  "DFB generation released more than once");
          }
        } else if (extended && (call->op.same_as(accumulator_init()) ||
                                call->op.same_as(gemm_update()) ||
                                call->op.same_as(accumulator_materialize()))) {
          Check(slot == "trisc", "accumulator operation requires TRISC");
          const bool update = call->op.same_as(gemm_update());
          const bool init = call->op.same_as(accumulator_init());
          Check(call->args.size() == (update ? 5
                                      : init ? 1
                                             : 2),
                "accumulator operation arity mismatch");
          int64_t accumulator = RequireStaticInteger(call->args[update ? 2 : 0],
                                                     "accumulator ID");
          Check(accumulators.count(accumulator),
                "operation references missing accumulator");
          if (update) {
            read(id_at(0));
            read(id_at(1));
          } else if (!init) {
            int64_t output = id_at(1);
            ffi::Array<PrimExpr> shape;
            for (const auto &range :
                 accumulators.at(accumulator)->accumulator_region->region)
              shape.push_back(range->extent);
            logical_shapes[output] = shape;
            publish(output);
          }
        } else if (value_program && (call->op.same_as(compute_value()) ||
                                     call->op.same_as(compute_value_gemm()) ||
                                     call->op.same_as(compute_value_store()))) {
          Check(slot == "trisc", "compute values cannot cross processor slots");
          if (call->op.same_as(compute_value())) {
            for (int64_t id : values.Expression(call, dfbs))
              read(id);
          } else if (call->op.same_as(compute_value_gemm())) {
            for (int64_t id : values.Gemm(call, dfbs))
              read(id);
          } else {
            for (int64_t id : values.Store(call, dfbs))
              read(id);
            auto source = values.Read(
                RequireStaticInteger(call->args[0], "stored compute value"));
            logical_shapes[id_at(1)] = source->buffer->shape;
            publish(id_at(1));
          }
        } else if (call->op.same_as(dfb_compute())) {
          Check(slot == "trisc", "dfb_compute requires TRISC");
          if (value_program) {
            for (int64_t id : values.Expression(call, dfbs))
              read(id);
          } else
            VerifyGeneralCompute(call, dfbs);
          int64_t output = id_at(0);
          logical_shapes[output] = ComputeShape(call, "tt.logical_domain");
          for (size_t i = 1; i < call->args.size(); ++i) {
            int64_t input = id_at(i);
            read(input);
            if (pipeline.enabled)
              Check(pipeline.ordinals.at(input) == -1 ||
                        pipeline.ordinals.at(output) == -1 ||
                        pipeline.ordinals.at(input) ==
                            pipeline.ordinals.at(output),
                    "dfb_compute transaction relation crosses pipeline "
                    "iterations");
          }
          publish(output);
        } else if (call->op.same_as(tensor_to_dfb_nd()) ||
                   call->op.same_as(dfb_to_tensor_nd())) {
          Check(slot == "ncrisc" && call->args.size() >= 2,
                "Tensor transfer requires NCRISC and Tensor/DFB operands");
          bool input = call->op.same_as(tensor_to_dfb_nd());
          int64_t id = id_at(input ? 1 : 0);
          int64_t tensor_id =
              RequireStaticInteger(call->args[input ? 0 : 1], "Tensor ID");
          Check(tensors.count(tensor_id), "transfer references missing Tensor");
          TensorDescriptor tensor = tensors.at(tensor_id);
          Check(call->args.size() == 2 + 2 * tensor->shape.size(),
                "ND transfer rank/arity mismatch");
          ffi::Array<PrimExpr> shape, bounds;
          for (size_t axis = 0; axis < tensor->shape.size(); ++axis) {
            int64_t start = RequireStaticInteger(call->args[2 + 2 * axis],
                                                 "transfer start");
            int64_t extent = RequireStaticInteger(call->args[3 + 2 * axis],
                                                  "transfer extent");
            int64_t limit =
                RequireStaticInteger(tensor->shape[axis], "Tensor extent");
            int64_t tile = axis + 2 >= tensor->shape.size() ? 32 : 1;
            Check(start >= 0 && start <= limit && extent > 0 &&
                      extent <= limit - start,
                  "ND transfer slice lies outside Tensor bounds");
            Check(start % tile == 0 && (extent == 1 || extent % tile == 0),
                  "Phase 6 Tensor transfer slice must align to whole tiles");
            shape.push_back(call->args[3 + 2 * axis]);
            bounds.push_back(call->args[2 + 2 * axis]);
            bounds.push_back(call->args[3 + 2 * axis]);
          }
          shape = LocalTransferShape(shape, dfbs.at(id));
          VerifyTileGrid(shape, dfbs.at(id)->block_shape_in_tiles,
                         "transfer DFB");
          const auto backing = dfbs.at(id)->tensor_backing;
          Check(backing.has_value() &&
                    backing.value()->global_arg_index == tensor_id,
                "transfer disagrees with DFB Tensor backing");
          effects[tensor_id] |= input ? 1 : 2;
          used_tensors.insert(tensor_id);
          Check(copy_issues.emplace(id, event).second,
                "DFB generation has multiple Tensor copy issues");
          if (input) {
            Check(reserves.count(id) && dfbs.at(id)->producer_slot == slot,
                  "Tensor copy issue must follow reserve in producer slot");
            logical_shapes[id] = shape;
          } else {
            read(id);
            writes.push_back({tensor_id, core, bounds});
          }
        } else if (call->op.same_as(dfb_pipe_send()) ||
                   call->op.same_as(dfb_pipe_recv()) ||
                   call->op.same_as(dfb_pipe_wait())) {
          Check(call->args.size() == 3 &&
                    RequireStaticInteger(call->args[2],
                                         "Pipe transaction count") == 1,
                "Pipe operation must request exactly one transaction");
          int64_t tid = RequireStaticInteger(call->args[0], "Pipe transfer ID");
          Check(transfer_table.count(tid),
                "Pipe operation references missing transfer");
          const PipeTransferDescriptor &transfer = transfer_table.at(tid);
          int64_t id = id_at(1);
          bool source = id == transfer->source_dfb_id;
          Check(source || id == transfer->destination_dfb_id,
                "Pipe operation payload DFB mismatch");
          if (call->op.same_as(dfb_pipe_send()) ||
              call->op.same_as(dfb_pipe_recv())) {
            auto ordinal =
                std::make_pair(transfer->occurrence, transfer->record_index);
            auto [previous_record, inserted] =
                last_record.emplace(transfer->pipe_net_id, ordinal);
            Check(inserted || previous_record->second <= ordinal,
                  "Pipe operations must preserve occurrence and foreach record "
                  "order");
            previous_record->second = ordinal;
          }
          if (call->op.same_as(dfb_pipe_send())) {
            const bool forwarding = extended && slot == "ncrisc" &&
                                    dfbs.at(id)->producer_slot == slot &&
                                    dfbs.at(id)->consumer_slot == "trisc";
            Check(source && (slot == "brisc" || forwarding),
                  "Pipe send requires source BRISC or forwarding NCRISC "
                  "affinity");
            if (forwarding) {
              Check(copy_completions.count(id) && producers.count(id) &&
                        events[copy_issues.at(id)].call->op.same_as(
                            tensor_to_dfb_nd()),
                    "Pipe forwarding must follow its Tensor copy completion");
            } else {
              read(id);
            }
            Check(sends.emplace(tid, event).second,
                  "Pipe transaction sent more than once");
          } else if (call->op.same_as(dfb_pipe_recv())) {
            Check(!source && slot == "ncrisc",
                  "Pipe receive requires destination NCRISC");
            Check(reserves.count(id),
                  "Pipe receive must follow destination reserve");
            Check(receives.emplace(tid, event).second,
                  "Pipe transaction received more than once");
          } else if (source) {
            Check((slot == "brisc" || (extended && slot == "ncrisc" &&
                                       dfbs.at(id)->producer_slot == slot)) &&
                      sends.count(tid) && events[sends.at(tid)].slot == slot,
                  "Pipe source completion must follow send in its source slot");
            Check(send_completions.emplace(tid, event).second,
                  "Pipe source completion duplicated");
          } else {
            Check(slot == "ncrisc" && receives.count(tid),
                  "Pipe destination completion must follow receive in NCRISC");
            Check(recv_completions.emplace(tid, event).second,
                  "Pipe destination completion duplicated");
            publish(id);
          }
        } else {
          Fail("Phase 6 slot retains unsupported Device operation");
        }
      } catch (const ffi::Error &error) {
        Fail(OperationContext(func, statement) + error.what());
      }
    }
    Check(declared == used_tensors, "Phase 6 NCRISC Tensor ABI must contain "
                                    "exactly its used Tensor arguments");
  }
  values.Finish();
  auto order =
      RequireModuleAttr<ffi::Array<ffi::String>>(mod, kKernelOrderAttr);
  Check(order.size() == mod->functions.size(),
        "Phase 6 kernel_order must cover every Core/slot");
  size_t order_index = 0;
  for (int64_t x = 0; x < grid->x; ++x)
    for (int64_t y = 0; y < grid->y; ++y)
      for (const char *slot : {"trisc", "ncrisc", "brisc"}) {
        auto key = std::make_pair(CoreKey{x, y}, std::string(slot));
        Check(function_names.count(key) &&
                  order[order_index++] == function_names.at(key),
              "Phase 6 kernel_order must follow x/y/TRISC/NCRISC/BRISC order");
      }
  for (const auto &[id, dfb] : dfbs) {
    auto Check = [&, resource = dfb](bool condition,
                                     const std::string &message) {
      if (!condition)
        Fail(ResourceContext(mod, resource) + message);
    };
    Check(reserves.count(id) && producers.count(id),
          "DFB generation is never published");
    Check(!uses[id].empty() || discards.count(id),
          "DFB generation is never consumed");
    Check(releases.count(id), "DFB generation is missing release");
    // Same-owner slot order is the authority for a release. Adding a missing
    // dependency here would silently repair malformed Device IR.
    for (size_t use : uses[id])
      Check(use < releases.at(id),
            "DFB release precedes last consumption (overwrite risk)");
    if (copy_issues.count(id)) {
      Check(copy_completions.count(id), "DFB copy is missing completion wait");
      if (events[copy_issues.at(id)].call->op.same_as(dfb_to_tensor_nd()))
        Check(
            copy_completions.at(id) < releases.at(id),
            "DFB output release before copy completion risks overwritten data");
    }
  }
  if (pipeline.enabled) {
    for (const auto &[group, ids] : pipeline.ordered) {
      int64_t capacity = RequireStaticInteger(dfbs.at(ids.front())->block_count,
                                              "pipeline pool block_count");
      for (size_t ordinal = 0; ordinal < ids.size(); ++ordinal) {
        int64_t id = ids[ordinal];
        if (ordinal > 0) {
          int64_t old = ids[ordinal - 1];
          Check(reserves.at(old) < reserves.at(id) &&
                    producers.at(old) < producers.at(id) &&
                    releases.at(old) < releases.at(id),
                "pipeline pool reserve/publication/release order disagrees "
                "with transaction relation");
          if (!uses.at(old).empty() && !uses.at(id).empty())
            Check(uses.at(old).back() < uses.at(id).front(),
                  "pipeline pool consumption order disagrees with transaction "
                  "relation");
        }
        if (ordinal >= static_cast<size_t>(capacity))
          dependencies[reserves.at(id)].push_back(
              releases.at(ids[ordinal - capacity]));
      }
    }
  }
  for (const auto &[tid, transfer] : transfer_table) {
    Check(sends.count(tid) && receives.count(tid),
          "Pipe transaction producer/consumer matching is not closed");
    Check(send_completions.count(tid) && recv_completions.count(tid),
          "Pipe transaction is missing completion synchronization");
    if (dfbs.at(transfer->source_dfb_id)->consumer_slot == "brisc")
      Check(
          send_completions.at(tid) < releases.at(transfer->source_dfb_id),
          "Pipe source release before send completion risks overwritten data");
    // Destination reservation must exist before transport completes. Reception
    // becomes readable only after the matching source transport has completed.
    dependencies[send_completions.at(tid)].push_back(receives.at(tid));
    dependencies[recv_completions.at(tid)].push_back(send_completions.at(tid));
  }
  for (size_t i = 0; i < events.size(); ++i)
    if (events[i].call->op.same_as(dfb_wait())) {
      int64_t id = RequireStaticInteger(events[i].call->args[0], "DFB wait ID");
      dependencies[i].push_back(producers.at(id));
      // A forwarding producer publishes readiness only after every outgoing
      // transfer completes. The consumer's wait protects its eventual release.
      if (dfbs.at(id)->producer_slot == "ncrisc" &&
          dfbs.at(id)->consumer_slot == "trisc")
        for (int64_t transfer : outgoing[id])
          dependencies[i].push_back(send_completions.at(transfer));
    }
  // Reject cross-Core writes to overlapping global regions; no ordering of
  // unrelated NCRISC streams can establish a deterministic winning write.
  for (size_t i = 0; i < writes.size(); ++i)
    for (size_t j = i + 1; j < writes.size(); ++j) {
      if (writes[i].tensor != writes[j].tensor ||
          writes[i].core == writes[j].core)
        continue;
      bool overlap = true;
      for (size_t axis = 0; axis < writes[i].bounds.size(); axis += 2) {
        int64_t a = RequireStaticInteger(writes[i].bounds[axis], "write start");
        int64_t b = RequireStaticInteger(writes[j].bounds[axis], "write start");
        int64_t na =
            RequireStaticInteger(writes[i].bounds[axis + 1], "write extent");
        int64_t nb =
            RequireStaticInteger(writes[j].bounds[axis + 1], "write extent");
        overlap &= a < b + nb && b < a + na;
      }
      Check(
          !overlap,
          "cross-Core Tensor output overlap risks nondeterministic overwrite");
    }
  std::vector<int> state(events.size(), 0);
  std::function<void(size_t)> visit = [&](size_t event) {
    Check(state[event] != 1,
          "Device transaction dependency cycle (cross-Core/slot deadlock)");
    if (state[event] == 2)
      return;
    state[event] = 1;
    for (size_t dependency : dependencies[event])
      visit(dependency);
    state[event] = 2;
  };
  for (size_t i = 0; i < events.size(); ++i)
    visit(i);
  // Forward logical shapes through communication in graph-independent order.
  std::function<ffi::Array<PrimExpr>(int64_t)> shape_of = [&](int64_t id) {
    if (logical_shapes.count(id))
      return logical_shapes.at(id);
    Check(incoming.count(id), "DFB generation has no logical shape producer");
    int64_t source = transfer_table.at(incoming.at(id))->source_dfb_id;
    auto shape = shape_of(source);
    VerifyTileGrid(shape, dfbs.at(id)->block_shape_in_tiles,
                   "Pipe destination DFB");
    logical_shapes[id] = shape;
    return shape;
  };
  for (const Event &event : events) {
    const Call &call = event.call;
    if (call->op.same_as(dfb_compute()) || call->op.same_as(compute_value())) {
      auto shapes = CheckedShapes(call->annotations.at("tt.input_shapes"),
                                  "tt.input_shapes");
      for (size_t i = 1; i < call->args.size(); ++i)
        Check(IsCompatibleLogicalInput(
                  shape_of(RequireStaticInteger(call->args[i], "input DFB ID")),
                  shapes[i - 1]),
              "dfb_compute input logical shape disagrees with its producer");
    } else if (call->op.same_as(gemm_update()) ||
               call->op.same_as(compute_value_gemm())) {
      for (size_t i = 0; i < 2; ++i) {
        size_t operand = call->op.same_as(compute_value_gemm()) ? i + 1 : i;
        int64_t id =
            RequireStaticInteger(call->args[operand], "GEMM input DFB ID");
        auto shape = shape_of(id);
        Check(shape.size() == 2, "GEMM input producer must have rank two");
        for (size_t axis = 0; axis < 2; ++axis)
          Check(
              RequireStaticInteger(shape[axis], "GEMM input extent") ==
                  RequireStaticInteger(dfbs.at(id)->block_shape_in_tiles[axis],
                                       "GEMM input tile extent") *
                      32,
              "GEMM input logical shape disagrees with its producer");
      }
    } else if (call->op.same_as(dfb_to_tensor_nd())) {
      values.Export(
          RequireStaticInteger(call->args[0], "export DFB ID"),
          tensors.at(RequireStaticInteger(call->args[1], "Tensor ID"))->dtype);
      ffi::Array<PrimExpr> shape;
      for (size_t i = 3; i < call->args.size(); i += 2)
        shape.push_back(call->args[i]);
      shape = LocalTransferShape(
          shape, dfbs.at(RequireStaticInteger(call->args[0], "export DFB ID")));
      Check(ffi::StructuralEqual()(
                shape_of(RequireStaticInteger(call->args[0], "export DFB ID")),
                shape),
            "Tensor export logical shape disagrees with its producer");
    }
  }
  for (const auto &[id, tensor] : tensors) {
    ffi::String expected = effects[id] == 3   ? "inout"
                           : effects[id] == 2 ? "output"
                                              : "input";
    Check(tensor->effect == expected,
          "Tensor effect disagrees with scheduled multicore dataflow");
    Check(effects[id] != 3,
          "Phase 6 cross-Core Tensor inout ordering is unsupported");
  }
}

IRModule VerifyCurrentDeviceModule(IRModule mod) {
  auto diagnostic_expansion =
      mod->GetAttr<Integer>("tt.current_device_table_expansion");
  if (diagnostic_expansion.has_value()) {
    bool stream_operands = mod->GetAttr<Integer>("tt.stream_operands")
                               .value_or(Integer(0))
                               ->value == 1;
    Check(diagnostic_expansion.value()->value == 1 &&
              (stream_operands ||
               mod->attrs->dict.count("tt.expanded_storage_group_descriptors")),
          "invalid current Device diagnostic expansion marker");
    IRModule parameterized = ParameterizeCurrentDeviceTables(mod);
    VerifyCurrentDeviceModule(std::move(parameterized));
    return mod;
  }
  IRModule parameterized = mod;
  for (const auto &[global, base] : parameterized->functions) {
    auto func = base.as<tirx::PrimFunc>();
    if (!func.has_value())
      continue;
    PostOrderVisit(func.value()->body, [&](const ffi::ObjectRef &object) {
      const auto *call = object.as<CallNode>();
      if (!call || (!call->op.same_as(dfb_compute()) &&
                    !call->op.same_as(compute_value())))
        return;
      for (const char *key :
           {"tt.access_maps", "tt.access_ids", "tt.value_access_maps"})
        Check(!call->annotations.count(key),
              std::string("current Device compute must not carry ") + key);
    });
  }
  ffi::String stage =
      RequireModuleAttr<ffi::String>(parameterized, "tt.ir_stage");
  Check(stage == "device", "current Device IR requires tt.ir_stage='device'");
  Check(!parameterized->attrs->dict.count(kDeviceIRVersionAttr),
        "current Device IR must not carry a legacy version discriminator");
  ffi::String target_arch =
      RequireModuleAttr<ffi::String>(parameterized, kTargetArchAttr);
  Check(target_arch == "wormhole_b0" || target_arch == "blackhole",
        "unsupported tt.target_arch `" + std::string(target_arch) + "`");
  CoreCoord launch_grid =
      RequireModuleAttr<CoreCoord>(parameterized, kLaunchGridAttr);
  Check(launch_grid->x > 0 && launch_grid->y > 0 && launch_grid->x <= 256 &&
            launch_grid->y <= 256 && launch_grid->x * launch_grid->y <= 256,
        "shared-grid Device IR requires 1..256 static Cores");
  OperationIdentity operation = RequireModuleAttr<OperationIdentity>(
      parameterized, kOperationIdentityAttr);
  Check(!operation->operation_id.empty(),
        "tt.operation_identity has an empty operation_id");
  ffi::Array<TensorDescriptor> tensors =
      RequireModuleAttr<ffi::Array<TensorDescriptor>>(parameterized,
                                                      kTensorTableAttr);
  TensorTable tensor_table =
      VerifyTensorTable(tensors, launch_grid, true, false);
  CurrentStoragePlan storage =
      VerifyCurrentStoragePlan(parameterized, tensor_table, launch_grid);
  bool stream_operands = parameterized->GetAttr<Integer>("tt.stream_operands")
                             .value_or(Integer(0))
                             ->value == 1;
  mod = ExpandCurrentDeviceTables(mod);
  ffi::Array<DFBDescriptor> dfbs =
      RequireModuleAttr<ffi::Array<DFBDescriptor>>(mod, kDFBTableAttr);
  ffi::Array<PipeDescriptor> transports =
      RequireModuleAttr<ffi::Array<PipeDescriptor>>(mod, "tt.transport_table");
  auto communication = mod->GetAttr<ffi::Array<CommunicationDescriptor>>(
      "tt.communication_table");
  auto steps = RequireModuleAttr<ffi::Array<CommunicationStepDescriptor>>(
      mod, "tt.communication_step_table");
  ffi::Array<ffi::String> order =
      RequireModuleAttr<ffi::Array<ffi::String>>(mod, kKernelOrderAttr);
  Check(order.size() == 3 && mod->functions.size() == 3,
        "current Device IR must contain exactly three shared slot functions");
  const std::array<std::string, 3> suffixes = {"_trisc", "_ncrisc", "_brisc"};
  for (size_t i = 0; i < order.size(); ++i) {
    std::string symbol = order[i];
    Check(symbol.size() >= suffixes[i].size() &&
              symbol.compare(symbol.size() - suffixes[i].size(),
                             suffixes[i].size(), suffixes[i]) == 0,
          "tt.kernel_order must use TRISC/NCRISC/BRISC symbol order");
  }
  DFBTable dfb_table =
      VerifyDFBTable(dfbs, tensor_table, launch_grid, true, false);
  auto accumulators =
      mod->GetAttr<ffi::Array<AccumulatorDescriptor>>(kAccumulatorTableAttr);
  std::map<CurrentComputeKey, int64_t> current_accumulator_ids;
  std::map<int64_t, AccumulatorDescriptor> current_accumulators;
  if (accumulators.has_value()) {
    auto local_ids = RequireModuleAttr<ffi::Array<Integer>>(
        mod, "tt.expanded_accumulator_local_ids");
    Check(local_ids.size() == accumulators.value().size(),
          "expanded accumulator local-ID relation has the wrong size");
    for (size_t index = 0; index < accumulators.value().size(); ++index) {
      const AccumulatorDescriptor &accumulator = accumulators.value()[index];
      Check(accumulator.defined() && accumulator->accumulator_id >= 0,
            "invalid accumulator descriptor");
      Check(
          current_accumulators.emplace(accumulator->accumulator_id, accumulator)
              .second,
          "duplicate accumulator descriptor ID");
      VerifyDomain(accumulator->core_domain, launch_grid,
                   "accumulator core domain");
      Check(
          accumulator->core_domain->end->x ==
                  accumulator->core_domain->begin->x + 1 &&
              accumulator->core_domain->end->y ==
                  accumulator->core_domain->begin->y + 1 &&
              current_accumulator_ids
                  .emplace(CurrentComputeKey{accumulator->core_domain->begin->x,
                                             accumulator->core_domain->begin->y,
                                             local_ids[index]->value},
                           accumulator->accumulator_id)
                  .second,
          "duplicate Core-local accumulator descriptor ID");
    }
  }
  auto values =
      mod->GetAttr<ffi::Array<ComputeValueDescriptor>>(kComputeValueTableAttr);
  std::map<CurrentComputeKey, int64_t> current_value_ids;
  if (values.has_value()) {
    auto local_ids = RequireModuleAttr<ffi::Array<Integer>>(
        mod, "tt.expanded_compute_value_local_ids");
    Check(local_ids.size() == values.value().size(),
          "expanded compute-value local-ID relation has the wrong size");
    for (size_t index = 0; index < values.value().size(); ++index) {
      const ComputeValueDescriptor &value = values.value()[index];
      Check(value.defined() && value->value_id >= 0 &&
                value->buffer.defined() &&
                value->value_kind == "compute_fragment" &&
                value->owner_slot == "trisc" &&
                value->dtype == value->buffer->dtype &&
                ffi::StructuralEqual()(value->logical_shape,
                                       value->buffer->shape) &&
                value->compute_tile_shape.size() == 2,
            "invalid compute-local value descriptor");
      VerifyDomain(value->core_domain, launch_grid,
                   "compute value core domain");
      Check(value->core_domain->end->x == value->core_domain->begin->x + 1 &&
                value->core_domain->end->y ==
                    value->core_domain->begin->y + 1 &&
                current_value_ids
                    .emplace(CurrentComputeKey{value->core_domain->begin->x,
                                               value->core_domain->begin->y,
                                               local_ids[index]->value},
                             value->value_id)
                    .second,
            "duplicate Core-local compute value descriptor ID");
    }
  }
  std::map<std::pair<int64_t, int64_t>, PipeDescriptor> transport_records;
  for (const PipeDescriptor &transport : transports) {
    Check(transport.defined() && transport->pipe_net_id >= 0 &&
              transport->communication_id >= 0 && transport->record_index >= 0,
          "transport descriptor has an invalid identity");
    VerifyCoord(transport->src_coord, launch_grid, "transport source");
    VerifyDomain(CoreDomain(transport->dst_begin, transport->dst_end),
                 launch_grid, "transport destination");
    Check((transport->contract == "point_to_point" ||
           transport->contract == "collective") &&
              dfb_table.count(transport->payload_dfb_id),
          "transport descriptor has an invalid contract or payload DFB");
    Check(communication.has_value() &&
              transport->communication_id <
                  static_cast<int64_t>(communication.value().size()),
          "transport descriptor references a missing parent communication");
    Check(transport_records
              .emplace(std::make_pair(transport->pipe_net_id,
                                      transport->record_index),
                       transport)
              .second,
          "transport descriptor has a duplicate Pipe record identity");
  }
  VerifyFunctions(mod, target_arch, launch_grid, tensor_table, nullptr,
                  /*general=*/true, /*multicore=*/false,
                  /*require_span=*/false, /*current=*/true);

  std::unordered_set<std::string> ordered_symbols;
  for (const ffi::String &symbol : order)
    Check(ordered_symbols.insert(symbol).second,
          "tt.kernel_order contains duplicate symbols");
  std::unordered_set<std::string> actual_symbols;
  struct LocalLifecycle {
    bool reserved{false};
    bool produced{false};
    bool waited{false};
    bool released{false};
    ffi::String reserve_slot;
    ffi::String producer_slot;
    ffi::String consumer_slot;
    int64_t last_use{-1};
    int64_t release_order{-1};
  };
  using StreamKey = std::tuple<int64_t, int64_t, int64_t>;
  using LocalKey = std::tuple<int64_t, int64_t, int64_t, int64_t>;
  using SlotKey = std::tuple<int64_t, int64_t, std::string>;
  std::map<LocalKey, LocalLifecycle> lifecycles;
  std::map<int64_t, std::set<CoreKey>> resource_cores;
  std::map<StreamKey, int64_t> reservation_transactions;
  std::map<LocalKey, int64_t> reserve_events;
  std::map<LocalKey, int64_t> release_events;
  std::map<LocalKey, ffi::Array<PrimExpr>> tensor_load_shapes;
  std::map<LocalKey, std::pair<int64_t, bool>> copy_issues;
  std::map<LocalKey, int64_t> copy_completions;
  std::map<LocalKey, std::vector<int64_t>> forwarding_completions;
  struct InputShapeUse {
    LocalKey key;
    ffi::Array<PrimExpr> shape;
  };
  std::vector<InputShapeUse> input_shape_uses;
  std::map<LocalKey, int64_t> producer_events;
  std::map<LocalKey, std::vector<int64_t>> wait_events;
  std::map<SlotKey, int64_t> last_slot_event;
  std::map<int64_t, int> accumulator_state;
  std::map<int64_t, int64_t> accumulator_k_tiles;
  std::vector<std::vector<int64_t>> event_successors;
  ComputeValueVerifier compute_values(mod);
  std::vector<std::pair<int64_t, DataType>> value_exports;
  std::vector<int64_t> step_ref_counts(steps.size(), 0);
  std::vector<int64_t> step_ref_dynamic_counts(steps.size(), 0);
  std::function<void(const Stmt &, std::vector<Call> *, std::vector<int64_t> *,
                     int64_t, int64_t, const ffi::Map<Var, PrimExpr> &,
                     int64_t)>
      collect_calls = [&](const Stmt &stmt, std::vector<Call> *calls,
                          std::vector<int64_t> *multipliers, int64_t core_x,
                          int64_t core_y,
                          const ffi::Map<Var, PrimExpr> &bindings,
                          int64_t multiplier) {
        if (const auto *seq = stmt.as<SeqStmtNode>()) {
          for (const Stmt &child : seq->seq)
            collect_calls(child, calls, multipliers, core_x, core_y, bindings,
                          multiplier);
        } else if (const auto *loop = stmt.as<ForNode>()) {
          if (std::string(loop->loop_var->name_hint).rfind("tt_compact_", 0) ==
              0) {
            int64_t minimum =
                RequireStaticInteger(loop->min, "compact loop minimum");
            int64_t extent =
                RequireStaticInteger(loop->extent, "compact loop extent");
            for (int64_t offset = 0; offset < extent; ++offset) {
              PrimExpr iteration =
                  make_const(loop->loop_var.dtype(), minimum + offset);
              ffi::Map<Var, PrimExpr> nested = bindings;
              nested.Set(loop->loop_var, iteration);
              collect_calls(
                  Substitute(loop->body, {{loop->loop_var, iteration}}), calls,
                  multipliers, core_x, core_y, nested, multiplier);
            }
          } else {
            int64_t extent =
                RequireStaticInteger(loop->extent, "Device loop extent");
            collect_calls(
                loop->body, calls, multipliers, core_x, core_y, bindings,
                CheckedProduct(multiplier, extent,
                               "Device operation dynamic multiplicity"));
          }
        } else if (const auto *branch = stmt.as<IfThenElseNode>()) {
          CoreCoordinateSubstituter bind_core(core_x, core_y);
          PrimExpr condition = arith::Analyzer().Simplify(
              bind_core.VisitExpr(Substitute(branch->condition, bindings)));
          if (is_one(condition)) {
            collect_calls(branch->then_case, calls, multipliers, core_x, core_y,
                          bindings, multiplier);
          } else if (is_zero(condition)) {
            if (branch->else_case.has_value())
              collect_calls(branch->else_case.value(), calls, multipliers,
                            core_x, core_y, bindings, multiplier);
          } else {
            collect_calls(branch->then_case, calls, multipliers, core_x, core_y,
                          bindings, multiplier);
            if (branch->else_case.has_value())
              collect_calls(branch->else_case.value(), calls, multipliers,
                            core_x, core_y, bindings, multiplier);
          }
        } else if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
          collect_calls(realize->block->body, calls, multipliers, core_x,
                        core_y, bindings, multiplier);
        } else if (const auto *evaluate = stmt.as<EvaluateNode>()) {
          if (const auto *call = evaluate->value.as<CallNode>()) {
            CoreCoordinateSubstituter bind_core(core_x, core_y);
            Call concrete =
                Downcast<Call>(bind_core.VisitExpr(ffi::GetRef<Call>(call)));
            auto annotations = concrete->annotations;
            auto concretize = [&](const PrimExpr &expr) {
              return arith::Analyzer().Simplify(
                  bind_core.VisitExpr(Substitute(expr, bindings)));
            };
            auto value_inputs = annotations.Get("tt.value_inputs");
            if (value_inputs.has_value()) {
              ffi::Array<PrimExpr> rewritten;
              for (const PrimExpr &value :
                   Downcast<ffi::Array<PrimExpr>>(value_inputs.value()))
                rewritten.push_back(concretize(value));
              annotations.Set("tt.value_inputs", rewritten);
            }
            auto expression = annotations.Get("tt.expression");
            if (expression.has_value())
              annotations.Set("tt.expression", concretize(Downcast<PrimExpr>(
                                                   expression.value())));
            auto step_refs = annotations.Get("tt_communication_step_refs");
            if (step_refs.has_value()) {
              ffi::Array<PrimExpr> rewritten;
              for (const PrimExpr &ref :
                   Downcast<ffi::Array<PrimExpr>>(step_refs.value()))
                rewritten.push_back(concretize(ref));
              annotations.Set("tt_communication_step_refs", rewritten);
            }
            if (!ffi::StructuralEqual()(annotations, concrete->annotations))
              concrete = Call(concrete->dtype, concrete->op, concrete->args,
                              annotations, concrete->span);
            PostOrderVisit(concrete, [&](const ffi::ObjectRef &object) {
              if (const auto *nested = object.as<CallNode>()) {
                calls->push_back(ffi::GetRef<Call>(nested));
                multipliers->push_back(multiplier);
              }
            });
          }
        }
      };
  for (const auto &[global, base] : mod->functions) {
    PrimFunc func = Downcast<PrimFunc>(base);
    compute_values.BindFunction(func);
    BoundVars bound_loop_vars;
    PostOrderVisit(func->body, [&](const ffi::ObjectRef &object) {
      if (const auto *loop = object.as<ForNode>())
        bound_loop_vars.insert(loop->loop_var.get());
    });
    ffi::String symbol =
        func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).value();
    actual_symbols.insert(symbol);
    CoreDomain domain = func->GetAttr<CoreDomain>(kCoreDomainAttr).value();
    Check(domain->begin->x == 0 && domain->begin->y == 0 &&
              domain->end->x == launch_grid->x &&
              domain->end->y == launch_grid->y,
          "shared slot tt.core_domain must equal the complete launch grid");
    PostOrderVisit(func->body, [&](const ffi::ObjectRef &object) {
      const auto *node = object.as<CallNode>();
      if (!node)
        return;
      Call call = ffi::GetRef<Call>(node);
      const auto *op = call->op.as<OpNode>();
      std::string name = op ? std::string(op->name) : "";
      Check(name != "tl.tt.transfer" && name != "tl.tt.all_gather" &&
                name != "tl.tt.all_reduce" && name.rfind("tl.tt.pipe_", 0) != 0,
            "Device body retains a frontend communication operation");
      auto check_dfb = [&](size_t index) {
        Check(index < call->args.size(),
              "Device operation DFB operand missing");
        const PrimExpr &operand = call->args[index];
        const int64_t *constant = as_const_int(operand);
        if (!constant) {
          Check(operand.dtype().is_int() && operand.dtype().lanes() == 1 &&
                    SideEffect(operand) <= CallEffectKind::kPure,
                "loop-carried Device DFB ID must be a pure scalar integer");
          return;
        }
        int64_t id = *constant;
        Check(dfb_table.count(id),
              "Device operation references missing DFB " + std::to_string(id));
      };
      if (call->op.same_as(dfb_reserve()) || call->op.same_as(dfb_wait()) ||
          call->op.same_as(dfb_release()) ||
          call->op.same_as(dfb_copy_wait())) {
        Check(call->args.size() == 1,
              "current Device DFB lifecycle operation requires one stream "
              "operand");
        check_dfb(0);
      } else if (call->op.same_as(dfb_load())) {
        Check(call->args.size() == 1,
              "current Device dfb_load requires one stream operand");
        check_dfb(0);
      } else if (call->op.same_as(tensor_to_dfb_nd()) ||
                 call->op.same_as(dfb_pipe_send()) ||
                 call->op.same_as(dfb_pipe_recv()) ||
                 call->op.same_as(dfb_pipe_wait()))
        check_dfb(1);
      else if (call->op.same_as(dfb_pipe_send_record()) ||
               call->op.same_as(dfb_pipe_recv_record()) ||
               call->op.same_as(dfb_pipe_wait_record()))
        check_dfb(3);
      else if (call->op.same_as(dfb_to_tensor_nd()) ||
               call->op.same_as(accumulator_materialize()) ||
               call->op.same_as(compute_value_store()))
        check_dfb(call->op.same_as(dfb_to_tensor_nd()) ? 0 : 1);
      else if (call->op.same_as(gemm_update())) {
        check_dfb(0);
        check_dfb(1);
      } else if (call->op.same_as(dfb_local_copy()) ||
                 call->op.same_as(dfb_gather_copy())) {
        check_dfb(0);
        check_dfb(1);
      } else if (call->op.same_as(dfb_collective_reduce())) {
        check_dfb(0);
        check_dfb(1);
        check_dfb(2);
      }
      if (call->op.same_as(tensor_to_dfb_nd()) ||
          call->op.same_as(dfb_to_tensor_nd())) {
        bool load = call->op.same_as(tensor_to_dfb_nd());
        size_t tensor_operand = load ? 0 : 1;
        size_t dfb_operand = load ? 1 : 0;
        const PrimExpr &tensor_expr = call->args[tensor_operand];
        const int64_t *static_tensor = as_const_int(tensor_expr);
        const int64_t *static_dfb = as_const_int(call->args[dfb_operand]);
        if (!static_tensor || !static_dfb) {
          Check(tensor_expr.dtype().is_int() &&
                    tensor_expr.dtype().lanes() == 1 &&
                    SideEffect(tensor_expr) <= CallEffectKind::kPure,
                "parameterized Tensor ABI index must be a pure scalar integer");
          return;
        }
        int64_t tensor_id = *static_tensor;
        int64_t dfb_id = *static_dfb;
        Check(tensor_table.count(tensor_id) && dfb_table.count(dfb_id),
              "Tensor transfer references missing Tensor/DFB");
        const TensorDescriptor &tensor = tensor_table.at(tensor_id);
        const DFBDescriptor &dfb = dfb_table.at(dfb_id);
        if (!stream_operands)
          Check(dfb->tensor_backing.has_value() &&
                    dfb->tensor_backing.value()->global_arg_index == tensor_id,
                "Tensor transfer disagrees with DFB Tensor backing");
        Check(call->args.size() == 2 + 2 * tensor->shape.size(),
              "Tensor transfer rank/arity mismatch");
        Check(tensor->shape.size() >= dfb->block_shape_in_tiles.size(),
              "Tensor transfer rank is smaller than DFB rank");
        size_t leading =
            tensor->shape.size() - dfb->block_shape_in_tiles.size();
        for (size_t axis = 0; axis < leading; ++axis)
          Check(is_one(call->args[3 + 2 * axis]),
                "Tensor transfer may remove only leading singleton slice axes");
        for (size_t axis = 0; axis < tensor->shape.size(); ++axis) {
          const PrimExpr &start = call->args[2 + 2 * axis];
          const PrimExpr &extent = call->args[3 + 2 * axis];
          if (as_const_int(start) && as_const_int(extent) &&
              as_const_int(tensor->shape[axis]))
            Check(*as_const_int(start) >= 0 && *as_const_int(extent) > 0 &&
                      *as_const_int(start) + *as_const_int(extent) <=
                          *as_const_int(tensor->shape[axis]),
                  "Tensor transfer slice is out of bounds");
        }
      }
    });
    bool compute =
        func->GetAttr<ffi::String>(kKernelSlotAttr).value() == "trisc";
    auto requirements =
        func->GetAttr<ComputeRequirements>(kComputeRequirementsAttr);
    Check(!compute || requirements.has_value(),
          "shared TRISC is missing tt.compute_requirements");
    Check(compute || !requirements.has_value(),
          "data-movement slot carries tt.compute_requirements");
    if (compute)
      Check(ffi::StructuralEqual()(requirements.value(),
                                   DeriveComputeRequirements(mod, func)),
            "tt.compute_requirements disagrees with actual Device operations "
            "across shared TRISC Core paths");
    auto work_domain = [&](const SBlock &block) {
      auto domain = block->annotations.Get("tt.core_domain");
      if (domain.has_value())
        return Downcast<CoreDomain>(domain.value());
      int64_t x = Downcast<Integer>(block->annotations.at("tt.core_x"))->value;
      int64_t y = Downcast<Integer>(block->annotations.at("tt.core_y"))->value;
      return CoreDomain(CoreCoord(x, y), CoreCoord(x + 1, y + 1));
    };
    std::set<CoreKey> dispatched_cores;
    std::function<void(const Stmt &)> verify_dispatch = [&](const Stmt &stmt) {
      if (const auto *sequence = stmt.as<SeqStmtNode>()) {
        for (const Stmt &child : sequence->seq)
          verify_dispatch(child);
        return;
      }
      ffi::Optional<PrimExpr> predicate = std::nullopt;
      Stmt work = stmt;
      if (const auto *branch = stmt.as<IfThenElseNode>()) {
        Check(!branch->else_case.has_value(),
              "Core work dispatch must not have an else branch");
        predicate = branch->condition;
        work = branch->then_case;
      }
      const auto *realize = work.as<SBlockRealizeNode>();
      if (!realize) {
        Check(IsCanonicalNoOp(work),
              "shared slot body must contain only Core work dispatch or idle");
        return;
      }
      Check(realize->block->name_hint == "tt_device_core_work",
            "shared slot dispatch must realize tt_device_core_work");
      CoreDomain domain = work_domain(realize->block);
      VerifyDomain(domain, launch_grid, "Core work domain");
      for (int64_t x = 0; x < launch_grid->x; ++x) {
        for (int64_t y = 0; y < launch_grid->y; ++y) {
          bool expected = x >= domain->begin->x && x < domain->end->x &&
                          y >= domain->begin->y && y < domain->end->y;
          bool actual = true;
          if (predicate.has_value()) {
            CoreCoordinateSubstituter bind_core(x, y);
            PrimExpr value = bind_core(predicate.value());
            const int64_t *constant = as_const_int(value);
            Check(constant != nullptr,
                  "Core work predicate must simplify for every static Core");
            actual = *constant != 0;
          }
          Check(actual == expected,
                "Core work predicate disagrees with tt.core_domain");
          if (actual)
            Check(dispatched_cores.insert({x, y}).second,
                  "shared slot has overlapping Core work domains");
        }
      }
    };
    Stmt dispatch_body = func->body;
    if (const auto *root = dispatch_body.as<SBlockRealizeNode>();
        root && root->block->name_hint == "root") {
      Check(root->iter_values.empty() && is_one(root->predicate) &&
                root->block->iter_vars.empty() && root->block->reads.empty() &&
                root->block->writes.empty() &&
                root->block->annotations.empty() &&
                root->block->alloc_buffers.empty() &&
                root->block->match_buffers.empty() &&
                !root->block->init.defined(),
            "TVMScript root block must be an unannotated structural wrapper");
      dispatch_body = root->block->body;
    }
    verify_dispatch(dispatch_body);
    std::function<void(const Stmt &, ffi::Optional<PrimExpr>)>
        verify_stream_dispatch = [&](const Stmt &stmt,
                                     ffi::Optional<PrimExpr> predicate) {
          if (const auto *sequence = stmt.as<SeqStmtNode>()) {
            Check(!predicate.has_value(),
                  "Device stream predicate must guard one stream work region");
            for (const Stmt &child : sequence->seq)
              verify_stream_dispatch(child, std::nullopt);
            return;
          }
          if (const auto *loop = stmt.as<ForNode>()) {
            Check(!predicate.has_value(),
                  "Device stream predicate cannot guard a loop");
            verify_stream_dispatch(loop->body, std::nullopt);
            return;
          }
          if (const auto *branch = stmt.as<IfThenElseNode>()) {
            Check(!predicate.has_value() && !branch->else_case.has_value(),
                  "Device stream dispatch requires one predicate without else");
            verify_stream_dispatch(branch->then_case, branch->condition);
            return;
          }
          if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
            if (realize->block->name_hint != "tt_device_stream_work") {
              Check(!predicate.has_value(),
                    "Device stream predicate guards a non-stream region");
              verify_stream_dispatch(realize->block->body, std::nullopt);
              return;
            }
            CoreDomain domain = Downcast<CoreDomain>(
                realize->block->annotations.at("tt.core_domain"));
            for (int64_t x = 0; x < launch_grid->x; ++x) {
              for (int64_t y = 0; y < launch_grid->y; ++y) {
                bool expected = x >= domain->begin->x && x < domain->end->x &&
                                y >= domain->begin->y && y < domain->end->y;
                bool actual = true;
                if (predicate.has_value()) {
                  CoreCoordinateSubstituter bind_core(x, y);
                  PrimExpr condition = arith::Analyzer().Simplify(
                      bind_core.VisitExpr(predicate.value()));
                  const int64_t *constant = as_const_int(condition);
                  Check(constant,
                        "Device stream predicate must simplify on every Core");
                  actual = *constant != 0;
                }
                Check(actual == expected,
                      "Device stream predicate disagrees with core_domain");
              }
            }
            return;
          }
          Check(!predicate.has_value(),
                "Device stream predicate guards a non-stream statement");
        };
    PostOrderVisit(func->body, [&](const ffi::ObjectRef &object) {
      const auto *block = object.as<SBlockNode>();
      if (!block || block->name_hint != "tt_device_core_work")
        return;
      bool has_stream_work = false;
      PostOrderVisit(block->body, [&](const ffi::ObjectRef &nested) {
        const auto *nested_block = nested.as<SBlockNode>();
        has_stream_work |=
            nested_block && nested_block->name_hint == "tt_device_stream_work";
      });
      if (has_stream_work)
        verify_stream_dispatch(block->body, std::nullopt);
    });
    PostOrderVisit(func->body, [&](const ffi::ObjectRef &object) {
      const auto *block = object.as<SBlockNode>();
      if (!block)
        return;
      if (block->name_hint == "tt_device_stream_work") {
        auto annotation_integer = [&](const ffi::Any &value,
                                      const char *label) {
          if (auto scalar = value.as<int64_t>())
            return scalar.value();
          if (auto integer = value.as<Integer>())
            return integer.value()->value;
          Fail(std::string(label) + " must be an integer");
          return int64_t{0};
        };
        auto domain = block->annotations.Get("tt.core_domain");
        auto owner = block->annotations.Get("tt.owner_slot");
        auto work_group = block->annotations.Get("tt.work_group");
        Check(domain.has_value() && owner.has_value() && work_group.has_value(),
              "Device stream work requires core_domain, owner_slot and "
              "work_group annotations");
        VerifyDomain(Downcast<CoreDomain>(domain.value()), launch_grid,
                     "Device stream work domain");
        auto owner_string = owner.value().as<ffi::String>();
        Check(owner_string.has_value() &&
                  owner_string.value() ==
                      func->GetAttr<ffi::String>(kKernelSlotAttr).value(),
              "Device stream work owner disagrees with containing slot");
        int64_t work_group_id =
            annotation_integer(work_group.value(), "Device stream work group");
        Check(work_group_id >= 0,
              "Device stream work group must be a nonnegative integer");
        auto role = block->annotations.Get("tt.communication_role");
        auto sites = block->annotations.Get("tt.communication_sites");
        Check(role.has_value() == sites.has_value(),
              "Device communication role and site annotations must coexist");
        if (role.has_value()) {
          auto role_string = role.value().as<ffi::String>();
          Check(role_string.has_value(),
                "Device communication role must be a string");
          ffi::String role_name = role_string.value();
          Check(role_name == "source" || role_name == "destination",
                "Device communication role must be source or destination");
          auto site_array = sites.value().as<ffi::Array<ffi::Any>>();
          Check(site_array.has_value() && !site_array.value().empty(),
                "Device communication role requires at least one site");
          std::set<int64_t> allowed_sites;
          for (const ffi::Any &site_value : site_array.value()) {
            int64_t site =
                annotation_integer(site_value, "Device communication site");
            Check(site >= 0 && allowed_sites.insert(site).second,
                  "Device communication sites must be unique nonnegative IDs");
          }
          bool found_pipe = false;
          PostOrderVisit(block->body, [&](const ffi::ObjectRef &nested) {
            const auto *call = nested.as<CallNode>();
            if (!call || (!call->op.same_as(dfb_pipe_send()) &&
                          !call->op.same_as(dfb_pipe_recv()) &&
                          !call->op.same_as(dfb_pipe_wait()) &&
                          !call->op.same_as(dfb_pipe_send_record()) &&
                          !call->op.same_as(dfb_pipe_recv_record()) &&
                          !call->op.same_as(dfb_pipe_wait_record())))
              return;
            found_pipe = true;
            if (role_name == "source")
              Check(!call->op.same_as(dfb_pipe_recv()) &&
                        !call->op.same_as(dfb_pipe_recv_record()),
                    "source communication role contains a receive");
            else
              Check(!call->op.same_as(dfb_pipe_send()) &&
                        !call->op.same_as(dfb_pipe_send_record()),
                    "destination communication role contains a send");
            auto call_role = call->annotations.Get("tt.communication_role");
            auto call_site = call->annotations.Get("tt.communication_site");
            auto call_role_string = call_role.has_value()
                                        ? call_role.value().as<StringImm>()
                                        : ffi::Optional<StringImm>();
            Check(call_role_string.has_value() && call_site.has_value() &&
                      call_role_string.value()->value == role_name,
                  "Pipe operation role annotation disagrees with its stream "
                  "work region");
            int64_t site =
                annotation_integer(call_site.value(), "Pipe operation site");
            Check(allowed_sites.count(site),
                  "Pipe operation site is outside its stream work region");
          });
          Check(found_pipe,
                "Device communication role region contains no Pipe operation");
        }
        return;
      }
      if (block->name_hint != "tt_device_core_work")
        return;
      CoreDomain instance_domain;
      auto domain_annotation = block->annotations.Get("tt.core_domain");
      if (domain_annotation.has_value()) {
        instance_domain = Downcast<CoreDomain>(domain_annotation.value());
      } else {
        int64_t x =
            Downcast<Integer>(block->annotations.at("tt.core_x"))->value;
        int64_t y =
            Downcast<Integer>(block->annotations.at("tt.core_y"))->value;
        instance_domain = CoreDomain(CoreCoord(x, y), CoreCoord(x + 1, y + 1));
      }
      VerifyDomain(instance_domain, launch_grid, "Core work domain");
      ffi::String owner =
          Downcast<ffi::String>(block->annotations.at("tt.owner_slot"));
      Check(owner == func->GetAttr<ffi::String>(kKernelSlotAttr).value(),
            "Core work owner disagrees with containing slot");
      for (int64_t x = instance_domain->begin->x; x < instance_domain->end->x;
           ++x) {
        for (int64_t y = instance_domain->begin->y; y < instance_domain->end->y;
             ++y) {
          CoreCoordinateSubstituter bind_core(x, y);
          Stmt core_body = bind_core(block->body);
          if (compute_values.Enabled() && owner == "trisc")
            compute_values.EnterCoreInstance(x, y);
          std::vector<Call> calls;
          std::vector<int64_t> call_multipliers;
          collect_calls(core_body, &calls, &call_multipliers, x, y, {}, 1);
          Check(calls.size() == call_multipliers.size(),
                "Device call multiplicity analysis is incomplete");
          std::map<int64_t, int64_t> block_reservations;
          std::function<void(const Stmt &, int64_t)> count_reservations =
              [&](const Stmt &stmt, int64_t multiplier) {
                if (const auto *seq = stmt.as<SeqStmtNode>()) {
                  for (const Stmt &child : seq->seq)
                    count_reservations(child, multiplier);
                } else if (const auto *loop = stmt.as<ForNode>()) {
                  Check(loop->HasTrivialStep(),
                        "current Device structured loop requires unit step");
                  int64_t extent =
                      RequireStaticInteger(loop->extent, "Device loop extent");
                  if (std::string(loop->loop_var->name_hint)
                          .rfind("tt_compact_", 0) == 0) {
                    int64_t minimum =
                        RequireStaticInteger(loop->min, "compact loop minimum");
                    for (int64_t offset = 0; offset < extent; ++offset)
                      count_reservations(
                          Substitute(loop->body,
                                     {{loop->loop_var,
                                       make_const(loop->loop_var.dtype(),
                                                  minimum + offset)}}),
                          multiplier);
                  } else {
                    count_reservations(
                        loop->body,
                        CheckedProduct(multiplier, extent,
                                       "Device DFB transaction count"));
                  }
                } else if (const auto *branch = stmt.as<IfThenElseNode>()) {
                  PrimExpr condition =
                      arith::Analyzer().Simplify(branch->condition);
                  if (is_one(condition)) {
                    count_reservations(branch->then_case, multiplier);
                  } else if (is_zero(condition)) {
                    if (branch->else_case.has_value())
                      count_reservations(branch->else_case.value(), multiplier);
                  } else {
                    count_reservations(branch->then_case, multiplier);
                    if (branch->else_case.has_value())
                      count_reservations(branch->else_case.value(), multiplier);
                  }
                } else if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
                  count_reservations(realize->block->body, multiplier);
                } else if (const auto *evaluate = stmt.as<EvaluateNode>()) {
                  const auto *call = evaluate->value.as<CallNode>();
                  if (call && call->op.same_as(dfb_reserve())) {
                    PrimExpr resource_expr =
                        arith::Analyzer().Simplify(call->args[0]);
                    int64_t resource = RequireStaticInteger(
                        resource_expr, "reserved DFB transaction identity");
                    Check(block_reservations[resource] <=
                              std::numeric_limits<int64_t>::max() - multiplier,
                          "Device DFB transaction count overflows");
                    block_reservations[resource] += multiplier;
                  }
                }
              };
          count_reservations(core_body, 1);
          for (const auto &[resource, transactions] : block_reservations) {
            Check(dfb_table.count(resource),
                  "DFB reservation references a missing logical stream");
            int64_t expected_transactions = RequireStaticInteger(
                dfb_table.at(resource)->transaction_count_or_loop_relation,
                "logical stream transaction count");
            Check(transactions == expected_transactions,
                  "DFB stream " + std::to_string(resource) + " has " +
                      std::to_string(transactions) +
                      " reservations but its structured loop relation "
                      "requires " +
                      std::to_string(expected_transactions) +
                      "; transaction count disagrees");
            Check(reservation_transactions
                      .emplace(StreamKey{x, y, resource}, transactions)
                      .second,
                  "DFB stream is reserved by multiple Core work blocks");
          }
          int64_t ordinal = 0;
          std::map<std::pair<int64_t, int64_t>, bool> sent_transfers;
          std::set<std::pair<int64_t, int64_t>> received_transfers;
          using RecordTransferKey =
              std::tuple<int64_t, int64_t, int64_t, int64_t>;
          std::map<RecordTransferKey, bool> sent_records;
          std::set<RecordTransferKey> received_records;
          std::map<int64_t, int64_t> next_producer_transaction;
          std::map<int64_t, int64_t> next_consumer_transaction;
          std::map<int64_t, int64_t> active_producer_transaction;
          std::map<int64_t, std::vector<int64_t>> active_consumer_transactions;
          auto id = [&](const Call &call, size_t index) {
            Check(index < call->args.size(),
                  "Core work DFB operand is missing");
            PrimExpr simplified = arith::Analyzer().Simplify(call->args[index]);
            Check(as_const_int(simplified), "Core work DFB IDs must be static "
                                            "inside preserved outer loops");
            return *as_const_int(simplified);
          };
          auto state = [&](const LocalKey &key) -> LocalLifecycle & {
            int64_t resource = std::get<2>(key);
            Check(dfb_table.count(resource),
                  "Core work references missing DFB stream " +
                      std::to_string(resource));
            resource_cores[resource].insert({x, y});
            return lifecycles[key];
          };
          auto begin_producer = [&](int64_t resource) {
            auto active = active_producer_transaction.find(resource);
            if (active != active_producer_transaction.end()) {
              LocalKey previous{x, y, resource, active->second};
              Check(state(previous).produced,
                    "DFB stream has nested producer reservations");
              active_producer_transaction.erase(active);
            }
            int64_t transaction = next_producer_transaction[resource]++;
            active_producer_transaction.emplace(resource, transaction);
            return LocalKey{x, y, resource, transaction};
          };
          auto producer_key = [&](int64_t resource) {
            Check(active_producer_transaction.count(resource),
                  "DFB producer operation has no active reservation");
            return LocalKey{x, y, resource,
                            active_producer_transaction.at(resource)};
          };
          auto begin_consumer = [&](int64_t resource) {
            int64_t transaction = next_consumer_transaction[resource]++;
            auto &transactions = active_consumer_transactions[resource];
            if (transactions.empty() || transactions.back() != transaction)
              transactions.push_back(transaction);
            if (compute_values.Enabled())
              compute_values.AcquireDFB(resource, transaction);
            return LocalKey{x, y, resource, transaction};
          };
          auto consumer_key = [&](int64_t resource) {
            auto consumer = active_consumer_transactions.find(resource);
            if (consumer != active_consumer_transactions.end() &&
                !consumer->second.empty())
              return LocalKey{x, y, resource, consumer->second.back()};
            Check(false, "DFB consumer operation has no active acquisition");
            return LocalKey{x, y, resource, 0};
          };
          auto release_key = [&](int64_t resource) {
            auto consumer = active_consumer_transactions.find(resource);
            Check(consumer != active_consumer_transactions.end() &&
                      !consumer->second.empty(),
                  "DFB consumer operation has no active acquisition");
            return LocalKey{x, y, resource, consumer->second.front()};
          };
          auto use = [&](int64_t resource) {
            LocalKey key = consumer_key(resource);
            auto &entry = state(key);
            if (entry.consumer_slot.empty())
              entry.consumer_slot = owner;
            Check(entry.consumer_slot == owner,
                  "one Core-local DFB transaction has consumers in multiple "
                  "slots");
            entry.last_use = ordinal;
          };
          auto produce = [&](int64_t resource, int64_t event,
                             bool partial = false) {
            LocalKey key = producer_key(resource);
            auto &entry = state(key);
            if (entry.producer_slot.empty())
              entry.producer_slot = owner;
            Check(entry.producer_slot == owner,
                  "one Core-local DFB transaction has multiple producer slots");
            auto [producer, inserted] = producer_events.emplace(key, event);
            Check(inserted || partial,
                  "DFB transaction is published more than once");
            if (partial)
              producer->second = event;
            entry.produced = true;
            const DFBDescriptor &descriptor = dfb_table.at(resource);
            if (descriptor->producer_slot == owner &&
                descriptor->consumer_slot == owner) {
              auto &transactions = active_consumer_transactions[resource];
              if (transactions.empty() ||
                  transactions.back() != std::get<3>(key))
                transactions.push_back(std::get<3>(key));
              if (compute_values.Enabled())
                compute_values.AcquireDFB(resource, std::get<3>(key));
            }
          };
          auto verify_step_refs = [&](const Call &call,
                                      int64_t dynamic_multiplier) {
            bool pipe_send = call->op.same_as(dfb_pipe_send());
            bool pipe_recv = call->op.same_as(dfb_pipe_recv());
            bool pipe_wait = call->op.same_as(dfb_pipe_wait());
            bool record_send = call->op.same_as(dfb_pipe_send_record());
            bool record_recv = call->op.same_as(dfb_pipe_recv_record());
            bool record_wait = call->op.same_as(dfb_pipe_wait_record());
            bool local_copy = call->op.same_as(dfb_local_copy());
            bool gather_copy = call->op.same_as(dfb_gather_copy());
            bool local_reduce = call->op.same_as(dfb_collective_reduce());
            bool communication_op = pipe_send || pipe_recv || pipe_wait ||
                                    record_send || record_recv || record_wait ||
                                    local_copy || gather_copy || local_reduce;
            auto attribute =
                call->annotations.Get("tt_communication_step_refs");
            Check(communication_op == attribute.has_value(),
                  communication_op
                      ? "communication operation is missing step refs"
                      : "non-communication operation carries step refs");
            if (!communication_op)
              return;
            auto refs = attribute.value().as<ffi::Array<PrimExpr>>();
            Check(
                refs.has_value() && !refs.value().empty(),
                "communication step refs must be a nonempty expression array");
            Check(((pipe_wait || record_wait) && refs.value().size() <= 2) ||
                      (!pipe_wait && !record_wait && refs.value().size() == 1),
                  "communication operation has the wrong number of step refs");
            std::set<int64_t> unique;
            for (size_t ref_index = 0; ref_index < refs.value().size();
                 ++ref_index) {
              int64_t ref = RequireStaticInteger(refs.value()[ref_index],
                                                 "communication step ref");
              Check(ref >= 0 && ref < static_cast<int64_t>(steps.size()) &&
                        unique.insert(ref).second,
                    "communication operation has an invalid or duplicate step "
                    "ref");
              const CommunicationStepDescriptor &step = steps[ref];
              Check(step.defined() && step->step_id == ref &&
                        step->owner_slot == owner,
                    "communication step ref has the wrong identity or owner");
              ++step_ref_counts[ref];
              Check(step_ref_dynamic_counts[ref] <=
                        std::numeric_limits<int64_t>::max() -
                            dynamic_multiplier,
                    "communication step dynamic multiplicity overflows");
              step_ref_dynamic_counts[ref] += dynamic_multiplier;
              auto coord_is = [&](const ffi::Optional<CoreCoord> &coord) {
                return coord.has_value() && coord.value()->x == x &&
                       coord.value()->y == y;
              };
              if (record_send || record_recv || record_wait) {
                int64_t pipe_net = id(call, 0);
                int64_t record = id(call, 1);
                int64_t occurrence = id(call, 2);
                int64_t resource = id(call, 3);
                Check(step->pipe_net_id == pipe_net &&
                          step->record_index == record &&
                          step->occurrence == occurrence,
                      "Pipe record operation and step use different record "
                      "identities");
                auto transport = transport_records.find({pipe_net, record});
                Check(transport != transport_records.end(),
                      "Pipe record operation references missing transport");
                Check(
                    step->communication_id ==
                        transport->second->communication_id,
                    "Pipe record operation has the wrong parent communication");
                if (record_send) {
                  Check(step->kind == "send" && coord_is(step->source_core) &&
                            !step->destination_core.has_value() &&
                            step->source_dfb_id == resource &&
                            step->destination_dfb_id < 0,
                        "Pipe record send disagrees with its communication "
                        "step");
                } else if (record_recv) {
                  Check(step->kind == "receive" &&
                            coord_is(step->destination_core) &&
                            step->destination_dfb_id == resource,
                        "Pipe record receive disagrees with its communication "
                        "step");
                } else if (ref_index == 0) {
                  auto role = call->annotations.Get("tt.communication_role");
                  Check(role.has_value() && role.value().as<StringImm>(),
                        "Pipe record wait has no communication role");
                  bool source =
                      Downcast<StringImm>(role.value())->value == "source";
                  Check(step->kind == "completion" &&
                            (source ? coord_is(step->source_core) &&
                                          !step->destination_core.has_value() &&
                                          step->source_dfb_id == resource
                                    : coord_is(step->destination_core) &&
                                          step->destination_dfb_id == resource),
                        "Pipe record wait disagrees with its completion step");
                } else {
                  Check(step->kind == "publish" &&
                            coord_is(step->destination_core) &&
                            step->destination_dfb_id == resource,
                        "Pipe record wait disagrees with its publication step");
                }
                continue;
              }
              if (pipe_send || pipe_recv || pipe_wait) {
                int64_t transfer = id(call, 0);
                int64_t resource = id(call, 1);
                Check(
                    as_const_int(step->parameter) &&
                        *as_const_int(step->parameter) == transfer,
                    "transport operation and step use different transfer IDs");
                if (pipe_send) {
                  Check(step->kind == "send" && coord_is(step->source_core) &&
                            step->source_dfb_id == resource,
                        "Pipe send disagrees with its communication step");
                } else if (pipe_recv) {
                  Check(step->kind == "receive" &&
                            coord_is(step->destination_core) &&
                            step->destination_dfb_id == resource,
                        "Pipe receive disagrees with its communication step");
                } else if (ref_index == 0) {
                  Check(step->kind == "completion" &&
                            ((coord_is(step->source_core) &&
                              step->source_dfb_id == resource) ||
                             (coord_is(step->destination_core) &&
                              step->destination_dfb_id == resource)),
                        "Pipe wait disagrees with its completion step");
                } else {
                  Check(step->kind == "publish" &&
                            coord_is(step->destination_core) &&
                            step->destination_dfb_id == resource,
                        "Pipe wait disagrees with its publication step");
                }
                continue;
              }
              int64_t source = local_reduce ? id(call, 1) : id(call, 0);
              int64_t destination = local_reduce ? id(call, 2) : id(call, 1);
              PrimExpr parameter = local_copy    ? PrimExpr(Integer(0))
                                   : gather_copy ? call->args[2]
                                                 : call->args[3];
              ffi::String kind =
                  Downcast<StringImm>(
                      call->annotations.at("tt.communication_step_kind"))
                      ->value;
              int64_t site = Downcast<Integer>(
                                 call->annotations.at("tt.communication_site"))
                                 ->value;
              int64_t occurrence =
                  Downcast<Integer>(
                      call->annotations.at("tt.communication_occurrence"))
                      ->value;
              Check(step->kind == kind && step->communication_id == site &&
                        step->occurrence == occurrence &&
                        coord_is(step->source_core) &&
                        coord_is(step->destination_core) &&
                        step->source_dfb_id == source &&
                        step->destination_dfb_id == destination &&
                        ffi::StructuralEqual()(step->parameter, parameter),
                    "local communication operation disagrees with its step");
            }
          };
          for (size_t call_index = 0; call_index < calls.size(); ++call_index) {
            const Call &call = calls[call_index];
            ++ordinal;
            int64_t event = static_cast<int64_t>(event_successors.size());
            event_successors.emplace_back();
            SlotKey slot{x, y, std::string(owner)};
            auto previous = last_slot_event.find(slot);
            if (previous != last_slot_event.end())
              event_successors[previous->second].push_back(event);
            last_slot_event[slot] = event;
            try {
              verify_step_refs(call, call_multipliers[call_index]);
              if (call->op.same_as(accumulator_init())) {
                int64_t accumulator = id(call, 0);
                Check(current_accumulators.count(accumulator),
                      "accumulator initialization references missing "
                      "accumulator");
                const CoreDomain &domain =
                    current_accumulators.at(accumulator)->core_domain;
                Check(x >= domain->begin->x && x < domain->end->x &&
                          y >= domain->begin->y && y < domain->end->y,
                      "accumulator initialization crosses Core ownership");
                Check(accumulator_state[accumulator] == 0,
                      "accumulator initialized more than once");
                accumulator_state[accumulator] = 1;
              } else if (call->op.same_as(dfb_reserve())) {
                int64_t resource = id(call, 0);
                LocalKey key = begin_producer(resource);
                auto &entry = state(key);
                Check(!entry.reserved,
                      "DFB transaction is reserved more than once");
                entry.reserved = true;
                entry.reserve_slot = owner;
                Check(reserve_events.emplace(key, event).second,
                      "DFB transaction has multiple reserve events");
              } else if (call->op.same_as(dfb_wait())) {
                int64_t resource = id(call, 0);
                LocalKey key = begin_consumer(resource);
                auto &entry = state(key);
                entry.waited = true;
                entry.last_use = ordinal;
                wait_events[key].push_back(event);
                if (entry.consumer_slot.empty())
                  entry.consumer_slot = owner;
              } else if (call->op.same_as(dfb_release())) {
                int64_t resource = id(call, 0);
                LocalKey key = release_key(resource);
                auto &entry = state(key);
                Check(!entry.released,
                      "DFB transaction is released more than once");
                entry.released = true;
                entry.release_order = ordinal;
                Check(release_events.emplace(key, event).second,
                      "DFB transaction has multiple release events");
                auto &transactions = active_consumer_transactions[resource];
                transactions.erase(transactions.begin());
                if (transactions.empty())
                  active_consumer_transactions.erase(resource);
                auto producer = active_producer_transaction.find(resource);
                if (producer != active_producer_transaction.end() &&
                    producer->second == std::get<3>(key))
                  active_producer_transaction.erase(producer);
                if (compute_values.Enabled())
                  compute_values.ReleaseDFB(resource, std::get<3>(key));
              } else if (call->op.same_as(tensor_to_dfb_nd())) {
                int64_t resource = id(call, 1);
                int64_t tensor = RequireStaticInteger(call->args[0],
                                                      "Tensor load ABI index");
                Check(tensor_table.count(tensor),
                      "Tensor load references a missing Tensor");
                LocalKey key = producer_key(resource);
                Check(copy_issues.emplace(key, std::make_pair(event, true))
                          .second,
                      "DFB transaction has multiple asynchronous copy issues");
                ffi::Array<PrimExpr> shape;
                for (size_t index = 3; index < call->args.size(); index += 2)
                  shape.push_back(call->args[index]);
                Check(tensor_load_shapes.emplace(key, shape).second,
                      "Tensor-backed DFB transaction has multiple load shapes");
              } else if (call->op.same_as(dfb_to_tensor_nd())) {
                int64_t resource = id(call, 0);
                int64_t tensor = RequireStaticInteger(call->args[1],
                                                      "Tensor store ABI index");
                Check(tensor_table.count(tensor),
                      "Tensor store references a missing Tensor");
                use(resource);
                LocalKey key = consumer_key(resource);
                Check(copy_issues.emplace(key, std::make_pair(event, false))
                          .second,
                      "DFB transaction has multiple asynchronous copy issues");
              } else if (call->op.same_as(dfb_copy_wait())) {
                int64_t resource = id(call, 0);
                Check(active_producer_transaction.count(resource) ||
                          active_consumer_transactions.count(resource),
                      "DFB copy completion has no active transaction");
                LocalKey key = active_producer_transaction.count(resource)
                                   ? producer_key(resource)
                                   : consumer_key(resource);
                auto issue = copy_issues.find(key);
                Check(issue != copy_issues.end(),
                      "DFB copy completion has no matching issue");
                Check(copy_completions.emplace(key, event).second,
                      "DFB transaction has multiple copy completions");
                if (issue->second.second) {
                  produce(resource, event);
                }
              } else if (call->op.same_as(dfb_compute())) {
                produce(id(call, 0), event);
                for (size_t index = 1; index < call->args.size(); ++index)
                  use(id(call, index));
                auto shapes = call->annotations.Get("tt.input_shapes");
                if (shapes.has_value()) {
                  auto typed =
                      shapes.value().as<ffi::Array<ffi::Array<PrimExpr>>>();
                  Check(typed.has_value() &&
                            typed.value().size() == call->args.size() - 1,
                        "dfb_compute input shapes must cover every input "
                        "operand");
                  for (size_t index = 1; index < call->args.size(); ++index)
                    input_shape_uses.push_back({consumer_key(id(call, index)),
                                                typed.value()[index - 1]});
                }
              } else if (call->op.same_as(compute_value()) ||
                         call->op.same_as(compute_value_gemm())) {
                size_t begin = call->op.same_as(compute_value()) ? 1 : 1;
                size_t end = call->op.same_as(compute_value_gemm())
                                 ? 3
                                 : call->args.size();
                for (size_t index = begin; index < end; ++index)
                  use(id(call, index));
              } else if (call->op.same_as(compute_value_store())) {
                produce(id(call, 1), event);
              } else if (call->op.same_as(gemm_update())) {
                int64_t accumulator = id(call, 2);
                Check(current_accumulators.count(accumulator),
                      "GEMM update references missing accumulator");
                Check(accumulator_state[accumulator] == 1,
                      "GEMM update requires a live initialized accumulator");
                use(id(call, 0));
                use(id(call, 1));
                int64_t transpose_a = id(call, 3), transpose_b = id(call, 4);
                Check((transpose_a == 0 || transpose_a == 1) &&
                          (transpose_b == 0 || transpose_b == 1),
                      "GEMM transpose flags must be Boolean");
                const DFBDescriptor &lhs = dfb_table.at(id(call, 0));
                const DFBDescriptor &rhs = dfb_table.at(id(call, 1));
                int64_t k = RequireStaticInteger(
                    lhs->block_shape_in_tiles[transpose_a ? 0 : 1],
                    "GEMM K tiles");
                Check(k == RequireStaticInteger(
                               rhs->block_shape_in_tiles[transpose_b ? 1 : 0],
                               "GEMM K tiles"),
                      "GEMM update K shapes disagree");
                const AccumulatorDescriptor &descriptor =
                    current_accumulators.at(accumulator);
                Check(accumulator_k_tiles[accumulator] <=
                          descriptor->full_k_tiles - k,
                      "GEMM updates exceed accumulator full-K updates");
                accumulator_k_tiles[accumulator] += k;
              } else if (call->op.same_as(accumulator_materialize())) {
                int64_t accumulator = id(call, 0);
                Check(current_accumulators.count(accumulator),
                      "materialization references missing accumulator");
                Check(
                    accumulator_state[accumulator] == 1,
                    "materialization requires a live initialized accumulator");
                Check(accumulator_k_tiles[accumulator] ==
                          current_accumulators.at(accumulator)->full_k_tiles,
                      "materialization must follow all full-K updates");
                accumulator_state[accumulator] = 2;
                produce(id(call, 1), event);
              } else if (call->op.same_as(dfb_pipe_send())) {
                int64_t transfer = id(call, 0);
                int64_t resource = id(call, 1);
                const DFBDescriptor &descriptor = dfb_table.at(resource);
                bool forwarding =
                    owner == descriptor->producer_slot &&
                    (stream_operands
                         ? active_producer_transaction.count(resource)
                         : descriptor->tensor_backing.has_value());
                if (forwarding) {
                  auto &entry = state(producer_key(resource));
                  Check(
                      entry.produced,
                      "producer forwarding must follow Tensor copy completion");
                } else {
                  use(resource);
                }
                Check(
                    sent_transfers
                        .emplace(std::make_pair(transfer, resource), forwarding)
                        .second,
                    "Pipe transfer is sent more than once in one Core work "
                    "item");
              } else if (call->op.same_as(dfb_pipe_recv())) {
                int64_t transfer = id(call, 0);
                int64_t resource = id(call, 1);
                state(producer_key(resource));
                Check(received_transfers.insert({transfer, resource}).second,
                      "Pipe transfer is received more than once in one Core "
                      "work item");
              } else if (call->op.same_as(dfb_pipe_wait())) {
                int64_t transfer = id(call, 0);
                int64_t resource = id(call, 1);
                auto sent = sent_transfers.find({transfer, resource});
                if (sent != sent_transfers.end() && sent->second) {
                  forwarding_completions[producer_key(resource)].push_back(
                      event);
                } else if (received_transfers.count({transfer, resource})) {
                  produce(resource, event);
                } else if (sent != sent_transfers.end()) {
                  use(resource);
                } else {
                  Check(false,
                        "Pipe completion has no matching send or receive");
                }
              } else if (call->op.same_as(dfb_pipe_send_record())) {
                int64_t resource = id(call, 3);
                const DFBDescriptor &descriptor = dfb_table.at(resource);
                bool forwarding =
                    owner == descriptor->producer_slot &&
                    (stream_operands
                         ? active_producer_transaction.count(resource)
                         : descriptor->tensor_backing.has_value());
                if (forwarding) {
                  auto &entry = state(producer_key(resource));
                  Check(entry.produced,
                        "record forwarding must follow Tensor copy completion");
                } else {
                  use(resource);
                }
                RecordTransferKey record{id(call, 0), id(call, 1), id(call, 2),
                                         resource};
                Check(sent_records.emplace(record, forwarding).second,
                      "Pipe record is sent more than once in one Core work "
                      "item");
              } else if (call->op.same_as(dfb_pipe_recv_record())) {
                int64_t resource = id(call, 3);
                state(producer_key(resource));
                RecordTransferKey record{id(call, 0), id(call, 1), id(call, 2),
                                         resource};
                Check(received_records.insert(record).second,
                      "Pipe record is received more than once in one Core work "
                      "item");
              } else if (call->op.same_as(dfb_pipe_wait_record())) {
                int64_t resource = id(call, 3);
                RecordTransferKey record{id(call, 0), id(call, 1), id(call, 2),
                                         resource};
                auto sent = sent_records.find(record);
                if (sent != sent_records.end() && sent->second) {
                  forwarding_completions[producer_key(resource)].push_back(
                      event);
                } else if (received_records.count(record)) {
                  produce(resource, event);
                } else if (sent != sent_records.end()) {
                  use(resource);
                } else {
                  Check(
                      false,
                      "Pipe record completion has no matching send or receive");
                }
              } else if (call->op.same_as(dfb_local_copy()) ||
                         call->op.same_as(dfb_gather_copy())) {
                use(id(call, 0));
                produce(id(call, 1), event, /*partial=*/true);
              } else if (call->op.same_as(dfb_collective_reduce())) {
                use(id(call, 0));
                use(id(call, 1));
                produce(id(call, 2), event, /*partial=*/true);
              } else if (call->op.same_as(dfb_load())) {
                use(id(call, 0));
              }
              if (compute_values.Enabled()) {
                if (call->op.same_as(compute_value()) ||
                    call->op.same_as(dfb_compute()))
                  compute_values.Expression(call, dfb_table);
                else if (call->op.same_as(compute_value_gemm()))
                  compute_values.Gemm(call, dfb_table);
                else if (call->op.same_as(compute_value_store()))
                  compute_values.Store(call, dfb_table);
                else if (call->op.same_as(dfb_to_tensor_nd())) {
                  int64_t resource = id(call, 0);
                  int64_t tensor = RequireStaticInteger(
                      call->args[1], "Tensor export ABI index");
                  Check(tensor_table.count(tensor),
                        "Tensor export references missing ABI descriptor");
                  value_exports.emplace_back(resource,
                                             tensor_table.at(tensor)->dtype);
                }
              } else if (call->op.same_as(dfb_compute())) {
                VerifyGeneralCompute(call, dfb_table, &bound_loop_vars,
                                     stream_operands);
              }
            } catch (const ffi::Error &error) {
              Fail(OperationContext(func, Evaluate(call)) + error.what());
            }
          }
        }
      }
    });
  }
  if (!compute_values.Enabled())
    for (const auto &[id, accumulator] : current_accumulators)
      Check(accumulator_state[id] == 2,
            "accumulator lifetime is missing final materialization");
  for (const auto &[key, waits] : wait_events) {
    auto producer = producer_events.find(key);
    Check(producer != producer_events.end(),
          "DFB wait has no matching producer publication");
    for (int64_t wait : waits)
      event_successors[producer->second].push_back(wait);
  }
  for (const auto &[key, issue] : copy_issues) {
    auto completion = copy_completions.find(key);
    Check(completion != copy_completions.end(),
          "DFB asynchronous copy is missing completion");
    event_successors[issue.first].push_back(completion->second);
    if (!issue.second) {
      auto release = release_events.find(key);
      Check(release != release_events.end(),
            "Tensor export DFB is missing release");
      event_successors[completion->second].push_back(release->second);
    }
  }
  for (const auto &[key, completions] : forwarding_completions) {
    auto release = release_events.find(key);
    Check(release != release_events.end(),
          "producer-forwarded DFB is missing final release");
    for (int64_t completion : completions)
      event_successors[completion].push_back(release->second);
  }
  struct GroupReserve {
    int64_t ordinal;
    int64_t resource;
    int64_t transaction;
    int64_t event;
  };
  std::map<std::tuple<int64_t, int64_t, int64_t>, std::vector<GroupReserve>>
      group_reserves;
  for (const auto &[key, event] : reserve_events) {
    int64_t x = std::get<0>(key), y = std::get<1>(key);
    int64_t resource = std::get<2>(key);
    int64_t transaction = std::get<3>(key);
    int64_t group = resource;
    int64_t ordinal = transaction;
    if (stream_operands) {
      Check(storage.groups.count(resource),
            "reserved DFB references a missing logical stream");
    } else {
      Check(storage.resource_groups.count(resource) &&
                storage.resource_ordinals.count(resource),
            "reserved DFB has no storage-group relation");
      group = storage.resource_groups.at(resource);
      ordinal = storage.resource_ordinals.at(resource);
    }
    group_reserves[{x, y, group}].push_back(
        {ordinal, resource, transaction, event});
  }
  for (auto &[key, events] : group_reserves) {
    std::sort(events.begin(), events.end(),
              [](const GroupReserve &a, const GroupReserve &b) {
                return a.event < b.event;
              });
    int64_t group = std::get<2>(key);
    int64_t capacity = RequireStaticInteger(
        storage.groups.at(group)->block_count, "storage group capacity");
    for (size_t index = 1; index < events.size(); ++index)
      Check(events[index - 1].ordinal < events[index].ordinal,
            "Device body order disagrees with pipeline transaction relation");
    std::sort(events.begin(), events.end(),
              [](const GroupReserve &a, const GroupReserve &b) {
                return std::tie(a.ordinal, a.event) <
                       std::tie(b.ordinal, b.event);
              });
    for (size_t index = static_cast<size_t>(capacity); index < events.size();
         ++index) {
      const GroupReserve &old = events[index - capacity];
      const GroupReserve &next = events[index];
      auto release = release_events.find(LocalKey{
          std::get<0>(key), std::get<1>(key), old.resource, old.transaction});
      Check(release != release_events.end(),
            "storage-group reuse is missing release-before-reserve dependency");
      event_successors[release->second].push_back(next.event);
    }
  }
  std::vector<int64_t> indegree(event_successors.size(), 0);
  for (const auto &successors : event_successors)
    for (int64_t successor : successors)
      ++indegree[successor];
  std::vector<int64_t> ready;
  for (size_t event = 0; event < indegree.size(); ++event)
    if (indegree[event] == 0)
      ready.push_back(static_cast<int64_t>(event));
  size_t visited = 0;
  while (!ready.empty()) {
    int64_t event = ready.back();
    ready.pop_back();
    ++visited;
    for (int64_t successor : event_successors[event])
      if (--indegree[successor] == 0)
        ready.push_back(successor);
  }
  Check(visited == event_successors.size(),
        "cross-slot Device transaction dependency cycle");
  for (const InputShapeUse &use : input_shape_uses) {
    auto loaded = tensor_load_shapes.find(use.key);
    if (loaded != tensor_load_shapes.end()) {
      Check(
          IsCompatibleLogicalInput(loaded->second, use.shape),
          "dfb_compute input logical shape disagrees with its Tensor producer");
    }
  }
  Check(actual_symbols == ordered_symbols,
        "tt.kernel_order does not match Device function symbols");
  for (const auto &[key, state] : lifecycles) {
    int64_t resource = std::get<2>(key);
    const DFBDescriptor &descriptor = dfb_table.at(resource);
    if (state.reserved) {
      auto transactions = reservation_transactions.find(
          StreamKey{std::get<0>(key), std::get<1>(key), resource});
      Check(
          transactions != reservation_transactions.end() &&
              std::get<3>(key) < transactions->second,
          "DFB transaction count disagrees with its structured loop relation");
    }
    Check(!state.reserved || state.produced,
          "reserved DFB transaction is never published");
    Check(!state.produced || state.reserved,
          "DFB producer writes without reserve");
    if (!state.producer_slot.empty())
      Check(state.producer_slot == descriptor->producer_slot,
            "DFB producer slot disagrees with descriptor");
    if (!state.consumer_slot.empty())
      Check(state.consumer_slot == descriptor->consumer_slot,
            "DFB consumer slot disagrees with descriptor");
    if (!state.producer_slot.empty() && !state.consumer_slot.empty() &&
        state.producer_slot != state.consumer_slot)
      Check(state.waited, "cross-slot DFB use is missing consumer wait");
    Check(!state.released || state.release_order >= state.last_use,
          "DFB release precedes its last local use");
    Check(!state.produced || state.released,
          "published DFB transaction is missing release");
  }
  VerifyCurrentL1Budget(mod, dfb_table, resource_cores, storage);
  compute_values.Finish();
  for (const auto &[resource, dtype] : value_exports)
    compute_values.Export(resource, dtype);

  if (communication.has_value()) {
    int64_t expected_site = 0;
    for (const CommunicationDescriptor &entry : communication.value()) {
      Check(entry.defined() && entry->communication_id == expected_site++,
            "communication site IDs must be dense and deterministic");
      Check(entry->kind == "transfer" || entry->kind == "all_gather" ||
                entry->kind == "all_reduce",
            "unknown high-level communication kind");
      Check(entry->source_region.defined() &&
                entry->destination_region.defined() &&
                entry->group_begin.size() == 2 && entry->group_end.size() == 2,
            "communication descriptor is incomplete");
      Check(!entry->algorithm.empty(),
            "communication descriptor has no selected algorithm");
      Check((entry->kind == "transfer" &&
             (entry->algorithm == "direct_p2p" ||
              entry->algorithm == "direct_multicast")) ||
                (entry->kind == "all_gather" &&
                 entry->algorithm == "direct_all_to_all") ||
                (entry->kind == "all_reduce" &&
                 entry->algorithm == "root0_left_fold_broadcast"),
            "communication kind and selected algorithm disagree");
    }
  }
  for (size_t index = 0; index < steps.size(); ++index) {
    const CommunicationStepDescriptor &step = steps[index];
    Check(step.defined() && step->step_id == static_cast<int64_t>(index),
          "communication step IDs must be dense and deterministic");
    Check(communication.has_value() && step->communication_id >= 0 &&
              step->communication_id <
                  static_cast<int64_t>(communication.value().size()),
          "communication step references a missing site");
    Check(step->owner_slot == "trisc" || step->owner_slot == "ncrisc" ||
              step->owner_slot == "brisc",
          "communication step has an invalid owner slot");
    Check(step->kind == "send" || step->kind == "receive" ||
              step->kind == "completion" || step->kind == "publish" ||
              step->kind == "local_copy" || step->kind == "local_reduce",
          "communication step has an unknown kind");
    bool transport_step = step->kind == "send" || step->kind == "receive" ||
                          step->kind == "completion" || step->kind == "publish";
    Check(
        (transport_step && step->pipe_net_id >= 0 && step->record_index >= 0) ||
            (!transport_step && step->pipe_net_id == -1 &&
             step->record_index == -1),
        "communication step has an invalid Pipe record identity");
    if (transport_step) {
      auto record =
          transport_records.find({step->pipe_net_id, step->record_index});
      Check(record != transport_records.end(),
            "communication step references a missing Pipe record");
      Check(record->second->communication_id == step->communication_id,
            "communication step and Pipe record have different parents");
    }
    Check(
        step_ref_counts[index] == 1,
        "communication step must be referenced by exactly one body operation");
    for (const Integer &dependency : step->dependencies)
      Check(dependency->value >= 0 && dependency->value < step->step_id,
            "communication step dependency must reference an earlier step");
    if (step->source_dfb_id >= 0)
      Check(dfb_table.count(step->source_dfb_id),
            "communication step source DFB is missing");
    if (step->destination_dfb_id >= 0)
      Check(dfb_table.count(step->destination_dfb_id),
            "communication step destination DFB is missing");
  }
  if (communication.has_value()) {
    using TransferPair = std::tuple<int64_t, int64_t, int64_t, int64_t>;
    for (const CommunicationDescriptor &entry : communication.value()) {
      bool collective =
          entry->kind == "all_gather" || entry->kind == "all_reduce";
      int64_t bx = 0, by = 0, ex = launch_grid->x, ey = launch_grid->y;
      if (collective) {
        bx = RequireStaticInteger(entry->group_begin[0],
                                  "communication group begin x");
        by = RequireStaticInteger(entry->group_begin[1],
                                  "communication group begin y");
        ex = RequireStaticInteger(entry->group_end[0],
                                  "communication group end x");
        ey = RequireStaticInteger(entry->group_end[1],
                                  "communication group end y");
        Check(bx >= 0 && by >= 0 && bx < ex && by < ey &&
                  ex <= launch_grid->x && ey <= launch_grid->y,
              "communication group is outside the launch grid");
      }
      int64_t group_size =
          CheckedProduct(ex - bx, ey - by, "communication group size");
      int64_t occurrence_count = 1;
      for (const PrimExpr &extent : entry->occurrence_extents)
        occurrence_count = CheckedProduct(
            occurrence_count,
            RequireStaticInteger(extent, "communication occurrence extent"),
            "communication occurrence count");
      Check(occurrence_count > 0,
            "communication occurrence domain must be nonempty");

      std::map<int64_t, std::vector<CommunicationStepDescriptor>> by_occurrence;
      for (const CommunicationStepDescriptor &step : steps)
        if (step->communication_id == entry->communication_id)
          by_occurrence[step->occurrence].push_back(step);
      Check(!by_occurrence.empty() &&
                occurrence_count % static_cast<int64_t>(by_occurrence.size()) ==
                    0,
            "communication steps and enclosing loops do not cover every site "
            "occurrence");
      int64_t static_occurrences = by_occurrence.size();
      int64_t dynamic_multiplier = occurrence_count / static_occurrences;
      for (int64_t occurrence = 0; occurrence < static_occurrences;
           ++occurrence) {
        Check(by_occurrence.count(occurrence),
              "communication occurrence IDs must be dense");
        std::map<std::string, int64_t> kinds;
        std::set<TransferPair> sends, receives;
        std::set<std::pair<int64_t, int64_t>> source_records;
        for (const CommunicationStepDescriptor &step :
             by_occurrence.at(occurrence)) {
          Check(step_ref_dynamic_counts[step->step_id] == dynamic_multiplier,
                "communication step dynamic multiplicity disagrees with its "
                "occurrence relation");
          ++kinds[std::string(step->kind)];
          if (step->kind != "send" && step->kind != "receive")
            continue;
          Check(step->source_core.has_value(),
                "transport step is missing source coordinates");
          CoreCoord source = step->source_core.value();
          if (step->kind == "send") {
            Check(!step->destination_core.has_value(),
                  "record-level send must not name one destination Core");
            auto record =
                transport_records.find({step->pipe_net_id, step->record_index});
            Check(record != transport_records.end() &&
                      ffi::StructuralEqual()(record->second->src_coord, source),
                  "record-level send disagrees with its transport record");
            Check(source_records.insert({step->pipe_net_id, step->record_index})
                      .second,
                  "communication occurrence has a duplicate record send");
            for (int64_t dx = record->second->dst_begin->x;
                 dx < record->second->dst_end->x; ++dx)
              for (int64_t dy = record->second->dst_begin->y;
                   dy < record->second->dst_end->y; ++dy) {
                Check(dx != source->x || dy != source->y,
                      "record-level send has a loopback destination");
                Check(sends.insert({source->x, source->y, dx, dy}).second,
                      "communication occurrence has a duplicate transport "
                      "endpoint");
              }
          } else {
            Check(step->destination_core.has_value(),
                  "receive step is missing destination coordinates");
            CoreCoord destination = step->destination_core.value();
            Check((!collective || (source->x >= bx && source->x < ex &&
                                   source->y >= by && source->y < ey)) &&
                      destination->x >= bx && destination->x < ex &&
                      destination->y >= by && destination->y < ey &&
                      (source->x != destination->x ||
                       source->y != destination->y),
                  "transport step has an invalid or loopback group endpoint");
            Check(receives
                      .insert({source->x, source->y, destination->x,
                               destination->y})
                      .second,
                  "communication occurrence has a duplicate receive endpoint");
          }
        }
        Check(sends == receives,
              "communication send/receive endpoint coverage disagrees");
        Check(kinds["completion"] ==
                      static_cast<int64_t>(sends.size() +
                                           source_records.size()) &&
                  kinds["publish"] == static_cast<int64_t>(sends.size()),
              "communication completion/publication coverage is incomplete");

        if (entry->kind == "all_gather") {
          int64_t axis =
              RequireStaticInteger(entry->parameter, "all_gather axis");
          Check(axis == 0 || axis == 1,
                "all_gather axis must be normalized to zero or one");
          std::set<TransferPair> expected;
          for (int64_t sx = bx; sx < ex; ++sx)
            for (int64_t sy = by; sy < ey; ++sy)
              for (int64_t dx = bx; dx < ex; ++dx)
                for (int64_t dy = by; dy < ey; ++dy)
                  if (sx != dx || sy != dy)
                    expected.insert({sx, sy, dx, dy});
          Check(sends == expected &&
                    kinds["local_copy"] == group_size * group_size &&
                    kinds["local_reduce"] == 0,
                "direct all_gather does not cover every source/destination "
                "contribution");
        } else if (entry->kind == "all_reduce") {
          const auto *reduction = entry->parameter.as<StringImmNode>();
          Check(reduction &&
                    (reduction->value == "sum" || reduction->value == "max" ||
                     reduction->value == "min"),
                "all_reduce has an unsupported reduction operation");
          std::set<TransferPair> expected;
          for (int64_t x = bx; x < ex; ++x)
            for (int64_t y = by; y < ey; ++y)
              if (x != bx || y != by) {
                expected.insert({x, y, bx, by});
                expected.insert({bx, by, x, y});
              }
          Check(sends == expected && kinds["local_copy"] == 1 &&
                    kinds["local_reduce"] == group_size - 1,
                "direct all_reduce has incomplete gather/reduce/broadcast "
                "coverage");
        }
      }
    }
  }
  return parameterized;
}

IRModule VerifyModule(IRModule mod) {
  auto stage = mod->GetAttr<ffi::String>("tt.ir_stage");
  if (stage.has_value())
    return VerifyCurrentDeviceModule(std::move(mod));
  Integer version = RequireModuleAttr<Integer>(mod, kDeviceIRVersionAttr);
  Check(version->value == kDeviceIRVersion || version->value == 2 ||
            version->value == 3 || version->value == 4 || version->value == 5 ||
            version->value == 6 || version->value == 7 || version->value == 8,
        "unsupported tt.device_ir_version " + std::to_string(version->value) +
            "; expected 1, 2, 3, 4, 5, 6, 7, or 8");

  for (const char *key :
       {kDFBStorageGroupsAttr, kPipelineRelationsAttr, kPipelineStagesAttr,
        kPipelineExtentAttr, "tt.pipeline_wait_policy"})
    Check(version->value >= 3 || !mod->attrs->dict.count(key),
          std::string(key) +
              " pipeline metadata requires Device IR schema v3 or newer");

  Check(version->value == 5 || version->value == 6 || version->value >= 7 ||
            !mod->attrs->dict.count(kAccumulatorTableAttr),
        "tt.accumulator_table requires Device IR schema v5 or v6");
  if (version->value == 5 ||
      (version->value == 6 && mod->attrs->dict.count(kAccumulatorTableAttr))) {
    Check(!RequireModuleAttr<ffi::Array<AccumulatorDescriptor>>(
               mod, kAccumulatorTableAttr)
               .empty(),
          "schema v5 requires a nonempty accumulator table");
  }
  Check(version->value >= 7 || !mod->attrs->dict.count(kComputeValueTableAttr),
        "tt.compute_value_table requires Device IR v7");
  Check(version->value == 4 || version->value == 6 || version->value >= 7 ||
            !mod->attrs->dict.count(kPipeTransferTableAttr),
        "tt.pipe_transfer_table requires Device IR schema v4, v6 or v7");
  const bool multicore =
      version->value == 4 || version->value == 6 ||
      (version->value >= 7 && mod->attrs->dict.count(kPipeTransferTableAttr));

  ffi::String target_arch =
      RequireModuleAttr<ffi::String>(mod, kTargetArchAttr);
  Check(target_arch == "wormhole_b0" || target_arch == "blackhole",
        "unsupported tt.target_arch `" + std::string(target_arch) + "`");

  CoreCoord launch_grid = RequireModuleAttr<CoreCoord>(mod, kLaunchGridAttr);
  Check(launch_grid->x > 0 && launch_grid->y > 0,
        "tt.launch_grid must contain two positive extents");

  OperationIdentity operation =
      RequireModuleAttr<OperationIdentity>(mod, kOperationIdentityAttr);
  Check(!operation->operation_id.empty(),
        "tt.operation_identity has an empty operation_id");
  Check(operation->source_span.defined(),
        "tt.operation_identity has no source_span");

  ffi::Array<TensorDescriptor> tensors =
      RequireModuleAttr<ffi::Array<TensorDescriptor>>(mod, kTensorTableAttr);
  ffi::Array<DFBDescriptor> dfbs =
      RequireModuleAttr<ffi::Array<DFBDescriptor>>(mod, kDFBTableAttr);
  ffi::Array<PipeDescriptor> pipes =
      RequireModuleAttr<ffi::Array<PipeDescriptor>>(mod, kPipeTableAttr);
  ffi::Array<ffi::String> kernel_order =
      RequireModuleAttr<ffi::Array<ffi::String>>(mod, kKernelOrderAttr);
  Check(multicore ||
            (kernel_order.size() == 3 && kernel_order[0] == "trisc" &&
             kernel_order[1] == "ncrisc" && kernel_order[2] == "brisc"),
        "tt.kernel_order must be exactly [trisc, ncrisc, brisc]");

  const bool general = version->value >= 2;
  TensorTable tensor_table = VerifyTensorTable(tensors, launch_grid, general);
  DFBTable dfb_table = VerifyDFBTable(dfbs, tensor_table, launch_grid, general);
  VerifyPipeTable(pipes, dfb_table, launch_grid);
  if (!multicore && !mod->attrs->dict.count(kPipelineStagesAttr) &&
      (version->value < 3 || version->value == 5 || version->value >= 7))
    VerifyLegacyL1Budget(mod, dfb_table);
  if (multicore) {
    VerifyFunctions(mod, target_arch, launch_grid, tensor_table, nullptr, true,
                    true);
    if (version->value == 6)
      for (const auto &[global, base] : mod->functions)
        DeriveComputeRequirements(mod, Downcast<PrimFunc>(base));
    VerifyMulticoreProgram(mod, tensor_table, dfb_table, launch_grid, pipes);
  } else if (general) {
    Check(launch_grid->x == 1 && launch_grid->y == 1 && pipes.empty(),
          "Phase 4 is single-Core without Pipe communication");
    VerifyFunctions(mod, target_arch, launch_grid, tensor_table, nullptr, true);
    if (version->value == 5) {
      for (const auto &[global, base] : mod->functions)
        DeriveComputeRequirements(mod, Downcast<PrimFunc>(base));
    }
    VerifyGeneralProgram(mod, tensor_table, dfb_table);
  } else if (dfbs.empty()) {
    Check(pipes.empty(), "Device IR without DFBs must not contain Pipes");
    VerifyFunctions(mod, target_arch, launch_grid, tensor_table, nullptr);
  } else {
    Phase2AddPlan phase2_plan =
        VerifyPhase2AddDescriptors(dfbs, tensor_table, pipes, launch_grid);
    VerifyFunctions(mod, target_arch, launch_grid, tensor_table, &phase2_plan);
  }
  for (const auto &[global, base] : mod->functions) {
    PrimFunc func = Downcast<PrimFunc>(base);
    auto requirements =
        func->GetAttr<ComputeRequirements>(kComputeRequirementsAttr);
    bool compute =
        func->GetAttr<ffi::String>(kKernelSlotAttr).value() == "trisc";
    Check(compute || !requirements.has_value(),
          "data movement kernel must not carry tt.compute_requirements");
    auto region_requirements = func->GetAttr<ffi::Array<ComputeRequirements>>(
        "tt.compute_region_requirements");
    Check((version->value == 8 && compute) || !region_requirements.has_value(),
          "compute region requirements require v8 TRISC");
    if (version->value == 8 && compute) {
      Check(region_requirements.has_value(),
            "v8 missing compute region requirements");
      Check(
          ffi::StructuralEqual()(region_requirements.value(),
                                 DeriveRegionComputeRequirements(mod, func)),
          "compute region requirements disagree with actual Device operations");
    }
    if (compute && (version->value == 5 || version->value == 6 ||
                    version->value >= 7 || requirements.has_value())) {
      Check(requirements.has_value(),
            "schema v5 compute kernel missing tt.compute_requirements");
      ComputeRequirements expected = DeriveComputeRequirements(mod, func);
      Check(ffi::StructuralEqual()(requirements.value(), expected),
            "tt.compute_requirements disagrees with actual Device operations");
    }
  }
  return mod;
}

} // namespace

tvm::transform::Pass VerifyTenstorrentDeviceIR() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext context) {
    IRModule expanded = ExpandDeviceProgram(mod);
    VerifyModule(expanded);
    return mod;
  };
  return tvm::transform::CreateModulePass(
      pass_func, 0, "tl.tenstorrent.VerifyTenstorrentDeviceIR", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::GlobalDef().def("tl.tenstorrent.transform.VerifyTenstorrentDeviceIR",
                        VerifyTenstorrentDeviceIR);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file tenstorrent/transform/verify_device_ir.cc
 * \brief Read-only verifier for the Tenstorrent Device TIR v1 schema.
 */
#include "verify_device_ir.h"

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
#include <cstdint>
#include <iterator>
#include <sstream>
#include <string>
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

void VerifyNoForbiddenOps(const tirx::PrimFunc &func,
                          const std::string &symbol) {
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
      Check(op_name != "tl.tt.tile_add",
            "PrimFunc `" + symbol +
                "` retains intermediate op `tl.tt.tile_add`");
    }
  });
}

using TensorTable = std::unordered_map<int64_t, TensorDescriptor>;
using DFBTable = std::unordered_map<int64_t, DFBDescriptor>;

TensorTable VerifyTensorTable(const ffi::Array<TensorDescriptor> &tensors,
                              const CoreCoord &launch_grid) {
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
    Check(tensor->tile_shape.size() == tensor->shape.size(),
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
    Check(tensor->source_span.defined(), label + " has no source_span");
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
                        const CoreCoord &launch_grid) {
  DFBTable table;
  std::unordered_set<std::string> source_buffer_ids;
  for (const DFBDescriptor &dfb : dfbs) {
    const std::string label = "DFB " + std::to_string(dfb->dfb_id);
    Check(dfb->dfb_id >= 0, label + " has a negative dfb_id");
    Check(table.emplace(dfb->dfb_id, dfb).second,
          label + " duplicates dfb_id");
    Check(!dfb->source_buffer_identity.empty(),
          label + " has no source_buffer_identity");
    Check(source_buffer_ids.insert(dfb->source_buffer_identity).second,
          label + " duplicates source_buffer_identity `" +
              std::string(dfb->source_buffer_identity) + "`");
    Check(!dfb->element_dtype.is_void(), label + " has a void dtype");
    VerifyPositiveShape(dfb->tile_shape, label + " tile_shape");
    VerifyPositiveShape(dfb->block_shape_in_tiles,
                        label + " block_shape_in_tiles");
    Check(dfb->tile_shape.size() == dfb->block_shape_in_tiles.size(),
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
    Check(dfb->producer_slot != dfb->consumer_slot,
          label + " must have distinct SPSC producer and consumer slots");
    VerifyDomain(dfb->producer_domain, launch_grid, label + " producer_domain");
    VerifyDomain(dfb->consumer_domain, launch_grid, label + " consumer_domain");
    Check(dfb->source_span.defined(), label + " has no source_span");
  }
  return table;
}

void VerifyPipeTable(const ffi::Array<PipeDescriptor> &pipes,
                     const DFBTable &dfbs,
                     const CoreCoord &launch_grid) {
  std::unordered_set<std::string> event_ids;
  std::unordered_map<int64_t, int64_t> next_event;
  for (const PipeDescriptor &pipe : pipes) {
    const std::string label = "PipeNet " + std::to_string(pipe->pipe_net_id) +
                              " event " + std::to_string(pipe->event_index);
    Check(pipe->pipe_net_id >= 0, label + " has a negative pipe_net_id");
    Check(pipe->event_index >= 0, label + " has a negative event_index");
    const std::string event_id = std::to_string(pipe->pipe_net_id) + ":" +
                                 std::to_string(pipe->event_index);
    Check(event_ids.insert(event_id).second,
          label + " duplicates a Pipe event identity");
    int64_t &expected = next_event[pipe->pipe_net_id];
    Check(pipe->event_index == expected,
          label + " is out of order; expected event_index " +
              std::to_string(expected));
    ++expected;
    VerifyCoord(pipe->src_coord, launch_grid, label + " src_coord");
    VerifyDomain(CoreDomain(pipe->dst_begin, pipe->dst_end), launch_grid,
                 label + " destination range");
    Check(pipe->contract == "point_to_point" || pipe->contract == "collective",
          label + " has invalid contract `" + std::string(pipe->contract) +
              "`");
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
            ffi::StructuralEqual()(tensor_a->tile_shape,
                                   tensor_c->tile_shape),
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
  return {std::move(input_a), std::move(input_b), std::move(output), rows,
          cols, tile_rows * tile_cols};
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

MarkerKind ParseMarkerKind(const tirx::Call &call,
                           const std::string &symbol) {
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
  const std::string name =
      op == nullptr ? std::string(call->op->GetTypeKey())
                    : std::string(op->name);
  Fail("Phase 2 PrimFunc `" + symbol + "` contains non-canonical op `" +
       name + "`");
}

size_t MarkerArity(MarkerKind kind) {
  switch (kind) {
  case MarkerKind::kDFBReserve:
  case MarkerKind::kDFBWait:
    return 2;
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
          "Phase 2 PrimFunc `" + symbol + "` marker " +
              std::to_string(index) + " must be `" +
              MarkerName(expected[index].kind) + "`");
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
    Check(logical_kernel->kind == "compute" && logical_kernel->role == "add",
          "Phase 2 trisc LogicalKernel must have kind `compute` and role `add`");
    VerifyMarkerSequence(
        ParseMarkers(func->body, symbol),
        {{MarkerKind::kDFBReserve, {output_id, 1}},
         {MarkerKind::kDFBWait, {input_a_id, 1}},
         {MarkerKind::kDFBWait, {input_b_id, 1}},
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
    VerifyMarkerSequence(
        ParseMarkers(func->body, symbol),
        {{MarkerKind::kDFBReserve, {input_a_id, 1}},
         {MarkerKind::kTensorToDFB,
          {0, input_a_id, 0, 0, plan.rows, plan.cols}},
         {MarkerKind::kDFBReserve, {input_b_id, 1}},
         {MarkerKind::kTensorToDFB,
          {1, input_b_id, 0, 0, plan.rows, plan.cols}},
         {MarkerKind::kDFBWait, {output_id, 1}},
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
                     const Phase2AddPlan *phase2_plan) {
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
    Check(slots.insert(slot).second,
          "operation contains duplicate slot `" + std::string(slot) + "`");

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
    Check(logical_kernel->source_span.defined(),
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
        Check(tensor_arg_indices.empty(), "Phase 2 " + std::string(slot) +
                                              " must not own Tensor ABI args");
      }
    }
    Check(IsVoidType(func->ret_type),
          "PrimFunc `" + symbol + "` must return void");
    VerifyNoForbiddenOps(func, symbol);
    if (phase2_plan != nullptr) {
      VerifyPhase2Body(func, symbol, slot, logical_kernel, *phase2_plan);
    } else {
      Check(IsCanonicalNoOp(func->body),
            "Phase 1 PrimFunc `" + symbol +
                "` body must be the canonical Evaluate(0) no-op");
    }
    Check(!func->attrs->dict.count(kBufferMetadataTableAttr),
          "PrimFunc `" + symbol + "` retains Form-input-only attr `" +
              std::string(kBufferMetadataTableAttr) + "`");

    auto record_var = [&](const tirx::Var &var) {
      auto owner = var_owners.find(var);
      if (owner != var_owners.end() && owner->second != std::string(slot)) {
        Fail("Var `" + std::string(var->name_hint) +
             "` is shared across Device IR slots `" + owner->second +
             "` and `" + std::string(slot) + "`");
      }
      var_owners.emplace(var, std::string(slot));
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

  Check(prim_func_count == 3,
        "operation must contain exactly three slot PrimFuncs");
  Check(slots.size() == 3 && slots.count("trisc") != 0 &&
            slots.count("ncrisc") != 0 && slots.count("brisc") != 0,
        "operation must contain exactly trisc, ncrisc, and brisc slots");
}

IRModule VerifyModule(IRModule mod) {
  Integer version = RequireModuleAttr<Integer>(mod, kDeviceIRVersionAttr);
  Check(version->value == kDeviceIRVersion,
        "unsupported tt.device_ir_version " + std::to_string(version->value) +
            "; expected 1");

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
  Check(kernel_order.size() == 3 && kernel_order[0] == "trisc" &&
            kernel_order[1] == "ncrisc" && kernel_order[2] == "brisc",
        "tt.kernel_order must be exactly [trisc, ncrisc, brisc]");

  TensorTable tensor_table = VerifyTensorTable(tensors, launch_grid);
  DFBTable dfb_table = VerifyDFBTable(dfbs, tensor_table, launch_grid);
  VerifyPipeTable(pipes, dfb_table, launch_grid);
  if (dfbs.empty()) {
    Check(pipes.empty(), "Device IR without DFBs must not contain Pipes");
    VerifyFunctions(mod, target_arch, launch_grid, tensor_table, nullptr);
  } else {
    Phase2AddPlan phase2_plan =
        VerifyPhase2AddDescriptors(dfbs, tensor_table, pipes, launch_grid);
    VerifyFunctions(mod, target_arch, launch_grid, tensor_table, &phase2_plan);
  }
  return mod;
}

} // namespace

tvm::transform::Pass VerifyTenstorrentDeviceIR() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext context) {
    return VerifyModule(std::move(mod));
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

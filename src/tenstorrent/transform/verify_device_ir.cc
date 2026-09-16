/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file tenstorrent/transform/verify_device_ir.cc
 * \brief Read-only verifier for the Tenstorrent Device TIR v1/v2 schemas.
 */
#include "verify_device_ir.h"

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
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
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
      Check(op_name != "tl.tt.tile_add" && op_name != "tl.tt.tile_compute",
            "PrimFunc `" + symbol +
                "` retains intermediate op `tl.tt.tile_add`");
    }
  });
}

using TensorTable = std::unordered_map<int64_t, TensorDescriptor>;
using DFBTable = std::unordered_map<int64_t, DFBDescriptor>;

TensorTable VerifyTensorTable(const ffi::Array<TensorDescriptor> &tensors,
                              const CoreCoord &launch_grid,
                              bool general = false) {
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
                        const CoreCoord &launch_grid, bool general = false) {
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
    Check(dfb->source_span.defined(), label + " has no source_span");
  }
  return table;
}

void VerifyPipeTable(const ffi::Array<PipeDescriptor> &pipes,
                     const DFBTable &dfbs, const CoreCoord &launch_grid) {
  std::unordered_set<std::string> event_ids;
  std::unordered_map<int64_t, int64_t> next_event;
  std::unordered_map<int64_t, ffi::String> contracts;
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
    VerifyMarkerSequence(ParseMarkers(func->body, symbol),
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
                     const Phase2AddPlan *phase2_plan, bool general = false,
                     bool multicore = false) {
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
        Check(tensor_arg_indices.empty(),
              "Phase 2 " + std::string(slot) + " must not own Tensor ABI args");
      }
    }
    Check(IsVoidType(func->ret_type),
          "PrimFunc `" + symbol + "` must return void");
    VerifyNoForbiddenOps(func, symbol);
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

void VerifyGeneralCompute(const Call &call, const DFBTable &dfbs) {
  Check(!call->args.empty(), "dfb_compute must have an output DFB");
  std::vector<DFBDescriptor> operands;
  std::unordered_set<int64_t> input_ids;
  for (size_t i = 0; i < call->args.size(); ++i) {
    int64_t id = RequireStaticInteger(call->args[i], "dfb_compute resource ID");
    Check(dfbs.count(id), "dfb_compute references missing DFB");
    operands.push_back(dfbs.at(id));
    if (i)
      input_ids.insert(id);
  }
  Check(!input_ids.count(operands[0]->dfb_id),
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
  Check(maps_value.has_value(), "dfb_compute missing tt.access_maps");
  auto maps = CheckedMaps(maps_value.value());
  Check(maps.size() + 1 == operands.size(),
        "dfb_compute access maps must cover every input operand");
  ffi::String kind = ComputeString(call, "tt.compute_kind");
  auto input_shapes_value = call->annotations.Get("tt.input_shapes");
  Check(input_shapes_value.has_value(), "dfb_compute missing tt.input_shapes");
  auto input_shapes =
      CheckedShapes(input_shapes_value.value(), "tt.input_shapes");
  Check(input_shapes.size() + 1 == operands.size(),
        "dfb_compute input shape count mismatch");
  std::unordered_map<int64_t, ffi::Array<Integer>> maps_by_id;
  for (size_t i = 0; i < maps.size(); ++i) {
    auto input_shape = input_shapes[i];
    VerifyTileGrid(input_shape, operands[i + 1]->block_shape_in_tiles,
                   "dfb_compute input");
    Check(maps[i].size() == input_shape.size(),
          "dfb_compute access map rank disagrees with input");
    auto [previous, inserted] =
        maps_by_id.emplace(operands[i + 1]->dfb_id, maps[i]);
    Check(inserted || ffi::StructuralEqual()(previous->second, maps[i]),
          "one DFB input has conflicting access maps");
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
      Check(!object.as<BufferLoadNode>() && !object.as<VarNode>(),
            "Device expression retains BufferLoad or free scalar Var");
      if (auto expr = object.as<PrimExpr>()) {
        Check(object.as<IntImmNode>() ||
                  SupportedComputeDType(expr.value().dtype()),
              "Device expression has unsupported intermediate dtype");
        Check(object.as<IntImmNode>() || object.as<FloatImmNode>() ||
                  object.as<CastNode>() || object.as<CallNode>() ||
                  object.as<AddNode>() || object.as<SubNode>() ||
                  object.as<MulNode>() || object.as<DivNode>() ||
                  object.as<MinNode>() || object.as<MaxNode>(),
              "Device expression contains an unsupported expression node");
        if (const auto *cast = object.as<CastNode>())
          Check(cast->annotations.empty(),
                "Device Cast annotations are unsupported");
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
      Check(ffi::StructuralEqual()(expected, shape),
            "reduction output logical shape mismatch");
    }
  }
}

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
      RequireModuleAttr<Integer>(mod, kDeviceIRVersionAttr)->value == 3;
  if (!result.enabled)
    return result;
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
    Check(ordinal >= 0 && ordinal < extent &&
              relation[1]->value == ordinal % result.depth,
          "pipeline transaction relation has invalid iteration/stage");
    Check(result.ordinals.emplace(id, ordinal).second,
          "duplicate pipeline transaction relation DFB ID");
    int64_t &entry = result.ordered.at(result.groups.at(id))[ordinal];
    Check(entry == -1, "pipeline storage pool has duplicate iteration");
    entry = id;
  }
  int64_t payload_bytes = 0;
  for (const auto &[group, ids] : result.ordered) {
    const std::string label = "pipeline storage pool " + std::to_string(group);
    Check(
        std::all_of(ids.begin(), ids.end(), [](int64_t id) { return id >= 0; }),
        label + " must have exactly one generation per iteration");
    DFBDescriptor first = dfbs.at(ids.front());
    int64_t capacity =
        RequireStaticInteger(first->block_count, label + " block_count");
    Check(capacity >= result.depth,
          label + " capacity insufficient for pipeline window: requires " +
              std::to_string(result.depth) + ", has " +
              std::to_string(capacity));
    for (int64_t id : ids) {
      DFBDescriptor dfb = dfbs.at(id);
      Check(ffi::StructuralEqual()(first->block_count, dfb->block_count) &&
                first->element_dtype == dfb->element_dtype &&
                ffi::StructuralEqual()(first->tile_shape, dfb->tile_shape) &&
                ffi::StructuralEqual()(first->block_shape_in_tiles,
                                       dfb->block_shape_in_tiles) &&
                first->producer_slot == dfb->producer_slot &&
                first->consumer_slot == dfb->consumer_slot &&
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
  }
  // This is logical payload storage, counted once per pool.  It does not
  // estimate alignment, TT-Lang scratch, firmware reservations, or free L1.
  auto budget = mod->GetAttr<Integer>(kL1CapacityBytesAttr);
  if (budget.has_value()) {
    Check(budget.value()->value > 0, "tt.l1_capacity_bytes must be positive");
    Check(payload_bytes <= budget.value()->value,
          "L1 logical payload lower bound " + std::to_string(payload_bytes) +
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
  // Verify an independent structured iteration by induction: every active
  // slot has the same loop boundary, each resource is defined and fully used
  // in one iteration, and no DFB value escapes that iteration. Tensor inout
  // dependencies remain ordered in the NCRISC stream.
  bool has_loop = false;
  for (const auto &[global, base] : mod->functions)
    has_loop |= Downcast<PrimFunc>(base)->body.as<ForNode>() != nullptr;
  if (has_loop) {
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
                ffi::StructuralEqual()(dfb->block_shape_in_tiles,
                                       tensor->tile_grid_shape),
            "DFB Tensor backing metadata mismatch");
    }
  }
  for (const auto &[global, base] : mod->functions) {
    PrimFunc function = Downcast<PrimFunc>(base);
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
      const auto *evaluate = statement.as<EvaluateNode>();
      Check(evaluate != nullptr,
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
        Check(dfbs.count(id),
              "Device operation references missing DFB " + std::to_string(id));
        return id;
      };
      auto read = [&](int64_t id) {
        Check(waited.count(id),
              "DFB use must be preceded by dfb_wait in consumer slot");
        Check(dfbs.at(id)->consumer_slot == slot,
              "DFB consumer slot metadata mismatch");
        Check(!pipeline.enabled || !releases.count(id),
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
      if (call->op.same_as(dfb_reserve()) || call->op.same_as(dfb_wait())) {
        Check(call->args.size() == 2 &&
                  RequireStaticInteger(call->args[1], "transaction count") == 1,
              "immutable DFB reserve/wait must request one transaction");
        int64_t id = id_at(0);
        if (call->op.same_as(dfb_reserve())) {
          Check(dfbs.at(id)->producer_slot == slot,
                "DFB reserve in wrong slot");
          Check(reserves.emplace(id, event).second,
                "DFB generation reserved more than once");
        } else {
          Check(dfbs.at(id)->consumer_slot == slot, "DFB wait in wrong slot");
          Check(!pipeline.enabled || !releases.count(id),
                "DFB wait after release risks overwritten data");
          waited.insert(id);
        }
      } else if (pipeline.enabled && call->op.same_as(dfb_copy_wait())) {
        Check(slot == "ncrisc" && call->args.size() == 2 &&
                  RequireStaticInteger(call->args[1], "copy wait count") == 1,
              "dfb_copy_wait requires ncrisc and one transaction");
        int64_t id = id_at(0);
        Check(copy_issues.count(id),
              "dfb_copy_wait must follow its copy issue");
        Check(copy_completions.emplace(id, event).second,
              "DFB copy completed more than once");
        if (events[copy_issues.at(id)].call->op.same_as(tensor_to_dfb_nd()))
          publish(id);
      } else if (pipeline.enabled && call->op.same_as(dfb_release())) {
        Check(call->args.size() == 2 &&
                  RequireStaticInteger(call->args[1], "release count") == 1,
              "dfb_release must release one transaction");
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
      } else if (call->op.same_as(dfb_compute())) {
        Check(slot == "trisc", "dfb_compute must execute in trisc");
        VerifyGeneralCompute(call, dfbs);
        logical_shapes[id_at(0)] = ComputeShape(call, "tt.logical_domain");
        for (size_t i = 1; i < call->args.size(); ++i) {
          read(id_at(i));
          if (pipeline.enabled)
            Check(
                pipeline.ordinals.at(id_at(i)) ==
                    pipeline.ordinals.at(id_at(0)),
                "dfb_compute transaction relation crosses pipeline iterations");
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
        for (size_t axis = 0; axis < tensor->shape.size(); ++axis) {
          Check(RequireStaticInteger(call->args[2 + 2 * axis],
                                     "transfer start") == 0 &&
                    ffi::StructuralEqual()(call->args[3 + 2 * axis],
                                           tensor->shape[axis]),
                "ND transfer must cover full Tensor");
        }
        auto dfb = dfbs.at(id);
        Check(dfb->tensor_backing.has_value() &&
                  dfb->tensor_backing.value()->global_arg_index == tensor_id,
              "transfer disagrees with DFB Tensor backing");
        effects[tensor_id] |= input ? 1 : 2;
        if (input) {
          logical_shapes[id] = tensor->shape;
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
    }
  }
  for (const auto &[id, descriptor] : dfbs) {
    Check(producers.count(id) && reserves.count(id),
          "DFB generation is never published");
    Check(use_count[id] > 0, "DFB generation is never consumed");
    if (pipeline.enabled) {
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
    if (call->op.same_as(dfb_compute())) {
      auto shapes = Downcast<ffi::Array<ffi::Array<PrimExpr>>>(
          call->annotations.at("tt.input_shapes"));
      for (size_t i = 1; i < call->args.size(); ++i) {
        int64_t id = RequireStaticInteger(call->args[i], "input DFB ID");
        Check(logical_shapes.count(id) &&
                  ffi::StructuralEqual()(logical_shapes.at(id), shapes[i - 1]),
              "dfb_compute input logical shape disagrees with its producer");
      }
    } else if (call->op.same_as(dfb_to_tensor_nd())) {
      int64_t id = RequireStaticInteger(call->args[0], "export DFB ID");
      int64_t tensor = RequireStaticInteger(call->args[1], "export Tensor ID");
      Check(logical_shapes.count(id) &&
                ffi::StructuralEqual()(logical_shapes.at(id),
                                       tensors.at(tensor)->shape),
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
        int64_t id = RequireStaticInteger(call->args[index], "cycle DFB ID");
        detail = " involving pipeline capacity/release dependencies for DFB " +
                 std::to_string(id) + " storage pool " +
                 std::to_string(pipeline.groups.at(id));
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
  Check(
      !mod->attrs->dict.count(kPipelineRelationsAttr) &&
          !mod->attrs->dict.count(kDFBStorageGroupsAttr) &&
          !mod->attrs->dict.count(kPipelineStagesAttr) &&
          !mod->attrs->dict.count(kPipelineExtentAttr) &&
          !mod->attrs->dict.count("tt.pipeline_wait_policy"),
      "Phase 6 Pipe communication with pipeline storage reuse is unsupported");
  const auto transfers = RequireModuleAttr<ffi::Array<PipeTransferDescriptor>>(
      mod, kPipeTransferTableAttr);
  std::unordered_map<int64_t, PipeTransferDescriptor> transfer_table;
  std::map<std::pair<int64_t, int64_t>, PipeDescriptor> records;
  for (const PipeDescriptor &pipe : pipes)
    records.emplace(std::make_pair(pipe->pipe_net_id, pipe->event_index), pipe);
  std::map<std::pair<int64_t, int64_t>, std::map<CoreKey, int64_t>>
      destinations;
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
  if (budget.has_value()) {
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
    Check(transfer->occurrence == 0,
          "Phase 6 supports exactly one occurrence per Pipe record");
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
    Check(destinations[record_key].emplace(dest, transfer->transfer_id).second,
          "Pipe record has a duplicate destination transaction");
    Check(dfbs.count(transfer->source_dfb_id) &&
              dfbs.count(transfer->destination_dfb_id),
          "Pipe transfer references missing source/destination DFB");
    Check(transfer->source_dfb_id == record->payload_dfb_id,
          "Pipe transfer source DFB disagrees with original record payload");
    DFBDescriptor source = dfbs.at(transfer->source_dfb_id);
    DFBDescriptor destination = dfbs.at(transfer->destination_dfb_id);
    Check(CoreOf(source->consumer_domain) == CoreOf(transfer->src_coord) &&
              CoreOf(destination->producer_domain) == dest,
          "Pipe transfer endpoint owner Core mismatch");
    Check(source->consumer_slot == "brisc" &&
              destination->producer_slot == "ncrisc",
          "Pipe source requires BRISC affinity and receiver requires NCRISC");
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
    Check(destinations[key].size() == static_cast<size_t>(count),
          "Pipe record destination transactions are not closed (missing "
          "receiver)");
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
    std::string slot = func->GetAttr<ffi::String>(kKernelSlotAttr).value();
    CoreKey core = CoreOf(func->GetAttr<CoreDomain>(kCoreDomainAttr).value());
    function_names.emplace(std::make_pair(core, slot), global->name_hint);
    std::unordered_set<int64_t> declared, used_tensors, waited;
    std::unordered_map<int64_t, int64_t> last_record;
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
      const auto *evaluate = statement.as<EvaluateNode>();
      Check(evaluate != nullptr,
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
      if (call->op.same_as(dfb_reserve()) || call->op.same_as(dfb_wait()) ||
          call->op.same_as(dfb_release()) ||
          call->op.same_as(dfb_copy_wait())) {
        Check(call->args.size() == 2 &&
                  RequireStaticInteger(call->args[1], "transaction count") == 1,
              "Phase 6 DFB operation must request exactly one transaction");
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
          Check(
              dfbs.at(id)->consumer_slot == slot && waited.count(id) &&
                  (!uses[id].empty() || discard),
              "DFB release must follow wait and consumption in consumer slot");
          if (discard) {
            Check(recv_completions.count(incoming.at(id)),
                  "Pipe discard must follow receive completion, wait, and "
                  "release");
            discards.insert(id);
          }
          Check(releases.emplace(id, event).second,
                "DFB generation released more than once");
        }
      } else if (call->op.same_as(dfb_compute())) {
        Check(slot == "trisc", "dfb_compute requires TRISC");
        VerifyGeneralCompute(call, dfbs);
        int64_t output = id_at(0);
        logical_shapes[output] = ComputeShape(call, "tt.logical_domain");
        for (size_t i = 1; i < call->args.size(); ++i)
          read(id_at(i));
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
          int64_t start =
              RequireStaticInteger(call->args[2 + 2 * axis], "transfer start");
          int64_t extent =
              RequireStaticInteger(call->args[3 + 2 * axis], "transfer extent");
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
          auto [previous_record, inserted] = last_record.emplace(
              transfer->pipe_net_id, transfer->record_index);
          Check(inserted || previous_record->second <= transfer->record_index,
                "Pipe operations must preserve original foreach record order");
          previous_record->second = transfer->record_index;
        }
        if (call->op.same_as(dfb_pipe_send())) {
          Check(source && slot == "brisc",
                "Pipe send requires source BRISC affinity");
          read(id);
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
          Check(slot == "brisc" && sends.count(tid),
                "Pipe source completion must follow send in BRISC");
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
    }
    Check(declared == used_tensors, "Phase 6 NCRISC Tensor ABI must contain "
                                    "exactly its used Tensor arguments");
  }
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
  for (const auto &[tid, transfer] : transfer_table) {
    Check(sends.count(tid) && receives.count(tid),
          "Pipe transaction producer/consumer matching is not closed");
    Check(send_completions.count(tid) && recv_completions.count(tid),
          "Pipe transaction is missing completion synchronization");
    Check(send_completions.at(tid) < releases.at(transfer->source_dfb_id),
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
    if (call->op.same_as(dfb_compute())) {
      auto shapes = CheckedShapes(call->annotations.at("tt.input_shapes"),
                                  "tt.input_shapes");
      for (size_t i = 1; i < call->args.size(); ++i)
        Check(ffi::StructuralEqual()(
                  shape_of(RequireStaticInteger(call->args[i], "input DFB ID")),
                  shapes[i - 1]),
              "dfb_compute input logical shape disagrees with its producer");
    } else if (call->op.same_as(dfb_to_tensor_nd())) {
      ffi::Array<PrimExpr> shape;
      for (size_t i = 3; i < call->args.size(); i += 2)
        shape.push_back(call->args[i]);
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

IRModule VerifyModule(IRModule mod) {
  Integer version = RequireModuleAttr<Integer>(mod, kDeviceIRVersionAttr);
  Check(version->value == kDeviceIRVersion || version->value == 2 ||
            version->value == 3 || version->value == 4,
        "unsupported tt.device_ir_version " + std::to_string(version->value) +
            "; expected 1, 2, 3, or 4");

  Check(version->value == 4 || !mod->attrs->dict.count(kPipeTransferTableAttr),
        "tt.pipe_transfer_table requires Device IR schema v4");

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
  Check(version->value == 4 ||
            (kernel_order.size() == 3 && kernel_order[0] == "trisc" &&
             kernel_order[1] == "ncrisc" && kernel_order[2] == "brisc"),
        "tt.kernel_order must be exactly [trisc, ncrisc, brisc]");

  const bool general = version->value >= 2;
  TensorTable tensor_table = VerifyTensorTable(tensors, launch_grid, general);
  DFBTable dfb_table = VerifyDFBTable(dfbs, tensor_table, launch_grid, general);
  VerifyPipeTable(pipes, dfb_table, launch_grid);
  if (version->value < 3)
    VerifyLegacyL1Budget(mod, dfb_table);
  if (version->value == 4) {
    VerifyFunctions(mod, target_arch, launch_grid, tensor_table, nullptr, true,
                    true);
    VerifyMulticoreProgram(mod, tensor_table, dfb_table, launch_grid, pipes);
  } else if (general) {
    Check(launch_grid->x == 1 && launch_grid->y == 1 && pipes.empty(),
          "Phase 4 is single-Core without Pipe communication");
    VerifyFunctions(mod, target_arch, launch_grid, tensor_table, nullptr, true);
    VerifyGeneralProgram(mod, tensor_table, dfb_table);
  } else if (dfbs.empty()) {
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

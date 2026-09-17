/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#include "infer_compute_requirements.h"

#include <functional>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {
namespace tenstorrent {
namespace {
using namespace tirx;
void Check(bool valid, const std::string &message) {
  if (!valid)
    TVM_FFI_THROW(ValueError) << "[TenstorrentComputeRequirements] " << message;
}
int64_t IntegerValue(const PrimExpr &value) {
  const auto *imm = value.as<IntImmNode>();
  Check(imm != nullptr,
        "resource IDs, flags and tile extents must be static integers");
  return imm->value;
}
ffi::String StringAnnotation(const Call &call, const char *key) {
  auto value = call->annotations.Get(key);
  Check(value.has_value(), std::string("missing ") + key);
  const auto *text = value.value().as<StringImmNode>();
  Check(text != nullptr, std::string("invalid ") + key);
  return text->value;
}
} // namespace

ComputeRequirements DeriveComputeRequirements(const IRModule &mod,
                                              const tirx::PrimFunc &func) {
  std::map<int64_t, AccumulatorDescriptor> table;
  auto entries =
      mod->GetAttr<ffi::Array<AccumulatorDescriptor>>(kAccumulatorTableAttr);
  // Fragment storage is local to a specialized Core. A frontend Buffer handle
  // may therefore occur in descriptors owned by different compute functions.
  std::map<int64_t, PrimFunc> owners;
  if (entries.has_value()) {
    for (const auto &[global, base] : mod->functions) {
      PrimFunc owner = Downcast<PrimFunc>(base);
      PostOrderVisit(owner->body, [&](const ffi::ObjectRef &node) {
        const auto *call = node.as<CallNode>();
        if (!call || (!call->op.same_as(accumulator_init()) &&
                      !call->op.same_as(gemm_update()) &&
                      !call->op.same_as(accumulator_materialize())))
          return;
        size_t index = call->op.same_as(gemm_update()) ? 2 : 0;
        Check(call->args.size() > index,
              "accumulator operation arity mismatch");
        int64_t id = IntegerValue(call->args[index]);
        auto [it, inserted] = owners.emplace(id, owner);
        Check(inserted || it->second.same_as(owner),
              "accumulator operations cross Core or processor ownership");
      });
    }
  }
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> storage;
  if (entries.has_value()) {
    for (const auto &entry : entries.value()) {
      Check(entry.defined() && entry->accumulator_id >= 0,
            "invalid accumulator descriptor");
      Check(table.emplace(entry->accumulator_id, entry).second,
            "duplicate accumulator ID");
      const auto &region = entry->accumulator_region;
      Check(region.defined() && region->buffer.defined() &&
                region->region.size() == 2 && region->buffer->shape.size() == 2,
            "accumulator requires a rank-2 BufferRegion");
      Check(owners.count(entry->accumulator_id),
            "unused accumulator descriptor");
      if (owners.at(entry->accumulator_id).same_as(func))
        Check(storage.insert(region->buffer->data).second,
              "accumulator descriptors alias the same fragment storage");
      Check(
          region->buffer.scope() == "local.fragment" &&
              region->buffer->dtype == entry->accumulation_dtype,
          "accumulator Buffer must have fragment scope and accumulation dtype");
      Check(IsSupportedAccumulatorDTypeTriple(entry->input_dtype,
                                              entry->accumulation_dtype,
                                              entry->output_dtype),
            "unsupported accumulator dtype triple");
      Check(entry->full_k_tiles > 0 && entry->source_span.defined(),
            "accumulator requires positive full_k_tiles and source_span");
      arith::Analyzer analyzer;
      for (size_t axis = 0; axis < 2; ++axis) {
        int64_t extent = IntegerValue(region->region[axis]->extent);
        Check(extent > 0 && extent % 32 == 0 &&
                  IntegerValue(region->region[axis]->min) % 32 == 0 &&
                  analyzer.CanProve(region->region[axis]->min >= 0) &&
                  analyzer.CanProve(region->region[axis]->min +
                                        region->region[axis]->extent <=
                                    region->buffer->shape[axis]),
              "accumulator region must be tile-aligned and in bounds");
      }
    }
  }
  std::map<int64_t, DFBDescriptor> dfbs;
  auto dfb_table = mod->GetAttr<ffi::Array<DFBDescriptor>>(kDFBTableAttr);
  Check(dfb_table.has_value(), "missing tt.dfb_table");
  for (const auto &dfb : dfb_table.value())
    dfbs.emplace(dfb->dfb_id, dfb);
  ffi::String width = "unconstrained", full = "allowed";
  auto merge = [&](ffi::String next_width, ffi::String next_full) {
    Check(width == "unconstrained" || next_width == "unconstrained" ||
              width == next_width,
          "conflicting hard destination width requirements in one compute "
          "kernel");
    Check(full == "allowed" || next_full == "allowed" || full == next_full,
          "conflicting matmul_full_fp32 requirements in one compute kernel");
    if (next_width != "unconstrained")
      width = next_width;
    if (next_full != "allowed")
      full = next_full;
  };
  std::map<int64_t, int> state;
  std::map<int64_t, int64_t> k_tiles;
  ffi::Array<AccumulatorDescriptor> used;
  const bool compute =
      func->GetAttr<ffi::String>(kKernelSlotAttr).value_or("") == "trisc";
  auto dfb_at = [&](const PrimExpr &arg) {
    int64_t id = IntegerValue(arg);
    Check(dfbs.count(id), "accumulator operation references missing DFB");
    return dfbs.at(id);
  };
  // Traverse lexical occurrences, including repeated handles. PostOrderVisit
  // deduplicates shared nodes and would miss duplicate init/update statements.
  std::vector<Call> calls;
  std::function<void(const Stmt &)> collect = [&](const Stmt &stmt) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : seq->seq)
        collect(child);
    } else if (const auto *evaluate = stmt.as<EvaluateNode>()) {
      if (const auto *call = evaluate->value.as<CallNode>())
        calls.push_back(ffi::GetRef<Call>(call));
    } else if (const auto *loop = stmt.as<ForNode>()) {
      Check(table.empty(),
            "schema v5 requires statically expanded serial K updates");
      collect(loop->body);
    } else {
      Check(table.empty(),
            "schema v5 lifetime must contain only scheduled operations");
    }
  };
  collect(func->body);
  for (const Call &call : calls) {
    bool init = call->op.same_as(accumulator_init());
    bool update = call->op.same_as(gemm_update());
    bool materialize = call->op.same_as(accumulator_materialize());
    if (init || update || materialize) {
      Check(compute, "accumulator operations may execute only in trisc");
      Check(
          call->dtype.is_void() && call->annotations.empty(),
          "accumulator operation must be a void intrinsic without annotations");
      Check(call->args.size() == (update ? 5
                                  : init ? 1
                                         : 2),
            "accumulator operation arity mismatch");
      int64_t id = IntegerValue(call->args[update ? 2 : 0]);
      Check(table.count(id), "operation references missing accumulator");
      const auto &entry = table.at(id);
      if (init) {
        Check(state[id] == 0, "accumulator initialized more than once");
        state[id] = 1;
        used.push_back(entry);
        bool fp32 = entry->accumulation_dtype == DataType::Float(32);
        merge(fp32 ? "bits32_required" : "bits16_required",
              fp32 ? "required" : "forbidden");
      } else if (update) {
        Check(state[id] == 1,
              "GEMM update requires a live initialized accumulator");
        auto lhs = dfb_at(call->args[0]), rhs = dfb_at(call->args[1]);
        int64_t ta = IntegerValue(call->args[3]),
                tb = IntegerValue(call->args[4]);
        Check((ta == 0 || ta == 1) && (tb == 0 || tb == 1),
              "GEMM transpose flags must be Boolean");
        Check(lhs->element_dtype == entry->input_dtype &&
                  rhs->element_dtype == entry->input_dtype,
              "GEMM update input dtype disagrees with accumulator");
        Check(lhs->block_shape_in_tiles.size() == 2 &&
                  rhs->block_shape_in_tiles.size() == 2,
              "GEMM update requires rank-2 DFBs");
        auto a = lhs->block_shape_in_tiles, b = rhs->block_shape_in_tiles;
        int64_t k = IntegerValue(a[ta ? 0 : 1]);
        Check(
            k > 0 && k == IntegerValue(b[tb ? 1 : 0]) &&
                IntegerValue(a[ta ? 1 : 0]) ==
                    IntegerValue(entry->accumulator_region->region[0]->extent) /
                        32 &&
                IntegerValue(b[tb ? 0 : 1]) ==
                    IntegerValue(entry->accumulator_region->region[1]->extent) /
                        32,
            "GEMM update M/N/K shape mismatch");
        Check(k_tiles[id] <= entry->full_k_tiles - k,
              "GEMM updates exceed accumulator full-K extent");
        k_tiles[id] += k;
      } else {
        Check(state[id] == 1 && k_tiles[id] == entry->full_k_tiles,
              "materialization must follow all full-K updates exactly once");
        auto output = dfb_at(call->args[1]);
        Check(output->element_dtype == entry->output_dtype &&
                  output->block_shape_in_tiles.size() == 2,
              "materialized DFB dtype or rank disagrees with accumulator");
        for (size_t axis = 0; axis < 2; ++axis)
          Check(IntegerValue(output->block_shape_in_tiles[axis]) ==
                    IntegerValue(
                        entry->accumulator_region->region[axis]->extent) /
                        32,
                "materialized DFB shape disagrees with accumulator");
        state[id] = 2;
      }
    } else if (call->op.same_as(dfb_compute()) &&
               StringAnnotation(call, "tt.compute_kind") == "gemm") {
      ffi::String accumulation = StringAnnotation(call, "tt.accum_dtype");
      Check(accumulation == "float32" || accumulation == "bfloat16",
            "unsupported GEMM accumulation dtype");
      bool fp32 = accumulation == "float32";
      // Legacy DFB GEMM materializes each operation and does not promise full-K
      // retention.
      merge(fp32 ? "bits32_required" : "bits16_required",
            fp32 ? "allowed" : "forbidden");
    }
  }
  for (const auto &[id, value] : state)
    Check(value == 2, "accumulator lifetime has no final materialization");
  if (compute && entries.has_value()) {
    size_t owned = 0;
    for (const auto &entry : entries.value())
      owned += owners.at(entry->accumulator_id).same_as(func);
    Check(used.size() == owned, "unused accumulator descriptor");
  }
  return ComputeRequirements(width, full, used);
}

tvm::transform::Pass InferTenstorrentComputeRequirements() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext context) {
    IRModule result = mod;
    result.CopyOnWrite();
    for (const auto &[global, base] : mod->functions) {
      auto function = base.as<tirx::PrimFunc>();
      if (!function.has_value())
        continue;
      PrimFunc func = function.value();
      if (func->GetAttr<ffi::String>(kKernelSlotAttr).value_or("") != "trisc")
        continue;
      ComputeRequirements requirements = DeriveComputeRequirements(mod, func);
      auto previous =
          func->GetAttr<ComputeRequirements>(kComputeRequirementsAttr);
      Check(
          !previous.has_value() ||
              ffi::StructuralEqual()(previous.value(), requirements),
          "existing tt.compute_requirements conflicts with Device operations");
      result->Update(global,
                     WithAttr(func, kComputeRequirementsAttr, requirements));
    }
    return result;
  };
  return tvm::transform::CreateModulePass(
      pass_func, 0, "tl.tenstorrent.InferTenstorrentComputeRequirements", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.tenstorrent.transform.InferTenstorrentComputeRequirements",
      InferTenstorrentComputeRequirements);
}
} // namespace tenstorrent
} // namespace tl
} // namespace tvm

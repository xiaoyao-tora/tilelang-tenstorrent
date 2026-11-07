/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file tenstorrent/transform/legalize_tile_ops.cc
 * \brief Consume structured elementwise operations supported by Device TIR.
 */

#include "../../op/copy.h"
#include "../../op/fill.h"
#include "../../op/gemm.h"
#include "../../op/reduce.h"
#include "../../op/region.h"
#include "../../op/transpose.h"
#include "../../op/utils.h"
#include "../ir/device_ir.h"
#include "../op/builtin.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace tenstorrent {

using namespace tirx;

namespace {

bool IsElementwiseBlock(const SBlock &block) {
  auto kind = block->annotations.Get("tl.tt.compute_kind");
  return kind.has_value() && kind.value().as<ffi::String>().has_value() &&
         Downcast<ffi::String>(kind.value()) == "elementwise";
}

[[noreturn]] void ThrowMalformed(const std::string &message) {
  TVM_FFI_THROW(ValueError)
      << "[LegalizeTenstorrentTileOps] malformed input: " << message;
}

[[noreturn]] void ThrowUnsupported(const std::string &message) {
  TVM_FFI_THROW(NotImplementedError)
      << "[LegalizeTenstorrentTileOps] unsupported program: " << message;
}

void RequireMalformed(bool condition, const std::string &message) {
  if (!condition)
    ThrowMalformed(message);
}

// Capability guards use Require; verified-input invariants use
// RequireMalformed.
void Require(bool condition, const std::string &message) {
  if (!condition)
    ThrowUnsupported(message);
}

std::string DTypeName(DataType dtype) {
  std::ostringstream stream;
  stream << dtype;
  return stream.str();
}

void ValidateDType(DataType dtype) {
  Require(dtype.lanes() == 1 &&
              (dtype.is_bfloat16() || (dtype.is_float() && dtype.bits() == 32)),
          "compute supports scalar bfloat16 and float32, got " +
              DTypeName(dtype));
}

void ValidateRegion(const BufferRegion &region, const std::string &owner,
                    bool allow_fragment = false, bool require_full = true) {
  const Buffer &buffer = region->buffer;
  Require(buffer.scope() == "shared" || buffer.scope() == "shared.dyn" ||
              (allow_fragment && buffer.scope() == "local.fragment"),
          owner + " requires a shared DFB buffer");
  ValidateDType(buffer->dtype);
  Require(!buffer->shape.empty() &&
              region->region.size() == buffer->shape.size(),
          owner + " requires a non-scalar full BufferRegion");
  arith::Analyzer analyzer;
  Require(buffer->axis_separators.empty() &&
              analyzer.CanProveEqual(buffer->elem_offset, 0),
          owner +
              " requires compact storage without offsets or axis separators");
  if (!buffer->strides.empty()) {
    Require(buffer->strides.size() == buffer->shape.size(),
            owner + " storage strides must match rank");
    PrimExpr stride = Integer(1);
    for (int axis = static_cast<int>(buffer->shape.size()) - 1; axis >= 0;
         --axis) {
      Require(analyzer.CanProveEqual(buffer->strides[axis], stride),
              owner +
                  " requires compact strides; strided aliases are unsupported");
      stride = stride * buffer->shape[axis];
    }
  }
  for (size_t axis = 0; axis < buffer->shape.size(); ++axis) {
    const auto *extent = buffer->shape[axis].as<IntImmNode>();
    Require(extent && extent->value > 0 &&
                (axis + 2 < buffer->shape.size() || extent->value % 32 == 0 ||
                 (allow_fragment && buffer.scope() == "local.fragment" &&
                  extent->value == 1)),
            owner + " requires static positive batch dimensions and "
                    "tile-aligned final axes");
    Require(is_zero(region->region[axis]->min),
            owner + " region must start at zero");
    if (require_full)
      Require(ffi::StructuralEqual()(region->region[axis]->extent,
                                     buffer->shape[axis]),
              owner + " requires a full BufferRegion");
    else
      Require(analyzer.CanProve(region->region[axis]->extent > 0) &&
                  analyzer.CanProve(region->region[axis]->extent <=
                                    buffer->shape[axis]),
              owner + " region extent must be positive and in bounds");
  }
}

bool IsFullBufferRegion(const BufferRegion &region) {
  if (region->region.size() != region->buffer->shape.size())
    return false;
  for (size_t axis = 0; axis < region->region.size(); ++axis)
    if (!is_zero(region->region[axis]->min) ||
        !ffi::StructuralEqual()(region->region[axis]->extent,
                                region->buffer->shape[axis]))
      return false;
  return true;
}

using Annotations = ffi::Map<ffi::String, ffi::ObjectRef>;

struct GemmAccumulatorPlanEntry {
  Buffer accumulator;
  BufferRegion region;
  DataType input_dtype;
  DataType output_dtype;
  ffi::Array<Annotations> elementwise_updates;
  Span source_span;
};

struct GemmAccumulatorPlan {
  std::vector<GemmAccumulatorPlanEntry> entries;
  std::unordered_map<Call, size_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      operation_lifetimes;

  int Count(const Buffer &buffer) const {
    return std::count_if(
        entries.begin(), entries.end(),
        [&](const auto &entry) { return entry.accumulator.same_as(buffer); });
  }

  bool Contains(const Buffer &buffer) const {
    for (const GemmAccumulatorPlanEntry &entry : entries)
      if (entry.accumulator.same_as(buffer))
        return true;
    return false;
  }

  Annotations Contract(const Call &call) const {
    auto found = operation_lifetimes.find(call);
    RequireMalformed(found != operation_lifetimes.end() &&
                         found->second < entries.size(),
                     "structured accumulator operation has no lifetime plan");
    const GemmAccumulatorPlanEntry &entry = entries[found->second];
    const bool fp32 = entry.accumulator->dtype == DataType::Float(32);
    Annotations annotations{
        {"tt.input_dtype", StringImm(DTypeName(entry.input_dtype))},
        {"tt.accum_dtype", StringImm(DTypeName(entry.accumulator->dtype))},
        {"tt.output_dtype", StringImm(DTypeName(entry.output_dtype))},
        {"tt.dest_precision_requirement",
         StringImm(fp32 ? "bits32_required" : "bits16_required")},
        {"tt.materialization", StringImm("after_complete_k_reduction")}};
    if (!fp32)
      annotations.Set("tt.matmul_full_fp32", StringImm("forbidden"));
    return annotations;
  }
};

Annotations BaseAnnotations(const Buffer &output, const char *kind) {
  return {
      {"tt.compute_kind", StringImm(kind)},
      {"tt.compute_dtype", StringImm(DTypeName(output->dtype))},
      {"tt.compute_tile_shape", ffi::Array<PrimExpr>{Integer(32), Integer(32)}},
      {"tt.logical_domain", output->shape}};
}

// Fragment precision belongs to a complete GEMM lifetime, not to all FP32
// temporaries. Keep this analysis before any DFB generation or loop lowering.
class GemmAccumulatorAnalyzer : public StmtExprVisitor {
  struct Lifetime {
    Buffer buffer;
    BufferRegion region;
    DataType input_dtype;
    bool initialized{false};
    bool updated{false};
    bool materialized{false};
    DataType output_dtype;
    int declaration_depth{-1};
    int64_t core_x{-1};
    int64_t core_y{-1};
    ffi::Array<Annotations> elementwise_updates;
    int initialization_depth{-1};
    bool final_update{false};
    Span source_span;
  };

  class Collector : public StmtExprVisitor {
  public:
    struct Fragment {
      Buffer buffer;
      int64_t core_x;
      int64_t core_y;
      Span source_span;
    };
    std::vector<Fragment> fragments;
    std::vector<Call> calls;
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(Gemm::Get()) || op->op.same_as(Fill::Get()))
        calls.push_back(ffi::GetRef<Call>(op));
      if (op->op.same_as(Gemm::Get())) {
        Gemm gemm = Downcast<Gemm>(ParseOperator(ffi::GetRef<Call>(op)));
        if (gemm->c_.scope() == "local.fragment") {
          for (const Fragment &fragment : fragments)
            if (fragment.buffer.same_as(gemm->c_) &&
                fragment.core_x == core_x_ && fragment.core_y == core_y_)
              return;
          fragments.push_back({gemm->c_, core_x_, core_y_, op->span});
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const SBlockNode *op) final {
      int64_t saved_x = core_x_, saved_y = core_y_;
      if (op->annotations.count("tt.core_x") &&
          op->annotations.count("tt.core_y")) {
        core_x_ = Downcast<Integer>(op->annotations.at("tt.core_x"))->value;
        core_y_ = Downcast<Integer>(op->annotations.at("tt.core_y"))->value;
      }
      StmtExprVisitor::VisitStmt_(op);
      core_x_ = saved_x;
      core_y_ = saved_y;
    }

  private:
    int64_t core_x_{-1}, core_y_{-1};
  };

public:
  static GemmAccumulatorPlan Analyze(const PrimFunc &func) {
    Collector collector;
    collector(func->body);
    if (collector.fragments.empty())
      return {};
    GemmAccumulatorAnalyzer analyzer;
    // Determine whether a syntactic GEMM has another update before the next
    // explicit initialization. Loop-depth checks below still prohibit treating
    // an update inside a K loop as the complete reduction.
    std::unordered_map<Var, bool, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
        pending_update;
    for (auto it = collector.calls.rbegin(); it != collector.calls.rend();
         ++it) {
      const Call &call = *it;
      if (call->op.same_as(Fill::Get())) {
        Fill fill = Downcast<Fill>(ParseOperator(call));
        pending_update.erase(fill->dst->data);
      } else {
        Gemm gemm = Downcast<Gemm>(ParseOperator(call));
        bool final = !pending_update[gemm->c_->data];
        auto [entry, inserted] = analyzer.final_updates_.emplace(call, final);
        if (!inserted)
          entry->second &= final;
        pending_update[gemm->c_->data] = !is_one(gemm->clearAccum_);
      }
    }
    auto metadata =
        func->GetAttr<ffi::Array<TTBufferMetadata>>(kBufferMetadataTableAttr);
    for (const Collector::Fragment &fragment : collector.fragments) {
      const Buffer &buffer = fragment.buffer;
      RequireMalformed(
          metadata.has_value(),
          "GEMM accumulator verification requires normalized buffer metadata");
      int matches = 0;
      for (const TTBufferMetadata &entry : metadata.value()) {
        if (entry->buffer->data.same_as(buffer->data)) {
          RequireMalformed(
              entry->buffer.same_as(buffer) &&
                  entry->kind == "compute_fragment" &&
                  !entry->alias_of.has_value(),
              "GEMM accumulator must resolve to one unique compute_fragment; "
              "aliases are unsupported");
          ++matches;
        }
      }
      RequireMalformed(
          matches == 1,
          "GEMM accumulator must resolve to one unique compute_fragment");
      ValidateDType(buffer->dtype);
      Require(buffer->shape.size() == 2,
              "GEMM accumulator requires a rank-2 fragment");
      analyzer.lifetimes_.push_back({buffer, BufferRegion::FullRegion(buffer),
                                     DataType::Void(), false, false, false,
                                     DataType::Void(), -1, fragment.core_x,
                                     fragment.core_y});
      analyzer.lifetimes_.back().source_span = fragment.source_span;
      analyzer.active_lifetimes_.push_back(analyzer.lifetimes_.size() - 1);
    }
    if (!analyzer.lifetimes_.empty()) {
      auto slot = func->GetAttr<ffi::String>(kKernelSlotAttr);
      Require(!slot.has_value() || slot.value() == "trisc",
              "GEMM fragment cannot cross processor slots; materialize through "
              "an output DFB");
      analyzer(func->body);
    }
    GemmAccumulatorPlan plan;
    for (const Lifetime &lifetime : analyzer.lifetimes_) {
      RequireMalformed(
          lifetime.updated && lifetime.materialized,
          "GEMM accumulator must have one final materialization after "
          "its complete K reduction");
      plan.entries.push_back({lifetime.buffer, lifetime.region,
                              lifetime.input_dtype, lifetime.output_dtype,
                              lifetime.elementwise_updates,
                              lifetime.source_span});
    }
    plan.operation_lifetimes = std::move(analyzer.operation_lifetimes_);
    return plan;
  }

private:
  Lifetime *Find(const Buffer &buffer) {
    for (size_t index : active_lifetimes_) {
      Lifetime &lifetime = lifetimes_[index];
      // Topology specialization reuses lexical Buffer handles across Core
      // blocks. Their storage and reduction lifetimes remain Core-local.
      if (lifetime.core_x != core_x_ || lifetime.core_y != core_y_)
        continue;
      if (buffer->data.same_as(lifetime.buffer->data)) {
        RequireMalformed(
            buffer.same_as(lifetime.buffer),
            "GEMM accumulator alias makes fragment identity ambiguous");
        return &lifetime;
      }
    }
    return nullptr;
  }

  std::vector<size_t> Origins(const Buffer &buffer) {
    if (Lifetime *lifetime = Find(buffer))
      return {static_cast<size_t>(lifetime - lifetimes_.data())};
    auto found = provenance_.find(buffer);
    return found == provenance_.end() ? std::vector<size_t>{} : found->second;
  }

  size_t Index(const Lifetime *lifetime) const {
    return static_cast<size_t>(lifetime - lifetimes_.data());
  }

  void Materialize(size_t origin, const Buffer &output) {
    Lifetime &lifetime = lifetimes_[origin];
    CheckContext();
    RequireMalformed(
        loop_depth_ == lifetime.initialization_depth,
        "GEMM accumulator cannot materialize inside its K reduction "
        "loop; pack only after all updates");
    RequireMalformed(
        lifetime.updated,
        "GEMM accumulator requires final materialization after all updates");
    Require(IsSupportedAccumulatorDTypeTriple(
                lifetime.input_dtype, lifetime.buffer->dtype, output->dtype),
            "GEMM capability registry has no (input=" +
                DTypeName(lifetime.input_dtype) +
                ", accum=" + DTypeName(lifetime.buffer->dtype) +
                ", output=" + DTypeName(output->dtype) + ") combination");
    RequireMalformed(
        !lifetime.materialized || lifetime.output_dtype == output->dtype,
        "all GEMM accumulator materializations must use one output dtype");
    lifetime.output_dtype = output->dtype;
    lifetime.materialized = true;
  }

  void CheckContext() {
    Require(conditional_depth_ == 0,
            "GEMM accumulator conditional access has no proven dominating "
            "initialization or final materialization");
    Require(unsupported_loop_depth_ == 0,
            "GEMM accumulator requires static positive serial K loops");
  }

  void CheckRegion(const BufferRegion &region, const Lifetime &lifetime) {
    RequireMalformed(
        ffi::StructuralEqual()(region, lifetime.region),
        "GEMM accumulator updates/materialization must preserve the same "
        "full BufferRegion");
    arith::Analyzer analyzer;
    Require(region->buffer->strides.empty() &&
                region->buffer->axis_separators.empty() &&
                analyzer.CanProveEqual(region->buffer->elem_offset, 0),
            "GEMM accumulator aliases and storage offsets are unsupported");
  }

  Lifetime *Initialize(Lifetime *lifetime, const Span &span) {
    CheckContext();
    // Independent outer-loop reductions initialize and publish at one depth.
    // A clear inside a K loop followed by an outer store remains invalid.
    if (lifetime->materialized) {
      size_t previous = Index(lifetime);
      Lifetime next;
      next.buffer = lifetime->buffer;
      next.region = lifetime->region;
      next.declaration_depth = lifetime->declaration_depth;
      next.core_x = lifetime->core_x;
      next.core_y = lifetime->core_y;
      next.source_span = span;
      lifetimes_.push_back(std::move(next));
      size_t current = lifetimes_.size() - 1;
      auto active = std::find(active_lifetimes_.begin(),
                              active_lifetimes_.end(), previous);
      ICHECK(active != active_lifetimes_.end());
      *active = current;
      lifetime = &lifetimes_[current];
    }
    RequireMalformed(
        !lifetime->initialized && !lifetime->updated,
        "GEMM accumulator requires unique initialization; an ordinary "
        "write must not overwrite its live value");
    lifetime->initialized = true;
    lifetime->initialization_depth = loop_depth_;
    if (!lifetime->source_span.defined())
      lifetime->source_span = span;
    return lifetime;
  }

  void VisitStmt_(const EvaluateNode *op) final {
    const auto *node = op->value.as<CallNode>();
    if (!node) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    Call call = ffi::GetRef<Call>(node);
    if (call->op.same_as(Gemm::Get())) {
      Gemm gemm = Downcast<Gemm>(ParseOperator(call));
      Lifetime *lifetime = Find(gemm->c_);
      if (lifetime) {
        CheckContext();
        CheckRegion(gemm->cRegion_, *lifetime);
        Require(call->annotations.empty(),
                "GEMM scheduling/precision annotations are unsupported");
        Require(gemm->m_ > 0 && gemm->n_ > 0 && gemm->k_ > 0 &&
                    gemm->m_ % 32 == 0 && gemm->n_ % 32 == 0 &&
                    gemm->k_ % 32 == 0,
                "GEMM accumulator M/N/K must be positive multiples of 32");
        RequireMalformed(
            gemm->a_->shape.size() == 2 && gemm->b_->shape.size() == 2 &&
                is_const_int(gemm->a_->shape[gemm->transA_ ? 1 : 0],
                             gemm->m_) &&
                is_const_int(gemm->a_->shape[gemm->transA_ ? 0 : 1],
                             gemm->k_) &&
                is_const_int(gemm->b_->shape[gemm->transB_ ? 1 : 0],
                             gemm->k_) &&
                is_const_int(gemm->b_->shape[gemm->transB_ ? 0 : 1],
                             gemm->n_) &&
                is_const_int(gemm->c_->shape[0], gemm->m_) &&
                is_const_int(gemm->c_->shape[1], gemm->n_),
            "GEMM accumulator M/N/K disagree with full operand shapes");

        RequireMalformed(
            !lifetime->materialized || is_one(gemm->clearAccum_),
            "GEMM update after final materialization would lose its "
            "complete K reduction lifetime");
        RequireMalformed(gemm->a_->dtype == gemm->b_->dtype,
                         "GEMM requires matching input storage dtypes");
        ValidateRegion(gemm->aRegion_, "GEMM matrix input A", true);
        ValidateRegion(gemm->bRegion_, "GEMM matrix input B", true);
        RequireMalformed(!Find(gemm->a_) && !Find(gemm->b_),
                         "GEMM matrix inputs must not alias an accumulator");
        Require(is_zero(gemm->clearAccum_) || is_one(gemm->clearAccum_),
                "GEMM clear_accum must be static");
        if (is_one(gemm->clearAccum_))
          lifetime = Initialize(lifetime, call->span);
        RequireMalformed(
            lifetime->initialized,
            "GEMM accumulator first update requires dominating T.clear "
            "or clear_accum=True");
        RequireMalformed(
            !lifetime->updated || lifetime->input_dtype == gemm->a_->dtype,
            "GEMM accumulator input dtype changed during K updates");
        lifetime->input_dtype = gemm->a_->dtype;
        lifetime->updated = true;
        lifetime->final_update = final_updates_.at(call);
        operation_lifetimes_[call] = Index(lifetime);
        return;
      }
    } else if (call->op.same_as(Fill::Get())) {
      Fill fill = Downcast<Fill>(ParseOperator(call));
      if (Lifetime *lifetime = Find(fill->dst)) {
        CheckRegion(BufferRegion(fill->dst, fill->region), *lifetime);
        // Fill preserves the explicit dtype conversion around T.clear's
        // integer zero. Casting zero to either supported fragment dtype is
        // exact; do not infer initialization from an arbitrary expression.
        PrimExpr initializer = fill->value;
        while (const auto *cast = initializer.as<CastNode>())
          initializer = cast->value;
        const auto *float_value = initializer.as<FloatImmNode>();
        RequireMalformed(
            is_zero(initializer) || (float_value && float_value->value == 0.0),
            "GEMM accumulator initialization requires T.clear (zero fill)");
        lifetime = Initialize(lifetime, call->span);
        operation_lifetimes_[call] = Index(lifetime);
        return;
      }
      provenance_.erase(fill->dst);
    } else if (call->op.same_as(ReduceOp::Get())) {
      ReduceOp reduce = Downcast<ReduceOp>(ParseOperator(call));
      RequireMalformed(!Find(reduce->dst),
                       "Reduction must not overwrite a live GEMM accumulator");
      // A reduction consumes an exact-dtype snapshot of the completed GEMM.
      // Subsequent values are ordinary compute results; Device def-use still
      // retains their numerical provenance and borrowed storage dependencies.
      for (size_t origin : Origins(reduce->src))
        Materialize(origin, reduce->src);
      provenance_.erase(reduce->dst);
      return;
    } else if (call->op.same_as(Copy::Get())) {
      Copy copy = Downcast<Copy>(ParseOperator(call));
      RequireMalformed(!Find(copy->dst),
                       "ordinary copy must not write a GEMM accumulator");
      std::vector<size_t> origins = Origins(copy->src);
      if (!origins.empty()) {
        if (Lifetime *lifetime = Find(copy->src)) {
          CheckRegion(BufferRegion(copy->src, copy->src_range), *lifetime);
          operation_lifetimes_[call] = Index(lifetime);
        }
        RequireMalformed(copy->src_range.size() == copy->dst_range.size(),
                         "GEMM output materialization rank mismatch");
        arith::Analyzer analyzer;
        for (size_t i = 0; i < copy->dst_range.size(); ++i)
          RequireMalformed(analyzer.CanProveEqual(copy->dst_range[i]->extent,
                                                  copy->src_range[i]->extent),
                           "GEMM output materialization shape mismatch");
        if (copy->dst.scope() == "local.fragment") {
          provenance_[copy->dst] = origins;
        } else {
          Require(copy->dst.scope() == "shared" ||
                      copy->dst.scope() == "shared.dyn" ||
                      copy->dst.scope() == "global" ||
                      copy->dst.scope().empty(),
                  "GEMM fragment must materialize to output DFB or Tensor");
          for (size_t origin : origins)
            Materialize(origin, copy->dst);
        }
        return;
      }
      provenance_.erase(copy->dst);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Require(Origins(op->buffer).empty(),
            "GEMM accumulator may only be read by GEMM updates and final copy; "
            "ordinary reads or cross-slot uses are unsupported");
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode *op) final {
    Var var = ffi::GetRef<Var>(op);
    for (const Lifetime &lifetime : lifetimes_)
      Require(!var.same_as(lifetime.buffer->data),
              "opaque pointer access to a GEMM accumulator is unsupported; "
              "use GEMM updates and final copy");
    for (const auto &entry : provenance_)
      Require(
          !var.same_as(entry.first->data),
          "opaque pointer access to a GEMM epilogue fragment is unsupported");
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    RequireMalformed(
        !Find(op->buffer),
        "ordinary write must not overwrite a live GEMM accumulator");
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    arith::Analyzer analyzer;
    PrimExpr condition = analyzer.Simplify(op->condition);
    if (is_one(condition)) {
      VisitStmt(op->then_case);
      return;
    }
    if (is_zero(condition)) {
      if (op->else_case.has_value())
        VisitStmt(op->else_case.value());
      return;
    }
    VisitExpr(op->condition);
    auto entry = lifetimes_;
    auto entry_active = active_lifetimes_;
    auto entry_provenance = provenance_;
    auto entry_operations = operation_lifetimes_;
    VisitStmt(op->then_case);
    auto then_lifetimes = lifetimes_;
    auto then_active = active_lifetimes_;
    auto then_provenance = provenance_;
    auto then_operations = operation_lifetimes_;
    lifetimes_ = entry;
    active_lifetimes_ = entry_active;
    provenance_ = entry_provenance;
    operation_lifetimes_ = entry_operations;
    if (op->else_case.has_value())
      VisitStmt(op->else_case.value());
    Require(lifetimes_.size() == then_lifetimes.size() &&
                active_lifetimes_ == then_active,
            "GEMM accumulator branches create different lifetime boundaries");
    for (size_t i = 0; i < lifetimes_.size(); ++i) {
      Lifetime &merged = lifetimes_[i];
      const Lifetime &then_value = then_lifetimes[i];
      Require(merged.initialized == then_value.initialized,
              "GEMM accumulator conditional access has no proven dominating "
              "initialization or final materialization");
      Require(!merged.initialized || merged.initialization_depth ==
                                         then_value.initialization_depth,
              "GEMM accumulator branch initialization loop depths disagree");
      Require(!merged.updated || !then_value.updated ||
                  merged.input_dtype == then_value.input_dtype,
              "GEMM accumulator branch merge has incompatible input dtypes");
      merged.updated &= then_value.updated;
      merged.final_update &= then_value.final_update;
      merged.materialized &= then_value.materialized;
      if (merged.materialized)
        Require(merged.output_dtype == then_value.output_dtype,
                "GEMM accumulator branch merge has incompatible output dtypes");
      for (const Annotations &update : then_value.elementwise_updates)
        merged.elementwise_updates.push_back(update);
    }
    for (auto it = provenance_.begin(); it != provenance_.end();) {
      auto other = then_provenance.find(it->first);
      if (other == then_provenance.end() || other->second != it->second)
        it = provenance_.erase(it);
      else
        ++it;
    }
    for (const auto &[call, lifetime] : then_operations) {
      auto [found, inserted] = operation_lifetimes_.emplace(call, lifetime);
      Require(inserted || found->second == lifetime,
              "one accumulator operation maps to different branch lifetimes");
    }
  }

  void VisitStmt_(const SBlockNode *op) final {
    int64_t saved_x = core_x_, saved_y = core_y_;
    if (op->annotations.count("tt.core_x") &&
        op->annotations.count("tt.core_y")) {
      core_x_ = Downcast<Integer>(op->annotations.at("tt.core_x"))->value;
      core_y_ = Downcast<Integer>(op->annotations.at("tt.core_y"))->value;
    }
    for (const Buffer &buffer : op->alloc_buffers)
      if (Lifetime *lifetime = Find(buffer))
        lifetime->declaration_depth = loop_depth_;
    if (IsElementwiseBlock(ffi::GetRef<SBlock>(op))) {
      // The compute verifier has proved a pure, single-store expression and
      // complete effects. Read the old value before recording the new value;
      // keep the relation for future SSA/materialization consumers.
      const auto *store = op->body.as<BufferStoreNode>();
      ICHECK(store != nullptr);
      Lifetime *output = Find(store->buffer);
      bool reads_output = false;
      std::vector<size_t> origins;
      for (const BufferRegion &read : op->reads) {
        if (Lifetime *input = Find(read->buffer)) {
          CheckContext();
          CheckRegion(read, *input);
          RequireMalformed(
              input->updated,
              "GEMM Tiles read requires a live initialized accumulator "
              "before final materialization");
          reads_output |= input == output;
        }
        for (size_t origin : Origins(read->buffer)) {
          if (std::find(origins.begin(), origins.end(), origin) ==
              origins.end())
            origins.push_back(origin);
        }
      }
      if (!origins.empty()) {
        if (store->buffer.scope() == "local.fragment") {
          if (!output) {
            // A distinct value may consume a completed GEMM without keeping
            // that GEMM's mutable accumulator alive until the derived value
            // is exported. The boundary is exact-dtype and at
            // the GEMM initialization depth, never inside its K update loop.
            for (auto it = origins.begin(); it != origins.end();) {
              const Lifetime &origin = lifetimes_[*it];
              if (origin.final_update &&
                  origin.initialization_depth == loop_depth_ &&
                  origin.buffer->dtype == store->buffer->dtype &&
                  IsSupportedAccumulatorDTypeTriple(origin.input_dtype,
                                                    origin.buffer->dtype,
                                                    store->buffer->dtype)) {
                Materialize(*it, store->buffer);
                it = origins.erase(it);
              } else {
                ++it;
              }
            }
          }
          if (!output)
            provenance_[store->buffer] = origins;
        } else {
          Require(store->buffer.scope() == "shared" ||
                      store->buffer.scope() == "shared.dyn",
                  "GEMM Tiles epilogue must materialize to an output DFB");
          for (size_t origin : origins)
            Materialize(origin, store->buffer);
        }
      } else {
        provenance_.erase(store->buffer);
      }
      if (output) {
        RequireMalformed(
            reads_output,
            "ordinary write must not overwrite a live GEMM accumulator; "
            "Tiles update must read its old value");
        CheckRegion(op->writes[0], *output);
        output->elementwise_updates.push_back(
            {{"accumulator", output->buffer},
             {"region", output->region},
             {"expression", store->value},
             {"relation", StringImm("pointwise_old_to_new")}});
      }
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
    core_x_ = saved_x;
    core_y_ = saved_y;
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    if (Lifetime *lifetime = Find(op->buffer))
      lifetime->declaration_depth = loop_depth_;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileNode *op) final {
    ++unsupported_loop_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --unsupported_loop_depth_;
  }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    bool conditional = !is_one(op->predicate);
    conditional_depth_ += conditional;
    StmtExprVisitor::VisitStmt_(op);
    conditional_depth_ -= conditional;
  }

  void VisitStmt_(const ForNode *op) final {
    const auto *extent = op->extent.as<IntImmNode>();
    bool unsupported =
        op->kind != ForKind::kSerial || !extent || extent->value <= 0;
    ++loop_depth_;
    unsupported_loop_depth_ += unsupported;
    StmtExprVisitor::VisitStmt_(op);
    unsupported_loop_depth_ -= unsupported;
    --loop_depth_;
  }

  std::vector<Lifetime> lifetimes_;
  std::vector<size_t> active_lifetimes_;
  std::unordered_map<Buffer, std::vector<size_t>, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      provenance_;
  std::unordered_map<Call, size_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      operation_lifetimes_;
  std::unordered_map<Call, bool, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      final_updates_;
  int64_t core_x_{-1}, core_y_{-1};
  int loop_depth_{0};
  int conditional_depth_{0};
  int unsupported_loop_depth_{0};
};

Stmt MakeCompute(const BufferRegion &output,
                 const ffi::Array<BufferRegion> &inputs,
                 Annotations annotations, Span span,
                 bool allow_tensor_output = false) {
  if (allow_tensor_output &&
      (output->buffer.scope() == "global" || output->buffer.scope().empty()))
    ValidateDType(output->buffer->dtype);
  else
    ValidateRegion(output, "compute output", true);
  ffi::Array<PrimExpr> arguments{output->ToPrimExpr()};
  ffi::Array<ffi::Array<PrimExpr>> input_shapes;
  for (const BufferRegion &input : inputs) {
    ValidateRegion(input, "compute input", true, false);
    arguments.push_back(input->ToPrimExpr());
    ffi::Array<PrimExpr> shape;
    for (const Range &range : input->region)
      shape.push_back(range->extent);
    input_shapes.push_back(shape);
  }
  annotations.Set("tt.input_shapes", input_shapes);
  return Evaluate(Call(DataType::Handle(), tile_compute(), arguments,
                       std::move(annotations), span),
                  span);
}

class ExplicitBroadcastBuilder : public ExprMutator {
public:
  ExplicitBroadcastBuilder(const Array<BufferRegion> &reads,
                           Array<PrimExpr> domain)
      : domain_(std::move(domain)) {
    for (const BufferRegion &region : reads)
      RequireMalformed(regions_.emplace(region->buffer->data, region).second,
                       "elementwise reads contain duplicate Buffer identities");
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto found = regions_.find(op->buffer->data);
    RequireMalformed(
        found != regions_.end() && found->second->buffer.same_as(op->buffer),
        "elementwise expression reads a Buffer missing from block reads");
    const BufferRegion &region = found->second;
    RequireMalformed(region->region.size() == domain_.size(),
                     "elementwise read rank must match logical domain");
    ffi::Array<PrimExpr> axes;
    for (size_t axis = 0; axis < domain_.size(); ++axis) {
      RequireMalformed(is_zero(region->region[axis]->min),
                       "elementwise read region must start at zero");
      if (ffi::StructuralEqual()(region->region[axis]->extent, domain_[axis]))
        continue;
      Require(is_one(region->region[axis]->extent),
              "elementwise read extent must match the domain or be one");
      axes.push_back(Integer(axis));
    }
    PrimExpr load = ffi::GetRef<BufferLoad>(op);
    if (axes.empty())
      return load;
    ffi::Array<PrimExpr> args{load};
    for (const PrimExpr &axis : axes)
      args.push_back(axis);
    return Call(op->dtype, tenstorrent::compute_broadcast(), args, {},
                op->span);
  }

private:
  std::unordered_map<Var, BufferRegion, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      regions_;
  Array<PrimExpr> domain_;
};

class ComputeDTypeVerifier : public ExprVisitor {
public:
  void VisitExpr(const PrimExpr &expr) final {
    // CanonicalizeTTElementwise only preserves Vars from enclosing static
    // scalar loops. Form replaces them by integer constants; storage and
    // arithmetic nodes still require FP dtypes.
    if (!expr.as<IntImmNode>() && !expr.as<VarNode>() &&
        !expr.dtype().is_bool())
      ValidateDType(expr.dtype());
    ExprVisitor::VisitExpr(expr);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    // Template coordinates are address metadata, not scalar DAG values.
  }
};

// Each buffer has exactly one access map in an elementwise expression. The
// canonical verifier establishes this invariant before any binders disappear.
// Scalar DAG nodes keep their own dtype, including all intermediate cast and
// rounding points; program formation only renames their buffer leaves.
class StructuredComputeLowerer : public StmtExprMutator {
public:
  StructuredComputeLowerer(const PrimFunc &func,
                           const GemmAccumulatorPlan &accumulator_plan)
      : accumulator_plan_(accumulator_plan) {
    std::unordered_map<Buffer, int, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
        copies;
    PostOrderVisit(func->body, [&](const ffi::ObjectRef &object) {
      if (auto call = object.as<Call>()) {
        if (call.value()->op.same_as(Copy::Get())) {
          Copy copy = Downcast<Copy>(ParseOperator(call.value()));
          if (copy->src.scope() == "local.fragment")
            has_fragment_compute_ |=
                !IsAccumulator(copy->src) ||
                ++copies[copy->src] > AccumulatorLifetimes(copy->src);
          has_fragment_compute_ |= copy->dst.scope() == "local.fragment";
        } else if (call.value()->op.same_as(Fill::Get())) {
          Fill fill = Downcast<Fill>(ParseOperator(call.value()));
          has_fragment_compute_ |= fill->dst.scope() == "local.fragment" &&
                                   !IsAccumulator(fill->dst);
        }
      }
      if (auto block = object.as<SBlock>()) {
        if (!IsElementwiseBlock(block.value()))
          return;
        for (const BufferRegion &region : block.value()->reads)
          has_fragment_compute_ |= region->buffer.scope() == "local.fragment";
        for (const BufferRegion &region : block.value()->writes)
          has_fragment_compute_ |= region->buffer.scope() == "local.fragment";
      }
    });
  }

private:
  bool IsAccumulator(const Buffer &buffer) const {
    return accumulator_plan_.Contains(buffer);
  }

  int AccumulatorLifetimes(const Buffer &buffer) const {
    return accumulator_plan_.Count(buffer);
  }

  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    const SBlock &block = op->block;
    if (!IsElementwiseBlock(block))
      return StmtExprMutator::VisitStmt_(op);
    const auto *store = block->body.as<BufferStoreNode>();
    ICHECK(store != nullptr);
    const Buffer &output = store->buffer;
    PostOrderVisit(store->value, [&](const ffi::ObjectRef &object) {
      if (auto expr = object.as<PrimExpr>()) {
        if (!expr.value().as<IntImmNode>() && !expr.value().as<VarNode>()) {
          DataType dtype = expr.value().dtype();
          if (dtype.lanes() != 1 ||
              (dtype != DataType::BFloat(16) && dtype != DataType::Float(32) &&
               !dtype.is_bool()))
            ThrowUnsupported(
                "structured elementwise expression supports scalar bfloat16, "
                "float32 and bool values, got " +
                DTypeName(dtype) + " in output buffer '" +
                std::string(output->name) + "'");
        }
      }
    });
    ValidateDType(output->dtype);
    Span span = block->span.defined() ? block->span : op->span;
    ComputeDTypeVerifier dtype_verifier;
    dtype_verifier(store->value);
    RequireMalformed(
        block->writes.size() == 1 && block->writes[0]->buffer.same_as(output),
        "structured elementwise block requires one matching output region");
    ffi::Array<BufferRegion> inputs = block->reads;
    const char *kind = inputs.empty()                ? "fill"
                       : store->value.as<CastNode>() ? "typecast"
                                                     : "elementwise";
    Annotations annotations = BaseAnnotations(output, kind);
    auto domain = Downcast<ffi::Array<PrimExpr>>(
        block->annotations.at("tl.tt.logical_domain"));
    ExplicitBroadcastBuilder broadcasts(inputs, domain);
    annotations.Set("tt.expression", broadcasts(store->value));
    return MakeCompute(block->writes[0], inputs, annotations, span);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *node = op->value.as<CallNode>();
    if (!node)
      return StmtExprMutator::VisitStmt_(op);
    Call call = ffi::GetRef<Call>(node);
    Span span = call->span.defined() ? call->span : op->span;
    if (call->op.same_as(Fill::Get())) {
      Fill fill = Downcast<Fill>(ParseOperator(call));
      if (fill->dst.scope() == "local.fragment") {
        if (!IsAccumulator(fill->dst)) {
          Require(call->annotations.empty(),
                  "Fill scheduling annotations are unsupported");
          ComputeDTypeVerifier dtype_verifier;
          dtype_verifier(cast(fill->dst->dtype, fill->value));
          Annotations annotations = BaseAnnotations(fill->dst, "fill");
          annotations.Set("tt.expression", cast(fill->dst->dtype, fill->value));
          return MakeCompute(BufferRegion(fill->dst, fill->region), {},
                             annotations, span);
        }
        Annotations contract = accumulator_plan_.Contract(call);
        return MakeAccumulatorCompute(BufferRegion(fill->dst, fill->region), {},
                                      "accumulator_init", std::move(contract),
                                      span);
      }
      Require(call->annotations.empty(),
              "Fill scheduling annotations are unsupported");
      Require(call->args[1].as<IntImmNode>() ||
                  call->args[1].as<FloatImmNode>(),
              "Fill requires a scalar constant");
      Annotations annotations = BaseAnnotations(fill->dst, "fill");
      annotations.Set("tt.expression", cast(fill->dst->dtype, fill->value));
      return MakeCompute(BufferRegion(fill->dst, fill->region), {}, annotations,
                         span);
    }
    if (call->op.same_as(Transpose::Get())) {
      Transpose transpose = Downcast<Transpose>(ParseOperator(call));
      Require(call->annotations.empty(),
              "Transpose scheduling annotations are unsupported");
      RequireMalformed(!transpose->src->data.same_as(transpose->dst->data),
                       "Transpose source/destination alias requires a separate "
                       "temporary buffer");
      size_t rank = transpose->src->shape.size();
      Require(rank >= 2 && transpose->dst->shape.size() == rank,
              "Transpose requires equal rank >= 2");
      RequireMalformed(transpose->src->dtype == transpose->dst->dtype,
                       "Transpose requires matching input/output dtype");
      ffi::Array<Integer> axes;
      for (size_t axis = 0; axis < rank; ++axis) {
        size_t mapped =
            axis + 2 < rank ? axis : (axis == rank - 2 ? rank - 1 : rank - 2);
        RequireMalformed(
            ffi::StructuralEqual()(transpose->dst->shape[axis],
                                   transpose->src->shape[mapped]),
            "Transpose output shape must exchange the final two input axes");
        axes.push_back(Integer(mapped));
      }
      Annotations annotations = BaseAnnotations(transpose->dst, "transpose");
      annotations.Set("tt.axes", axes);
      return MakeCompute(BufferRegion(transpose->dst, transpose->dst_range),
                         {BufferRegion(transpose->src, transpose->src_range)},
                         annotations, span);
    }
    if (call->op.same_as(Gemm::Get())) {
      Gemm gemm = Downcast<Gemm>(ParseOperator(call));
      Require(call->annotations.empty(),
              "GEMM scheduling/precision annotations are unsupported");
      Require(
          gemm->a_->dtype == gemm->b_->dtype &&
              (gemm->c_->dtype == DataType::Float(32) ||
               (gemm->a_->dtype == DataType::BFloat(16) &&
                gemm->c_->dtype == DataType::BFloat(16))),
          "GEMM requires equal input dtype and supported accumulation/output");
      RequireMalformed(!gemm->c_->data.same_as(gemm->a_->data) &&
                           !gemm->c_->data.same_as(gemm->b_->data),
                       "GEMM output must not alias its matrix inputs");
      Require(gemm->a_->shape.size() == 2 && gemm->b_->shape.size() == 2 &&
                  gemm->c_->shape.size() == 2,
              "GEMM currently requires rank-2 full matrices; batched GEMM is "
              "unsupported");
      Require(gemm->m_ > 0 && gemm->n_ > 0 && gemm->k_ > 0 &&
                  gemm->m_ % 32 == 0 && gemm->n_ % 32 == 0 &&
                  gemm->k_ % 32 == 0,
              "GEMM M/N/K must be positive multiples of 32");
      Require((is_zero(gemm->clearAccum_) || is_one(gemm->clearAccum_)) &&
                  gemm->kPack_ == 1 && gemm->wgWait_ == 0 && !gemm->isWgmma_ &&
                  !gemm->isTcgen05_ && !gemm->mbar_.defined() &&
                  !gemm->sfaRegion_.defined() && !gemm->sfbRegion_.defined(),
              "GEMM requires static clear_accum and default scheduling without "
              "target-specific extensions");
      int64_t m = gemm->m_, n = gemm->n_, k = gemm->k_;
      auto equal = [](const PrimExpr &value, int64_t expected) {
        return is_const_int(value, expected);
      };
      RequireMalformed(
          equal(gemm->a_->shape[gemm->transA_ ? 1 : 0], m) &&
              equal(gemm->a_->shape[gemm->transA_ ? 0 : 1], k) &&
              equal(gemm->b_->shape[gemm->transB_ ? 1 : 0], k) &&
              equal(gemm->b_->shape[gemm->transB_ ? 0 : 1], n) &&
              equal(gemm->c_->shape[0], m) && equal(gemm->c_->shape[1], n),
          "GEMM M/N/K and transpose flags disagree with full buffer shapes");
      Annotations annotations = BaseAnnotations(gemm->c_, "gemm");
      annotations.Set("tt.transpose_a", Integer(gemm->transA_));
      annotations.Set("tt.transpose_b", Integer(gemm->transB_));
      annotations.Set("tt.clear", Integer(is_one(gemm->clearAccum_)));
      annotations.Set("tt.input_dtype", StringImm(DTypeName(gemm->a_->dtype)));
      annotations.Set("tt.accum_dtype", StringImm(DTypeName(gemm->c_->dtype)));
      annotations.Set("tt.output_dtype", StringImm(DTypeName(gemm->c_->dtype)));
      const bool fp32 = gemm->c_->dtype == DataType::Float(32);
      annotations.Set("tt.dest_precision_requirement",
                      StringImm(fp32 ? "bits32_required" : "bits16_required"));
      if (!fp32)
        annotations.Set("tt.matmul_full_fp32", StringImm("forbidden"));
      ffi::Array<BufferRegion> inputs{gemm->aRegion_, gemm->bRegion_};
      if (gemm->c_.scope() == "local.fragment") {
        RequireMalformed(IsAccumulator(gemm->c_),
                         "fragment GEMM requires a verified accumulator plan");
        Annotations contract = accumulator_plan_.Contract(call);
        for (const auto &[key, value] : contract)
          annotations.Set(key, value);
        Stmt update = MakeAccumulatorCompute(gemm->cRegion_, inputs,
                                             "gemm_update", annotations, span);
        if (is_one(gemm->clearAccum_))
          return SeqStmt(
              {MakeAccumulatorCompute(gemm->cRegion_, {}, "accumulator_init",
                                      std::move(contract), span),
               update},
              span);
        return update;
      }
      if (!is_one(gemm->clearAccum_))
        inputs.push_back(gemm->cRegion_);
      return MakeCompute(gemm->cRegion_, inputs, annotations, span);
    }
    if (call->op.same_as(ReduceOp::Get())) {
      ReduceOp reduce = Downcast<ReduceOp>(ParseOperator(call));
      for (const auto &entry : call->annotations)
        Require(entry.first == "nan_propagate",
                "Reduction scheduling annotations are unsupported");
      Require(reduce->type->IsSum() || reduce->type->IsMax() ||
                  reduce->type->IsMin(),
              "Reduction supports sum, max and min");
      RequireMalformed(!reduce->src->data.same_as(reduce->dst->data),
                       "Reduction output must not alias its source");
      size_t rank = reduce->src->shape.size();
      RequireMalformed(rank >= 2 && reduce->dim >= 0 &&
                           static_cast<size_t>(reduce->dim) < rank,
                       "Reduction requires a valid axis on a rank >= 2 input");
      bool keepdims = reduce->dst->shape.size() == rank;
      RequireMalformed(
          keepdims || reduce->dst->shape.size() + 1 == rank,
          "Reduction output must remove exactly the selected axis (no "
          "extra dimensions)");
      for (size_t src_axis = 0, dst_axis = 0; src_axis < rank; ++src_axis) {
        if (static_cast<int>(src_axis) == reduce->dim) {
          if (keepdims) {
            RequireMalformed(is_one(reduce->dst->shape[dst_axis++]),
                             "Reduction keepdims axis must have extent one");
          }
          continue;
        }
        RequireMalformed(
            ffi::StructuralEqual()(reduce->src->shape[src_axis],
                                   reduce->dst->shape[dst_axis++]),
            "Reduction output shape does not match the non-reduced axes");
      }
      Require(reduce->dst->dtype == reduce->src->dtype ||
                  reduce->dst->dtype == DataType::Float(32),
              "Reduction output must use input dtype or float32 accumulation");
      Annotations annotations = BaseAnnotations(reduce->dst, "reduce");
      annotations.Set("tt.reduce_axis", Integer(reduce->dim));
      annotations.Set("tt.reduce_kind",
                      StringImm(reduce->type->IsSum()   ? "sum"
                                : reduce->type->IsMax() ? "max"
                                                        : "min"));
      annotations.Set("tt.clear", Integer(reduce->clear));
      annotations.Set("tt.nan_propagate", Integer(reduce->nan_propagate));
      annotations.Set("tt.accum_dtype",
                      StringImm(reduce->type->IsSum()
                                    ? "float32"
                                    : DTypeName(reduce->dst->dtype)));
      ffi::Array<BufferRegion> inputs{reduce->srcRegion_};
      if (!reduce->clear)
        inputs.push_back(reduce->dstRegion_);
      return MakeCompute(reduce->dstRegion_, inputs, annotations, span);
    }
    if (call->op.same_as(Copy::Get())) {
      Copy copy = Downcast<Copy>(ParseOperator(call));
      if ((has_fragment_compute_ && copy->src.scope() == "local.fragment") ||
          copy->dst.scope() == "local.fragment") {
        for (const auto &entry : call->annotations)
          Require(entry.first == "tt.transfer_kind",
                  "Fragment copy scheduling annotations are unsupported");
        const char *kind =
            copy->src->dtype == copy->dst->dtype ? "copy" : "typecast";
        BufferRegion output(copy->dst, copy->dst_range);
        if (copy->dst.scope() != "global" && !copy->dst.scope().empty() &&
            !IsFullBufferRegion(output))
          ThrowUnsupported("fragment materialization to buffer '" +
                           std::string(copy->dst->name) +
                           "' requires a complete destination region; partial "
                           "materialization has no Device consumer");
        Annotations annotations = BaseAnnotations(copy->dst, kind);
        ffi::Array<PrimExpr> zeros, domain;
        for (const Range &range : copy->src_range)
          zeros.push_back(make_zero(range->min.dtype()));
        for (const Range &range : copy->dst_range)
          domain.push_back(range->extent);
        annotations.Set("tt.logical_domain", domain);
        annotations.Set("tt.expression",
                        cast(copy->dst->dtype, BufferLoad(copy->src, zeros)));
        return MakeCompute(output, {BufferRegion(copy->src, copy->src_range)},
                           annotations, span, true);
      }
      if (copy->src.scope() == "local.fragment")
        return MakeAccumulatorCompute(
            BufferRegion(copy->src, copy->src_range),
            {BufferRegion(copy->dst, copy->dst_range)},
            "accumulator_materialize", accumulator_plan_.Contract(call), span);
      if ((copy->src.scope() == "shared" ||
           copy->src.scope() == "shared.dyn") &&
          (copy->dst.scope() == "shared" ||
           copy->dst.scope() == "shared.dyn")) {
        RequireMalformed(
            ffi::StructuralEqual()(copy->src->shape, copy->dst->shape),
            "shared copy/typecast requires matching full buffer shape");
        Annotations annotations = BaseAnnotations(copy->dst, "typecast");
        ffi::Array<PrimExpr> zeros;
        for (const PrimExpr &extent : copy->src->shape)
          zeros.push_back(make_zero(extent.dtype()));
        annotations.Set("tt.expression",
                        cast(copy->dst->dtype, BufferLoad(copy->src, zeros)));
        return MakeCompute(BufferRegion(copy->dst, copy->dst_range),
                           {BufferRegion(copy->src, copy->src_range)},
                           annotations, span);
      }
      return ffi::GetRef<Stmt>(op);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    Require(op->kind != ForKind::kParallel,
            "Tenstorrent elementwise compute requires T.Tiles; T.Parallel is "
            "unsupported");
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    ThrowMalformed(
        "BufferStore compute requires a structured T.Tiles operation");
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (const auto *named = op->op.as<OpNode>()) {
      const std::string name = named->name;
      if (name.rfind("tl.tileop.", 0) == 0 && !op->op.same_as(Copy::Get()) &&
          !op->op.same_as(RegionOp::Get()))
        ThrowUnsupported("TileOp '" + name + "' has no Device TIR lowering");
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  static Stmt MakeAccumulatorCompute(const BufferRegion &accumulator,
                                     const ffi::Array<BufferRegion> &regions,
                                     const char *kind, Annotations annotations,
                                     const Span &span) {
    annotations.Set("tt.compute_kind", StringImm(kind));
    ffi::Array<PrimExpr> args{accumulator->ToPrimExpr()};
    for (const BufferRegion &region : regions)
      args.push_back(region->ToPrimExpr());
    return Evaluate(
        Call(DataType::Void(), tile_compute(), args, annotations, span), span);
  }

  const GemmAccumulatorPlan &accumulator_plan_;
  bool has_fragment_compute_{false};
};

bool HasStructuredCompute(const Stmt &body) {
  bool found = false;
  PostOrderVisit(body, [&](const ffi::ObjectRef &object) {
    if (auto block = object.as<SBlock>())
      found |= block.value()->annotations.count("tl.tt.compute_kind") != 0;
  });
  return found;
}

} // namespace

tvm::transform::Pass LegalizeTenstorrentTileOps() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &context) {
    if (func->attrs.defined() &&
        func->attrs->dict.count("tt.gemm_accumulator_requirements"))
      func = WithoutAttr(std::move(func), "tt.gemm_accumulator_requirements");
    GemmAccumulatorPlan accumulator_plan =
        GemmAccumulatorAnalyzer::Analyze(func);
    StructuredComputeLowerer lowerer(func, accumulator_plan);
    Stmt body = lowerer(func->body);
    if (HasStructuredCompute(body))
      ThrowUnsupported("structured compute block has no Device consumer");
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
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

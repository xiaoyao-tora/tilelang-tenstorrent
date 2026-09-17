/*!
 * \file tenstorrent/transform/normalize_regions.cc
 * \brief Normalize Tenstorrent copy regions without lowering transfers.
 */

#include "../../op/copy.h"
#include "../../op/utils.h"
#include "../op/builtin.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace tl {
namespace tenstorrent {

using namespace tirx;

namespace {

constexpr const char *kTransferKind = "tt.transfer_kind";

void ThrowMalformed(const std::string &message) {
  TVM_FFI_THROW(ValueError)
      << "Malformed Tenstorrent transfer region: " << message;
}

void ThrowUnsupported(const std::string &message) {
  TVM_FFI_THROW(NotImplementedError)
      << "Unsupported Tenstorrent transfer region: " << message;
}

bool IsGlobalBuffer(const Buffer &buffer) {
  const std::string scope = buffer.scope();
  return scope.empty() || scope == "global";
}

bool IsDFBBuffer(const Buffer &buffer) {
  const std::string scope = buffer.scope();
  return scope == "shared" || scope == "shared.dyn";
}

int64_t RequireStaticInteger(const PrimExpr &expr, const std::string &field) {
  arith::Analyzer analyzer;
  PrimExpr simplified = analyzer.Simplify(expr);
  const auto *integer = simplified.as<IntImmNode>();
  if (integer == nullptr) {
    ThrowUnsupported(field +
                     " is dynamic; Phase 1 requires statically provable "
                     "copy regions");
  }
  return integer->value;
}

struct NormalizedRegion {
  BufferRegion region;
  bool changed;
};

NormalizedRegion NormalizeRegion(const BufferRegion &region,
                                 const std::string &endpoint,
                                 arith::Analyzer *loop_analyzer = nullptr) {
  const Buffer &buffer = region->buffer;
  if (region->region.size() != buffer->shape.size()) {
    ThrowMalformed(endpoint + " region rank does not match buffer '" +
                   std::string(buffer->name) + "'");
  }

  ffi::Array<Range> normalized;
  bool changed = false;
  for (size_t axis = 0; axis < region->region.size(); ++axis) {
    const Range &range = region->region[axis];
    int64_t shape = RequireStaticInteger(buffer->shape[axis],
                                         endpoint + " buffer shape axis " +
                                             std::to_string(axis));
    PrimExpr minimum_expr =
        loop_analyzer ? loop_analyzer->Simplify(range->min) : range->min;
    int64_t extent = RequireStaticInteger(
        range->extent, endpoint + " extent axis " + std::to_string(axis));
    if (shape <= 0 || extent <= 0) {
      ThrowMalformed(endpoint + " shape and region extents must be positive");
    }
    if (!as_const_int(minimum_expr) && loop_analyzer) {
      // Static pipeline binders will be substituted by formation. Prove the
      // complete range now so delayed specialization cannot hide an invalid
      // iteration or turn a dynamic/runtime index into an accepted transfer.
      if (SideEffect(minimum_expr) > CallEffectKind::kPure ||
          !loop_analyzer->CanProve(minimum_expr >= 0) ||
          !loop_analyzer->CanProve(minimum_expr +
                                       IntImm(minimum_expr.dtype(), extent) <=
                                   IntImm(minimum_expr.dtype(), shape))) {
        ThrowMalformed(endpoint +
                       " pipeline region is out of bounds or "
                       "unproven for buffer '" +
                       std::string(buffer->name) + "' on axis " +
                       std::to_string(axis));
      }
      normalized.push_back(Range::FromMinExtent(
          minimum_expr, IntImm(range->extent.dtype(), extent)));
      changed |= !minimum_expr.same_as(range->min) ||
                 !is_const_int(range->extent, extent);
      continue;
    }
    int64_t minimum = RequireStaticInteger(
        minimum_expr, endpoint + " minimum axis " + std::to_string(axis));
    bool axis_changed = minimum < 0;
    changed |= !is_const_int(range->min, minimum) ||
               !is_const_int(range->extent, extent);
    if (axis_changed) {
      minimum += shape;
      changed = true;
    }
    if (minimum < 0 || minimum + extent > shape) {
      ThrowMalformed(endpoint + " region is out of bounds for buffer '" +
                     std::string(buffer->name) + "' on axis " +
                     std::to_string(axis));
    }
    PrimExpr normalized_min = PrimExpr(IntImm(range->min.dtype(), minimum));
    normalized.push_back(Range::FromMinExtent(
        std::move(normalized_min), IntImm(range->extent.dtype(), extent)));
  }
  if (!changed) {
    return {region, false};
  }
  return {BufferRegion(buffer, std::move(normalized)), true};
}

ffi::String ClassifyTransfer(const Buffer &source, const Buffer &destination) {
  if (destination.scope() == "local.fragment" &&
      (IsDFBBuffer(source) || source.scope() == "local.fragment")) {
    return "compute_value_copy";
  }
  if (source.scope() == "local.fragment" &&
      (IsGlobalBuffer(destination) || IsDFBBuffer(destination))) {
    return "accumulator_materialize";
  }
  if (IsGlobalBuffer(source) && IsDFBBuffer(destination)) {
    return "tensor_to_dfb";
  }
  if (IsDFBBuffer(source) && IsGlobalBuffer(destination)) {
    return "dfb_to_tensor";
  }
  if (IsDFBBuffer(source) && IsDFBBuffer(destination)) {
    return "dfb_to_dfb";
  }
  ThrowUnsupported("copy from scope '" + std::string(source.scope()) +
                   "' to scope '" + std::string(destination.scope()) +
                   "' is not a Phase 1 transfer kind");
  return ffi::String();
}

void ValidateMatchingExtents(const BufferRegion &source,
                             const BufferRegion &destination) {
  if (source->region.size() != destination->region.size()) {
    ThrowMalformed("source and destination region ranks differ");
  }
  for (size_t axis = 0; axis < source->region.size(); ++axis) {
    int64_t source_extent =
        RequireStaticInteger(source->region[axis]->extent,
                             "source extent axis " + std::to_string(axis));
    int64_t destination_extent =
        RequireStaticInteger(destination->region[axis]->extent,
                             "destination extent axis " + std::to_string(axis));
    if (source_extent != destination_extent) {
      ThrowMalformed("source and destination extents differ on axis " +
                     std::to_string(axis));
    }
  }
}

// Preserve operation provenance while specializing K-dependent transfer slices.
class AccumulatorIterationSubstituter : public StmtExprMutator {
public:
  AccumulatorIterationSubstituter(Var variable, PrimExpr value) {
    substitutions_.emplace(std::move(variable), std::move(value));
  }

  PrimExpr VisitExpr(const PrimExpr &expr) final {
    PrimExpr result = StmtExprMutator::VisitExpr(expr);
    return result.dtype().is_handle() ||
                   SideEffect(result) > CallEffectKind::kPure
               ? result
               : analyzer_.Simplify(result);
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var variable = ffi::GetRef<Var>(op);
    auto found = substitutions_.find(variable);
    return found == substitutions_.end() ? variable : found->second;
  }

  Stmt VisitStmt_(const BindNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    // A serial iteration can make aliases such as k_begin = stage * block_k
    // constant. Eliminate their definitions with their uses, so expanded
    // iterations neither retain dynamic slices nor redefine the same binder.
    if (value.as<IntImmNode>() || value.as<FloatImmNode>()) {
      substitutions_[op->var] = value;
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
  arith::Analyzer analyzer_;
  std::unordered_map<Var, PrimExpr, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      substitutions_;
};

class RegionNormalizer : public StmtExprMutator {
public:
  explicit RegionNormalizer(bool expand_accumulator_loops)
      : expand_accumulator_loops_(expand_accumulator_loops) {}

  Stmt VisitStmt_(const ForNode *op) final {
    if (op->annotations.count("num_stages")) {
      int64_t extent = RequireStaticInteger(op->extent, "pipeline extent");
      int64_t minimum = RequireStaticInteger(op->min, "pipeline minimum");
      if (extent <= 0)
        ThrowUnsupported("pipeline extent must be positive");
      if (extent > 1024)
        ThrowUnsupported("pipeline region proof requires extent <= 1024");
      if (op->kind != ForKind::kSerial ||
          (op->step.has_value() && !is_one(op->step.value())))
        ThrowUnsupported("pipeline region proof requires a static serial "
                         "unit-step loop");
      loop_analyzer_.Bind(
          op->loop_var,
          Range::FromMinExtent(IntImm(op->loop_var.dtype(), minimum),
                               IntImm(op->loop_var.dtype(), extent)));
      auto saved_aliases = aliases_;
      ++pipeline_depth_;
      Stmt result = StmtExprMutator::VisitStmt_(op);
      --pipeline_depth_;
      aliases_ = std::move(saved_aliases);
      return result;
    }
    if (!expand_accumulator_loops_)
      return StmtExprMutator::VisitStmt_(op);
    // The single-Core path does not run topology specialization. Retain its
    // canonical launch prefix while resolving coordinates and scalar aliases.
    if (op->annotations.count("tt.logical_core_axis") && is_one(op->extent)) {
      AccumulatorIterationSubstituter substitute(op->loop_var, op->min);
      For loop = ffi::GetRef<For>(op);
      loop.CopyOnWrite()->body = VisitStmt(substitute(op->body));
      return loop;
    }
    if (!op->annotations.empty())
      return StmtExprMutator::VisitStmt_(op);
    int64_t extent =
        RequireStaticInteger(op->extent, "accumulator K loop extent");
    int64_t minimum =
        RequireStaticInteger(op->min, "accumulator K loop minimum");
    if (op->kind != ForKind::kSerial || extent <= 0 || extent > 1024 ||
        (op->step.has_value() && !is_one(op->step.value())))
      ThrowUnsupported(
          "accumulator K loop requires serial unit step and extent <= 1024");
    ffi::Array<Stmt> iterations;
    for (int64_t i = 0; i < extent; ++i) {
      if (++expanded_iterations_ > 65536)
        ThrowUnsupported(
            "accumulator static loop expansion exceeds 65536 iterations");
      AccumulatorIterationSubstituter substitute(
          op->loop_var, IntImm(op->loop_var.dtype(), minimum + i));
      iterations.push_back(VisitStmt(substitute(op->body)));
    }
    return iterations.size() == 1 ? iterations[0]
                                  : SeqStmt(iterations, op->span);
  }

  Stmt VisitStmt_(const BindNode *op) final {
    if (pipeline_depth_ && op->var.dtype().is_int() &&
        SideEffect(op->value) == CallEffectKind::kPure) {
      aliases_[op->var] = VisitExpr(op->value);
      return Evaluate(Integer(0), op->span);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    auto found = aliases_.find(ffi::GetRef<Var>(op));
    return found == aliases_.end() ? ffi::GetRef<Var>(op) : found->second;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call.same_as(ffi::GetRef<Call>(op)))
      call.CopyOnWrite()->span = op->span;
    return call;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call_node = op->value.as<CallNode>();
    if (call_node == nullptr) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Call call = Downcast<Call>(VisitExpr(ffi::GetRef<Call>(call_node)));
    if (call->op.same_as(Copy::Get())) {
      return NormalizeCopy(ffi::GetRef<Evaluate>(op), call);
    }
    if (call->op.same_as(tenstorrent::pipe_send())) {
      ValidatePipeRegion(call, 0, kAccessRead, "Pipe source");
      return ffi::GetRef<Stmt>(op);
    }
    if (call->op.same_as(tenstorrent::pipe_recv())) {
      ValidatePipeRegion(call, 1, kAccessWrite, "Pipe destination");
      return ffi::GetRef<Stmt>(op);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

private:
  bool expand_accumulator_loops_;
  size_t expanded_iterations_{0};
  int pipeline_depth_{0};
  arith::Analyzer loop_analyzer_;
  std::unordered_map<Var, PrimExpr, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      aliases_;

  static void ValidatePipeRegion(const Call &call, size_t argument_index,
                                 int access_mask, const std::string &endpoint) {
    if (call->args.size() <= argument_index) {
      ThrowMalformed(endpoint + " operation is missing its BufferRegion");
    }
    BufferRegion region =
        NormalizeToAccessRegion(call->args[argument_index], access_mask).region;
    NormalizedRegion normalized = NormalizeRegion(region, endpoint);
    if (normalized.changed) {
      ThrowUnsupported(endpoint +
                       " negative indexing is not canonicalized until the "
                       "typed Pipe endpoint path is implemented");
    }
    if (!IsDFBBuffer(region->buffer)) {
      ThrowMalformed(endpoint + " payload must use a shared DFB buffer");
    }
  }

  Stmt NormalizeCopy(const Evaluate &evaluate, const Call &call) {
    if (call->args.size() < 2) {
      ThrowMalformed("T.copy requires source and destination BufferRegions");
    }
    Copy copy = Downcast<Copy>(ParseOperator(call));
    BufferRegion source(copy->src, copy->src_range);
    BufferRegion destination(copy->dst, copy->dst_range);

    arith::Analyzer *analyzer = pipeline_depth_ ? &loop_analyzer_ : nullptr;
    NormalizedRegion normalized_source =
        NormalizeRegion(source, "source", analyzer);
    NormalizedRegion normalized_destination =
        NormalizeRegion(destination, "destination", analyzer);
    ValidateMatchingExtents(normalized_source.region,
                            normalized_destination.region);
    ffi::String transfer_kind =
        ClassifyTransfer(source->buffer, destination->buffer);

    ffi::Map<ffi::String, ffi::ObjectRef> annotations = call->annotations;
    bool annotation_matches = false;
    if (auto existing = annotations.Get(kTransferKind)) {
      const auto *existing_kind = existing.value().as<StringImmNode>();
      if (existing_kind == nullptr) {
        ThrowMalformed("'tt.transfer_kind' must be a string");
      }
      if (existing_kind->value != transfer_kind) {
        ThrowMalformed("'tt.transfer_kind' conflicts with source and "
                       "destination buffer scopes");
      }
      annotation_matches = true;
    }
    if (!normalized_source.changed && !normalized_destination.changed &&
        annotation_matches && call.same_as(evaluate->value)) {
      return evaluate;
    }

    ffi::Array<PrimExpr> args = call->args;
    args.Set(0, normalized_source.region->ToPrimExpr());
    args.Set(1, normalized_destination.region->ToPrimExpr());
    annotations.Set(kTransferKind, StringImm(transfer_kind));
    Call normalized_call(call.dtype(), call->op, std::move(args),
                         std::move(annotations), call->span);
    return Evaluate(std::move(normalized_call), evaluate->span);
  }
};

PrimFunc NormalizeRegions(PrimFunc func) {
  auto accumulators =
      func->GetAttr<ffi::Array<ffi::Map<ffi::String, ffi::ObjectRef>>>(
          "tt.gemm_accumulator_requirements");
  RegionNormalizer normalizer(accumulators.has_value() &&
                              !accumulators.value().empty());
  Stmt body = normalizer(func->body);
  if (body.same_as(func->body)) {
    return func;
  }
  func.CopyOnWrite()->body = std::move(body);
  return func;
}

} // namespace

tvm::transform::Pass NormalizeTenstorrentRegions() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    return NormalizeRegions(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.NormalizeTenstorrentRegions", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.tenstorrent.transform.NormalizeTenstorrentRegions",
                        NormalizeTenstorrentRegions);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

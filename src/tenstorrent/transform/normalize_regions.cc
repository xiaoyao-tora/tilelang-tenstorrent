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
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
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
  const auto *integer = expr.as<IntImmNode>();
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
                                 const std::string &endpoint) {
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
    int64_t minimum = RequireStaticInteger(
        range->min, endpoint + " minimum axis " + std::to_string(axis));
    int64_t extent = RequireStaticInteger(
        range->extent, endpoint + " extent axis " + std::to_string(axis));
    if (shape <= 0 || extent <= 0) {
      ThrowMalformed(endpoint + " shape and region extents must be positive");
    }
    bool axis_changed = minimum < 0;
    if (axis_changed) {
      minimum += shape;
      changed = true;
    }
    if (minimum < 0 || minimum + extent > shape) {
      ThrowMalformed(endpoint + " region is out of bounds for buffer '" +
                     std::string(buffer->name) + "' on axis " +
                     std::to_string(axis));
    }
    PrimExpr normalized_min =
        axis_changed ? PrimExpr(IntImm(range->min.dtype(), minimum))
                     : range->min;
    normalized.push_back(
        Range::FromMinExtent(std::move(normalized_min), range->extent));
  }
  if (!changed) {
    return {region, false};
  }
  return {BufferRegion(buffer, std::move(normalized)), true};
}

ffi::String ClassifyTransfer(const Buffer &source, const Buffer &destination) {
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

class RegionNormalizer : public StmtExprMutator {
public:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call_node = op->value.as<CallNode>();
    if (call_node == nullptr) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Call call = ffi::GetRef<Call>(call_node);
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

  static Stmt NormalizeCopy(const Evaluate &evaluate, const Call &call) {
    if (call->args.size() < 2) {
      ThrowMalformed("T.copy requires source and destination BufferRegions");
    }
    Copy copy = Downcast<Copy>(ParseOperator(call));
    BufferRegion source(copy->src, copy->src_range);
    BufferRegion destination(copy->dst, copy->dst_range);

    NormalizedRegion normalized_source = NormalizeRegion(source, "source");
    NormalizedRegion normalized_destination =
        NormalizeRegion(destination, "destination");
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
        annotation_matches) {
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
  RegionNormalizer normalizer;
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

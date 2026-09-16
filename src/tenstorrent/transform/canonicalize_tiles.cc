/*!
 * \file tenstorrent/transform/canonicalize_tiles.cc
 * \brief Capture Tiles and Parallel elementwise semantics before logical loop
 * binders disappear.
 */
#include "attr.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm::tl::tenstorrent {
using namespace tirx;
using ffi::Array;
using ffi::Map;
using ffi::String;
namespace {
using Entries = Map<String, ffi::Any>;
using Metadata =
    std::unordered_map<Var, Entries, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using Allocations =
    std::unordered_map<Var, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using AxisMaps = Map<Var, Array<Integer>>;

[[noreturn]] void ThrowElementwiseError(const std::string &message) {
  TVM_FFI_THROW(ValueError) << "Tenstorrent elementwise: " << message;
}
int64_t Static(const PrimExpr &expr, const std::string &field) {
  const auto *value = expr.as<IntImmNode>();
  if (!value)
    ThrowElementwiseError(field + " must be a compile-time constant");
  return value->value;
}
void Require(bool condition, const std::string &message) {
  if (!condition)
    ThrowElementwiseError(message);
}

class AllocationCollector : public StmtVisitor {
public:
  void Collect(const Stmt &body) { VisitStmt(body); }
  Metadata metadata;
  Allocations allocations;
  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers)
      Record(buffer);
    if (auto value = op->annotations.Get("tl.alloc_buffer_annotations")) {
      auto entries = value.value().as<Map<Var, Entries>>();
      Require(entries.has_value(), "invalid allocation metadata schema");
      for (const auto &[data, item] : entries.value()) {
        bool owned = false;
        for (const Buffer &buffer : op->alloc_buffers) {
          owned |= buffer->data.same_as(data);
        }
        Require(
            owned,
            "allocation metadata refers to a buffer not owned by its SBlock");
        RecordMetadata(data, item);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const AllocBufferNode *op) final {
    Record(op->buffer);
    RecordMetadata(op->buffer->data, op->annotations);
    StmtVisitor::VisitStmt_(op);
  }

private:
  void RecordMetadata(const Var &data, const Entries &entries) {
    auto [it, added] = metadata.emplace(data, entries);
    Require(added || ffi::StructuralEqual()(it->second, entries),
            "conflicting allocation metadata for one buffer identity");
  }

  void Record(const Buffer &buffer) {
    auto [it, added] = allocations.emplace(buffer->data, buffer);
    Require(added || it->second.same_as(buffer),
            "aliased allocation Buffer objects");
  }
};

struct Access {
  Buffer buffer;
  Array<Integer> axes;
  bool read{false};
  bool write{false};
};
struct ScopePlan {
  Array<Var> variables;
  Array<PrimExpr> domain;
  BufferStore store;
  std::vector<Access> accesses;
  Span span;
};

// An enclosing bounded serial loop is expanded by program formation. Its
// binder is a scalar template parameter, unlike the elementwise coordinates.
bool IsStaticScalarLoop(const For &loop) {
  const int64_t *extent = as_const_int(loop->extent);
  return (loop->kind == ForKind::kSerial || loop->kind == ForKind::kUnrolled) &&
         as_const_int(loop->min) && extent &&
         (!loop->step.has_value() || is_one(loop->step.value()));
}

// Analyze the complete scope first. Mutation only starts after this plan
// validates.
class ScopeAnalyzer {
public:
  ScopeAnalyzer(const AllocationCollector &allocation, bool require_metadata,
                Array<Var> scalar_parameters = {})
      : allocation_(allocation), require_metadata_(require_metadata),
        scalar_parameters_(std::move(scalar_parameters)) {}

  ScopePlan Analyze(const For &root, bool tiles) {
    if (root->span.defined() && root->span->source_name.defined()) {
      context_ = "scope at " + std::string(root->span->source_name->name) +
                 ":" + std::to_string(root->span->line) + ": ";
    }
    ScopePlan plan;
    plan.span = root->span;
    if (tiles) {
      auto domain = root->annotations.Get(kTTTilesDomain);
      Require(domain.has_value(), "frontend loop is missing logical domain");
      auto values = domain.value().as<Array<PrimExpr>>();
      Require(values.has_value(), "logical domain must be an Array of extents");
      plan.domain = values.value();
    }
    Stmt body = root;
    for (size_t axis = 0;
         tiles ? axis < plan.domain.size()
               : body.as<ForNode>() &&
                     body.as<ForNode>()->kind == ForKind::kParallel;
         ++axis) {
      const auto *node = body.as<ForNode>();
      Require(node != nullptr, "expected a direct loop chain");
      For loop = ffi::GetRef<For>(node);
      Require(Static(loop->min, "loop minimum") == 0,
              "loop minimum must be zero");
      Require(!loop->step.has_value() ||
                  Static(loop->step.value(), "loop step") == 1,
              "loop step must be one");
      if (tiles) {
        Require(loop->kind == ForKind::kSerial,
                "Tiles binders must be serial loops");
        CheckIntAnnotation(loop->annotations, kTTTilesStage, 0);
        CheckIntAnnotation(loop->annotations, kTTTilesParallel, 1);
        Require(axis == 0 || !loop->annotations.count(kTTTilesScope),
                "nested Tiles scope");
        if (axis == 0)
          CheckIntAnnotation(loop->annotations, kTTTilesScope, 1);
        for (const auto &[key, value] : loop->annotations) {
          Require(key == kTTTilesStage || key == kTTTilesParallel ||
                      (axis == 0 &&
                       (key == kTTTilesScope || key == kTTTilesDomain)),
                  "unsupported Tiles loop annotation schema: " +
                      std::string(key));
        }
      } else {
        Require(loop->kind == ForKind::kParallel,
                "expected parallel loop chain");
        Require(loop->annotations.empty(),
                "Parallel scheduling annotations are not supported in the "
                "elementwise Phase 1 path");
        plan.domain.push_back(loop->extent);
      }
      Require(plan.domain.size() >= 2 || !tiles,
              "logical domain must have rank at least 2");
      Require(analyzer_.CanProveEqual(loop->extent, plan.domain[axis]),
              "loop extent disagrees with domain");
      plan.variables.push_back(loop->loop_var);
      body = loop->body;
    }
    Require(plan.domain.size() >= 2,
            "logical domain must have rank at least 2");
    for (size_t axis = 0; axis < plan.domain.size(); ++axis) {
      int64_t size = Static(plan.domain[axis], "logical domain");
      Require(size > 0 && (axis + 2 < plan.domain.size() || size % 32 == 0),
              "logical domain must be positive and divisible by 32");
    }
    const auto *store = body.as<BufferStoreNode>();
    Require(store != nullptr,
            "scope body must contain exactly one BufferStore; nested scopes, "
            "predicates and side effects are unsupported");
    plan.store = ffi::GetRef<BufferStore>(store);
    Require(!store->predicate.has_value(), "store predicate is unsupported");
    Require(store->value.dtype() == store->buffer->dtype,
            "store expression dtype mismatch");
    RecordAccess(plan, store->buffer, store->indices, false);
    AnalyzeExpression(plan, store->value);
    return plan;
  }

  void CheckIntAnnotation(const Map<String, ffi::Any> &annotations,
                          const char *key, int expected) {
    auto value = annotations.Get(key);
    Require(value.has_value(), std::string("missing ") + key);
    auto expr = value.value().as<PrimExpr>();
    Require(expr.has_value() && Static(expr.value(), key) == expected,
            std::string(key) + " has invalid stage/parallel value for Phase 1");
  }

private:
  [[noreturn]] void Reject(const std::string &message) const {
    ThrowElementwiseError(context_ + message);
  }

  void Require(bool condition, const std::string &message) const {
    if (!condition)
      Reject(message);
  }

  int64_t Static(const PrimExpr &expr, const std::string &field) const {
    const auto *value = expr.as<IntImmNode>();
    if (!value)
      Reject(field + " must be a compile-time constant");
    return value->value;
  }

  void ValidateBuffer(const Buffer &buffer, const Array<Integer> &axes,
                      const Array<PrimExpr> &domain) {
    Require(buffer.scope() == "shared" || buffer.scope() == "shared.dyn",
            "buffer '" + std::string(buffer->name) +
                "' must have shared scope");
    Require(buffer->shape.size() == domain.size(),
            "buffer shape must match logical domain rank");
    Require(buffer->dtype.lanes() == 1,
            "elementwise buffers must have scalar element dtypes");
    auto allocation = allocation_.allocations.find(buffer->data);
    Require(allocation == allocation_.allocations.end() ||
                allocation->second.same_as(buffer),
            "access uses an alias instead of the allocation Buffer object");
    Require(buffer->axis_separators.empty(),
            "axis-separated storage alias is unsupported");
    Require(analyzer_.CanProveEqual(buffer->elem_offset, 0),
            "storage offset must be zero");
    if (!buffer->strides.empty()) {
      Require(buffer->strides.size() == domain.size(),
              "storage strides must match rank");
      PrimExpr stride = Integer(1);
      for (int axis = static_cast<int>(domain.size()) - 1; axis >= 0; --axis) {
        Require(analyzer_.CanProveEqual(buffer->strides[axis], stride),
                "storage must have compact strides; strided aliases are "
                "unsupported");
        stride = stride * buffer->shape[axis];
      }
    }
    for (size_t axis = 0; axis < domain.size(); ++axis) {
      int64_t padding = axis + 2 < domain.size() ? 1 : 32;
      int64_t expected =
          axes[axis]->value < 0 ? padding : Static(domain[axis], "domain");
      Require(Static(buffer->shape[axis], "buffer shape") == expected,
              "buffer '" + std::string(buffer->name) +
                  "' shape does not match identity or compact padded broadcast "
                  "shape");
    }
    auto record = allocation_.metadata.find(buffer->data);
    if (record == allocation_.metadata.end()) {
      Require(!require_metadata_,
              "buffer is missing Tenstorrent allocation metadata");
      return; // Ordinary Parallel uses the backend's explicit default 32x32
              // tile.
    }
    auto shape = record->second.Get("tt.tile_shape");
    auto count = record->second.Get("tt.dfb_block_count");
    Require(
        !require_metadata_ || (shape.has_value() && count.has_value()),
        "allocation metadata requires tt.tile_shape and tt.dfb_block_count");
    if (shape.has_value()) {
      auto extents = shape.value().as<Array<PrimExpr>>();
      Require(extents.has_value() && extents.value().size() == 2,
              "tt.tile_shape metadata must have rank 2");
      for (const PrimExpr &extent : extents.value())
        Require(Static(extent, "tt.tile_shape") == 32,
                "tt.tile_shape must be 32x32");
    }
    if (count.has_value()) {
      auto integer = count.value().as<IntImm>();
      Require(integer.has_value() && integer.value()->value > 0 &&
                  integer.value()->value <= 32,
              "tt.dfb_block_count metadata must be in [1, 32]");
    }
  }

  void ValidateIndex(const PrimExpr &expr) {
    // Inspect before simplification: a canceled gather/effect still changes
    // semantics.
    PostOrderVisit(expr, [this](const ffi::ObjectRef &node) {
      Require(!node.as<BufferLoadNode>() && !node.as<CallNode>(),
              "access indices cannot contain gather loads or calls with side "
              "effects");
    });
  }

  void RecordAccess(ScopePlan &plan, const Buffer &buffer,
                    const Array<PrimExpr> &indices, bool read) {
    Require(indices.size() == plan.domain.size(),
            "buffer access rank must match logical domain");
    Array<Integer> axes;
    for (size_t axis = 0; axis < plan.domain.size(); ++axis) {
      ValidateIndex(indices[axis]);
      if (analyzer_.CanProveEqual(indices[axis], plan.variables[axis])) {
        axes.push_back(Integer(axis));
      } else if (read && analyzer_.CanProveEqual(indices[axis], 0)) {
        axes.push_back(Integer(-1));
      } else
        Reject("access indices in buffer '" + std::string(buffer->name) +
               "' must be identity or zero-axis broadcast in Phase 1");
    }
    ValidateBuffer(buffer, axes, plan.domain);
    for (Access &access : plan.accesses) {
      if (access.buffer->data.same_as(buffer->data)) {
        Require(access.buffer.same_as(buffer),
                "alias Buffer objects share one data identity");
        Require(ffi::StructuralEqual()(access.axes, axes),
                "one buffer has conflicting access maps");
        access.read |= read;
        access.write |= !read;
        return;
      }
    }
    plan.accesses.push_back({buffer, axes, read, !read});
  }

  void AnalyzeExpression(ScopePlan &plan, const PrimExpr &expr) {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      Require(!load->predicate.has_value(), "load predicate is unsupported");
      RecordAccess(plan, load->buffer, load->indices, true);
      return;
    }
    if (expr.as<IntImmNode>() || expr.as<FloatImmNode>())
      return;
    if (auto variable = expr.as<Var>()) {
      for (const Var &parameter : scalar_parameters_) {
        if (parameter.same_as(variable.value()))
          return;
      }
    }
    if (const auto *cast = expr.as<CastNode>()) {
      Require(cast->annotations.empty(),
              "Cast annotations have no supported template schema");
      AnalyzeExpression(plan, cast->value);
      return;
    }
#define TT_BINARY(Node)                                                        \
  if (const auto *binary = expr.as<Node>()) {                                  \
    AnalyzeExpression(plan, binary->a);                                        \
    AnalyzeExpression(plan, binary->b);                                        \
    return;                                                                    \
  }
    TT_BINARY(AddNode)
    TT_BINARY(SubNode)
    TT_BINARY(MulNode)
    TT_BINARY(DivNode)
    TT_BINARY(MinNode)
    TT_BINARY(MaxNode)
#undef TT_BINARY
    if (const auto *call = expr.as<CallNode>()) {
      Require(call->annotations.empty(),
              "Call annotations have no supported template schema");
      const auto *op = call->op.as<OpNode>();
      Require(op != nullptr && call->args.size() == 1,
              "only known pure unary Call operations are supported");
      const std::string name = op->name;
      Require(name == "tirx.exp" || name == "tirx.exp2" || name == "tirx.log" ||
                  name == "tirx.log2" || name == "tirx.sqrt" ||
                  name == "tirx.rsqrt" || name == "tirx.tanh" ||
                  name == "tirx.sin" || name == "tirx.cos" ||
                  name == "tirx.fabs" || name == "tirx.floor" ||
                  name == "tirx.ceil",
              "Call operation '" + name +
                  "' is not a supported pure unary expression");
      static auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
      Require(effects.count(ffi::GetRef<Op>(op)) &&
                  effects[ffi::GetRef<Op>(op)]->value ==
                      static_cast<int>(CallEffectKind::kPure),
              "Call must be pure");
      AnalyzeExpression(plan, call->args[0]);
      return;
    }
    Reject("unsupported Phase 1 expression; loop variables cannot be numeric "
           "RHS values");
  }
  const AllocationCollector &allocation_;
  bool require_metadata_;
  Array<Var> scalar_parameters_;
  std::string context_;
  arith::Analyzer analyzer_;
};

// Validation has already consumed the access relations. Replace the complete
// index expressions, including equivalent spellings such as (i + 0), with
// canonical zeros without simplifying the scalar expression DAG itself.
class TemplateBuilder : public StmtExprMutator {
private:
  static Array<PrimExpr> ZeroIndices(const Array<PrimExpr> &indices) {
    Array<PrimExpr> zeros;
    for (const PrimExpr &index : indices)
      zeros.push_back(make_zero(index.dtype()));
    return zeros;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    return BufferLoad(op->buffer, ZeroIndices(op->indices), std::nullopt,
                      op->span);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    return BufferStore(op->buffer, VisitExpr(op->value),
                       ZeroIndices(op->indices), std::nullopt, op->span);
  }
};

SBlockRealize ApplyPlan(const ScopePlan &plan) {
  Array<BufferRegion> reads, writes;
  AxisMaps maps;
  Map<Var, Map<String, ffi::Any>> recipes;
  for (const Access &access : plan.accesses) {
    Array<Range> ranges;
    Array<PrimExpr> logical, physical_tiles;
    Array<Integer> broadcast_axes;
    for (size_t axis = 0; axis < plan.domain.size(); ++axis) {
      PrimExpr extent = access.axes[axis]->value < 0
                            ? make_const(plan.domain[axis].dtype(), 1)
                            : plan.domain[axis];
      ranges.push_back(Range::FromMinExtent(make_zero(extent.dtype()), extent));
      logical.push_back(extent);
      physical_tiles.push_back(
          make_const(access.buffer->shape[axis].dtype(),
                     Static(access.buffer->shape[axis], "shape") /
                         (axis + 2 < plan.domain.size() ? 1 : 32)));
      if (access.axes[axis]->value < 0)
        broadcast_axes.push_back(Integer(axis));
    }
    BufferRegion region(access.buffer, ranges);
    if (access.read)
      reads.push_back(region);
    if (access.write)
      writes.push_back(region);
    maps.Set(access.buffer->data, access.axes);
    if (!broadcast_axes.empty()) {
      String kind =
          plan.domain.size() == 2
              ? (broadcast_axes.size() == 2
                     ? "scalar"
                     : (broadcast_axes[0]->value == 0 ? "row" : "column"))
              : "batch";
      recipes.Set(access.buffer->data,
                  {{"broadcast_kind", kind},
                   {"broadcast_axes", broadcast_axes},
                   {"logical_region", logical},
                   {"physical_shape", access.buffer->shape},
                   {"block_shape", physical_tiles}});
    }
  }
  Array<PrimExpr> block_shape;
  Array<String> iterator_types;
  for (size_t axis = 0; axis < plan.domain.size(); ++axis) {
    iterator_types.push_back("parallel");
    block_shape.push_back(
        make_const(plan.domain[axis].dtype(),
                   Static(plan.domain[axis], "domain") /
                       (axis + 2 < plan.domain.size() ? 1 : 32)));
  }
  Map<String, ffi::Any> annotations{
      {kTTComputeKind, String("elementwise")},
      {kTTTilesStage, Integer(1)},
      {kTTLogicalDomain, plan.domain},
      {kTTPhysicalTileShape, Array<PrimExpr>{Integer(32), Integer(32)}},
      {kTTBlockShape, block_shape},
      {kTTIteratorTypes, iterator_types},
      {kTTAccessMaps, maps},
      {kTTBroadcastRecipes, recipes}};
  TemplateBuilder builder;
  SBlock block({}, reads, writes, "tl.tt.elementwise", builder(plan.store),
               std::nullopt, {}, {}, annotations, plan.span);
  return SBlockRealize({}, const_true(), block, plan.span);
}

class Canonicalizer : public StmtMutator {
public:
  explicit Canonicalizer(const PrimFunc &func) : func_(func) {
    allocation_.Collect(func->body);
  }
  Stmt VisitStmt_(const ForNode *op) final {
    bool tiles = op->annotations.count(kTTTilesScope);
    if (tiles || op->kind == ForKind::kParallel) {
      auto target = func_->GetAttr<Target>(tvm::attr::kTarget);
      Require(target.has_value() && target.value()->kind->name == "tenstorrent",
              "canonicalization requires a bound Tenstorrent target");
      // The explicit Tiles marker selects its stricter frontend contract.
      // Capture the whole loop chain before visiting its nested binders.
      ScopeAnalyzer analyzer(allocation_, tiles, scalar_parameters_);
      return ApplyPlan(analyzer.Analyze(ffi::GetRef<For>(op), tiles));
    }
    bool scalar = IsStaticScalarLoop(ffi::GetRef<For>(op));
    if (scalar)
      scalar_parameters_.push_back(op->loop_var);
    Stmt body = StmtMutator::VisitStmt_(op);
    if (scalar)
      scalar_parameters_.pop_back();
    return body;
  }
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->annotations.count(kTTComputeKind))
      return ffi::GetRef<Stmt>(op);
    return StmtMutator::VisitStmt_(op);
  }

private:
  PrimFunc func_;
  AllocationCollector allocation_;
  Array<Var> scalar_parameters_;
};

// Reconstruct only the logical binders from access descriptors, then reuse the
// analyzer to verify all geometry, effects, recipes and expression semantics.
// The original template itself is required to contain zero indices exclusively.
class TemplateRestorer : public StmtExprMutator {
public:
  TemplateRestorer(AxisMaps maps, Array<Var> variables)
      : maps_(std::move(maps)), variables_(std::move(variables)) {}
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    Require(!op->predicate.has_value(),
            "structured load predicate is unsupported");
    return BufferLoad(op->buffer, RestoreIndices(op->buffer, op->indices),
                      std::nullopt, op->span);
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Require(!op->predicate.has_value(),
            "structured store predicate is unsupported");
    return BufferStore(op->buffer, VisitExpr(op->value),
                       RestoreIndices(op->buffer, op->indices), std::nullopt,
                       op->span);
  }

private:
  Array<PrimExpr> RestoreIndices(const Buffer &buffer,
                                 const Array<PrimExpr> &indices) {
    Require(indices.size() == variables_.size(),
            "structured template access rank must match domain");
    auto axes = maps_.Get(buffer->data);
    Require(axes.has_value() && axes.value().size() == variables_.size(),
            "structured template buffer is missing tl.tt.access_maps");
    Array<PrimExpr> restored;
    for (size_t axis = 0; axis < variables_.size(); ++axis) {
      Require(Static(indices[axis], "structured template index") == 0,
              "structured template indices must be zero");
      int64_t mapped = axes.value()[axis]->value;
      Require(mapped == -1 || mapped == static_cast<int64_t>(axis),
              "structured access map must be identity or zero-axis broadcast");
      restored.push_back(mapped < 0 ? make_zero(indices[axis].dtype())
                                    : PrimExpr(variables_[axis]));
    }
    return restored;
  }
  AxisMaps maps_;
  Array<Var> variables_;
};

class ComputeVerifier : public StmtVisitor {
public:
  void Verify(const Stmt &body) { VisitStmt(body); }
  explicit ComputeVerifier(const PrimFunc &func) : func_(func) {
    allocation_.Collect(func->body);
  }
  void VisitStmt_(const ForNode *op) final {
    Require(!op->annotations.count(kTTTilesScope),
            "uncanonicalized frontend T.Tiles scope");
    bool scalar = IsStaticScalarLoop(ffi::GetRef<For>(op));
    if (scalar)
      scalar_parameters_.push_back(op->loop_var);
    StmtVisitor::VisitStmt_(op);
    if (scalar)
      scalar_parameters_.pop_back();
  }
  void VisitStmt_(const SBlockRealizeNode *op) final {
    if (op->block->annotations.count(kTTComputeKind)) {
      Require(
          op->iter_values.empty() && is_one(op->predicate),
          "structured elementwise realization must be opaque and unpredicated");
    }
    StmtVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const SBlockNode *op) final {
    if (!op->annotations.count(kTTComputeKind)) {
      StmtVisitor::VisitStmt_(op);
      return;
    }
    auto target = func_->GetAttr<Target>(tvm::attr::kTarget);
    Require(target.has_value() && target.value()->kind->name == "tenstorrent",
            "structured block requires a bound Tenstorrent target");
    Require(op->iter_vars.empty() && !op->init.has_value() &&
                op->alloc_buffers.empty() && op->match_buffers.empty(),
            "structured elementwise block must be opaque with no local "
            "allocations");
    for (const char *key :
         {kTTComputeKind, kTTTilesStage, kTTLogicalDomain, kTTPhysicalTileShape,
          kTTBlockShape, kTTIteratorTypes, kTTAccessMaps,
          kTTBroadcastRecipes}) {
      Require(op->annotations.count(key),
              std::string("structured block missing metadata ") + key);
    }
    auto domain = op->annotations.at(kTTLogicalDomain).as<Array<PrimExpr>>();
    auto maps = op->annotations.at(kTTAccessMaps).as<AxisMaps>();
    Require(
        domain.has_value() && domain.value().size() >= 2 && maps.has_value(),
        "invalid structured tl.tt.logical_domain or tl.tt.access_maps schema");
    Array<Var> variables;
    for (const PrimExpr &extent : domain.value()) {
      Static(extent, "structured logical domain");
      variables.push_back(Var("tt_verify_axis", extent.dtype()));
    }
    TemplateRestorer restorer(maps.value(), variables);
    Stmt body = restorer(op->body);
    for (int axis = static_cast<int>(variables.size()) - 1; axis >= 0; --axis) {
      body = For(variables[axis], make_zero(variables[axis].dtype()),
                 domain.value()[axis], ForKind::kParallel, body);
    }
    ScopeAnalyzer analyzer(allocation_, false, scalar_parameters_);
    SBlock expected =
        ApplyPlan(analyzer.Analyze(Downcast<For>(body), false))->block;
    Require(ffi::StructuralEqual()(op->reads, expected->reads),
            "structured elementwise read regions disagree with "
            "expression/access maps");
    Require(ffi::StructuralEqual()(op->writes, expected->writes),
            "structured elementwise write regions disagree with "
            "expression/access maps");
    for (const auto &[key, value] : expected->annotations) {
      Require(ffi::StructuralEqual()(op->annotations.at(key), value),
              "structured elementwise metadata is inconsistent: " +
                  std::string(key));
    }
    Require(op->name_hint == "tl.tt.elementwise",
            "structured elementwise block name is invalid");
    for (const auto &[key, value] : op->annotations) {
      Require(expected->annotations.count(key),
              "unsupported structured block annotation schema: " +
                  std::string(key));
    }
  }

private:
  PrimFunc func_;
  AllocationCollector allocation_;
  Array<Var> scalar_parameters_;
};

} // namespace

tvm::transform::Pass CanonicalizeTTElementwise() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    Canonicalizer rewriter(func);
    Stmt body = rewriter(func->body);
    if (!body.same_as(func->body))
      func.CopyOnWrite()->body = std::move(body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.CanonicalizeTTElementwise", {});
}

tvm::transform::Pass VerifyTTComputeBlocks() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    ComputeVerifier verifier(func);
    verifier.Verify(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.VerifyTTComputeBlocks", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.tenstorrent.transform.CanonicalizeTTElementwise",
           CanonicalizeTTElementwise)
      .def("tl.tenstorrent.transform.VerifyTTComputeBlocks",
           VerifyTTComputeBlocks);
}
} // namespace tvm::tl::tenstorrent

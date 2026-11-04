/*!
 * \file tenstorrent/transform/canonicalize_tiles.cc
 * \brief Capture Tiles elementwise semantics before logical loop binders
 * disappear.
 */
#include "../../op/utils.h"
#include "../ir/device_ir.h"
#include "../op/builtin.h"
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
using ValueKinds = Map<Var, String>;

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

String GetValueKind(const Buffer &buffer) {
  if (buffer.scope() == "shared" || buffer.scope() == "shared.dyn")
    return "shared_dfb";
  Require(buffer.scope() == "local.fragment",
          "buffer '" + std::string(buffer->name) +
              "' must have shared, shared.dyn or local.fragment scope");
  return "compute_fragment";
}

bool HasFrontendTilesAnnotations(const Entries &annotations) {
  return annotations.count(kTTTilesScope) ||
         annotations.count(kTTTilesDomain) ||
         annotations.count(kTTTilesParallel) ||
         annotations.count(kTTTilesStage);
}

bool HasStructuredComputeAnnotations(const Entries &annotations) {
  return annotations.count(kTTComputeKind) ||
         annotations.count(kTTTilesStage) ||
         annotations.count(kTTLogicalDomain) ||
         annotations.count(kTTPhysicalTileShape) ||
         annotations.count(kTTBlockShape) ||
         annotations.count(kTTIteratorTypes) ||
         annotations.count(kTTValueKinds);
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
  ScopeAnalyzer(const AllocationCollector &allocation,
                Array<Var> scalar_parameters = {})
      : allocation_(allocation),
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
      Require(plan.domain.size() == 2,
              "explicit T.Tiles logical domain must have rank 2");
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
      // Check the original extent before arithmetic proof: simplifying an
      // expression such as effectful_call() * 0 + N would erase its effects
      // when the logical loop binders are removed.
      Require(!loop->extent.dtype().is_bool(),
              "loop extent must be an integer constant, not bool");
      Require(Static(loop->extent, "loop extent") > 0,
              "loop extent must be positive");
      Require(analyzer_.CanProveEqual(loop->extent, plan.domain[axis]),
              "loop extent disagrees with domain");
      plan.variables.push_back(loop->loop_var);
      body = loop->body;
    }
    Require(plan.domain.size() >= 2,
            "logical domain must have rank at least 2");
    for (const PrimExpr &extent : plan.domain)
      Require(Static(extent, "logical domain") > 0,
              "logical domain must be positive");
    const auto *store = body.as<BufferStoreNode>();
    Require(store != nullptr,
            "scope body must contain exactly one BufferStore; nested scopes, "
            "predicates and side effects are unsupported");
    plan.store = ffi::GetRef<BufferStore>(store);
    Require(!store->predicate.has_value(), "store predicate is unsupported");
    Require(store->value.dtype() == store->buffer->dtype,
            "store expression dtype mismatch");
    AnalyzeExpression(plan, store->value);
    RecordAccess(plan, store->buffer, store->indices, false, store->span);
    // Storage metadata belongs to the actual effects, never to the Buffer
    // that merely supplied the frontend domain shape.
    for (const Access &access : plan.accesses)
      ValidateBuffer(access.buffer, access.axes, plan.domain);
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
    // Keep the original allocation identity, so its storage descriptor must
    // remain valid after the logical coordinate binders have been removed.
    // Check the unsimplified expressions so canceled free variables cannot
    // bypass verification of an already-structured block either.
    auto is_variable = [](const VarNode *) { return true; };
    Require(!UsesVar(buffer->elem_offset, is_variable),
            "buffer storage offset must be static and cannot reference loop "
            "variables");
    for (const PrimExpr &stride : buffer->strides) {
      Require(!UsesVar(stride, is_variable),
              "buffer storage strides must be static and cannot reference loop "
              "variables");
    }
    String value_kind = GetValueKind(buffer);
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
      int64_t actual = Static(buffer->shape[axis], "buffer shape");
      int64_t expected = Static(domain[axis], "domain");
      Require(actual > 0 && (axes[axis]->value < 0 || actual == expected),
              "buffer '" + std::string(buffer->name) +
                  "' shape does not match its logical access on axis " +
                  std::to_string(axis) + " (actual=" + std::to_string(actual) +
                  ", domain=" + std::to_string(expected) + ")");
    }
    if (value_kind == "compute_fragment") {
      Require(domain.size() == 2,
              "fragment elementwise computation requires a rank-2 domain");
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
                    const Array<PrimExpr> &indices, bool read, const Span &) {
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
    for (const Access &access : plan.accesses) {
      if (!access.buffer->data.same_as(buffer->data) || access.read != read)
        continue;
      Require(access.buffer.same_as(buffer),
              "access uses a distinct Buffer object sharing one data identity");
      Require(ffi::StructuralEqual()(access.axes, axes),
              "one elementwise block cannot read buffer '" +
                  std::string(buffer->name) +
                  "' through different logical regions");
      return;
    }
    plan.accesses.push_back({buffer, axes, read, !read});
  }

  void AnalyzeExpression(ScopePlan &plan, const PrimExpr &expr) {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      Require(!load->predicate.has_value(), "load predicate is unsupported");
      RecordAccess(plan, load->buffer, load->indices, true, load->span);
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
    TT_BINARY(EQNode)
    TT_BINARY(NENode)
    TT_BINARY(LTNode)
    TT_BINARY(LENode)
    TT_BINARY(GTNode)
    TT_BINARY(GENode)
    TT_BINARY(AndNode)
    TT_BINARY(OrNode)
#undef TT_BINARY
    if (const auto *logical_not = expr.as<NotNode>()) {
      AnalyzeExpression(plan, logical_not->a);
      return;
    }
    if (const auto *select = expr.as<SelectNode>()) {
      Require(select->condition.dtype().is_bool() &&
                  select->true_value.dtype() == select->false_value.dtype(),
              "Select requires a Boolean condition and equal arm dtypes");
      AnalyzeExpression(plan, select->condition);
      AnalyzeExpression(plan, select->true_value);
      AnalyzeExpression(plan, select->false_value);
      return;
    }
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
  Array<Var> scalar_parameters_;
  std::string context_;
  arith::Analyzer analyzer_;
};

// Validation has already consumed the access relations. Replace complete index
// expressions with canonical zeros while retaining BufferLoad identities.
class TemplateBuilder : public StmtExprMutator {
public:
  explicit TemplateBuilder(const ScopePlan &) {}

private:
  static Array<PrimExpr> ZeroIndices(const Array<PrimExpr> &indices) {
    Array<PrimExpr> zeros;
    for (const PrimExpr &index : indices)
      zeros.push_back(make_zero(index.dtype()));
    return zeros;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    return BufferLoad(op->buffer, ZeroIndices(op->indices), op->predicate,
                      op->span);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    return BufferStore(op->buffer, VisitExpr(op->value),
                       ZeroIndices(op->indices), std::nullopt, op->span);
  }
};

SBlockRealize ApplyPlan(const ScopePlan &plan) {
  Array<BufferRegion> reads, writes;
  ValueKinds value_kinds;
  for (const Access &access : plan.accesses) {
    Array<Range> ranges;
    for (size_t axis = 0; axis < plan.domain.size(); ++axis) {
      PrimExpr extent = access.axes[axis]->value < 0
                            ? make_const(plan.domain[axis].dtype(), 1)
                            : plan.domain[axis];
      ranges.push_back(Range::FromMinExtent(make_zero(extent.dtype()), extent));
    }
    BufferRegion region(access.buffer, ranges);
    if (access.read)
      reads.push_back(region);
    if (access.write)
      writes.push_back(region);
    value_kinds.Set(access.buffer->data, GetValueKind(access.buffer));
  }
  Array<PrimExpr> block_shape;
  Array<String> iterator_types;
  for (size_t axis = 0; axis < plan.domain.size(); ++axis) {
    iterator_types.push_back("parallel");
    const int64_t extent = Static(plan.domain[axis], "domain");
    const int64_t tile = axis + 2 < plan.domain.size() ? 1 : 32;
    block_shape.push_back(
        make_const(plan.domain[axis].dtype(), (extent + tile - 1) / tile));
  }
  Map<String, ffi::Any> annotations{
      {kTTComputeKind, String("elementwise")},
      {kTTTilesStage, Integer(1)},
      {kTTLogicalDomain, plan.domain},
      {kTTPhysicalTileShape, Array<PrimExpr>{Integer(32), Integer(32)}},
      {kTTBlockShape, block_shape},
      {kTTIteratorTypes, iterator_types},
      {kTTValueKinds, value_kinds}};
  TemplateBuilder builder(plan);
  SBlock block({}, reads, writes, "tl.tt.elementwise", builder(plan.store),
               std::nullopt, {}, {}, annotations, plan.span);
  return SBlockRealize({}, const_true(), block, plan.span);
}

Stmt MergeComputeBranch(const IfThenElse &branch,
                        const AllocationCollector &allocation,
                        const Array<Var> &scalar_parameters);

class Canonicalizer : public StmtMutator {
public:
  explicit Canonicalizer(const PrimFunc &func) : func_(func) {
    allocation_.Collect(func->body);
  }
  Stmt VisitStmt_(const ForNode *op) final {
    bool tiles = op->annotations.count(kTTTilesScope);
    if (tiles) {
      auto target = func_->GetAttr<Target>(tvm::attr::kTarget);
      Require(target.has_value() && target.value()->kind->name == "tenstorrent",
              "canonicalization requires a bound Tenstorrent target");
      // Capture the whole loop chain before visiting its nested binders.
      ScopeAnalyzer analyzer(allocation_, scalar_parameters_);
      return ApplyPlan(analyzer.Analyze(ffi::GetRef<For>(op), true));
    }
    Require(!HasFrontendTilesAnnotations(op->annotations),
            "orphan frontend T.Tiles annotations without a tiles_scope root");
    bool scalar = IsStaticScalarLoop(ffi::GetRef<For>(op));
    if (scalar)
      scalar_parameters_.push_back(op->loop_var);
    Stmt body = StmtMutator::VisitStmt_(op);
    if (scalar)
      scalar_parameters_.pop_back();
    return body;
  }
  Stmt VisitStmt_(const IfThenElseNode *op) final {
    IfThenElse branch = Downcast<IfThenElse>(StmtMutator::VisitStmt_(op));
    return MergeComputeBranch(branch, allocation_, scalar_parameters_);
  }

private:
  PrimFunc func_;
  AllocationCollector allocation_;
  Array<Var> scalar_parameters_;
};

// Reconstruct logical binders from precise block regions, then reuse the scope
// analyzer to verify geometry, effects and expression semantics.
class TemplateRestorer : public StmtExprMutator {
public:
  TemplateRestorer(Array<BufferRegion> reads, Array<BufferRegion> writes,
                   Array<PrimExpr> domain, Array<Var> variables)
      : domain_(std::move(domain)), variables_(std::move(variables)) {
    for (const BufferRegion &region : reads)
      Require(reads_.emplace(region->buffer->data, region).second,
              "structured reads contain duplicate Buffer identities");
    for (const BufferRegion &region : writes)
      Require(writes_.emplace(region->buffer->data, region).second,
              "structured writes contain duplicate Buffer identities");
  }
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    Require(!op->predicate.has_value(),
            "structured load predicate is unsupported");
    for (const PrimExpr &index : op->indices)
      Require(is_zero(index),
              "structured BufferLoad coordinates must be canonical zeros");
    auto found = reads_.find(op->buffer->data);
    Require(found != reads_.end() && found->second->buffer.same_as(op->buffer),
            "structured expression reads a Buffer missing from block reads");
    return BufferLoad(op->buffer, RestoreIndices(found->second), std::nullopt,
                      op->span);
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Require(!op->predicate.has_value(),
            "structured store predicate is unsupported");
    for (const PrimExpr &index : op->indices)
      Require(is_zero(index),
              "structured BufferStore coordinates must be canonical zeros");
    auto found = writes_.find(op->buffer->data);
    Require(found != writes_.end() && found->second->buffer.same_as(op->buffer),
            "structured output Buffer is missing from block writes");
    return BufferStore(op->buffer, VisitExpr(op->value),
                       RestoreIndices(found->second), std::nullopt, op->span);
  }

private:
  [[noreturn]] static void Reject(const std::string &message) {
    ThrowElementwiseError(message);
  }

  Array<PrimExpr> RestoreIndices(const BufferRegion &region) {
    Require(region->region.size() == variables_.size() &&
                domain_.size() == variables_.size(),
            "structured region rank must match domain");
    Array<PrimExpr> restored;
    for (size_t axis = 0; axis < variables_.size(); ++axis) {
      Require(is_zero(region->region[axis]->min),
              "structured logical region must start at zero");
      if (ffi::StructuralEqual()(region->region[axis]->extent, domain_[axis]))
        restored.push_back(variables_[axis]);
      else {
        Require(is_one(region->region[axis]->extent),
                "structured read extent must match the domain or be one");
        restored.push_back(make_zero(variables_[axis].dtype()));
      }
    }
    return restored;
  }
  std::unordered_map<Var, BufferRegion, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      reads_, writes_;
  Array<PrimExpr> domain_;
  Array<Var> variables_;
};

// If-conversion is safe only for one pure, full-region assignment per arm.
// It creates a value merge without introducing conditional DFB transactions.
// Other branches retain their control-flow boundary for later diagnostics.
Stmt MergeComputeBranch(const IfThenElse &branch,
                        const AllocationCollector &allocation,
                        const Array<Var> &scalar_parameters) {
  if (as_const_int(branch->condition))
    return branch;
  auto get_compute = [](const Stmt &stmt) -> ffi::Optional<SBlock> {
    auto realize = stmt.as<SBlockRealize>();
    if (!realize.has_value() || !is_one(realize.value()->predicate) ||
        !realize.value()->iter_values.empty())
      return std::nullopt;
    SBlock block = realize.value()->block;
    if (!block->annotations.count(kTTComputeKind) ||
        !block->body.as<BufferStoreNode>() || block->init.has_value() ||
        !block->iter_vars.empty() || !block->alloc_buffers.empty() ||
        !block->match_buffers.empty())
      return std::nullopt;
    return block;
  };
  auto then_block = get_compute(branch->then_case);
  auto else_block = branch->else_case.has_value()
                        ? get_compute(branch->else_case.value())
                        : ffi::Optional<SBlock>();
  bool empty_else = !branch->else_case.has_value();
  if (branch->else_case.has_value()) {
    if (const auto *evaluate = branch->else_case.value().as<EvaluateNode>())
      empty_else = is_zero(evaluate->value);
  }
  if (!then_block.has_value() || (!empty_else && !else_block.has_value()))
    return branch;
  bool local_condition = true;
  bool reads_local_value = false;
  PostOrderVisit(branch->condition, [&](const ffi::ObjectRef &object) {
    if (const auto *load = object.as<BufferLoadNode>()) {
      reads_local_value = true;
      std::string scope = load->buffer.scope();
      local_condition &= scope == "shared" || scope == "shared.dyn" ||
                         scope == "local.fragment";
      for (const PrimExpr &index : load->indices)
        local_condition &= is_zero(index);
    }
  });
  if (!local_condition || !reads_local_value)
    return branch;

  Array<PrimExpr> domain = Downcast<Array<PrimExpr>>(
      then_block.value()->annotations.at(kTTLogicalDomain));
  BufferStore then_store = Downcast<BufferStore>(then_block.value()->body);
  if (else_block.has_value()) {
    BufferStore other = Downcast<BufferStore>(else_block.value()->body);
    if (!other->buffer.same_as(then_store->buffer) ||
        !ffi::StructuralEqual()(
            else_block.value()->annotations.at(kTTLogicalDomain), domain))
      return branch;
  }
  Array<Var> variables;
  Array<PrimExpr> indices;
  for (const PrimExpr &extent : domain) {
    Var variable("tt_merge_axis", extent.dtype());
    variables.push_back(variable);
    indices.push_back(variable);
  }
  TemplateRestorer then_restorer(then_block.value()->reads,
                                 then_block.value()->writes, domain, variables);
  then_store = Downcast<BufferStore>(then_restorer(then_store));
  PrimExpr otherwise =
      BufferLoad(then_store->buffer, indices, std::nullopt, branch->span);
  if (else_block.has_value()) {
    TemplateRestorer else_restorer(else_block.value()->reads,
                                   else_block.value()->writes, domain,
                                   variables);
    otherwise =
        Downcast<BufferStore>(else_restorer(else_block.value()->body))->value;
  }
  Stmt merged = BufferStore(
      then_store->buffer,
      Select(branch->condition, then_store->value, otherwise, branch->span),
      indices, std::nullopt, branch->span);
  for (int axis = static_cast<int>(variables.size()) - 1; axis >= 0; --axis)
    merged = For(variables[axis], make_zero(variables[axis].dtype()),
                 domain[axis], ForKind::kParallel, merged, std::nullopt, {},
                 std::nullopt, branch->span);
  ScopeAnalyzer analyzer(allocation, scalar_parameters);
  return ApplyPlan(analyzer.Analyze(Downcast<For>(merged), false));
}

class ComputeVerifier : public StmtVisitor {
public:
  void Verify(const Stmt &body) { VisitStmt(body); }
  explicit ComputeVerifier(const PrimFunc &func) : func_(func) {
    allocation_.Collect(func->body);
  }
  void VisitStmt_(const ForNode *op) final {
    Require(!HasFrontendTilesAnnotations(op->annotations),
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
      Require(!HasStructuredComputeAnnotations(op->annotations),
              "structured block missing metadata tl.tt.compute_kind");
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
          kTTBlockShape, kTTIteratorTypes, kTTValueKinds}) {
      Require(op->annotations.count(key),
              std::string("structured block missing metadata ") + key);
    }
    auto domain = op->annotations.at(kTTLogicalDomain).as<Array<PrimExpr>>();
    auto value_kinds = op->annotations.at(kTTValueKinds).as<ValueKinds>();
    Require(value_kinds.has_value(),
            "invalid structured tl.tt.value_kinds schema: expected "
            "Map<Var, String> keyed by Buffer data identity");
    Require(domain.has_value() && domain.value().size() == 2,
            "structured elementwise logical domain must have rank 2");
    Array<Var> variables;
    for (const PrimExpr &extent : domain.value()) {
      Static(extent, "structured logical domain");
      variables.push_back(Var("tt_verify_axis", extent.dtype()));
    }
    TemplateRestorer restorer(op->reads, op->writes, domain.value(), variables);
    Stmt body = restorer(op->body);
    for (int axis = static_cast<int>(variables.size()) - 1; axis >= 0; --axis) {
      body = For(variables[axis], make_zero(variables[axis].dtype()),
                 domain.value()[axis], ForKind::kParallel, body);
    }
    ScopeAnalyzer analyzer(allocation_, scalar_parameters_);
    SBlock expected =
        ApplyPlan(analyzer.Analyze(Downcast<For>(body), false))->block;
    Require(ffi::StructuralEqual()(op->reads, expected->reads),
            "structured elementwise read regions disagree with expression");
    Require(ffi::StructuralEqual()(op->writes, expected->writes),
            "structured elementwise write regions disagree with expression");
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

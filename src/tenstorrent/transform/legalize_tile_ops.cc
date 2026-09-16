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

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace tenstorrent {

using namespace tirx;

// The verifier owns the structured-operation contract. Device instruction
// selection must not reconstruct that contract from zero-coordinate stores.
tvm::transform::Pass VerifyTTComputeBlocks();

namespace {

bool IsElementwiseBlock(const SBlock &block) {
  auto kind = block->annotations.Get("tl.tt.compute_kind");
  return kind.has_value() && kind.value().as<ffi::String>().has_value() &&
         Downcast<ffi::String>(kind.value()) == "elementwise";
}

bool IsDeviceAdd(const SBlock &block,
                 const ffi::Array<TTBufferMetadata> &metadata) {
  if (!IsElementwiseBlock(block)) {
    return false;
  }
  const auto *store = block->body.as<BufferStoreNode>();
  const auto *add = store == nullptr ? nullptr : store->value.as<AddNode>();
  if (add == nullptr || !add->a.as<BufferLoadNode>() ||
      !add->b.as<BufferLoadNode>()) {
    return false;
  }
  const Buffer &output = store->buffer;
  const Buffer &lhs = add->a.as<BufferLoadNode>()->buffer;
  const Buffer &rhs = add->b.as<BufferLoadNode>()->buffer;
  if (lhs.same_as(rhs) || lhs.same_as(output) || rhs.same_as(output)) {
    return false;
  }
  // The implemented Device TIR transaction protocol currently requires two
  // blocks per DFB. Other capacities remain valid structured computations.
  for (const Buffer &buffer : {lhs, rhs, output}) {
    bool supported_capacity = false;
    for (const TTBufferMetadata &item : metadata) {
      if (item->buffer.same_as(buffer) && item->dfb_block_count.has_value() &&
          is_const_int(item->dfb_block_count.value(), 2)) {
        supported_capacity = true;
        break;
      }
    }
    if (!supported_capacity)
      return false;
  }
  const DataType dtype = output->dtype;
  if (dtype.lanes() != 1 ||
      !(dtype.is_bfloat16() || (dtype.is_float() && dtype.bits() == 32))) {
    return false;
  }
  auto maps = Downcast<ffi::Map<Var, ffi::Array<Integer>>>(
      block->annotations.at("tl.tt.access_maps"));
  for (const BufferRegion &region : block->reads) {
    if (region->buffer->dtype != dtype) {
      return false;
    }
  }
  for (const auto &entry : maps) {
    if (entry.second.size() != 2 || entry.second[0]->value != 0 ||
        entry.second[1]->value != 1) {
      return false;
    }
  }
  auto shape = Downcast<ffi::Array<PrimExpr>>(
      block->annotations.at("tl.tt.logical_domain"));
  return shape.size() == 2 && is_const_int(shape[0], 32) &&
         is_const_int(shape[1], 32);
}

[[noreturn]] void Reject(const std::string &message) {
  TVM_FFI_THROW(NotImplementedError)
      << "[LegalizeTenstorrentTileOps] " << message;
}

void Require(bool condition, const std::string &message) {
  if (!condition)
    Reject(message);
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

void ValidateRegion(const BufferRegion &region, const std::string &owner) {
  const Buffer &buffer = region->buffer;
  Require(buffer.scope() == "shared" || buffer.scope() == "shared.dyn",
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
                (axis + 2 < buffer->shape.size() || extent->value % 32 == 0),
            owner + " requires static positive batch dimensions and "
                    "tile-aligned final axes");
    Require(is_zero(region->region[axis]->min) &&
                ffi::StructuralEqual()(region->region[axis]->extent,
                                       buffer->shape[axis]),
            owner + " currently requires a full BufferRegion; partial compute "
                    "is unsupported");
  }
}

ffi::Array<Integer> IdentityAxes(const Buffer &buffer) {
  ffi::Array<Integer> axes;
  for (size_t axis = 0; axis < buffer->shape.size(); ++axis)
    axes.push_back(Integer(axis));
  return axes;
}

using Annotations = ffi::Map<ffi::String, ffi::ObjectRef>;

Annotations BaseAnnotations(const Buffer &output, const char *kind) {
  return {
      {"tt.compute_kind", StringImm(kind)},
      {"tt.compute_dtype", StringImm(DTypeName(output->dtype))},
      {"tt.compute_tile_shape", ffi::Array<PrimExpr>{Integer(32), Integer(32)}},
      {"tt.logical_domain", output->shape}};
}

// Fragment precision belongs to a complete GEMM lifetime, not to all FP32
// temporaries. Keep this analysis before any DFB generation or loop lowering.
class GemmAccumulatorVerifier : public StmtExprVisitor {
  struct Lifetime {
    Buffer buffer;
    BufferRegion region;
    DataType input_dtype;
    bool initialized{false};
    bool updated{false};
    bool materialized{false};
    DataType output_dtype;
    int declaration_depth{-1};
  };

  class Collector : public StmtExprVisitor {
  public:
    std::vector<Buffer> fragments;
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(Gemm::Get())) {
        Gemm gemm = Downcast<Gemm>(ParseOperator(ffi::GetRef<Call>(op)));
        if (gemm->c_.scope() == "local.fragment") {
          for (const Buffer &buffer : fragments)
            if (buffer.same_as(gemm->c_))
              return;
          fragments.push_back(gemm->c_);
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  };

public:
  static PrimFunc Verify(PrimFunc func) {
    Collector collector;
    collector(func->body);
    if (collector.fragments.empty()) {
      if (func->attrs.defined() &&
          func->attrs->dict.count("tt.gemm_accumulator_requirements"))
        return WithoutAttr(std::move(func), "tt.gemm_accumulator_requirements");
      return func;
    }
    GemmAccumulatorVerifier verifier;
    auto metadata =
        func->GetAttr<ffi::Array<TTBufferMetadata>>(kBufferMetadataTableAttr);
    for (const Buffer &buffer : collector.fragments) {
      Require(
          metadata.has_value(),
          "GEMM accumulator verification requires normalized buffer metadata");
      int matches = 0;
      for (const TTBufferMetadata &entry : metadata.value()) {
        if (entry->buffer->data.same_as(buffer->data)) {
          Require(
              entry->buffer.same_as(buffer) &&
                  entry->kind == "compute_fragment" &&
                  !entry->alias_of.has_value(),
              "GEMM accumulator must resolve to one unique compute_fragment; "
              "aliases are unsupported");
          ++matches;
        }
      }
      Require(matches == 1,
              "GEMM accumulator must resolve to one unique compute_fragment");
      ValidateDType(buffer->dtype);
      Require(buffer->shape.size() == 2,
              "GEMM accumulator requires a rank-2 fragment");
      verifier.lifetimes_.push_back({buffer, BufferRegion::FullRegion(buffer),
                                     DataType::Void(), false, false, false,
                                     DataType::Void()});
    }
    if (!verifier.lifetimes_.empty()) {
      auto slot = func->GetAttr<ffi::String>(kKernelSlotAttr);
      Require(!slot.has_value() || slot.value() == "trisc",
              "GEMM fragment cannot cross processor slots; materialize through "
              "an output DFB");
      verifier(func->body);
    }
    ffi::Array<Annotations> requirements;
    for (const Lifetime &lifetime : verifier.lifetimes_) {
      Require(lifetime.updated && lifetime.materialized,
              "GEMM accumulator must have one final materialization after "
              "its complete K reduction");
      const bool fp32 = lifetime.buffer->dtype == DataType::Float(32);
      Annotations requirement{
          {"accumulator", lifetime.buffer},
          {"region", lifetime.region},
          {"input_dtype", StringImm(DTypeName(lifetime.input_dtype))},
          {"accum_dtype", StringImm(DTypeName(lifetime.buffer->dtype))},
          {"output_dtype", StringImm(DTypeName(lifetime.output_dtype))},
          {"dest_precision_requirement",
           StringImm(fp32 ? "bits32_required" : "bits16_required")},
          {"materialization", StringImm("after_complete_k_reduction")}};
      if (!fp32)
        requirement.Set("matmul_full_fp32", StringImm("forbidden"));
      requirements.push_back(requirement);
    }
    // Recompute this table on every invocation; frontend annotations are not a
    // proof of the def-use, dominance, or precision contract.
    return WithAttr(std::move(func), "tt.gemm_accumulator_requirements",
                    requirements);
  }

private:
  Lifetime *Find(const Buffer &buffer) {
    for (Lifetime &lifetime : lifetimes_) {
      if (buffer->data.same_as(lifetime.buffer->data)) {
        Require(buffer.same_as(lifetime.buffer),
                "GEMM accumulator alias makes fragment identity ambiguous");
        return &lifetime;
      }
    }
    return nullptr;
  }

  void CheckContext() {
    Require(conditional_depth_ == 0,
            "GEMM accumulator conditional access has no proven dominating "
            "initialization or final materialization");
    Require(unsupported_loop_depth_ == 0,
            "GEMM accumulator requires static positive serial K loops");
  }

  void CheckRegion(const BufferRegion &region, const Lifetime &lifetime) {
    Require(ffi::StructuralEqual()(region, lifetime.region),
            "GEMM accumulator updates/materialization must preserve the same "
            "full BufferRegion");
    arith::Analyzer analyzer;
    Require(region->buffer->strides.empty() &&
                region->buffer->axis_separators.empty() &&
                analyzer.CanProveEqual(region->buffer->elem_offset, 0),
            "GEMM accumulator aliases and storage offsets are unsupported");
  }

  void Initialize(Lifetime *lifetime) {
    CheckContext();
    Require(loop_depth_ == lifetime->declaration_depth,
            "GEMM accumulator initialization inside a K loop cannot preserve "
            "one complete reduction lifetime");
    Require(!lifetime->initialized && !lifetime->updated,
            "GEMM accumulator requires unique initialization; an ordinary "
            "write must not overwrite its live value");
    lifetime->initialized = true;
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
        Require(gemm->a_->shape.size() == 2 && gemm->b_->shape.size() == 2 &&
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

        Require(!lifetime->materialized,
                "GEMM update after final materialization would lose its "
                "complete K reduction lifetime");
        Require(gemm->a_->dtype == gemm->b_->dtype,
                "GEMM requires matching input storage dtypes");
        ValidateRegion(gemm->aRegion_, "GEMM matrix input A");
        ValidateRegion(gemm->bRegion_, "GEMM matrix input B");
        Require(!Find(gemm->a_) && !Find(gemm->b_),
                "GEMM matrix inputs must not alias an accumulator");
        Require(is_zero(gemm->clearAccum_) || is_one(gemm->clearAccum_),
                "GEMM clear_accum must be static");
        if (is_one(gemm->clearAccum_))
          Initialize(lifetime);
        Require(lifetime->initialized,
                "GEMM accumulator first update requires dominating T.clear "
                "or clear_accum=True");
        Require(!lifetime->updated || lifetime->input_dtype == gemm->a_->dtype,
                "GEMM accumulator input dtype changed during K updates");
        lifetime->input_dtype = gemm->a_->dtype;
        lifetime->updated = true;
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
        Require(is_zero(initializer) ||
                    (float_value && float_value->value == 0.0),
                "GEMM accumulator initialization requires T.clear (zero fill)");
        Initialize(lifetime);
        return;
      }
    } else if (call->op.same_as(Copy::Get())) {
      Copy copy = Downcast<Copy>(ParseOperator(call));
      Require(!Find(copy->dst),
              "ordinary copy must not write a GEMM accumulator");
      if (Lifetime *lifetime = Find(copy->src)) {
        CheckContext();
        CheckRegion(BufferRegion(copy->src, copy->src_range), *lifetime);
        Require(loop_depth_ == lifetime->declaration_depth,
                "GEMM accumulator cannot materialize inside its K reduction "
                "loop; pack only after all updates");
        Require(lifetime->updated && !lifetime->materialized,
                "GEMM accumulator requires one final materialization after "
                "all updates");
        Require(copy->dst.scope() == "shared" ||
                    copy->dst.scope() == "shared.dyn" ||
                    copy->dst.scope() == "global" || copy->dst.scope().empty(),
                "GEMM fragment must materialize to output DFB or Tensor");
        Require(copy->dst_range.size() == lifetime->region->region.size(),
                "GEMM output materialization rank mismatch");
        arith::Analyzer analyzer;
        for (size_t i = 0; i < copy->dst_range.size(); ++i)
          Require(analyzer.CanProveEqual(copy->dst_range[i]->extent,
                                         lifetime->region->region[i]->extent),
                  "GEMM output materialization shape mismatch");
        DataType input = lifetime->input_dtype;
        DataType accum = lifetime->buffer->dtype;
        DataType output = copy->dst->dtype;
        Require(IsSupportedAccumulatorDTypeTriple(input, accum, output),
                "GEMM capability registry has no (input=" + DTypeName(input) +
                    ", accum=" + DTypeName(accum) +
                    ", output=" + DTypeName(output) + ") combination");
        lifetime->output_dtype = output;
        lifetime->materialized = true;
        return;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Require(!Find(op->buffer),
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
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Require(!Find(op->buffer),
            "ordinary write must not overwrite a live GEMM accumulator");
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    ++conditional_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --conditional_depth_;
  }

  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers)
      if (Lifetime *lifetime = Find(buffer))
        lifetime->declaration_depth = loop_depth_;
    StmtExprVisitor::VisitStmt_(op);
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
  int loop_depth_{0};
  int conditional_depth_{0};
  int unsupported_loop_depth_{0};
};

Stmt MakeCompute(const BufferRegion &output,
                 const ffi::Array<BufferRegion> &inputs,
                 Annotations annotations, Span span) {
  ValidateRegion(output, "compute output");
  ffi::Array<PrimExpr> arguments{output->ToPrimExpr()};
  ffi::Array<ffi::Array<Integer>> maps;
  ffi::Array<ffi::Array<PrimExpr>> input_shapes;
  for (const BufferRegion &input : inputs) {
    ValidateRegion(input, "compute input");
    arguments.push_back(input->ToPrimExpr());
    maps.push_back(IdentityAxes(input->buffer));
    input_shapes.push_back(input->buffer->shape);
  }
  annotations.Set("tt.input_shapes", input_shapes);
  if (!annotations.count("tt.access_maps"))
    annotations.Set("tt.access_maps", maps);
  return Evaluate(Call(DataType::Handle(), tile_compute(), arguments,
                       std::move(annotations), span),
                  span);
}

class ComputeDTypeVerifier : public ExprVisitor {
public:
  void VisitExpr(const PrimExpr &expr) final {
    // VerifyTTComputeBlocks has already proved every remaining Var is an
    // enclosing static loop's scalar parameter. Form replaces it by an
    // integer constant; storage and arithmetic nodes still require FP dtypes.
    if (!expr.as<IntImmNode>() && !expr.as<VarNode>())
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
  explicit StructuredComputeLowerer(const PrimFunc &func)
      : metadata_(func->GetAttr<ffi::Array<TTBufferMetadata>>(
                          kBufferMetadataTableAttr)
                      .value_or(ffi::Array<TTBufferMetadata>{})),
        accumulator_requirements_(func->GetAttr<ffi::Array<Annotations>>(
                                          "tt.gemm_accumulator_requirements")
                                      .value_or(ffi::Array<Annotations>{})) {}

private:
  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    const SBlock &block = op->block;
    if (!IsElementwiseBlock(block))
      return StmtExprMutator::VisitStmt_(op);
    const auto *store = block->body.as<BufferStoreNode>();
    ICHECK(store != nullptr);
    const Buffer &output = store->buffer;
    bool supported_dtype = true;
    PostOrderVisit(store->value, [&](const ffi::ObjectRef &object) {
      if (auto expr = object.as<PrimExpr>()) {
        if (!expr.value().as<IntImmNode>() && !expr.value().as<VarNode>()) {
          DataType dtype = expr.value().dtype();
          supported_dtype &=
              dtype.lanes() == 1 &&
              (dtype == DataType::BFloat(16) || dtype == DataType::Float(32));
        }
      }
    });
    supported_dtype &= output->dtype == DataType::BFloat(16) ||
                       output->dtype == DataType::Float(32);
    if (!supported_dtype)
      return ffi::GetRef<Stmt>(op);
    Span span = block->span.defined() ? block->span : op->span;
    if (IsDeviceAdd(block, metadata_)) {
      const auto *add = store->value.as<AddNode>();
      const Buffer &lhs = add->a.as<BufferLoadNode>()->buffer;
      const Buffer &rhs = add->b.as<BufferLoadNode>()->buffer;
      Annotations annotations{
          {"tt.compute_dtype", StringImm(DTypeName(output->dtype))},
          {"tt.compute_tile_shape",
           ffi::Array<PrimExpr>{Integer(32), Integer(32)}}};
      return Evaluate(Call(DataType::Handle(), tile_add(),
                           {BufferRegion::FullRegion(lhs)->ToPrimExpr(),
                            BufferRegion::FullRegion(rhs)->ToPrimExpr(),
                            BufferRegion::FullRegion(output)->ToPrimExpr()},
                           annotations, span),
                      span);
    }
    ComputeDTypeVerifier dtype_verifier;
    dtype_verifier(store->value);
    auto access_maps = Downcast<ffi::Map<Var, ffi::Array<Integer>>>(
        block->annotations.at("tl.tt.access_maps"));
    ffi::Array<BufferRegion> inputs;
    ffi::Array<ffi::Array<Integer>> maps;
    for (const BufferRegion &read : block->reads) {
      inputs.push_back(BufferRegion::FullRegion(read->buffer));
      maps.push_back(access_maps.at(read->buffer->data));
    }
    const char *kind = inputs.empty()                ? "fill"
                       : store->value.as<CastNode>() ? "typecast"
                                                     : "elementwise";
    Annotations annotations = BaseAnnotations(output, kind);
    annotations.Set("tt.expression", store->value);
    annotations.Set("tt.access_maps", maps);
    return MakeCompute(BufferRegion::FullRegion(output), inputs, annotations,
                       span);
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
        bool verified = false;
        for (const auto &requirement : accumulator_requirements_)
          verified |= Downcast<Buffer>(requirement.at("accumulator"))
                          .same_as(fill->dst);
        Require(verified, "fragment Fill requires a verified GEMM accumulator");
        return MakeAccumulatorCompute(BufferRegion(fill->dst, fill->region), {},
                                      "accumulator_init", {}, span);
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
      Require(!transpose->src->data.same_as(transpose->dst->data),
              "Transpose source/destination alias requires a separate "
              "temporary buffer");
      size_t rank = transpose->src->shape.size();
      Require(rank >= 2 && transpose->dst->shape.size() == rank,
              "Transpose requires equal rank >= 2");
      Require(transpose->src->dtype == transpose->dst->dtype,
              "Transpose requires matching input/output dtype");
      ffi::Array<Integer> axes;
      for (size_t axis = 0; axis < rank; ++axis) {
        size_t mapped =
            axis + 2 < rank ? axis : (axis == rank - 2 ? rank - 1 : rank - 2);
        Require(
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
      Require(!gemm->c_->data.same_as(gemm->a_->data) &&
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
      Require(
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
        Stmt update = MakeAccumulatorCompute(gemm->cRegion_, inputs,
                                             "gemm_update", annotations, span);
        if (is_one(gemm->clearAccum_))
          return SeqStmt({MakeAccumulatorCompute(gemm->cRegion_, {},
                                                 "accumulator_init", {}, span),
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
      Require(!reduce->src->data.same_as(reduce->dst->data),
              "Reduction output must not alias its source");
      size_t rank = reduce->src->shape.size();
      Require(rank >= 2 && reduce->dim >= 0 &&
                  static_cast<size_t>(reduce->dim) < rank,
              "Reduction requires a valid axis on a rank >= 2 input");
      Require(reduce->dst->shape.size() + 1 == rank,
              "Reduction output must remove exactly the selected axis (no "
              "keepdims)");
      for (size_t src_axis = 0, dst_axis = 0; src_axis < rank; ++src_axis) {
        if (static_cast<int>(src_axis) == reduce->dim)
          continue;
        Require(ffi::StructuralEqual()(reduce->src->shape[src_axis],
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
      if (copy->src.scope() == "local.fragment")
        return MakeAccumulatorCompute(
            BufferRegion(copy->src, copy->src_range),
            {BufferRegion(copy->dst, copy->dst_range)},
            "accumulator_materialize", {}, span);
      if ((copy->src.scope() == "shared" ||
           copy->src.scope() == "shared.dyn") &&
          (copy->dst.scope() == "shared" ||
           copy->dst.scope() == "shared.dyn")) {
        Require(ffi::StructuralEqual()(copy->src->shape, copy->dst->shape),
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
            "T.Parallel must first pass through CanonicalizeTTElementwise");
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Reject("BufferStore compute requires a structured T.Parallel or T.Tiles "
           "operation");
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (const auto *named = op->op.as<OpNode>()) {
      const std::string name = named->name;
      if (name.rfind("tl.tileop.", 0) == 0 && !op->op.same_as(Copy::Get()) &&
          !op->op.same_as(RegionOp::Get()))
        Reject("TileOp '" + name + "' has no Device TIR lowering");
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

  ffi::Array<TTBufferMetadata> metadata_;
  ffi::Array<Annotations> accumulator_requirements_;
};

} // namespace

tvm::transform::Pass VerifyTTGemmAccumulators() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &context) {
    return GemmAccumulatorVerifier::Verify(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.VerifyTTGemmAccumulators", {});
}

tvm::transform::Pass LegalizeTenstorrentTileOps() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &context) {
    StructuredComputeLowerer lowerer(func);
    Stmt body = lowerer(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(body);
    }
    return func;
  };
  return tvm::transform::Sequential(
      {VerifyTTComputeBlocks(), VerifyTTGemmAccumulators(),
       tirx::transform::CreatePrimFuncPass(
           pass_func, 0, "tl.tenstorrent.LegalizeTenstorrentTileOps", {})},
      "tl.tenstorrent.LegalizeTenstorrentTileOps");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::GlobalDef().def("tl.tenstorrent.transform.VerifyTTGemmAccumulators",
                        VerifyTTGemmAccumulators);
  refl::GlobalDef().def("tl.tenstorrent.transform.LegalizeTenstorrentTileOps",
                        LegalizeTenstorrentTileOps);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*! \file tenstorrent/transform/normalize_topology.cc
 * \brief Specialize bounded logical Core and ordered PipeNet domains.
 */

#include "../../op/utils.h"
#include "../ir/device_ir.h"
#include "../op/builtin.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {
namespace tenstorrent {
namespace {
using namespace tirx;
using Record = ffi::Array<Integer>;
using Records = ffi::Array<Record>;

[[noreturn]] void Reject(const std::string &message) {
  TVM_FFI_THROW(ValueError) << "[NormalizeTenstorrentTopology] " << message;
}

class CoreSpecializer : public StmtExprMutator {
public:
  CoreSpecializer(int64_t x, int64_t y, int64_t gx, int64_t gy, const Var &vx,
                  const Var &vy, std::map<int64_t, std::string> *definitions)
      : x_(x), y_(y), gx_(gx), gy_(gy), definitions_(definitions) {
    substitutions_.emplace(vx, Integer(x));
    substitutions_.emplace(vy, Integer(y));
  }

  PrimExpr VisitExpr(const PrimExpr &expr) final {
    PrimExpr result = StmtExprMutator::VisitExpr(expr);
    return result.dtype().is_handle() ||
                   SideEffect(result) > CallEffectKind::kPure
               ? result
               : analyzer_.Simplify(result);
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    auto found = substitutions_.find(ffi::GetRef<Var>(op));
    return found == substitutions_.end() ? ffi::GetRef<Var>(op) : found->second;
  }

  Stmt VisitStmt_(const BindNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    if (value.as<IntImmNode>() || value.as<FloatImmNode>()) {
      substitutions_[op->var] = value;
      return Evaluate(Integer(0), op->span);
    }
    return Bind(op->var, value, op->span);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = analyzer_.Simplify(VisitExpr(op->condition));
    if (is_one(condition))
      return VisitStmt(op->then_case);
    if (is_zero(condition))
      return op->else_case.has_value() ? VisitStmt(op->else_case.value())
                                       : Evaluate(Integer(0), op->span);
    return IfThenElse(condition, VisitStmt(op->then_case),
                      op->else_case.has_value() ? ffi::Optional<Stmt>(VisitStmt(
                                                      op->else_case.value()))
                                                : std::nullopt,
                      op->span);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    auto src = op->annotations.Get("tl.tt.foreach_src");
    auto dst = op->annotations.Get("tl.tt.foreach_dst");
    if (!src.has_value() && !dst.has_value()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    if (src.has_value() && dst.has_value())
      Reject("foreach cannot select source and destination simultaneously");
    Records records = Decode(
        Downcast<ffi::String>(src.has_value() ? src.value() : dst.value()));
    if (!is_zero(op->min) || !is_const_int(op->extent, records.size()) ||
        op->kind != ForKind::kSerial ||
        (op->step.has_value() && !is_one(op->step.value())))
      Reject(
          "foreach bounds must retain the complete ordered Pipe record table");
    ffi::Array<Stmt> body;
    Var variable = op->loop_var;
    for (const Record &record : records) {
      bool matches = src.has_value() ? IsSource(record) : IsDestination(record);
      if (!matches)
        continue;
      selected_[variable] = record;
      source_side_[variable] = src.has_value();
      substitutions_[variable] = record[1];
      body.push_back(VisitStmt(op->body));
      selected_.erase(variable);
      source_side_.erase(variable);
      substitutions_.erase(variable);
    }
    if (body.empty())
      return Evaluate(Integer(0), op->span);
    return body.size() == 1 ? body[0] : SeqStmt(body, op->span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    const auto *operator_node = op->op.as<OpNode>();
    std::string name = operator_node ? std::string(operator_node->name) : "";
    if (name == "tl.tt.is_src" || name == "tl.tt.is_dst" ||
        name == "tl.tt.is_active") {
      if (op->args.size() != 1 || !op->args[0].as<StringImmNode>())
        Reject("PipeNet predicate requires one frozen descriptor");
      Records records = Decode(op->args[0].as<StringImmNode>()->value);
      bool match = false;
      for (const Record &record : records)
        match |= (name != "tl.tt.is_dst" && IsSource(record)) ||
                 (name != "tl.tt.is_src" && IsDestination(record));
      return make_const(op->dtype, match);
    }
    if (name == "tl.tt.pipe_send" || name == "tl.tt.pipe_recv" ||
        name == "tl.tt.pipe_src" || name == "tl.tt.pipe_dst" ||
        name == "tl.tt.pipe_dst_range") {
      size_t index = name == "tl.tt.pipe_send" ? 1 : 0;
      if (op->args.size() <= index || !op->args[index].as<VarNode>())
        Reject("PipeRef must be an active foreach binder");
      Var variable = Downcast<Var>(op->args[index]);
      auto found = selected_.find(variable);
      if (found == selected_.end())
        Reject("PipeRef escaped its foreach region");
      const Record &record = found->second;
      if ((name == "tl.tt.pipe_send" && !source_side_.at(variable)) ||
          (name == "tl.tt.pipe_recv" && source_side_.at(variable)))
        Reject("pipe_send requires foreach_src and pipe_recv requires "
               "foreach_dst");
      if ((name == "tl.tt.pipe_dst" && record[2]->value != 0) ||
          (name == "tl.tt.pipe_dst_range" && record[2]->value != 1))
        Reject("Pipe coordinate accessor disagrees with "
               "point-to-point/collective kind");
      if (name == "tl.tt.pipe_src" || name == "tl.tt.pipe_dst" ||
          name == "tl.tt.pipe_dst_range") {
        size_t arity = name == "tl.tt.pipe_dst_range" ? 3 : 2;
        if (op->args.size() != arity || !as_const_int(op->args.back()))
          Reject("Pipe coordinate accessor requires static axis and correct "
                 "arity");
        int64_t axis = *as_const_int(op->args.back());
        if (axis < 0 || axis > 1)
          Reject("Pipe coordinate axis must be zero or one");
        int offset = name == "tl.tt.pipe_src" ? 3 : 5;
        if (name == "tl.tt.pipe_dst_range") {
          if (!as_const_int(op->args[1]))
            Reject("Pipe destination range selector must be static");
          int64_t which = *as_const_int(op->args[1]);
          if (which < 0 || which > 1)
            Reject("Pipe destination range selector must be zero or one");
          offset += which * 2;
        }
        return IntImm(op->dtype, record[offset + axis]->value);
      }
      Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
      call.CopyOnWrite()->span = op->span;
      call.CopyOnWrite()->annotations.Set("tt.pipe_record", record);
      return call;
    }
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call.same_as(ffi::GetRef<Call>(op)))
      call.CopyOnWrite()->span = op->span;
    return call;
  }

private:
  Records Decode(const ffi::String &text) {
    auto decoder = ffi::Function::GetGlobal("tl.tenstorrent.DecodePipeNet");
    if (!decoder.has_value())
      Reject("PipeNet decoder is not registered; import tilelang.tenstorrent");
    Records records = (*decoder)(text).cast<Records>();
    int64_t net = records[0][0]->value;
    auto previous = definitions_->find(net);
    if (previous != definitions_->end() &&
        previous->second != std::string(text))
      Reject("one PipeNet ID has conflicting frozen descriptors");
    (*definitions_)[net] = text;
    for (const Record &r : records) {
      if (r[3]->value >= gx_ || r[4]->value >= gy_ || r[7]->value > gx_ ||
          r[8]->value > gy_)
        Reject("Pipe endpoint lies outside the logical Core grid");
    }
    return records;
  }
  bool IsSource(const Record &r) const {
    return r[3]->value == x_ && r[4]->value == y_;
  }
  bool IsDestination(const Record &r) const {
    return r[5]->value <= x_ && x_ < r[7]->value && r[6]->value <= y_ &&
           y_ < r[8]->value;
  }
  int64_t x_, y_, gx_, gy_;
  arith::Analyzer analyzer_;
  std::map<int64_t, std::string> *definitions_;
  std::unordered_map<Var, PrimExpr, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      substitutions_;
  std::unordered_map<Var, Record, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      selected_;
  std::unordered_map<Var, bool, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      source_side_;
};

class HighLevelTopologyCollector : public StmtExprVisitor {
public:
  HighLevelTopologyCollector(const Var &core_x, const Var &core_y, int64_t gx,
                             int64_t gy)
      : core_x_(core_x), core_y_(core_y), gx_(gx), gy_(gy) {
    analyzer_.Bind(core_x_, Range::FromMinExtent(Integer(0), Integer(gx_)));
    analyzer_.Bind(core_y_, Range::FromMinExtent(Integer(0), Integer(gy_)));
  }

  ffi::Array<NormalizedTopologyDescriptor> Collect(const Stmt &body) {
    VisitStmt(body);
    int64_t expected = 0;
    for (const NormalizedTopologyDescriptor &entry : entries_) {
      if (entry->site_id != expected++)
        Reject("communication site IDs must be dense and follow source order");
    }
    return entries_;
  }

  void VisitStmt_(const ForNode *op) final {
    bool contains_communication = false;
    PostOrderVisit(op->body, [&](const ffi::ObjectRef &object) {
      const auto *call = object.as<CallNode>();
      const auto *operator_node = call ? call->op.as<OpNode>() : nullptr;
      std::string name = operator_node ? std::string(operator_node->name) : "";
      contains_communication |= name == "tl.tt.transfer" ||
                                name == "tl.tt.all_gather" ||
                                name == "tl.tt.all_reduce";
    });
    if (!contains_communication) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled)
      Reject("communication occurrence loops must be serial or unrolled");
    const auto *minimum = op->min.as<IntImmNode>();
    const auto *extent = op->extent.as<IntImmNode>();
    if (!minimum || !extent || extent->value <= 0 ||
        (op->step.has_value() && !is_one(op->step.value())))
      Reject("communication occurrence loops require positive static unit-step "
             "bounds");
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    occurrence_extents_.push_back(op->extent);
    VisitStmt(op->body);
    occurrence_extents_.pop_back();
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    bool contains_communication = false;
    PostOrderVisit(
        ffi::GetRef<IfThenElse>(op), [&](const ffi::ObjectRef &object) {
          const auto *call = object.as<CallNode>();
          const auto *operator_node = call ? call->op.as<OpNode>() : nullptr;
          std::string name =
              operator_node ? std::string(operator_node->name) : "";
          contains_communication |= name == "tl.tt.transfer" ||
                                    name == "tl.tt.all_gather" ||
                                    name == "tl.tt.all_reduce";
        });
    if (!contains_communication) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    if (SideEffect(op->condition) > CallEffectKind::kPure)
      Reject("communication guard must be a pure expression");
    PrimExpr previous = guard_;
    guard_ = analyzer_.Simplify(previous && op->condition);
    VisitStmt(op->then_case);
    if (op->else_case.has_value()) {
      guard_ = analyzer_.Simplify(previous && !op->condition);
      VisitStmt(op->else_case.value());
    }
    guard_ = std::move(previous);
  }

  void VisitExpr_(const CallNode *op) final {
    const auto *operator_node = op->op.as<OpNode>();
    const std::string name =
        operator_node ? std::string(operator_node->name) : "";
    if (name != "tl.tt.transfer" && name != "tl.tt.all_gather" &&
        name != "tl.tt.all_reduce") {
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    Call call = ffi::GetRef<Call>(op);
    const bool transfer_call = name == "tl.tt.transfer";
    const size_t expected_arity = transfer_call ? 10 : 8;
    if (call->args.size() != expected_arity)
      Reject(
          "high-level communication arity changed after frontend validation");
    BufferRegion source =
        NormalizeToAccessRegion(call->args[0], kAccessRead).region;
    BufferRegion destination =
        NormalizeToAccessRegion(call->args[1], kAccessWrite).region;
    ffi::Array<PrimExpr> source_coord;
    ffi::Array<PrimExpr> begin;
    ffi::Array<PrimExpr> end;
    PrimExpr parameter;
    ffi::String contract;
    if (transfer_call) {
      source_coord = {call->args[2], call->args[3]};
      begin = {call->args[4], call->args[5]};
      end = {call->args[6], call->args[7]};
      parameter = call->args[8];
      contract = Downcast<StringImm>(call->args[8])->value;
    } else {
      begin = {call->args[2], call->args[3]};
      end = {call->args[4], call->args[5]};
      parameter = call->args[6];
      contract = "collective";
    }
    for (size_t axis = 0; axis < 2; ++axis) {
      PrimExpr limit = Integer(axis == 0 ? gx_ : gy_);
      if (!analyzer_.CanProve(begin[axis] >= 0) ||
          !analyzer_.CanProve(begin[axis] < end[axis]) ||
          !analyzer_.CanProve(end[axis] <= limit))
        Reject("communication group/endpoints are not statically in the launch "
               "grid");
      if (transfer_call && (!analyzer_.CanProve(source_coord[axis] >= 0) ||
                            !analyzer_.CanProve(source_coord[axis] < limit)))
        Reject("communication source is not statically in the launch grid");
    }
    if (transfer_call && contract != "collective") {
      PrimExpr source_in_destination =
          source_coord[0] >= begin[0] && source_coord[0] < end[0] &&
          source_coord[1] >= begin[1] && source_coord[1] < end[1];
      if (analyzer_.CanProve(source_in_destination))
        Reject("point transfer destination equals its source");
    }
    int64_t site_id = *as_const_int(call->args[expected_arity - 1]);
    if (!sites_.insert(site_id).second)
      Reject("duplicate communication site ID");
    ffi::String kind =
        transfer_call ? ffi::String("transfer") : ffi::String(name.substr(6));
    entries_.push_back(NormalizedTopologyDescriptor(
        site_id, kind, source, destination, source_coord, begin, end, parameter,
        contract, guard_, occurrence_extents_, call->span));
  }

private:
  Var core_x_, core_y_;
  int64_t gx_, gy_;
  arith::Analyzer analyzer_;
  PrimExpr guard_{const_true()};
  ffi::Array<PrimExpr> occurrence_extents_;
  ffi::Array<NormalizedTopologyDescriptor> entries_;
  std::unordered_set<int64_t> sites_;
};

PrimFunc NormalizeTopology(PrimFunc func) {
  auto grid = func->GetAttr<ffi::Array<PrimExpr>>("tt.launch_grid");
  if (!grid.has_value() || grid.value().size() != 2)
    Reject("normalized launch grid is missing");
  if (!as_const_int(grid.value()[0]) || !as_const_int(grid.value()[1]))
    Reject("logical Core grid must have static integer extents");
  int64_t gx = *as_const_int(grid.value()[0]),
          gy = *as_const_int(grid.value()[1]);
  if (gx <= 0 || gy <= 0 || gx > 256 || gy > 256 || gx * gy > 256)
    TVM_FFI_THROW(NotImplementedError)
        << "[NormalizeTenstorrentTopology] static Core grid is limited to 256 "
           "cores";
  if (!func->body.as<ForNode>() ||
      !func->body.as<ForNode>()->body.as<ForNode>())
    Reject("normalized launch must retain the logical Core x/y loop prefix");
  For xloop = Downcast<For>(func->body);
  For yloop = Downcast<For>(xloop->body);
  HighLevelTopologyCollector collector(xloop->loop_var, yloop->loop_var, gx,
                                       gy);
  auto topology = collector.Collect(yloop->body);
  func = WithAttr(std::move(func), "tt.normalized_topology", topology);
  return WithAttr(std::move(func), "tt.topology_normalized", Integer(1));
}
} // namespace

tvm::transform::Pass NormalizeTenstorrentTopology() {
  auto pass_func = [](tirx::PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &context) {
    return NormalizeTopology(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.NormalizeTenstorrentTopology", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::GlobalDef().def("tl.tenstorrent.transform.NormalizeTenstorrentTopology",
                        NormalizeTenstorrentTopology);
}
} // namespace tenstorrent
} // namespace tl
} // namespace tvm

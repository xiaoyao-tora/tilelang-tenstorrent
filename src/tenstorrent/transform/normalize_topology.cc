/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*! \file tenstorrent/transform/normalize_topology.cc
 * \brief Specialize bounded logical Core and ordered PipeNet domains.
 */

#include "../op/builtin.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <map>
#include <string>
#include <unordered_map>

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
    return result.dtype().is_handle() ? result : analyzer_.Simplify(result);
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
      if (op->annotations.count("num_stages"))
        TVM_FFI_THROW(NotImplementedError)
            << "[NormalizeTenstorrentTopology] multi-Core/PipeNet combined "
               "with T.Pipelined is deferred";
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

PrimFunc NormalizeTopology(PrimFunc func) {
  if (func->GetAttr<Integer>("tt.topology_normalized").has_value())
    return func;
  auto grid = func->GetAttr<ffi::Array<PrimExpr>>("tt.launch_grid");
  if (!grid.has_value() || grid.value().size() != 2)
    Reject("normalized launch grid is missing");
  if (!as_const_int(grid.value()[0]) || !as_const_int(grid.value()[1]))
    Reject("logical Core grid must have static integer extents");
  int64_t gx = *as_const_int(grid.value()[0]),
          gy = *as_const_int(grid.value()[1]);
  bool pipes = false;
  PostOrderVisit(func->body, [&](const ffi::ObjectRef &object) {
    if (const auto *call = object.as<CallNode>()) {
      if (const auto *op = call->op.as<OpNode>())
        pipes |= std::string(op->name).rfind("tl.tt.pipe_", 0) == 0 ||
                 std::string(op->name).rfind("tl.tt.is_", 0) == 0;
    }
    if (const auto *loop = object.as<ForNode>())
      pipes |= loop->annotations.count("tl.tt.foreach_src") ||
               loop->annotations.count("tl.tt.foreach_dst");
  });
  if (gx == 1 && gy == 1 && !pipes)
    return func;
  if (gx <= 0 || gy <= 0 || gx > 256 || gy > 256 || gx * gy > 256)
    TVM_FFI_THROW(NotImplementedError)
        << "[NormalizeTenstorrentTopology] static Core grid is limited to 256 "
           "cores";
  if (!func->body.as<ForNode>() ||
      !func->body.as<ForNode>()->body.as<ForNode>())
    Reject("normalized launch must retain the logical Core x/y loop prefix");
  For xloop = Downcast<For>(func->body);
  For yloop = Downcast<For>(xloop->body);
  ffi::Array<Stmt> cores;
  std::map<int64_t, std::string> definitions;
  for (int64_t x = 0; x < gx; ++x) {
    for (int64_t y = 0; y < gy; ++y) {
      CoreSpecializer specializer(x, y, gx, gy, xloop->loop_var,
                                  yloop->loop_var, &definitions);
      Stmt body = specializer(yloop->body);
      SBlock block({}, {}, {}, "tt_core", body, std::nullopt, {}, {},
                   {{"tt.core_x", Integer(x)}, {"tt.core_y", Integer(y)}},
                   func->span);
      cores.push_back(SBlockRealize({}, const_true(), block, func->span));
    }
  }
  yloop.CopyOnWrite()->body =
      cores.size() == 1 ? cores[0] : SeqStmt(cores, func->span);
  xloop.CopyOnWrite()->body = yloop;
  func.CopyOnWrite()->body = xloop;
  Records original_records;
  auto decoder = ffi::Function::GetGlobal("tl.tenstorrent.DecodePipeNet");
  for (const auto &[net, text] : definitions) {
    Records records = (*decoder)(text).cast<Records>();
    for (const Record &record : records)
      original_records.push_back(record);
  }
  func = WithAttr(std::move(func), "tt.topology_records", original_records);
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

// Internal plan-contract checks: compile this translation unit against the same
// headers and libraries as form_device_program.cc. No production registry hooks.
#include <tvm/ffi/base_details.h>
#undef TVM_FFI_STATIC_INIT_BLOCK
#define TVM_FFI_STATIC_INIT_BLOCK() static void UnusedTestRegistration()
#include "../../src/tenstorrent/transform/form_device_program.cc"
#include <iostream>

int main() {
  using namespace tvm;
  using namespace tvm::tirx;
  using namespace tvm::tl::tenstorrent;
  using Planner = GeneralDataflowPlanner;
  auto op = [](const Op &kind, ffi::Array<PrimExpr> args) -> Stmt {
    return Evaluate(Call(DataType::Handle(), kind, args));
  };
  Stmt issue = op(tensor_to_dfb_nd(), {Integer(0), Integer(0)});
  Stmt wait = op(dfb_copy_wait(), {Integer(0), Integer(1)});
  Stmt send = op(dfb_pipe_send(), {Integer(0), Integer(0)});
  ffi::Array<Stmt> original{issue, wait, send};
  auto verify = [&](ffi::Array<Stmt> operations) {
    Planner::VerifyWindowSchedulePlan({operations}, original, 1);
  };
  verify({issue, wait, send});
  Stmt issue1 = op(tensor_to_dfb_nd(), {Integer(0), Integer(1)});
  Stmt wait1 = op(dfb_copy_wait(), {Integer(1), Integer(1)});
  Stmt send1 = op(dfb_pipe_send(), {Integer(0), Integer(1)});
  Planner::VerifyWindowSchedulePlan(
      {{issue1, wait1, send1, issue, wait, send}},
      {issue, wait, send, issue1, wait1, send1}, 2);
  size_t rejected = 0;
  auto reject = [&](ffi::Array<Stmt> operations) {
    try {
      verify(operations);
    } catch (const std::exception &) {
      ++rejected;
      return;
    }
    throw std::runtime_error("invalid window plan accepted");
  };
  reject({issue, send, wait});
  reject({send, issue, wait});
  reject({issue, wait});
  reject({issue, wait, send, send});
  reject({issue, issue, wait, send});
  reject({wait, issue, send});
  reject({issue, wait, wait, send});
  reject({issue, send});
  reject({issue, op(dfb_copy_wait(), {Integer(0), Integer(2)}), send});
  reject({issue, op(dfb_copy_wait(), {Integer(1), Integer(1)}), send});
  reject({op(tensor_to_dfb_nd(), {Integer(0), Integer(1)}), wait, send});
  reject({Evaluate(Integer(0))});
  reject({op(dfb_copy_wait(), {})});
  reject({op(dfb_copy_wait(), {Var("id", DataType::Int(32)), Integer(1)})});
  size_t loop_rejected = 0;
  auto reject_loop = [&](const Planner::LoopSchedulePlan &plan) {
    try {
      Planner::VerifyLoopSchedulePlan(plan);
    } catch (const std::exception &) {
      ++loop_rejected;
      return;
    }
    throw std::runtime_error("invalid loop plan accepted");
  };
  For loop(Var("i", DataType::Int(32)), Integer(0), Integer(3),
           ForKind::kSerial, Evaluate(Integer(0)));
  Planner::LoopSchedulePlan plan;
  plan.analysis = {loop, LoopStrategy::kPreserve};
  reject_loop(plan);
  plan.preserved_body = loop->body;
  Planner::VerifyLoopSchedulePlan(plan);
  plan.analysis.strategy = LoopStrategy::kExpand;
  reject_loop(plan);
  plan.extent = 1025;
  reject_loop(plan);
  plan.extent = 3;
  Planner::VerifyLoopSchedulePlan(plan);
  plan.analysis.strategy = LoopStrategy::kWindow;
  plan.stages = plan.depth = 2;
  reject_loop(plan);
  plan.analysis.loop = For(loop->loop_var, loop->min, loop->extent,
                          ForKind::kSerial, loop->body, std::nullopt,
                          {{"num_stages", Integer(2)}});
  Planner::VerifyLoopSchedulePlan(plan);
  plan.depth = 1;
  reject_loop(plan);
  plan.depth = 0;
  reject_loop(plan);
  plan.stages = 33;
  plan.depth = 3;
  reject_loop(plan);
  plan.stages = plan.depth = 2;
  plan.extent = 1025;
  reject_loop(plan);
  plan.analysis.loop = For();
  reject_loop(plan);
  std::cout << "window plan: forwarding and independent reorder accepted, "
            << rejected << " malformed schedules and " << loop_rejected
            << " malformed loop plans rejected\n";
}

// Internal planner-contract checks: compile this translation unit against the same
// headers and libraries as form_device_program.cc. No production registry hooks.
//
// 说明：本测试是从旧分支 test_tenstorrent_window_plan.cc 移植后的版本。
// 旧版测的是 GeneralDataflowPlanner::VerifyWindowSchedulePlan / VerifyLoopSchedulePlan
// 两个静态纯函数；refactor/lower 把「计划 + 校验」合并进了 planner 的实例方法，
// 这两个静态函数已不存在，因此这里改为直接测 planner 的可观测契约：
//   1. 构造阶段的前置条件校验（缺 launch grid 必须抛错）
//   2. 分核特化是纯函数（不改输入）、且只替换自身 loop var
//   3. 同一核重复规划结果一致（可重入）
#include <tvm/ffi/base_details.h>
#undef TVM_FFI_STATIC_INIT_BLOCK
#define TVM_FFI_STATIC_INIT_BLOCK() static void UnusedTestRegistration()
#include "../../src/tenstorrent/transform/form_device_program.cc"
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

using namespace tvm;
using namespace tvm::tirx;
using namespace tvm::tl::tenstorrent;

std::string Text(const Stmt &stmt) {
  std::ostringstream os;
  os << stmt;
  return os.str();
}

PrimFunc MakeFrontend(Stmt body, ffi::Optional<ffi::Array<PrimExpr>> grid) {
  ffi::Map<ffi::String, ffi::Any> attrs;
  if (grid.has_value())
    attrs.Set(kLaunchGridAttr, grid.value());
  return PrimFunc({}, body, VoidType(), {}, DictAttrs(std::move(attrs)));
}

int g_failed = 0;

void ExpectTrue(bool condition, const char *what) {
  if (!condition) {
    std::cerr << "FAILED: " << what << "\n";
    ++g_failed;
  }
}

template <typename F> bool Throws(F &&fn) {
  try {
    fn();
  } catch (const std::exception &) {
    return true;
  }
  return false;
}

} // namespace

int main() {
  // ── 1. 构造前置条件：缺少 launch grid 必须拒绝 ──────────────────
  PrimFunc no_grid = MakeFrontend(Evaluate(Integer(0)), std::nullopt);
  ExpectTrue(Throws([&] { GeneralDataflowPlanner p(no_grid, {}); }),
             "planner must reject a frontend without launch grid");

  // ── 2. 分核特化是纯函数，且只替换自身 loop var ─────────────────
  Var core_x("core_x", DataType::Int(32));
  Var core_y("core_y", DataType::Int(32));
  // 非对称表达式：保证 (gx,gy) 的每个组合都产生不同的常量，
  // 否则 core_x + core_y 会让 (0,1) 和 (1,0) 代入后相等而掩盖错误。
  PrimExpr expr = core_x * Integer(16) + core_y + Integer(1);
  Stmt original = Evaluate(expr);
  const std::string before = Text(original);

  StaticIterationSubstituter specialize_x(core_x, Integer(3));
  Stmt step1 = specialize_x(original);
  ExpectTrue(Text(original) == before,
             "specializing core_x must not mutate the input statement");

  StaticIterationSubstituter specialize_y(core_y, Integer(5));
  Stmt step2 = specialize_y(step1);
  ExpectTrue(Text(original) == before,
             "specializing core_y must not mutate the input statement");
  ExpectTrue(Text(step2).find("core_x") == std::string::npos,
             "specialized statement must not mention core_x");
  ExpectTrue(Text(step2).find("core_y") == std::string::npos,
             "specialized statement must not mention core_y");

  // 同一个 substituter 复用两次，结果必须一致（可重入、不累积状态）。
  Stmt again = specialize_y(step1);
  ExpectTrue(Text(again) == Text(step2),
             "repeated specialization must be idempotent");

  // ── 3. 分核特化对 gx*gy 个核互不影响 ───────────────────────────
  std::vector<std::string> lowered;
  for (int64_t index = 0; index < 4; ++index) {
    int64_t x = index / 2, y = index % 2;
    StaticIterationSubstituter sx(core_x, Integer(x));
    StaticIterationSubstituter sy(core_y, Integer(y));
    lowered.push_back(Text(sy(sx(original))));
  }
  ExpectTrue(lowered[0] != lowered[1] && lowered[1] != lowered[2] &&
                 lowered[2] != lowered[3],
             "each core must get its own specialized statement");
  ExpectTrue(Text(original) == before,
             "specializing all cores must leave the shared input untouched");

  if (g_failed != 0) {
    std::cerr << g_failed << " check(s) failed\n";
    return 1;
  }
  std::cout << "test_tenstorrent_planner_contract: all checks passed\n";
  return 0;
}

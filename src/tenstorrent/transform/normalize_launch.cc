/*!
 * \file tenstorrent/transform/normalize_launch.cc
 * \brief Normalize T.Kernel launch loops into logical Tenstorrent Core axes.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace tenstorrent {

using namespace tirx;

namespace {

constexpr const char *kLaunchGridAttr = "tt.launch_grid";
constexpr const char *kLogicalCoreAxis = "tt.logical_core_axis";

using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

void ThrowMalformed(const std::string &message) {
  TVM_FFI_THROW(ValueError) << "Malformed Tenstorrent launch: " << message;
}

void ThrowUnsupported(const std::string &message) {
  TVM_FFI_THROW(NotImplementedError)
      << "Unsupported Tenstorrent launch: " << message;
}

int64_t RequireStaticExtent(const PrimExpr &extent, const std::string &axis) {
  const auto *integer = extent.as<IntImmNode>();
  if (integer == nullptr) {
    ThrowUnsupported("axis '" + axis +
                     "' has a dynamic extent; Phase 1 requires a static "
                     "Core grid");
  }
  if (integer->value <= 0) {
    ThrowMalformed("axis '" + axis + "' must have positive extent");
  }
  return integer->value;
}

void RequireZeroMinimum(const For &loop, const std::string &axis) {
  const auto *minimum = loop->min.as<IntImmNode>();
  if (minimum == nullptr || minimum->value != 0) {
    ThrowMalformed("axis '" + axis + "' must start at zero");
  }
}

ffi::Optional<ffi::String> GetThreadTag(const For &loop) {
  if (loop->kind != ForKind::kThreadBinding ||
      !loop->thread_binding.defined()) {
    return std::nullopt;
  }
  return loop->thread_binding.value()->thread_tag;
}

class ThreadVarUseFinder : public StmtExprVisitor {
public:
  explicit ThreadVarUseFinder(VarSet variables)
      : variables_(std::move(variables)) {}

  bool found() const { return found_; }

  void VisitExpr_(const VarNode *op) final {
    if (variables_.count(ffi::GetRef<Var>(op))) {
      found_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

private:
  VarSet variables_;
  bool found_{false};
};

class ThreadBindingFinder : public StmtVisitor {
public:
  bool found() const { return found_; }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kThreadBinding) {
      found_ = true;
    }
    StmtVisitor::VisitStmt_(op);
  }

private:
  bool found_{false};
};

std::vector<For> CollectLaunchPrefix(const Stmt &body) {
  std::vector<For> loops;
  Stmt current = body;
  if (const auto *realize_node = current.as<SBlockRealizeNode>()) {
    const SBlockRealize realize = ffi::GetRef<SBlockRealize>(realize_node);
    if (!realize->iter_values.empty() || !is_one(realize->predicate) ||
        realize->block->name_hint != "root") {
      return loops;
    }
    current = realize->block->body;
  }
  while (const auto *loop_node = current.as<ForNode>()) {
    For loop = ffi::GetRef<For>(loop_node);
    if (loop->kind != ForKind::kThreadBinding) {
      break;
    }
    loops.push_back(loop);
    current = loop->body;
  }
  return loops;
}

void ValidateCanonicalLaunch(const PrimFunc &func,
                             const ffi::Array<PrimExpr> &grid) {
  if (grid.size() != 2) {
    ThrowMalformed("'tt.launch_grid' must contain exactly x and y extents");
  }
  RequireStaticExtent(grid[0], "core.x");
  RequireStaticExtent(grid[1], "core.y");

  Stmt current = func->body;
  for (const char *axis : {"x", "y"}) {
    const auto *loop_node = current.as<ForNode>();
    if (loop_node == nullptr || loop_node->kind != ForKind::kSerial) {
      ThrowMalformed("canonical launch must retain serial logical Core x/y "
                     "loops");
    }
    For loop = ffi::GetRef<For>(loop_node);
    auto annotation = loop->annotations.Get(kLogicalCoreAxis);
    if (!annotation.has_value()) {
      ThrowMalformed("canonical logical Core loop is missing '" +
                     std::string(kLogicalCoreAxis) + "'");
    }
    auto actual_axis = annotation.value().as<ffi::String>();
    if (!actual_axis.has_value() || actual_axis.value() != axis) {
      ThrowMalformed("canonical logical Core axes must be ordered x then y");
    }
    RequireZeroMinimum(loop, std::string("core.") + axis);
    current = loop->body;
  }

  ThreadBindingFinder finder;
  finder(func->body);
  if (finder.found()) {
    ThrowMalformed("canonical launch still contains a GPU thread-binding "
                   "loop");
  }
}

PrimFunc NormalizeLaunch(PrimFunc func) {
  if (auto launch_grid = func->GetAttr<ffi::Array<PrimExpr>>(kLaunchGridAttr)) {
    ValidateCanonicalLaunch(func, launch_grid.value());
    return func;
  }

  std::vector<For> loops = CollectLaunchPrefix(func->body);
  const std::vector<std::string> expected = {
      "blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "threadIdx.z"};
  if (loops.size() != expected.size()) {
    ThrowMalformed("expected the T.Kernel blockIdx.x/y and "
                   "threadIdx.x/y/z launch prefix");
  }

  VarSet thread_vars;
  for (size_t index = 0; index < loops.size(); ++index) {
    ffi::Optional<ffi::String> tag = GetThreadTag(loops[index]);
    if (!tag.has_value() || tag.value() != expected[index]) {
      const std::string actual =
          tag.has_value() ? std::string(tag.value()) : std::string("<missing>");
      if (actual == "blockIdx.z") {
        ThrowUnsupported("blockIdx.z is not part of the two-dimensional "
                         "Tenstorrent Core grid");
      }
      ThrowMalformed("launch axis " + std::to_string(index) + " is '" + actual +
                     "', expected '" + expected[index] + "'");
    }
    RequireZeroMinimum(loops[index], expected[index]);
    int64_t extent = RequireStaticExtent(loops[index]->extent, expected[index]);
    if (index >= 2) {
      if (extent != 1) {
        ThrowUnsupported(expected[index] + " extent is " +
                         std::to_string(extent) +
                         "; Tenstorrent Phase 1 requires threads=1");
      }
      thread_vars.insert(loops[index]->loop_var);
    }
  }

  Stmt body = loops.back()->body;
  ThreadVarUseFinder use_finder(std::move(thread_vars));
  use_finder(body);
  if (use_finder.found()) {
    ThrowUnsupported("threadIdx values are used in the kernel body");
  }

  for (int index = 1; index >= 0; --index) {
    const For &loop = loops[index];
    ffi::Map<ffi::String, ffi::Any> annotations = loop->annotations;
    annotations.Set(kLogicalCoreAxis, ffi::String(index == 0 ? "x" : "y"));
    body = For(loop->loop_var, loop->min, loop->extent, ForKind::kSerial,
               std::move(body), /*thread_binding=*/std::nullopt,
               std::move(annotations), loop->step, loop->span);
  }

  func.CopyOnWrite()->body = std::move(body);
  ffi::Array<PrimExpr> launch_grid = {loops[0]->extent, loops[1]->extent};
  return WithAttr(std::move(func), kLaunchGridAttr, std::move(launch_grid));
}

} // namespace

tvm::transform::Pass NormalizeTenstorrentLaunch() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    return NormalizeLaunch(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.NormalizeTenstorrentLaunch", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.tenstorrent.transform.NormalizeTenstorrentLaunch",
                        NormalizeTenstorrentLaunch);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

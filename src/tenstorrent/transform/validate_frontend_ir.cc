/*!
 * \file tenstorrent/transform/validate_frontend_ir.cc
 * \brief Validate the frozen Tenstorrent Frontend TIRX contract.
 */

#include "../../op/builtin.h"
#include "../../op/utils.h"
#include "attr.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace tenstorrent {

using namespace tirx;

namespace {

constexpr const char *kAllocBufferAnnotations = "tl.alloc_buffer_annotations";
constexpr const char *kDFBBlockCount = "tt.dfb_block_count";
constexpr const char *kTileShape = "tt.tile_shape";
constexpr const char *kTensorBacked = "tt.tensor_backed";
constexpr const char *kForeachSrc = "tl.tt.foreach_src";
constexpr const char *kForeachDst = "tl.tt.foreach_dst";

bool IsIntegerScalar(const PrimExpr &expr) {
  return expr.dtype().lanes() == 1 &&
         (expr.dtype().is_int() || expr.dtype().is_uint()) &&
         !expr.dtype().is_bool();
}

using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

void ThrowMalformed(const std::string &message) {
  TVM_FFI_THROW(ValueError) << "Malformed Tenstorrent frontend IR: " << message;
}

void ThrowUnsupported(const std::string &message) {
  TVM_FFI_THROW(NotImplementedError)
      << "Unsupported Tenstorrent frontend IR: " << message;
}

bool StartsWith(const std::string &value, const char *prefix) {
  return value.rfind(prefix, 0) == 0;
}

ffi::Optional<ffi::String> GetThreadTag(const For &loop) {
  if (loop->kind != ForKind::kThreadBinding ||
      !loop->thread_binding.defined()) {
    return std::nullopt;
  }
  return loop->thread_binding.value()->thread_tag;
}

int64_t RequirePositiveStaticExtent(const For &loop, const std::string &tag) {
  const auto *extent = loop->extent.as<IntImmNode>();
  if (extent == nullptr) {
    ThrowUnsupported("launch axis '" + tag +
                     "' has a dynamic extent; Phase 1 requires a static "
                     "grid");
  }
  if (extent->value <= 0) {
    ThrowMalformed("launch axis '" + tag + "' must have positive extent, got " +
                   std::to_string(extent->value));
  }
  const auto *minimum = loop->min.as<IntImmNode>();
  if (minimum == nullptr || minimum->value != 0) {
    ThrowMalformed("launch axis '" + tag + "' must start at zero");
  }
  return extent->value;
}

class VarUseCollector : public StmtExprVisitor {
public:
  explicit VarUseCollector(VarSet variables)
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

class BufferStoreCounter : public StmtExprVisitor {
public:
  static size_t Count(const Stmt &body) {
    BufferStoreCounter counter;
    counter.VisitStmt(body);
    return counter.count_;
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    ++count_;
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  size_t count_{0};
};

class FrontendBodyValidator : public StmtExprVisitor {
public:
  void Validate(const Stmt &body) { VisitStmt(body); }

  void VisitStmt_(const ForNode *op) final {
    For loop = ffi::GetRef<For>(op);
    if (op->kind == ForKind::kParallel) {
      ThrowUnsupported(
          "Tenstorrent elementwise compute requires T.Tiles; T.Parallel is "
          "unsupported");
    }
    if (op->kind == ForKind::kVectorized) {
      ThrowUnsupported(
          "vectorized loops are not supported at the Tenstorrent frontend "
          "boundary");
    }
    if (op->kind == ForKind::kThreadBinding) {
      ffi::Optional<ffi::String> tag = GetThreadTag(loop);
      const std::string tag_string =
          tag.has_value() ? std::string(tag.value()) : std::string("<missing>");
      ThrowUnsupported("nested thread binding '" + tag_string +
                       "' is not supported in a Tenstorrent kernel body");
    }

    for (const char *key :
         {"tl_pipeline_order", "tl_pipeline_stage", "tl_pipeline_group"})
      if (op->annotations.count(key))
        ThrowUnsupported(
            "T.Pipelined manual stage/order/group schedules are unsupported");

    bool has_foreach_src = op->annotations.count(kForeachSrc);
    bool has_foreach_dst = op->annotations.count(kForeachDst);
    if (has_foreach_src && has_foreach_dst) {
      ThrowMalformed("a PipeNet foreach loop cannot be both source and "
                     "destination");
    }
    if (has_foreach_src || has_foreach_dst) {
      ThrowUnsupported(
          "legacy PipeNet foreach is not an active Lower input; use "
          "T.comm.transfer/put/broadcast/all_gather/all_reduce");
    }
    for (const char *key : {kForeachSrc, kForeachDst}) {
      if (auto descriptor = op->annotations.Get(key)) {
        if (!descriptor.value().as<ffi::String>().has_value()) {
          ThrowMalformed(std::string("PipeNet annotation '") + key +
                         "' must contain the frozen serialized descriptor");
        }
        if (op->kind != ForKind::kSerial) {
          ThrowMalformed(std::string("PipeNet annotation '") + key +
                         "' must be attached to a serial foreach loop");
        }
      }
    }
    const bool tiles_scope = op->annotations.count(kTTTilesScope);
    if (tiles_scope) {
      const size_t stores = BufferStoreCounter::Count(op->body);
      if (stores != 1) {
        ThrowUnsupported(
            "T.Tiles body must contain exactly one BufferStore assignment; "
            "found " +
            std::to_string(stores));
      }
      ++tiles_depth_;
    }
    StmtExprVisitor::VisitStmt_(op);
    if (tiles_scope)
      --tiles_depth_;
  }

  void VisitStmt_(const WhileNode *op) final {
    ThrowUnsupported("while loops are not supported in the Phase 1 "
                     "structured frontend subset");
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      ThrowUnsupported("pre-materialized GPU thread_extent attributes are "
                       "not accepted at the Tenstorrent frontend boundary");
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (tiles_depth_ == 0) {
      ThrowUnsupported(
          "BufferStore compute must be enclosed by T.Tiles or represented by "
          "a supported TileOp");
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    if (!is_one(op->predicate)) {
      ThrowUnsupported(
          "non-trivial block predicates/T.where are not supported at the "
          "Tenstorrent frontend boundary");
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (const auto *operator_node = op->op.as<OpNode>()) {
      const std::string name = operator_node->name;
      if (StartsWith(name, "tl.cuda.") || StartsWith(name, "tl.rocm.") ||
          StartsWith(name, "tl.metal.") || StartsWith(name, "tir.ptx") ||
          StartsWith(name, "tl.tileop.wgmma") ||
          StartsWith(name, "tl.tileop.tcgen05") ||
          StartsWith(name, "tl.tileop.mfma")) {
        ThrowUnsupported("operation '" + name +
                         "' has target-specific GPU semantics");
      }
      if (StartsWith(name, "tl.tt.pipe_") || StartsWith(name, "tl.tt.is_")) {
        ThrowUnsupported(
            "legacy Pipe/PipeRef operations are not active Lower inputs; use "
            "the high-level T.comm communication operations");
      }
      if (name == "tl.tt.transfer") {
        ValidateCommunicationCall(ffi::GetRef<Call>(op), name, 10);
        const auto *contract = op->args[8].as<StringImmNode>();
        if (contract == nullptr || (contract->value != "point_to_point" &&
                                    contract->value != "collective")) {
          ThrowMalformed(
              "tl.tt.transfer contract must be point_to_point or collective");
        }
      } else if (name == "tl.tt.all_gather") {
        ValidateCommunicationCall(ffi::GetRef<Call>(op), name, 8);
        const auto *axis = op->args[6].as<IntImmNode>();
        if (axis == nullptr || (axis->value != 0 && axis->value != 1)) {
          ThrowMalformed(
              "tl.tt.all_gather axis must be the normalized value 0 or 1");
        }
      } else if (name == "tl.tt.all_reduce") {
        ValidateCommunicationCall(ffi::GetRef<Call>(op), name, 8);
        const auto *reduction = op->args[6].as<StringImmNode>();
        if (reduction == nullptr ||
            (reduction->value != "sum" && reduction->value != "max" &&
             reduction->value != "min")) {
          ThrowMalformed("tl.tt.all_reduce operation must be sum, max or min");
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const SBlockNode *op) final {
    SBlock block = ffi::GetRef<SBlock>(op);
    ValidateAllocationMetadata(block);
    if (!op->match_buffers.empty()) {
      ThrowUnsupported(
          "match-buffer aliases are not supported at the Tenstorrent frontend "
          "boundary");
    }
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  static void ValidateCommunicationCall(const Call &call,
                                        const std::string &name,
                                        size_t expected_arity) {
    if (call->args.size() != expected_arity || !call.dtype().is_handle()) {
      ThrowMalformed(name + " must retain its frozen " +
                     std::to_string(expected_arity) +
                     "-operand opaque frontend encoding");
    }
    BufferRegion source =
        NormalizeToAccessRegion(call->args[0], kAccessRead).region;
    BufferRegion destination =
        NormalizeToAccessRegion(call->args[1], kAccessWrite).region;
    auto shared = [](const Buffer &buffer) {
      const std::string scope = buffer.scope();
      return scope == "shared" || scope == "shared.dyn";
    };
    if (!shared(source->buffer) || !shared(destination->buffer)) {
      ThrowMalformed(
          name + " requires shared/shared.dyn source and destination buffers");
    }
    for (size_t index = 2; index < 6; ++index) {
      if (!IsIntegerScalar(call->args[index])) {
        ThrowMalformed(name +
                       " coordinates must be scalar integer expressions");
      }
    }
    if (name == "tl.tt.transfer") {
      for (size_t index = 6; index < 8; ++index) {
        if (!IsIntegerScalar(call->args[index])) {
          ThrowMalformed(name +
                         " coordinates must be scalar integer expressions");
        }
      }
    }
    const auto *site = call->args[expected_arity - 1].as<IntImmNode>();
    if (site == nullptr || site->dtype.is_bool() || site->value < 0) {
      ThrowMalformed(name +
                     " site_id must be a nonnegative compile-time integer");
    }
  }

  static void ValidateAllocationMetadata(const SBlock &block) {
    auto annotation = block->annotations.Get(kAllocBufferAnnotations);
    if (!annotation.has_value()) {
      return;
    }
    auto metadata =
        annotation.value()
            .as<ffi::Map<Var, ffi::Map<ffi::String, ffi::ObjectRef>>>();
    if (!metadata.has_value()) {
      ThrowMalformed(std::string("'") + kAllocBufferAnnotations +
                     "' must be Map<Var, Map<String, ObjectRef>>");
    }

    VarSet allocated_data;
    for (const Buffer &buffer : block->alloc_buffers) {
      allocated_data.insert(buffer->data);
    }
    for (const auto &[data, entries] : metadata.value()) {
      if (!allocated_data.count(data)) {
        ThrowMalformed("allocation metadata refers to a buffer that is not "
                       "allocated by the containing SBlock");
      }
      ValidateMetadataEntries(entries);
    }
  }

  static void ValidateMetadataEntries(
      const ffi::Map<ffi::String, ffi::ObjectRef> &entries) {
    for (const auto &[key, value] : entries) {
      const std::string key_string = key;
      if (key_string == kTensorBacked) {
        ThrowUnsupported("'tt.tensor_backed' is deferred beyond Phase 1");
      }
      if (StartsWith(key_string, "tt.") && key_string != kDFBBlockCount &&
          key_string != kTileShape) {
        ThrowUnsupported("allocation annotation '" + key_string +
                         "' is not part of the frozen Phase 1 contract");
      }
      if (key_string == kDFBBlockCount) {
        const auto *count = value.as<IntImmNode>();
        if (count == nullptr) {
          ThrowMalformed("'tt.dfb_block_count' must be a compile-time "
                         "integer");
        }
        if (count->value < 1 || count->value > 32) {
          ThrowMalformed("'tt.dfb_block_count' must be in [1, 32]");
        }
      }
      if (key_string == kTileShape) {
        auto shape = value.as<ffi::Array<PrimExpr>>();
        if (!shape.has_value() || shape.value().size() != 2) {
          ThrowMalformed("'tt.tile_shape' must be a pair of compile-time "
                         "integers");
        }
        for (const PrimExpr &extent : shape.value()) {
          const auto *integer = extent.as<IntImmNode>();
          if (integer == nullptr || integer->value <= 0) {
            ThrowMalformed("'tt.tile_shape' entries must be positive "
                           "compile-time integers");
          }
          if (integer->value != 32) {
            ThrowUnsupported("Phase 1 supports only 32x32 Tenstorrent "
                             "tiles");
          }
        }
      }
    }
  }

  int tiles_depth_{0};
};

std::vector<For> CollectLaunchPrefix(const PrimFunc &func) {
  std::vector<For> loops;
  Stmt current = func->body;
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

void ValidateTarget(const PrimFunc &func, const std::string &function_name) {
  ffi::Optional<Target> target = func->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.defined()) {
    ThrowMalformed("function '" + function_name +
                   "' has no bound target; BindTarget must run first");
  }
  if (target.value()->kind->name != "tenstorrent") {
    ThrowMalformed("function '" + function_name + "' is bound to target '" +
                   std::string(target.value()->kind->name) +
                   "', expected 'tenstorrent'");
  }
  ffi::Optional<ffi::String> arch =
      target.value()->GetAttr<ffi::String>("arch");
  if (!arch.has_value()) {
    ThrowMalformed("function '" + function_name +
                   "' has no Tenstorrent architecture");
  }
  if (arch.value() != "wormhole_b0" && arch.value() != "blackhole") {
    ThrowUnsupported("architecture '" + std::string(arch.value()) +
                     "' is not supported");
  }
}

void ValidateParameters(const PrimFunc &func,
                        const std::string &function_name) {
  for (const Var &parameter : func->params) {
    if (!parameter->dtype.is_handle()) {
      ThrowUnsupported("runtime scalar parameter '" +
                       std::string(parameter->name_hint) + "' in function '" +
                       function_name + "' has no Phase 1 TTL ABI");
    }
    if (!func->buffer_map.count(parameter)) {
      ThrowMalformed("handle parameter '" + std::string(parameter->name_hint) +
                     "' in function '" + function_name +
                     "' is missing from buffer_map");
    }
  }
  for (const auto &[parameter, buffer] : func->buffer_map) {
    bool is_parameter = false;
    for (const Var &candidate : func->params) {
      if (candidate.same_as(parameter)) {
        is_parameter = true;
        break;
      }
    }
    if (!is_parameter) {
      ThrowMalformed("buffer_map contains a key that is not a function "
                     "parameter in function '" +
                     function_name + "'");
    }
  }
}

void ValidateLaunch(const PrimFunc &func, const std::string &function_name) {
  std::vector<For> loops = CollectLaunchPrefix(func);
  const std::vector<std::string> expected = {
      "blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "threadIdx.z"};
  if (loops.empty()) {
    ThrowMalformed("function '" + function_name +
                   "' has no recognizable T.Kernel launch nest");
  }
  if (loops.size() != expected.size()) {
    ThrowMalformed("function '" + function_name +
                   "' must have exactly one blockIdx.x/y and "
                   "threadIdx.x/y/z launch prefix");
  }

  VarSet thread_vars;
  for (size_t index = 0; index < loops.size(); ++index) {
    ffi::Optional<ffi::String> tag = GetThreadTag(loops[index]);
    if (!tag.has_value() || tag.value() != expected[index]) {
      const std::string actual =
          tag.has_value() ? std::string(tag.value()) : std::string("<missing>");
      ThrowMalformed("function '" + function_name + "' launch axis " +
                     std::to_string(index) + " is '" + actual +
                     "', expected '" + expected[index] + "'");
    }
    int64_t extent = RequirePositiveStaticExtent(loops[index], expected[index]);
    if (StartsWith(expected[index], "threadIdx.")) {
      if (extent != 1) {
        ThrowUnsupported("function '" + function_name + "' uses " +
                         expected[index] + " extent " + std::to_string(extent) +
                         "; Tenstorrent Phase 1 requires threads=1");
      }
      thread_vars.insert(loops[index]->loop_var);
    }
  }

  Stmt kernel_body = loops.back()->body;
  VarUseCollector thread_use(std::move(thread_vars));
  thread_use(kernel_body);
  if (thread_use.found()) {
    ThrowUnsupported("function '" + function_name +
                     "' uses threadIdx values in its kernel body");
  }

  FrontendBodyValidator body_validator;
  body_validator.Validate(kernel_body);
}

IRModule ValidateModule(IRModule mod) {
  if (mod->functions.empty())
    ThrowMalformed("IRModule contains no TIRX PrimFunc");
  if (mod->functions.size() != 1)
    ThrowUnsupported("Tenstorrent Lower accepts exactly one frontend PrimFunc");
  for (const auto &[global_var, base_func] : mod->functions) {
    auto func = base_func.as<PrimFunc>();
    if (!func.has_value()) {
      ThrowUnsupported("non-PrimFunc global '" +
                       std::string(global_var->name_hint) +
                       "' is outside the single-operation Lower contract");
    }
    const std::string function_name = global_var->name_hint;
    ValidateTarget(func.value(), function_name);
    ValidateParameters(func.value(), function_name);
    ValidateLaunch(func.value(), function_name);
  }
  return mod;
}

} // namespace

tvm::transform::Pass ValidateTenstorrentFrontendIR() {
  auto pass_func = [](IRModule mod,
                      const tvm::transform::PassContext &ctx) -> IRModule {
    return ValidateModule(std::move(mod));
  };
  return tvm::transform::CreateModulePass(
      pass_func, 0, "tl.tenstorrent.ValidateTenstorrentFrontendIR", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.tenstorrent.transform.ValidateTenstorrentFrontendIR",
      ValidateTenstorrentFrontendIR);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

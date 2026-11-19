/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */
#ifndef TVM_TL_TENSTORRENT_TRANSFORM_COMPUTE_SUPPORT_H_
#define TVM_TL_TENSTORRENT_TRANSFORM_COMPUTE_SUPPORT_H_

#include <string>
#include <tvm/ir/op.h>
#include <tvm/tirx/expr.h>

namespace tvm::tl::tenstorrent {

// Lower-stage scalar DAG support; TTL codegen has a separate capability gate.
inline bool IsSupportedComputeNode(const PrimExpr &expr) {
  using namespace tirx;
  return expr.as<IntImmNode>() || expr.as<FloatImmNode>() ||
         expr.as<CastNode>() || expr.as<CallNode>() || expr.as<AddNode>() ||
         expr.as<SubNode>() || expr.as<MulNode>() || expr.as<DivNode>() ||
         expr.as<MinNode>() || expr.as<MaxNode>() || expr.as<EQNode>() ||
         expr.as<NENode>() || expr.as<LTNode>() || expr.as<LENode>() ||
         expr.as<GTNode>() || expr.as<GENode>() || expr.as<AndNode>() ||
         expr.as<OrNode>() || expr.as<NotNode>() || expr.as<SelectNode>();
}

// Integer tiles only survive TT-Lang lowering through SFPU integer binary ops
// (add/sub/mul) and the explicit broadcast call; compare, min/max, div, select
// and every unary intrinsic are rejected downstream
// (ConvertTTLTileOpsToTTKernel.cpp:427,487).
inline bool IsSupportedIntegerComputeNode(const PrimExpr &expr) {
  using namespace tirx;
  if (expr.as<IntImmNode>() || expr.as<BufferLoadNode>() ||
      expr.as<CastNode>() || expr.as<AddNode>() || expr.as<SubNode>() ||
      expr.as<MulNode>())
    return true;
  if (const auto *call = expr.as<CallNode>()) {
    const auto *operation = call->op.as<OpNode>();
    return operation != nullptr &&
           operation->name.find("compute_broadcast") != std::string::npos;
  }
  return false;
}

// Device IR expressions have already replaced buffer leaves by typed load
// calls, so the verifier accepts the same integer subset plus those loads.
inline bool IsSupportedDeviceIntegerComputeNode(const PrimExpr &expr) {
  using namespace tirx;
  if (expr.as<IntImmNode>() || expr.as<VarNode>() || expr.as<CastNode>() ||
      expr.as<AddNode>() || expr.as<SubNode>() || expr.as<MulNode>())
    return true;
  if (const auto *call = expr.as<CallNode>()) {
    const auto *operation = call->op.as<OpNode>();
    if (operation == nullptr)
      return false;
    const std::string &name = operation->name;
    return name == "tl.tt.dfb_operand" ||
           name == "tl.tt.compute_value_operand" ||
           name == "tl.tt.dfb_load" ||
           name == "tl.tt.compute_value_load" ||
           name.find("compute_broadcast") != std::string::npos;
  }
  return false;
}

inline bool IsSupportedComputeUnary(const std::string &name) {
  return name == "tirx.exp" || name == "tirx.log" || name == "tirx.sqrt" ||
         name == "tirx.tanh" || name == "tirx.rsqrt" || name == "tirx.floor" ||
         name == "tirx.ceil" || name == "tirx.exp2" || name == "tirx.log2" ||
         name == "tirx.sin" || name == "tirx.cos" || name == "tirx.fabs";
}

} // namespace tvm::tl::tenstorrent
#endif

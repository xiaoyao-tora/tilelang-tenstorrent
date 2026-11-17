/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */
#ifndef TVM_TL_TENSTORRENT_TRANSFORM_COMPUTE_SUPPORT_H_
#define TVM_TL_TENSTORRENT_TRANSFORM_COMPUTE_SUPPORT_H_

#include <string>
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

inline bool IsSupportedComputeUnary(const std::string &name) {
  return name == "tirx.exp" || name == "tirx.log" || name == "tirx.sqrt" ||
         name == "tirx.tanh" || name == "tirx.rsqrt" || name == "tirx.floor" ||
         name == "tirx.ceil" || name == "tirx.exp2" || name == "tirx.log2" ||
         name == "tirx.sin" || name == "tirx.cos" || name == "tirx.fabs";
}

} // namespace tvm::tl::tenstorrent
#endif

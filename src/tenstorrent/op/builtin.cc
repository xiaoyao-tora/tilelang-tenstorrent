/*!
 * \file tl/tenstorrent/op/builtin.cc
 * \brief Registration of Tenstorrent-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {
namespace tenstorrent {

using namespace tirx;

#define TIR_DEFINE_TT_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl.tt." #OpName);                           \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl.tt." #OpName)                                            \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", "tt." #OpName)

TIR_DEFINE_TT_BUILTIN(is_src).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TT_BUILTIN(is_dst).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TT_BUILTIN(is_active).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TT_BUILTIN(pipe_src).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TT_BUILTIN(pipe_dst).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TT_BUILTIN(pipe_dst_range)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TT_BUILTIN(pipe_send).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(pipe_recv).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(tile_add).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(tile_compute)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(dfb_compute)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(dfb_load).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TT_BUILTIN(tensor_to_dfb_nd)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(dfb_to_tensor_nd)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(dfb_reserve)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(dfb_wait).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(dfb_copy_wait).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(dfb_release).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(tensor_to_dfb)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(dfb_to_tensor)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TT_BUILTIN(dfb_add).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

#undef TIR_DEFINE_TT_BUILTIN

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

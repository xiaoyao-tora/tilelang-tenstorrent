/*!
 * \file tl/tenstorrent/op/builtin.h
 * \brief Tenstorrent-specific TileLang intrinsic Ops.
 */

#ifndef TVM_TL_TENSTORRENT_OP_BUILTIN_H_
#define TVM_TL_TENSTORRENT_OP_BUILTIN_H_

#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {
namespace tenstorrent {

/*!
 * \brief PipeNet execution-domain predicates.
 *
 * Each op receives the serialized, operation-local PipeNet descriptor. The
 * current Core coordinate is implicit and is materialized by the Tenstorrent
 * lowering pipeline.
 */
TVM_DLL const Op &is_src();
TVM_DLL const Op &is_dst();
TVM_DLL const Op &is_active();

/*!
 * \brief Access coordinates of the selected Pipe in a PipeNet foreach region.
 *
 * The first argument is the foreach loop's selected-pipe index. Accessors
 * additionally receive a dimension or endpoint selector as appropriate.
 */
TVM_DLL const Op &pipe_src();
TVM_DLL const Op &pipe_dst();
TVM_DLL const Op &pipe_dst_range();

/*!
 * \brief Transfer a complete shared-memory payload through a selected Pipe.
 */
TVM_DLL const Op &pipe_send();
TVM_DLL const Op &pipe_recv();

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TENSTORRENT_OP_BUILTIN_H_

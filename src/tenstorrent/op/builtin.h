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

/*!
 * \brief Canonical Phase 2 Add operation over normalized BufferRegions.
 *
 * LegalizeTenstorrentTileOps creates this intermediate operation. Program
 * formation consumes it and replaces it with logical-DFB Device IR ops.
 */
TVM_DLL const Op &tile_add();

/*!
 * \brief General computation over full normalized BufferRegions.
 *
 * Operands are output followed by inputs. Legalization fixes operation kind,
 * dtypes, logical domain and operation-specific parameters in annotations.
 * Formation replaces regions with versioned DFB IDs in dfb_compute, and
 * expression BufferLoads with pure dfb_load values. No frontend block survives.
 */
TVM_DLL const Op &tile_compute();
TVM_DLL const Op &dfb_compute();
TVM_DLL const Op &dfb_load();

/*!
 * \brief Logical-DFB transaction and dataflow operations in Device TIR.
 *
 * DFB and Tensor operands are represented by stable module-table indices.
 * Transfer operations additionally carry a two-dimensional Tensor region as
 * row start, column start, row extent, and column extent.
 */
TVM_DLL const Op &dfb_reserve();
TVM_DLL const Op &dfb_wait();
TVM_DLL const Op &tensor_to_dfb();
TVM_DLL const Op &dfb_to_tensor();
TVM_DLL const Op &dfb_add();

/*!
 * \brief Schema v2 transfers with rank-preserving Tensor slices.
 *
 * Arguments are tensor/DFB (or DFB/tensor), then (minimum, extent) pairs
 * for every logical Tensor axis. The entire logical DFB value is transferred.
 */
TVM_DLL const Op &tensor_to_dfb_nd();
TVM_DLL const Op &dfb_to_tensor_nd();

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TENSTORRENT_OP_BUILTIN_H_

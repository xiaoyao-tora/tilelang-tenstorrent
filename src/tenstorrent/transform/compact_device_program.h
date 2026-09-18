/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#ifndef TVM_TL_TENSTORRENT_TRANSFORM_COMPACT_DEVICE_PROGRAM_H_
#define TVM_TL_TENSTORRENT_TRANSFORM_COMPACT_DEVICE_PROGRAM_H_

#include <tvm/ir/module.h>

namespace tvm {
namespace tl {
namespace tenstorrent {

// Representation helpers used by the existing requirement inference/verifier.
// These are not compiler passes and do not alter the backend's pass order.
IRModule CompactDeviceProgram(const IRModule &mod, bool force = false);
IRModule ExpandDeviceProgram(const IRModule &mod);

} // namespace tenstorrent
} // namespace tl
} // namespace tvm
#endif // TVM_TL_TENSTORRENT_TRANSFORM_COMPACT_DEVICE_PROGRAM_H_

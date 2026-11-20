/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file tenstorrent/transform/verify_device_ir.h
 * \brief Verify the frozen Tenstorrent Device TIR schema.
 */
#ifndef TVM_TL_TENSTORRENT_TRANSFORM_VERIFY_DEVICE_IR_H_
#define TVM_TL_TENSTORRENT_TRANSFORM_VERIFY_DEVICE_IR_H_

#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {
namespace tenstorrent {

/*! \brief Create the read-only Tenstorrent Device TIR verifier. */
TVM_DLL tvm::transform::Pass VerifyTenstorrentDeviceIR();

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TENSTORRENT_TRANSFORM_VERIFY_DEVICE_IR_H_

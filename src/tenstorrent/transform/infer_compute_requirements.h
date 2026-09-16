/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#ifndef TVM_TL_TENSTORRENT_TRANSFORM_INFER_COMPUTE_REQUIREMENTS_H_
#define TVM_TL_TENSTORRENT_TRANSFORM_INFER_COMPUTE_REQUIREMENTS_H_
#include "../ir/device_ir.h"
#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>
namespace tvm {
namespace tl {
namespace tenstorrent {
/*! \brief Independently derive requirements and check persistent fragment
 * lifetimes. */
TVM_DLL ComputeRequirements
DeriveComputeRequirements(const IRModule &mod, const tirx::PrimFunc &func);
TVM_DLL tvm::transform::Pass InferTenstorrentComputeRequirements();
} // namespace tenstorrent
} // namespace tl
} // namespace tvm
#endif

/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#ifndef TVM_TL_TENSTORRENT_TRANSFORM_DEVICE_STATEMENT_COMPACTION_H_
#define TVM_TL_TENSTORRENT_TRANSFORM_DEVICE_STATEMENT_COMPACTION_H_

#include <cstdint>
#include <vector>

#include <tvm/ffi/string.h>
#include <tvm/ir/source_map.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {
namespace tenstorrent {

// Fold repeated device statements into serial loops without changing operation
// order, source provenance, or identity-bearing operands.
tirx::Stmt CompactDeviceStatements(const tirx::Stmt &body);
tirx::Stmt CompactDeviceStatementsRecursive(const tirx::Stmt &body);

struct DeviceCoreBody {
  int64_t x;
  int64_t y;
  tirx::Stmt body;
};

// Merge concrete analysis bodies into shared work-domain regions. Integer
// leaves that vary with Core coordinates become typed affine expressions;
// repeated flat schedules are folded into serial loops afterwards.
tirx::Stmt ParameterizeDeviceCoreStatements(
    const std::vector<DeviceCoreBody> &bodies, int64_t grid_x, int64_t grid_y,
    const ffi::String &owner_slot, const Span &span);

// Restore generated loops, charging each Evaluate against a shared module-wide
// budget. Unsupported control flow and overflowing integer expressions fail.
tirx::Stmt ExpandDeviceStatements(const tirx::Stmt &body, int64_t *budget);

} // namespace tenstorrent
} // namespace tl
} // namespace tvm
#endif // TVM_TL_TENSTORRENT_TRANSFORM_DEVICE_STATEMENT_COMPACTION_H_

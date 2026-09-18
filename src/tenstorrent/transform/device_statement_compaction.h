/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#ifndef TVM_TL_TENSTORRENT_TRANSFORM_DEVICE_STATEMENT_COMPACTION_H_
#define TVM_TL_TENSTORRENT_TRANSFORM_DEVICE_STATEMENT_COMPACTION_H_

#include <cstdint>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {
namespace tenstorrent {

// Fold repeated flat device instructions into serial loops without changing
// operation order, source provenance, or identity-bearing operands.
tirx::Stmt CompactDeviceStatements(const tirx::Stmt &body);

// Restore generated loops, charging each Evaluate against a shared module-wide
// budget. Unsupported control flow and overflowing integer expressions fail.
tirx::Stmt ExpandDeviceStatements(const tirx::Stmt &body, int64_t *budget);

} // namespace tenstorrent
} // namespace tl
} // namespace tvm
#endif // TVM_TL_TENSTORRENT_TRANSFORM_DEVICE_STATEMENT_COMPACTION_H_

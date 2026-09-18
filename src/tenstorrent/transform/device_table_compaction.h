/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#ifndef TVM_TL_TENSTORRENT_TRANSFORM_DEVICE_TABLE_COMPACTION_H_
#define TVM_TL_TENSTORRENT_TRANSFORM_DEVICE_TABLE_COMPACTION_H_

#include <cstdint>
#include <vector>

#include <tvm/ir/module.h>

#include "../ir/device_ir.h"

namespace tvm {
namespace tl {
namespace tenstorrent {

/*! \brief Encode a lossless integer column, preferring affine/periodic forms.
 */
DeviceIRIntColumn MakeDeviceIntColumn(const std::vector<int64_t> &values);
/*! \brief Validate and decode a column within the logical expansion budget. */
std::vector<int64_t> ExpandDeviceIntColumn(const DeviceIRIntColumn &column,
                                           int64_t expected_count = -1);
/*! \brief Compact DFB/value tables without changing their logical instances. */
IRModule CompactDeviceTables(IRModule mod);
/*! \brief Check compact table allocation budgets without decoding columns. */
bool DeviceTablesFitExpansionBudget(const IRModule &mod);
/*! \brief Validate and restore compact tables in their original order. */
IRModule ExpandDeviceTables(IRModule mod);

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TENSTORRENT_TRANSFORM_DEVICE_TABLE_COMPACTION_H_

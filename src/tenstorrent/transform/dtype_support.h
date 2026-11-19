/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */
#ifndef TVM_TL_TENSTORRENT_TRANSFORM_DTYPE_SUPPORT_H_
#define TVM_TL_TENSTORRENT_TRANSFORM_DTYPE_SUPPORT_H_

#include <sstream>
#include <string>
#include <tvm/runtime/data_type.h>

namespace tvm::tl::tenstorrent {

// Tenstorrent dtype capability table. Every predicate here is derived from the
// downstream contract verified in
// demonstrate/temp_codegen_test/scaleup/int_spike_20261117/REPORT.md:
//
//   storage / copy / passthrough : bfloat16, float32, si32, u32, u16
//   integer compute              : si32, u32, u16 (add, sub, mul, broadcast)
//   rejected downstream          : u8 (ComputeTarget.cpp:51-56),
//                                  float16 (tt-metal has no FLOAT16 type),
//                                  integer div/max/min/compare/unary/reduce/
//                                  transpose/fill/matmul/typecast
//
// Keep this header the single source of truth: passes must not re-derive their
// own bf16||f32 predicates.

inline std::string DTypeName(DataType dtype) {
  std::ostringstream stream;
  stream << dtype;
  return stream.str();
}

inline bool IsFloat32DType(DataType dtype) {
  return dtype.lanes() == 1 && dtype.is_float() && dtype.bits() == 32;
}

// float16 has no tt-metal storage type (tensor_types.hpp:26-37) and TT-Lang
// aliases it to bfloat16 (dtype_utils.py:188-189), so it must be rejected
// explicitly instead of silently becoming bf16.
inline bool IsFloat16DType(DataType dtype) {
  return dtype.lanes() == 1 && dtype.is_float() && dtype.bits() == 16;
}

// Tensor / DFB / copy element dtype.
inline bool IsSupportedStorageDType(DataType dtype) {
  if (dtype.lanes() != 1)
    return false;
  if (dtype.is_bfloat16())
    return true;
  if (dtype.is_float())
    return dtype.bits() == 32;
  if (dtype.is_int())
    return dtype.bits() == 32;
  if (dtype.is_uint())
    return dtype.bits() == 32 || dtype.bits() == 16;
  return false;
}

// Structured elementwise expression dtype.
inline bool IsSupportedFloatComputeDType(DataType dtype) {
  return dtype.lanes() == 1 &&
         (dtype.is_bfloat16() ||
         (dtype.is_float() && dtype.bits() == 32));
}

inline bool IsSupportedIntegerComputeDType(DataType dtype) {
  if (dtype.lanes() != 1)
    return false;
  if (dtype.is_int())
    return dtype.bits() == 32;
  if (dtype.is_uint())
    return dtype.bits() == 32 || dtype.bits() == 16;
  return false;
}

inline bool IsSupportedExpressionDType(DataType dtype) {
  return IsSupportedFloatComputeDType(dtype) ||
         IsSupportedIntegerComputeDType(dtype);
}

inline bool IsIntegerDType(DataType dtype) {
  return dtype.is_int() || dtype.is_uint();
}

// Destination register width that integer arithmetic pins. TT-Lang derives it
// from the element width when the kernel leaves fp32_dest_acc_en unset, and
// validates an explicit value: si32/u32 add_int_tile requires a 32-bit
// destination while u16 uses InstrModLoadStore::LO16.
inline bool RequiresBits32Destination(DataType dtype) {
  if (IsFloat32DType(dtype))
    return true;
  return IsIntegerDType(dtype) && dtype.bits() == 32;
}

inline bool RequiresBits16Destination(DataType dtype) {
  return IsIntegerDType(dtype) && dtype.bits() == 16;
}

// ---------------------------------------------------------------------------
// Diagnostics. Each message names the downstream constraint that causes the
// rejection so users do not have to bisect the pipeline.
// ---------------------------------------------------------------------------

inline std::string UnsupportedFloat16Message() {
  return "Tenstorrent Device Lower does not support float16: tt-metal defines "
         "no FLOAT16 tensor type (tensor_types.hpp:26-37) and TT-Lang aliases "
         "f16 to bfloat16 (dtype_utils.py:188-189); use bfloat16";
}

inline std::string UnsupportedUInt8Message() {
  return "Tenstorrent Device Lower does not support uint8: TT-Lang compute "
         "passthrough accepts bf16, f16, f32, BFP, si32, u32 and u16 only "
         "(ComputeTarget.cpp:51-56), so no uint8 tile can ever be consumed";
}

inline std::string UnsupportedBFPTypeMessage() {
  return "Tenstorrent Device Lower cannot represent block floating point: TVM "
         "DataType has no BFP type code, so a BFP Tensor/DFB dtype cannot be "
         "expressed in Device IR";
}

// Why a scalar dtype is rejected. Shared by every pass so the reason is
// identical whichever gate fires first.
inline std::string DTypeRejectionReason(DataType dtype) {
  if (IsFloat16DType(dtype))
    return UnsupportedFloat16Message();
  if (dtype.lanes() == 1 && dtype.is_uint() && dtype.bits() == 8)
    return UnsupportedUInt8Message();
  if (dtype.is_bfloat16())
    return UnsupportedBFPTypeMessage();
  return "unsupported scalar dtype " + DTypeName(dtype);
}

inline std::string UnsupportedStorageDTypeMessage(DataType dtype) {
  return "Tenstorrent Device Lower storage supports scalar bfloat16, float32, "
         "int32, uint32 and uint16, got " +
         DTypeName(dtype) + ": " + DTypeRejectionReason(dtype);
}

inline std::string UnsupportedIntegerComputeMessage(DataType dtype) {
  if (dtype.lanes() == 1 && dtype.is_uint() && dtype.bits() == 8)
    return UnsupportedUInt8Message();
  return "Tenstorrent integer compute supports scalar int32, uint32 and uint16, "
         "got " +
         DTypeName(dtype);
}

inline std::string UnsupportedIntegerOperatorMessage(const std::string &owner) {
  return "integer tiles only support elementwise add, sub, mul and broadcast "
         "in '" +
         owner +
         "': TT-Lang lowers integer tiles through SFPU "
         "createIntegerBinaryOp (ConvertTTLTileOpsToTTKernel.cpp:427) and "
         "rejects every other compute primitive (:487)";
}

inline std::string UnsupportedMixedIntegerFloatMessage(
    const std::string &owner) {
  return "structured expression in '" + owner +
         "' mixes integer and floating-point values; TT-Lang elementwise ops "
         "require identical operand and result types";
}

inline std::string UnsupportedIntegerFillMessage(const std::string &owner) {
  return "integer tiles cannot be filled in '" + owner +
         "': TT-Lang fill only accepts an f32 constant attribute "
         "(TTLOps.td:1512-1522), so an integer DFB must be produced by a "
         "copy or an add/sub/mul/broadcast expression";
}

} // namespace tvm::tl::tenstorrent
#endif // TVM_TL_TENSTORRENT_TRANSFORM_DTYPE_SUPPORT_H_

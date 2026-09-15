/*!
 * \file tenstorrent/transform/infer_tensor_layout.cc
 * \brief Infer the basic 32x32 tiled Tenstorrent frontend layout.
 */

#include "../ir/device_ir.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <utility>

namespace tvm {
namespace tl {
namespace tenstorrent {

using namespace tirx;

namespace {

void ThrowMalformed(const std::string &message) {
  TVM_FFI_THROW(ValueError)
      << "Malformed Tenstorrent tensor layout: " << message;
}

void ThrowUnsupported(const std::string &message) {
  TVM_FFI_THROW(NotImplementedError)
      << "Unsupported Tenstorrent tensor layout: " << message;
}

int64_t RequireStaticExtent(const PrimExpr &extent,
                            const std::string &buffer_id, size_t axis) {
  const auto *integer = extent.as<IntImmNode>();
  if (integer == nullptr) {
    ThrowUnsupported("buffer '" + buffer_id + "' shape axis " +
                     std::to_string(axis) + " is dynamic");
  }
  if (integer->value <= 0) {
    ThrowMalformed("buffer '" + buffer_id +
                   "' has a non-positive shape extent");
  }
  return integer->value;
}

bool IsSupportedDType(DataType dtype) {
  return dtype.lanes() == 1 &&
         (dtype.is_bfloat16() || (dtype.is_float() && dtype.bits() == 32));
}

void ValidateTarget(const PrimFunc &func) {
  ffi::Optional<Target> target = func->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.defined() || target.value()->kind->name != "tenstorrent") {
    ThrowMalformed("InferTenstorrentTensorLayout requires a bound "
                   "Tenstorrent target");
  }
  auto arch = target.value()->GetAttr<ffi::String>("arch");
  if (!arch.has_value()) {
    ThrowMalformed("the Tenstorrent target is missing 'arch'");
  }
  if (arch.value() != "wormhole_b0" && arch.value() != "blackhole") {
    ThrowUnsupported("target architecture '" + std::string(arch.value()) + "'");
  }
}

TTBufferMetadata InferOne(const TTBufferMetadata &metadata, bool *changed) {
  if (!metadata.defined() || !metadata->buffer.defined()) {
    ThrowMalformed("'tt.buffer_metadata_table' contains an undefined entry");
  }
  const std::string buffer_id = metadata->buffer_id;
  const std::string kind = metadata->kind;
  if (kind == "scalar_local") {
    return metadata;
  }
  if (kind == "compute_fragment") {
    ThrowUnsupported("compute fragment layout for buffer '" + buffer_id +
                     "' is deferred beyond the Phase 1 basic path");
  }
  if (kind != "tensor" && kind != "logical_dfb_candidate") {
    ThrowMalformed("buffer '" + buffer_id + "' has unknown kind '" + kind +
                   "'");
  }
  if (metadata->shard_spec.has_value()) {
    ThrowUnsupported("sharded layout for buffer '" + buffer_id +
                     "' is not implemented in the Phase 1 basic path");
  }
  const size_t rank = metadata->buffer->shape.size();
  if (rank == 0) {
    ThrowUnsupported("buffer '" + buffer_id +
                     "' must have at least one logical axis");
  }
  arith::Analyzer analyzer;
  const Buffer &buffer = metadata->buffer;
  if (metadata->alias_of.has_value() || !buffer->axis_separators.empty() ||
      !analyzer.CanProveEqual(buffer->elem_offset, 0)) {
    ThrowUnsupported("buffer '" + buffer_id +
                     "' uses a storage alias/view; use the same Buffer for "
                     "supported in-place updates");
  }
  if (!buffer->strides.empty()) {
    if (buffer->strides.size() != rank)
      ThrowMalformed("buffer '" + buffer_id + "' stride rank mismatch");
    PrimExpr stride = Integer(1);
    for (size_t axis = rank; axis-- > 0;) {
      if (!analyzer.CanProveEqual(buffer->strides[axis], stride))
        ThrowUnsupported("buffer '" + buffer_id +
                         "' requires compact row-major strides");
      stride = stride * buffer->shape[axis];
    }
  }
  if (!IsSupportedDType(metadata->buffer->dtype)) {
    ThrowUnsupported("buffer '" + buffer_id +
                     "' has an unsupported dtype; Device Lower supports "
                     "bfloat16 and float32");
  }

  ffi::Array<PrimExpr> tile_shape = metadata->tile_shape;
  ffi::String tile_shape_origin = metadata->tile_shape_origin;
  if (tile_shape.empty()) {
    tile_shape = {IntImm(DataType::Int(32), 32), IntImm(DataType::Int(32), 32)};
    tile_shape_origin = "inferred";
    *changed = true;
  }
  if (tile_shape.size() != 2) {
    ThrowMalformed("buffer '" + buffer_id + "' tile shape must have rank two");
  }
  for (size_t axis = 0; axis < 2; ++axis) {
    const auto *tile_extent = tile_shape[axis].as<IntImmNode>();
    if (tile_extent == nullptr || tile_extent->value <= 0) {
      ThrowMalformed("buffer '" + buffer_id +
                     "' tile extents must be positive static integers");
    }
    if (tile_extent->value != 32) {
      ThrowUnsupported("buffer '" + buffer_id +
                       "' does not use the supported 32x32 tile shape");
    }
  }

  ffi::Array<PrimExpr> tile_grid_shape;
  for (size_t axis = 0; axis < rank; ++axis) {
    int64_t element_extent =
        RequireStaticExtent(metadata->buffer->shape[axis], buffer_id, axis);
    // Batch axes are not tiled. A rank-one reduction result occupies logical
    // row zero of a padded 32xN physical value; the row axis is implicit.
    const int64_t tile_extent = axis + 2 >= rank ? 32 : 1;
    if (element_extent % tile_extent != 0) {
      ThrowUnsupported("buffer '" + buffer_id + "' shape axis " +
                       std::to_string(axis) + " (" +
                       std::to_string(element_extent) +
                       ") requires a padding/mask contract");
    }
    tile_grid_shape.push_back(IntImm(metadata->buffer->shape[axis].dtype(),
                                     element_extent / tile_extent));
  }
  if (metadata->tile_grid_shape.empty()) {
    *changed = true;
  } else {
    if (metadata->tile_grid_shape.size() != rank) {
      ThrowMalformed("buffer '" + buffer_id +
                     "' tile-grid rank must match its logical shape");
    }
    for (size_t axis = 0; axis < rank; ++axis) {
      const auto *existing = metadata->tile_grid_shape[axis].as<IntImmNode>();
      const auto *inferred = tile_grid_shape[axis].as<IntImmNode>();
      if (existing == nullptr || existing->value != inferred->value) {
        ThrowMalformed("buffer '" + buffer_id +
                       "' tile-grid shape conflicts with its element shape");
      }
    }
    tile_grid_shape = metadata->tile_grid_shape;
  }

  ffi::String memory_layout = metadata->memory_layout;
  ffi::String layout_origin = metadata->layout_origin;
  if (memory_layout.empty()) {
    memory_layout = "interleaved";
    layout_origin = "inferred";
    *changed = true;
  } else if (memory_layout != "interleaved") {
    ThrowUnsupported("buffer '" + buffer_id + "' requests memory layout '" +
                     std::string(memory_layout) + "'");
  }

  ffi::Optional<PrimExpr> block_count = metadata->dfb_block_count;
  ffi::String block_count_origin = metadata->block_count_origin;
  if (kind == "logical_dfb_candidate" && !block_count.has_value()) {
    block_count = IntImm(DataType::Int(32), 1);
    block_count_origin = "inferred";
    *changed = true;
  }

  return TTBufferMetadata(
      metadata->buffer_id, metadata->buffer, metadata->kind,
      metadata->global_arg_index, std::move(tile_shape),
      std::move(tile_grid_shape), std::move(memory_layout),
      metadata->shard_spec, std::move(block_count), metadata->tensor_backing,
      metadata->alias_of, std::move(tile_shape_origin),
      std::move(block_count_origin), metadata->tensor_backing_origin,
      std::move(layout_origin), metadata->source_span);
}

PrimFunc InferLayout(PrimFunc func) {
  ValidateTarget(func);
  auto table =
      func->GetAttr<ffi::Array<TTBufferMetadata>>(kBufferMetadataTableAttr);
  if (!table.has_value()) {
    ThrowMalformed("'tt.buffer_metadata_table' is missing; "
                   "NormalizeTenstorrentBufferMetadata must run first");
  }

  bool changed = false;
  ffi::Array<TTBufferMetadata> inferred;
  for (const TTBufferMetadata &metadata : table.value()) {
    inferred.push_back(InferOne(metadata, &changed));
  }
  if (!changed) {
    return func;
  }
  return WithAttr(std::move(func), kBufferMetadataTableAttr,
                  std::move(inferred));
}

} // namespace

tvm::transform::Pass InferTenstorrentTensorLayout() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    return InferLayout(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.InferTenstorrentTensorLayout", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.tenstorrent.transform.InferTenstorrentTensorLayout",
                        InferTenstorrentTensorLayout);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

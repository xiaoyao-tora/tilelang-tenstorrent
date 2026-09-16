/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file tenstorrent/ir/device_ir.h
 * \brief Typed metadata for the Tenstorrent Device TIR contract.
 */
#ifndef TVM_TL_TENSTORRENT_IR_DEVICE_IR_H_
#define TVM_TL_TENSTORRENT_IR_DEVICE_IR_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/source_map.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {
namespace tenstorrent {

/*! \brief Shared Lower capability registry for persistent GEMM dtype triples.
 */
TVM_DLL bool IsSupportedAccumulatorDTypeTriple(DataType input,
                                               DataType accumulation,
                                               DataType output);

constexpr const char *kDeviceIRVersionAttr = "tt.device_ir_version";
constexpr const char *kTargetArchAttr = "tt.target_arch";
constexpr const char *kLaunchGridAttr = "tt.launch_grid";
constexpr const char *kOperationIdentityAttr = "tt.operation_identity";
constexpr const char *kTensorTableAttr = "tt.tensor_table";
constexpr const char *kAccumulatorTableAttr = "tt.accumulator_table";
constexpr const char *kComputeRequirementsAttr = "tt.compute_requirements";
constexpr const char *kDFBTableAttr = "tt.dfb_table";
constexpr const char *kPipeTableAttr = "tt.pipe_table";
constexpr const char *kPipeTransferTableAttr = "tt.pipe_transfer_table";
constexpr const char *kKernelOrderAttr = "tt.kernel_order";

// Schema v3 retains immutable DFB descriptors and adds explicit storage reuse.
// Keys are canonical decimal DFB IDs, with stable value-based string equality.
// Groups: Map<String, Integer>, relation: Map<String, Array<Integer>> with
// [zero-based iteration ordinal, ordinal % min(stages, extent)].
constexpr const char *kDFBStorageGroupsAttr = "tt.dfb_storage_groups";
constexpr const char *kPipelineRelationsAttr = "tt.pipeline_relations";
constexpr const char *kPipelineStagesAttr = "tt.pipeline_stages";
constexpr const char *kPipelineExtentAttr = "tt.pipeline_extent";
// Optional caller-supplied DFB payload budget, not total physical L1 capacity.
constexpr const char *kL1CapacityBytesAttr = "tt.l1_capacity_bytes";

constexpr const char *kKernelSlotAttr = "tt.kernel_slot";
constexpr const char *kKernelThreadAttr = "tt.kernel_thread";
constexpr const char *kNocIndexAttr = "tt.noc_index";
constexpr const char *kLogicalKernelAttr = "tt.logical_kernel";
constexpr const char *kTensorArgIndicesAttr = "tt.tensor_arg_indices";
constexpr const char *kCoreDomainAttr = "tt.core_domain";

// Typed intermediate metadata consumed by FormTenstorrentDeviceProgram.
constexpr const char *kBufferMetadataTableAttr = "tt.buffer_metadata_table";

class CoreCoordNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  int64_t x;
  int64_t y;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.CoreCoord", CoreCoordNode,
                                    ffi::Object);
};

class CoreCoord : public ffi::ObjectRef {
public:
  TVM_DLL CoreCoord(int64_t x, int64_t y);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(CoreCoord, ffi::ObjectRef,
                                             CoreCoordNode);
};

/*! \brief Half-open rectangular core domain `[begin, end)`. */
class CoreDomainNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  CoreCoord begin;
  CoreCoord end;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.CoreDomain", CoreDomainNode,
                                    ffi::Object);
};

class CoreDomain : public ffi::ObjectRef {
public:
  TVM_DLL CoreDomain(CoreCoord begin, CoreCoord end);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(CoreDomain, ffi::ObjectRef,
                                             CoreDomainNode);
};

class ShardSpecNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  CoreDomain core_domain;
  ffi::Array<PrimExpr> shard_shape;
  ffi::String orientation;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.ShardSpec", ShardSpecNode,
                                    ffi::Object);
};

class ShardSpec : public ffi::ObjectRef {
public:
  TVM_DLL ShardSpec(CoreDomain core_domain, ffi::Array<PrimExpr> shard_shape,
                    ffi::String orientation);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ShardSpec, ffi::ObjectRef,
                                             ShardSpecNode);
};

class TensorBackingNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  int64_t global_arg_index;
  PrimExpr byte_offset;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.TensorBacking",
                                    TensorBackingNode, ffi::Object);
};

class TensorBacking : public ffi::ObjectRef {
public:
  TVM_DLL TensorBacking(int64_t global_arg_index, PrimExpr byte_offset);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TensorBacking, ffi::ObjectRef,
                                             TensorBackingNode);
};

class OperationIdentityNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  ffi::String operation_id;
  Span source_span;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.OperationIdentity",
                                    OperationIdentityNode, ffi::Object);
};

class OperationIdentity : public ffi::ObjectRef {
public:
  TVM_DLL OperationIdentity(ffi::String operation_id, Span source_span);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(OperationIdentity, ffi::ObjectRef,
                                             OperationIdentityNode);
};

class TensorDescriptorNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  int64_t global_arg_index;
  ffi::Array<PrimExpr> shape;
  DataType dtype;
  ffi::Array<PrimExpr> strides;
  ffi::Array<PrimExpr> tile_shape;
  ffi::Array<PrimExpr> tile_grid_shape;
  ffi::String memory_space;
  ffi::String memory_layout;
  ffi::Optional<ShardSpec> shard_spec;
  ffi::String effect;
  int64_t alias_group;
  Span source_span;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.TensorDescriptor",
                                    TensorDescriptorNode, ffi::Object);
};

class TensorDescriptor : public ffi::ObjectRef {
public:
  TVM_DLL TensorDescriptor(int64_t global_arg_index, ffi::Array<PrimExpr> shape,
                           DataType dtype, ffi::Array<PrimExpr> strides,
                           ffi::Array<PrimExpr> tile_shape,
                           ffi::Array<PrimExpr> tile_grid_shape,
                           ffi::String memory_space, ffi::String memory_layout,
                           ffi::Optional<ShardSpec> shard_spec,
                           ffi::String effect, int64_t alias_group,
                           Span source_span);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TensorDescriptor, ffi::ObjectRef,
                                             TensorDescriptorNode);
};

class DFBDescriptorNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  int64_t dfb_id;
  ffi::String source_buffer_identity;
  DataType element_dtype;
  ffi::Array<PrimExpr> tile_shape;
  ffi::Array<PrimExpr> block_shape_in_tiles;
  PrimExpr block_count;
  ffi::Optional<TensorBacking> tensor_backing;
  ffi::String producer_slot;
  CoreDomain producer_domain;
  ffi::String consumer_slot;
  CoreDomain consumer_domain;
  PrimExpr transaction_count_or_loop_relation;
  Span source_span;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.DFBDescriptor",
                                    DFBDescriptorNode, ffi::Object);
};

class DFBDescriptor : public ffi::ObjectRef {
public:
  TVM_DLL DFBDescriptor(int64_t dfb_id, ffi::String source_buffer_identity,
                        DataType element_dtype, ffi::Array<PrimExpr> tile_shape,
                        ffi::Array<PrimExpr> block_shape_in_tiles,
                        PrimExpr block_count,
                        ffi::Optional<TensorBacking> tensor_backing,
                        ffi::String producer_slot, CoreDomain producer_domain,
                        ffi::String consumer_slot, CoreDomain consumer_domain,
                        PrimExpr transaction_count_or_loop_relation,
                        Span source_span);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DFBDescriptor, ffi::ObjectRef,
                                             DFBDescriptorNode);
};

/*! \brief Schema v5 persistent fragment, separate from published DFB values. */
class AccumulatorDescriptorNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  int64_t accumulator_id;
  tirx::BufferRegion accumulator_region;
  DataType input_dtype;
  DataType accumulation_dtype;
  DataType output_dtype;
  int64_t full_k_tiles;
  Span source_span;
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.AccumulatorDescriptor",
                                    AccumulatorDescriptorNode, ffi::Object);
};
class AccumulatorDescriptor : public ffi::ObjectRef {
public:
  TVM_DLL AccumulatorDescriptor(int64_t accumulator_id,
                                tirx::BufferRegion accumulator_region,
                                DataType input_dtype,
                                DataType accumulation_dtype,
                                DataType output_dtype, int64_t full_k_tiles,
                                Span source_span);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AccumulatorDescriptor,
                                             ffi::ObjectRef,
                                             AccumulatorDescriptorNode);
};

/*! \brief Hard requirements merged across one actual compute kernel. */
class ComputeRequirementsNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  ffi::String destination_width;
  ffi::String matmul_full_fp32;
  ffi::Array<AccumulatorDescriptor> accumulators;
  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.ComputeRequirements",
                                    ComputeRequirementsNode, ffi::Object);
};
class ComputeRequirements : public ffi::ObjectRef {
public:
  TVM_DLL ComputeRequirements(ffi::String destination_width,
                              ffi::String matmul_full_fp32,
                              ffi::Array<AccumulatorDescriptor> accumulators);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ComputeRequirements,
                                             ffi::ObjectRef,
                                             ComputeRequirementsNode);
};

class PipeDescriptorNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  int64_t pipe_net_id;
  int64_t event_index;
  CoreCoord src_coord;
  CoreCoord dst_begin;
  CoreCoord dst_end;
  ffi::String contract;
  int64_t payload_dfb_id;
  Span source_span;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.PipeDescriptor",
                                    PipeDescriptorNode, ffi::Object);
};

class PipeDescriptor : public ffi::ObjectRef {
public:
  TVM_DLL PipeDescriptor(int64_t pipe_net_id, int64_t event_index,
                         CoreCoord src_coord, CoreCoord dst_begin,
                         CoreCoord dst_end, ffi::String contract,
                         int64_t payload_dfb_id, Span source_span);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PipeDescriptor, ffi::ObjectRef,
                                             PipeDescriptorNode);
};

/*! \brief Schema v4 point delivery of one immutable PipeNet record payload.
 *
 * A collective record has one entry per destination, in x/y order. Record
 * identity is never deduplicated by endpoint coordinates. Version 4 currently
 * requires occurrence=0 and transaction_count=1; repeated dynamic occurrences
 * need a future scheduling contract. Existing descriptor layouts are unchanged.
 */
class PipeTransferDescriptorNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  int64_t transfer_id;
  int64_t pipe_net_id;
  int64_t record_index;
  int64_t occurrence;
  CoreCoord src_coord;
  CoreCoord dst_coord;
  int64_t source_dfb_id;
  int64_t destination_dfb_id;
  int64_t transaction_count;
  Span source_span;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.PipeTransferDescriptor",
                                    PipeTransferDescriptorNode, ffi::Object);
};

class PipeTransferDescriptor : public ffi::ObjectRef {
public:
  TVM_DLL PipeTransferDescriptor(int64_t transfer_id, int64_t pipe_net_id,
                                 int64_t record_index, int64_t occurrence,
                                 CoreCoord src_coord, CoreCoord dst_coord,
                                 int64_t source_dfb_id,
                                 int64_t destination_dfb_id,
                                 int64_t transaction_count, Span source_span);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PipeTransferDescriptor,
                                             ffi::ObjectRef,
                                             PipeTransferDescriptorNode);
};

class LogicalKernelNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  ffi::String kernel_id;
  ffi::String kind;
  ffi::String role;
  Span source_span;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.LogicalKernel",
                                    LogicalKernelNode, ffi::Object);
};

class LogicalKernel : public ffi::ObjectRef {
public:
  TVM_DLL LogicalKernel(ffi::String kernel_id, ffi::String kind,
                        ffi::String role, Span source_span);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LogicalKernel, ffi::ObjectRef,
                                             LogicalKernelNode);
};

/*! \brief Typed intermediate metadata attached to one frontend Buffer. */
class TTBufferMetadataNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  ffi::String buffer_id;
  tirx::Buffer buffer;
  ffi::String kind;
  ffi::Optional<Integer> global_arg_index;
  ffi::Array<PrimExpr> tile_shape;
  ffi::Array<PrimExpr> tile_grid_shape;
  ffi::String memory_layout;
  ffi::Optional<ShardSpec> shard_spec;
  ffi::Optional<PrimExpr> dfb_block_count;
  ffi::Optional<TensorBacking> tensor_backing;
  ffi::Optional<ffi::String> alias_of;
  ffi::String tile_shape_origin;
  ffi::String block_count_origin;
  ffi::String tensor_backing_origin;
  ffi::String layout_origin;
  Span source_span;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.TTBufferMetadata",
                                    TTBufferMetadataNode, ffi::Object);
};

class TTBufferMetadata : public ffi::ObjectRef {
public:
  TVM_DLL TTBufferMetadata(
      ffi::String buffer_id, tirx::Buffer buffer, ffi::String kind,
      ffi::Optional<Integer> global_arg_index, ffi::Array<PrimExpr> tile_shape,
      ffi::Array<PrimExpr> tile_grid_shape, ffi::String memory_layout,
      ffi::Optional<ShardSpec> shard_spec,
      ffi::Optional<PrimExpr> dfb_block_count,
      ffi::Optional<TensorBacking> tensor_backing,
      ffi::Optional<ffi::String> alias_of, ffi::String tile_shape_origin,
      ffi::String block_count_origin, ffi::String tensor_backing_origin,
      ffi::String layout_origin, Span source_span);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTBufferMetadata, ffi::ObjectRef,
                                             TTBufferMetadataNode);
};

class DeviceModuleMetadataNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  int64_t device_ir_version;
  ffi::String target_arch;
  CoreCoord launch_grid;
  OperationIdentity operation_identity;
  ffi::Array<TensorDescriptor> tensor_table;
  ffi::Array<DFBDescriptor> dfb_table;
  ffi::Array<PipeDescriptor> pipe_table;
  ffi::Array<ffi::String> kernel_order;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.DeviceModuleMetadata",
                                    DeviceModuleMetadataNode, ffi::Object);
};

class DeviceModuleMetadata : public ffi::ObjectRef {
public:
  TVM_DLL DeviceModuleMetadata(int64_t device_ir_version,
                               ffi::String target_arch, CoreCoord launch_grid,
                               OperationIdentity operation_identity,
                               ffi::Array<TensorDescriptor> tensor_table,
                               ffi::Array<DFBDescriptor> dfb_table,
                               ffi::Array<PipeDescriptor> pipe_table,
                               ffi::Array<ffi::String> kernel_order);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DeviceModuleMetadata,
                                             ffi::ObjectRef,
                                             DeviceModuleMetadataNode);
};

class DeviceFunctionMetadataNode : public ffi::Object {
public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;
  ffi::String kernel_slot;
  ffi::String kernel_thread;
  ffi::Optional<Integer> noc_index;
  LogicalKernel logical_kernel;
  ffi::Array<Integer> tensor_arg_indices;
  CoreDomain core_domain;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.tenstorrent.DeviceFunctionMetadata",
                                    DeviceFunctionMetadataNode, ffi::Object);
};

class DeviceFunctionMetadata : public ffi::ObjectRef {
public:
  TVM_DLL DeviceFunctionMetadata(ffi::String kernel_slot,
                                 ffi::String kernel_thread,
                                 ffi::Optional<Integer> noc_index,
                                 LogicalKernel logical_kernel,
                                 ffi::Array<Integer> tensor_arg_indices,
                                 CoreDomain core_domain);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DeviceFunctionMetadata,
                                             ffi::ObjectRef,
                                             DeviceFunctionMetadataNode);
};

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TENSTORRENT_IR_DEVICE_IR_H_

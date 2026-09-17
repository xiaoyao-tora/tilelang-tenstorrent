/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file tenstorrent/ir/device_ir.cc
 * \brief Typed Tenstorrent Device TIR metadata definitions.
 */
#include "device_ir.h"

#include <tvm/ffi/reflection/registry.h>

#include <utility>

namespace tvm {
namespace tl {
namespace tenstorrent {

bool IsSupportedAccumulatorDTypeTriple(DataType input, DataType accumulation,
                                       DataType output) {
  return (input == DataType::BFloat(16) && output == DataType::BFloat(16) &&
          (accumulation == DataType::BFloat(16) ||
           accumulation == DataType::Float(32))) ||
         (input == DataType::Float(32) && accumulation == DataType::Float(32) &&
          output == DataType::Float(32));
}

AccumulatorDescriptor::AccumulatorDescriptor(
    int64_t accumulator_id, tirx::BufferRegion accumulator_region,
    DataType input_dtype, DataType accumulation_dtype, DataType output_dtype,
    int64_t full_k_tiles, Span source_span) {
  auto node = ffi::make_object<AccumulatorDescriptorNode>();
  node->accumulator_id = accumulator_id;
  node->accumulator_region = std::move(accumulator_region);
  node->input_dtype = input_dtype;
  node->accumulation_dtype = accumulation_dtype;
  node->output_dtype = output_dtype;
  node->full_k_tiles = full_k_tiles;
  node->source_span = std::move(source_span);
  data_ = std::move(node);
}
void AccumulatorDescriptorNode::RegisterReflection() {
  ffi::reflection::ObjectDef<AccumulatorDescriptorNode>()
      .def_ro("accumulator_id", &AccumulatorDescriptorNode::accumulator_id)
      .def_ro("accumulator_region",
              &AccumulatorDescriptorNode::accumulator_region)
      .def_ro("input_dtype", &AccumulatorDescriptorNode::input_dtype)
      .def_ro("accumulation_dtype",
              &AccumulatorDescriptorNode::accumulation_dtype)
      .def_ro("output_dtype", &AccumulatorDescriptorNode::output_dtype)
      .def_ro("full_k_tiles", &AccumulatorDescriptorNode::full_k_tiles)
      .def_ro("source_span", &AccumulatorDescriptorNode::source_span);
}
ComputeValueDescriptor::ComputeValueDescriptor(
    int64_t value_id, tirx::Buffer buffer, int64_t version,
    int64_t previous_value_id, int64_t accumulator_id, Span source_span) {
  auto node = ffi::make_object<ComputeValueDescriptorNode>();
  node->value_id = value_id;
  node->buffer = std::move(buffer);
  node->version = version;
  node->previous_value_id = previous_value_id;
  node->accumulator_id = accumulator_id;
  node->source_span = std::move(source_span);
  data_ = std::move(node);
}
void ComputeValueDescriptorNode::RegisterReflection() {
  ffi::reflection::ObjectDef<ComputeValueDescriptorNode>()
      .def_ro("value_id", &ComputeValueDescriptorNode::value_id)
      .def_ro("buffer", &ComputeValueDescriptorNode::buffer)
      .def_ro("version", &ComputeValueDescriptorNode::version)
      .def_ro("previous_value_id",
              &ComputeValueDescriptorNode::previous_value_id)
      .def_ro("accumulator_id", &ComputeValueDescriptorNode::accumulator_id)
      .def_ro("source_span", &ComputeValueDescriptorNode::source_span);
}
ComputeRequirements::ComputeRequirements(
    ffi::String destination_width, ffi::String matmul_full_fp32,
    ffi::Array<AccumulatorDescriptor> accumulators) {
  auto node = ffi::make_object<ComputeRequirementsNode>();
  node->destination_width = std::move(destination_width);
  node->matmul_full_fp32 = std::move(matmul_full_fp32);
  node->accumulators = std::move(accumulators);
  data_ = std::move(node);
}
void ComputeRequirementsNode::RegisterReflection() {
  ffi::reflection::ObjectDef<ComputeRequirementsNode>()
      .def_ro("destination_width", &ComputeRequirementsNode::destination_width)
      .def_ro("matmul_full_fp32", &ComputeRequirementsNode::matmul_full_fp32)
      .def_ro("accumulators", &ComputeRequirementsNode::accumulators);
}

CoreCoord::CoreCoord(int64_t x, int64_t y) {
  ffi::ObjectPtr<CoreCoordNode> node = ffi::make_object<CoreCoordNode>();
  node->x = x;
  node->y = y;
  data_ = std::move(node);
}

void CoreCoordNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<CoreCoordNode>()
      .def_ro("x", &CoreCoordNode::x)
      .def_ro("y", &CoreCoordNode::y);
}

CoreDomain::CoreDomain(CoreCoord begin, CoreCoord end) {
  ffi::ObjectPtr<CoreDomainNode> node = ffi::make_object<CoreDomainNode>();
  node->begin = std::move(begin);
  node->end = std::move(end);
  data_ = std::move(node);
}

void CoreDomainNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<CoreDomainNode>()
      .def_ro("begin", &CoreDomainNode::begin)
      .def_ro("end", &CoreDomainNode::end);
}

ShardSpec::ShardSpec(CoreDomain core_domain, ffi::Array<PrimExpr> shard_shape,
                     ffi::String orientation) {
  ffi::ObjectPtr<ShardSpecNode> node = ffi::make_object<ShardSpecNode>();
  node->core_domain = std::move(core_domain);
  node->shard_shape = std::move(shard_shape);
  node->orientation = std::move(orientation);
  data_ = std::move(node);
}

void ShardSpecNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<ShardSpecNode>()
      .def_ro("core_domain", &ShardSpecNode::core_domain)
      .def_ro("shard_shape", &ShardSpecNode::shard_shape)
      .def_ro("orientation", &ShardSpecNode::orientation);
}

TensorBacking::TensorBacking(int64_t global_arg_index, PrimExpr byte_offset) {
  ffi::ObjectPtr<TensorBackingNode> node =
      ffi::make_object<TensorBackingNode>();
  node->global_arg_index = global_arg_index;
  node->byte_offset = std::move(byte_offset);
  data_ = std::move(node);
}

void TensorBackingNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<TensorBackingNode>()
      .def_ro("global_arg_index", &TensorBackingNode::global_arg_index)
      .def_ro("byte_offset", &TensorBackingNode::byte_offset);
}

OperationIdentity::OperationIdentity(ffi::String operation_id,
                                     Span source_span) {
  ffi::ObjectPtr<OperationIdentityNode> node =
      ffi::make_object<OperationIdentityNode>();
  node->operation_id = std::move(operation_id);
  node->source_span = std::move(source_span);
  data_ = std::move(node);
}

void OperationIdentityNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<OperationIdentityNode>()
      .def_ro("operation_id", &OperationIdentityNode::operation_id)
      .def_ro("source_span", &OperationIdentityNode::source_span);
}

TensorDescriptor::TensorDescriptor(
    int64_t global_arg_index, ffi::Array<PrimExpr> shape, DataType dtype,
    ffi::Array<PrimExpr> strides, ffi::Array<PrimExpr> tile_shape,
    ffi::Array<PrimExpr> tile_grid_shape, ffi::String memory_space,
    ffi::String memory_layout, ffi::Optional<ShardSpec> shard_spec,
    ffi::String effect, int64_t alias_group, Span source_span) {
  ffi::ObjectPtr<TensorDescriptorNode> node =
      ffi::make_object<TensorDescriptorNode>();
  node->global_arg_index = global_arg_index;
  node->shape = std::move(shape);
  node->dtype = dtype;
  node->strides = std::move(strides);
  node->tile_shape = std::move(tile_shape);
  node->tile_grid_shape = std::move(tile_grid_shape);
  node->memory_space = std::move(memory_space);
  node->memory_layout = std::move(memory_layout);
  node->shard_spec = std::move(shard_spec);
  node->effect = std::move(effect);
  node->alias_group = alias_group;
  node->source_span = std::move(source_span);
  data_ = std::move(node);
}

void TensorDescriptorNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<TensorDescriptorNode>()
      .def_ro("global_arg_index", &TensorDescriptorNode::global_arg_index)
      .def_ro("shape", &TensorDescriptorNode::shape)
      .def_ro("dtype", &TensorDescriptorNode::dtype)
      .def_ro("strides", &TensorDescriptorNode::strides)
      .def_ro("tile_shape", &TensorDescriptorNode::tile_shape)
      .def_ro("tile_grid_shape", &TensorDescriptorNode::tile_grid_shape)
      .def_ro("memory_space", &TensorDescriptorNode::memory_space)
      .def_ro("memory_layout", &TensorDescriptorNode::memory_layout)
      .def_ro("shard_spec", &TensorDescriptorNode::shard_spec)
      .def_ro("effect", &TensorDescriptorNode::effect)
      .def_ro("alias_group", &TensorDescriptorNode::alias_group)
      .def_ro("source_span", &TensorDescriptorNode::source_span);
}

DFBDescriptor::DFBDescriptor(
    int64_t dfb_id, ffi::String source_buffer_identity, DataType element_dtype,
    ffi::Array<PrimExpr> tile_shape, ffi::Array<PrimExpr> block_shape_in_tiles,
    PrimExpr block_count, ffi::Optional<TensorBacking> tensor_backing,
    ffi::String producer_slot, CoreDomain producer_domain,
    ffi::String consumer_slot, CoreDomain consumer_domain,
    PrimExpr transaction_count_or_loop_relation, Span source_span) {
  ffi::ObjectPtr<DFBDescriptorNode> node =
      ffi::make_object<DFBDescriptorNode>();
  node->dfb_id = dfb_id;
  node->source_buffer_identity = std::move(source_buffer_identity);
  node->element_dtype = element_dtype;
  node->tile_shape = std::move(tile_shape);
  node->block_shape_in_tiles = std::move(block_shape_in_tiles);
  node->block_count = std::move(block_count);
  node->tensor_backing = std::move(tensor_backing);
  node->producer_slot = std::move(producer_slot);
  node->producer_domain = std::move(producer_domain);
  node->consumer_slot = std::move(consumer_slot);
  node->consumer_domain = std::move(consumer_domain);
  node->transaction_count_or_loop_relation =
      std::move(transaction_count_or_loop_relation);
  node->source_span = std::move(source_span);
  data_ = std::move(node);
}

void DFBDescriptorNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<DFBDescriptorNode>()
      .def_ro("dfb_id", &DFBDescriptorNode::dfb_id)
      .def_ro("source_buffer_identity",
              &DFBDescriptorNode::source_buffer_identity)
      .def_ro("element_dtype", &DFBDescriptorNode::element_dtype)
      .def_ro("tile_shape", &DFBDescriptorNode::tile_shape)
      .def_ro("block_shape_in_tiles", &DFBDescriptorNode::block_shape_in_tiles)
      .def_ro("block_count", &DFBDescriptorNode::block_count)
      .def_ro("tensor_backing", &DFBDescriptorNode::tensor_backing)
      .def_ro("producer_slot", &DFBDescriptorNode::producer_slot)
      .def_ro("producer_domain", &DFBDescriptorNode::producer_domain)
      .def_ro("consumer_slot", &DFBDescriptorNode::consumer_slot)
      .def_ro("consumer_domain", &DFBDescriptorNode::consumer_domain)
      .def_ro("transaction_count_or_loop_relation",
              &DFBDescriptorNode::transaction_count_or_loop_relation)
      .def_ro("source_span", &DFBDescriptorNode::source_span);
}

PipeDescriptor::PipeDescriptor(int64_t pipe_net_id, int64_t event_index,
                               CoreCoord src_coord, CoreCoord dst_begin,
                               CoreCoord dst_end, ffi::String contract,
                               int64_t payload_dfb_id, Span source_span) {
  ffi::ObjectPtr<PipeDescriptorNode> node =
      ffi::make_object<PipeDescriptorNode>();
  node->pipe_net_id = pipe_net_id;
  node->event_index = event_index;
  node->src_coord = std::move(src_coord);
  node->dst_begin = std::move(dst_begin);
  node->dst_end = std::move(dst_end);
  node->contract = std::move(contract);
  node->payload_dfb_id = payload_dfb_id;
  node->source_span = std::move(source_span);
  data_ = std::move(node);
}

void PipeDescriptorNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<PipeDescriptorNode>()
      .def_ro("pipe_net_id", &PipeDescriptorNode::pipe_net_id)
      .def_ro("event_index", &PipeDescriptorNode::event_index)
      .def_ro("src_coord", &PipeDescriptorNode::src_coord)
      .def_ro("dst_begin", &PipeDescriptorNode::dst_begin)
      .def_ro("dst_end", &PipeDescriptorNode::dst_end)
      .def_ro("contract", &PipeDescriptorNode::contract)
      .def_ro("payload_dfb_id", &PipeDescriptorNode::payload_dfb_id)
      .def_ro("source_span", &PipeDescriptorNode::source_span);
}

PipeTransferDescriptor::PipeTransferDescriptor(
    int64_t transfer_id, int64_t pipe_net_id, int64_t record_index,
    int64_t occurrence, CoreCoord src_coord, CoreCoord dst_coord,
    int64_t source_dfb_id, int64_t destination_dfb_id,
    int64_t transaction_count, Span source_span) {
  auto node = ffi::make_object<PipeTransferDescriptorNode>();
  node->transfer_id = transfer_id;
  node->pipe_net_id = pipe_net_id;
  node->record_index = record_index;
  node->occurrence = occurrence;
  node->src_coord = std::move(src_coord);
  node->dst_coord = std::move(dst_coord);
  node->source_dfb_id = source_dfb_id;
  node->destination_dfb_id = destination_dfb_id;
  node->transaction_count = transaction_count;
  node->source_span = std::move(source_span);
  data_ = std::move(node);
}

void PipeTransferDescriptorNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<PipeTransferDescriptorNode>()
      .def_ro("transfer_id", &PipeTransferDescriptorNode::transfer_id)
      .def_ro("pipe_net_id", &PipeTransferDescriptorNode::pipe_net_id)
      .def_ro("record_index", &PipeTransferDescriptorNode::record_index)
      .def_ro("occurrence", &PipeTransferDescriptorNode::occurrence)
      .def_ro("src_coord", &PipeTransferDescriptorNode::src_coord)
      .def_ro("dst_coord", &PipeTransferDescriptorNode::dst_coord)
      .def_ro("source_dfb_id", &PipeTransferDescriptorNode::source_dfb_id)
      .def_ro("destination_dfb_id",
              &PipeTransferDescriptorNode::destination_dfb_id)
      .def_ro("transaction_count",
              &PipeTransferDescriptorNode::transaction_count)
      .def_ro("source_span", &PipeTransferDescriptorNode::source_span);
}

LogicalKernel::LogicalKernel(ffi::String kernel_id, ffi::String kind,
                             ffi::String role, Span source_span) {
  ffi::ObjectPtr<LogicalKernelNode> node =
      ffi::make_object<LogicalKernelNode>();
  node->kernel_id = std::move(kernel_id);
  node->kind = std::move(kind);
  node->role = std::move(role);
  node->source_span = std::move(source_span);
  data_ = std::move(node);
}

void LogicalKernelNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<LogicalKernelNode>()
      .def_ro("kernel_id", &LogicalKernelNode::kernel_id)
      .def_ro("kind", &LogicalKernelNode::kind)
      .def_ro("role", &LogicalKernelNode::role)
      .def_ro("source_span", &LogicalKernelNode::source_span);
}

TTBufferMetadata::TTBufferMetadata(
    ffi::String buffer_id, tirx::Buffer buffer, ffi::String kind,
    ffi::Optional<Integer> global_arg_index, ffi::Array<PrimExpr> tile_shape,
    ffi::Array<PrimExpr> tile_grid_shape, ffi::String memory_layout,
    ffi::Optional<ShardSpec> shard_spec,
    ffi::Optional<PrimExpr> dfb_block_count,
    ffi::Optional<TensorBacking> tensor_backing,
    ffi::Optional<ffi::String> alias_of, ffi::String tile_shape_origin,
    ffi::String block_count_origin, ffi::String tensor_backing_origin,
    ffi::String layout_origin, Span source_span) {
  ffi::ObjectPtr<TTBufferMetadataNode> node =
      ffi::make_object<TTBufferMetadataNode>();
  node->buffer_id = std::move(buffer_id);
  node->buffer = std::move(buffer);
  node->kind = std::move(kind);
  node->global_arg_index = std::move(global_arg_index);
  node->tile_shape = std::move(tile_shape);
  node->tile_grid_shape = std::move(tile_grid_shape);
  node->memory_layout = std::move(memory_layout);
  node->shard_spec = std::move(shard_spec);
  node->dfb_block_count = std::move(dfb_block_count);
  node->tensor_backing = std::move(tensor_backing);
  node->alias_of = std::move(alias_of);
  node->tile_shape_origin = std::move(tile_shape_origin);
  node->block_count_origin = std::move(block_count_origin);
  node->tensor_backing_origin = std::move(tensor_backing_origin);
  node->layout_origin = std::move(layout_origin);
  node->source_span = std::move(source_span);
  data_ = std::move(node);
}

void TTBufferMetadataNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<TTBufferMetadataNode>()
      .def_ro("buffer_id", &TTBufferMetadataNode::buffer_id)
      .def_ro("buffer", &TTBufferMetadataNode::buffer)
      .def_ro("kind", &TTBufferMetadataNode::kind)
      .def_ro("global_arg_index", &TTBufferMetadataNode::global_arg_index)
      .def_ro("tile_shape", &TTBufferMetadataNode::tile_shape)
      .def_ro("tile_grid_shape", &TTBufferMetadataNode::tile_grid_shape)
      .def_ro("memory_layout", &TTBufferMetadataNode::memory_layout)
      .def_ro("shard_spec", &TTBufferMetadataNode::shard_spec)
      .def_ro("dfb_block_count", &TTBufferMetadataNode::dfb_block_count)
      .def_ro("tensor_backing", &TTBufferMetadataNode::tensor_backing)
      .def_ro("alias_of", &TTBufferMetadataNode::alias_of)
      .def_ro("tile_shape_origin", &TTBufferMetadataNode::tile_shape_origin)
      .def_ro("block_count_origin", &TTBufferMetadataNode::block_count_origin)
      .def_ro("tensor_backing_origin",
              &TTBufferMetadataNode::tensor_backing_origin)
      .def_ro("layout_origin", &TTBufferMetadataNode::layout_origin)
      .def_ro("source_span", &TTBufferMetadataNode::source_span);
}

DeviceModuleMetadata::DeviceModuleMetadata(
    int64_t device_ir_version, ffi::String target_arch, CoreCoord launch_grid,
    OperationIdentity operation_identity,
    ffi::Array<TensorDescriptor> tensor_table,
    ffi::Array<DFBDescriptor> dfb_table, ffi::Array<PipeDescriptor> pipe_table,
    ffi::Array<ffi::String> kernel_order) {
  ffi::ObjectPtr<DeviceModuleMetadataNode> node =
      ffi::make_object<DeviceModuleMetadataNode>();
  node->device_ir_version = device_ir_version;
  node->target_arch = std::move(target_arch);
  node->launch_grid = std::move(launch_grid);
  node->operation_identity = std::move(operation_identity);
  node->tensor_table = std::move(tensor_table);
  node->dfb_table = std::move(dfb_table);
  node->pipe_table = std::move(pipe_table);
  node->kernel_order = std::move(kernel_order);
  data_ = std::move(node);
}

void DeviceModuleMetadataNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<DeviceModuleMetadataNode>()
      .def_ro("device_ir_version", &DeviceModuleMetadataNode::device_ir_version)
      .def_ro("target_arch", &DeviceModuleMetadataNode::target_arch)
      .def_ro("launch_grid", &DeviceModuleMetadataNode::launch_grid)
      .def_ro("operation_identity",
              &DeviceModuleMetadataNode::operation_identity)
      .def_ro("tensor_table", &DeviceModuleMetadataNode::tensor_table)
      .def_ro("dfb_table", &DeviceModuleMetadataNode::dfb_table)
      .def_ro("pipe_table", &DeviceModuleMetadataNode::pipe_table)
      .def_ro("kernel_order", &DeviceModuleMetadataNode::kernel_order);
}

DeviceFunctionMetadata::DeviceFunctionMetadata(
    ffi::String kernel_slot, ffi::String kernel_thread,
    ffi::Optional<Integer> noc_index, LogicalKernel logical_kernel,
    ffi::Array<Integer> tensor_arg_indices, CoreDomain core_domain) {
  ffi::ObjectPtr<DeviceFunctionMetadataNode> node =
      ffi::make_object<DeviceFunctionMetadataNode>();
  node->kernel_slot = std::move(kernel_slot);
  node->kernel_thread = std::move(kernel_thread);
  node->noc_index = std::move(noc_index);
  node->logical_kernel = std::move(logical_kernel);
  node->tensor_arg_indices = std::move(tensor_arg_indices);
  node->core_domain = std::move(core_domain);
  data_ = std::move(node);
}

void DeviceFunctionMetadataNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<DeviceFunctionMetadataNode>()
      .def_ro("kernel_slot", &DeviceFunctionMetadataNode::kernel_slot)
      .def_ro("kernel_thread", &DeviceFunctionMetadataNode::kernel_thread)
      .def_ro("noc_index", &DeviceFunctionMetadataNode::noc_index)
      .def_ro("logical_kernel", &DeviceFunctionMetadataNode::logical_kernel)
      .def_ro("tensor_arg_indices",
              &DeviceFunctionMetadataNode::tensor_arg_indices)
      .def_ro("core_domain", &DeviceFunctionMetadataNode::core_domain);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  AccumulatorDescriptorNode::RegisterReflection();
  ComputeValueDescriptorNode::RegisterReflection();
  ComputeRequirementsNode::RegisterReflection();
  CoreCoordNode::RegisterReflection();
  CoreDomainNode::RegisterReflection();
  ShardSpecNode::RegisterReflection();
  TensorBackingNode::RegisterReflection();
  OperationIdentityNode::RegisterReflection();
  TensorDescriptorNode::RegisterReflection();
  DFBDescriptorNode::RegisterReflection();
  PipeDescriptorNode::RegisterReflection();
  PipeTransferDescriptorNode::RegisterReflection();
  LogicalKernelNode::RegisterReflection();
  TTBufferMetadataNode::RegisterReflection();
  DeviceModuleMetadataNode::RegisterReflection();
  DeviceFunctionMetadataNode::RegisterReflection();

  namespace refl = ffi::reflection;
  refl::GlobalDef()
      .def("tl.tenstorrent.CoreCoord",
           [](int64_t x, int64_t y) { return CoreCoord(x, y); })
      .def("tl.tenstorrent.CoreDomain",
           [](CoreCoord begin, CoreCoord end) {
             return CoreDomain(std::move(begin), std::move(end));
           })
      .def("tl.tenstorrent.ShardSpec",
           [](CoreDomain core_domain, ffi::Array<PrimExpr> shard_shape,
              ffi::String orientation) {
             return ShardSpec(std::move(core_domain), std::move(shard_shape),
                              std::move(orientation));
           })
      .def("tl.tenstorrent.TensorBacking",
           [](int64_t global_arg_index, PrimExpr byte_offset) {
             return TensorBacking(global_arg_index, std::move(byte_offset));
           })
      .def("tl.tenstorrent.OperationIdentity",
           [](ffi::String operation_id, Span source_span) {
             return OperationIdentity(std::move(operation_id),
                                      std::move(source_span));
           })
      .def("tl.tenstorrent.TensorDescriptor",
           [](int64_t global_arg_index, ffi::Array<PrimExpr> shape,
              DataType dtype, ffi::Array<PrimExpr> strides,
              ffi::Array<PrimExpr> tile_shape,
              ffi::Array<PrimExpr> tile_grid_shape, ffi::String memory_space,
              ffi::String memory_layout, ffi::Optional<ShardSpec> shard_spec,
              ffi::String effect, int64_t alias_group, Span source_span) {
             return TensorDescriptor(
                 global_arg_index, std::move(shape), dtype, std::move(strides),
                 std::move(tile_shape), std::move(tile_grid_shape),
                 std::move(memory_space), std::move(memory_layout),
                 std::move(shard_spec), std::move(effect), alias_group,
                 std::move(source_span));
           })
      .def("tl.tenstorrent.DFBDescriptor",
           [](int64_t dfb_id, ffi::String source_buffer_identity,
              DataType element_dtype, ffi::Array<PrimExpr> tile_shape,
              ffi::Array<PrimExpr> block_shape_in_tiles, PrimExpr block_count,
              ffi::Optional<TensorBacking> tensor_backing,
              ffi::String producer_slot, CoreDomain producer_domain,
              ffi::String consumer_slot, CoreDomain consumer_domain,
              PrimExpr transaction_count_or_loop_relation, Span source_span) {
             return DFBDescriptor(
                 dfb_id, std::move(source_buffer_identity), element_dtype,
                 std::move(tile_shape), std::move(block_shape_in_tiles),
                 std::move(block_count), std::move(tensor_backing),
                 std::move(producer_slot), std::move(producer_domain),
                 std::move(consumer_slot), std::move(consumer_domain),
                 std::move(transaction_count_or_loop_relation),
                 std::move(source_span));
           })
      .def("tl.tenstorrent.IsSupportedAccumulatorDTypeTriple",
           IsSupportedAccumulatorDTypeTriple)
      .def("tl.tenstorrent.AccumulatorDescriptor",
           [](int64_t id, tirx::BufferRegion region, DataType input,
              DataType accumulation, DataType output, int64_t full_k,
              Span span) {
             return AccumulatorDescriptor(id, std::move(region), input,
                                          accumulation, output, full_k,
                                          std::move(span));
           })
      .def("tl.tenstorrent.ComputeValueDescriptor",
           [](int64_t id, tirx::Buffer buffer, int64_t version,
              int64_t previous, int64_t accumulator, Span span) {
             return ComputeValueDescriptor(id, std::move(buffer), version,
                                           previous, accumulator,
                                           std::move(span));
           })
      .def("tl.tenstorrent.ComputeRequirements",
           [](ffi::String width, ffi::String full,
              ffi::Array<AccumulatorDescriptor> accumulators) {
             return ComputeRequirements(std::move(width), std::move(full),
                                        std::move(accumulators));
           })
      .def("tl.tenstorrent.PipeDescriptor",
           [](int64_t pipe_net_id, int64_t event_index, CoreCoord src_coord,
              CoreCoord dst_begin, CoreCoord dst_end, ffi::String contract,
              int64_t payload_dfb_id, Span source_span) {
             return PipeDescriptor(pipe_net_id, event_index,
                                   std::move(src_coord), std::move(dst_begin),
                                   std::move(dst_end), std::move(contract),
                                   payload_dfb_id, std::move(source_span));
           })
      .def("tl.tenstorrent.PipeTransferDescriptor",
           [](int64_t transfer_id, int64_t pipe_net_id, int64_t record_index,
              int64_t occurrence, CoreCoord src_coord, CoreCoord dst_coord,
              int64_t source_dfb_id, int64_t destination_dfb_id,
              int64_t transaction_count, Span source_span) {
             return PipeTransferDescriptor(
                 transfer_id, pipe_net_id, record_index, occurrence,
                 std::move(src_coord), std::move(dst_coord), source_dfb_id,
                 destination_dfb_id, transaction_count, std::move(source_span));
           })
      .def("tl.tenstorrent.LogicalKernel",
           [](ffi::String kernel_id, ffi::String kind, ffi::String role,
              Span source_span) {
             return LogicalKernel(std::move(kernel_id), std::move(kind),
                                  std::move(role), std::move(source_span));
           })
      .def("tl.tenstorrent.TTBufferMetadata",
           [](ffi::String buffer_id, tirx::Buffer buffer, ffi::String kind,
              ffi::Optional<Integer> global_arg_index,
              ffi::Array<PrimExpr> tile_shape,
              ffi::Array<PrimExpr> tile_grid_shape, ffi::String memory_layout,
              ffi::Optional<ShardSpec> shard_spec,
              ffi::Optional<PrimExpr> dfb_block_count,
              ffi::Optional<TensorBacking> tensor_backing,
              ffi::Optional<ffi::String> alias_of,
              ffi::String tile_shape_origin, ffi::String block_count_origin,
              ffi::String tensor_backing_origin, ffi::String layout_origin,
              Span source_span) {
             return TTBufferMetadata(
                 std::move(buffer_id), std::move(buffer), std::move(kind),
                 std::move(global_arg_index), std::move(tile_shape),
                 std::move(tile_grid_shape), std::move(memory_layout),
                 std::move(shard_spec), std::move(dfb_block_count),
                 std::move(tensor_backing), std::move(alias_of),
                 std::move(tile_shape_origin), std::move(block_count_origin),
                 std::move(tensor_backing_origin), std::move(layout_origin),
                 std::move(source_span));
           })
      .def("tl.tenstorrent.DeviceModuleMetadata",
           [](int64_t device_ir_version, ffi::String target_arch,
              CoreCoord launch_grid, OperationIdentity operation_identity,
              ffi::Array<TensorDescriptor> tensor_table,
              ffi::Array<DFBDescriptor> dfb_table,
              ffi::Array<PipeDescriptor> pipe_table,
              ffi::Array<ffi::String> kernel_order) {
             return DeviceModuleMetadata(
                 device_ir_version, std::move(target_arch),
                 std::move(launch_grid), std::move(operation_identity),
                 std::move(tensor_table), std::move(dfb_table),
                 std::move(pipe_table), std::move(kernel_order));
           })
      .def("tl.tenstorrent.DeviceFunctionMetadata",
           [](ffi::String kernel_slot, ffi::String kernel_thread,
              ffi::Optional<Integer> noc_index, LogicalKernel logical_kernel,
              ffi::Array<Integer> tensor_arg_indices, CoreDomain core_domain) {
             return DeviceFunctionMetadata(
                 std::move(kernel_slot), std::move(kernel_thread),
                 std::move(noc_index), std::move(logical_kernel),
                 std::move(tensor_arg_indices), std::move(core_domain));
           });
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

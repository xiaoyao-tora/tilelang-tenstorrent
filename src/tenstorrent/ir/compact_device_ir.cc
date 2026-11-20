/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#include "device_ir.h"

#include <utility>

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {
namespace tenstorrent {

DeviceIRIntColumn::DeviceIRIntColumn(int64_t start, int64_t step, int64_t count,
                                     ffi::Array<int64_t> values,
                                     int64_t period) {
  auto node = ffi::make_object<DeviceIRIntColumnNode>();
  node->start = start;
  node->step = step;
  node->count = count;
  node->period = period;
  node->values = std::move(values);
  data_ = std::move(node);
}

void DeviceIRIntColumnNode::RegisterReflection() {
  ffi::reflection::ObjectDef<DeviceIRIntColumnNode>()
      .def_ro("start", &DeviceIRIntColumnNode::start)
      .def_ro("step", &DeviceIRIntColumnNode::step)
      .def_ro("count", &DeviceIRIntColumnNode::count)
      .def_ro("period", &DeviceIRIntColumnNode::period)
      .def_ro("values", &DeviceIRIntColumnNode::values);
}

DeviceIRDescriptorFamily::DeviceIRDescriptorFamily(
    ffi::ObjectRef prototype, DeviceIRIntColumn positions,
    ffi::Map<ffi::String, DeviceIRIntColumn> fields,
    ffi::Array<ffi::String> source_parts) {
  auto node = ffi::make_object<DeviceIRDescriptorFamilyNode>();
  node->prototype = std::move(prototype);
  node->positions = std::move(positions);
  node->fields = std::move(fields);
  node->source_parts = std::move(source_parts);
  data_ = std::move(node);
}

void DeviceIRDescriptorFamilyNode::RegisterReflection() {
  ffi::reflection::ObjectDef<DeviceIRDescriptorFamilyNode>()
      .def_ro("prototype", &DeviceIRDescriptorFamilyNode::prototype)
      .def_ro("positions", &DeviceIRDescriptorFamilyNode::positions)
      .def_ro("fields", &DeviceIRDescriptorFamilyNode::fields)
      .def_ro("source_parts", &DeviceIRDescriptorFamilyNode::source_parts);
}

DFBStorageGroupRelation::DFBStorageGroupRelation(
    int64_t dfb_family_id, DeviceIRIntColumn storage_group_ids,
    DeviceIRIntColumn transaction_ordinals,
    DeviceIRIntColumn pipeline_iterations) {
  auto node = ffi::make_object<DFBStorageGroupRelationNode>();
  node->dfb_family_id = dfb_family_id;
  node->storage_group_ids = std::move(storage_group_ids);
  node->transaction_ordinals = std::move(transaction_ordinals);
  node->pipeline_iterations = std::move(pipeline_iterations);
  data_ = std::move(node);
}

void DFBStorageGroupRelationNode::RegisterReflection() {
  ffi::reflection::ObjectDef<DFBStorageGroupRelationNode>()
      .def_ro("dfb_family_id", &DFBStorageGroupRelationNode::dfb_family_id)
      .def_ro("storage_group_ids",
              &DFBStorageGroupRelationNode::storage_group_ids)
      .def_ro("transaction_ordinals",
              &DFBStorageGroupRelationNode::transaction_ordinals)
      .def_ro("pipeline_iterations",
              &DFBStorageGroupRelationNode::pipeline_iterations);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  DeviceIRIntColumnNode::RegisterReflection();
  DeviceIRDescriptorFamilyNode::RegisterReflection();
  DFBStorageGroupRelationNode::RegisterReflection();
  ffi::reflection::GlobalDef()
      .def("tl.tenstorrent.DeviceIRIntColumn",
           [](int64_t start, int64_t step, int64_t count,
              ffi::Array<int64_t> values, int64_t period) {
             return DeviceIRIntColumn(start, step, count, std::move(values),
                                      period);
           })
      .def("tl.tenstorrent.DeviceIRDescriptorFamily",
           [](ffi::ObjectRef prototype, DeviceIRIntColumn positions,
              ffi::Map<ffi::String, DeviceIRIntColumn> fields,
              ffi::Array<ffi::String> source_parts) {
             return DeviceIRDescriptorFamily(
                 std::move(prototype), std::move(positions), std::move(fields),
                 std::move(source_parts));
           })
      .def("tl.tenstorrent.DFBStorageGroupRelation",
           [](int64_t dfb_family_id, DeviceIRIntColumn storage_group_ids,
              DeviceIRIntColumn transaction_ordinals,
              DeviceIRIntColumn pipeline_iterations) {
             return DFBStorageGroupRelation(
                 dfb_family_id, std::move(storage_group_ids),
                 std::move(transaction_ordinals),
                 std::move(pipeline_iterations));
           });
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

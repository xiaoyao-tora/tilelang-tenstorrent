/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#include "compact_device_program.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>

#include "../ir/device_ir.h"
#include "device_metadata_interner.h"
#include "device_statement_compaction.h"
#include "device_table_compaction.h"

namespace tvm {
namespace tl {
namespace tenstorrent {
namespace {

constexpr int64_t kExpansionBudget = 4000000;

void Check(bool valid, const char *message) {
  if (!valid)
    TVM_FFI_THROW(ValueError) << "[TenstorrentCompactDeviceIR] " << message;
}

bool ConsumeStatementBudget(const tirx::Stmt &body, int64_t *budget) {
  if (body.as<tirx::EvaluateNode>()) {
    if (*budget == 0)
      return false;
    --*budget;
    return true;
  }
  auto sequence = body.as<tirx::SeqStmt>();
  if (!sequence.has_value() || sequence.value()->seq.empty())
    return false;
  for (const tirx::Stmt &statement : sequence.value()->seq)
    if (!ConsumeStatementBudget(statement, budget))
      return false;
  return true;
}

bool FitsExpansionBudget(const IRModule &mod, size_t resources) {
  __int128 total = resources;
  total +=
      mod->GetAttr<ffi::Array<AccumulatorDescriptor>>(kAccumulatorTableAttr)
          .value_or(ffi::Array<AccumulatorDescriptor>())
          .size();
  total +=
      mod->GetAttr<ffi::Array<PipeTransferDescriptor>>(kPipeTransferTableAttr)
          .value_or(ffi::Array<PipeTransferDescriptor>())
          .size();
  if (total > kExpansionBudget)
    return false;
  int64_t budget = kExpansionBudget - static_cast<int64_t>(total);
  for (const auto &[global, base] : mod->functions) {
    auto function = base.as<tirx::PrimFunc>();
    if (!function.has_value() ||
        !ConsumeStatementBudget(function.value()->body, &budget))
      return false;
  }
  return true;
}

bool HasCompactMetadata(const IRModule &mod) {
  return mod->attrs.defined() &&
         (mod->attrs->dict.count(kCompactOriginalVersionAttr) ||
          mod->attrs->dict.count(kCompactDFBFamiliesAttr) ||
          mod->attrs->dict.count(kCompactValueFamiliesAttr));
}

bool HasControlFlow(const IRModule &mod) {
  bool found = false;
  for (const auto &[global, base] : mod->functions) {
    auto function = base.as<tirx::PrimFunc>();
    if (!function.has_value())
      return true;
    tirx::PostOrderVisit(
        function.value()->body, [&](const ffi::ObjectRef &obj) {
          found |= obj.as<tirx::ForNode>() || obj.as<tirx::IfThenElseNode>() ||
                   obj.as<tirx::WhileNode>();
        });
  }
  return found;
}

} // namespace

IRModule CompactDeviceProgram(const IRModule &mod, bool force) {
  int64_t version =
      mod->GetAttr<Integer>(kDeviceIRVersionAttr).value_or(Integer(0))->value;
  if (version == 9)
    return mod;
  Check(!HasCompactMetadata(mod), "compact metadata requires Device IR v9");
  if ((version != 7 && version != 8) || HasControlFlow(mod))
    return mod;
  size_t resources =
      mod->GetAttr<ffi::Array<DFBDescriptor>>(kDFBTableAttr)
          .value_or(ffi::Array<DFBDescriptor>())
          .size() +
      mod->GetAttr<ffi::Array<ComputeValueDescriptor>>(kComputeValueTableAttr)
          .value_or(ffi::Array<ComputeValueDescriptor>())
          .size();
  if (!force && resources < 4096)
    return mod;

  // Compaction is optional: do not make an otherwise supported expanded
  // module unrepresentable by selecting a bounded compact representation.
  if (!FitsExpansionBudget(mod, resources)) {
    Check(!force, "module exceeds compact expansion budget");
    return mod;
  }
  IRModule result = CompactDeviceTables(mod);
  if (!DeviceTablesFitExpansionBudget(result)) {
    Check(!force, "metadata exceeds compact expansion budget");
    return mod;
  }
  size_t families = result
                        ->GetAttr<ffi::Array<DeviceIRDescriptorFamily>>(
                            kCompactDFBFamiliesAttr)
                        .value_or(ffi::Array<DeviceIRDescriptorFamily>())
                        .size() +
                    result
                        ->GetAttr<ffi::Array<DeviceIRDescriptorFamily>>(
                            kCompactValueFamiliesAttr)
                        .value_or(ffi::Array<DeviceIRDescriptorFamily>())
                        .size();
  if (!force && families * 4 >= resources)
    return mod;
  result.CopyOnWrite();
  for (const auto &[global, base] : mod->functions) {
    tirx::PrimFunc func = Downcast<tirx::PrimFunc>(base);
    func.CopyOnWrite()->body = CompactDeviceStatements(func->body);
    result->Update(global, func);
  }
  result = WithAttr(result, kCompactOriginalVersionAttr,
                    mod->attrs->dict.at(kDeviceIRVersionAttr));
  result = WithAttr(result, kDeviceIRVersionAttr, Integer(9));
  return DeviceMetadataInterner::Rewrite(result);
}

IRModule ExpandDeviceProgram(const IRModule &mod) {
  int64_t version =
      mod->GetAttr<Integer>(kDeviceIRVersionAttr).value_or(Integer(0))->value;
  if (version != 9) {
    Check(!HasCompactMetadata(mod), "compact metadata requires Device IR v9");
    return mod;
  }
  auto original = mod->GetAttr<Integer>(kCompactOriginalVersionAttr);
  Check(original.has_value() &&
            (original.value()->value == 7 || original.value()->value == 8),
        "v9 requires original version 7 or 8");
  Check(mod->attrs->dict.count(kCompactDFBFamiliesAttr) &&
            mod->attrs->dict.count(kCompactValueFamiliesAttr),
        "v9 requires DFB and compute-value families");
  int64_t budget = kExpansionBudget;
  for (const char *key : {kCompactDFBFamiliesAttr, kCompactValueFamiliesAttr}) {
    for (const auto &family :
         mod->GetAttr<ffi::Array<DeviceIRDescriptorFamily>>(key).value()) {
      Check(family.defined() && family->positions.defined(),
            "undefined descriptor family");
      int64_t count = family->positions->count;
      Check(count >= 0 && count <= budget,
            "descriptor expansion exceeds budget");
      budget -= count;
    }
  }
  size_t other_resources =
      mod->GetAttr<ffi::Array<AccumulatorDescriptor>>(kAccumulatorTableAttr)
          .value_or(ffi::Array<AccumulatorDescriptor>())
          .size() +
      mod->GetAttr<ffi::Array<PipeTransferDescriptor>>(kPipeTransferTableAttr)
          .value_or(ffi::Array<PipeTransferDescriptor>())
          .size();
  Check(other_resources <= static_cast<size_t>(budget),
        "descriptor expansion exceeds budget");
  budget -= other_resources;
  IRModule result = ExpandDeviceTables(mod);
  result.CopyOnWrite();
  for (const auto &[global, base] : mod->functions) {
    auto function = base.as<tirx::PrimFunc>();
    Check(function.has_value(), "Device globals must be PrimFuncs");
    tirx::PrimFunc func = function.value();
    func.CopyOnWrite()->body = ExpandDeviceStatements(func->body, &budget);
    result->Update(global, func);
  }
  result = WithAttr(result, kDeviceIRVersionAttr,
                    mod->attrs->dict.at(kCompactOriginalVersionAttr));
  return WithoutAttr(result, kCompactOriginalVersionAttr);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.tenstorrent.CompactDeviceIR", CompactDeviceProgram)
      .def("tl.tenstorrent.ExpandDeviceIR", ExpandDeviceProgram);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

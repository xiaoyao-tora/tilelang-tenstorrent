/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#include "device_table_compaction.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/ffi/extra/structural_hash.h>

namespace tvm {
namespace tl {
namespace tenstorrent {
namespace {

constexpr int64_t kExpansionBudget = 4000000;
// Bound metadata amplification independently of the number of logical rows.
// Both limits cover the whole module, before allocating any expanded table.
constexpr int64_t kMetadataByteBudget = 256LL * 1024 * 1024;
constexpr int64_t kDecodedIntegerBudget = kMetadataByteBudget / sizeof(int64_t);
constexpr const char *kDFBFamilies = "tt.compact_dfb_families";
constexpr const char *kValueFamilies = "tt.compact_value_families";

void Check(bool valid, const std::string &message) {
  if (!valid)
    TVM_FFI_THROW(ValueError) << "[TenstorrentDeviceTables] " << message;
}

bool FitsInt64(__int128 value) {
  return value >= std::numeric_limits<int64_t>::min() &&
         value <= std::numeric_limits<int64_t>::max();
}

int64_t CheckedInt64(__int128 value) {
  Check(FitsInt64(value), "integer column arithmetic overflow");
  return static_cast<int64_t>(value);
}

struct SourcePattern {
  ffi::Array<ffi::String> parts;
  std::vector<int64_t> values;
};

SourcePattern SplitSource(const ffi::String &identity) {
  std::string source = identity;
  SourcePattern result;
  size_t begin = 0;
  for (size_t offset = 0; offset < source.size();) {
    if (source[offset] < '0' || source[offset] > '9') {
      ++offset;
      continue;
    }
    size_t end = offset;
    __int128 value = 0;
    while (end < source.size() && source[end] >= '0' && source[end] <= '9') {
      value = value * 10 + (source[end++] - '0');
      if (!FitsInt64(value))
        return {};
    }
    // A noncanonical spelling is retained verbatim in its own family.
    if (end - offset > 1 && source[offset] == '0')
      return {};
    result.parts.push_back(source.substr(begin, offset - begin));
    result.values.push_back(static_cast<int64_t>(value));
    begin = offset = end;
  }
  if (result.values.empty())
    return {};
  result.parts.push_back(source.substr(begin));
  return result;
}

std::string SourcePatternKey(const SourcePattern &pattern) {
  std::string result;
  for (const auto &part : pattern.parts)
    result += std::to_string(part.size()) + ":" + std::string(part);
  return result;
}

struct DFBHash {
  size_t operator()(const DFBDescriptor &descriptor) const {
    return ffi::StructuralHash()(descriptor);
  }
};

struct DFBEqual {
  bool operator()(const DFBDescriptor &left, const DFBDescriptor &right) const {
    // Reuse only identical nested objects. This preserves even spans on
    // otherwise equal expressions, and any symbolic Var identities.
    return left->dfb_id == right->dfb_id &&
           left->source_buffer_identity == right->source_buffer_identity &&
           left->element_dtype == right->element_dtype &&
           left->tile_shape.same_as(right->tile_shape) &&
           left->block_shape_in_tiles.same_as(right->block_shape_in_tiles) &&
           left->block_count.same_as(right->block_count) &&
           left->tensor_backing.same_as(right->tensor_backing) &&
           left->producer_slot == right->producer_slot &&
           left->producer_domain.same_as(right->producer_domain) &&
           left->consumer_slot == right->consumer_slot &&
           left->consumer_domain.same_as(right->consumer_domain) &&
           left->transaction_count_or_loop_relation.same_as(
               right->transaction_count_or_loop_relation) &&
           left->source_span.same_as(right->source_span);
  }
};

struct ValueKey {
  tirx::Buffer buffer;
  Span span;
};

struct ValueHash {
  size_t operator()(const ValueKey &key) const {
    return ffi::ObjectPtrHash()(key.buffer) ^
           (ffi::ObjectPtrHash()(key.span) << 1);
  }
};

struct ValueEqual {
  bool operator()(const ValueKey &left, const ValueKey &right) const {
    return left.buffer.same_as(right.buffer) && left.span.same_as(right.span);
  }
};

struct FamilyBuilder {
  ffi::ObjectRef prototype;
  std::vector<int64_t> positions;
  std::map<std::string, std::vector<int64_t>> fields;
  ffi::Array<ffi::String> source_parts;

  DeviceIRDescriptorFamily Finish() const {
    ffi::Map<ffi::String, DeviceIRIntColumn> columns;
    for (const auto &[name, values] : fields)
      columns.Set(name, MakeDeviceIntColumn(values));
    return DeviceIRDescriptorFamily(prototype, MakeDeviceIntColumn(positions),
                                    columns, source_parts);
  }
};

DFBDescriptor RebuildDFB(const DFBDescriptor &prototype, int64_t identifier,
                         const ffi::String &source) {
  return DFBDescriptor(identifier, source, prototype->element_dtype,
                       prototype->tile_shape, prototype->block_shape_in_tiles,
                       prototype->block_count, prototype->tensor_backing,
                       prototype->producer_slot, prototype->producer_domain,
                       prototype->consumer_slot, prototype->consumer_domain,
                       prototype->transaction_count_or_loop_relation,
                       prototype->source_span);
}

ffi::Array<DeviceIRDescriptorFamily>
CompactDFBs(const ffi::Array<DFBDescriptor> &descriptors) {
  std::vector<FamilyBuilder> builders;
  std::unordered_map<DFBDescriptor, size_t, DFBHash, DFBEqual> groups;
  for (size_t position = 0; position < descriptors.size(); ++position) {
    const DFBDescriptor &descriptor = descriptors[position];
    Check(descriptor.defined(), "undefined DFB descriptor");
    SourcePattern source = SplitSource(descriptor->source_buffer_identity);
    size_t family_index = builders.size();
    bool inserted = true;
    if (!source.parts.empty()) {
      DFBDescriptor key = RebuildDFB(descriptor, 0, SourcePatternKey(source));
      auto found = groups.emplace(key, family_index);
      family_index = found.first->second;
      inserted = found.second;
    }
    if (inserted) {
      FamilyBuilder builder;
      builder.prototype = descriptor;
      builder.source_parts = source.parts;
      builders.push_back(std::move(builder));
    }
    FamilyBuilder &builder = builders[family_index];
    builder.positions.push_back(position);
    builder.fields["dfb_id"].push_back(descriptor->dfb_id);
    for (size_t index = 0; index < source.values.size(); ++index)
      builder.fields["source." + std::to_string(index)].push_back(
          source.values[index]);
  }
  ffi::Array<DeviceIRDescriptorFamily> result;
  for (const auto &builder : builders)
    result.push_back(builder.Finish());
  return result;
}

ffi::Array<DeviceIRDescriptorFamily>
CompactValues(const ffi::Array<ComputeValueDescriptor> &descriptors) {
  std::vector<FamilyBuilder> builders;
  std::unordered_map<ValueKey, size_t, ValueHash, ValueEqual> groups;
  for (size_t position = 0; position < descriptors.size(); ++position) {
    const ComputeValueDescriptor &descriptor = descriptors[position];
    Check(descriptor.defined(), "undefined compute-value descriptor");
    ValueKey key{descriptor->buffer, descriptor->source_span};
    auto [found, inserted] = groups.emplace(key, builders.size());
    if (inserted) {
      FamilyBuilder builder;
      builder.prototype = descriptor;
      builders.push_back(std::move(builder));
    }
    FamilyBuilder &builder = builders[found->second];
    builder.positions.push_back(position);
    builder.fields["value_id"].push_back(descriptor->value_id);
    builder.fields["version"].push_back(descriptor->version);
    builder.fields["previous_value_id"].push_back(
        descriptor->previous_value_id);
    builder.fields["accumulator_id"].push_back(descriptor->accumulator_id);
  }
  ffi::Array<DeviceIRDescriptorFamily> result;
  for (const auto &builder : builders)
    result.push_back(builder.Finish());
  return result;
}

int64_t TableSize(const ffi::Array<DeviceIRDescriptorFamily> &families) {
  int64_t total = 0;
  for (const auto &family : families) {
    Check(family.defined() && family->positions.defined(),
          "undefined descriptor family or positions column");
    int64_t count = family->positions->count;
    Check(count > 0 && count <= kExpansionBudget - total,
          "invalid family size or descriptor expansion budget exceeded");
    total += count;
  }
  return total;
}

void AccumulateMetadataExpansion(
    const ffi::Array<DeviceIRDescriptorFamily> &families, bool source_strings,
    __int128 *integer_count, __int128 *string_bytes) {
  for (const auto &family : families) {
    // TableSize has already validated the family and its positive row count.
    __int128 count = family->positions->count;
    *integer_count +=
        count * (static_cast<__int128>(family->fields.size()) + 1);
    if (!source_strings || family->source_parts.empty())
      continue;
    __int128 bytes_per_row = 0;
    for (const auto &part : family->source_parts)
      bytes_per_row += part.size();
    // Source columns are nonnegative int64 values: each needs at most 19
    // decimal digits. The conservative bound needs no decoded column storage.
    bytes_per_row +=
        19 * (static_cast<__int128>(family->source_parts.size()) - 1);
    *string_bytes += count * bytes_per_row;
  }
}

struct ExpansionSize {
  int64_t dfbs{0};
  int64_t values{0};
  __int128 integers{0};
  __int128 source_bytes{0};

  bool FitsBudget() const {
    return values <= kExpansionBudget - dfbs &&
           integers <= kDecodedIntegerBudget &&
           source_bytes <= kMetadataByteBudget;
  }
};

ExpansionSize MeasureExpansion(const IRModule &mod) {
  ExpansionSize size;
  auto dfbs = mod->GetAttr<ffi::Array<DeviceIRDescriptorFamily>>(kDFBFamilies);
  auto values =
      mod->GetAttr<ffi::Array<DeviceIRDescriptorFamily>>(kValueFamilies);
  if (dfbs.has_value()) {
    size.dfbs = TableSize(dfbs.value());
    AccumulateMetadataExpansion(dfbs.value(), true, &size.integers,
                                &size.source_bytes);
  }
  if (values.has_value()) {
    size.values = TableSize(values.value());
    AccumulateMetadataExpansion(values.value(), false, &size.integers,
                                &size.source_bytes);
  }
  return size;
}

using DecodedFields = std::map<std::string, std::vector<int64_t>>;

DecodedFields ReadFields(const DeviceIRDescriptorFamily &family,
                         const std::set<std::string> &expected) {
  Check(family->fields.size() == expected.size(),
        "descriptor family has missing or unknown fields");
  DecodedFields result;
  for (const auto &[name, column] : family->fields) {
    Check(expected.count(name),
          "unknown descriptor family field: " + std::string(name));
    result.emplace(name,
                   ExpandDeviceIntColumn(column, family->positions->count));
  }
  return result;
}

template <typename Descriptor, typename Restore>
ffi::Array<Descriptor>
ExpandTable(const ffi::Array<DeviceIRDescriptorFamily> &families, int64_t total,
            Restore restore) {
  std::vector<Descriptor> table(total);
  for (const auto &family : families) {
    auto prototype = family->prototype.as<Descriptor>();
    Check(prototype.has_value(), "wrong descriptor family prototype type");
    std::vector<int64_t> positions = ExpandDeviceIntColumn(family->positions);
    std::vector<Descriptor> entries = restore(family, prototype.value());
    for (size_t index = 0; index < positions.size(); ++index) {
      int64_t position = positions[index];
      Check(position >= 0 && position < total,
            "descriptor position is outside the original table");
      Check(!table[position].defined(), "duplicate descriptor table position");
      table[position] = entries[index];
    }
  }
  for (const Descriptor &descriptor : table)
    Check(descriptor.defined(), "missing descriptor table position");
  return ffi::Array<Descriptor>(table.begin(), table.end());
}

std::vector<DFBDescriptor> RestoreDFBs(const DeviceIRDescriptorFamily &family,
                                       const DFBDescriptor &prototype) {
  std::set<std::string> names{"dfb_id"};
  size_t source_columns = 0;
  if (!family->source_parts.empty()) {
    Check(family->source_parts.size() >= 2,
          "source pattern requires at least one integer column");
    source_columns = family->source_parts.size() - 1;
    Check(source_columns <= family->fields.size(),
          "source pattern has missing integer columns");
    for (size_t index = 0; index < family->source_parts.size(); ++index) {
      const ffi::String &part = family->source_parts[index];
      Check(!std::any_of(
                part.data(), part.data() + part.size(),
                [](char value) { return value >= '0' && value <= '9'; }) &&
                (index == 0 || index == source_columns || !part.empty()),
            "source pattern must separate canonical decimal integer fields");
    }
    for (size_t index = 0; index < source_columns; ++index)
      names.insert("source." + std::to_string(index));
  }
  DecodedFields fields = ReadFields(family, names);
  std::vector<DFBDescriptor> result;
  result.reserve(family->positions->count);
  for (int64_t row = 0; row < family->positions->count; ++row) {
    ffi::String identity = prototype->source_buffer_identity;
    if (source_columns) {
      std::string source = family->source_parts[0];
      for (size_t index = 0; index < source_columns; ++index) {
        int64_t value = fields.at("source." + std::to_string(index))[row];
        Check(value >= 0, "source identity integer must be nonnegative");
        source += std::to_string(value) +
                  std::string(family->source_parts[index + 1]);
      }
      identity = source;
    }
    result.push_back(RebuildDFB(prototype, fields.at("dfb_id")[row], identity));
  }
  return result;
}

std::vector<ComputeValueDescriptor>
RestoreValues(const DeviceIRDescriptorFamily &family,
              const ComputeValueDescriptor &prototype) {
  Check(family->source_parts.empty(),
        "compute-value family cannot have a source string pattern");
  DecodedFields fields = ReadFields(
      family, {"value_id", "version", "previous_value_id", "accumulator_id"});
  std::vector<ComputeValueDescriptor> result;
  result.reserve(family->positions->count);
  for (int64_t row = 0; row < family->positions->count; ++row) {
    result.emplace_back(
        fields.at("value_id")[row], prototype->buffer,
        fields.at("version")[row], fields.at("previous_value_id")[row],
        fields.at("accumulator_id")[row], prototype->source_span);
  }
  return result;
}

} // namespace

DeviceIRIntColumn MakeDeviceIntColumn(const std::vector<int64_t> &values) {
  Check(values.size() <= kExpansionBudget,
        "integer column exceeds expansion budget");
  int64_t count = values.size();
  if (count == 0)
    return DeviceIRIntColumn(0, 0, 0, {});
  if (count == 1)
    return DeviceIRIntColumn(values[0], 0, 1, {});
  __int128 step = static_cast<__int128>(values[1]) - values[0];
  bool affine = FitsInt64(step);
  for (int64_t index = 0; affine && index < count; ++index)
    affine = static_cast<__int128>(values[0]) + step * index == values[index];
  if (affine)
    return DeviceIRIntColumn(values[0], static_cast<int64_t>(step), count, {});

  // A periodic first difference gives a repeated value pattern with an affine
  // increment per period. KMP finds the minimum candidate in linear time.
  std::vector<__int128> differences(count - 1);
  for (int64_t index = 1; index < count; ++index)
    differences[index - 1] =
        static_cast<__int128>(values[index]) - values[index - 1];
  std::vector<size_t> prefix(differences.size(), 0);
  for (size_t index = 1; index < differences.size(); ++index) {
    size_t length = prefix[index - 1];
    while (length && differences[index] != differences[length])
      length = prefix[length - 1];
    if (differences[index] == differences[length])
      ++length;
    prefix[index] = length;
  }
  size_t period = differences.size() - prefix.back();
  if (period < values.size() && period * 2 <= values.size()) {
    step = static_cast<__int128>(values[period]) - values[0];
    bool periodic = FitsInt64(step);
    for (size_t index = 0; periodic && index < values.size(); ++index)
      periodic = static_cast<__int128>(values[index % period]) +
                     step * (index / period) ==
                 values[index];
    if (periodic)
      return DeviceIRIntColumn(
          0, static_cast<int64_t>(step), count,
          ffi::Array<int64_t>(values.begin(), values.begin() + period), period);
  }
  return DeviceIRIntColumn(0, 0, count,
                           ffi::Array<int64_t>(values.begin(), values.end()));
}

std::vector<int64_t> ExpandDeviceIntColumn(const DeviceIRIntColumn &column,
                                           int64_t expected_count) {
  Check(column.defined(), "undefined integer column");
  int64_t count = column->count;
  Check(count >= 0 && count <= kExpansionBudget,
        "invalid integer column length or expansion budget exceeded");
  Check(expected_count < 0 || count == expected_count,
        "integer column length differs from family positions");
  Check(column->period >= 0, "integer column period must be nonnegative");
  if (column->period) {
    Check(column->start == 0 && column->period < count &&
              column->values.size() == static_cast<size_t>(column->period),
          "invalid periodic integer column");
  } else if (!column->values.empty()) {
    Check(column->start == 0 && column->step == 0 &&
              column->values.size() == static_cast<size_t>(count),
          "invalid explicit integer column");
  }
  std::vector<int64_t> result;
  result.reserve(count);
  for (int64_t index = 0; index < count; ++index) {
    __int128 value;
    if (column->period)
      value = static_cast<__int128>(column->values[index % column->period]) +
              static_cast<__int128>(column->step) * (index / column->period);
    else if (!column->values.empty())
      value = column->values[index];
    else
      value = static_cast<__int128>(column->start) +
              static_cast<__int128>(column->step) * index;
    result.push_back(CheckedInt64(value));
  }
  return result;
}

IRModule CompactDeviceTables(IRModule mod) {
  Check(!mod->attrs->dict.count(kDFBFamilies) &&
            !mod->attrs->dict.count(kValueFamilies),
        "cannot compact tables that already have a family representation");
  auto dfbs = mod->GetAttr<ffi::Array<DFBDescriptor>>(kDFBTableAttr);
  auto values =
      mod->GetAttr<ffi::Array<ComputeValueDescriptor>>(kComputeValueTableAttr);
  size_t count = (dfbs.has_value() ? dfbs.value().size() : 0) +
                 (values.has_value() ? values.value().size() : 0);
  Check(count <= kExpansionBudget, "descriptor expansion budget exceeded");
  if (dfbs.has_value()) {
    mod = WithAttr(mod, kDFBFamilies, CompactDFBs(dfbs.value()));
    mod = WithoutAttr(mod, kDFBTableAttr);
  }
  if (values.has_value()) {
    mod = WithAttr(mod, kValueFamilies, CompactValues(values.value()));
    mod = WithoutAttr(mod, kComputeValueTableAttr);
  }
  return mod;
}

bool DeviceTablesFitExpansionBudget(const IRModule &mod) {
  return MeasureExpansion(mod).FitsBudget();
}

IRModule ExpandDeviceTables(IRModule mod) {
  auto dfbs = mod->GetAttr<ffi::Array<DeviceIRDescriptorFamily>>(kDFBFamilies);
  auto values =
      mod->GetAttr<ffi::Array<DeviceIRDescriptorFamily>>(kValueFamilies);
  Check(!dfbs.has_value() || !mod->attrs->dict.count(kDFBTableAttr),
        "both DFB table and compact families are present");
  Check(!values.has_value() || !mod->attrs->dict.count(kComputeValueTableAttr),
        "both compute-value table and compact families are present");
  ExpansionSize size = MeasureExpansion(mod);
  Check(size.values <= kExpansionBudget - size.dfbs,
        "combined descriptor expansion budget exceeded");
  Check(size.integers <= kDecodedIntegerBudget,
        "decoded integer column budget exceeds 256 MiB");
  Check(size.source_bytes <= kMetadataByteBudget,
        "expanded source string budget exceeds 256 MiB");
  if (dfbs.has_value()) {
    mod = WithAttr(
        mod, kDFBTableAttr,
        ExpandTable<DFBDescriptor>(dfbs.value(), size.dfbs, RestoreDFBs));
    mod = WithoutAttr(mod, kDFBFamilies);
  }
  if (values.has_value()) {
    mod = WithAttr(mod, kComputeValueTableAttr,
                   ExpandTable<ComputeValueDescriptor>(
                       values.value(), size.values, RestoreValues));
    mod = WithoutAttr(mod, kValueFamilies);
  }
  return mod;
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

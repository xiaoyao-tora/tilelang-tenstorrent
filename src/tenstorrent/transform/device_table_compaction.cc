/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#include "device_table_compaction.h"
#include "../op/builtin.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>

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
constexpr const char *kExpandedAccumulatorLocalIDs =
    "tt.expanded_accumulator_local_ids";
constexpr const char *kExpandedComputeValueLocalIDs =
    "tt.expanded_compute_value_local_ids";

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

struct CurrentStorageGroups {
  ffi::Array<DFBStorageGroupDescriptor> descriptors;
  ffi::Array<DFBStorageGroupRelation> relations;
};

bool SameStorageGroupContract(const DFBStorageGroupDescriptor &group,
                              const DFBDescriptor &dfb) {
  return group->element_dtype == dfb->element_dtype &&
         ffi::StructuralEqual()(group->tile_shape, dfb->tile_shape) &&
         ffi::StructuralEqual()(group->block_shape_in_tiles,
                                dfb->block_shape_in_tiles) &&
         ffi::StructuralEqual()(group->block_count, dfb->block_count) &&
         group->producer_slot == dfb->producer_slot &&
         ffi::StructuralEqual()(group->producer_domain, dfb->producer_domain) &&
         group->consumer_slot == dfb->consumer_slot &&
         ffi::StructuralEqual()(group->consumer_domain, dfb->consumer_domain);
}

CurrentStorageGroups BuildCurrentStorageGroups(
    const ffi::Array<DFBDescriptor> &dfbs,
    const ffi::Array<DeviceIRDescriptorFamily> &families,
    const ffi::Optional<ffi::Map<ffi::String, Integer>> &raw_groups,
    const ffi::Optional<ffi::Map<ffi::String, ffi::Array<Integer>>>
        &raw_relations,
    int64_t pipeline_depth) {
  std::map<int64_t, DFBDescriptor> by_id;
  for (const DFBDescriptor &dfb : dfbs)
    Check(dfb.defined() && by_id.emplace(dfb->dfb_id, dfb).second,
          "DFB IDs must be dense and unique before storage grouping");
  if (raw_groups.has_value())
    Check(raw_groups.value().size() == dfbs.size(),
          "internal storage groups must cover every DFB generation");
  if (raw_relations.has_value())
    Check(raw_relations.value().size() == dfbs.size(),
          "internal transaction relations must cover every DFB generation");
  Check(pipeline_depth == 0 || raw_groups.has_value(),
        "pipelined DFB families require an explicit storage-group plan");
  std::set<std::string> canonical_keys;
  for (const auto &[id, dfb] : by_id)
    canonical_keys.insert(std::to_string(id));
  if (raw_groups.has_value())
    for (const auto &[key, group] : raw_groups.value())
      Check(canonical_keys.count(std::string(key)),
            "storage metadata requires a canonical decimal DFB key");
  if (raw_relations.has_value()) {
    for (const auto &[key, relation] : raw_relations.value()) {
      Check(canonical_keys.count(std::string(key)),
            "storage metadata requires a canonical decimal DFB key");
      Check(relation.size() == 2,
            "internal transaction relation must be [ordinal, stage]");
      int64_t iteration = relation[0]->value;
      int64_t stage = relation[1]->value;
      Check(pipeline_depth > 0 &&
                ((iteration == -1 && stage == 0) ||
                 (iteration >= 0 && stage == iteration % pipeline_depth)),
            "pipeline relation has invalid iteration/stage");
    }
  }

  CurrentStorageGroups result;
  std::map<int64_t, int64_t> canonical_groups;
  std::map<int64_t, std::set<int64_t>> ordinals;
  std::map<int64_t, int64_t> next_raw_ordinal;
  int64_t next_distinct_group = 0;
  for (size_t family_id = 0; family_id < families.size(); ++family_id) {
    const DeviceIRDescriptorFamily &family = families[family_id];
    Check(family.defined() && family->fields.count("dfb_id"),
          "DFB family is missing its dfb_id relation");
    std::vector<int64_t> ids = ExpandDeviceIntColumn(
        family->fields.at("dfb_id"), family->positions->count);
    std::vector<int64_t> group_column;
    std::vector<int64_t> ordinal_column;
    std::vector<int64_t> pipeline_iteration_column;
    group_column.reserve(ids.size());
    ordinal_column.reserve(ids.size());
    pipeline_iteration_column.reserve(ids.size());
    auto prototype = family->prototype.as<DFBDescriptor>();
    Check(prototype.has_value(), "DFB family has the wrong prototype type");
    const auto *capacity_value =
        prototype.value()->block_count.as<IntImmNode>();
    Check(capacity_value && capacity_value->value > 0,
          "DFB family capacity must be a positive static integer");
    // A missing internal plan carries no FIFO/lifetime proof. Keep generations
    // independent instead of inferring logical reuse from capacity alone.
    int64_t group_period = static_cast<int64_t>(ids.size());
    int64_t family_group_base = next_distinct_group;
    next_distinct_group += group_period;
    for (size_t row = 0; row < ids.size(); ++row) {
      int64_t id = ids[row];
      Check(by_id.count(id), "DFB family references a missing generation");
      int64_t raw_group =
          family_group_base + static_cast<int64_t>(row) % group_period;
      int64_t ordinal = static_cast<int64_t>(row) / group_period;
      int64_t pipeline_iteration = -1;
      if (raw_relations.has_value()) {
        ffi::String key = std::to_string(id);
        Check(raw_relations.value().count(key),
              "internal transaction relations have a missing DFB generation");
        ffi::Array<Integer> relation = raw_relations.value().at(key);
        Check(relation.size() == 2,
              "internal transaction relation must be [ordinal, stage]");
        pipeline_iteration = relation[0]->value;
      }
      if (raw_groups.has_value()) {
        ffi::String key = std::to_string(id);
        Check(raw_groups.value().count(key),
              "internal storage groups have a missing DFB generation");
        raw_group = raw_groups.value().at(key)->value;
        ordinal = raw_relations.has_value()
                      ? (pipeline_iteration < 0 ? 0 : pipeline_iteration)
                      : next_raw_ordinal[raw_group]++;
      }
      Check(raw_group >= 0 && ordinal >= 0,
            "storage group IDs and transaction ordinals must be non-negative");
      auto [found, inserted] =
          canonical_groups.emplace(raw_group, canonical_groups.size());
      int64_t group_id = found->second;
      const DFBDescriptor &dfb = by_id.at(id);
      if (inserted) {
        result.descriptors.push_back(DFBStorageGroupDescriptor(
            group_id, dfb->element_dtype, dfb->tile_shape,
            dfb->block_shape_in_tiles, dfb->block_count, dfb->producer_slot,
            dfb->producer_domain, dfb->consumer_slot, dfb->consumer_domain,
            dfb->transaction_count_or_loop_relation, std::nullopt,
            dfb->source_span));
      } else {
        Check(
            SameStorageGroupContract(result.descriptors[group_id], dfb),
            "storage-group generations disagree on capacity, layout, or slots");
      }
      Check(ordinals[group_id].insert(ordinal).second,
            "one logical storage group has a duplicate generation ordinal");
      group_column.push_back(group_id);
      ordinal_column.push_back(ordinal);
      pipeline_iteration_column.push_back(pipeline_iteration);
    }
    result.relations.push_back(DFBStorageGroupRelation(
        family_id, MakeDeviceIntColumn(group_column),
        MakeDeviceIntColumn(ordinal_column),
        MakeDeviceIntColumn(pipeline_iteration_column)));
  }
  Check(result.descriptors.size() == canonical_groups.size() &&
            result.relations.size() == families.size(),
        "incomplete current storage-group construction");
  return result;
}

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

ffi::Array<DeviceIRDescriptorFamily>
LocalizeFamilyPositions(const ffi::Array<DeviceIRDescriptorFamily> &families) {
  ffi::Array<DeviceIRDescriptorFamily> result;
  for (const DeviceIRDescriptorFamily &family : families) {
    std::vector<int64_t> positions(family->positions->count);
    std::iota(positions.begin(), positions.end(), 0);
    result.push_back(DeviceIRDescriptorFamily(
        family->prototype, MakeDeviceIntColumn(positions), family->fields,
        family->source_parts));
  }
  return result;
}

template <typename Descriptor, typename Restore, typename Identifier>
ffi::Array<Descriptor>
ExpandCurrentTable(const ffi::Array<DeviceIRDescriptorFamily> &families,
                   int64_t total, Restore restore, Identifier identifier) {
  std::vector<Descriptor> table;
  table.reserve(total);
  for (const DeviceIRDescriptorFamily &family : families) {
    auto prototype = family->prototype.as<Descriptor>();
    Check(prototype.has_value(), "wrong current family prototype type");
    std::vector<int64_t> positions = ExpandDeviceIntColumn(family->positions);
    for (size_t index = 0; index < positions.size(); ++index)
      Check(positions[index] == static_cast<int64_t>(index),
            "current family positions must be dense local ordinals");
    std::vector<Descriptor> entries = restore(family, prototype.value());
    table.insert(table.end(), entries.begin(), entries.end());
  }
  Check(table.size() == static_cast<size_t>(total),
        "current family expansion size mismatch");
  std::sort(table.begin(), table.end(),
            [&](const Descriptor &a, const Descriptor &b) {
              return identifier(a) < identifier(b);
            });
  for (size_t index = 0; index < table.size(); ++index)
    Check(identifier(table[index]) == static_cast<int64_t>(index),
          "current family identifiers must be dense and unique");
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
    std::string prefix = family->source_parts[0];
    constexpr const char *generation_marker = ".generation";
    size_t marker_size = std::char_traits<char>::length(generation_marker);
    bool generation_only = source_columns == 1 &&
                           prefix.size() >= marker_size &&
                           prefix.compare(prefix.size() - marker_size,
                                          marker_size, generation_marker) == 0;
    for (size_t index = 0; index < family->source_parts.size(); ++index) {
      const ffi::String &part = family->source_parts[index];
      Check((generation_only ||
             !std::any_of(
                 part.data(), part.data() + part.size(),
                 [](char value) { return value >= '0' && value <= '9'; })) &&
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

std::string SpanKey(const Span &span) {
  if (!span.defined() || !span->source_name.defined())
    return "<unknown>";
  return std::string(span->source_name->name) + ":" +
         std::to_string(span->line) + ":" + std::to_string(span->column) + ":" +
         std::to_string(span->end_line) + ":" +
         std::to_string(span->end_column);
}

std::string BufferFamilyKey(const tirx::Buffer &buffer) {
  std::ostringstream key;
  key << buffer->name << '|' << buffer.scope() << '|' << buffer->dtype << '|'
      << ffi::StructuralHash()(buffer->shape);
  return key.str();
}

ffi::Array<DeviceIRDescriptorFamily> CompactCurrentAccumulators(
    const ffi::Array<AccumulatorDescriptor> &descriptors) {
  std::vector<FamilyBuilder> builders;
  std::unordered_map<std::string, size_t> groups;
  for (size_t position = 0; position < descriptors.size(); ++position) {
    const AccumulatorDescriptor &descriptor = descriptors[position];
    Check(descriptor.defined() && descriptor->accumulator_region.defined(),
          "undefined current accumulator descriptor");
    std::ostringstream key;
    key << BufferFamilyKey(descriptor->accumulator_region->buffer) << '|'
        << descriptor->input_dtype << '|' << descriptor->accumulation_dtype
        << '|' << descriptor->output_dtype << '|' << descriptor->full_k_tiles
        << '|' << ffi::StructuralHash()(descriptor->accumulator_region->region)
        << '|' << SpanKey(descriptor->source_span);
    auto [found, inserted] = groups.emplace(key.str(), builders.size());
    if (inserted) {
      FamilyBuilder builder;
      builder.prototype = descriptor;
      builders.push_back(std::move(builder));
    }
    FamilyBuilder &builder = builders[found->second];
    builder.positions.push_back(builder.positions.size());
    builder.fields["accumulator_id"].push_back(descriptor->accumulator_id);
    builder.fields["core_begin_x"].push_back(descriptor->core_domain->begin->x);
    builder.fields["core_begin_y"].push_back(descriptor->core_domain->begin->y);
    builder.fields["core_end_x"].push_back(descriptor->core_domain->end->x);
    builder.fields["core_end_y"].push_back(descriptor->core_domain->end->y);
  }
  ffi::Array<DeviceIRDescriptorFamily> result;
  for (const FamilyBuilder &builder : builders)
    result.push_back(builder.Finish());
  return result;
}

ffi::Array<DeviceIRDescriptorFamily>
CompactCurrentValues(const ffi::Array<ComputeValueDescriptor> &descriptors) {
  std::vector<FamilyBuilder> builders;
  std::unordered_map<std::string, size_t> groups;
  for (size_t position = 0; position < descriptors.size(); ++position) {
    const ComputeValueDescriptor &descriptor = descriptors[position];
    Check(descriptor.defined() && descriptor->buffer.defined(),
          "undefined current compute-value descriptor");
    std::ostringstream key;
    key << BufferFamilyKey(descriptor->buffer) << '|' << descriptor->value_kind
        << '|' << ffi::StructuralHash()(descriptor->logical_shape) << '|'
        << ffi::StructuralHash()(descriptor->compute_tile_shape) << '|'
        << descriptor->dtype << '|' << descriptor->owner_slot << '|'
        << descriptor->materialization_refs.size() << '|'
        << SpanKey(descriptor->source_span);
    auto [found, inserted] = groups.emplace(key.str(), builders.size());
    if (inserted) {
      FamilyBuilder builder;
      builder.prototype = descriptor;
      builders.push_back(std::move(builder));
    }
    FamilyBuilder &builder = builders[found->second];
    builder.positions.push_back(builder.positions.size());
    builder.fields["value_id"].push_back(descriptor->value_id);
    builder.fields["version"].push_back(descriptor->version);
    builder.fields["previous_value_id"].push_back(
        descriptor->previous_value_id);
    builder.fields["accumulator_id"].push_back(descriptor->accumulator_id);
    builder.fields["core_begin_x"].push_back(descriptor->core_domain->begin->x);
    builder.fields["core_begin_y"].push_back(descriptor->core_domain->begin->y);
    builder.fields["core_end_x"].push_back(descriptor->core_domain->end->x);
    builder.fields["core_end_y"].push_back(descriptor->core_domain->end->y);
    for (size_t index = 0; index < descriptor->materialization_refs.size();
         ++index)
      builder.fields["materialization." + std::to_string(index)].push_back(
          descriptor->materialization_refs[index]->value);
  }
  ffi::Array<DeviceIRDescriptorFamily> result;
  for (const FamilyBuilder &builder : builders)
    result.push_back(builder.Finish());
  return result;
}

ffi::Array<DeviceIRDescriptorFamily>
CompactCurrentTransports(const ffi::Array<PipeDescriptor> &descriptors) {
  std::vector<FamilyBuilder> builders;
  std::unordered_map<std::string, size_t> groups;
  for (size_t position = 0; position < descriptors.size(); ++position) {
    const PipeDescriptor &descriptor = descriptors[position];
    Check(descriptor.defined(), "undefined current transport descriptor");
    std::string key = std::string(descriptor->contract) + "|" +
                      SpanKey(descriptor->source_span);
    auto [found, inserted] = groups.emplace(key, builders.size());
    if (inserted) {
      FamilyBuilder builder;
      builder.prototype = descriptor;
      builders.push_back(std::move(builder));
    }
    FamilyBuilder &builder = builders[found->second];
    builder.positions.push_back(position);
    builder.fields["pipe_net_id"].push_back(descriptor->pipe_net_id);
    builder.fields["communication_id"].push_back(descriptor->communication_id);
    builder.fields["record_index"].push_back(descriptor->record_index);
    builder.fields["source_x"].push_back(descriptor->src_coord->x);
    builder.fields["source_y"].push_back(descriptor->src_coord->y);
    builder.fields["destination_begin_x"].push_back(descriptor->dst_begin->x);
    builder.fields["destination_begin_y"].push_back(descriptor->dst_begin->y);
    builder.fields["destination_end_x"].push_back(descriptor->dst_end->x);
    builder.fields["destination_end_y"].push_back(descriptor->dst_end->y);
    builder.fields["payload_dfb_id"].push_back(descriptor->payload_dfb_id);
  }
  ffi::Array<DeviceIRDescriptorFamily> result;
  for (const FamilyBuilder &builder : builders)
    result.push_back(builder.Finish());
  return result;
}

std::vector<AccumulatorDescriptor>
RestoreCurrentAccumulators(const DeviceIRDescriptorFamily &family,
                           const AccumulatorDescriptor &prototype) {
  Check(family->source_parts.empty(),
        "accumulator family cannot have source string parts");
  DecodedFields fields =
      ReadFields(family, {"accumulator_id", "core_begin_x", "core_begin_y",
                          "core_end_x", "core_end_y"});
  std::vector<AccumulatorDescriptor> result;
  result.reserve(family->positions->count);
  for (int64_t row = 0; row < family->positions->count; ++row)
    result.emplace_back(fields.at("accumulator_id")[row],
                        prototype->accumulator_region, prototype->input_dtype,
                        prototype->accumulation_dtype, prototype->output_dtype,
                        prototype->full_k_tiles,
                        CoreDomain(CoreCoord(fields.at("core_begin_x")[row],
                                             fields.at("core_begin_y")[row]),
                                   CoreCoord(fields.at("core_end_x")[row],
                                             fields.at("core_end_y")[row])),
                        prototype->source_span);
  return result;
}

std::vector<ComputeValueDescriptor>
RestoreCurrentValues(const DeviceIRDescriptorFamily &family,
                     const ComputeValueDescriptor &prototype) {
  Check(family->source_parts.empty(),
        "compute-value family cannot have source string parts");
  std::set<std::string> names{
      "value_id",     "version",      "previous_value_id", "accumulator_id",
      "core_begin_x", "core_begin_y", "core_end_x",        "core_end_y"};
  for (size_t index = 0; index < prototype->materialization_refs.size();
       ++index)
    names.insert("materialization." + std::to_string(index));
  DecodedFields fields = ReadFields(family, names);
  std::vector<ComputeValueDescriptor> result;
  result.reserve(family->positions->count);
  for (int64_t row = 0; row < family->positions->count; ++row) {
    ffi::Array<Integer> materializations;
    for (size_t index = 0; index < prototype->materialization_refs.size();
         ++index)
      materializations.push_back(
          Integer(fields.at("materialization." + std::to_string(index))[row]));
    result.emplace_back(fields.at("value_id")[row], prototype->buffer,
                        fields.at("version")[row],
                        fields.at("previous_value_id")[row],
                        fields.at("accumulator_id")[row], prototype->value_kind,
                        prototype->logical_shape, prototype->compute_tile_shape,
                        prototype->dtype, prototype->owner_slot,
                        CoreDomain(CoreCoord(fields.at("core_begin_x")[row],
                                             fields.at("core_begin_y")[row]),
                                   CoreCoord(fields.at("core_end_x")[row],
                                             fields.at("core_end_y")[row])),
                        materializations, prototype->source_span);
  }
  return result;
}

using LocalComputeKey = std::tuple<int64_t, int64_t, int64_t>;

template <typename Descriptor, typename Restore>
std::vector<Descriptor>
RestoreCurrentInstanceRows(const ffi::Array<DeviceIRDescriptorFamily> &families,
                           Restore restore) {
  std::vector<Descriptor> result;
  for (const DeviceIRDescriptorFamily &family : families) {
    auto prototype = family->prototype.as<Descriptor>();
    Check(prototype.has_value(), "wrong current family prototype type");
    std::vector<int64_t> positions = ExpandDeviceIntColumn(family->positions);
    for (size_t index = 0; index < positions.size(); ++index)
      Check(positions[index] == static_cast<int64_t>(index),
            "current family positions must be dense local ordinals");
    std::vector<Descriptor> entries = restore(family, prototype.value());
    result.insert(result.end(), entries.begin(), entries.end());
  }
  return result;
}

LocalComputeKey ComputeInstanceKey(const CoreDomain &domain, int64_t local_id,
                                   const char *label) {
  Check(domain.defined() && domain->end->x == domain->begin->x + 1 &&
            domain->end->y == domain->begin->y + 1,
        std::string(label) + " requires a singleton Core domain");
  return {domain->begin->x, domain->begin->y, local_id};
}

struct CurrentComputeExpansion {
  ffi::Array<AccumulatorDescriptor> accumulators;
  ffi::Array<ComputeValueDescriptor> values;
  ffi::Array<Integer> accumulator_local_ids;
  ffi::Array<Integer> value_local_ids;
};

CurrentComputeExpansion ExpandCurrentComputeInstances(
    const ffi::Optional<ffi::Array<DeviceIRDescriptorFamily>>
        &accumulator_families,
    const ffi::Optional<ffi::Array<DeviceIRDescriptorFamily>> &value_families) {
  CurrentComputeExpansion result;
  std::map<LocalComputeKey, int64_t> accumulator_ids;
  std::vector<AccumulatorDescriptor> local_accumulators;
  if (accumulator_families.has_value())
    local_accumulators = RestoreCurrentInstanceRows<AccumulatorDescriptor>(
        accumulator_families.value(), RestoreCurrentAccumulators);
  std::sort(
      local_accumulators.begin(), local_accumulators.end(),
      [](const AccumulatorDescriptor &left,
         const AccumulatorDescriptor &right) {
        return std::tie(left->core_domain->begin->x,
                        left->core_domain->begin->y, left->accumulator_id) <
               std::tie(right->core_domain->begin->x,
                        right->core_domain->begin->y, right->accumulator_id);
      });
  for (const AccumulatorDescriptor &entry : local_accumulators) {
    int64_t global_id = result.accumulators.size();
    LocalComputeKey key = ComputeInstanceKey(
        entry->core_domain, entry->accumulator_id, "current accumulator");
    Check(accumulator_ids.emplace(key, global_id).second,
          "duplicate Core-local accumulator ID");
    result.accumulator_local_ids.push_back(Integer(entry->accumulator_id));
    result.accumulators.push_back(AccumulatorDescriptor(
        global_id, entry->accumulator_region, entry->input_dtype,
        entry->accumulation_dtype, entry->output_dtype, entry->full_k_tiles,
        entry->core_domain, entry->source_span));
  }

  std::map<LocalComputeKey, int64_t> value_ids;
  std::vector<ComputeValueDescriptor> local_values;
  if (value_families.has_value())
    local_values = RestoreCurrentInstanceRows<ComputeValueDescriptor>(
        value_families.value(), RestoreCurrentValues);
  std::sort(local_values.begin(), local_values.end(),
            [](const ComputeValueDescriptor &left,
               const ComputeValueDescriptor &right) {
              return std::tie(left->core_domain->begin->x,
                              left->core_domain->begin->y, left->value_id) <
                     std::tie(right->core_domain->begin->x,
                              right->core_domain->begin->y, right->value_id);
            });
  for (const ComputeValueDescriptor &entry : local_values) {
    int64_t global_id = value_ids.size();
    LocalComputeKey key = ComputeInstanceKey(
        entry->core_domain, entry->value_id, "current compute value");
    Check(value_ids.emplace(key, global_id).second,
          "duplicate Core-local compute value ID");
  }
  for (const ComputeValueDescriptor &entry : local_values) {
    LocalComputeKey key = ComputeInstanceKey(
        entry->core_domain, entry->value_id, "current compute value");
    int64_t global_id = value_ids.at(key);
    int64_t previous = -1;
    if (entry->previous_value_id >= 0) {
      auto found = value_ids.find(
          ComputeInstanceKey(entry->core_domain, entry->previous_value_id,
                             "compute value predecessor"));
      Check(found != value_ids.end(),
            "compute value predecessor references a missing Core-local ID");
      previous = found->second;
    }
    int64_t accumulator = -1;
    if (entry->accumulator_id >= 0) {
      auto found = accumulator_ids.find(
          ComputeInstanceKey(entry->core_domain, entry->accumulator_id,
                             "compute value accumulator"));
      Check(found != accumulator_ids.end(),
            "compute value references a missing Core-local accumulator ID");
      accumulator = found->second;
    }
    result.value_local_ids.push_back(Integer(entry->value_id));
    result.values.push_back(ComputeValueDescriptor(
        global_id, entry->buffer, entry->version, previous, accumulator,
        entry->value_kind, entry->logical_shape, entry->compute_tile_shape,
        entry->dtype, entry->owner_slot, entry->core_domain,
        entry->materialization_refs, entry->source_span));
  }
  return result;
}

using CoreKey = std::pair<int64_t, int64_t>;

class CurrentComputeIDShifter : public tirx::StmtExprMutator {
public:
  CurrentComputeIDShifter(std::map<CoreKey, int64_t> accumulator_offsets,
                          std::map<CoreKey, int64_t> value_offsets,
                          int64_t direction)
      : accumulator_offsets_(std::move(accumulator_offsets)),
        value_offsets_(std::move(value_offsets)), direction_(direction) {
    Check(direction == 1 || direction == -1,
          "current compute ID shift direction must be +/-1");
  }

  PrimExpr VisitExpr_(const tirx::CallNode *op) final {
    tirx::Call call = Downcast<tirx::Call>(StmtExprMutator::VisitExpr_(op));
    ffi::Array<PrimExpr> args = call->args;
    auto shift = [&](size_t index, const std::map<CoreKey, int64_t> &offsets,
                     const char *label, bool optional = false) {
      Check(index < args.size(), std::string(label) + " operand is missing");
      if (optional) {
        const int64_t *constant = tirx::as_const_int(args[index]);
        if (constant && *constant < 0)
          return;
      }
      args.Set(index, Shift(args[index], offsets));
    };
    if (call->op.same_as(accumulator_init()) ||
        call->op.same_as(accumulator_materialize())) {
      shift(0, accumulator_offsets_, "accumulator");
    } else if (call->op.same_as(gemm_update())) {
      shift(2, accumulator_offsets_, "GEMM accumulator");
    } else if (call->op.same_as(compute_value())) {
      shift(0, value_offsets_, "compute value output");
    } else if (call->op.same_as(compute_value_gemm())) {
      shift(0, value_offsets_, "compute value GEMM output");
      shift(3, value_offsets_, "compute value GEMM predecessor",
            /*optional=*/true);
    } else if (call->op.same_as(compute_value_store()) ||
               call->op.same_as(compute_value_load())) {
      shift(0, value_offsets_, "compute value");
    } else if (call->op.same_as(compute_value_state_init()) ||
               call->op.same_as(compute_value_state_yield())) {
      shift(1, value_offsets_, "loop-state value");
    } else if (call->op.same_as(compute_value_state_input()) ||
               call->op.same_as(compute_value_state_result())) {
      shift(0, value_offsets_, "loop-state value");
    }
    ffi::Map<ffi::String, ffi::ObjectRef> annotations = call->annotations;
    auto value_inputs = annotations.Get("tt.value_inputs");
    if (value_inputs.has_value()) {
      ffi::Array<PrimExpr> shifted;
      for (const PrimExpr &input :
           Downcast<ffi::Array<PrimExpr>>(value_inputs.value()))
        shifted.push_back(Shift(input, value_offsets_));
      annotations.Set("tt.value_inputs", shifted);
    }
    return tirx::Call(call->dtype, call->op, args, annotations, op->span);
  }

private:
  PrimExpr Offset(const std::map<CoreKey, int64_t> &offsets,
                  DataType dtype) const {
    PrimExpr result = tirx::make_const(dtype, 0);
    PrimExpr x = tirx::Call(DataType::Int(32), core_x(), {});
    PrimExpr y = tirx::Call(DataType::Int(32), core_y(), {});
    for (auto it = offsets.rbegin(); it != offsets.rend(); ++it) {
      PrimExpr predicate = x == tirx::make_const(x.dtype(), it->first.first) &&
                           y == tirx::make_const(y.dtype(), it->first.second);
      result = tirx::Select(
          predicate, tirx::make_const(dtype, direction_ * it->second), result);
    }
    return result;
  }

  PrimExpr Shift(const PrimExpr &value,
                 const std::map<CoreKey, int64_t> &offsets) const {
    if (offsets.empty())
      return value;
    return analyzer_.Simplify(value + Offset(offsets, value.dtype()));
  }

  std::map<CoreKey, int64_t> accumulator_offsets_;
  std::map<CoreKey, int64_t> value_offsets_;
  int64_t direction_;
  mutable arith::Analyzer analyzer_;
};

IRModule ShiftCurrentComputeIDs(
    IRModule mod, const ffi::Array<AccumulatorDescriptor> &accumulators,
    const ffi::Array<Integer> &accumulator_local_ids,
    const ffi::Array<ComputeValueDescriptor> &values,
    const ffi::Array<Integer> &value_local_ids, int64_t direction) {
  Check(accumulators.size() == accumulator_local_ids.size(),
        "duplicate accumulator descriptor or expanded local-ID relation has "
        "the wrong size");
  if (values.size() != value_local_ids.size()) {
    std::set<int64_t> ids;
    bool duplicate = false;
    for (const ComputeValueDescriptor &value : values)
      duplicate |= !ids.insert(value->value_id).second;
    Check(false, duplicate ? "duplicate compute value descriptor ID"
                           : "undefined compute value descriptor ID");
  }
  std::map<CoreKey, int64_t> accumulator_offsets;
  std::map<CoreKey, int64_t> value_offsets;
  // The diagnostic tables are dense in their synthetic IDs, so the parallel
  // local-ID arrays carry the source IDs at the same positions.
  for (size_t index = 0; index < accumulators.size(); ++index) {
    const AccumulatorDescriptor &descriptor = accumulators[index];
    CoreKey core{descriptor->core_domain->begin->x,
                 descriptor->core_domain->begin->y};
    int64_t offset =
        descriptor->accumulator_id - accumulator_local_ids[index]->value;
    auto [found, inserted] = accumulator_offsets.emplace(core, offset);
    Check(inserted || found->second == offset,
          "expanded accumulator local-to-diagnostic relation is not affine");
  }
  for (size_t index = 0; index < values.size(); ++index) {
    const ComputeValueDescriptor &descriptor = values[index];
    CoreKey core{descriptor->core_domain->begin->x,
                 descriptor->core_domain->begin->y};
    int64_t offset = descriptor->value_id - value_local_ids[index]->value;
    auto [found, inserted] = value_offsets.emplace(core, offset);
    Check(inserted || found->second == offset,
          "expanded compute-value local-to-diagnostic relation is not affine");
  }
  IRModule result = mod;
  result.CopyOnWrite();
  for (const auto &[global, base] : mod->functions) {
    auto function = base.as<tirx::PrimFunc>();
    if (!function.has_value())
      continue;
    tirx::PrimFunc rewritten = function.value();
    rewritten.CopyOnWrite()->body = CurrentComputeIDShifter(
        accumulator_offsets, value_offsets, direction)(rewritten->body);
    result->Update(global, rewritten);
  }
  return result;
}

std::vector<PipeDescriptor>
RestoreCurrentTransports(const DeviceIRDescriptorFamily &family,
                         const PipeDescriptor &prototype) {
  Check(family->source_parts.empty(),
        "transport family cannot have source string parts");
  DecodedFields fields = ReadFields(
      family, {"pipe_net_id", "communication_id", "record_index", "source_x",
               "source_y", "destination_begin_x", "destination_begin_y",
               "destination_end_x", "destination_end_y", "payload_dfb_id"});
  std::vector<PipeDescriptor> result;
  result.reserve(family->positions->count);
  for (int64_t row = 0; row < family->positions->count; ++row)
    result.emplace_back(
        fields.at("pipe_net_id")[row], fields.at("communication_id")[row],
        fields.at("record_index")[row],
        CoreCoord(fields.at("source_x")[row], fields.at("source_y")[row]),
        CoreCoord(fields.at("destination_begin_x")[row],
                  fields.at("destination_begin_y")[row]),
        CoreCoord(fields.at("destination_end_x")[row],
                  fields.at("destination_end_y")[row]),
        prototype->contract, fields.at("payload_dfb_id")[row],
        prototype->source_span);
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

bool HasCurrentDeviceFamilies(const IRModule &mod) {
  if (!mod->attrs.defined())
    return false;
  for (const char *key : {kDFBFamiliesAttr, kAccumulatorFamiliesAttr,
                          kComputeValueFamiliesAttr, kTransportFamiliesAttr})
    if (mod->attrs->dict.count(key))
      return true;
  return false;
}

IRModule ParameterizeCurrentDeviceTables(IRModule mod) {
  Check(mod->GetAttr<ffi::String>("tt.ir_stage").value_or("") == "device",
        "current table families require tt.ir_stage='device'");
  Check(!HasCurrentDeviceFamilies(mod),
        "current Device tables are already parameterized");
  auto dfbs = mod->GetAttr<ffi::Array<DFBDescriptor>>(kDFBTableAttr);
  auto accumulators =
      mod->GetAttr<ffi::Array<AccumulatorDescriptor>>(kAccumulatorTableAttr);
  auto values =
      mod->GetAttr<ffi::Array<ComputeValueDescriptor>>(kComputeValueTableAttr);
  auto transports =
      mod->GetAttr<ffi::Array<PipeDescriptor>>("tt.transport_table");
  bool stream_operands =
      mod->GetAttr<Integer>("tt.stream_operands").value_or(Integer(0))->value ==
      1;
  Check(dfbs.has_value(), "current Device IR is missing tt.dfb_table");
  Check(transports.has_value(),
        "current Device IR is missing tt.transport_table");
  auto expanded_accumulator_local_ids =
      mod->GetAttr<ffi::Array<Integer>>(kExpandedAccumulatorLocalIDs);
  auto expanded_value_local_ids =
      mod->GetAttr<ffi::Array<Integer>>(kExpandedComputeValueLocalIDs);
  if (expanded_accumulator_local_ids.has_value() ||
      expanded_value_local_ids.has_value())
    mod = ShiftCurrentComputeIDs(
        mod, accumulators.value_or(ffi::Array<AccumulatorDescriptor>()),
        expanded_accumulator_local_ids.value_or(ffi::Array<Integer>()),
        values.value_or(ffi::Array<ComputeValueDescriptor>()),
        expanded_value_local_ids.value_or(ffi::Array<Integer>()), -1);
  if (expanded_accumulator_local_ids.has_value()) {
    Check(accumulators.has_value() &&
              accumulators.value().size() ==
                  expanded_accumulator_local_ids.value().size(),
          "expanded accumulator local-ID relation has the wrong size");
    ffi::Array<AccumulatorDescriptor> localized;
    for (size_t index = 0; index < accumulators.value().size(); ++index) {
      const AccumulatorDescriptor &entry = accumulators.value()[index];
      localized.push_back(AccumulatorDescriptor(
          expanded_accumulator_local_ids.value()[index]->value,
          entry->accumulator_region, entry->input_dtype,
          entry->accumulation_dtype, entry->output_dtype, entry->full_k_tiles,
          entry->core_domain, entry->source_span));
    }
    accumulators = localized;
    mod = WithAttr(mod, kAccumulatorTableAttr, localized);
  }
  if (expanded_value_local_ids.has_value()) {
    Check(values.has_value() &&
              values.value().size() == expanded_value_local_ids.value().size(),
          "duplicate compute value descriptor or expanded local-ID relation "
          "has the wrong size");
    ffi::Array<ComputeValueDescriptor> localized;
    for (size_t index = 0; index < values.value().size(); ++index) {
      const ComputeValueDescriptor &entry = values.value()[index];
      int64_t previous = -1;
      if (entry->previous_value_id >= 0) {
        Check(entry->previous_value_id <
                  static_cast<int64_t>(expanded_value_local_ids.value().size()),
              "compute value predecessor references a missing descriptor");
        previous =
            expanded_value_local_ids.value()[entry->previous_value_id]->value;
      }
      int64_t accumulator = -1;
      if (entry->accumulator_id >= 0) {
        Check(expanded_accumulator_local_ids.has_value() &&
                  entry->accumulator_id <
                      static_cast<int64_t>(
                          expanded_accumulator_local_ids.value().size()),
              "compute value references a missing accumulator descriptor");
        accumulator =
            expanded_accumulator_local_ids.value()[entry->accumulator_id]
                ->value;
      }
      localized.push_back(ComputeValueDescriptor(
          expanded_value_local_ids.value()[index]->value, entry->buffer,
          entry->version, previous, accumulator, entry->value_kind,
          entry->logical_shape, entry->compute_tile_shape, entry->dtype,
          entry->owner_slot, entry->core_domain, entry->materialization_refs,
          entry->source_span));
    }
    values = localized;
    mod = WithAttr(mod, kComputeValueTableAttr, localized);
  }
  if (values.has_value()) {
    std::set<LocalComputeKey> value_ids;
    for (const ComputeValueDescriptor &value : values.value())
      Check(value.defined() && value_ids
                                   .insert(ComputeInstanceKey(
                                       value->core_domain, value->value_id,
                                       "current compute value"))
                                   .second,
            "duplicate Core-local compute value descriptor ID");
  }
  if (accumulators.has_value()) {
    std::set<LocalComputeKey> accumulator_ids;
    for (const AccumulatorDescriptor &accumulator : accumulators.value())
      Check(accumulator.defined() &&
                accumulator_ids
                    .insert(ComputeInstanceKey(accumulator->core_domain,
                                               accumulator->accumulator_id,
                                               "current accumulator"))
                    .second,
            "duplicate Core-local accumulator descriptor ID");
  }
  ffi::Array<DeviceIRDescriptorFamily> dfb_families =
      LocalizeFamilyPositions(CompactDFBs(dfbs.value()));
  bool diagnostic_expansion =
      mod->attrs->dict.count("tt.current_device_table_expansion");
  auto stages = mod->GetAttr<Integer>(kPipelineStagesAttr);
  auto extent = mod->GetAttr<Integer>(kPipelineExtentAttr);
  if (stages.has_value() != extent.has_value())
    Check(false, stages.has_value()
                     ? "missing required Module attr tt.pipeline_extent"
                     : "missing required Module attr tt.pipeline_stages");
  int64_t pipeline_depth = stages.has_value() ? std::min(stages.value()->value,
                                                         extent.value()->value)
                                              : 0;
  ffi::Optional<ffi::Map<ffi::String, Integer>> raw_groups;
  ffi::Optional<ffi::Map<ffi::String, ffi::Array<Integer>>> raw_relations;
  if (!stream_operands) {
    raw_groups = diagnostic_expansion && pipeline_depth > 0
                     ? mod->GetAttr<ffi::Map<ffi::String, Integer>>(
                           kDFBStorageGroupsAttr)
                     : mod->GetAttr<ffi::Map<ffi::String, Integer>>(
                           kInternalDFBStorageGroupsAttr);
    raw_relations =
        diagnostic_expansion && pipeline_depth > 0
            ? mod->GetAttr<ffi::Map<ffi::String, ffi::Array<Integer>>>(
                  kPipelineRelationsAttr)
            : mod->GetAttr<ffi::Map<ffi::String, ffi::Array<Integer>>>(
                  kInternalDFBTransactionRelationsAttr);
  }
  auto planned_streams = mod->GetAttr<ffi::Array<DFBStorageGroupDescriptor>>(
      kDFBStorageGroupsAttr);
  CurrentStorageGroups storage;
  if (stream_operands) {
    Check(planned_streams.has_value(),
          "stream-centric Device IR requires materialized logical streams");
    storage.descriptors = planned_streams.value();
    mod = WithoutAttr(mod, kDFBFamiliesAttr);
    mod = WithoutAttr(mod, kDFBStorageGroupRelationsAttr);
  } else {
    storage = BuildCurrentStorageGroups(dfbs.value(), dfb_families, raw_groups,
                                        raw_relations, pipeline_depth);
    mod = WithAttr(mod, kDFBFamiliesAttr, dfb_families);
    mod = WithAttr(mod, kDFBStorageGroupRelationsAttr, storage.relations);
  }
  mod = WithoutAttr(mod, kDFBTableAttr);
  mod = WithAttr(mod, kDFBStorageGroupsAttr, storage.descriptors);
  if (mod->attrs->dict.count("tt.expanded_storage_group_descriptors"))
    mod = WithoutAttr(mod, "tt.expanded_storage_group_descriptors");
  if (mod->attrs->dict.count("tt.current_device_table_expansion"))
    mod = WithoutAttr(mod, "tt.current_device_table_expansion");
  if (mod->attrs->dict.count("tt.expanded_dfb_transaction_relations"))
    mod = WithoutAttr(mod, "tt.expanded_dfb_transaction_relations");
  if (mod->attrs->dict.count(kExpandedAccumulatorLocalIDs))
    mod = WithoutAttr(mod, kExpandedAccumulatorLocalIDs);
  if (mod->attrs->dict.count(kExpandedComputeValueLocalIDs))
    mod = WithoutAttr(mod, kExpandedComputeValueLocalIDs);
  if (mod->attrs->dict.count(kInternalDFBStorageGroupsAttr))
    mod = WithoutAttr(mod, kInternalDFBStorageGroupsAttr);
  if (mod->attrs->dict.count(kInternalDFBTransactionRelationsAttr))
    mod = WithoutAttr(mod, kInternalDFBTransactionRelationsAttr);
  if (mod->attrs->dict.count(kPipelineRelationsAttr))
    mod = WithoutAttr(mod, kPipelineRelationsAttr);
  mod = WithAttr(mod, kTransportFamiliesAttr,
                 CompactCurrentTransports(transports.value()));
  mod = WithoutAttr(mod, "tt.transport_table");
  if (accumulators.has_value()) {
    mod = WithAttr(mod, kAccumulatorFamiliesAttr,
                   CompactCurrentAccumulators(accumulators.value()));
    mod = WithoutAttr(mod, kAccumulatorTableAttr);
  }
  if (values.has_value()) {
    mod = WithAttr(mod, kComputeValueFamiliesAttr,
                   CompactCurrentValues(values.value()));
    mod = WithoutAttr(mod, kComputeValueTableAttr);
  }
  return mod;
}

IRModule ExpandCurrentDeviceTables(IRModule mod) {
  if (!HasCurrentDeviceFamilies(mod))
    return mod;
  bool stream_operands =
      mod->GetAttr<Integer>("tt.stream_operands").value_or(Integer(0))->value ==
      1;
  auto dfbs =
      mod->GetAttr<ffi::Array<DeviceIRDescriptorFamily>>(kDFBFamiliesAttr);
  auto accumulators = mod->GetAttr<ffi::Array<DeviceIRDescriptorFamily>>(
      kAccumulatorFamiliesAttr);
  auto values = mod->GetAttr<ffi::Array<DeviceIRDescriptorFamily>>(
      kComputeValueFamiliesAttr);
  auto transports = mod->GetAttr<ffi::Array<DeviceIRDescriptorFamily>>(
      kTransportFamiliesAttr);
  auto storage_groups = mod->GetAttr<ffi::Array<DFBStorageGroupDescriptor>>(
      kDFBStorageGroupsAttr);
  auto storage_relations = mod->GetAttr<ffi::Array<DFBStorageGroupRelation>>(
      kDFBStorageGroupRelationsAttr);
  Check(transports.has_value(),
        "current Device family representation requires transport families");
  Check(storage_groups.has_value(),
        "current Device representation requires typed logical streams");
  if (stream_operands) {
    Check(!dfbs.has_value() && !storage_relations.has_value(),
          "stream-centric Device IR cannot persist DFB generation families");
  } else {
    Check(dfbs.has_value() && storage_relations.has_value() &&
              storage_relations.value().size() == dfbs.value().size(),
          "current Device family representation requires DFB families and "
          "storage relations");
  }
  Check(!mod->attrs->dict.count(kDFBTableAttr) &&
            !mod->attrs->dict.count("tt.transport_table") &&
            (!accumulators.has_value() ||
             !mod->attrs->dict.count(kAccumulatorTableAttr)) &&
            (!values.has_value() ||
             !mod->attrs->dict.count(kComputeValueTableAttr)),
        "current Device IR cannot mix a table with its family representation");
  int64_t dfb_count = dfbs.has_value() ? TableSize(dfbs.value()) : 0;
  int64_t accumulator_count =
      accumulators.has_value() ? TableSize(accumulators.value()) : 0;
  int64_t value_count = values.has_value() ? TableSize(values.value()) : 0;
  int64_t transport_count = TableSize(transports.value());
  CurrentComputeExpansion compute =
      ExpandCurrentComputeInstances(accumulators, values);
  Check(compute.accumulators.size() == static_cast<size_t>(accumulator_count) &&
            compute.values.size() == static_cast<size_t>(value_count),
        "current compute family expansion size mismatch");
  Check(dfb_count <= kExpansionBudget - accumulator_count &&
            dfb_count + accumulator_count <= kExpansionBudget - value_count &&
            dfb_count + accumulator_count + value_count <=
                kExpansionBudget - transport_count,
        "current Device family expansion exceeds the bounded proof budget");
  ffi::Map<ffi::String, Integer> expanded_groups;
  ffi::Map<ffi::String, ffi::Array<Integer>> expanded_relations;
  int64_t pipeline_depth = 1;
  auto pipeline_stages = mod->GetAttr<Integer>(kPipelineStagesAttr);
  auto pipeline_extent = mod->GetAttr<Integer>(kPipelineExtentAttr);
  if (pipeline_stages.has_value() && pipeline_extent.has_value())
    pipeline_depth = std::min(pipeline_stages.value()->value,
                              pipeline_extent.value()->value);
  if (!stream_operands) {
    for (size_t family_id = 0; family_id < dfbs.value().size(); ++family_id) {
      const DeviceIRDescriptorFamily &family = dfbs.value()[family_id];
      const DFBStorageGroupRelation &relation =
          storage_relations.value()[family_id];
      Check(relation.defined() &&
                relation->dfb_family_id == static_cast<int64_t>(family_id) &&
                family->fields.count("dfb_id"),
            "storage-group relation does not match its DFB family");
      std::vector<int64_t> ids = ExpandDeviceIntColumn(
          family->fields.at("dfb_id"), family->positions->count);
      std::vector<int64_t> groups = ExpandDeviceIntColumn(
          relation->storage_group_ids, family->positions->count);
      std::vector<int64_t> ordinals = ExpandDeviceIntColumn(
          relation->transaction_ordinals, family->positions->count);
      std::vector<int64_t> pipeline_iterations = ExpandDeviceIntColumn(
          relation->pipeline_iterations, family->positions->count);
      for (size_t row = 0; row < ids.size(); ++row) {
        ffi::String key = std::to_string(ids[row]);
        int64_t iteration = pipeline_iterations[row];
        expanded_groups.Set(key, Integer(groups[row]));
        expanded_relations.Set(
            key,
            {Integer(iteration), Integer(iteration < 0 || pipeline_depth <= 0
                                             ? 0
                                             : iteration % pipeline_depth)});
      }
    }
    mod = WithAttr(mod, "tt.expanded_storage_group_descriptors",
                   storage_groups.value());
    mod = WithAttr(mod, kDFBStorageGroupsAttr, expanded_groups);
    mod = WithAttr(mod, kInternalDFBStorageGroupsAttr, expanded_groups);
    mod = WithoutAttr(mod, kDFBStorageGroupRelationsAttr);
    mod = WithAttr(mod, "tt.expanded_dfb_transaction_relations",
                   expanded_relations);
    if (pipeline_stages.has_value() && pipeline_extent.has_value())
      mod = WithAttr(mod, kInternalDFBTransactionRelationsAttr,
                     expanded_relations);
    if (pipeline_stages.has_value() && pipeline_extent.has_value())
      mod = WithAttr(mod, kPipelineRelationsAttr, expanded_relations);
  }
  mod = WithAttr(mod, "tt.current_device_table_expansion", Integer(1));
  if (stream_operands) {
    ffi::Array<DFBDescriptor> stream_table;
    for (const DFBStorageGroupDescriptor &stream : storage_groups.value())
      stream_table.push_back(DFBDescriptor(
          stream->storage_group_id,
          "stream." + std::to_string(stream->storage_group_id),
          stream->element_dtype, stream->tile_shape,
          stream->block_shape_in_tiles, stream->block_count, std::nullopt,
          stream->producer_slot, stream->producer_domain, stream->consumer_slot,
          stream->consumer_domain, stream->transaction_count_or_loop_relation,
          stream->source_span));
    mod = WithAttr(mod, kDFBTableAttr, stream_table);
  } else {
    mod = WithAttr(
        mod, kDFBTableAttr,
        ExpandCurrentTable<DFBDescriptor>(dfbs.value(), dfb_count, RestoreDFBs,
                                          [](const DFBDescriptor &descriptor) {
                                            return descriptor->dfb_id;
                                          }));
    mod = WithoutAttr(mod, kDFBFamiliesAttr);
  }
  mod =
      WithAttr(mod, "tt.transport_table",
               ExpandTable<PipeDescriptor>(transports.value(), transport_count,
                                           RestoreCurrentTransports));
  mod = WithoutAttr(mod, kTransportFamiliesAttr);
  if (accumulators.has_value()) {
    mod = WithAttr(mod, kAccumulatorTableAttr, compute.accumulators);
    mod = WithAttr(mod, kExpandedAccumulatorLocalIDs,
                   compute.accumulator_local_ids);
    mod = WithoutAttr(mod, kAccumulatorFamiliesAttr);
  }
  if (values.has_value()) {
    mod = WithAttr(mod, kComputeValueTableAttr, compute.values);
    mod = WithAttr(mod, kExpandedComputeValueLocalIDs, compute.value_local_ids);
    mod = WithoutAttr(mod, kComputeValueFamiliesAttr);
  }
  return ShiftCurrentComputeIDs(mod, compute.accumulators,
                                compute.accumulator_local_ids, compute.values,
                                compute.value_local_ids, 1);
}

IRModule DiscardCurrentDeviceExpansionMetadata(IRModule mod) {
  for (const char *key :
       {"tt.current_device_table_expansion",
        "tt.expanded_storage_group_descriptors",
        "tt.expanded_dfb_transaction_relations", kExpandedAccumulatorLocalIDs,
        kExpandedComputeValueLocalIDs})
    if (mod->attrs.defined() && mod->attrs->dict.count(key))
      mod = WithoutAttr(mod, key);
  if (mod->attrs.defined() && mod->attrs->dict.count(kPipelineRelationsAttr))
    mod = WithoutAttr(mod, kPipelineRelationsAttr);
  return mod;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.tenstorrent.ParameterizeCurrentDeviceTables",
           ParameterizeCurrentDeviceTables)
      .def("tl.tenstorrent.ExpandCurrentDeviceTables",
           ExpandCurrentDeviceTables);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

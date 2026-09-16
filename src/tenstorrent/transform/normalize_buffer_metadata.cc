/*!
 * \file tenstorrent/transform/normalize_buffer_metadata.cc
 * \brief Attach typed, deterministic metadata to frontend Buffers.
 */

#include "../ir/device_ir.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace tenstorrent {

using namespace tirx;

namespace {

constexpr const char *kAllocBufferAnnotations = "tl.alloc_buffer_annotations";
constexpr const char *kDFBBlockCount = "tt.dfb_block_count";
constexpr const char *kTileShape = "tt.tile_shape";
constexpr const char *kTensorBacked = "tt.tensor_backed";

using BufferSet =
    std::unordered_set<Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using VarMetadataMap =
    std::unordered_map<Var, ffi::Map<ffi::String, ffi::Any>, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>;
using VarStringMap = std::unordered_map<Var, ffi::String, ffi::ObjectPtrHash,
                                        ffi::ObjectPtrEqual>;

void ThrowMalformed(const std::string &message) {
  TVM_FFI_THROW(ValueError)
      << "Malformed Tenstorrent buffer metadata: " << message;
}

void ThrowUnsupported(const std::string &message) {
  TVM_FFI_THROW(NotImplementedError)
      << "Unsupported Tenstorrent buffer metadata: " << message;
}

struct AllocationRecord {
  Buffer buffer;
  ffi::Map<ffi::String, ffi::Any> annotations;
};

class AllocationCollector : public StmtExprVisitor {
public:
  static std::vector<AllocationRecord> Collect(const Stmt &body) {
    AllocationCollector collector;
    collector(body);
    return std::move(collector.allocations_);
  }

  void VisitStmt_(const SBlockNode *op) final {
    VarMetadataMap annotation_by_data;
    if (auto annotation = op->annotations.Get(kAllocBufferAnnotations)) {
      auto table = annotation.value()
                       .as<ffi::Map<Var, ffi::Map<ffi::String, ffi::Any>>>();
      if (!table.has_value()) {
        ThrowMalformed(std::string("'") + kAllocBufferAnnotations +
                       "' must be Map<Var, Map<String, Any>>");
      }
      for (const auto &[data, entries] : table.value()) {
        annotation_by_data.emplace(data, entries);
      }
    }

    BufferSet allocated;
    for (const Buffer &buffer : op->alloc_buffers) {
      if (!allocated.insert(buffer).second) {
        ThrowMalformed("an SBlock allocates the same Buffer object twice");
      }
      ffi::Map<ffi::String, ffi::Any> entries;
      auto it = annotation_by_data.find(buffer->data);
      if (it != annotation_by_data.end()) {
        entries = it->second;
        annotation_by_data.erase(it);
      }
      allocations_.push_back({buffer, std::move(entries)});
    }
    if (!annotation_by_data.empty()) {
      ThrowMalformed("allocation annotations contain a dangling Buffer data "
                     "reference");
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    // LowerOpaqueBlock moves per-buffer annotations onto AllocBuffer. Keep
    // collecting the same Buffer identity across that representation change.
    allocations_.push_back({op->buffer, op->annotations});
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  std::vector<AllocationRecord> allocations_;
};

ffi::String ClassifyBuffer(const Buffer &buffer) {
  const std::string scope = buffer.scope();
  if (scope == "shared" || scope == "shared.dyn") {
    return "logical_dfb_candidate";
  }
  if (scope == "local.fragment") {
    return "compute_fragment";
  }
  if (scope == "local" || scope == "local.var") {
    return "scalar_local";
  }
  if (scope.empty() || scope == "global") {
    return "tensor";
  }
  ThrowUnsupported("buffer '" + std::string(buffer->name) + "' uses scope '" +
                   scope + "'");
  return ffi::String();
}

struct ParsedAnnotations {
  ffi::Array<PrimExpr> tile_shape;
  ffi::Optional<PrimExpr> block_count;
  ffi::String tile_shape_origin{"unset"};
  ffi::String block_count_origin{"unset"};
};

ParsedAnnotations
ParseAnnotations(const Buffer &buffer,
                 const ffi::Map<ffi::String, ffi::Any> &annotations) {
  ParsedAnnotations result;
  for (const auto &[key, value] : annotations) {
    const std::string name = key;
    if (name == kTensorBacked) {
      ThrowUnsupported("'tt.tensor_backed' is deferred beyond Phase 1");
    }
    if (name == kTileShape) {
      auto shape = value.as<ffi::Array<PrimExpr>>();
      if (!shape.has_value() || shape.value().size() != 2) {
        ThrowMalformed("buffer '" + std::string(buffer->name) +
                       "' has a non-pair 'tt.tile_shape'");
      }
      for (const PrimExpr &extent : shape.value()) {
        const auto *integer = extent.as<IntImmNode>();
        if (integer == nullptr || integer->dtype.is_bool() ||
            integer->value <= 0) {
          ThrowMalformed("'tt.tile_shape' entries must be positive static "
                         "integers");
        }
      }
      result.tile_shape = shape.value();
      result.tile_shape_origin = "explicit";
      continue;
    }
    if (name == kDFBBlockCount) {
      const auto *integer = value.as<IntImmNode>();
      if (integer == nullptr || integer->dtype.is_bool()) {
        ThrowMalformed("'tt.dfb_block_count' must be a compile-time integer");
      }
      if (integer->value < 1 || integer->value > 32) {
        ThrowMalformed("'tt.dfb_block_count' must be in [1, 32]");
      }
      result.block_count = ffi::GetRef<PrimExpr>(integer);
      result.block_count_origin = "explicit";
      continue;
    }
    if (name.rfind("tt.", 0) == 0) {
      ThrowUnsupported("allocation annotation '" + name +
                       "' is not part of the Phase 1 contract");
    }
  }
  return result;
}

void ValidateExistingTable(const PrimFunc &func,
                           const ffi::Array<TTBufferMetadata> &table) {
  std::unordered_set<std::string> ids;
  BufferSet buffers;
  VarSet allocation_data;
  for (const TTBufferMetadata &metadata : table) {
    if (!metadata.defined() || !metadata->buffer.defined()) {
      ThrowMalformed("'tt.buffer_metadata_table' contains an undefined entry");
    }
    if (!ids.insert(metadata->buffer_id).second) {
      ThrowMalformed("duplicate buffer_id '" +
                     std::string(metadata->buffer_id) + "'");
    }
    if (!buffers.insert(metadata->buffer).second) {
      ThrowMalformed("'tt.buffer_metadata_table' contains a Buffer more than "
                     "once");
    }
    allocation_data.insert(metadata->buffer->data);
  }

  for (const auto &[parameter, buffer] : func->buffer_map) {
    if (!buffers.count(buffer)) {
      ThrowMalformed("'tt.buffer_metadata_table' omits parameter Buffer '" +
                     std::string(buffer->name) + "'");
    }
  }
  for (const AllocationRecord &allocation :
       AllocationCollector::Collect(func->body)) {
    // LowerOpaqueBlock retains the logical Buffer in DeclBuffer but creates
    // a physical allocation Buffer sharing its data Var. Allocation metadata
    // is attached to that storage identity, not to a particular wrapper.
    if (!allocation_data.count(allocation.buffer->data)) {
      ThrowMalformed("'tt.buffer_metadata_table' omits allocated Buffer '" +
                     std::string(allocation.buffer->name) + "'");
    }
  }
}

PrimFunc NormalizeBufferMetadata(PrimFunc func) {
  if (auto table = func->GetAttr<ffi::Array<TTBufferMetadata>>(
          kBufferMetadataTableAttr)) {
    ValidateExistingTable(func, table.value());
    return func;
  }

  ffi::Array<TTBufferMetadata> table;
  BufferSet seen_buffers;
  VarStringMap first_id_by_data;

  for (size_t index = 0; index < func->params.size(); ++index) {
    const Var &parameter = func->params[index];
    auto buffer_it = func->buffer_map.find(parameter);
    if (buffer_it == func->buffer_map.end()) {
      ThrowMalformed("parameter '" + std::string(parameter->name_hint) +
                     "' has no Buffer");
    }
    const Buffer &buffer = (*buffer_it).second;
    if (!seen_buffers.insert(buffer).second) {
      ThrowMalformed("the same Buffer object is bound to multiple ABI "
                     "parameters");
    }
    ffi::String buffer_id = "tensor." + std::to_string(index);
    ffi::Optional<ffi::String> alias_of;
    auto alias = first_id_by_data.find(buffer->data);
    if (alias == first_id_by_data.end()) {
      first_id_by_data.emplace(buffer->data, buffer_id);
    } else {
      alias_of = alias->second;
    }
    table.push_back(TTBufferMetadata(
        buffer_id, buffer, "tensor", Integer(index), {}, {}, "",
        /*shard_spec=*/std::nullopt, /*dfb_block_count=*/std::nullopt,
        /*tensor_backing=*/std::nullopt, std::move(alias_of), "unset", "unset",
        "unset", "unset", buffer->span));
  }

  std::vector<AllocationRecord> allocations =
      AllocationCollector::Collect(func->body);
  for (size_t index = 0; index < allocations.size(); ++index) {
    const Buffer &buffer = allocations[index].buffer;
    if (!seen_buffers.insert(buffer).second) {
      ThrowMalformed("Buffer '" + std::string(buffer->name) +
                     "' is allocated more than once");
    }
    ffi::String kind = ClassifyBuffer(buffer);
    ffi::String buffer_id = "buffer." + std::to_string(index);
    ffi::Optional<ffi::String> alias_of;
    auto alias = first_id_by_data.find(buffer->data);
    if (alias == first_id_by_data.end()) {
      first_id_by_data.emplace(buffer->data, buffer_id);
    } else {
      alias_of = alias->second;
    }
    ParsedAnnotations annotations =
        ParseAnnotations(buffer, allocations[index].annotations);
    if (kind != "logical_dfb_candidate" &&
        (!annotations.tile_shape.empty() ||
         annotations.block_count.has_value())) {
      ThrowMalformed("Tenstorrent DFB annotations are attached to non-shared "
                     "buffer '" +
                     std::string(buffer->name) + "'");
    }
    table.push_back(TTBufferMetadata(
        buffer_id, buffer, kind, /*global_arg_index=*/std::nullopt,
        std::move(annotations.tile_shape), {}, "",
        /*shard_spec=*/std::nullopt, std::move(annotations.block_count),
        /*tensor_backing=*/std::nullopt, std::move(alias_of),
        std::move(annotations.tile_shape_origin),
        std::move(annotations.block_count_origin), "unset", "unset",
        buffer->span));
  }

  return WithAttr(std::move(func), kBufferMetadataTableAttr, std::move(table));
}

} // namespace

tvm::transform::Pass NormalizeTenstorrentBufferMetadata() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    return NormalizeBufferMetadata(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tenstorrent.NormalizeTenstorrentBufferMetadata", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.tenstorrent.transform.NormalizeTenstorrentBufferMetadata",
      NormalizeTenstorrentBufferMetadata);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

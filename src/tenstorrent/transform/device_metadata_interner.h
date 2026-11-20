/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#ifndef TVM_TL_TENSTORRENT_TRANSFORM_DEVICE_METADATA_INTERNER_H_
#define TVM_TL_TENSTORRENT_TRANSFORM_DEVICE_METADATA_INTERNER_H_

#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <tvm/ffi/reflection/creator.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>

#include "../ir/device_ir.h"

namespace tvm {
namespace tl {
namespace tenstorrent {

// Static scheduling creates many identical shape/access-map containers and
// literals. Share these when finishing compute requirements, without changing
// the Device schema, resource IDs, instruction order or diagnostic contents.
// Keys compare canonical children by identity, including every source span.
// In particular, structural equality must not merge distinct buffers/variables,
// resource descriptors, or effectful operations.
class DeviceMetadataInterner {
public:
  static IRModule Rewrite(const IRModule &mod) {
    DeviceMetadataInterner interner;
    return Downcast<IRModule>(interner.Visit(mod));
  }

private:
  static uint64_t DoubleBits(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }
  struct Key {
    int32_t type;
    std::vector<ffi::Any> fields;
  };
  struct Hash {
    size_t operator()(const Key &key) const {
      size_t hash = key.type;
      for (const auto &field : key.fields) {
        auto number = field.as<double>();
        size_t field_hash =
            number.has_value()
                ? std::hash<uint64_t>()(DoubleBits(number.value()))
                : ffi::AnyHash()(field);
        hash ^= field_hash + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };
  struct Equal {
    bool operator()(const Key &a, const Key &b) const {
      if (a.type != b.type || a.fields.size() != b.fields.size())
        return false;
      for (size_t i = 0; i < a.fields.size(); ++i) {
        auto x = a.fields[i].as<double>(), y = b.fields[i].as<double>();
        if (x.has_value() || y.has_value()) {
          if (!x.has_value() || !y.has_value() ||
              DoubleBits(x.value()) != DoubleBits(y.value()))
            return false;
        } else if (!ffi::AnyEqual()(a.fields[i], b.fields[i])) {
          return false;
        }
      }
      return true;
    }
  };

  ffi::Any VisitAny(const ffi::Any &value) {
    if (auto object = value.as<ffi::ObjectRef>())
      return Visit(object.value());
    return value;
  }

  ffi::ObjectRef Intern(const ffi::ObjectRef &object, Key key) {
    auto [it, inserted] = shared_.emplace(std::move(key), object);
    return it->second;
  }

  ffi::ObjectRef Visit(const ffi::ObjectRef &object) {
    if (!object.defined())
      return object;
    auto found = rewritten_.find(object);
    if (found != rewritten_.end())
      return found->second;
    ffi::ObjectRef result = RewriteObject(object);
    rewritten_.emplace(object, result);
    return result;
  }

  ffi::ObjectRef RewriteObject(const ffi::ObjectRef &object) {
    Key key{object->type_index(), {}};
    if (auto array = object.as<ffi::Array<ffi::Any>>()) {
      ffi::Array<ffi::Any> result;
      bool changed = false;
      for (const auto &item : array.value()) {
        ffi::Any value = VisitAny(item);
        changed |= !ffi::AnyEqual()(item, value);
        result.push_back(value);
        key.fields.push_back(value);
      }
      return Intern(changed ? result : object, std::move(key));
    }
    if (auto map = object.as<ffi::Map<ffi::Any, ffi::Any>>()) {
      ffi::Map<ffi::Any, ffi::Any> result;
      bool changed = false;
      // Preserve identity-bearing keys. Attribute keys are strings; their
      // iteration order is stable for maps with the same keys.
      for (const auto &[name, item] : map.value()) {
        ffi::Any value = VisitAny(item);
        changed |= !ffi::AnyEqual()(item, value);
        result.Set(name, value);
        key.fields.push_back(name);
        key.fields.push_back(value);
      }
      return Intern(changed ? result : object, std::move(key));
    }
    // These objects define identity or contain compiler/runtime state. They
    // remain exactly the original handles, including their provenance.
    if (object.as<tirx::BufferNode>() || object.as<tirx::VarNode>() ||
        object.as<GlobalVarNode>() || object.as<OpNode>() ||
        object.as<SpanNode>() || object.as<SourceNameNode>() ||
        object.as<TargetNode>())
      return object;

    const TVMFFITypeInfo *info = TVMFFIGetTypeInfo(object->type_index());
    if (!ffi::HasCreator(info))
      return object;
    ffi::Map<ffi::String, ffi::Any> fields;
    bool changed = false;
    ffi::reflection::ForEachFieldInfo(info, [&](const TVMFFIFieldInfo *field) {
      ffi::Any original = ffi::reflection::FieldGetter(field)(object);
      ffi::Any value = VisitAny(original);
      changed |= !ffi::AnyEqual()(original, value);
      fields.Set(ffi::String(field->name), value);
      key.fields.push_back(value);
    });
    ffi::ObjectRef result = changed
                                ? ffi::reflection::ObjectCreator(info)(fields)
                                      .cast<ffi::ObjectRef>()
                                : object;
    // Only immutable metadata with identity-sensitive child keys is interned.
    // Float literals are deliberately excluded (signed zero / NaN payloads).
    if (object.as<IntImmNode>() || object.as<tirx::StringImmNode>() ||
        object.as<RangeNode>() || object.as<tirx::BufferRegionNode>() ||
        object.as<CoreCoordNode>() || object.as<CoreDomainNode>() ||
        object.as<TensorBackingNode>() || object.as<ComputeRequirementsNode>())
      return Intern(result, std::move(key));
    return result;
  }

  std::unordered_map<ffi::ObjectRef, ffi::ObjectRef, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      rewritten_;
  std::unordered_map<Key, ffi::ObjectRef, Hash, Equal> shared_;
};

} // namespace tenstorrent
} // namespace tl
} // namespace tvm
#endif // TVM_TL_TENSTORRENT_TRANSFORM_DEVICE_METADATA_INTERNER_H_

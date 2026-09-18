/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#include "device_statement_compaction.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

#include <tvm/ffi/reflection/creator.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>

namespace tvm {
namespace tl {
namespace tenstorrent {
namespace {
using namespace tirx;

constexpr int64_t kMaxExpansion = 4000000;
constexpr size_t kMaxBlock = 4096;

[[noreturn]] void Invalid(const char *message) {
  TVM_FFI_THROW(ValueError) << "[TenstorrentDeviceStatements] " << message;
}

bool Container(const ffi::ObjectRef &object) {
  return object.as<ffi::Array<ffi::Any>>() ||
         object.as<ffi::Map<ffi::Any, ffi::Any>>();
}

size_t Combine(size_t a, size_t b) {
  return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
}

bool Opaque(const ffi::ObjectRef &object) {
  return object.as<BufferNode>() || object.as<VarNode>() ||
         object.as<GlobalVarNode>() || object.as<OpNode>() ||
         object.as<SpanNode>() || object.as<SourceNameNode>() ||
         object.as<TargetNode>();
}

// AnyEqual follows numeric equality for doubles. Preserve signed zero and NaN
// payloads in all literal fields instead.
uint64_t DoubleBits(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
size_t ScalarHash(const ffi::Any &value) {
  if (auto number = value.as<double>())
    return std::hash<uint64_t>()(DoubleBits(number.value()));
  return ffi::AnyHash()(value);
}
bool ScalarEqual(const ffi::Any &a, const ffi::Any &b) {
  auto x = a.as<double>(), y = b.as<double>();
  if (x.has_value() || y.has_value())
    return x.has_value() && y.has_value() &&
           DoubleBits(x.value()) == DoubleBits(y.value());
  return ffi::AnyEqual()(a, b);
}

using Fields = std::vector<ffi::Any>;
Fields GetFields(const ffi::ObjectRef &object) {
  Fields fields;
  if (auto array = object.as<ffi::Array<ffi::Any>>()) {
    for (const auto &value : array.value())
      fields.push_back(value);
  } else if (auto map = object.as<ffi::Map<ffi::Any, ffi::Any>>()) {
    // Call annotations use string keys. Sorting makes matching independent of
    // insertion order while leaving the actual map unchanged when rebuilding.
    std::vector<std::pair<ffi::Any, ffi::Any>> entries;
    bool strings = true;
    for (const auto &[key, value] : map.value()) {
      entries.emplace_back(key, value);
      strings &= key.as<ffi::String>().has_value();
    }
    if (strings)
      std::sort(entries.begin(), entries.end(),
                [](const auto &a, const auto &b) {
                  return a.first.template cast<ffi::String>() <
                         b.first.template cast<ffi::String>();
                });
    for (const auto &[key, value] : entries) {
      fields.push_back(key);
      fields.push_back(value);
    }
  } else {
    const TVMFFITypeInfo *info = TVMFFIGetTypeInfo(object->type_index());
    ffi::reflection::ForEachFieldInfo(info, [&](const TVMFFIFieldInfo *field) {
      fields.push_back(ffi::reflection::FieldGetter(field)(object));
    });
  }
  return fields;
}

struct StatementInfo {
  size_t hash;
  std::vector<IntImm> integers;
};

size_t Describe(const ffi::Any &value, std::vector<IntImm> *integers) {
  auto object = value.as<ffi::ObjectRef>();
  if (!object.has_value() || !object.value().defined())
    return ScalarHash(value);
  const ffi::ObjectRef &ref = object.value();
  size_t hash = ref->type_index();
  if (auto integer = ref.as<IntImm>()) {
    integers->push_back(integer.value());
    hash = Combine(hash, ScalarHash(integer.value()->dtype));
    return Combine(hash, ScalarHash(integer.value()->span));
  }
  if (Opaque(ref) || (!Container(ref) &&
                      !ffi::HasCreator(TVMFFIGetTypeInfo(ref->type_index()))))
    return Combine(hash, ScalarHash(value));
  Fields fields = GetFields(ref);
  hash = Combine(hash, fields.size());
  bool map = ref.as<ffi::Map<ffi::Any, ffi::Any>>().has_value();
  for (size_t i = 0; i < fields.size(); ++i)
    hash = Combine(hash, map && i % 2 == 0 ? ScalarHash(fields[i])
                                           : Describe(fields[i], integers));
  return hash;
}

bool SameSkeleton(const ffi::Any &a, const ffi::Any &b) {
  auto x = a.as<ffi::ObjectRef>(), y = b.as<ffi::ObjectRef>();
  if (!x.has_value() || !y.has_value() || !x.value().defined() ||
      !y.value().defined())
    return ScalarEqual(a, b);
  if (x.value()->type_index() != y.value()->type_index())
    return false;
  if (auto integer = x.value().as<IntImm>()) {
    IntImm other = Downcast<IntImm>(y.value());
    return integer.value()->dtype == other->dtype &&
           integer.value()->span.same_as(other->span);
  }
  if (Opaque(x.value()) ||
      (!Container(x.value()) &&
       !ffi::HasCreator(TVMFFIGetTypeInfo(x.value()->type_index()))))
    return ScalarEqual(a, b);
  Fields lhs = GetFields(x.value()), rhs = GetFields(y.value());
  if (lhs.size() != rhs.size())
    return false;
  bool map = x.value().as<ffi::Map<ffi::Any, ffi::Any>>().has_value();
  for (size_t i = 0; i < lhs.size(); ++i)
    if (!(map && i % 2 == 0 ? ScalarEqual(lhs[i], rhs[i])
                            : SameSkeleton(lhs[i], rhs[i])))
      return false;
  return true;
}

bool Fits(__int128 value, DataType dtype) {
  if (dtype.lanes() != 1 || (!dtype.is_int() && !dtype.is_uint()) ||
      dtype.bits() <= 0 || dtype.bits() > 64)
    return false;
  if (dtype.is_uint()) {
    // IntImm stores its payload in int64_t, even for unsigned dtypes.
    __int128 maximum = dtype.bits() == 64 ? std::numeric_limits<int64_t>::max()
                                          : ((__int128{1} << dtype.bits()) - 1);
    return value >= 0 && value <= maximum;
  }
  __int128 bound = __int128{1} << (dtype.bits() - 1);
  return value >= -bound && value < bound;
}

bool CanRepresentColumn(const IntImm &base, int64_t step, size_t repeats) {
  if (!step)
    return true;
  DataType dtype = base->dtype;
  __int128 last = repeats - 1;
  return Fits(base->value, dtype) && Fits(step, dtype) && Fits(last, dtype) &&
         Fits(last * step, dtype) && Fits(base->value + last * step, dtype);
}

// Rebuild only paths whose integer leaf changes. Reflection also covers nested
// arrays/maps in Call annotations, which the standard expression mutator skips.
class ObjectRewriter {
public:
  virtual ~ObjectRewriter() = default;
  ffi::ObjectRef Rewrite(const ffi::ObjectRef &object) {
    if (!object.defined())
      return object;
    if (auto replacement = RewriteLeaf(object))
      return replacement.value();
    if (Opaque(object) ||
        (!Container(object) &&
         !ffi::HasCreator(TVMFFIGetTypeInfo(object->type_index()))))
      return object;
    if (auto array = object.as<ffi::Array<ffi::Any>>()) {
      ffi::Array<ffi::Any> result;
      bool changed = false;
      for (const auto &item : array.value()) {
        ffi::Any value = RewriteAny(item);
        changed |= !ScalarEqual(item, value);
        result.push_back(value);
      }
      return changed ? result : object;
    }
    if (auto map = object.as<ffi::Map<ffi::Any, ffi::Any>>()) {
      ffi::Map<ffi::Any, ffi::Any> result;
      bool changed = false;
      // Match the same canonical field order used by Describe.
      Fields fields = GetFields(object);
      for (size_t i = 0; i < fields.size(); i += 2) {
        ffi::Any value = RewriteAny(fields[i + 1]);
        changed |= !ScalarEqual(fields[i + 1], value);
        result.Set(fields[i], value);
      }
      return changed ? result : object;
    }
    ffi::Map<ffi::String, ffi::Any> fields;
    bool changed = false;
    const TVMFFITypeInfo *info = TVMFFIGetTypeInfo(object->type_index());
    ffi::reflection::ForEachFieldInfo(info, [&](const TVMFFIFieldInfo *field) {
      ffi::Any original = ffi::reflection::FieldGetter(field)(object);
      ffi::Any value = RewriteAny(original);
      changed |= !ScalarEqual(original, value);
      fields.Set(ffi::String(field->name), value);
    });
    return changed ? ffi::reflection::ObjectCreator(info)(fields)
                         .cast<ffi::ObjectRef>()
                   : object;
  }

protected:
  virtual ffi::Optional<ffi::ObjectRef>
  RewriteLeaf(const ffi::ObjectRef &object) = 0;

private:
  ffi::Any RewriteAny(const ffi::Any &value) {
    if (auto object = value.as<ffi::ObjectRef>())
      return Rewrite(object.value());
    return value;
  }
};

class AffineWriter : public ObjectRewriter {
public:
  AffineWriter(const std::vector<int64_t> &steps, Var iteration)
      : steps_(steps), iteration_(std::move(iteration)) {}

private:
  ffi::Optional<ffi::ObjectRef>
  RewriteLeaf(const ffi::ObjectRef &object) final {
    if (auto integer = object.as<IntImm>()) {
      if (position_ >= steps_.size())
        Invalid("integer template traversal disagrees with its columns");
      int64_t step = steps_[position_++];
      if (!step)
        return object;
      const IntImm &base = integer.value();
      PrimExpr index = iteration_;
      if (index.dtype() != base->dtype)
        index = Cast(base->dtype, index, base->span);
      return Add(base,
                 Mul(index, IntImm(base->dtype, step, base->span), base->span),
                 base->span);
    }
    return std::nullopt;
  }
  const std::vector<int64_t> &steps_;
  Var iteration_;
  size_t position_{0};
};

class AffineReader : public ObjectRewriter {
public:
  using Bindings =
      std::unordered_map<Var, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
  explicit AffineReader(const Bindings &bindings) : bindings_(bindings) {}

private:
  ffi::Optional<ffi::ObjectRef>
  RewriteLeaf(const ffi::ObjectRef &object) final {
    if (auto var = object.as<Var>()) {
      if (bindings_.count(var.value()))
        Invalid("loop variable occurs outside a generated affine integer");
      return object;
    }
    auto add = object.as<Add>();
    if (!add.has_value())
      return std::nullopt;
    auto base = add.value()->a.as<IntImm>();
    auto multiply = add.value()->b.as<Mul>();
    if (!base.has_value() || !multiply.has_value())
      return std::nullopt;
    auto step = multiply.value()->b.as<IntImm>();
    PrimExpr index = multiply.value()->a;
    if (auto cast = index.as<Cast>()) {
      if (cast.value()->dtype != base.value()->dtype ||
          !cast.value()->annotations.empty())
        return std::nullopt;
      index = cast.value()->value;
    }
    auto variable = index.as<Var>();
    if (!step.has_value() || !variable.has_value())
      return std::nullopt;
    auto found = bindings_.find(variable.value());
    if (found == bindings_.end())
      return std::nullopt;
    DataType dtype = base.value()->dtype;
    if (step.value()->dtype != dtype || add.value()->dtype != dtype ||
        multiply.value()->dtype != dtype)
      Invalid("affine integer has inconsistent dtypes");
    __int128 product = __int128{found->second} * step.value()->value;
    __int128 value = base.value()->value + product;
    if (!Fits(base.value()->value, dtype) ||
        !Fits(step.value()->value, dtype) ||
        multiply.value()->a.dtype() != dtype || !Fits(found->second, dtype) ||
        !Fits(product, dtype) || !Fits(value, dtype))
      Invalid("affine integer overflows its dtype");
    return IntImm(dtype, static_cast<int64_t>(value), add.value()->span);
  }
  const Bindings &bindings_;
};

int64_t ExpandedCount(const Stmt &body, int64_t limit, size_t depth = 0) {
  if (depth > 64)
    Invalid("compact statement nesting exceeds the limit of 64");
  if (body.as<EvaluateNode>())
    return 1;
  if (auto sequence = body.as<SeqStmt>()) {
    if (sequence.value()->seq.empty())
      Invalid("compact statement sequences must not be empty");
    int64_t count = 0;
    for (const Stmt &child : sequence.value()->seq) {
      int64_t child_count = ExpandedCount(child, limit - count, depth + 1);
      if (child_count > limit - count)
        Invalid("expanded operation budget exceeded");
      count += child_count;
    }
    return count;
  }
  auto loop = body.as<For>();
  if (!loop.has_value())
    Invalid("only Evaluate, SeqStmt and serial loops can be expanded");
  auto minimum = loop.value()->min.as<IntImm>();
  auto extent = loop.value()->extent.as<IntImm>();
  auto step = loop.value()->step;
  if (extent.has_value() && extent.value()->value > kMaxExpansion)
    Invalid("expanded operation budget exceeded");
  if (loop.value()->kind != ForKind::kSerial ||
      loop.value()->thread_binding.has_value() ||
      !loop.value()->annotations.empty() || !minimum.has_value() ||
      minimum.value()->value != 0 || !extent.has_value() ||
      extent.value()->value <= 0 || extent.value()->value > kMaxExpansion ||
      (step.has_value() && (!step.value().as<IntImmNode>() ||
                            step.value().as<IntImmNode>()->value != 1)))
    Invalid("compact loops require constant zero minimum and positive serial "
            "unit-step extent without annotations");
  DataType dtype = loop.value()->loop_var.dtype();
  if (!Fits(0, dtype) || minimum.value()->dtype != dtype ||
      extent.value()->dtype != dtype || !Fits(extent.value()->value, dtype) ||
      !Fits(extent.value()->value - 1, dtype) ||
      (step.has_value() && step.value().dtype() != dtype))
    Invalid("compact loop bounds must match a scalar integer loop variable "
            "without overflow");
  int64_t count = ExpandedCount(loop.value()->body, limit, depth + 1);
  if (count > limit / extent.value()->value)
    Invalid("expanded operation budget exceeded");
  return count * extent.value()->value;
}

void Expand(const Stmt &body, AffineReader::Bindings *bindings,
            ffi::Array<Stmt> *output) {
  if (auto sequence = body.as<SeqStmt>()) {
    for (const Stmt &child : sequence.value()->seq)
      Expand(child, bindings, output);
  } else if (auto loop = body.as<For>()) {
    Var variable = loop.value()->loop_var;
    if (bindings->count(variable))
      Invalid("compact loop reuses an enclosing loop variable");
    int64_t extent = loop.value()->extent.as<IntImmNode>()->value;
    for (int64_t i = 0; i < extent; ++i) {
      (*bindings)[variable] = i;
      Expand(loop.value()->body, bindings, output);
    }
    bindings->erase(variable);
  } else {
    AffineReader reader(*bindings);
    output->push_back(Downcast<Stmt>(reader.Rewrite(body)));
  }
}
} // namespace

Stmt CompactDeviceStatements(const Stmt &body) {
  auto sequence = body.as<SeqStmt>();
  if (!sequence.has_value() || sequence.value()->seq.size() < 2)
    return body;
  const auto &statements = sequence.value()->seq;
  size_t count = statements.size();
  std::vector<StatementInfo> info;
  std::unordered_map<size_t, std::vector<size_t>> positions;
  std::vector<uint64_t> prefix(count + 1), powers(count + 1, 1);
  constexpr uint64_t kBase = 1000000007;
  for (size_t i = 0; i < count; ++i) {
    const auto *evaluate = statements[i].as<EvaluateNode>();
    if (!evaluate || !evaluate->value.as<CallNode>())
      return body;
    StatementInfo item;
    item.hash = Describe(statements[i], &item.integers);
    positions[item.hash].push_back(i);
    prefix[i + 1] = prefix[i] * kBase + item.hash;
    powers[i + 1] = powers[i] * kBase;
    info.push_back(std::move(item));
  }
  auto same_block = [&](size_t a, size_t b, size_t length) {
    return prefix[a + length] - prefix[a] * powers[length] ==
           prefix[b + length] - prefix[b] * powers[length];
  };
  ffi::Array<Stmt> result;
  for (size_t begin = 0; begin < count;) {
    size_t best_length = 0, best_repeats = 0, best_saving = 0;
    std::vector<std::vector<int64_t>> best_steps;
    const auto &candidates = positions.at(info[begin].hash);
    auto next = std::upper_bound(candidates.begin(), candidates.end(), begin);
    size_t attempts = 0;
    for (; next != candidates.end() && attempts < 256; ++next, ++attempts) {
      size_t length = *next - begin;
      if (length > kMaxBlock || length > (count - begin) / 2)
        break;
      // Even a perfect repetition cannot save its retained template and loop.
      // Once this monotone bound loses, larger candidates cannot improve it.
      if (count - begin - length - 1 <= best_saving)
        break;
      size_t maximum_repeats = (count - begin) / length;
      if (length * (maximum_repeats - 1) - 1 <= best_saving)
        continue;
      if (!same_block(begin, begin + length, length))
        continue;
      std::vector<std::vector<int64_t>> steps(length);
      bool valid = true;
      for (size_t j = 0; j < length && valid; ++j) {
        const auto &first = info[begin + j].integers;
        const auto &second = info[begin + length + j].integers;
        if (first.size() != second.size() ||
            !SameSkeleton(statements[begin + j],
                          statements[begin + length + j])) {
          valid = false;
          break;
        }
        for (size_t k = 0; k < first.size(); ++k) {
          __int128 step = __int128{second[k]->value} - first[k]->value;
          if (step < std::numeric_limits<int64_t>::min() ||
              step > std::numeric_limits<int64_t>::max() ||
              !CanRepresentColumn(first[k], static_cast<int64_t>(step), 2)) {
            valid = false;
            break;
          }
          steps[j].push_back(static_cast<int64_t>(step));
        }
      }
      if (!valid)
        continue;
      size_t repeats = 2;
      while (repeats < kMaxExpansion &&
             length <= (count - begin) / (repeats + 1) &&
             same_block(begin, begin + repeats * length, length)) {
        valid = true;
        for (size_t j = 0; j < length && valid; ++j) {
          const auto &first = info[begin + j].integers;
          const auto &current = info[begin + repeats * length + j].integers;
          if (first.size() != current.size() ||
              !SameSkeleton(statements[begin + j],
                            statements[begin + repeats * length + j])) {
            valid = false;
            break;
          }
          for (size_t k = 0; k < first.size(); ++k)
            if (__int128{first[k]->value} + __int128{steps[j][k]} * repeats !=
                    current[k]->value ||
                !CanRepresentColumn(first[k], steps[j][k], repeats + 1)) {
              valid = false;
              break;
            }
        }
        if (!valid)
          break;
        ++repeats;
      }
      size_t saving = length * (repeats - 1) - 1;
      if (saving > best_saving) {
        best_saving = saving;
        best_length = length;
        best_repeats = repeats;
        best_steps = std::move(steps);
      }
    }
    if (!best_length) {
      result.push_back(statements[begin++]);
      continue;
    }
    Var iteration("tt_compact_iteration", DataType::Int(64));
    ffi::Array<Stmt> loop_body;
    for (size_t j = 0; j < best_length; ++j) {
      AffineWriter writer(best_steps[j], iteration);
      loop_body.push_back(
          Downcast<Stmt>(writer.Rewrite(statements[begin + j])));
    }
    result.push_back(For(iteration, IntImm(DataType::Int(64), 0),
                         IntImm(DataType::Int(64), best_repeats),
                         ForKind::kSerial, SeqStmt::Flatten(loop_body),
                         std::nullopt, {}, std::nullopt,
                         statements[begin]->span));
    begin += best_length * best_repeats;
  }
  if (result.size() == count)
    return body;
  if (result.size() > 1)
    return SeqStmt(result, body->span);
  For loop = Downcast<For>(result[0]);
  loop.CopyOnWrite()->span = body->span;
  return loop;
}

Stmt ExpandDeviceStatements(const Stmt &body, int64_t *budget) {
  if (!budget || *budget < 0 || *budget > kMaxExpansion)
    Invalid("expanded operation budget must be between zero and 4000000");
  int64_t count = ExpandedCount(body, *budget);
  if (count > *budget)
    Invalid("expanded operation budget exceeded");
  if (body.as<EvaluateNode>()) {
    *budget -= count;
    return body;
  }
  AffineReader::Bindings bindings;
  ffi::Array<Stmt> result;
  Expand(body, &bindings, &result);
  *budget -= count;
  return result.size() == 1 ? result[0] : SeqStmt(result, body->span);
}

} // namespace tenstorrent
} // namespace tl
} // namespace tvm

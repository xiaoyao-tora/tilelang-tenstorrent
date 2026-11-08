/* Copyright (c) Tile-AI Corporation. Licensed under the MIT License. */
#include "device_statement_compaction.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>
#include <vector>

#include <tvm/ffi/reflection/creator.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include "../ir/device_ir.h"
#include "../op/builtin.h"

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

size_t CoreLocalHash(const ffi::ObjectRef &object) {
  if (!object.defined())
    return 0;
  size_t hash = object->type_index();
  if (auto buffer = object.as<Buffer>()) {
    hash = Combine(hash, ScalarHash(buffer.value()->name));
    hash = Combine(hash, ScalarHash(buffer.value().scope()));
    hash = Combine(hash, ScalarHash(buffer.value()->dtype));
    hash = Combine(hash, ffi::StructuralHash()(buffer.value()->shape));
    hash = Combine(hash, ffi::StructuralHash()(buffer.value()->strides));
    hash =
        Combine(hash, ffi::StructuralHash()(buffer.value()->axis_separators));
    hash = Combine(hash, ffi::StructuralHash()(buffer.value()->elem_offset));
    hash = Combine(hash, std::hash<int>()(buffer.value()->data_alignment));
    hash = Combine(hash, std::hash<int>()(buffer.value()->offset_factor));
    hash = Combine(
        hash, std::hash<int>()(static_cast<int>(buffer.value()->buffer_type)));
    return Combine(hash, CoreLocalHash(buffer.value()->span));
  }
  if (auto variable = object.as<Var>()) {
    hash = Combine(hash, ScalarHash(variable.value()->name_hint));
    return Combine(hash, ScalarHash(variable.value().dtype()));
  }
  if (auto span = object.as<Span>()) {
    if (span.value()->source_name.defined())
      hash = Combine(hash, ScalarHash(span.value()->source_name->name));
    for (int64_t value : {span.value()->line, span.value()->column,
                          span.value()->end_line, span.value()->end_column})
      hash = Combine(hash, std::hash<int64_t>()(value));
    return hash;
  }
  if (auto source = object.as<SourceName>())
    return Combine(hash, ScalarHash(source.value()->name));
  return Combine(hash, ScalarHash(object));
}

bool SameCoreLocal(const ffi::ObjectRef &left, const ffi::ObjectRef &right) {
  if (!left.defined() || !right.defined())
    return left.defined() == right.defined();
  if (auto a = left.as<Buffer>()) {
    auto b = right.as<Buffer>();
    return b.has_value() && a.value()->name == b.value()->name &&
           a.value().scope() == b.value().scope() &&
           a.value()->dtype == b.value()->dtype &&
           ffi::StructuralEqual()(a.value()->shape, b.value()->shape) &&
           ffi::StructuralEqual()(a.value()->strides, b.value()->strides) &&
           ffi::StructuralEqual()(a.value()->axis_separators,
                                  b.value()->axis_separators) &&
           ffi::StructuralEqual()(a.value()->elem_offset,
                                  b.value()->elem_offset) &&
           a.value()->data_alignment == b.value()->data_alignment &&
           a.value()->offset_factor == b.value()->offset_factor &&
           a.value()->buffer_type == b.value()->buffer_type &&
           SameCoreLocal(a.value()->span, b.value()->span);
  }
  if (auto a = left.as<Var>()) {
    auto b = right.as<Var>();
    return b.has_value() && a.value()->name_hint == b.value()->name_hint &&
           a.value().dtype() == b.value().dtype();
  }
  if (auto a = left.as<Span>()) {
    auto b = right.as<Span>();
    if (!b.has_value())
      return false;
    ffi::String a_source = a.value()->source_name.defined()
                               ? a.value()->source_name->name
                               : ffi::String("");
    ffi::String b_source = b.value()->source_name.defined()
                               ? b.value()->source_name->name
                               : ffi::String("");
    return a_source == b_source && a.value()->line == b.value()->line &&
           a.value()->column == b.value()->column &&
           a.value()->end_line == b.value()->end_line &&
           a.value()->end_column == b.value()->end_column;
  }
  if (auto a = left.as<SourceName>()) {
    auto b = right.as<SourceName>();
    return b.has_value() && a.value()->name == b.value()->name;
  }
  return ScalarEqual(left, right);
}

size_t Describe(const ffi::Any &value, std::vector<IntImm> *integers,
                bool semantic_core_locals = false) {
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
  if (semantic_core_locals && (ref.as<BufferNode>() || ref.as<VarNode>() ||
                               ref.as<SpanNode>() || ref.as<SourceNameNode>()))
    return CoreLocalHash(ref);
  if (Opaque(ref) || (!Container(ref) &&
                      !ffi::HasCreator(TVMFFIGetTypeInfo(ref->type_index()))))
    return Combine(hash, ScalarHash(value));
  Fields fields = GetFields(ref);
  hash = Combine(hash, fields.size());
  bool map = ref.as<ffi::Map<ffi::Any, ffi::Any>>().has_value();
  for (size_t i = 0; i < fields.size(); ++i)
    hash = Combine(hash, map && i % 2 == 0 ? ScalarHash(fields[i])
                                           : Describe(fields[i], integers,
                                                      semantic_core_locals));
  return hash;
}

bool SameSkeleton(const ffi::Any &a, const ffi::Any &b,
                  bool semantic_core_locals = false) {
  auto x = a.as<ffi::ObjectRef>(), y = b.as<ffi::ObjectRef>();
  if (!x.has_value() || !y.has_value() || !x.value().defined() ||
      !y.value().defined())
    return ScalarEqual(a, b);
  if (x.value()->type_index() != y.value()->type_index())
    return false;
  if (auto integer = x.value().as<IntImm>()) {
    IntImm other = Downcast<IntImm>(y.value());
    return integer.value()->dtype == other->dtype &&
           (semantic_core_locals
                ? SameCoreLocal(integer.value()->span, other->span)
                : integer.value()->span.same_as(other->span));
  }
  if (semantic_core_locals &&
      (x.value().as<BufferNode>() || x.value().as<VarNode>() ||
       x.value().as<SpanNode>() || x.value().as<SourceNameNode>()))
    return SameCoreLocal(x.value(), y.value());
  if (Opaque(x.value()) ||
      (!Container(x.value()) &&
       !ffi::HasCreator(TVMFFIGetTypeInfo(x.value()->type_index()))))
    return ScalarEqual(a, b);
  Fields lhs = GetFields(x.value()), rhs = GetFields(y.value());
  if (lhs.size() != rhs.size())
    return false;
  bool map = x.value().as<ffi::Map<ffi::Any, ffi::Any>>().has_value();
  for (size_t i = 0; i < lhs.size(); ++i)
    if (!(map && i % 2 == 0
              ? ScalarEqual(lhs[i], rhs[i])
              : SameSkeleton(lhs[i], rhs[i], semantic_core_locals)))
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

struct CoreAffineColumn {
  int64_t base;
  int64_t x_step;
  int64_t y_step;
};

class CoreAffineWriter : public ObjectRewriter {
public:
  explicit CoreAffineWriter(const std::vector<CoreAffineColumn> &columns)
      : columns_(columns) {}

  bool complete() const { return position_ == columns_.size(); }

private:
  ffi::Optional<ffi::ObjectRef>
  RewriteLeaf(const ffi::ObjectRef &object) final {
    auto integer = object.as<IntImm>();
    if (!integer.has_value())
      return std::nullopt;
    if (position_ >= columns_.size())
      Invalid("Core parameterization traversed more integers than expected");
    const CoreAffineColumn &column = columns_[position_++];
    if (!column.x_step && !column.y_step)
      return object;
    const DataType dtype = integer.value()->dtype;
    PrimExpr result = IntImm(dtype, column.base, integer.value()->span);
    auto coordinate = [&](const Op &op) -> PrimExpr {
      PrimExpr value = Call(DataType::Int(32), op, {});
      return value.dtype() == dtype ? value : Cast(dtype, value);
    };
    if (column.x_step)
      result = Add(result,
                   Mul(coordinate(tenstorrent::core_x()),
                       IntImm(dtype, column.x_step), integer.value()->span),
                   integer.value()->span);
    if (column.y_step)
      result = Add(result,
                   Mul(coordinate(tenstorrent::core_y()),
                       IntImm(dtype, column.y_step), integer.value()->span),
                   integer.value()->span);
    return result;
  }

  const std::vector<CoreAffineColumn> &columns_;
  size_t position_{0};
};

int64_t ExtendedGCD(int64_t a, int64_t b, int64_t *x, int64_t *y) {
  if (b == 0) {
    *x = 1;
    *y = 0;
    return a;
  }
  int64_t next_x = 0, next_y = 0;
  int64_t gcd = ExtendedGCD(b, a % b, &next_x, &next_y);
  *x = next_y;
  *y = next_x - (a / b) * next_y;
  return gcd;
}

bool FitCoreAffine(const std::vector<DeviceCoreBody> &points,
                   const std::vector<int64_t> &values, DataType dtype,
                   CoreAffineColumn *result) {
  if (points.empty() || points.size() != values.size())
    return false;
  const int64_t x0 = points[0].x, y0 = points[0].y, value0 = values[0];
  bool constant = std::all_of(values.begin(), values.end(),
                              [&](int64_t value) { return value == value0; });
  if (constant) {
    *result = {value0, 0, 0};
    return Fits(value0, dtype);
  }

  size_t first = points.size();
  for (size_t i = 1; i < points.size(); ++i)
    if (points[i].x != x0 || points[i].y != y0) {
      first = i;
      break;
    }
  if (first == points.size())
    return false;
  __int128 dx1 = points[first].x - x0;
  __int128 dy1 = points[first].y - y0;
  __int128 dv1 = static_cast<__int128>(values[first]) - value0;
  __int128 x_step = 0, y_step = 0;
  bool solved = false;
  for (size_t i = first + 1; i < points.size(); ++i) {
    __int128 dx2 = points[i].x - x0;
    __int128 dy2 = points[i].y - y0;
    __int128 determinant = dx1 * dy2 - dx2 * dy1;
    if (!determinant)
      continue;
    __int128 dv2 = static_cast<__int128>(values[i]) - value0;
    __int128 x_numerator = dv1 * dy2 - dv2 * dy1;
    __int128 y_numerator = dx1 * dv2 - dx2 * dv1;
    if (x_numerator % determinant || y_numerator % determinant)
      return false;
    x_step = x_numerator / determinant;
    y_step = y_numerator / determinant;
    solved = true;
    break;
  }
  if (!solved) {
    int64_t dx = static_cast<int64_t>(dx1);
    int64_t dy = static_cast<int64_t>(dy1);
    int64_t abs_x = std::abs(dx), abs_y = std::abs(dy);
    int64_t coefficient_x = 0, coefficient_y = 0;
    int64_t gcd = ExtendedGCD(abs_x, abs_y, &coefficient_x, &coefficient_y);
    if (!gcd || dv1 % gcd)
      return false;
    if (dx < 0)
      coefficient_x = -coefficient_x;
    if (dy < 0)
      coefficient_y = -coefficient_y;
    x_step = static_cast<__int128>(coefficient_x) * (dv1 / gcd);
    y_step = static_cast<__int128>(coefficient_y) * (dv1 / gcd);
  }
  __int128 base = static_cast<__int128>(value0) - x_step * x0 - y_step * y0;
  if (!Fits(base, dtype) || !Fits(x_step, dtype) || !Fits(y_step, dtype))
    return false;
  for (size_t i = 0; i < points.size(); ++i)
    if (base + x_step * points[i].x + y_step * points[i].y != values[i])
      return false;
  *result = {static_cast<int64_t>(base), static_cast<int64_t>(x_step),
             static_cast<int64_t>(y_step)};
  return true;
}

ffi::Optional<Stmt>
TryParameterizeCoreBody(const std::vector<DeviceCoreBody> &points) {
  if (points.empty())
    return std::nullopt;
  std::vector<StatementInfo> descriptions;
  descriptions.reserve(points.size());
  for (const DeviceCoreBody &point : points) {
    StatementInfo info;
    info.hash = Describe(point.body, &info.integers,
                         /*semantic_core_locals=*/true);
    if (!descriptions.empty() &&
        (info.hash != descriptions[0].hash ||
         !SameSkeleton(points[0].body, point.body,
                       /*semantic_core_locals=*/true) ||
         info.integers.size() != descriptions[0].integers.size()))
      return std::nullopt;
    descriptions.push_back(std::move(info));
  }
  std::vector<CoreAffineColumn> columns;
  columns.reserve(descriptions[0].integers.size());
  for (size_t column = 0; column < descriptions[0].integers.size(); ++column) {
    std::vector<int64_t> values;
    values.reserve(points.size());
    for (const StatementInfo &description : descriptions)
      values.push_back(description.integers[column]->value);
    CoreAffineColumn relation;
    if (!FitCoreAffine(points, values, descriptions[0].integers[column]->dtype,
                       &relation))
      return std::nullopt;
    columns.push_back(relation);
  }
  CoreAffineWriter writer(columns);
  Stmt result = Downcast<Stmt>(writer.Rewrite(points[0].body));
  if (!writer.complete())
    Invalid("Core parameterization traversed fewer integers than expected");
  return result;
}

struct CoreRectangle {
  int64_t begin_x;
  int64_t begin_y;
  int64_t end_x;
  int64_t end_y;
  std::vector<DeviceCoreBody> points;
};

std::vector<CoreRectangle>
PartitionRectangles(const std::vector<DeviceCoreBody> &bodies) {
  std::map<std::pair<int64_t, int64_t>, Stmt> remaining;
  for (const DeviceCoreBody &body : bodies)
    if (!remaining.emplace(std::make_pair(body.x, body.y), body.body).second)
      Invalid("duplicate Core body coordinate");
  std::vector<CoreRectangle> rectangles;
  while (!remaining.empty()) {
    int64_t begin_x = remaining.begin()->first.first;
    int64_t begin_y = remaining.begin()->first.second;
    int64_t end_y = begin_y + 1;
    while (remaining.count({begin_x, end_y}))
      ++end_y;
    int64_t end_x = begin_x + 1;
    for (;; ++end_x) {
      bool complete = true;
      for (int64_t y = begin_y; y < end_y; ++y)
        complete &= remaining.count({end_x, y}) != 0;
      if (!complete)
        break;
    }
    CoreRectangle rectangle{begin_x, begin_y, end_x, end_y, {}};
    for (int64_t x = begin_x; x < end_x; ++x)
      for (int64_t y = begin_y; y < end_y; ++y) {
        auto found = remaining.find({x, y});
        rectangle.points.push_back({x, y, found->second});
        remaining.erase(found);
      }
    rectangles.push_back(std::move(rectangle));
  }
  return rectangles;
}

PrimExpr CoreRangePredicate(const CoreRectangle &rectangle, int64_t grid_x,
                            int64_t grid_y) {
  PrimExpr x = Call(DataType::Int(32), tenstorrent::core_x(), {});
  PrimExpr y = Call(DataType::Int(32), tenstorrent::core_y(), {});
  PrimExpr x_active =
      rectangle.end_x == rectangle.begin_x + 1
          ? x == Integer(rectangle.begin_x)
          : x >= Integer(rectangle.begin_x) && x < Integer(rectangle.end_x);
  PrimExpr y_active =
      rectangle.end_y == rectangle.begin_y + 1
          ? y == Integer(rectangle.begin_y)
          : y >= Integer(rectangle.begin_y) && y < Integer(rectangle.end_y);
  bool full_x = rectangle.begin_x == 0 && rectangle.end_x == grid_x;
  bool full_y = rectangle.begin_y == 0 && rectangle.end_y == grid_y;
  if (full_x)
    return y_active;
  if (full_y)
    return x_active;
  return x_active && y_active;
}

void EmitParameterizedRectangle(const CoreRectangle &rectangle, int64_t grid_x,
                                int64_t grid_y, const ffi::String &owner_slot,
                                const Span &fallback_span,
                                const ffi::String &block_name,
                                ffi::Optional<Integer> work_group,
                                ffi::Array<Stmt> *result) {
  ffi::Optional<Stmt> parameterized = TryParameterizeCoreBody(rectangle.points);
  if (!parameterized.has_value() && rectangle.points.size() > 1) {
    bool split_x = rectangle.end_x - rectangle.begin_x >=
                   rectangle.end_y - rectangle.begin_y;
    int64_t middle = split_x ? (rectangle.begin_x + rectangle.end_x) / 2
                             : (rectangle.begin_y + rectangle.end_y) / 2;
    CoreRectangle first = rectangle, second = rectangle;
    first.points.clear();
    second.points.clear();
    if (split_x) {
      first.end_x = middle;
      second.begin_x = middle;
    } else {
      first.end_y = middle;
      second.begin_y = middle;
    }
    for (const DeviceCoreBody &point : rectangle.points)
      ((split_x ? point.x : point.y) < middle ? first.points : second.points)
          .push_back(point);
    if (!first.points.empty())
      EmitParameterizedRectangle(first, grid_x, grid_y, owner_slot,
                                 fallback_span, block_name, work_group, result);
    if (!second.points.empty())
      EmitParameterizedRectangle(second, grid_x, grid_y, owner_slot,
                                 fallback_span, block_name, work_group, result);
    return;
  }
  if (!parameterized.has_value())
    Invalid("singleton Core body could not be parameterized");
  Stmt body = CompactDeviceStatementsRecursive(parameterized.value());
  Span span = body->span.defined() ? body->span : fallback_span;
  CoreDomain domain(CoreCoord(rectangle.begin_x, rectangle.begin_y),
                    CoreCoord(rectangle.end_x, rectangle.end_y));
  ffi::Map<ffi::String, ffi::Any> annotations{{"tt.core_domain", domain},
                                              {"tt.owner_slot", owner_slot}};
  if (work_group.has_value())
    annotations.Set("tt.work_group", work_group.value());
  if (block_name == "tt_device_stream_work") {
    bool sends = false, receives = false;
    std::set<int64_t> sites;
    PostOrderVisit(body, [&](const ffi::ObjectRef &object) {
      const auto *node = object.as<CallNode>();
      if (!node)
        return;
      sends |= node->op.same_as(tenstorrent::dfb_pipe_send()) ||
               node->op.same_as(tenstorrent::dfb_pipe_send_record());
      receives |= node->op.same_as(tenstorrent::dfb_pipe_recv()) ||
                  node->op.same_as(tenstorrent::dfb_pipe_recv_record());
      auto site = node->annotations.Get("tt.communication_site");
      if (site.has_value()) {
        auto integer = site.value().as<Integer>();
        if (!integer.has_value())
          Invalid("communication site annotation must be an integer");
        sites.insert(integer.value()->value);
      }
    });
    if (sends && receives)
      Invalid("one communication role region cannot send and receive");
    if (sends || receives) {
      annotations.Set("tt.communication_role",
                      ffi::String(sends ? "source" : "destination"));
      ffi::Array<Integer> site_array;
      for (int64_t site : sites)
        site_array.push_back(Integer(site));
      if (site_array.empty())
        Invalid("communication role region has no communication site");
      annotations.Set("tt.communication_sites", site_array);
    }
  }
  SBlock work({}, {}, {}, block_name, body, std::nullopt, {}, {}, annotations,
              span);
  Stmt realize = SBlockRealize({}, const_true(), work, span);
  if (rectangle.begin_x == 0 && rectangle.begin_y == 0 &&
      rectangle.end_x == grid_x && rectangle.end_y == grid_y)
    result->push_back(realize);
  else
    result->push_back(IfThenElse(CoreRangePredicate(rectangle, grid_x, grid_y),
                                 realize, std::nullopt, span));
}

class AffineReader : public ObjectRewriter {
public:
  using Bindings =
      std::unordered_map<Var, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
  explicit AffineReader(const Bindings &bindings) : bindings_(bindings) {}

private:
  ffi::Optional<ffi::ObjectRef>
  RewriteLeaf(const ffi::ObjectRef &object) final {
    if (auto var = object.as<Var>()) {
      auto found = bindings_.find(var.value());
      if (found != bindings_.end())
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
    return 1;
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

static Stmt CompactDeviceStatementsImpl(const Stmt &body,
                                        bool semantic_core_locals) {
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
    if (!semantic_core_locals) {
      const auto *evaluate = statements[i].as<EvaluateNode>();
      if (!evaluate || !evaluate->value.as<CallNode>())
        return body;
    }
    StatementInfo item;
    item.hash = Describe(statements[i], &item.integers, semantic_core_locals);
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
            !SameSkeleton(statements[begin + j], statements[begin + length + j],
                          semantic_core_locals)) {
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
                            statements[begin + repeats * length + j],
                            semantic_core_locals)) {
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

Stmt CompactDeviceStatements(const Stmt &body) {
  return CompactDeviceStatementsImpl(body, /*semantic_core_locals=*/false);
}

namespace {
class NestedDeviceStatementCompactor : public StmtMutator {
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = ffi::GetRef<For>(op);
    Stmt body = CompactDeviceStatementsRecursive(op->body);
    if (body.same_as(op->body))
      return loop;
    loop.CopyOnWrite()->body = std::move(body);
    return loop;
  }
};
} // namespace

Stmt CompactDeviceStatementsRecursive(const Stmt &body) {
  Stmt compact = CompactDeviceStatements(body);
  NestedDeviceStatementCompactor nested;
  return nested(compact);
}

static Stmt CompactDeviceRoleStatements(const Stmt &body) {
  Stmt compact =
      CompactDeviceStatementsImpl(body, /*semantic_core_locals=*/true);
  NestedDeviceStatementCompactor nested;
  return nested(compact);
}

namespace {

Stmt ParameterizeDeviceCoreStatementsImpl(
    const std::vector<DeviceCoreBody> &bodies, int64_t grid_x, int64_t grid_y,
    const ffi::String &owner_slot, const Span &span,
    const ffi::String &block_name = "tt_device_core_work",
    ffi::Optional<Integer> work_group = std::nullopt) {
  if (bodies.empty())
    return Evaluate(Integer(0), span);
  if (grid_x <= 0 || grid_y <= 0 ||
      bodies.size() > static_cast<size_t>(grid_x * grid_y))
    Invalid("Core parameterization received an invalid launch domain");

  std::vector<std::vector<DeviceCoreBody>> skeletons;
  for (const DeviceCoreBody &body : bodies) {
    bool inserted = false;
    for (auto &group : skeletons) {
      if (SameSkeleton(group.front().body, body.body,
                       /*semantic_core_locals=*/true)) {
        group.push_back(body);
        inserted = true;
        break;
      }
    }
    if (!inserted)
      skeletons.push_back({body});
  }

  ffi::Array<Stmt> result;
  for (const auto &skeleton : skeletons)
    for (const CoreRectangle &rectangle : PartitionRectangles(skeleton))
      EmitParameterizedRectangle(rectangle, grid_x, grid_y, owner_slot, span,
                                 block_name, work_group, &result);
  if (result.empty())
    return Evaluate(Integer(0), span);
  return result.size() == 1 ? result[0] : SeqStmt(result, span);
}

std::vector<size_t> DFBOperandPositions(const Call &call) {
  if (call->op.same_as(tenstorrent::dfb_compute())) {
    std::vector<size_t> positions(call->args.size());
    std::iota(positions.begin(), positions.end(), 0);
    return positions;
  }
  if (call->op.same_as(tenstorrent::compute_value())) {
    std::vector<size_t> positions(call->args.size() > 0 ? call->args.size() - 1
                                                        : 0);
    std::iota(positions.begin(), positions.end(), 1);
    return positions;
  }
  if (call->op.same_as(tenstorrent::dfb_reserve()) ||
      call->op.same_as(tenstorrent::dfb_wait()) ||
      call->op.same_as(tenstorrent::dfb_copy_wait()) ||
      call->op.same_as(tenstorrent::dfb_release()) ||
      call->op.same_as(tenstorrent::dfb_load()))
    return {0};
  if (call->op.same_as(tenstorrent::dfb_pipe_send()) ||
      call->op.same_as(tenstorrent::dfb_pipe_recv()) ||
      call->op.same_as(tenstorrent::dfb_pipe_wait()) ||
      call->op.same_as(tenstorrent::tensor_to_dfb_nd()))
    return {1};
  if (call->op.same_as(tenstorrent::dfb_pipe_send_record()) ||
      call->op.same_as(tenstorrent::dfb_pipe_recv_record()) ||
      call->op.same_as(tenstorrent::dfb_pipe_wait_record()))
    return {3};
  if (call->op.same_as(tenstorrent::dfb_to_tensor_nd()))
    return {0};
  if (call->op.same_as(tenstorrent::gemm_update()))
    return {0, 1};
  if (call->op.same_as(tenstorrent::accumulator_materialize()) ||
      call->op.same_as(tenstorrent::compute_value_store()))
    return {1};
  if (call->op.same_as(tenstorrent::compute_value_gemm()))
    return {1, 2};
  if (call->op.same_as(tenstorrent::dfb_local_copy()) ||
      call->op.same_as(tenstorrent::dfb_gather_copy()))
    return {0, 1};
  if (call->op.same_as(tenstorrent::dfb_collective_reduce()))
    return {0, 1, 2};
  return {};
}

bool StatementDFBs(const Stmt &statement, std::vector<int64_t> *ids) {
  const auto *evaluate = statement.as<EvaluateNode>();
  if (!evaluate)
    return false;
  const auto *node = evaluate->value.as<CallNode>();
  if (!node)
    return is_zero(evaluate->value);
  Call call = ffi::GetRef<Call>(node);
  for (size_t position : DFBOperandPositions(call)) {
    if (position >= call->args.size())
      Invalid("Device DFB operation is missing an operand");
    const int64_t *value = as_const_int(call->args[position]);
    if (!value || *value < 0)
      return false;
    ids->push_back(*value);
  }
  return !ids->empty();
}

struct StatementSegment {
  int64_t component;
  Stmt body;
};

ffi::Optional<Stmt>
TryParameterizeDFBComponents(const std::vector<DeviceCoreBody> &bodies,
                             int64_t grid_x, int64_t grid_y,
                             const ffi::String &owner_slot, const Span &span) {
  if (owner_slot == "trisc" ||
      bodies.size() != static_cast<size_t>(grid_x * grid_y))
    return std::nullopt;

  std::map<int64_t, int64_t> parent;
  std::vector<std::vector<Stmt>> statements;
  auto find = [&](auto &&self, int64_t value) -> int64_t {
    auto found = parent.find(value);
    if (found == parent.end())
      return parent.emplace(value, value).first->second;
    if (found->second != value)
      found->second = self(self, found->second);
    return found->second;
  };
  auto unite = [&](int64_t left, int64_t right) {
    left = find(find, left);
    right = find(find, right);
    if (left == right)
      return;
    if (left > right)
      std::swap(left, right);
    parent[right] = left;
  };

  for (const DeviceCoreBody &core : bodies) {
    ffi::Array<Stmt> flat;
    if (auto sequence = core.body.as<SeqStmt>())
      flat = sequence.value()->seq;
    else
      flat = {core.body};
    std::vector<Stmt> local;
    for (const Stmt &statement : flat) {
      std::vector<int64_t> ids;
      if (!StatementDFBs(statement, &ids))
        return std::nullopt;
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      for (int64_t id : ids)
        find(find, id);
      for (size_t index = 1; index < ids.size(); ++index)
        unite(ids[0], ids[index]);
      local.push_back(statement);
    }
    statements.push_back(std::move(local));
  }
  std::set<int64_t> components;
  for (const auto &[id, _] : parent)
    components.insert(find(find, id));
  if (components.size() < 2)
    return std::nullopt;

  std::vector<std::vector<StatementSegment>> segmented;
  for (const auto &local : statements) {
    std::vector<StatementSegment> segments;
    for (const Stmt &statement : local) {
      std::vector<int64_t> ids;
      if (!StatementDFBs(statement, &ids) || ids.empty())
        return std::nullopt;
      int64_t component = find(find, ids[0]);
      for (int64_t id : ids)
        if (find(find, id) != component)
          Invalid("one Device statement spans disconnected DFB components");
      if (!segments.empty() && segments.back().component == component) {
        segments.back().body =
            SeqStmt::Flatten(segments.back().body, statement);
      } else {
        segments.push_back({component, statement});
      }
    }
    if (segments.empty())
      return std::nullopt;
    if (!segmented.empty()) {
      if (segments.size() != segmented.front().size())
        return std::nullopt;
      for (size_t index = 0; index < segments.size(); ++index)
        if (segments[index].component != segmented.front()[index].component)
          return std::nullopt;
    }
    segmented.push_back(std::move(segments));
  }
  if (segmented.front().size() < 2)
    return std::nullopt;

  ffi::Array<Stmt> ordered;
  for (size_t segment = 0; segment < segmented.front().size(); ++segment) {
    std::vector<DeviceCoreBody> points;
    points.reserve(bodies.size());
    for (size_t core = 0; core < bodies.size(); ++core)
      points.push_back(
          {bodies[core].x, bodies[core].y, segmented[core][segment].body});
    Stmt role = ParameterizeDeviceCoreStatementsImpl(
        points, grid_x, grid_y, owner_slot, span, "tt_device_stream_work",
        Integer(segmented.front()[segment].component));
    if (auto sequence = role.as<SeqStmt>()) {
      for (const Stmt &statement : sequence.value()->seq)
        ordered.push_back(statement);
    } else {
      ordered.push_back(role);
    }
  }
  Stmt scheduled = ordered.size() == 1 ? ordered[0] : SeqStmt(ordered, span);
  scheduled = CompactDeviceRoleStatements(scheduled);
  CoreDomain domain(CoreCoord(0, 0), CoreCoord(grid_x, grid_y));
  SBlock work({}, {}, {}, "tt_device_core_work", scheduled, std::nullopt, {},
              {}, {{"tt.core_domain", domain}, {"tt.owner_slot", owner_slot}},
              span);
  return SBlockRealize({}, const_true(), work, span);
}

} // namespace

Stmt ParameterizeDeviceCoreStatements(const std::vector<DeviceCoreBody> &bodies,
                                      int64_t grid_x, int64_t grid_y,
                                      const ffi::String &owner_slot,
                                      const Span &span) {
  if (auto components = TryParameterizeDFBComponents(bodies, grid_x, grid_y,
                                                     owner_slot, span))
    return components.value();
  return ParameterizeDeviceCoreStatementsImpl(bodies, grid_x, grid_y,
                                              owner_slot, span);
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

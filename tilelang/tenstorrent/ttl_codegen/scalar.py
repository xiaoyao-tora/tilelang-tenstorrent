"""TIR scalar and Core-coordinate expression emission."""

from __future__ import annotations

from tvm import tirx


class ScalarEmitter:
    def __init__(self, function_emitter) -> None:
        self.function = function_emitter
        self.bindings = function_emitter.bindings
        self.context = function_emitter.context

    def index_constant(self, value: int):
        return self.bindings.arith.ConstantOp(self.bindings.ir.IndexType.get(self.context), int(value)).result

    def emit(self, expression):
        arith = self.bindings.arith
        if isinstance(expression, tirx.IntImm):
            return self.index_constant(int(expression))
        if isinstance(expression, tirx.Var):
            try:
                return self.function.scalar_values[expression]
            except KeyError as error:
                raise RuntimeError(f"unbound Device scalar {expression.name}") from error
        if isinstance(expression, tirx.Cast):
            value = self.emit(expression.value)
            if str(expression.dtype) in ("int32", "int64", "uint32", "uint64"):
                return value
            raise NotImplementedError(f"TTL scalar cast to {expression.dtype} is not supported")
        if isinstance(expression, tirx.Call) and hasattr(expression.op, "name"):
            name = str(expression.op.name)
            if name == "tl.tt.core_x":
                return self.bindings.ttl.core_x()
            if name == "tl.tt.core_y":
                return self.bindings.ttl.core_y()
            raise NotImplementedError(f"TTL scalar expression has no mapping for {name}")
        binary = {
            tirx.Add: arith.addi,
            tirx.Sub: arith.subi,
            tirx.Mul: arith.muli,
            tirx.FloorDiv: arith.floordivsi,
            tirx.FloorMod: arith.remsi,
            tirx.Min: arith.minsi,
            tirx.Max: arith.maxsi,
            tirx.And: arith.andi,
            tirx.Or: arith.ori,
        }
        operation = binary.get(type(expression))
        if operation is not None:
            return operation(self.emit(expression.a), self.emit(expression.b))
        predicates = {
            tirx.EQ: arith.CmpIPredicate.eq,
            tirx.NE: arith.CmpIPredicate.ne,
            tirx.LT: arith.CmpIPredicate.slt,
            tirx.LE: arith.CmpIPredicate.sle,
            tirx.GT: arith.CmpIPredicate.sgt,
            tirx.GE: arith.CmpIPredicate.sge,
        }
        predicate = predicates.get(type(expression))
        if predicate is not None:
            return arith.cmpi(predicate, self.emit(expression.a), self.emit(expression.b))
        if isinstance(expression, tirx.Not):
            return arith.xori(
                self.emit(expression.a),
                arith.ConstantOp(self.bindings.ir.IntegerType.get_signless(1, self.context), 1).result,
            )
        if isinstance(expression, tirx.Select):
            return arith.select(
                self.emit(expression.condition),
                self.emit(expression.true_value),
                self.emit(expression.false_value),
            )
        raise NotImplementedError(f"TTL scalar expression has no mapping for {type(expression).__name__}: {expression}")


__all__ = ["ScalarEmitter"]

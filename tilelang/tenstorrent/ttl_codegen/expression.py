"""Typed block expressions for verified Device compute values."""

import math

from tvm import tirx


_BINARY = {tirx.Add: "add", tirx.Sub: "sub", tirx.Mul: "mul", tirx.Div: "div", tirx.Min: "min", tirx.Max: "max"}
_UNARY = {
    "tirx.exp": "exp",
    "tirx.exp2": "exp2",
    "tirx.log": "log",
    "tirx.log2": "log2",
    "tirx.sqrt": "sqrt",
    "tirx.rsqrt": "rsqrt",
    "tirx.tanh": "tanh",
    "tirx.sin": "sin",
    "tirx.cos": "cos",
    "tirx.fabs": "abs",
    "tirx.floor": "floor",
    "tirx.ceil": "ceil",
}
_LOADS = ("tl.tt.dfb_load", "tl.tt.compute_value_load")


def check_expression(expr):
    """Check the complete expression before loading optional compiler bindings."""
    if isinstance(expr, (tirx.IntImm, tirx.FloatImm)):
        return
    if str(expr.dtype) not in ("bfloat16", "float32"):
        raise NotImplementedError(f"TTL block expression dtype {expr.dtype} is not supported")
    if isinstance(expr, tirx.Cast):
        check_expression(expr.value)
    elif type(expr) in _BINARY:
        check_expression(expr.a)
        check_expression(expr.b)
    elif isinstance(expr, tirx.Call) and expr.op.name in _LOADS:
        return
    elif isinstance(expr, tirx.Call) and expr.op.name in _UNARY:
        check_expression(expr.args[0])
    else:
        raise NotImplementedError(f"TTL block expression has no mapping for {type(expr).__name__}: {expr}")


class ExpressionEmitter:
    def __init__(self, emitter, call, result_type):
        self.e = emitter
        self.b = emitter.b
        self.shape = list(result_type.shape)
        self.result_type = result_type
        self.dtype = str(call.annotations["tt.compute_dtype"].value)
        self.maps = {
            ("tl.tt.dfb_load", int(index)): list(map(int, axes))
            for index, axes in zip(call.args[1:], call.annotations.get("tt.access_maps", []))
        }
        self.maps.update(
            {
                ("tl.tt.compute_value_load", int(index)): list(map(int, axes))
                for index, axes in zip(call.annotations.get("tt.value_inputs", []), call.annotations.get("tt.value_access_maps", []))
            }
        )

    def tensor_type(self, dtype):
        return self.b.ir.RankedTensorType.get(self.shape, self.e.tile_type(dtype, (32, 32)))

    def fill(self, value, dtype):
        attr = self.b.ir.FloatAttr.get(self.b.ir.F32Type.get(self.e.ctx), float(value))
        return self.b.ttl.fill(self.tensor_type(dtype), attr)

    def emit(self, expr, scalar_dtype=None):
        ttl = self.b.ttl
        if isinstance(expr, (tirx.IntImm, tirx.FloatImm)):
            return self.fill(expr.value, scalar_dtype or self.dtype)
        if isinstance(expr, tirx.Cast):
            # Constants are materialized directly in the explicitly requested dtype.
            if isinstance(expr.value, (tirx.IntImm, tirx.FloatImm)):
                return self.fill(expr.value.value, expr.dtype)
            return ttl.typecast(self.tensor_type(expr.dtype), self.emit(expr.value))
        if type(expr) in _BINARY:
            return getattr(ttl, _BINARY[type(expr)])(self.emit(expr.a, expr.dtype), self.emit(expr.b, expr.dtype))
        if isinstance(expr, tirx.Call) and expr.op.name in _LOADS:
            index = int(expr.args[0])
            value = self.e.values[index] if expr.op.name == "tl.tt.dfb_load" else self.e.compute_values[index]
            axes = self.maps[(expr.op.name, index)]
            dims = [axis for axis, source in enumerate(axes) if source == -1]
            if dims:
                dims_attr = self.b.ir.DenseI64ArrayAttr.get(dims)
                shape_attr = self.b.ir.DenseI64ArrayAttr.get(self.shape)
                value = ttl.block_broadcast(self.tensor_type(expr.dtype), value, dims_attr, shape_attr)
            return value
        operation = _UNARY[expr.op.name]
        value = self.emit(expr.args[0])
        if operation == "log2":
            scale = self.b.ir.FloatAttr.get(self.b.ir.F32Type.get(self.e.ctx), 1.0 / math.log(2.0))
            return ttl.mul_unary_const(ttl.log(value), scale)
        return getattr(ttl, operation)(value)

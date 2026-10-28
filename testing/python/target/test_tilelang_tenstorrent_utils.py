"""CPU numerical reference models, not a TT runtime or hardware simulator.

The value model preserves immutable snapshots. The composition model additionally
checks bounded DFB storage, asynchronous copies and randomized stream scheduling.
BF16 uses FP32 storage with round-to-nearest, ties-to-even conversion.
"""

import numpy as np
from tilelang import tvm
from tilelang.tenstorrent import device_ir as D
from tvm import tirx

TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})


def _specialize_expr(expr, bindings, core):
    expr = tirx.stmt_functor.substitute(expr, bindings)
    if core is not None:

        def rewrite(node):
            if isinstance(node, tirx.Call) and node.op.name == "tl.tt.core_x":
                return tirx.IntImm(node.dtype, core[0])
            if isinstance(node, tirx.Call) and node.op.name == "tl.tt.core_y":
                return tirx.IntImm(node.dtype, core[1])
            return None

        expr = tirx.stmt_functor.ir_transform(
            tirx.Evaluate(expr), rewrite, None, ["tirx.Call"]
        ).value
    return tvm.arith.Analyzer().simplify(expr)


def _specialize_call(call, bindings, core):
    args = [_specialize_expr(arg, bindings, core) for arg in call.args]
    annotations = {key: value for key, value in call.annotations.items()}
    if "tt.value_inputs" in annotations:
        annotations["tt.value_inputs"] = [
            _specialize_expr(value, bindings, core)
            for value in annotations["tt.value_inputs"]
        ]
    if "tt.expression" in annotations:
        annotations["tt.expression"] = _specialize_expr(
            annotations["tt.expression"], bindings, core
        )
    return tirx.Call(call.dtype, call.op, args, annotations, call.span)


def _calls(body, bindings=None, core=None):
    bindings = {} if bindings is None else bindings
    if isinstance(body, tirx.SeqStmt):
        for stmt in body.seq:
            yield from _calls(stmt, bindings, core)
    elif isinstance(body, tirx.Evaluate) and isinstance(body.value, tirx.Call):
        yield _specialize_call(body.value, bindings, core)
    elif isinstance(body, tirx.Evaluate) and int(body.value) == 0:
        return
    elif isinstance(body, tirx.IfThenElse):
        yield from _calls(body.then_case, bindings, core)
        if body.else_case is not None:
            yield from _calls(body.else_case, bindings, core)
    elif isinstance(body, tirx.SBlockRealize):
        domain = body.block.annotations.get("tt.core_domain")
        if domain is not None and core is None:
            for x in range(int(domain.begin.x), int(domain.end.x)):
                for y in range(int(domain.begin.y), int(domain.end.y)):
                    yield from _calls(body.block.body, bindings, (x, y))
        else:
            yield from _calls(body.block.body, bindings, core)
    elif isinstance(body, tirx.For):
        minimum = int(body.min)
        step = int(body.step) if body.step is not None else 1
        for offset in range(int(body.extent)):
            iteration = tirx.IntImm(body.loop_var.dtype, minimum + offset * step)
            nested = dict(bindings)
            nested[body.loop_var] = iteration
            yield from _calls(body.body, nested, core)
    else:
        raise AssertionError(f"Unexpected Device statement: {body}")


def _core_statements(func):
    dispatch = func.body
    if isinstance(dispatch, tirx.SeqStmt):
        assert len(dispatch.seq) == 1
        dispatch = dispatch.seq[0]
    realize = dispatch.then_case if isinstance(dispatch, tirx.IfThenElse) else dispatch
    assert isinstance(realize, tirx.SBlockRealize)
    body = realize.block.body
    return list(body.seq) if isinstance(body, tirx.SeqStmt) else [body]


def _with_core_statements(func, statements):
    dispatch = func.body
    realize = dispatch.then_case if isinstance(dispatch, tirx.IfThenElse) else dispatch
    assert isinstance(realize, tirx.SBlockRealize)
    block = realize.block
    body = tirx.SeqStmt(statements) if len(statements) != 1 else statements[0]
    updated = tirx.SBlock(
        block.iter_vars,
        block.reads,
        block.writes,
        block.name_hint,
        body,
        init=block.init,
        alloc_buffers=block.alloc_buffers,
        match_buffers=block.match_buffers,
        annotations=block.annotations,
    )
    realize = tirx.SBlockRealize(realize.iter_values, realize.predicate, updated)
    if isinstance(dispatch, tirx.IfThenElse):
        return func.with_body(tirx.IfThenElse(dispatch.condition, realize, dispatch.else_case))
    return func.with_body(realize)


def _round(value, dtype):
    if str(dtype) == "bfloat16":
        value = np.asarray(value, dtype=np.float32)
        bits = value.view(np.uint32)
        bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
        return bits.view(np.float32)
    target = np.dtype(str(dtype))
    if target.kind in "iub" or str(dtype) in ("float16", "float32", "float64"):
        return np.asarray(value).astype(target)
    raise AssertionError(f"Unsupported interpreter dtype: {dtype}")


def interpret_values(mod, tensors):
    """Run streams with waits, preserving snapshots after input DFB release."""
    mod = D.expand_current_device_tables(mod)
    arrays = [np.array(value, copy=True) for value in tensors]
    dfbs, values = {}, {}
    descriptors = {int(value.value_id): value for value in mod.attrs["tt.compute_value_table"]}
    dfb_descriptors = {int(dfb.dfb_id): dfb for dfb in mod.attrs["tt.dfb_table"]}
    streams = [list(_calls(func.body)) for func in mod.functions.values()]
    cursors = [0] * len(streams)

    def expression(expr, maps):
        if isinstance(expr, (tirx.FloatImm, tirx.IntImm)):
            return _round(expr.value, expr.dtype)
        if isinstance(expr, tirx.Cast):
            return _round(expression(expr.value, maps), expr.dtype)
        if isinstance(expr, tirx.Select):
            return _round(
                np.where(expression(expr.condition, maps), expression(expr.true_value, maps), expression(expr.false_value, maps)),
                expr.dtype,
            )
        if isinstance(expr, tirx.Not):
            return np.logical_not(expression(expr.a, maps))
        if isinstance(expr, tirx.Call):
            name = expr.op.name
            if name in ("tl.tt.dfb_load", "tl.tt.compute_value_load"):
                identifier = int(expr.args[0])
                source = dfbs[identifier] if name == "tl.tt.dfb_load" else values[identifier]
                axes = maps[(name, identifier)]
                source = source[tuple(slice(0, 1) if axis == -1 else slice(None) for axis in axes)]
                return source
            unary = {
                "tirx.exp": np.exp,
                "tirx.exp2": np.exp2,
                "tirx.log": np.log,
                "tirx.log2": np.log2,
                "tirx.sqrt": np.sqrt,
                "tirx.rsqrt": lambda value: 1 / np.sqrt(value),
                "tirx.abs": np.abs,
            }
            assert name in unary, name
            return _round(unary[name](expression(expr.args[0], maps)), expr.dtype)
        binary = {
            tirx.Add: np.add,
            tirx.Sub: np.subtract,
            tirx.Mul: np.multiply,
            tirx.Div: np.divide,
            tirx.Min: np.minimum,
            tirx.Max: np.maximum,
            tirx.EQ: np.equal,
            tirx.NE: np.not_equal,
            tirx.LT: np.less,
            tirx.LE: np.less_equal,
            tirx.GT: np.greater,
            tirx.GE: np.greater_equal,
            tirx.And: np.logical_and,
            tirx.Or: np.logical_or,
        }
        for cls, operation in binary.items():
            if isinstance(expr, cls):
                result = operation(expression(expr.a, maps), expression(expr.b, maps))
                return result if str(expr.dtype) == "bool" else _round(result, expr.dtype)
        raise AssertionError(f"Unexpected expression: {expr}")

    for _ in range(10000):
        if all(cursor == len(stream) for cursor, stream in zip(cursors, streams)):
            return arrays, values
        advanced = False
        for index, stream in enumerate(streams):
            if cursors[index] == len(stream):
                continue
            call = stream[cursors[index]]
            name, args = call.op.name, list(map(int, call.args))
            if name == "tl.tt.dfb_wait" and args[0] not in dfbs:
                continue
            if name == "tl.tt.dfb_to_tensor_nd" and args[0] not in dfbs:
                continue
            if name == "tl.tt.tensor_to_dfb_nd":
                region = tuple(slice(args[axis], args[axis] + args[axis + 1]) for axis in range(2, len(args), 2))
                dfbs[args[1]] = arrays[args[0]][region].copy()
            elif name == "tl.tt.dfb_to_tensor_nd":
                region = tuple(slice(args[axis], args[axis] + args[axis + 1]) for axis in range(2, len(args), 2))
                arrays[args[1]][region] = dfbs[args[0]]
            elif name in ("tl.tt.compute_value", "tl.tt.dfb_compute"):
                attrs = call.annotations
                maps = {
                    ("tl.tt.dfb_load", identifier): tuple(map(int, axes))
                    for identifier, axes in zip(args[1:], attrs.get("tt.access_maps", []))
                }
                maps.update(
                    {
                        ("tl.tt.compute_value_load", int(identifier)): tuple(map(int, axes))
                        for identifier, axes in zip(attrs.get("tt.value_inputs", []), attrs.get("tt.value_access_maps", []))
                    }
                )
                kind = str(attrs["tt.compute_kind"].value)
                result = dfbs[args[1]] if kind == "copy" else expression(attrs["tt.expression"], maps)
                shape = tuple(map(int, attrs["tt.logical_domain"]))
                result = np.broadcast_to(result, shape).copy()
                if name == "tl.tt.compute_value":
                    assert args[0] not in values, "An immutable value was redefined"
                    values[args[0]] = _round(result, descriptors[args[0]].buffer.dtype)
                else:
                    dfbs[args[0]] = _round(result, dfb_descriptors[args[0]].element_dtype)
            elif name == "tl.tt.compute_value_gemm":
                output, lhs, rhs, old, ta, tb = args
                a, b = dfbs[lhs], dfbs[rhs]
                result = (a.T if ta else a) @ (b.T if tb else b)
                if old >= 0:
                    result = result + values[old]
                assert output not in values
                values[output] = _round(result, descriptors[output].buffer.dtype)
            elif name == "tl.tt.compute_value_store":
                value_dtype = str(descriptors[args[0]].buffer.dtype)
                dfb_dtype = str(dfb_descriptors[args[1]].element_dtype)
                assert value_dtype == dfb_dtype, "compute_value_store must preserve dtype"
                dfbs[args[1]] = values[args[0]].copy()
            elif name == "tl.tt.dfb_release":
                del dfbs[args[0]]
            else:
                assert name in ("tl.tt.dfb_reserve", "tl.tt.dfb_wait", "tl.tt.dfb_copy_wait"), name
            cursors[index] += 1
            advanced = True
        assert advanced, "Device streams deadlocked"
    raise AssertionError("Device interpreter exceeded its instruction bound")


def interpret(mod, inputs, seed=0):
    """Execute value, DFB, and point-delivery contracts independently of lowering."""
    mod = D.expand_current_device_tables(mod)
    tensors = [np.array(value, copy=True) for value in inputs]
    values, dfbs, deliveries, accumulators = {}, {}, {}, {}
    pending = {}
    occupied = set()
    groups = {int(k): int(v) for k, v in mod.attrs.get("tt.dfb_storage_groups", {}).items()}
    accumulator_info = {int(a.accumulator_id): a for a in mod.attrs.get("tt.accumulator_table", [])}
    random = np.random.default_rng(seed)
    value_info = {int(value.value_id): value for value in mod.attrs.get("tt.compute_value_table", [])}
    dfb_info = {int(dfb.dfb_id): dfb for dfb in mod.attrs["tt.dfb_table"]}
    streams = [list(_calls(func.body)) for func in mod.functions.values()]
    explicit_release = any(call.op.name == "tl.tt.dfb_release" for stream in streams for call in stream)
    cursors = [0] * len(streams)
    value_owners = {
        int(call.args[0]): index
        for index, stream in enumerate(streams)
        for call in stream
        if call.op.name in ("tl.tt.compute_value", "tl.tt.compute_value_gemm")
    }
    precision = {}
    asynchronous = {int(call.args[0]) for stream in streams for call in stream if call.op.name == "tl.tt.dfb_copy_wait"}

    def expression(expr, maps):
        if isinstance(expr, (tirx.IntImm, tirx.FloatImm)):
            return _round(expr.value, expr.dtype)
        if isinstance(expr, tirx.Cast):
            return _round(expression(expr.value, maps), expr.dtype)
        if isinstance(expr, tirx.Call):
            if expr.op.name in ("tl.tt.dfb_load", "tl.tt.compute_value_load"):
                identifier = int(expr.args[0])
                source = dfbs[identifier] if expr.op.name == "tl.tt.dfb_load" else values[identifier]
                axes = maps[(expr.op.name, identifier)]
                return source[tuple(slice(0, 1) if axis == -1 else slice(None) for axis in axes)]
            if expr.op.name == "tirx.exp2":
                return _round(np.exp2(expression(expr.args[0], maps)), expr.dtype)
            raise AssertionError(expr.op.name)
        binary = {tirx.Add: np.add, tirx.Sub: np.subtract, tirx.Mul: np.multiply, tirx.Div: np.divide, tirx.Max: np.maximum}
        return _round(binary[type(expr)](expression(expr.a, maps), expression(expr.b, maps)), expr.dtype)

    for _ in range(10000):
        if all(cursor == len(stream) for cursor, stream in zip(cursors, streams)):
            assert not pending, "Copies remain incomplete"
            if explicit_release:
                assert not occupied, "DFB reservations remain live"
            return tensors, values
        advanced = False
        for index in random.permutation(len(streams)):
            stream = streams[index]
            if cursors[index] == len(stream):
                continue
            call = stream[cursors[index]]
            name, args = call.op.name, list(map(int, call.args))
            if name == "tl.tt.compute_precision":
                assert args[0] in (0, 1)
                precision[index] = args[0]
                # Compute-local values cannot survive a destination precision
                # transition. Only explicit DFB snapshots can reload them.
                for identifier in list(values):
                    if value_owners[identifier] == index:
                        del values[identifier]
            if name == "tl.tt.dfb_wait" and args[0] not in dfbs:
                continue
            if name == "tl.tt.dfb_pipe_recv" and args[0] not in deliveries:
                continue
            if name == "tl.tt.dfb_reserve":
                group = groups.get(args[0], args[0])
                if sum(groups.get(resource, resource) == group for resource in occupied) >= int(dfb_info[args[0]].block_count):
                    continue
                assert args[0] not in occupied
                occupied.add(args[0])
            elif name == "tl.tt.tensor_to_dfb_nd":
                region = tuple(slice(args[i], args[i] + args[i + 1]) for i in range(2, len(args), 2))
                value = tensors[args[0]][region].copy()
                local_rank = len(dfb_info[args[1]].block_shape_in_tiles)
                while value.ndim > local_rank:
                    assert value.shape[0] == 1
                    value = value[0]
                if args[1] in asynchronous:
                    pending[args[1]] = ("load", value)
                else:
                    dfbs[args[1]] = value
            elif name == "tl.tt.dfb_to_tensor_nd":
                region = tuple(slice(args[i], args[i] + args[i + 1]) for i in range(2, len(args), 2))
                if args[0] in asynchronous:
                    pending[args[0]] = ("store", args[1], region, dfbs[args[0]].copy())
                else:
                    tensors[args[1]][region] = dfbs[args[0]]
            elif name == "tl.tt.dfb_copy_wait":
                action = pending.pop(args[0])
                if action[0] == "load":
                    dfbs[args[0]] = action[1]
                else:
                    tensors[action[1]][action[2]] = action[3]
            elif name == "tl.tt.dfb_pipe_send":
                deliveries[args[0]] = dfbs[args[1]].copy()
            elif name == "tl.tt.dfb_pipe_recv":
                dfbs[args[1]] = deliveries[args[0]].copy()
            elif name in ("tl.tt.compute_value", "tl.tt.dfb_compute"):
                attrs = call.annotations
                kind = str(attrs["tt.compute_kind"].value)
                domain = tuple(map(int, attrs["tt.logical_domain"]))
                if kind == "transpose":
                    result = dfbs[args[1]].swapaxes(-1, -2)
                elif kind == "reduce":
                    reducer = {"sum": np.sum, "max": np.max, "min": np.min}[str(attrs["tt.reduce_kind"].value)]
                    result = reducer(dfbs[args[1]], axis=int(attrs["tt.reduce_axis"]), keepdims=len(domain) == dfbs[args[1]].ndim)
                elif kind == "copy":
                    result = dfbs[args[1]]
                else:
                    maps = {
                        ("tl.tt.dfb_load", identifier): tuple(map(int, axes))
                        for identifier, axes in zip(args[1:], attrs.get("tt.access_maps", []))
                    }
                    maps.update(
                        {
                            ("tl.tt.compute_value_load", int(identifier)): tuple(map(int, axes))
                            for identifier, axes in zip(attrs.get("tt.value_inputs", []), attrs.get("tt.value_access_maps", []))
                        }
                    )
                    result = expression(attrs["tt.expression"], maps)
                result = np.broadcast_to(result, domain).copy()
                if name == "tl.tt.compute_value":
                    assert args[0] not in values
                    values[args[0]] = _round(result, value_info[args[0]].buffer.dtype)
                else:
                    dfbs[args[0]] = _round(result, dfb_info[args[0]].element_dtype)
            elif name == "tl.tt.compute_value_gemm":
                output, lhs, rhs, old, ta, tb = args
                if index in precision:
                    assert precision[index] == (1 if value_info[output].buffer.dtype == "float32" else 0)
                a, b = dfbs[lhs], dfbs[rhs]
                result = (a.T if ta else a) @ (b.T if tb else b)
                if old >= 0:
                    result = result + values[old]
                values[output] = _round(result, value_info[output].buffer.dtype)
            elif name == "tl.tt.accumulator_init":
                info = accumulator_info[args[0]]
                accumulators[args[0]] = np.zeros(tuple(int(r.extent) for r in info.accumulator_region.region), "float32")
            elif name == "tl.tt.gemm_update":
                lhs, rhs, identifier, ta, tb = args
                a, b = dfbs[lhs], dfbs[rhs]
                result = accumulators[identifier] + (a.T if ta else a) @ (b.T if tb else b)
                accumulators[identifier] = _round(result, accumulator_info[identifier].accumulation_dtype)
            elif name == "tl.tt.accumulator_materialize":
                dfbs[args[1]] = _round(accumulators.pop(args[0]), dfb_info[args[1]].element_dtype)
            elif name == "tl.tt.compute_value_store":
                value_dtype = str(value_info[args[0]].buffer.dtype)
                dfb_dtype = str(dfb_info[args[1]].element_dtype)
                assert value_dtype == dfb_dtype, "compute_value_store must preserve dtype"
                dfbs[args[1]] = values[args[0]].copy()
            elif name == "tl.tt.dfb_release":
                assert args[0] not in pending, "Release before copy completion"
                occupied.remove(args[0])
                del dfbs[args[0]]
            else:
                assert name in (
                    "tl.tt.dfb_reserve",
                    "tl.tt.dfb_wait",
                    "tl.tt.dfb_copy_wait",
                    "tl.tt.dfb_pipe_wait",
                    "tl.tt.compute_precision",
                ), name
            cursors[index] += 1
            advanced = True
        assert advanced, "Device streams deadlocked"
    raise AssertionError("Device stream instruction bound exceeded")

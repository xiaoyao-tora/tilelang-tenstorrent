"""Numerical validation of immutable compute values without a TT device.

This interpreter executes the Device streams and expression contract against
NumPy. The expected results are independent whole-array frontend formulas.
"""

import numpy as np
from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import lower_tenstorrent_ir
from tvm import tirx

TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
METADATA = {"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2}


def _round(value, dtype):
    value = np.asarray(value, dtype=np.float32)
    if str(dtype) == "bfloat16":
        bits = value.view(np.uint32)
        bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
        return bits.view(np.float32)
    return value


def _calls(body):
    if isinstance(body, tirx.SeqStmt):
        for stmt in body.seq:
            yield from _calls(stmt)
    elif isinstance(body, tirx.Evaluate) and isinstance(body.value, tirx.Call):
        yield body.value
    elif isinstance(body, tirx.Evaluate) and int(body.value) == 0:
        return
    else:
        raise AssertionError(f"Unexpected Device statement: {body}")


def interpret_values(mod, tensors):
    """Run streams with waits, preserving snapshots after input DFB release."""
    arrays = [np.array(value, copy=True) for value in tensors]
    dfbs, values = {}, {}
    descriptors = {int(value.value_id): value for value in mod.attrs["tt.compute_value_table"]}
    dfb_descriptors = {int(dfb.dfb_id): dfb for dfb in mod.attrs["tt.dfb_table"]}
    streams = [list(_calls(func.body)) for func in mod.functions.values()]
    cursors = [0] * len(streams)

    def expression(expr, maps):
        if isinstance(expr, (tirx.FloatImm, tirx.IntImm)):
            return np.float32(expr.value)
        if isinstance(expr, tirx.Cast):
            return _round(expression(expr.value, maps), expr.dtype)
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
        }
        for cls, operation in binary.items():
            if isinstance(expr, cls):
                return _round(operation(expression(expr.a, maps), expression(expr.b, maps)), expr.dtype)
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
                dfbs[args[1]] = values[args[0]].copy()
            elif name == "tl.tt.dfb_release":
                del dfbs[args[0]]
            else:
                assert name in ("tl.tt.dfb_reserve", "tl.tt.dfb_wait"), name
            cursors[index] += 1
            advanced = True
        assert advanced, "Device streams deadlocked"
    raise AssertionError("Device interpreter exceeded its instruction bound")


def value_chain():
    @T.prim_func
    def main(
        A: T.Tensor((64, 64), "float32"),
        B: T.Tensor((32, 64), "float32"),
        C: T.Tensor((64, 64), "float32"),
        D: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((64, 64), "float32", annotations=METADATA)
            b = T.alloc_shared((32, 64), "float32", annotations=METADATA)
            f = T.alloc_fragment((64, 64), "float32")
            old = T.alloc_fragment((64, 64), "float32")
            bf = T.alloc_fragment((32, 64), "float32")
            out = T.alloc_shared((64, 64), "float32", annotations=METADATA)
            T.copy(A, a)
            T.copy(B, b)
            for i, j in T.Tiles(bf):
                bf[i, j] = b[i, j]
            for i, j in T.Tiles(f):
                f[i, j] = a[i, j] * 2 + b[0, j]
            for i, j in T.Tiles(old):
                old[i, j] = f[i, j]
            for _ in T.serial(3):
                for i, j in T.Tiles(f):
                    f[i, j] = f[i, j] * T.float32(0.5) + old[i, j] + a[i, j] + bf[0, j]
            for i, j in T.Tiles(out):
                out[i, j] = f[i, j] + old[i, j]
            T.copy(out, C)
            T.copy(old, D)

    return main


def test_old_values_survive_updates_broadcast_loop_and_input_release():
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": value_chain()}), TARGET)
    assert int(mod.attrs["tt.device_ir_version"]) == 7
    rng = np.random.default_rng(12)
    a, b = rng.normal(size=(64, 64)).astype("float32"), rng.normal(size=(32, 64)).astype("float32")
    outputs, snapshots = interpret_values(mod, [a, b, np.zeros_like(a), np.zeros_like(a)])
    old = a * 2 + b[0:1, :]
    f = old.copy()
    for _ in range(3):
        f = f * 0.5 + old + a + b[0:1, :]
    np.testing.assert_allclose(outputs[2], f + old, rtol=1e-6)
    np.testing.assert_array_equal(outputs[3], old)
    versions = [value for value in mod.attrs["tt.compute_value_table"] if value.buffer.name == "f"]
    assert [int(value.version) for value in versions] == [0, 1, 2, 3]
    assert [int(value.previous_value_id) for value in versions[1:]] == [int(value.value_id) for value in versions[:-1]]
    np.testing.assert_array_equal(snapshots[int(versions[0].value_id)], old)
    assert "tl.tt.compute_value_store" not in mod.script(), "Pure SSA expressions need no provisional fragment DFB"
    compute = next(func for func in mod.functions.values() if str(func.attrs["tt.kernel_slot"]) == "trisc")
    calls = list(_calls(compute.body))
    borrowed = {int(arg) for call in calls if call.op.name == "tl.tt.compute_value" for arg in call.args[1:]}
    final_value_use = max(index for index, call in enumerate(calls) if len(call.annotations.get("tt.value_inputs", [])))
    for index, call in enumerate(calls):
        if call.op.name == "tl.tt.dfb_release" and int(call.args[0]) in borrowed:
            assert index > final_value_use, "An acquired DFB was released while descendant SSA still borrowed it"


def gemm_snapshots():
    @T.prim_func
    def main(
        A: T.Tensor((32, 64), "bfloat16"),
        B: T.Tensor((64, 32), "bfloat16"),
        C: T.Tensor((32, 32), "bfloat16"),
        D: T.Tensor((32, 32), "bfloat16"),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            b = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            acc = T.alloc_fragment((32, 32), "float32")
            old = T.alloc_fragment((32, 32), "float32")
            fresh = T.alloc_fragment((32, 32), "float32")
            T.clear(acc)
            for k in T.serial(2):
                T.copy(A[0:32, k * 32 : k * 32 + 32], a)
                T.copy(B[k * 32 : k * 32 + 32, 0:32], b)
                T.gemm(a, b, acc)
            for i, j in T.Tiles(old):
                old[i, j] = acc[i, j]
            for i, j in T.Tiles(acc):
                acc[i, j] = acc[i, j] * T.float32(0.5)
            for i, j in T.Tiles(fresh):
                fresh[i, j] = old[i, j] + acc[i, j]
            T.copy(fresh, C)
            T.copy(old, D)

    return main


def test_full_k_accumulator_fresh_epilogue_and_multiple_consumers():
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": gemm_snapshots()}), TARGET)
    rng = np.random.default_rng(31)
    a = _round(rng.normal(size=(32, 64)), "bfloat16")
    b = _round(rng.normal(size=(64, 32)), "bfloat16")
    outputs, _ = interpret_values(mod, [a, b, np.zeros((32, 32), "float32"), np.zeros((32, 32), "float32")])
    full = a[:, :32] @ b[:32, :] + a[:, 32:] @ b[32:, :]
    np.testing.assert_array_equal(outputs[2], _round(full + full * 0.5, "bfloat16"))
    np.testing.assert_array_equal(outputs[3], _round(full, "bfloat16"))
    (accumulator,) = mod.attrs["tt.accumulator_table"]
    assert str(accumulator.accumulation_dtype) == "float32"
    assert int(accumulator.full_k_tiles) == 2
    assert len([call for func in mod.functions.values() for call in _calls(func.body) if call.op.name == "tl.tt.compute_value_gemm"]) == 2
    assert all(str(value.buffer.dtype) == "float32" for value in mod.attrs["tt.compute_value_table"])


def gemm_fragment_operand():
    @T.prim_func
    def main(A: T.Tensor((32, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            b = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            operand = T.alloc_fragment((32, 32), "bfloat16")
            acc = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            for i, j in T.Tiles(operand):
                operand[i, j] = a[i, j]
            T.gemm(operand, b, acc, clear_accum=True)
            T.copy(acc, C)

    return main


def test_fragment_dfb_only_consumer_materializes_exactly_when_needed():
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": gemm_fragment_operand()}), TARGET)
    stores = [call for func in mod.functions.values() for call in _calls(func.body) if call.op.name == "tl.tt.compute_value_store"]
    assert len(stores) == 1
    descriptors = {int(value.value_id): value for value in mod.attrs["tt.compute_value_table"]}
    dfbs = {int(dfb.dfb_id): dfb for dfb in mod.attrs["tt.dfb_table"]}
    assert str(descriptors[int(stores[0].args[0])].buffer.dtype) == str(dfbs[int(stores[0].args[1])].element_dtype)
    rng = np.random.default_rng(9)
    a = _round(rng.normal(size=(32, 32)), "bfloat16")
    b = _round(rng.normal(size=(32, 32)), "bfloat16")
    outputs, _ = interpret_values(mod, [a, b, np.zeros((32, 32), "float32")])
    np.testing.assert_array_equal(outputs[2], _round(a @ b, "bfloat16"))


def independent_gemm_lifetimes():
    @T.prim_func
    def main(A: T.Tensor((64, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((64, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            b = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            acc = T.alloc_fragment((32, 32), "float32")
            T.copy(B, b)
            for batch in T.serial(2):
                T.copy(A[batch * 32 : batch * 32 + 32, 0:32], a)
                T.clear(acc)
                T.gemm(a, b, acc)
                for i, j in T.Tiles(acc):
                    acc[i, j] = acc[i, j] * T.float32(0.5)
                T.copy(acc, C[batch * 32 : batch * 32 + 32, 0:32])

    return main


def test_outer_serial_iterations_have_distinct_accumulator_lifetimes():
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": independent_gemm_lifetimes()}), TARGET)
    assert len(mod.attrs["tt.accumulator_table"]) == 2
    assert {int(value.accumulator_id) for value in mod.attrs["tt.compute_value_table"]} == {0, 1}
    rng = np.random.default_rng(71)
    a = _round(rng.normal(size=(64, 32)), "bfloat16")
    b = _round(rng.normal(size=(32, 32)), "bfloat16")
    outputs, _ = interpret_values(mod, [a, b, np.zeros_like(a)])
    np.testing.assert_array_equal(outputs[2], _round((a @ b) * 0.5, "bfloat16"))


def combined_gemm_roots():
    @T.prim_func
    def main(A: T.Tensor((32, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            b = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            first = T.alloc_fragment((32, 32), "float32")
            second = T.alloc_fragment((32, 32), "float32")
            combined = T.alloc_fragment((32, 32), "float32")
            epilogue = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, first, clear_accum=True)
            T.gemm(a, b, second, clear_accum=True)
            for i, j in T.Tiles(combined):
                combined[i, j] = first[i, j] * T.float32(0.25) + second[i, j] * T.float32(0.75)
            for i, j in T.Tiles(epilogue):
                epilogue[i, j] = combined[i, j] * T.float32(2) + first[i, j] * T.float32(0.5)
            T.copy(epilogue, C)

    return main


def test_two_accumulator_roots_combine_and_preserve_transitive_provenance():
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": combined_gemm_roots()}), TARGET)
    assert len(mod.attrs["tt.accumulator_table"]) == 2
    derived = [value for value in mod.attrs["tt.compute_value_table"] if value.buffer.name in ("combined", "epilogue")]
    assert len(derived) == 2
    assert all(int(value.accumulator_id) == -1 for value in derived)
    rng = np.random.default_rng(83)
    a = _round(rng.normal(size=(32, 32)), "bfloat16")
    b = _round(rng.normal(size=(32, 32)), "bfloat16")
    outputs, _ = interpret_values(mod, [a, b, np.zeros((32, 32), "float32")])
    full = a @ b
    np.testing.assert_array_equal(outputs[2], _round((full * 0.25 + full * 0.75) * 2 + full * 0.5, "bfloat16"))

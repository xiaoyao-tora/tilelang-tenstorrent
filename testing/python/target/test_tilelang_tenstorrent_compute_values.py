"""Numerical validation of immutable compute values without a TT device.

The shared CPU reference model executes Device streams against NumPy.
The expected results are independent whole-array frontend formulas.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from tilelang import tvm
from tilelang.tenstorrent import device_ir as D
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import lower_tenstorrent_ir
from tvm import tirx

import test_tilelang_tenstorrent_utils
from test_tilelang_tenstorrent_utils import TARGET, _calls, _round, interpret, interpret_values

METADATA = {"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2}


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
    mod = D.expand_current_device_tables(
        lower_tenstorrent_ir(tvm.IRModule({"main": value_chain()}), TARGET)
    )
    assert str(mod.attrs["tt.ir_stage"]) == "device"
    assert "tt.device_ir_version" not in mod.attrs
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
    mod = D.expand_current_device_tables(
        lower_tenstorrent_ir(tvm.IRModule({"main": gemm_snapshots()}), TARGET)
    )
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
    mod = D.expand_current_device_tables(
        lower_tenstorrent_ir(tvm.IRModule({"main": gemm_fragment_operand()}), TARGET)
    )
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
    mod = D.expand_current_device_tables(
        lower_tenstorrent_ir(tvm.IRModule({"main": independent_gemm_lifetimes()}), TARGET)
    )
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
    mod = D.expand_current_device_tables(
        lower_tenstorrent_ir(tvm.IRModule({"main": combined_gemm_roots()}), TARGET)
    )
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


def _interpreter_expression_module(expr, output_dtype):
    calls = [
        tirx.Call(
            "handle", tvm.ir.Op.get("tl.tt.compute_value"), [0],
            {"tt.compute_kind": tirx.StringImm("elementwise"), "tt.expression": expr,
             "tt.logical_domain": [1, 1]},
        ),
        tirx.Call("handle", tvm.ir.Op.get("tl.tt.compute_value_store"), [0, 0]),
        tirx.Call("handle", tvm.ir.Op.get("tl.tt.dfb_to_tensor_nd"), [0, 0, 0, 1, 0, 1]),
    ]
    return SimpleNamespace(
        attrs={
            "tt.compute_value_table": [SimpleNamespace(
                value_id=0, buffer=SimpleNamespace(dtype=expr.dtype))],
            "tt.dfb_table": [SimpleNamespace(dfb_id=0, element_dtype=output_dtype)],
        },
        functions={"compute": SimpleNamespace(
            body=tirx.SeqStmt([tirx.Evaluate(call) for call in calls]))},
    )


@pytest.mark.parametrize("dtype, inputs, expected", [
    ("int32", [1.75, -1.75, 0.25, -0.25], [1, -1, 0, 0]),
    ("int64", [2**40 + 1, -(2**40 + 1)], [2**40 + 1, -(2**40 + 1)]),
    ("uint32", [0, 2**24 + 1], [0, 2**24 + 1]),
    ("bool", [0, 1, -1], [False, True, True]),
    ("float16", [1.0001], [1.0]),
    ("float32", [1.0 + 2**-30], [1.0]),
    ("float64", [1.0 + 2**-30], [1.0 + 2**-30]),
])
def test_interpreter_dtype_casts(dtype, inputs, expected):
    actual = _round(inputs, dtype)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("expr, expected", [
    (tirx.Cast("int32", tirx.FloatImm("float32", 1.75)), 1),
    (tirx.Cast("int32", tirx.FloatImm("float32", -1.75)), -1),
    (tirx.Cast("bool", tirx.IntImm("int32", -1)), True),
    (tirx.Cast("bool", tirx.IntImm("int32", 0)), False),
    (tirx.IntImm("int64", 2**40 + 1), 2**40 + 1),
    (tirx.FloatImm("float64", 1.0 + 2**-30), 1.0 + 2**-30),
])
@pytest.mark.parametrize("interpreter", [interpret_values, interpret])
def test_interpreter_typed_expressions(monkeypatch, expr, expected, interpreter):
    monkeypatch.setattr(D, "expand_current_device_tables", lambda mod: mod)
    # Exercise interpreter Cast evaluation rather than TVM constant folding.
    monkeypatch.setattr(test_tilelang_tenstorrent_utils, "_specialize_expr", lambda expr, *args: expr)
    mod = _interpreter_expression_module(expr, expr.dtype)
    arrays, values = interpreter(mod, [np.zeros((1, 1), dtype=str(expr.dtype))])
    assert values[0].dtype == np.dtype(str(expr.dtype))
    assert values[0][0, 0] == expected
    assert arrays[0][0, 0] == expected


@pytest.mark.parametrize("value, expression_dtype, output_dtype, expected", [
    (1.00390625, "float32", "bfloat16", 1.0),
    (1.01171875, "float32", "bfloat16", 1.015625),
    (-1.00390625, "float32", "bfloat16", -1.0),
    (1.75, "float32", "int32", 1),
    (-1.75, "float32", "int32", -1),
    (1.0001, "float64", "float16", 1.0),
    (1.0 + 2**-30, "float64", "float64", 1.0 + 2**-30),
])
@pytest.mark.parametrize("interpreter", [interpret_values, interpret])
def test_interpreter_explicit_cast_materializes_exact_dtype(
    monkeypatch, value, expression_dtype, output_dtype, expected, interpreter,
):
    monkeypatch.setattr(D, "expand_current_device_tables", lambda mod: mod)
    expression = tirx.Cast(output_dtype, tirx.FloatImm(expression_dtype, value))
    mod = _interpreter_expression_module(expression, output_dtype)
    tensor_dtype = "float32" if output_dtype == "bfloat16" else output_dtype
    arrays, values = interpreter(mod, [np.zeros((1, 1), dtype=tensor_dtype)])
    expected_storage_dtype = "float32" if output_dtype == "bfloat16" else output_dtype
    assert values[0].dtype == np.dtype(expected_storage_dtype)
    assert values[0][0, 0] == expected
    assert arrays[0][0, 0] == expected


@pytest.mark.parametrize("interpreter", [interpret_values, interpret])
def test_interpreter_rejects_cross_dtype_compute_value_store(monkeypatch, interpreter):
    monkeypatch.setattr(D, "expand_current_device_tables", lambda mod: mod)
    mod = _interpreter_expression_module(tirx.FloatImm("float32", 1.5), "bfloat16")
    with pytest.raises(AssertionError, match="compute_value_store must preserve dtype"):
        interpreter(mod, [np.zeros((1, 1), dtype="float32")])

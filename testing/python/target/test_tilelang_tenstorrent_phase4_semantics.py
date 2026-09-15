"""Execute final Device IR on NumPy to check Lower semantics without TT hardware.

This small reference model follows only the published Device operation contract;
it does not reconstruct frontend blocks or call Lower's analyses. It is not a
TTL compiler, simulator, or hardware numerical certification.
"""

import numpy as np
import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_phase4_compute import (
    batch_elementwise,
    cast_program,
    elementwise_program,
    fill_program,
    gemm_program,
    reduction_program,
    shared,
    transpose_program,
)


def cast(value, dtype):
    if str(dtype).startswith(("int", "uint")):
        return np.asarray(value, dtype=str(dtype))
    value = np.asarray(value, dtype=np.float32)
    if str(dtype) == "float32":
        return value
    assert str(dtype) == "bfloat16", dtype
    bits = value.view(np.uint32)
    # Round-to-nearest-even, preserving NaNs rather than turning them into inf.
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    rounded = np.where(np.isnan(value), bits | np.uint32(0x00400000), rounded).astype(np.uint32)
    return rounded.view(np.float32)


def expression(expr, values, maps, domain):
    if isinstance(expr, (tirx.IntImm, tirx.FloatImm)):
        result = expr.value
    elif isinstance(expr, tirx.Cast):
        result = expression(expr.value, values, maps, domain)
    elif isinstance(expr, tirx.Call):
        if expr.op.name == "tl.tt.dfb_load":
            resource = int(expr.args[0])
            coordinates = np.indices(domain, sparse=True)
            indices = tuple(0 if axis == -1 else coordinates[axis] for axis in maps[resource])
            result = values[resource][indices]
        else:
            unary = {
                "tirx.sqrt": np.sqrt,
                "tirx.rsqrt": lambda x: 1 / np.sqrt(x),
                "tirx.exp": np.exp,
                "tirx.exp2": np.exp2,
                "tirx.log": np.log,
                "tirx.log2": np.log2,
                "tirx.tanh": np.tanh,
                "tirx.sin": np.sin,
                "tirx.cos": np.cos,
                "tirx.fabs": np.abs,
                "tirx.floor": np.floor,
                "tirx.ceil": np.ceil,
            }
            result = unary[expr.op.name](expression(expr.args[0], values, maps, domain))
    else:
        operations = {
            tirx.Add: np.add,
            tirx.Sub: np.subtract,
            tirx.Mul: np.multiply,
            tirx.Div: np.divide,
            tirx.Min: np.minimum,
            tirx.Max: np.maximum,
        }
        operation = operations[type(expr)]
        result = operation(expression(expr.a, values, maps, domain), expression(expr.b, values, maps, domain))
    return cast(result, expr.dtype)


def run_device(mod, inputs):
    """Schedule the three slot streams, stalling wait until publication."""
    VerifyTenstorrentDeviceIR()(mod)
    loops = [func.body for func in mod.functions.values() if isinstance(func.body, tirx.For)]
    if loops:
        from testing.python.target.test_tilelang_tenstorrent_phase2_device_ir import _replace_dfb

        iteration = ir.load_json(ir.save_json(mod))
        for global_var, func in list(iteration.functions.items()):
            if isinstance(func.body, tirx.For):
                iteration[global_var] = func.with_body(func.body.body, func.span)
        iteration = iteration.with_attr(
            "tt.dfb_table", [_replace_dfb(dfb, transaction_count_or_loop_relation=1) for dfb in iteration.attrs["tt.dfb_table"]]
        )
        # New publication epoch each iteration; a wait cannot reuse an old value.
        result = inputs
        for _ in range(int(loops[0].extent)):
            result = run_device(iteration, result)
        return result
    tensors = [cast(value, desc.dtype).copy() for value, desc in zip(inputs, mod.attrs["tt.tensor_table"])]
    values = {}
    streams = []
    for func in mod.functions.values():
        statements = list(func.body.seq) if isinstance(func.body, tirx.SeqStmt) else [func.body]
        streams.append(
            [
                statement.value
                for statement in statements
                if not (isinstance(statement, tirx.Evaluate) and isinstance(statement.value, tirx.IntImm))
            ]
        )
    positions = [0] * len(streams)
    while any(position < len(stream) for position, stream in zip(positions, streams)):
        progressed = False
        for slot, stream in enumerate(streams):
            position = positions[slot]
            if position == len(stream):
                continue
            call = stream[position]
            name = call.op.name
            args = [int(arg) for arg in call.args]
            if name == "tl.tt.dfb_wait" and args[0] not in values:
                continue
            if name in ("tl.tt.dfb_wait", "tl.tt.dfb_reserve"):
                pass
            elif name in ("tl.tt.tensor_to_dfb_nd", "tl.tt.dfb_to_tensor_nd"):
                to_dfb = name == "tl.tt.tensor_to_dfb_nd"
                tensor, resource = args[:2] if to_dfb else args[1::-1]
                region = tuple(slice(start, start + extent) for start, extent in zip(args[2::2], args[3::2]))
                if to_dfb:
                    values[resource] = tensors[tensor][region].copy()
                else:
                    tensors[tensor][region] = values[resource]
            elif name == "tl.tt.dfb_compute":
                attrs = call.annotations
                output, *operands = args
                kind = attrs["tt.compute_kind"].value
                domain = tuple(int(x) for x in attrs["tt.logical_domain"])
                maps = {resource: [int(x) for x in axes] for resource, axes in zip(operands, attrs["tt.access_maps"])}
                if kind in ("elementwise", "typecast", "fill"):
                    result = np.broadcast_to(expression(attrs["tt.expression"], values, maps, domain), domain)
                elif kind == "copy":
                    result = values[operands[0]]
                elif kind == "transpose":
                    result = np.transpose(values[operands[0]], [int(x) for x in attrs["tt.axes"]])
                elif kind == "gemm":
                    a, b = (values[x] for x in operands[:2])
                    if int(attrs["tt.transpose_a"]):
                        a = a.T
                    if int(attrs["tt.transpose_b"]):
                        b = b.T
                    result = a.astype(np.float32) @ b.astype(np.float32)
                    if not int(attrs["tt.clear"]):
                        result = result + values[operands[2]].astype(np.float32)
                elif kind == "reduce":
                    axis = int(attrs["tt.reduce_axis"])
                    reduction = attrs["tt.reduce_kind"].value
                    source = values[operands[0]].astype(np.float32)
                    if reduction == "sum":
                        result = np.sum(source, axis=axis, dtype=np.float32)
                        combine = np.add
                    else:
                        propagate = bool(int(attrs["tt.nan_propagate"]))
                        combine = (np.maximum if propagate else np.fmax) if reduction == "max" else (np.minimum if propagate else np.fmin)
                        result = combine.reduce(source, axis=axis)
                    if not int(attrs["tt.clear"]):
                        result = combine(values[operands[1]].astype(np.float32), result)
                else:
                    raise AssertionError(kind)
                values[output] = cast(result, attrs["tt.compute_dtype"].value).copy()
            else:
                raise AssertionError(f"Unsupported reference Device op: {name}")
            positions[slot] += 1
            progressed = True
        assert progressed, "Device streams deadlocked"
    return tensors


def lower(func, arch="wormhole_b0"):
    target = tvm.target.Target({"kind": "tenstorrent", "arch": arch})
    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": func}), target)
    assert int(mod.attrs["tt.device_ir_version"]) == 2
    assert "tt.ir_stage" not in mod.attrs
    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(restored, mod)
    return restored


@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
def test_final_tiles_and_parallel_expression_semantics(broadcast, inplace, arch):
    rng = np.random.default_rng(12)
    a = rng.uniform(1, 5, (64, 64)).astype(np.float32)
    b = rng.uniform(1, 5, (32 if broadcast else 64, 64)).astype(np.float32)
    expected = np.sqrt(a * a + (b[0:1] if broadcast else b)) / np.float32(2)
    modules = [lower(elementwise_program(tiles=tiles, broadcast=broadcast, inplace=inplace), arch) for tiles in (False, True)]
    ir.assert_structural_equal(*modules)
    for module in modules:
        output = run_device(module, [a, b, np.zeros_like(a)])[-1]
        np.testing.assert_array_equal(output, expected)


def test_final_batch_broadcast_semantics():
    a = np.arange(2 * 64 * 64, dtype=np.float32).reshape(2, 64, 64)
    b = np.arange(32 * 64, dtype=np.float32).reshape(1, 32, 64)
    output = run_device(lower(batch_elementwise), [a, b, np.zeros_like(a)])[-1]
    np.testing.assert_array_equal(output, a - b[:, :1, :])


def test_final_fill_cast_transpose_semantics():
    a = np.arange(64 * 96, dtype=np.float32).reshape(64, 96) / 7
    filled = run_device(lower(fill_program), [np.zeros((64, 64), dtype=np.float32)])[0]
    np.testing.assert_array_equal(filled, np.full((64, 64), 3.25, dtype=np.float32))
    converted = run_device(lower(cast_program), [a[:, :64], np.zeros((64, 64), dtype=np.float32)])[-1]
    np.testing.assert_array_equal(converted, cast(a[:, :64], "bfloat16"))
    transposed = run_device(lower(transpose_program), [a, np.zeros((96, 64), dtype=np.float32)])[-1]
    np.testing.assert_array_equal(transposed, a.T)


@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("clear", [False, True])
def test_final_gemm_accumulation_semantics(transpose_a, transpose_b, clear):
    rng = np.random.default_rng(31)
    a = rng.integers(-3, 4, (32, 64)).astype(np.float32)
    b = rng.integers(-3, 4, (64, 96)).astype(np.float32)
    mod = lower(gemm_program(clear=clear, transpose_a=transpose_a, transpose_b=transpose_b))
    output = run_device(mod, [a.T if transpose_a else a, b.T if transpose_b else b, np.zeros((32, 96), dtype=np.float32)])[-1]
    np.testing.assert_array_equal(output, a @ b + (0 if clear else 1))


@pytest.mark.parametrize("kind", ["sum", "min", "max"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("dtype,output_dtype", [("float32", "float32"), ("bfloat16", "float32"), ("bfloat16", "bfloat16")])
def test_final_reduction_semantics(kind, axis, clear, dtype, output_dtype):
    a = np.random.default_rng(2).integers(-3, 4, (64, 96)).astype(np.float32)
    shape = (96,) if axis == 0 else (64,)
    mod = lower(reduction_program(kind=kind, axis=axis, clear=clear, dtype=dtype, output_dtype=output_dtype))
    output = run_device(mod, [a, np.zeros(shape, dtype=np.float32)])[-1]
    expected = {"sum": np.sum, "min": np.min, "max": np.max}[kind](a, axis=axis)
    if not clear:
        expected = {"sum": np.add, "min": np.minimum, "max": np.maximum}[kind](expected, 1)
    np.testing.assert_array_equal(output, cast(expected, output_dtype))


@T.prim_func
def tensor_inout(A: T.Tensor((64, 64), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = shared((64, 64))
        T.copy(A, a)
        for i, j in T.Parallel(64, 64):
            a[i, j] = a[i, j] + T.float32(2)
        T.copy(a, A)


def test_same_tensor_inout_preserves_read_before_write():
    a = np.arange(4096, dtype=np.float32).reshape(64, 64)
    mod = lower(tensor_inout)
    assert str(mod.attrs["tt.tensor_table"][0].effect) == "inout"
    np.testing.assert_array_equal(run_device(mod, [a])[0], a + 2)


@T.prim_func
def batch_transpose(A: T.Tensor((2, 64, 96), "float32"), C: T.Tensor((2, 96, 64), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = shared((2, 64, 96))
        c = shared((2, 96, 64))
        T.copy(A, a)
        T.transpose(a, c)
        T.copy(c, C)


def batch_reduce(axis):
    shape = (32, 64, 96)
    output_shape = tuple(size for i, size in enumerate(shape) if i != axis)

    @T.prim_func
    def program(A: T.Tensor(shape, "float32"), C: T.Tensor(output_shape, "float32")):
        with T.Kernel(1, 1, threads=1):
            a = shared(shape)
            c = shared(output_shape)
            T.copy(A, a)
            T.reduce_sum(a, c, dim=axis)
            T.copy(c, C)

    return program, output_shape


def test_final_batch_transpose_semantics():
    a = np.arange(2 * 64 * 96, dtype=np.float32).reshape(2, 64, 96)
    out = run_device(lower(batch_transpose), [a, np.zeros((2, 96, 64), np.float32)])[-1]
    np.testing.assert_array_equal(out, a.transpose(0, 2, 1))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_final_batch_reduction_semantics_and_layout(axis):
    program, shape = batch_reduce(axis)
    a = np.random.default_rng(7).integers(-3, 4, (32, 64, 96)).astype(np.float32)
    out = run_device(lower(program), [a, np.zeros(shape, np.float32)])[-1]
    np.testing.assert_array_equal(out, a.sum(axis=axis))


@T.prim_func
def unused_tensor(A: T.Tensor((64, 64), "float32"), C: T.Tensor((64, 64), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = shared((64, 64))
        c = shared((64, 64))
        T.copy(A, a)
        T.fill(c, 5)
        T.copy(c, C)


def test_dead_tensor_transfer_does_not_leak_into_processor_abi():
    mod = lower(unused_tensor)
    assert len(mod.attrs["tt.tensor_table"]) == 2
    ncrisc = next(func for func in mod.functions.values() if str(func.attrs["tt.kernel_slot"]) == "ncrisc")
    assert [int(index) for index in ncrisc.attrs["tt.tensor_arg_indices"]] == [1]
    a = np.zeros((64, 64), np.float32)
    np.testing.assert_array_equal(run_device(mod, [a, a])[-1], np.full_like(a, 5))

"""Integer dtype support across the Tenstorrent Lower pipeline.

Covers the storage/transport subset (bfloat16, float32, int32, uint32, uint16),
the integer elementwise subset (add, sub, mul, broadcast) and the classified
rejections that TT-Lang and tt-metal enforce downstream. See
progress_doc/int_dtype_support_plan.md for the evidence behind each boundary.
"""

from __future__ import annotations

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.capabilities import CapabilityStatus, lower_capability
from tilelang.tenstorrent.pipeline import lower_tenstorrent_ir
from tilelang.tenstorrent.ttl_codegen.expression import check_expression
from tvm import tirx

TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
METADATA = {"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2}
INTEGER_DTYPES = ("int32", "uint32", "uint16")


def storage_program(dtype):
    @T.prim_func
    def main(A: T.Tensor((32, 32), dtype), C: T.Tensor((32, 32), dtype)):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), dtype, annotations=METADATA)
            T.copy(A, a)
            T.copy(a, C)

    return main


def binary_program(dtype, op):
    @T.prim_func
    def main(A: T.Tensor((64, 32), dtype), B: T.Tensor((64, 32), dtype), C: T.Tensor((64, 32), dtype)):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((64, 32), dtype, annotations=METADATA)
            b = T.alloc_shared((64, 32), dtype, annotations=METADATA)
            c = T.alloc_shared((64, 32), dtype, annotations=METADATA)
            T.copy(A, a)
            T.copy(B, b)
            for i, j in T.Tiles(c):
                if op == "add":
                    c[i, j] = a[i, j] + b[i, j]
                elif op == "sub":
                    c[i, j] = a[i, j] - b[i, j]
                elif op == "mul":
                    c[i, j] = a[i, j] * b[i, j]
                elif op == "max":
                    c[i, j] = T.max(a[i, j], b[i, j])
                elif op == "exp":
                    c[i, j] = T.exp(a[i, j])
                else:  # pragma: no cover - defensive
                    raise AssertionError(op)
            T.copy(c, C)

    return main


def broadcast_program(dtype):
    @T.prim_func
    def main(A: T.Tensor((64, 32), dtype), C: T.Tensor((64, 32), dtype)):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((64, 32), dtype, annotations=METADATA)
            c = T.alloc_shared((64, 32), dtype, annotations=METADATA)
            T.copy(A, a)
            for i, j in T.Tiles(c):
                c[i, j] = a[i, 0]
            T.copy(c, C)

    return main


def fill_program(dtype):
    @T.prim_func
    def main(C: T.Tensor((32, 32), dtype)):
        with T.Kernel(1, 1, threads=1):
            c = T.alloc_shared((32, 32), dtype, annotations=METADATA)
            T.fill(c, 0)
            T.copy(c, C)

    return main


def device_tables(mod, key):
    value = None
    if mod.attrs is not None and key in mod.attrs:
        value = mod.attrs[key]
    for _, func in mod.functions.items():
        if isinstance(func, tirx.PrimFunc) and func.attrs is not None and key in func.attrs:
            value = func.attrs[key]
    assert value is not None, f"Device IR has no {key}"
    return value


def destination_width(mod):
    for _, func in mod.functions.items():
        if isinstance(func, tirx.PrimFunc) and func.attrs is not None:
            requirements = func.attrs.get("tt.compute_requirements")
            if requirements is not None:
                return str(requirements.destination_width)
    raise AssertionError("Device IR has no compute requirements")


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_integer_storage_reaches_device_ir(dtype):
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": storage_program(dtype)}), TARGET)
    assert str(mod.attrs["tt.ir_stage"]) == "device"
    assert {str(tensor.dtype) for tensor in device_tables(mod, "tt.tensor_table")} == {dtype}
    assert "tt.dfb_storage_groups" in mod.attrs


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_integer_binary_elementwise_reaches_device_ir(dtype, op):
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": binary_program(dtype, op)}), TARGET)
    assert str(mod.attrs["tt.ir_stage"]) == "device"


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_integer_broadcast_reaches_device_ir(dtype):
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": broadcast_program(dtype)}), TARGET)
    assert str(mod.attrs["tt.ir_stage"]) == "device"


@pytest.mark.parametrize("dtype,width", [
    ("int32", "bits32_required"),
    ("uint32", "bits32_required"),
    ("uint16", "bits16_required"),
])
def test_integer_destination_width(dtype, width):
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": binary_program(dtype, "add")}), TARGET)
    assert destination_width(mod) == width


def test_integer_storage_leaves_destination_width_unconstrained():
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": storage_program("int32")}), TARGET)
    assert destination_width(mod) == "unconstrained"


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_integer_fill_is_rejected(dtype):
    with pytest.raises(NotImplementedError, match="TT-Lang fill only accepts an f32 constant"):
        lower_tenstorrent_ir(tvm.IRModule({"main": fill_program(dtype)}), TARGET)


def test_integer_min_max_is_rejected_with_operator_reason():
    with pytest.raises(NotImplementedError, match="only support elementwise add, sub, mul and broadcast"):
        lower_tenstorrent_ir(tvm.IRModule({"main": binary_program("int32", "max")}), TARGET)


def test_integer_unary_intrinsic_is_rejected_as_mixed_expression():
    with pytest.raises(NotImplementedError, match="mixes integer and floating-point values"):
        lower_tenstorrent_ir(tvm.IRModule({"main": binary_program("int32", "exp")}), TARGET)


@pytest.mark.parametrize("dtype,reason", [
    ("uint8", "TT-Lang compute passthrough accepts"),
    ("float16", "no FLOAT16 tensor type"),
])
def test_unsupported_storage_dtype_diagnoses_downstream_cause(dtype, reason):
    with pytest.raises(NotImplementedError, match=reason):
        lower_tenstorrent_ir(tvm.IRModule({"main": storage_program(dtype)}), TARGET)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_capabilities_accept_integer_storage_and_elementwise(dtype):
    for operation in ("copy", "elementwise"):
        capability = lower_capability(operation, arch="wormhole_b0", input_dtype=dtype)
        assert capability.status == CapabilityStatus.SUPPORTED


@pytest.mark.parametrize("operation", ["fill", "typecast", "transpose", "reduce"])
def test_capabilities_reject_integer_non_elementwise_routes(operation):
    capability = lower_capability(operation, arch="wormhole_b0", input_dtype="int32")
    assert capability.status == CapabilityStatus.UNSUPPORTED
    assert "no integer route" in capability.reason


def test_capabilities_reject_mixed_integer_float():
    capability = lower_capability(
        "elementwise", arch="wormhole_b0", input_dtype="int32", output_dtype="float32"
    )
    assert capability.status == CapabilityStatus.UNSUPPORTED
    assert "cannot mix" in capability.reason


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_ttl_expression_check_accepts_integer_binary(dtype):
    lhs = tirx.Var("lhs", dtype)
    rhs = tirx.Var("rhs", dtype)
    check_expression(tirx.Add(lhs, rhs))
    check_expression(tirx.Mul(lhs, rhs))


def test_ttl_expression_check_rejects_integer_div():
    lhs = tirx.Var("lhs", "int32")
    rhs = tirx.Var("rhs", "int32")
    with pytest.raises(NotImplementedError, match="integer tiles only support elementwise add, sub, mul"):
        check_expression(tirx.Div(lhs, rhs))

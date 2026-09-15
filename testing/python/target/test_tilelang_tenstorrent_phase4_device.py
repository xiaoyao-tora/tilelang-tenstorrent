"""Device formation, dependency, metadata, and host-reference regressions."""

import numpy as np
import pytest

from tilelang import tvm
from tilelang.backend import create_backend_context
from tilelang.tenstorrent import execution_backend, transform
from tilelang.tenstorrent import language as T
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
from testing.python.target.test_tilelang_tenstorrent_phase2_device_ir import (
    _replace_dfb,
    _replace_slot_body,
    _slots,
)


def lower(monkeypatch, program):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    context = create_backend_context({"kind": "tenstorrent", "arch": "wormhole_b0"}, target_host="c", execution_backend="ttnn")
    result = context.lower(tvm.IRModule({"main": program}))
    assert int(result.attrs["tt.device_ir_version"]) == 2
    assert "tt.ir_stage" not in result.attrs
    assert transform.VerifyTenstorrentDeviceIR()(result).same_as(result)
    return result


def calls(function):
    body = function.body
    if isinstance(body, tirx.Evaluate) and isinstance(body.value, tirx.IntImm):
        return []
    statements = body.seq if isinstance(body, tirx.SeqStmt) else [body]
    return [statement.value for statement in statements]


def reference_device(mod, inputs):
    """Interpret logical Device v2 operations, without claiming TTL validation."""
    tensors = {index: array.copy() for index, array in inputs.items()}
    resources = {}
    pending = {slot: calls(function) for slot, function in _slots(mod).items()}

    def scalar(expression, values):
        if isinstance(expression, (tirx.FloatImm, tirx.IntImm)):
            return expression.value
        if isinstance(expression, tirx.Cast):
            return np.asarray(scalar(expression.value, values), dtype=np.float32)
        if isinstance(expression, tirx.Call):
            if expression.op.name == "tl.tt.dfb_load":
                return values[int(expression.args[0])]
            unary = {"tirx.sqrt": np.sqrt, "tirx.exp": np.exp, "tirx.log": np.log, "tirx.fabs": np.abs}
            return unary[expression.op.name](scalar(expression.args[0], values))
        binary = {
            tirx.Add: np.add,
            tirx.Sub: np.subtract,
            tirx.Mul: np.multiply,
            tirx.Div: np.divide,
            tirx.Min: np.minimum,
            tirx.Max: np.maximum,
        }
        return binary[type(expression)](scalar(expression.a, values), scalar(expression.b, values))

    while any(pending.values()):
        advanced = False
        for queue in pending.values():
            if not queue:
                continue
            call = queue[0]
            name = call.op.name
            args = [int(arg) for arg in call.args]
            if name == "tl.tt.dfb_wait" and args[0] not in resources:
                continue
            if name == "tl.tt.dfb_compute" and any(index not in resources for index in args[1:]):
                continue
            queue.pop(0)
            advanced = True
            if name in ("tl.tt.dfb_wait", "tl.tt.dfb_reserve"):
                continue
            if name == "tl.tt.tensor_to_dfb_nd":
                resources[args[1]] = tensors[args[0]].copy()
            elif name == "tl.tt.dfb_to_tensor_nd":
                tensors[args[1]] = resources[args[0]].copy()
            else:
                attrs = call.annotations
                kind = str(attrs["tt.compute_kind"].value)
                shape = tuple(int(dim) for dim in attrs["tt.logical_domain"])
                incoming = [resources[index] for index in args[1:]]
                if kind in ("elementwise", "fill", "typecast"):
                    values = {}
                    for index, value, axes in zip(args[1:], incoming, attrs["tt.access_maps"]):
                        mapping = [int(axis) for axis in axes]
                        slices = tuple(0 if axis < 0 else slice(None) for axis in mapping)
                        value = value[slices]
                        kept = [axis for axis in mapping if axis >= 0]
                        if kept:
                            value = np.transpose(value, np.argsort(kept))
                        expanded = [1] * len(shape)
                        for axis in kept:
                            expanded[axis] = shape[axis]
                        values[index] = value.reshape(expanded)
                    result = np.broadcast_to(scalar(attrs["tt.expression"], values), shape).copy()
                elif kind == "copy":
                    result = incoming[0].copy()
                elif kind == "transpose":
                    result = incoming[0].swapaxes(-1, -2)
                elif kind == "gemm":
                    a, b = incoming[:2]
                    if int(attrs["tt.transpose_a"]):
                        a = a.swapaxes(-1, -2)
                    if int(attrs["tt.transpose_b"]):
                        b = b.swapaxes(-1, -2)
                    result = a @ b
                    if not int(attrs["tt.clear"]):
                        result = result + incoming[-1]
                else:
                    op = {"sum": np.sum, "max": np.max, "min": np.min}[str(attrs["tt.reduce_kind"].value)]
                    result = op(incoming[0], axis=int(attrs["tt.reduce_axis"]))
                    if not int(attrs["tt.clear"]):
                        result = (
                            result + incoming[-1] if op == np.sum else (np.maximum if op == np.max else np.minimum)(result, incoming[-1])
                        )
                resources[args[0]] = result
        assert advanced, "Device interpreter encountered a dependency cycle"
    return tensors


@pytest.mark.parametrize("tiles", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_elementwise_semantics(monkeypatch, tiles, broadcast, inplace):
    mod = lower(monkeypatch, elementwise_program(tiles=tiles, broadcast=broadcast, inplace=inplace))
    rng = np.random.default_rng(3)
    a = rng.uniform(0.5, 2, (64, 64)).astype("float32")
    b = rng.uniform(0.5, 2, (32 if broadcast else 64, 64)).astype("float32")
    actual = reference_device(mod, {0: a, 1: b})[2]
    np.testing.assert_allclose(actual, np.sqrt(a * a + (b[0] if broadcast else b)) / 2, rtol=1e-6)
    assert all(".v" in str(d.source_buffer_identity) for d in mod.attrs["tt.dfb_table"])


def test_tiles_parallel_device_equivalence(monkeypatch):
    left = lower(monkeypatch, elementwise_program(tiles=False))
    right = lower(monkeypatch, elementwise_program(tiles=True))
    for slot in _slots(left):
        ir.assert_structural_equal(_slots(left)[slot].body, _slots(right)[slot].body, map_free_vars=True)


@pytest.mark.parametrize("program", [batch_elementwise, cast_program, fill_program, transpose_program])
def test_device_capabilities_and_round_trip(monkeypatch, program):
    mod = lower(monkeypatch, program)
    loaded = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(mod, loaded)
    transform.VerifyTenstorrentDeviceIR()(loaded)


@pytest.mark.parametrize("clear", [True, False])
@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (True, True)])
def test_gemm_semantics(monkeypatch, clear, transpose_a, transpose_b):
    mod = lower(monkeypatch, gemm_program(clear=clear, transpose_a=transpose_a, transpose_b=transpose_b, input_dtype="float32"))
    rng = np.random.default_rng(6)
    a = rng.uniform(-1, 1, (64, 32) if transpose_a else (32, 64)).astype("float32")
    b = rng.uniform(-1, 1, (96, 64) if transpose_b else (64, 96)).astype("float32")
    expected = (a.T if transpose_a else a) @ (b.T if transpose_b else b) + (0 if clear else 1)
    np.testing.assert_allclose(reference_device(mod, {0: a, 1: b})[2], expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["sum", "max", "min"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("clear", [True, False])
def test_reduction_semantics(monkeypatch, kind, axis, clear):
    mod = lower(monkeypatch, reduction_program(kind=kind, axis=axis, clear=clear))
    a = np.random.default_rng(5).uniform(-1, 1, (64, 96)).astype("float32")
    expected = {"sum": np.sum, "max": np.max, "min": np.min}[kind](a, axis=axis)
    if not clear:
        expected = expected + 1 if kind == "sum" else (np.maximum if kind == "max" else np.minimum)(expected, 1)
    np.testing.assert_allclose(reference_device(mod, {0: a})[1], expected, rtol=1e-6)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"consumer_slot": "brisc"}, "wait in wrong slot|consumer slot"),
        ({"transaction_count_or_loop_relation": 2}, "exactly one publication"),
        ({"element_dtype": "bfloat16"}, "backing metadata mismatch|dfb_load dtype mismatch"),
    ],
)
def test_resource_metadata_rejected(monkeypatch, change, match):
    mod = lower(monkeypatch, elementwise_program())
    dfbs = list(mod.attrs["tt.dfb_table"])
    dfbs[0] = _replace_dfb(dfbs[0], **change)
    mod = mod.with_attr("tt.dfb_table", dfbs)
    with pytest.raises(ValueError, match=match):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_missing_wait_rejected(monkeypatch):
    mod = lower(monkeypatch, elementwise_program())
    body = _slots(mod)["trisc"].body
    statements = list(body.seq)
    statements.pop(next(i for i, statement in enumerate(statements) if statement.value.op.name == "tl.tt.dfb_wait"))
    _replace_slot_body(mod, "trisc", tirx.SeqStmt(statements))
    with pytest.raises(ValueError, match="preceded by dfb_wait"):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_cross_slot_cycle_rejected(monkeypatch):
    mod = lower(monkeypatch, elementwise_program())
    body = _slots(mod)["ncrisc"].body
    statements = list(body.seq)
    export_wait = next(i for i, statement in enumerate(statements) if statement.value.op.name == "tl.tt.dfb_wait")
    statements.insert(0, statements.pop(export_wait))
    _replace_slot_body(mod, "ncrisc", tirx.SeqStmt(statements))
    with pytest.raises(ValueError, match="dependency cycle"):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_read_before_write_rejected(monkeypatch):
    @T.prim_func
    def program(C: T.Tensor((64, 64), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = shared((64, 64))
            c = shared((64, 64))
            for i, j in T.Parallel(64, 64):
                c[i, j] = a[i, j] * T.float32(2)
            T.copy(c, C)

    with pytest.raises(ValueError, match="read-before-write"):
        lower(monkeypatch, program)


def test_inout_and_static_loop(monkeypatch):
    @T.prim_func
    def program(A: T.Tensor((64, 64), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = shared((64, 64))
            T.copy(A, a)
            for _step in T.serial(3):
                for i, j in T.Parallel(64, 64):
                    a[i, j] = a[i, j] * T.float32(2)
            T.copy(a, A)

    mod = lower(monkeypatch, program)
    assert str(mod.attrs["tt.tensor_table"][0].effect) == "inout"
    a = np.ones((64, 64), dtype="float32")
    np.testing.assert_equal(reference_device(mod, {0: a})[0], a * 8)


def ir_metadata(value):
    if isinstance(value, list):
        return tvm.runtime.convert([ir_metadata(item) for item in value])
    if isinstance(value, int):
        return tirx.IntImm("int32", value)
    return value


def replace_compute_annotation(mod, key, value, *, kind=None):
    body = _slots(mod)["trisc"].body
    statements = list(body.seq)
    for index, statement in enumerate(statements):
        call = statement.value
        if call.op.name != "tl.tt.dfb_compute":
            continue
        if kind is not None and str(call.annotations["tt.compute_kind"].value) != kind:
            continue
        annotations = dict(call.annotations)
        annotations[key] = value
        statements[index] = tirx.Evaluate(tirx.Call(call.dtype, call.op, call.args, annotations, call.span))
        _replace_slot_body(mod, "trisc", tirx.SeqStmt(statements))
        return
    raise AssertionError("no matching compute operation")


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("tt.compute_dtype", tirx.StringImm("bfloat16"), "dtype annotation"),
        ("tt.logical_domain", [32, 64], "tile-grid metadata"),
        ("tt.access_maps", [[0, 2], [0, 1]], "axis is out of range"),
        ("tt.access_maps", [[0, 0], [0, 1]], "duplicates an output axis"),
        ("tt.compute_kind", tirx.StringImm("unknown"), "unknown compute kind"),
    ],
)
def test_compute_metadata_rejected(monkeypatch, key, value, match):
    mod = lower(monkeypatch, elementwise_program())
    replace_compute_annotation(mod, key, ir_metadata(value))
    with pytest.raises(ValueError, match=match):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_transpose_axes_rejected(monkeypatch):
    mod = lower(monkeypatch, transpose_program)
    replace_compute_annotation(mod, "tt.axes", ir_metadata([0, 1]), kind="transpose")
    with pytest.raises(ValueError, match="last-two-axis permutation"):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_gemm_accumulation_rejected(monkeypatch):
    mod = lower(monkeypatch, gemm_program())
    replace_compute_annotation(mod, "tt.accum_dtype", tirx.StringImm("bfloat16"), kind="gemm")
    with pytest.raises(ValueError, match="unsupported accumulation dtype"):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_double_publication_rejected(monkeypatch):
    mod = lower(monkeypatch, elementwise_program())
    body = _slots(mod)["trisc"].body
    statements = list(body.seq)
    compute = next(i for i, statement in enumerate(statements) if statement.value.op.name == "tl.tt.dfb_compute")
    statements.insert(compute + 1, statements[compute])
    _replace_slot_body(mod, "trisc", tirx.SeqStmt(statements))
    with pytest.raises(ValueError, match="published more than once"):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_dead_pure_write_eliminated(monkeypatch):
    @T.prim_func
    def program(C: T.Tensor((64, 64), "float32")):
        with T.Kernel(1, 1, threads=1):
            c = shared((64, 64))
            T.fill(c, 3)
            T.fill(c, 7)
            T.copy(c, C)

    mod = lower(monkeypatch, program)
    # One live Fill publication and one export snapshot. The overwritten Fill
    # neither leaks an unused resource nor changes the visible result.
    assert len(mod.attrs["tt.dfb_table"]) == 2
    np.testing.assert_equal(reference_device(mod, {})[0], np.full((64, 64), 7))


def test_padded_input_shape_matches_producer(monkeypatch):
    mod = lower(monkeypatch, batch_elementwise)
    replace_compute_annotation(mod, "tt.input_shapes", ir_metadata([[2, 64, 64], [1, 1, 64]]))
    with pytest.raises(ValueError, match="input logical shape disagrees with its producer"):
        transform.VerifyTenstorrentDeviceIR()(mod)


@pytest.mark.parametrize(
    "key,value",
    [
        ("tt.logical_domain", [32, 64]),
        ("tt.input_shapes", [[64, 64], [64, 64]]),
        ("tt.access_maps", [[0, 1], [0, 1]]),
    ],
)
def test_untyped_integer_array_annotation_rejected(monkeypatch, key, value):
    mod = lower(monkeypatch, elementwise_program())
    # FFI Array[int] is not Array[IntImm]. A malformed external Device module
    # must be diagnosed before typed-array element access.
    replace_compute_annotation(mod, key, tvm.runtime.convert(value))
    with pytest.raises(ValueError, match="annotation|shape|map|integer|IntImm|PrimExpr"):
        transform.VerifyTenstorrentDeviceIR()(mod)

"""Check collective frontend contracts and IR capture without device execution."""

import pytest
from tilelang.tenstorrent import language as T
from tvm import ir, tirx


def _calls(func):
    calls = []

    def collect(node):
        if (
            isinstance(node, tirx.Call)
            and isinstance(node.op, ir.Op)
            and node.op.name
            in (
                "tl.tt.transfer",
                "tl.tt.all_gather",
                "tl.tt.all_reduce",
            )
        ):
            calls.append(node)

    tirx.stmt_functor.post_order_visit(func.body, collect)
    return calls


def _program(
    *,
    api="all_gather",
    group=None,
    grid=(2, 2),
    shape=(32, 64),
    dst_shape=None,
    dtype="bfloat16",
    dst_dtype=None,
    scope="shared",
    axis=1,
    op="sum",
):
    group = T.comm.CoreRange((0, 0), (2, 2)) if group is None else group
    if dst_shape is None:
        dst_shape = shape if api == "all_reduce" else (32, 256)
    dst_dtype = dtype if dst_dtype is None else dst_dtype

    @T.prim_func
    def main():
        with T.Kernel(*grid, threads=1):
            src = T.alloc_buffer(shape, dtype, scope=scope)
            dst = T.alloc_shared(dst_shape, dst_dtype)
            if api == "all_gather":
                T.comm.all_gather(src=src, dst=dst, group=group, axis=axis)
            else:
                T.comm.all_reduce(src=src, dst=dst, group=group, op=op)

    return main


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("axis", [-2, -1, 0, 1])
@pytest.mark.parametrize(
    "begin, end, size",
    [((0, 0), (1, 1), 1), ((0, 1), (4, 2), 4), ((1, 0), (2, 3), 3), ((1, 1), (3, 3), 4)],
)
def test_all_gather_shapes_groups_and_axis_normalization(dtype, axis, begin, end, size):
    dst_shape = [32, 64]
    dst_shape[axis % 2] *= size
    func = _program(group=T.comm.CoreRange(begin, end), grid=(4, 3), dtype=dtype, axis=axis, dst_shape=tuple(dst_shape))
    (call,) = _calls(func)
    assert call.op.name == "tl.tt.all_gather"
    assert [int(coord.value) for coord in call.args[2:6]] == [*begin, *end]
    assert int(call.args[6].value) == axis % 2
    assert int(call.args[7].value) == 0
    for index, access, shape in ((0, 1, [32, 64]), (1, 2, dst_shape)):
        region = call.args[index]
        assert region.op.name == "tl.tileop.region"
        assert int(region.args[1].value) == access
        assert [int(extent.value) for extent in region.args[2:]] == shape
        assert region.args[0].buffer.dtype == dtype
    assert not call.args[0].args[0].buffer.data.same_as(call.args[1].args[0].buffer.data)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("end", [(1, 1), (2, 2)])
def test_all_reduce_preserves_shape_and_reduction_kind(dtype, op, end):
    func = _program(api="all_reduce", dtype=dtype, op=op, group=T.comm.CoreRange((0, 0), end))
    (call,) = _calls(func)
    assert call.op.name == "tl.tt.all_reduce"
    assert call.args[6].value == op
    assert [int(coord.value) for coord in call.args[2:6]] == [0, 0, *end]
    assert int(call.args[7].value) == 0
    for region in call.args[:2]:
        assert [int(extent.value) for extent in region.args[2:]] == [32, 64]
        assert region.args[0].buffer.dtype == dtype


def test_symbolic_groups_mixed_sites_loops_and_serialization():
    def build():
        @T.prim_func
        def main():
            with T.Kernel(4, 3, threads=1) as (x, y):
                src = T.alloc_shared((32, 32), "float32")
                gathered = T.alloc_shared((32, 128), "float32")
                reduced = T.alloc_shared((32, 32), "float32")
                for _stage in T.serial(5):
                    T.comm.transfer(src=src, dst=reduced, src_core=(0, y), dst_cores=(1, y))
                    T.comm.all_gather(src=src, dst=gathered, group=T.comm.CoreRange((0, y), (4, y + 1)), axis=-1)
                    T.comm.all_reduce(src=src, dst=reduced, group=T.comm.CoreRange((x, 0), (x + 1, 3)))
                    T.comm.put(src=src, dst=reduced, src_core=(0, y), dst_core=(1, y))
                    T.comm.broadcast(src=src, dst=reduced, src_core=(0, y), dst_cores=T.comm.CoreRange((1, y), (4, y + 1)))
                    T.comm.all_reduce(src=src, dst=reduced, group=T.comm.CoreRange((x, 0), (x + 1, 3)))

        return main

    func = build()
    calls = _calls(func)
    assert [int(call.args[-1].value) for call in calls] == list(range(6))
    assert [int(call.args[-1].value) for call in _calls(build())] == list(range(6))
    gather, reduce = calls[1:3]
    assert gather.args[6].value == 1
    assert reduce.args[6].value == "sum"
    assert isinstance(gather.args[3], tirx.Var)
    assert isinstance(reduce.args[2], tirx.Var)
    assert ir.structural_equal(gather.args[5], gather.args[3] + 1)
    assert ir.structural_equal(reduce.args[4], reduce.args[2] + 1)
    ir.assert_structural_equal(func, ir.load_json(ir.save_json(func)))
    simplified = tirx.transform.Simplify()(ir.IRModule({"main": func}))["main"]
    assert [int(call.args[-1].value) for call in _calls(simplified)] == list(range(6))
    loops = []
    tirx.stmt_functor.post_order_visit(func.body, lambda node: loops.append(node) if isinstance(node, tirx.For) else None)
    loop = next(node for node in loops if isinstance(node.extent, tirx.IntImm) and node.extent.value == 5)
    assert len(loop.body.seq) == 6
    assert func.script().count("T.tt.all_gather(") == 1
    assert func.script().count("T.tt.all_reduce(") == 2
    for name in ("all_gather", "all_reduce"):
        assert int(ir.Op.get("tl.tt." + name).get_attr("TCallEffectKind")) == tirx.CallEffectKind.Opaque.value


@pytest.mark.parametrize("axis", [True, 1.0, tirx.Var("axis", "int32"), tirx.IntImm("bool", True)])
def test_all_gather_requires_integer_axis(axis):
    with pytest.raises(TypeError, match="axis must be a compile-time integer"):
        _program(axis=axis)


@pytest.mark.parametrize("axis", [-3, 2])
def test_all_gather_rejects_out_of_range_axis(axis):
    with pytest.raises(ValueError, match="axis must be in"):
        _program(axis=axis)


def test_all_gather_accepts_constant_tir_axis():
    (call,) = _calls(_program(axis=tirx.IntImm("int64", -1)))
    assert int(call.args[6].value) == 1


@pytest.mark.parametrize("op, error", [("prod", ValueError), ("SUM", ValueError), (0, TypeError), (tirx.Var("op", "int32"), TypeError)])
def test_all_reduce_rejects_invalid_op(op, error):
    with pytest.raises(error, match="op must be"):
        _program(api="all_reduce", op=op)


@pytest.mark.parametrize("api", ["all_gather", "all_reduce"])
@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"group": (0, 0)}, TypeError, "group must be.*CoreRange"),
        ({"group": T.comm.CoreRange((0, 0), (3, 2))}, ValueError, "outside T.Kernel grid"),
        ({"group": T.comm.CoreRange((0, 0), (tirx.Var("n", "int32"), 1))}, ValueError, "positive compile-time integers"),
        ({"grid": (4,)}, ValueError, "2-D"),
        ({"grid": (tirx.Var("n", "int32"), 2)}, ValueError, "positive compile-time integer.*grid"),
        ({"shape": (32,)}, ValueError, "rank-2"),
        ({"shape": (48, 64)}, ValueError, "positive static payload extents"),
        ({"shape": (tirx.Var("m", "int32"), 64)}, ValueError, "positive static payload extents"),
        ({"dtype": "float16"}, ValueError, "only bfloat16 and float32"),
        ({"dtype": "int32"}, ValueError, "only bfloat16 and float32"),
        ({"dst_dtype": "float32"}, ValueError, "same dtype"),
        ({"scope": "local.fragment"}, ValueError, "shared or shared.dyn"),
        ({"dst_shape": (64, 64)}, ValueError, "shape"),
    ],
)
def test_collective_rejects_invalid_arguments(api, kwargs, error, match):
    with pytest.raises(error, match=match):
        _program(api=api, **kwargs)


@pytest.mark.parametrize("api", ["all_gather", "all_reduce"])
@pytest.mark.parametrize("alias_kind", ["same_buffer", "same_data", "match_buffer", "region"])
def test_collective_rejects_aliases_and_partial_regions(api, alias_kind):
    error, match = (TypeError, "complete tirx.Buffer") if alias_kind == "region" else (ValueError, "independent|views")
    with pytest.raises(error, match=match):

        @T.prim_func
        def main():
            with T.Kernel(1, 1, threads=1):
                src = T.alloc_shared((32, 64), "float32")
                if alias_kind == "same_buffer":
                    dst = src
                elif alias_kind == "same_data":
                    dst = T.Tensor((32, 64), "float32", data=src.data, scope="shared.dyn")
                elif alias_kind == "match_buffer":
                    dst = T.match_buffer(src[0:32, 0:64], (32, 64), "float32", scope="shared.dyn")
                else:
                    dst = src[0:32, 0:32]
                if api == "all_gather":
                    T.comm.all_gather(src=src, dst=dst, group=T.comm.CoreRange((0, 0), (1, 1)), axis=0)
                else:
                    T.comm.all_reduce(src=src, dst=dst, group=T.comm.CoreRange((0, 0), (1, 1)))


@pytest.mark.parametrize("kwargs", [{"elem_offset": 32}, {"strides": (128, 1)}, {"axis_separators": [1]}])
def test_collective_rejects_noncompact_storage(kwargs):
    src = tirx.decl_buffer((32, 64), "float32", scope="shared", **kwargs)
    dst = tirx.decl_buffer((32, 64), "float32", scope="shared")
    with pytest.raises(ValueError, match="compact"):

        @T.prim_func
        def main():
            with T.Kernel(1, 1, threads=1):
                T.comm.all_reduce(src=src, dst=dst, group=T.comm.CoreRange((0, 0), (1, 1)))


@pytest.mark.parametrize("api", ["all_gather", "all_reduce"])
def test_collective_requires_kernel_scope(api):
    with pytest.raises(RuntimeError, match="inside T.Kernel"):

        @T.prim_func
        def main():
            src = T.alloc_shared((32, 64), "float32")
            dst = T.alloc_shared((32, 64), "float32")
            if api == "all_gather":
                T.comm.all_gather(src=src, dst=dst, group=T.comm.CoreRange((0, 0), (1, 1)), axis=0)
            else:
                T.comm.all_reduce(src=src, dst=dst, group=T.comm.CoreRange((0, 0), (1, 1)))


@pytest.mark.parametrize("api", ["all_gather", "all_reduce"])
def test_collective_evaluate_retains_source_span(monkeypatch, api):
    from tilelang.ir import get_stmt_span, span_to_location

    monkeypatch.setenv("TILELANG_ENABLE_IR_SPAN", "1")
    locations = []

    def collect(node):
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call) and node.value.op.name == "tl.tt." + api:
            locations.append(span_to_location(get_stmt_span(node)))

    tirx.stmt_functor.post_order_visit(_program(api=api).body, collect)
    assert len(locations) == 1
    assert locations[0] is not None and locations[0][0] == __file__ and locations[0][1] > 0

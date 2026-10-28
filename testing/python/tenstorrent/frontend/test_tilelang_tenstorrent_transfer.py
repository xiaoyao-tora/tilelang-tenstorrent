"""Verify transfer capture without Tenstorrent lowering or device execution."""

import pytest
from tilelang.tenstorrent import language as T
from tvm import ir, tirx


def _transfers(func):
    calls = []

    def collect(node):
        if isinstance(node, tirx.Call) and isinstance(node.op, ir.Op) and node.op.name == "tl.tt.transfer":
            calls.append(node)

    tirx.stmt_functor.post_order_visit(func.body, collect)
    return calls


def _program(*, api="transfer", src_core=(0, 0), dst_cores=(1, 0), shape=(32, 64), dst_shape=None, dst_dtype="float32", grid=(2, 1)):
    dst_shape = shape if dst_shape is None else dst_shape

    @T.prim_func
    def main():
        with T.Kernel(*grid, threads=1):
            src = T.alloc_shared(shape, "float32")
            dst = T.alloc_shared(dst_shape, dst_dtype)
            if api == "put":
                T.comm.put(src=src, dst=dst, src_core=src_core, dst_core=dst_cores)
            elif api == "broadcast":
                T.comm.broadcast(src=src, dst=dst, src_core=src_core, dst_cores=dst_cores)
            else:
                T.comm.transfer(src=src, dst=dst, src_core=src_core, dst_cores=dst_cores)

    return main


def test_point_and_singleton_collective_keep_distinct_contracts():
    point = _transfers(_program())[0]
    collective = _transfers(_program(dst_cores=T.comm.CoreRange((1, 0), (2, 1))))[0]
    assert ir.structural_equal(point.args[2:8], collective.args[2:8])
    assert point.args[8].value == "point_to_point"
    assert collective.args[8].value == "collective"
    assert [int(coord.value) for coord in point.args[2:8]] == [0, 0, 1, 0, 2, 1]
    for index, access in ((0, 1), (1, 2)):
        region = point.args[index]
        assert region.op.name == "tl.tileop.region"
        assert int(region.args[1].value) == access
        assert [int(extent.value) for extent in region.args[2:]] == [32, 64]
    assert not point.args[0].args[0].buffer.same_as(point.args[1].args[0].buffer)


def test_symbolic_row_and_column_endpoints_survive_serialization():
    @T.prim_func
    def main():
        with T.Kernel(4, 3, threads=1) as (x, y):
            panel = T.alloc_shared((32, 32), "bfloat16")
            T.comm.transfer(src=panel, dst=panel, src_core=(0, y), dst_cores=T.comm.CoreRange((1, y), (4, y + 1)))
            T.comm.transfer(src=panel, dst=panel, src_core=(x, 0), dst_cores=T.comm.CoreRange((x, 1), (x + 1, 3)))

    row, col = _transfers(main)
    assert isinstance(row.args[3], tirx.Var)
    assert isinstance(col.args[2], tirx.Var)
    assert ir.structural_equal(row.args[3], row.args[5])
    assert ir.structural_equal(row.args[7], row.args[3] + 1)
    assert ir.structural_equal(col.args[2], col.args[4])
    assert ir.structural_equal(col.args[6], col.args[2] + 1)
    assert row.args[0].args[0].buffer.same_as(row.args[1].args[0].buffer)
    ir.assert_structural_equal(main, ir.load_json(ir.save_json(main)))
    assert main.script().count("T.tt.transfer(") == 2


def test_call_sites_and_loop_occurrences_are_preserved():
    def build():
        @T.prim_func
        def main():
            with T.Kernel(2, 1, threads=1):
                panel = T.alloc_shared((32, 32), "float32")
                for _stage in T.serial(4):
                    T.comm.transfer(src=panel, dst=panel, src_core=(0, 0), dst_cores=(1, 0))
                    T.comm.transfer(src=panel, dst=panel, src_core=(0, 0), dst_cores=(1, 0))

        return main

    func = build()
    assert [int(call.args[9].value) for call in _transfers(func)] == [0, 1]
    assert [int(call.args[9].value) for call in _transfers(build())] == [0, 1]
    loops = []
    tirx.stmt_functor.post_order_visit(func.body, lambda node: loops.append(node) if isinstance(node, tirx.For) else None)
    serial = next(loop for loop in loops if isinstance(loop.extent, tirx.IntImm) and loop.extent.value == 4)
    assert len(serial.body.seq) == 2
    assert int(ir.Op.get("tl.tt.transfer").get_attr("TCallEffectKind")) == tirx.CallEffectKind.Opaque.value
    simplified = tirx.transform.Simplify()(ir.IRModule({"main": func}))["main"]
    assert len(_transfers(simplified)) == 2


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"src_core": (True, 0)}, TypeError, "scalar integer"),
        ({"src_core": (0.5, 0)}, TypeError, "scalar integer"),
        ({"src_core": (tirx.Var("x", "float32"), 0)}, TypeError, "scalar integer"),
        ({"src_core": (tirx.Broadcast(tirx.const(0, "int32"), 2), 0)}, TypeError, "scalar integer"),
        ({"dst_cores": [1, 0]}, TypeError, "two-element"),
        ({"dst_cores": (-1, 0)}, ValueError, "non-negative"),
        ({"src_core": (2, 0)}, ValueError, "outside T.Kernel grid"),
        ({"dst_cores": (2, 0)}, ValueError, "outside T.Kernel grid"),
        ({"dst_cores": T.comm.CoreRange((1, 0), (3, 1))}, ValueError, "outside T.Kernel grid"),
        ({"dst_cores": (0, 0)}, ValueError, "source-exclusive"),
        ({"dst_cores": T.comm.CoreRange((0, 0), (2, 1))}, ValueError, "source-exclusive"),
        ({"shape": (32,)}, ValueError, "rank-2"),
        ({"shape": (48, 32)}, ValueError, "static payload extents"),
        ({"dst_shape": (32, 32)}, ValueError, "shape/dtype mismatch"),
        ({"dst_dtype": "bfloat16"}, ValueError, "shape/dtype mismatch"),
        ({"grid": (2,)}, ValueError, "2-D"),
    ],
)
def test_invalid_transfer_arguments(kwargs, error, match):
    with pytest.raises(error, match=match):
        _program(**kwargs)


@pytest.mark.parametrize("scope", ["local", "local.fragment"])
def test_transfer_rejects_non_shared_payloads(scope):
    with pytest.raises(ValueError, match="shared or shared.dyn"):

        @T.prim_func
        def main():
            with T.Kernel(2, 1, threads=1):
                panel = T.alloc_buffer((32, 32), "float32", scope=scope)
                T.comm.transfer(src=panel, dst=panel, src_core=(0, 0), dst_cores=(1, 0))


def test_transfer_rejects_partial_payloads():
    with pytest.raises(TypeError, match="complete tirx.Buffer"):

        @T.prim_func
        def main():
            with T.Kernel(2, 1, threads=1):
                panel = T.alloc_shared((64, 64), "float32")
                T.comm.transfer(src=panel[0:32, 0:32], dst=panel, src_core=(0, 0), dst_cores=(1, 0))


def test_transfer_requires_kernel_scope():
    with pytest.raises(RuntimeError, match="inside T.Kernel"):

        @T.prim_func
        def main():
            panel = T.alloc_shared((32, 32), "float32")
            T.comm.transfer(src=panel, dst=panel, src_core=(0, 0), dst_cores=(1, 0))


def test_symbolic_ranges_do_not_change_static_pipe_contract():
    y = tirx.Var("y", "int32")
    with pytest.raises(ValueError, match="non-empty"):
        T.comm.CoreRange((1, y), (2, y))
    with pytest.raises(TypeError, match="compile-time integers"):
        T.comm.Pipe((0, 0), T.comm.CoreRange((1, y), (2, y + 1)))


@pytest.mark.parametrize(
    "api, destination, grid, kind",
    [
        ("put", (1, 0), (2, 1), "point_to_point"),
        ("broadcast", T.comm.CoreRange((1, 0), (2, 1)), (2, 1), "collective"),
        ("broadcast", T.comm.CoreRange((1, 0), (4, 2)), (4, 2), "collective"),
    ],
)
def test_sugar_matches_transfer_ir(api, destination, grid, kind):
    direct = _program(dst_cores=destination, grid=grid)
    sugar = _program(api=api, dst_cores=destination, grid=grid)
    ir.assert_structural_equal(direct, sugar)
    calls = _transfers(sugar)
    assert len(calls) == 1
    assert calls[0].args[8].value == kind
    assert int(calls[0].args[9].value) == 0


def test_mixed_sugar_preserves_symbolic_endpoints_and_loop_identity():
    def build(sugar):
        @T.prim_func
        def main():
            with T.Kernel(4, 3, threads=1) as (x, y):
                panel = T.alloc_shared((32, 32), "bfloat16")
                for _stage in T.serial(4):
                    if sugar:
                        T.comm.put(src=panel, dst=panel, src_core=(0, y), dst_core=(1, y))
                        T.comm.broadcast(src=panel, dst=panel, src_core=(x, 0), dst_cores=T.comm.CoreRange((x, 1), (x + 1, 3)))
                    else:
                        T.comm.transfer(src=panel, dst=panel, src_core=(0, y), dst_cores=(1, y))
                        T.comm.transfer(src=panel, dst=panel, src_core=(x, 0), dst_cores=T.comm.CoreRange((x, 1), (x + 1, 3)))
                    T.comm.transfer(src=panel, dst=panel, src_core=(0, y), dst_cores=(1, y))

        return main

    direct, sugar = build(False), build(True)
    ir.assert_structural_equal(direct, sugar)
    ir.assert_structural_equal(sugar, ir.load_json(ir.save_json(sugar)))
    calls = _transfers(sugar)
    assert [int(call.args[9].value) for call in calls] == [0, 1, 2]
    assert [call.args[8].value for call in calls] == ["point_to_point", "collective", "point_to_point"]
    assert all(call.args[0].args[0].buffer.same_as(call.args[1].args[0].buffer) for call in calls)


@pytest.mark.parametrize(
    "api, destination, match",
    [
        ("put", T.comm.CoreRange((1, 0), (2, 1)), "put.dst_core.*two-element"),
        ("put", [1, 0], "put.dst_core.*two-element"),
        ("put", (True, 0), "put.dst_core.*scalar integer"),
        ("broadcast", (1, 0), "broadcast.*CoreRange"),
        ("broadcast", [(1, 0)], "broadcast.*CoreRange"),
    ],
)
def test_sugar_rejects_wrong_destination_types(api, destination, match):
    with pytest.raises(TypeError, match=match):
        _program(api=api, dst_cores=destination)


@pytest.mark.parametrize("api", ["put", "broadcast"])
@pytest.mark.parametrize("bad_payload", [False, True])
def test_sugar_retains_shared_transfer_validation(api, bad_payload):
    destination = (1, 0) if api == "put" else T.comm.CoreRange((1, 0), (2, 1))
    kwargs = {"dst_dtype": "bfloat16"} if bad_payload else {"src_core": (1, 0)}
    match = "payload shape/dtype mismatch" if bad_payload else "source-exclusive"
    with pytest.raises(ValueError, match=match):
        _program(api=api, dst_cores=destination, **kwargs)

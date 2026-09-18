import json

import pytest
from tvm import ir, tirx
from tvm.ir import Op

from tilelang.tenstorrent import language as T


def test_comm_namespace_is_exported():
    assert "comm" in T.__all__
    assert "tt" not in T.__all__
    assert set(T.comm.__all__) == {
        "CoreRange",
        "Pipe",
        "PipeNet",
        "foreach_dst",
        "foreach_src",
        "is_active",
        "is_dst",
        "is_src",
        "pipe_dst",
        "pipe_dst_range",
        "pipe_src",
    }
    assert callable(T.comm_reducer)


def _collect_calls_and_foreach_loops(func):
    calls = []
    loops = []

    def visit(node):
        if isinstance(node, tirx.Call) and isinstance(node.op, Op) and node.op.name.startswith("tl.tt."):
            calls.append(node)
        if isinstance(node, tirx.For) and ("tl.tt.foreach_src" in node.annotations or "tl.tt.foreach_dst" in node.annotations):
            loops.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls, loops


def _collect_op_calls(func, op_name):
    calls = []

    def visit(node):
        if isinstance(node, tirx.Call) and isinstance(node.op, Op) and node.op.name == op_name:
            calls.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls


def test_topology_primitives_construct_ordered_ir():
    gather = T.comm.PipeNet(
        [
            T.comm.Pipe(src=(1, 0), dst=(0, 0)),
            T.comm.Pipe(src=(2, 0), dst=(0, 0)),
            T.comm.Pipe(src=(2, 0), dst=(0, 0)),
        ]
    )

    @T.prim_func
    def main():
        with T.Kernel(4, 1, threads=1):
            if T.comm.is_src(gather):
                T.evaluate(0)
            if T.comm.is_dst(gather):
                T.evaluate(0)
            if T.comm.is_active(gather):
                T.evaluate(0)
            for pipe in T.comm.foreach_src(gather):
                src_x, src_y = T.comm.pipe_src(pipe)
                dst_x, dst_y = T.comm.pipe_dst(pipe)
                T.evaluate(src_x + src_y + dst_x + dst_y)
            for pipe in T.comm.foreach_dst(gather):
                src_x, _ = T.comm.pipe_src(pipe)
                T.evaluate(src_x)

    calls, loops = _collect_calls_and_foreach_loops(main)
    assert [call.op.name for call in calls].count("tl.tt.is_src") == 1
    assert [call.op.name for call in calls].count("tl.tt.is_dst") == 1
    assert [call.op.name for call in calls].count("tl.tt.is_active") == 1
    assert [call.op.name for call in calls].count("tl.tt.pipe_src") == 4
    assert [call.op.name for call in calls].count("tl.tt.pipe_dst") == 2
    assert len(loops) == 2

    descriptor = json.loads(str(loops[0].annotations["tl.tt.foreach_src"]))
    assert descriptor["kind"] == "point_to_point"
    assert descriptor["pipes"][1] == descriptor["pipes"][2]
    assert [pipe["src"] for pipe in descriptor["pipes"]] == [[1, 0], [2, 0], [2, 0]]


def test_collective_range_accessor_constructs_ir():
    broadcast = T.comm.PipeNet([T.comm.Pipe(src=(0, 0), dst=T.comm.CoreRange(begin=(1, 0), end=(4, 1)))])

    @T.prim_func
    def main():
        with T.Kernel(4, 1, threads=1):
            for pipe in T.comm.foreach_src(broadcast):
                begin, end = T.comm.pipe_dst_range(pipe)
                T.evaluate(begin[0] + begin[1] + end[0] + end[1])

    calls, loops = _collect_calls_and_foreach_loops(main)
    assert [call.op.name for call in calls].count("tl.tt.pipe_dst_range") == 4
    descriptor = json.loads(str(loops[0].annotations["tl.tt.foreach_src"]))
    assert descriptor["kind"] == "collective"
    assert descriptor["pipes"][0]["dst"] == {"begin": [1, 0], "end": [4, 1]}


@pytest.mark.parametrize(
    "factory, error, match",
    [
        (lambda: T.comm.CoreRange((0, 0), (0, 1)), ValueError, "non-empty"),
        (lambda: T.comm.Pipe(src=(0, 0), dst=(True, 1)), TypeError, "compile-time integers"),
        (lambda: T.comm.PipeNet([]), ValueError, "at least one"),
        (
            lambda: T.comm.PipeNet(
                [
                    T.comm.Pipe(src=(0, 0), dst=(1, 0)),
                    T.comm.Pipe(src=(0, 0), dst=T.comm.CoreRange((1, 0), (2, 1))),
                ]
            ),
            ValueError,
            "cannot mix",
        ),
    ],
)
def test_invalid_topologies_fail_at_construction(factory, error, match):
    with pytest.raises(error, match=match):
        factory()


def test_topology_must_fit_static_2d_kernel_grid():
    outside = T.comm.PipeNet([T.comm.Pipe(src=(0, 0), dst=(2, 0))])

    with pytest.raises(ValueError, match="outside T.Kernel grid"):

        @T.prim_func
        def outside_grid():
            with T.Kernel(2, 1, threads=1):
                T.evaluate(T.comm.is_dst(outside))

    with pytest.raises(ValueError, match="require a 2-D"):

        @T.prim_func
        def one_dimensional_grid():
            with T.Kernel(2, threads=1):
                T.evaluate(T.comm.is_dst(outside))


def test_selected_pipe_accessors_validate_region_and_contract():
    point_to_point = T.comm.PipeNet([T.comm.Pipe(src=(0, 0), dst=(1, 0))])
    collective = T.comm.PipeNet([T.comm.Pipe(src=(0, 0), dst=T.comm.CoreRange(begin=(1, 0), end=(2, 1)))])

    with pytest.raises(ValueError, match="requires a collective"):

        @T.prim_func
        def wrong_contract():
            with T.Kernel(2, 1, threads=1):
                for pipe in T.comm.foreach_src(point_to_point):
                    T.comm.pipe_dst_range(pipe)

    with pytest.raises(ValueError, match="requires a point-to-point"):

        @T.prim_func
        def other_wrong_contract():
            with T.Kernel(2, 1, threads=1):
                for pipe in T.comm.foreach_src(collective):
                    T.comm.pipe_dst(pipe)


def test_copy_constructs_pipe_send_and_recv_with_complete_regions():
    net = T.comm.PipeNet([T.comm.Pipe(src=(0, 0), dst=(1, 0))])

    @T.prim_func
    def main():
        with T.Kernel(2, 1, threads=1):
            send_block = T.alloc_shared((4, 8), T.float32)
            recv_block = T.alloc_shared((4, 8), T.float32)
            for pipe in T.comm.foreach_src(net):
                T.copy(send_block, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, recv_block)

    calls, loops = _collect_calls_and_foreach_loops(main)
    send = next(call for call in calls if call.op.name == "tl.tt.pipe_send")
    recv = next(call for call in calls if call.op.name == "tl.tt.pipe_recv")
    assert send.args[1].same_as(loops[0].loop_var)
    assert recv.args[0].same_as(loops[1].loop_var)

    send_region = send.args[0]
    recv_region = recv.args[1]
    for region, access_type in ((send_region, 1), (recv_region, 2)):
        assert isinstance(region, tirx.Call)
        assert region.op.name == "tl.tileop.region"
        assert isinstance(region.args[0], tirx.BufferLoad)
        assert region.args[0].buffer.scope() == "shared.dyn"
        assert all(ir.structural_equal(index, tirx.IntImm("int32", 0)) for index in region.args[0].indices)
        assert int(region.args[1].value) == access_type
        assert [int(extent.value) for extent in region.args[2:]] == [4, 8]


def test_copy_without_pipe_ref_delegates_to_common_copy():
    @T.prim_func
    def main():
        with T.Kernel(1, 1, threads=1):
            src = T.alloc_shared((4, 8), T.float32)
            dst = T.alloc_shared((4, 8), T.float32)
            T.copy(src, dst, annotations={"test.copy": 1})

    calls = _collect_op_calls(main, "tl.tileop.copy")
    assert len(calls) == 1
    assert calls[0].annotations["test.copy"] == 1


def test_copy_rejects_wrong_pipe_direction():
    net = T.comm.PipeNet([T.comm.Pipe(src=(0, 0), dst=(1, 0))])

    with pytest.raises(ValueError, match="send requires.*foreach_src"):

        @T.prim_func
        def send_from_destination():
            with T.Kernel(2, 1, threads=1):
                block = T.alloc_shared((4, 8), T.float32)
                for pipe in T.comm.foreach_dst(net):
                    T.copy(block, pipe)

    with pytest.raises(ValueError, match="receive requires.*foreach_dst"):

        @T.prim_func
        def receive_at_source():
            with T.Kernel(2, 1, threads=1):
                block = T.alloc_shared((4, 8), T.float32)
                for pipe in T.comm.foreach_src(net):
                    T.copy(pipe, block)


def test_copy_rejects_pipe_ref_after_foreach_region():
    net = T.comm.PipeNet([T.comm.Pipe(src=(0, 0), dst=(1, 0))])
    captured = []

    def capture(pipe):
        captured.append(pipe)
        return 0

    with pytest.raises(ValueError, match="only be used inside"):

        @T.prim_func
        def escaped_pipe():
            with T.Kernel(2, 1, threads=1):
                block = T.alloc_shared((4, 8), T.float32)
                for pipe in T.comm.foreach_src(net):
                    T.evaluate(capture(pipe))
                T.copy(block, captured[0])


def test_copy_rejects_global_and_partial_pipe_payloads():
    net = T.comm.PipeNet([T.comm.Pipe(src=(0, 0), dst=(1, 0))])

    with pytest.raises(ValueError, match="shared or shared.dyn"):

        @T.prim_func
        def global_payload(src: T.Tensor((4, 8), T.float32)):
            with T.Kernel(2, 1, threads=1):
                for pipe in T.comm.foreach_src(net):
                    T.copy(src, pipe)

    with pytest.raises(TypeError, match="complete tirx.Buffer"):

        @T.prim_func
        def partial_payload():
            with T.Kernel(2, 1, threads=1):
                block = T.alloc_shared((4, 8), T.float32)
                for pipe in T.comm.foreach_src(net):
                    T.copy(block[0, 0], pipe)


def test_copy_rejects_pipe_to_pipe_transfer():
    net = T.comm.PipeNet([T.comm.Pipe(src=(0, 0), dst=(1, 0))])

    with pytest.raises(TypeError, match="PipeRef-to-PipeRef"):

        @T.prim_func
        def pipe_to_pipe():
            with T.Kernel(2, 1, threads=1):
                for send_pipe in T.comm.foreach_src(net):
                    for recv_pipe in T.comm.foreach_dst(net):
                        T.copy(send_pipe, recv_pipe)


@pytest.mark.parametrize(
    "recv_shape, recv_dtype",
    [
        ((4, 16), "float32"),
        ((4, 8), "int32"),
    ],
)
def test_copy_rejects_pipenet_payload_shape_or_dtype_mismatch(recv_shape, recv_dtype):
    net = T.comm.PipeNet([T.comm.Pipe(src=(0, 0), dst=(1, 0))])

    with pytest.raises(ValueError, match="payload shape/dtype mismatch"):

        @T.prim_func
        def mismatched_payload():
            with T.Kernel(2, 1, threads=1):
                send_block = T.alloc_shared((4, 8), T.float32)
                recv_block = T.alloc_shared(recv_shape, recv_dtype)
                for pipe in T.comm.foreach_src(net):
                    T.copy(send_block, pipe)
                for pipe in T.comm.foreach_dst(net):
                    T.copy(pipe, recv_block)


def test_copy_rejects_non_default_options_for_pipe_transfer():
    net = T.comm.PipeNet([T.comm.Pipe(src=(0, 0), dst=(1, 0))])

    with pytest.raises(ValueError, match="non-default copy options"):

        @T.prim_func
        def configured_pipe_copy():
            with T.Kernel(2, 1, threads=1):
                block = T.alloc_shared((4, 8), T.float32)
                for pipe in T.comm.foreach_src(net):
                    T.copy(block, pipe, coalesced_width=4)

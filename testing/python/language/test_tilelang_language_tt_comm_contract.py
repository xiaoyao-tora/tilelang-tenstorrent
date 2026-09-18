"""Source-level communication contracts, independent of device execution."""

import json

import pytest
from tilelang.tenstorrent import language as T
from tvm import tirx


def _net():
    return T.comm.PipeNet([T.comm.Pipe((0, 0), (1, 0))])


def _transfer(shape=(32, 32), grid=(2, 1), receive_shape=None, receive_dtype="float32"):
    net = _net()
    receive_shape = shape if receive_shape is None else receive_shape

    @T.prim_func
    def program():
        with T.Kernel(*grid, threads=1):
            send = T.alloc_shared(shape, "float32")
            recv = T.alloc_shared(receive_shape, receive_dtype)
            for pipe in T.comm.foreach_src(net):
                T.copy(send, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, recv)

    return program


@pytest.mark.parametrize("shape", [(0, 32), (-32, 32), (48, 32), (32,), (1, 32, 32), (tirx.Var("n", "int32"), 32)])
def test_pipe_payload_requires_static_complete_tiles(shape):
    with pytest.raises(ValueError, match="payload.*(rank|extent)|rank-2 payload"):
        _transfer(shape=shape)


@pytest.mark.parametrize("grid", [(0, 1), (-1, 1), (tirx.Var("n", "int32"), 1)])
def test_pipe_grid_requires_positive_static_extents(grid):
    with pytest.raises(ValueError, match="positive compile-time integer.*grid"):
        _transfer(grid=grid)


@pytest.mark.parametrize("kwargs", [{"receive_shape": (64, 32)}, {"receive_dtype": "bfloat16"}])
def test_pipe_payload_contract_is_shared_by_send_and_receive(kwargs):
    with pytest.raises(ValueError, match="payload shape/dtype mismatch"):
        _transfer(**kwargs)


def test_pipenet_identity_is_operation_local_and_records_keep_multiplicity():
    first = T.comm.PipeNet([T.comm.Pipe((0, 0), (1, 0)), T.comm.Pipe((0, 0), (1, 0))])
    second = T.comm.PipeNet(first.pipes)

    def build():
        @T.prim_func
        def program():
            with T.Kernel(2, 1, threads=1):
                payload = T.alloc_shared((32, 32), "float32")
                for pipe in T.comm.foreach_src(first):
                    T.copy(payload, pipe)
                for pipe in T.comm.foreach_src(second):
                    T.copy(payload, pipe)
                for pipe in T.comm.foreach_dst(first):
                    T.copy(pipe, payload)

        descriptors = []

        def collect(node):
            if isinstance(node, tirx.For):
                for key in ("tl.tt.foreach_src", "tl.tt.foreach_dst"):
                    if key in node.annotations:
                        descriptors.append(json.loads(str(node.annotations[key])))

        tirx.stmt_functor.post_order_visit(program.body, collect)
        return descriptors

    descriptors = build()
    assert [item["id"] for item in descriptors] == [0, 1, 0]
    assert all(len(item["pipes"]) == 2 for item in descriptors)
    assert descriptors == build()


@pytest.mark.parametrize("side", ["src", "dst"])
def test_pipe_ref_cannot_escape_its_foreach_region(side):
    net = _net()
    saved = []

    def remember(pipe):
        saved.append(pipe)
        return T.evaluate(0)

    foreach = T.comm.foreach_src if side == "src" else T.comm.foreach_dst
    with pytest.raises(ValueError, match="only be used inside"):

        @T.prim_func
        def program():
            with T.Kernel(2, 1, threads=1):
                payload = T.alloc_shared((32, 32), "float32")
                for pipe in foreach(net):
                    remember(pipe)
                T.copy(payload, saved[0])


@pytest.mark.parametrize("side", ["src", "dst"])
def test_pipe_ref_enforces_endpoint_role(side):
    net = _net()
    foreach = T.comm.foreach_src if side == "src" else T.comm.foreach_dst
    with pytest.raises(ValueError, match="requires a PipeRef from"):

        @T.prim_func
        def program():
            with T.Kernel(2, 1, threads=1):
                payload = T.alloc_shared((32, 32), "float32")
                for pipe in foreach(net):
                    if side == "src":
                        T.copy(pipe, payload)
                    else:
                        T.copy(payload, pipe)

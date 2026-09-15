from __future__ import annotations

import pytest
from tvm import ir, tirx
from tvm.ir import Op

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import transform

from testing.python.target.test_tilelang_tenstorrent_phase0_contract import (
    FRONTEND_PROGRAMS,
)


TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})


def _shared(shape=(32, 32), dtype=T.bfloat16, block_count=2):
    return T.alloc_shared(
        shape,
        dtype,
        annotations={
            "tt.dfb_block_count": block_count,
            "tt.tile_shape": (32, 32),
        },
    )


@T.prim_func
def float32_add(
    A: T.Tensor((32, 32), T.float32),
    B: T.Tensor((32, 32), T.float32),
    C: T.Tensor((32, 32), T.float32),
):
    with T.Kernel(1, 1, threads=1):
        a = _shared(dtype=T.float32)
        b = _shared(dtype=T.float32)
        c = _shared(dtype=T.float32)
        T.copy(A, a)
        T.copy(B, b)
        for i, j in T.Parallel(32, 32):
            c[i, j] = a[i, j] + b[i, j]
        T.copy(c, C)


@T.prim_func
def subtract(
    A: T.Tensor((32, 32), T.bfloat16),
    B: T.Tensor((32, 32), T.bfloat16),
    C: T.Tensor((32, 32), T.bfloat16),
):
    with T.Kernel(1, 1, threads=1):
        a = _shared()
        b = _shared()
        c = _shared()
        T.copy(A, a)
        T.copy(B, b)
        for i, j in T.Parallel(32, 32):
            c[i, j] = a[i, j] - b[i, j]
        T.copy(c, C)


@T.prim_func
def broadcast_add(
    A: T.Tensor((32, 32), T.bfloat16),
    B: T.Tensor((32, 32), T.bfloat16),
    C: T.Tensor((32, 32), T.bfloat16),
):
    with T.Kernel(1, 1, threads=1):
        a = _shared()
        b = _shared()
        c = _shared()
        T.copy(A, a)
        T.copy(B, b)
        for i, j in T.Parallel(32, 32):
            c[i, j] = a[0, j] + b[i, j]
        T.copy(c, C)


@T.prim_func
def partial_copy_add(
    A: T.Tensor((32, 32), T.bfloat16),
    B: T.Tensor((32, 32), T.bfloat16),
    C: T.Tensor((32, 32), T.bfloat16),
):
    with T.Kernel(1, 1, threads=1):
        a = _shared()
        b = _shared()
        c = _shared()
        T.copy(A[0:16, 0:32], a[0:16, 0:32])
        T.copy(B, b)
        for i, j in T.Parallel(32, 32):
            c[i, j] = a[i, j] + b[i, j]
        T.copy(c, C)


@T.prim_func
def single_buffer_add(
    A: T.Tensor((32, 32), T.bfloat16),
    B: T.Tensor((32, 32), T.bfloat16),
    C: T.Tensor((32, 32), T.bfloat16),
):
    with T.Kernel(1, 1, threads=1):
        a = _shared(block_count=1)
        b = _shared()
        c = _shared()
        T.copy(A, a)
        T.copy(B, b)
        for i, j in T.Parallel(32, 32):
            c[i, j] = a[i, j] + b[i, j]
        T.copy(c, C)


def _normalized(func):
    mod = tirx.transform.BindTarget(TARGET)(tvm.IRModule({"main": func}))
    for compiler_pass in (
        transform.CanonicalizeTTElementwise(),
        transform.VerifyTTComputeBlocks(),
        transform.ValidateTenstorrentFrontendIR(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
        transform.NormalizeTenstorrentRegions(),
        transform.InferTenstorrentTensorLayout(),
    ):
        mod = compiler_pass(mod)
    return mod


def _calls(func):
    calls = []

    def collect(node):
        if isinstance(node, tirx.Call) and isinstance(node.op, Op):
            calls.append(node)

    tirx.stmt_functor.post_order_visit(func.body, collect)
    return calls


def _nodes(func, node_type):
    nodes = []
    tirx.stmt_functor.post_order_visit(func.body, lambda node: nodes.append(node) if isinstance(node, node_type) else None)
    return nodes


@pytest.mark.parametrize(
    ("func", "dtype"),
    (
        (FRONTEND_PROGRAMS["add"], "bfloat16"),
        (float32_add, "float32"),
    ),
)
def test_legalize_structured_add_to_one_canonical_tile_operation(func, dtype):
    before = _normalized(func)
    before_table = before["main"].attrs["tt.buffer_metadata_table"]

    legalized = transform.LegalizeTenstorrentTileOps()(before)
    result = legalized["main"]
    calls = _calls(result)
    tile_adds = [call for call in calls if call.op.name == "tl.tt.tile_add"]

    assert len(tile_adds) == 1
    assert [call.op.name for call in calls].count("tl.tileop.copy") == 3
    assert not [loop for loop in _nodes(result, tirx.For) if loop.kind == tirx.ForKind.PARALLEL]
    assert not _nodes(result, tirx.BufferStore)

    tile_add = tile_adds[0]
    assert tile_add.dtype == "handle"
    assert len(tile_add.args) == 3
    assert all(isinstance(argument, tirx.BufferLoad) for argument in tile_add.args)
    assert str(tile_add.annotations["tt.compute_dtype"].value) == dtype
    assert [int(value) for value in tile_add.annotations["tt.compute_tile_shape"]] == [
        32,
        32,
    ]
    assert tile_add.span is not None

    after_table = result.attrs["tt.buffer_metadata_table"]
    assert len(after_table) == len(before_table) == 6
    for before_item, after_item in zip(before_table, after_table):
        assert before_item.same_as(after_item)


def test_legalize_is_idempotent_deterministic_and_json_round_trips():
    once = transform.LegalizeTenstorrentTileOps()(_normalized(FRONTEND_PROGRAMS["add"]))
    twice = transform.LegalizeTenstorrentTileOps()(once)

    ir.assert_structural_equal(twice, once)
    assert ir.structural_hash(twice) == ir.structural_hash(once)
    assert str(twice) == str(once)
    restored = ir.load_json(ir.save_json(once))
    ir.assert_structural_equal(restored, once)


@pytest.mark.parametrize("func", [subtract, broadcast_add])
def test_legalize_consumes_subtract_and_broadcast(func):
    normalized = _normalized(func)
    result = transform.LegalizeTenstorrentTileOps()(normalized)
    assert not any("tl.tt.compute_kind" in node.annotations for node in _nodes(result["main"], tirx.SBlock))
    assert sum(call.op.name == "tl.tt.tile_compute" for call in _calls(result["main"])) == 1


@pytest.mark.parametrize("func", [partial_copy_add])
def test_legalize_leaves_transaction_constraints_to_device_formation(func):
    # Compute instruction selection does not inspect surrounding copies or ABI.
    result = transform.LegalizeTenstorrentTileOps()(_normalized(func))
    assert sum(call.op.name == "tl.tt.tile_add" for call in _calls(result["main"])) == 1


def test_legalize_consumes_fill_tileop():
    @T.prim_func
    def fill():
        with T.Kernel(1, 1, threads=1):
            a = _shared()
            T.fill(a, 0)

    normalized = _normalized(fill)
    legalized = transform.LegalizeTenstorrentTileOps()(normalized)
    assert sum(call.op.name == "tl.tt.tile_compute" for call in _calls(legalized["main"])) == 1


def test_topology_capabilities_are_checked_by_the_topology_pass():
    normalized = _normalized(FRONTEND_PROGRAMS["p2p"])
    ir.assert_structural_equal(transform.LegalizeTenstorrentTileOps()(normalized), normalized)
    with pytest.raises(NotImplementedError, match="NormalizeTenstorrentTopology"):
        transform.NormalizeTenstorrentTopology()(normalized)


def test_partial_copy_is_rejected_at_transaction_planning_boundary():
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    with pytest.raises(ValueError, match="FormTenstorrentDeviceProgram.*regions disagree"):
        TenstorrentPassPipelineBody(tvm.IRModule({"main": partial_copy_add}), TARGET)


def test_one_block_dfb_capacity_is_not_an_add_frontend_restriction():
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    result = TenstorrentPassPipelineBody(tvm.IRModule({"main": single_buffer_add}), TARGET)
    assert int(result.attrs["tt.device_ir_version"]) == 2
    assert "tt.ir_stage" not in result.attrs
    transform.VerifyTenstorrentDeviceIR()(result)

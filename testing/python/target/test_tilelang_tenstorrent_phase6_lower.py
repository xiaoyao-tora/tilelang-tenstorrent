"""Static multi-Core/PipeNet Device Lower; no execution backend is invoked."""

import json

import pytest
import tvm_ffi

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
from tvm import ir, tirx


def shared():
    return T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 1})


def lower_multicore(func, arch="wormhole_b0"):
    return TenstorrentPassPipelineBody(tvm.IRModule({"main": func}), tvm.target.Target({"kind": "tenstorrent", "arch": arch}))


def make_point_to_point():
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), (1, 0))])

    @T.prim_func
    def program(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(3, 1, threads=1):
            a = shared()
            if T.comm.is_src(net):
                T.copy(A, a)
            for pipe in T.comm.foreach_src(net):
                T.copy(a, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, a)
                T.copy(a, C)

    return program


def make_scatter(duplicate=False):
    records = [T.comm.Pipe((0, 0), (1, 0)), T.comm.Pipe((0, 0), (2, 0))]
    if duplicate:
        records.append(T.comm.Pipe((0, 0), (1, 0)))
    net = T.comm.PipeNet(records)

    @T.prim_func
    def program(C: T.Tensor((96, 32), "float32")):
        with T.Kernel(3, 1, threads=1) as (x, y):
            a = shared()
            for pipe in T.comm.foreach_src(net):
                dx, dy = T.comm.pipe_dst(pipe)
                T.fill(a, T.Cast("float32", dx + pipe * 10))
                T.copy(a, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, a)
                T.copy(a, C[x * 32 : (x + 1) * 32, :])

    return program


def make_collective():
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), T.comm.CoreRange((1, 0), (3, 2)))])

    @T.prim_func
    def program(A: T.Tensor((32, 32), "float32"), C: T.Tensor((96, 64), "float32")):
        with T.Kernel(3, 2, threads=1) as (x, y):
            a = shared()
            if T.comm.is_src(net):
                T.copy(A, a)
            for pipe in T.comm.foreach_src(net):
                T.copy(a, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, a)
                T.copy(a, C[x * 32 : (x + 1) * 32, y * 32 : (y + 1) * 32])

    return program


def make_gather(reverse=False, last_only=False):
    records = [T.comm.Pipe((1, 0), (0, 0)), T.comm.Pipe((2, 0), (0, 0))]
    net = T.comm.PipeNet(list(reversed(records)) if reverse else records)

    @T.prim_func
    def program(C: T.Tensor((64, 32), "float32")):
        with T.Kernel(3, 1, threads=1) as (x, y):
            a = shared()
            if T.comm.is_src(net):
                T.fill(a, T.Cast("float32", x))
            for pipe in T.comm.foreach_src(net):
                T.copy(a, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, a)
                if not last_only:
                    T.copy(a, C[pipe * 32 : (pipe + 1) * 32, :])
            if last_only and T.comm.is_dst(net):
                T.copy(a, C[0:32, :])

    return program


def make_gemm_2d():
    rows = T.comm.PipeNet([T.comm.Pipe((0, y), (1, y)) for y in range(2)])
    columns = T.comm.PipeNet([T.comm.Pipe((x, 0), (x, 1)) for x in range(2)])

    @T.prim_func
    def program(A: T.Tensor((64, 32), "float32"), B: T.Tensor((32, 64), "float32"), C: T.Tensor((64, 64), "float32")):
        with T.Kernel(2, 2, threads=1) as (x, y):
            a = shared()
            b = shared()
            c = shared()
            if x == 0:
                T.copy(A[y * 32 : (y + 1) * 32, :], a)
            if y == 0:
                T.copy(B[:, x * 32 : (x + 1) * 32], b)
            for pipe in T.comm.foreach_src(rows):
                T.copy(a, pipe)
            for pipe in T.comm.foreach_src(columns):
                T.copy(b, pipe)
            for pipe in T.comm.foreach_dst(rows):
                T.copy(pipe, a)
            for pipe in T.comm.foreach_dst(columns):
                T.copy(pipe, b)
            T.gemm(a, b, c, clear_accum=True)
            T.copy(c, C[y * 32 : (y + 1) * 32, x * 32 : (x + 1) * 32])

    return program


def device_calls(mod):
    calls = []
    for func in mod.functions.values():
        tirx.stmt_functor.post_order_visit(func.body, lambda node: calls.append(node) if isinstance(node, tirx.Call) else None)
    return calls


@pytest.mark.parametrize(
    "factory,records,edges,cores",
    [
        (make_point_to_point, 1, 1, 3),
        (make_scatter, 2, 2, 3),
        (lambda: make_scatter(True), 3, 3, 3),
        (make_collective, 1, 4, 6),
        (make_gather, 2, 2, 3),
        (make_gemm_2d, 4, 4, 4),
    ],
)
@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
def test_complete_multicore_device_lower(factory, records, edges, cores, arch):
    func = factory()
    mod = lower_multicore(func, arch)
    assert int(mod.attrs["tt.device_ir_version"]) == 4
    assert len(mod.functions) == cores * 3
    assert len(mod.attrs["tt.pipe_table"]) == records
    assert len(mod.attrs["tt.pipe_transfer_table"]) == edges
    names = [call.op.name for call in device_calls(mod)]
    assert names.count("tl.tt.dfb_pipe_send") == edges
    assert names.count("tl.tt.dfb_pipe_recv") == edges
    assert names.count("tl.tt.dfb_pipe_wait") == edges * 2
    assert names.count("tl.tt.dfb_release") == len(mod.attrs["tt.dfb_table"])
    assert "tl.tt.pipe_send" not in names and "tl.tt.pipe_recv" not in names
    for func in mod.functions.values():
        domain = func.attrs["tt.core_domain"]
        assert int(domain.end.x) == int(domain.begin.x) + 1
        assert int(domain.end.y) == int(domain.begin.y) + 1
    repeated = lower_multicore(func=factory(), arch=arch)
    assert mod.script() == repeated.script()
    assert ir.save_json(mod) == ir.save_json(repeated)
    ir.assert_structural_equal(mod, ir.load_json(ir.save_json(mod)))


def test_zero_matching_core_is_idle_and_record_indices_are_preserved():
    mod = lower_multicore(make_point_to_point())
    for func in mod.functions.values():
        if int(func.attrs["tt.core_domain"].begin.x) == 2:
            assert isinstance(func.body, tirx.Evaluate) and int(func.body.value) == 0
    mod = lower_multicore(make_scatter(True))
    assert [int(record.event_index) for record in mod.attrs["tt.pipe_table"]] == [0, 1, 2]
    assert len({int(edge.source_dfb_id) for edge in mod.attrs["tt.pipe_transfer_table"]}) == 3


def test_pipe_decode_rejects_malformed_frozen_schema():
    decoder = tvm_ffi.get_global_func("tl.tenstorrent.DecodePipeNet")
    for data in ({}, {"id": True, "kind": "point_to_point", "pipes": []}, {"id": 0, "kind": "bad", "pipes": []}):
        with pytest.raises(ValueError, match="NormalizeTenstorrentTopology"):
            decoder(json.dumps(data))


def test_reverse_source_record_order_and_discarded_receives():
    mod = lower_multicore(make_gather(reverse=True, last_only=True))
    records = mod.attrs["tt.pipe_table"]
    assert [int(record.event_index) for record in records] == [0, 1]
    assert [int(record.src_coord.x) for record in records] == [2, 1]
    assert len(mod.attrs["tt.pipe_transfer_table"]) == 2


def test_multicore_without_pipes_has_complete_device_ir():
    @T.prim_func
    def program(C: T.Tensor((64, 32), "float32")):
        with T.Kernel(2, 1, threads=1) as (x, y):
            a = shared()
            T.fill(a, T.Cast("float32", x + 1))
            T.copy(a, C[x * 32 : (x + 1) * 32, :])

    mod = lower_multicore(program)
    assert int(mod.attrs["tt.device_ir_version"]) == 4
    assert len(mod.functions) == 6
    assert len(mod.attrs["tt.pipe_transfer_table"]) == 0


def test_unit_core_self_pipe():
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), (0, 0))])

    @T.prim_func
    def program(C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = shared()
            T.fill(a, 4)
            for pipe in T.comm.foreach_src(net):
                T.copy(a, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, a)
            T.copy(a, C)

    mod = lower_multicore(program)
    assert len(mod.attrs["tt.pipe_transfer_table"]) == 1


def test_static_core_grid_limit_and_empty_domains():
    def make_empty(width):
        @T.prim_func
        def program():
            with T.Kernel(width, 1, threads=1):
                T.evaluate(0)

        return program

    mod = lower_multicore(make_empty(256))
    assert len(mod.functions) == 256 * 3
    with pytest.raises(NotImplementedError, match="limited to 256 cores"):
        lower_multicore(make_empty(257))


def test_repeated_occurrence_of_one_record_is_diagnosed():
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), (1, 0))])

    @T.prim_func
    def program():
        with T.Kernel(2, 1, threads=1):
            a = shared()
            T.fill(a, 4)
            for pipe in T.comm.foreach_src(net):
                T.copy(a, pipe)
                T.copy(a, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, a)

    with pytest.raises(ValueError, match="repeated transaction occurrence|more than one source occurrence"):
        lower_multicore(program)


def test_missing_receive_is_diagnosed_by_complete_lower():
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), (1, 0))])

    @T.prim_func
    def program():
        with T.Kernel(2, 1, threads=1):
            a = shared()
            T.fill(a, 4)
            for pipe in T.comm.foreach_src(net):
                T.copy(a, pipe)

    with pytest.raises(ValueError, match="producer/consumer transaction count mismatch"):
        lower_multicore(program)


def test_multicore_pipeline_composition_is_explicitly_deferred():
    @T.prim_func
    def program():
        with T.Kernel(2, 1, threads=1):
            a = shared()
            for _k in T.Pipelined(2, num_stages=2):
                T.fill(a, 4)

    with pytest.raises(NotImplementedError, match="combined with T.Pipelined is deferred"):
        lower_multicore(program)


def test_topology_normalization_is_idempotent():
    from tilelang.tenstorrent import transform

    target = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
    mod = tvm.IRModule({"main": make_scatter(True)})
    for compiler_pass in (
        tirx.transform.BindTarget(target),
        transform.CanonicalizeTTElementwise(),
        transform.VerifyTTComputeBlocks(),
        transform.ValidateTenstorrentFrontendIR(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
        transform.NormalizeTenstorrentTopology(),
    ):
        mod = compiler_pass(mod)
    ir.assert_structural_equal(mod, transform.NormalizeTenstorrentTopology()(mod))


def test_active_net_cannot_silently_drop_an_original_record():
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), (1, 0)), T.comm.Pipe((0, 0), (2, 0))])

    @T.prim_func
    def program():
        with T.Kernel(3, 1, threads=1):
            a = shared()
            T.fill(a, 4)
            for pipe in T.comm.foreach_src(net):
                if pipe == 0:
                    T.copy(a, pipe)
            for pipe in T.comm.foreach_dst(net):
                if pipe == 0:
                    T.copy(pipe, a)

    with pytest.raises(ValueError, match="every original record"):
        lower_multicore(program)


def test_pipe_coordinate_accessors_are_statically_eliminated():
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), T.comm.CoreRange((0, 0), (1, 2)))])

    @T.prim_func
    def program(C: T.Tensor((64, 32), "float32")):
        with T.Kernel(1, 2, threads=1) as (x, y):
            a = shared()
            for pipe in T.comm.foreach_src(net):
                begin, end = T.comm.pipe_dst_range(pipe)
                sx, sy = T.comm.pipe_src(pipe)
                T.fill(a, T.Cast("float32", begin[0] + end[1] + sx + sy))
                T.copy(a, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, a)
                T.copy(a, C[y * 32 : (y + 1) * 32, :])

    mod = lower_multicore(program)
    names = [call.op.name for call in device_calls(mod)]
    assert "tl.tt.pipe_src" not in names and "tl.tt.pipe_dst_range" not in names
    assert len(mod.attrs["tt.pipe_transfer_table"]) == 2


def make_row_column_gemm():
    rows = T.comm.PipeNet([T.comm.Pipe((0, y), T.comm.CoreRange((1, y), (3, y + 1))) for y in (1, 2)])
    cols = T.comm.PipeNet([T.comm.Pipe((x, 0), T.comm.CoreRange((x, 1), (x + 1, 3))) for x in (1, 2)])

    @T.prim_func
    def row_column_gemm(A: T.Tensor((64, 32), "float32"), B: T.Tensor((32, 64), "float32"), C: T.Tensor((64, 64), "float32")):
        with T.Kernel(3, 3, threads=1) as (x, y):
            a = T.alloc_shared((32, 32), "float32")
            b = T.alloc_shared((32, 32), "float32")
            c = T.alloc_shared((32, 32), "float32")
            for p in T.comm.foreach_src(rows):
                sx, sy = T.comm.pipe_src(p)
                T.copy(A[(sy - 1) * 32 : sy * 32, :], a)
                T.copy(a, p)
            for p in T.comm.foreach_src(cols):
                sx, sy = T.comm.pipe_src(p)
                T.copy(B[:, (sx - 1) * 32 : sx * 32], b)
                T.copy(b, p)
            for p in T.comm.foreach_dst(rows):
                T.copy(p, a)
            for p in T.comm.foreach_dst(cols):
                T.copy(p, b)
            if x > 0 and y > 0:
                T.gemm(a, b, c, clear_accum=True)
                T.copy(c, C[(y - 1) * 32 : y * 32, (x - 1) * 32 : x * 32])

    return row_column_gemm

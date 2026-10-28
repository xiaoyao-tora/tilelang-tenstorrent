"""Lower composition regressions, including CPU execution of Device streams."""

import numpy as np
import pytest
from tilelang import tvm
from tilelang.tenstorrent import device_ir as D
from tilelang.tenstorrent import language as T, lower_tenstorrent_ir, transform
from tvm import ir

from test_tilelang_tenstorrent_utils import TARGET, _calls, _round, interpret


def lower(program, version=7):
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": program}), TARGET)
    assert str(mod.attrs["tt.ir_stage"]) == "device"
    assert "tt.device_ir_version" not in mod.attrs
    ir.assert_structural_equal(mod, ir.load_json(ir.save_json(mod)))
    assert transform.VerifyTenstorrentDeviceIR()(mod).same_as(mod)
    return D.expand_current_device_tables(mod)


def lower_result(program):
    return lower_tenstorrent_ir(tvm.IRModule({"main": program}), TARGET)


def transpose_fragment(dtype="float32", source_fragment=False):
    @T.prim_func
    def main(A: T.Tensor((32, 64), dtype), C: T.Tensor((64, 32), dtype)):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 64), dtype)
            f = T.alloc_fragment((32, 64), dtype)
            g = T.alloc_fragment((64, 32), dtype)
            T.copy(A, a)
            if source_fragment:
                T.copy(a, f)
                T.transpose(f, g)
            else:
                T.transpose(a, g)
            for i, j in T.Tiles(g):
                g[i, j] = g[i, j] + T.Cast(dtype, 1)
            T.copy(g, C)

    return main


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("source_fragment", [False, True])
def test_transpose_fragment_exact_dtype_and_reentry(dtype, source_fragment):
    mod = lower(transpose_fragment(dtype, source_fragment))
    a = _round(np.random.default_rng(7).normal(size=(32, 64)), dtype)
    outputs, _ = interpret(mod, [a, np.zeros((64, 32), "float32")])
    np.testing.assert_array_equal(outputs[1], _round(a.T + 1, dtype))
    transpose = next(
        call
        for func in mod.functions.values()
        for call in _calls(func.body)
        if call.annotations.get("tt.compute_kind") is not None and call.annotations["tt.compute_kind"].value == "transpose"
    )
    assert transpose.op.name == "tl.tt.dfb_compute"
    assert all(str(dfb.element_dtype) == dtype for dfb in mod.attrs["tt.dfb_table"])


def multicore_values():
    @T.prim_func
    def main(A: T.Tensor((64, 32), "float32"), C: T.Tensor((64, 32), "float32")):
        with T.Kernel(2, 1, threads=1) as (x, y):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 3})
            f = T.alloc_fragment((32, 32), "float32")
            T.copy(A[x * 32 : (x + 1) * 32, 0:32], a)
            for i, j in T.Tiles(f):
                f[i, j] = a[i, j] + T.float32(1)
            for i, j in T.Tiles(f):
                f[i, j] = f[i, j] * T.float32(2)
            T.copy(f, C[x * 32 : (x + 1) * 32, 0:32])

    return main


def test_multicore_values_have_isolated_versions_and_numeric_outputs():
    mod = lower(multicore_values())
    assert len(mod.functions) == 3
    assert [str(symbol) for symbol in mod.attrs["tt.kernel_order"]] == [
        "main_trisc",
        "main_ncrisc",
        "main_brisc",
    ]
    a = np.arange(64 * 32, dtype="float32").reshape(64, 32)
    outputs, _ = interpret(mod, [a, np.zeros_like(a)])
    np.testing.assert_array_equal(outputs[1], (a + 1) * 2)
    definitions = list(mod.attrs["tt.compute_value_table"])
    assert len({int(value.value_id) for value in definitions}) == len(definitions)
    # One logical Buffer family is shared by the function; Core domain plus
    # value identity distinguishes the independent local instances.
    assert definitions[0].buffer.data.same_as(definitions[-1].buffer.data)
    domains = {
        (
            int(value.core_domain.begin.x),
            int(value.core_domain.begin.y),
            int(value.core_domain.end.x),
            int(value.core_domain.end.y),
        )
        for value in definitions
    }
    assert domains == {(0, 0, 1, 1), (1, 0, 2, 1)}


def pipe_values():
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), (1, 0))])

    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(2, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 3})
            f = T.alloc_fragment((32, 32), "float32")
            if T.comm.is_src(net):
                T.copy(A, a)
                for i, j in T.Tiles(f):
                    f[i, j] = a[i, j] * T.float32(2)
                T.copy(f, a)
            for pipe in T.comm.foreach_src(net):
                T.copy(a, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, a)
                for i, j in T.Tiles(f):
                    f[i, j] = a[i, j] + T.float32(3)
                T.copy(f, C)

    return main


def test_legacy_pipe_materialized_values_require_high_level_communication():
    with pytest.raises(NotImplementedError, match="legacy Pipe/PipeRef"):
        lower_result(pipe_values())


def pipeline_values(stages=2, extent=5):
    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 3})
            f = T.alloc_fragment((32, 32), "float32")
            for k in T.Pipelined(extent, num_stages=stages):
                T.copy(A, a)
                for i, j in T.Tiles(f):
                    f[i, j] = a[i, j] + T.Cast("float32", k)
                for i, j in T.Tiles(f):
                    f[i, j] = f[i, j] * T.float32(2)
                T.copy(f, C)

    return main


@pytest.mark.parametrize("stages,extent", [(1, 1), (2, 5), (3, 7)])
def test_independent_fragment_pipeline_prefetch_and_semantics(stages, extent):
    mod = lower(pipeline_values(stages, extent))
    a = np.random.default_rng(9).normal(size=(32, 32)).astype("float32")
    outputs, _ = interpret(mod, [a, np.zeros_like(a)])
    np.testing.assert_array_equal(outputs[1], (a + (extent - 1)) * 2)
    transfer = next(func for func in mod.functions.values() if str(func.attrs["tt.kernel_slot"]) == "ncrisc")
    calls = list(_calls(transfer.body))
    first_completion = next(i for i, call in enumerate(calls) if call.op.name == "tl.tt.dfb_copy_wait")
    assert sum(call.op.name == "tl.tt.tensor_to_dfb_nd" for call in calls[:first_completion]) == min(stages, extent)
    assert len(set(map(int, mod.attrs["tt.dfb_storage_groups"].values()))) == 2


def full_k_pipeline(blocks=5, stages=2):
    @T.prim_func
    def main(A: T.Tensor((32, blocks * 32), "bfloat16"), B: T.Tensor((blocks * 32, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16")
            b = T.alloc_shared((32, 32), "bfloat16")
            acc = T.alloc_fragment((32, 32), "float32")
            T.clear(acc)
            for k in T.Pipelined(blocks, num_stages=stages):
                start = k * 32
                T.copy(A[0:32, start : start + 32], a)
                T.copy(B[start : start + 32, 0:32], b)
                T.gemm(a, b, acc)
            T.copy(acc, C)

    return main


@pytest.mark.parametrize("seed", [0, 11, 71])
def test_full_k_pipeline_preserves_fp32_and_finite_pools(seed):
    mod = lower(full_k_pipeline(), version=5)
    contributions = [512.0, 1.0, 1.0, -512.0, 0.375]
    identity = np.eye(32, dtype="float32")
    a = np.concatenate([identity] * len(contributions), axis=1)
    b = np.concatenate([identity * scale for scale in contributions], axis=0)
    result, _ = interpret(mod, [a, b, np.zeros((32, 32), "float32")], seed)
    np.testing.assert_array_equal(result[2], identity * 2.375)
    assert int(mod.attrs["tt.accumulator_table"][0].full_k_tiles) == 5
    relations = mod.attrs["tt.pipeline_relations"]
    singletons = [dfb for dfb in mod.attrs["tt.dfb_table"] if int(relations[str(int(dfb.dfb_id))][0]) == -1]
    assert len(singletons) == 1 and int(singletons[0].block_count) == 1
    assert len(set(map(int, mod.attrs["tt.dfb_storage_groups"].values()))) == 3
    transfers = [call for f in mod.functions.values() if str(f.attrs["tt.kernel_slot"]) == "ncrisc" for call in _calls(f.body)]
    first_completion = next(i for i, call in enumerate(transfers) if call.op.name == "tl.tt.dfb_copy_wait")
    assert sum(call.op.name == "tl.tt.tensor_to_dfb_nd" for call in transfers[:first_completion]) == 4


def pipeline_pipe_values(extent=5, stages=2):
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), (1, 0))])

    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(2, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2})
            f = T.alloc_fragment((32, 32), "float32")
            for k in T.Pipelined(extent, num_stages=stages):
                if T.comm.is_src(net):
                    T.copy(A, a)
                    for i, j in T.Tiles(f):
                        f[i, j] = a[i, j] + T.Cast("float32", k)
                    T.copy(f, a)
                for pipe in T.comm.foreach_src(net):
                    T.copy(a, pipe)
                for pipe in T.comm.foreach_dst(net):
                    T.copy(pipe, a)
                    for i, j in T.Tiles(f):
                        f[i, j] = a[i, j] * T.float32(2)
                    T.copy(f, C)

    return main


@pytest.mark.parametrize("seed", [0, 11, 71])
def test_pipeline_pipe_values_finite_pool_lifetimes(seed):
    del seed
    with pytest.raises(NotImplementedError, match="legacy Pipe/PipeRef"):
        lower_result(pipeline_pipe_values())


def uninitialized_remote_fragment():
    @T.prim_func
    def main(C: T.Tensor((32, 32), "float32")):
        with T.Kernel(2, 1, threads=1) as (x, y):
            f = T.alloc_fragment((32, 32), "float32")
            if x == 0:
                T.fill(f, 1)
            if x == 1:
                T.copy(f, C)

    return main


def test_fragment_definition_on_other_core_does_not_dominate():
    with pytest.raises(
        ValueError,
        match="uninitialized|read-before|dominat|before.*definition|before definite initialization",
    ):
        lower(uninitialized_remote_fragment())


def test_rank3_reduction_fragment_output_numeric():
    from testing.python.target.test_tilelang_tenstorrent_capabilities import reduction_fragment_output

    mod = lower(reduction_fragment_output())
    shape = tuple(map(int, mod.attrs["tt.tensor_table"][0].shape))
    a = np.random.default_rng(91).normal(size=shape).astype("float32")
    outputs, _ = interpret(mod, [a, np.zeros((32, 32), "float32")])
    np.testing.assert_allclose(outputs[1], a.sum(axis=0), rtol=1e-6, atol=1e-6)


def multicore_gemm_epilogue():
    @T.prim_func
    def main(A: T.Tensor((64, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((64, 32), "bfloat16")):
        with T.Kernel(2, 1, threads=1) as (x, y):
            a = T.alloc_shared((32, 32), "bfloat16")
            b = T.alloc_shared((32, 32), "bfloat16")
            acc = T.alloc_fragment((32, 32), "float32")
            f = T.alloc_fragment((32, 32), "float32")
            T.copy(A[x * 32 : (x + 1) * 32, 0:32], a)
            T.copy(B, b)
            T.clear(acc)
            T.gemm(a, b, acc)
            for i, j in T.Tiles(f):
                f[i, j] = acc[i, j] + T.float32(1)
            T.copy(f, C[x * 32 : (x + 1) * 32, 0:32])

    return main


def test_multicore_gemm_epilogue_has_local_accumulator_provenance():
    mod = lower(multicore_gemm_epilogue())
    rng = np.random.default_rng(98)
    a, b = (_round(rng.normal(size=shape), "bfloat16") for shape in [(64, 32), (32, 32)])
    outputs, _ = interpret(mod, [a, b, np.zeros((64, 32), "float32")], 98)
    np.testing.assert_array_equal(outputs[2], _round(a @ b + 1, "bfloat16"))
    assert {int(value.accumulator_id) for value in mod.attrs["tt.compute_value_table"]} == {0, 1}


def sliced_pipeline(extent=5, stages=2, out_of_bounds=False):
    @T.prim_func
    def main(A: T.Tensor(((extent - int(out_of_bounds)) * 32, 32), "float32"), C: T.Tensor((extent * 32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2})
            f = T.alloc_fragment((32, 32), "float32")
            for k in T.Pipelined(extent, num_stages=stages):
                begin = k * 32
                T.copy(A[begin : begin + 32, 0:32], a)
                for i, j in T.Tiles(f):
                    f[i, j] = a[i, j] * T.float32(2)
                T.copy(f, C[begin : begin + 32, 0:32])

    return main


@pytest.mark.parametrize("seed", [0, 11, 71])
def test_fragment_pipeline_distinct_slices_and_scalar_alias(seed):
    mod = lower(sliced_pipeline())
    a = np.arange(160 * 32, dtype="float32").reshape(160, 32)
    outputs, _ = interpret(mod, [a, np.zeros_like(a)], seed)
    np.testing.assert_array_equal(outputs[1], a * 2)


def pipeline_summa(core_m=2, core_n=2, stages=5, block_m=32, block_n=32, block_k=32, input_dtype="bfloat16", accum_dtype="float32"):
    """Keep the example's Bind, two-slot panels, fixed PipeNets and K lifetime."""
    a_net = T.comm.PipeNet([T.comm.Pipe((0, y), T.comm.CoreRange((1, y), (core_n, y + 1))) for y in range(core_m)]) if core_n > 1 else None
    b_net = T.comm.PipeNet([T.comm.Pipe((x, 0), T.comm.CoreRange((x, 1), (x + 1, core_m))) for x in range(core_n)]) if core_m > 1 else None
    m, n, k = core_m * block_m, core_n * block_n, stages * block_k

    @T.prim_func
    def summa(A: T.Tensor((m, k), input_dtype), B: T.Tensor((k, n), input_dtype), C: T.Tensor((m, n), input_dtype)):
        with T.Kernel(core_n, core_m, threads=1) as (x, y):
            a = T.alloc_shared((block_m, block_k), input_dtype, annotations={"tt.dfb_block_count": 2, "tt.tile_shape": (32, 32)})
            b = T.alloc_shared((block_k, block_n), input_dtype, annotations={"tt.dfb_block_count": 2, "tt.tile_shape": (32, 32)})
            c = T.alloc_fragment((block_m, block_n), accum_dtype)
            T.clear(c)
            for stage in T.Pipelined(stages, num_stages=2):
                row = y * block_m
                col = x * block_n
                start = stage * block_k
                if a_net is None:
                    T.copy(A[row : row + block_m, start : start + block_k], a)
                else:
                    if T.comm.is_src(a_net):
                        T.copy(A[row : row + block_m, start : start + block_k], a)
                    for pipe in T.comm.foreach_src(a_net):
                        T.copy(a, pipe)
                    for pipe in T.comm.foreach_dst(a_net):
                        T.copy(pipe, a)
                if b_net is None:
                    T.copy(B[start : start + block_k, col : col + block_n], b)
                else:
                    if T.comm.is_src(b_net):
                        T.copy(B[start : start + block_k, col : col + block_n], b)
                    for pipe in T.comm.foreach_src(b_net):
                        T.copy(b, pipe)
                    for pipe in T.comm.foreach_dst(b_net):
                        T.copy(pipe, b)
                T.gemm(a, b, c, clear_accum=False)
            row = y * block_m
            col = x * block_n
            T.copy(c, C[row : row + block_m, col : col + block_n])

    return summa


def test_legacy_pipe_summa_requires_high_level_communication():
    with pytest.raises(NotImplementedError, match="legacy Pipe/PipeRef"):
        lower_result(pipeline_summa())


def test_pipeline_slice_range_proof_rejects_last_iteration_overrun():
    with pytest.raises((ValueError, NotImplementedError), match="bounds|region|prov"):
        lower(sliced_pipeline(out_of_bounds=True))


def pipeline_tensor_inout():
    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2})
            f = T.alloc_fragment((32, 32), "float32")
            for _k in T.Pipelined(3, num_stages=2):
                T.copy(A, a)
                for i, j in T.Tiles(f):
                    f[i, j] = a[i, j] + T.float32(1)
                T.copy(f, A)

    return main


def test_fragment_tensor_inout_cannot_be_prefetched():
    mod = lower_result(pipeline_tensor_inout())
    assert str(mod.attrs["tt.ir_stage"]) == "structured"
    assert any("inout Tensor" in str(reason) for reason in mod.attrs["tt.deferred_reasons"])  # codespell:ignore inout


def pipeline_carried_fragment():
    @T.prim_func
    def main(C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            f = T.alloc_fragment((32, 32), "float32")
            T.fill(f, 0)
            for _k in T.Pipelined(3, num_stages=2):
                for i, j in T.Tiles(f):
                    f[i, j] = f[i, j] + T.float32(1)
                T.copy(f, C)

    return main


def test_loop_carried_ordinary_fragment_requires_explicit_spill_plan():
    mod = lower_result(pipeline_carried_fragment())
    assert str(mod.attrs["tt.ir_stage"]) == "structured"
    assert any(
        "carries a value across iterations" in str(reason)
        for reason in mod.attrs["tt.deferred_reasons"]
    )

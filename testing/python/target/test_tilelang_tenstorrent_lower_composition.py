"""Lower composition regressions, including CPU execution of Device streams."""

import numpy as np
import pytest
from tilelang import tvm
from tilelang.tenstorrent import language as T, lower_tenstorrent_ir, transform
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_compute_values import TARGET, _calls, _round


def lower(program, version=7):
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": program}), TARGET)
    assert int(mod.attrs["tt.device_ir_version"]) == version
    ir.assert_structural_equal(mod, ir.load_json(ir.save_json(mod)))
    assert transform.VerifyTenstorrentDeviceIR()(mod).same_as(mod)
    return mod


def interpret(mod, inputs, seed=0):
    """Execute value, DFB, and point-delivery contracts independently of lowering."""
    tensors = [np.array(value, copy=True) for value in inputs]
    values, dfbs, deliveries, accumulators = {}, {}, {}, {}
    pending = {}
    occupied = set()
    groups = {int(k): int(v) for k, v in mod.attrs.get("tt.dfb_storage_groups", {}).items()}
    accumulator_info = {int(a.accumulator_id): a for a in mod.attrs.get("tt.accumulator_table", [])}
    random = np.random.default_rng(seed)
    value_info = {int(value.value_id): value for value in mod.attrs.get("tt.compute_value_table", [])}
    dfb_info = {int(dfb.dfb_id): dfb for dfb in mod.attrs["tt.dfb_table"]}
    streams = [list(_calls(func.body)) for func in mod.functions.values()]
    explicit_release = any(call.op.name == "tl.tt.dfb_release" for stream in streams for call in stream)
    cursors = [0] * len(streams)
    value_owners = {
        int(call.args[0]): index
        for index, stream in enumerate(streams)
        for call in stream
        if call.op.name in ("tl.tt.compute_value", "tl.tt.compute_value_gemm")
    }
    precision = {}
    asynchronous = {int(call.args[0]) for stream in streams for call in stream if call.op.name == "tl.tt.dfb_copy_wait"}

    def expression(expr, maps):
        if isinstance(expr, (tirx.IntImm, tirx.FloatImm)):
            return np.float32(expr.value)
        if isinstance(expr, tirx.Cast):
            return _round(expression(expr.value, maps), expr.dtype)
        if isinstance(expr, tirx.Call):
            if expr.op.name in ("tl.tt.dfb_load", "tl.tt.compute_value_load"):
                identifier = int(expr.args[0])
                source = dfbs[identifier] if expr.op.name == "tl.tt.dfb_load" else values[identifier]
                axes = maps[(expr.op.name, identifier)]
                return source[tuple(slice(0, 1) if axis == -1 else slice(None) for axis in axes)]
            if expr.op.name == "tirx.exp2":
                return _round(np.exp2(expression(expr.args[0], maps)), expr.dtype)
            raise AssertionError(expr.op.name)
        binary = {tirx.Add: np.add, tirx.Sub: np.subtract, tirx.Mul: np.multiply, tirx.Div: np.divide, tirx.Max: np.maximum}
        return _round(binary[type(expr)](expression(expr.a, maps), expression(expr.b, maps)), expr.dtype)

    for _ in range(10000):
        if all(cursor == len(stream) for cursor, stream in zip(cursors, streams)):
            assert not pending, "Copies remain incomplete"
            if explicit_release:
                assert not occupied, "DFB reservations remain live"
            return tensors, values
        advanced = False
        for index in random.permutation(len(streams)):
            stream = streams[index]
            if cursors[index] == len(stream):
                continue
            call = stream[cursors[index]]
            name, args = call.op.name, list(map(int, call.args))
            if name == "tl.tt.compute_precision":
                assert args[0] in (0, 1)
                precision[index] = args[0]
                # Compute-local values cannot survive a destination precision
                # transition. Only explicit DFB snapshots can reload them.
                for identifier in list(values):
                    if value_owners[identifier] == index:
                        del values[identifier]
            if name == "tl.tt.dfb_wait" and args[0] not in dfbs:
                continue
            if name == "tl.tt.dfb_pipe_recv" and args[0] not in deliveries:
                continue
            if name == "tl.tt.dfb_reserve":
                group = groups.get(args[0], args[0])
                if sum(groups.get(resource, resource) == group for resource in occupied) >= int(dfb_info[args[0]].block_count):
                    continue
                assert args[0] not in occupied
                occupied.add(args[0])
            elif name == "tl.tt.tensor_to_dfb_nd":
                region = tuple(slice(args[i], args[i] + args[i + 1]) for i in range(2, len(args), 2))
                value = tensors[args[0]][region].copy()
                local_rank = len(dfb_info[args[1]].block_shape_in_tiles)
                while value.ndim > local_rank:
                    assert value.shape[0] == 1
                    value = value[0]
                if args[1] in asynchronous:
                    pending[args[1]] = ("load", value)
                else:
                    dfbs[args[1]] = value
            elif name == "tl.tt.dfb_to_tensor_nd":
                region = tuple(slice(args[i], args[i] + args[i + 1]) for i in range(2, len(args), 2))
                if args[0] in asynchronous:
                    pending[args[0]] = ("store", args[1], region, dfbs[args[0]].copy())
                else:
                    tensors[args[1]][region] = dfbs[args[0]]
            elif name == "tl.tt.dfb_copy_wait":
                action = pending.pop(args[0])
                if action[0] == "load":
                    dfbs[args[0]] = action[1]
                else:
                    tensors[action[1]][action[2]] = action[3]
            elif name == "tl.tt.dfb_pipe_send":
                deliveries[args[0]] = dfbs[args[1]].copy()
            elif name == "tl.tt.dfb_pipe_recv":
                dfbs[args[1]] = deliveries[args[0]].copy()
            elif name in ("tl.tt.compute_value", "tl.tt.dfb_compute"):
                attrs = call.annotations
                kind = str(attrs["tt.compute_kind"].value)
                domain = tuple(map(int, attrs["tt.logical_domain"]))
                if kind == "transpose":
                    result = dfbs[args[1]].swapaxes(-1, -2)
                elif kind == "reduce":
                    reducer = {"sum": np.sum, "max": np.max, "min": np.min}[str(attrs["tt.reduce_kind"].value)]
                    result = reducer(dfbs[args[1]], axis=int(attrs["tt.reduce_axis"]), keepdims=len(domain) == dfbs[args[1]].ndim)
                elif kind == "copy":
                    result = dfbs[args[1]]
                else:
                    maps = {
                        ("tl.tt.dfb_load", identifier): tuple(map(int, axes))
                        for identifier, axes in zip(args[1:], attrs.get("tt.access_maps", []))
                    }
                    maps.update(
                        {
                            ("tl.tt.compute_value_load", int(identifier)): tuple(map(int, axes))
                            for identifier, axes in zip(attrs.get("tt.value_inputs", []), attrs.get("tt.value_access_maps", []))
                        }
                    )
                    result = expression(attrs["tt.expression"], maps)
                result = np.broadcast_to(result, domain).copy()
                if name == "tl.tt.compute_value":
                    assert args[0] not in values
                    values[args[0]] = _round(result, value_info[args[0]].buffer.dtype)
                else:
                    dfbs[args[0]] = _round(result, dfb_info[args[0]].element_dtype)
            elif name == "tl.tt.compute_value_gemm":
                output, lhs, rhs, old, ta, tb = args
                if index in precision:
                    assert precision[index] == (1 if value_info[output].buffer.dtype == "float32" else 0)
                a, b = dfbs[lhs], dfbs[rhs]
                result = (a.T if ta else a) @ (b.T if tb else b)
                if old >= 0:
                    result = result + values[old]
                values[output] = _round(result, value_info[output].buffer.dtype)
            elif name == "tl.tt.accumulator_init":
                info = accumulator_info[args[0]]
                accumulators[args[0]] = np.zeros(tuple(int(r.extent) for r in info.accumulator_region.region), "float32")
            elif name == "tl.tt.gemm_update":
                lhs, rhs, identifier, ta, tb = args
                a, b = dfbs[lhs], dfbs[rhs]
                result = accumulators[identifier] + (a.T if ta else a) @ (b.T if tb else b)
                accumulators[identifier] = _round(result, accumulator_info[identifier].accumulation_dtype)
            elif name == "tl.tt.accumulator_materialize":
                dfbs[args[1]] = _round(accumulators.pop(args[0]), dfb_info[args[1]].element_dtype)
            elif name == "tl.tt.compute_value_store":
                dfbs[args[1]] = values[args[0]].copy()
            elif name == "tl.tt.dfb_release":
                assert args[0] not in pending, "Release before copy completion"
                occupied.remove(args[0])
                del dfbs[args[0]]
            else:
                assert name in (
                    "tl.tt.dfb_reserve",
                    "tl.tt.dfb_wait",
                    "tl.tt.dfb_copy_wait",
                    "tl.tt.dfb_pipe_wait",
                    "tl.tt.compute_precision",
                ), name
            cursors[index] += 1
            advanced = True
        assert advanced, "Device streams deadlocked"
    raise AssertionError("Device stream instruction bound exceeded")


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
    assert len(mod.functions) == 6
    a = np.arange(64 * 32, dtype="float32").reshape(64, 32)
    outputs, _ = interpret(mod, [a, np.zeros_like(a)])
    np.testing.assert_array_equal(outputs[1], (a + 1) * 2)
    definitions = list(mod.attrs["tt.compute_value_table"])
    assert len({int(value.value_id) for value in definitions}) == len(definitions)
    assert not definitions[0].buffer.data.same_as(definitions[-1].buffer.data)


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


def test_pipe_materialized_values_and_receiver_local_epilogue():
    mod = lower(pipe_values())
    a = np.random.default_rng(8).normal(size=(32, 32)).astype("float32")
    outputs, _ = interpret(mod, [a, np.zeros_like(a)])
    np.testing.assert_array_equal(outputs[1], a * 2 + 3)
    assert len(mod.attrs["tt.pipe_transfer_table"]) == 1


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
    mod = lower(pipeline_pipe_values())
    a = np.random.default_rng(19).normal(size=(32, 32)).astype("float32")
    actual, _ = interpret(mod, [a, np.zeros_like(a)], seed)
    np.testing.assert_array_equal(actual[1], (a + 4) * 2)
    assert len(mod.attrs["tt.pipe_transfer_table"]) == 5
    assert {int(t.occurrence) for t in mod.attrs["tt.pipe_transfer_table"]} == set(range(5))


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
    with pytest.raises(ValueError, match="uninitialized|read-before|dominat|before.*definition"):
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


@pytest.mark.parametrize("seed", [0, 11, 71])
def test_full_k_summa_pipeline_with_finite_pools(seed):
    mod = lower(pipeline_summa(), version=6)
    rng = np.random.default_rng(199)
    a = _round(rng.normal(size=(64, 160)), "bfloat16")
    b = _round(rng.normal(size=(160, 64)), "bfloat16")
    outputs, _ = interpret(mod, [a, b, np.zeros((64, 64), "float32")], seed)
    expected = sum(a[:, k : k + 32] @ b[k : k + 32, :] for k in range(0, 160, 32))
    np.testing.assert_array_equal(outputs[2], _round(expected, "bfloat16"))
    assert len(mod.attrs["tt.pipe_transfer_table"]) == 20
    assert all(int(acc.full_k_tiles) == 5 for acc in mod.attrs["tt.accumulator_table"])


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
    with pytest.raises(NotImplementedError, match="inout Tensor"):  # codespell:ignore inout
        lower(pipeline_tensor_inout())


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
    with pytest.raises(NotImplementedError, match="carries a value across iterations"):
        lower(pipeline_carried_fragment())

"""Independent asynchronous Device IR model; no TTL or hardware execution."""

from collections import defaultdict

import numpy as np
import pytest

from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_phase4_semantics import cast, expression


def run_pipeline_device(mod, inputs, *, seed=0):
    """Execute slot streams with finite storage and delayed copy completion.

    Memory is keyed by (pool, epoch modulo capacity), so illegal overwrite
    cannot hide behind an unlimited dictionary of immutable generations.
    Randomized slot order and copy latency exercise legal interleavings.
    """
    VerifyTenstorrentDeviceIR()(mod)
    assert int(mod.attrs["tt.device_ir_version"]) == 3
    rng = np.random.default_rng(seed)
    tensors = [cast(value, desc.dtype).copy() for value, desc in zip(inputs, mod.attrs["tt.tensor_table"])]
    dfbs = {int(d.dfb_id): d for d in mod.attrs["tt.dfb_table"]}
    groups = {int(k): int(v) for k, v in mod.attrs["tt.dfb_storage_groups"].items()}
    epochs = {int(k): int(v[0]) for k, v in mod.attrs["tt.pipeline_relations"].items()}
    addresses = {i: (groups[i], epochs[i] % int(d.block_count)) for i, d in dfbs.items()}
    streams = []
    for slot in ("ncrisc", "trisc", "brisc"):
        func = next(f for f in mod.functions.values() if str(f.attrs["tt.kernel_slot"]) == slot)
        stmts = list(func.body.seq) if isinstance(func.body, tirx.SeqStmt) else [func.body]
        streams.append([s.value for s in stmts if isinstance(s.value, tirx.Call)])
    positions = [0] * len(streams)
    owners, storage, pending = {}, {}, {}
    published, released = set(), set()
    high_water = defaultdict(int)
    copy_high_water = 0
    exports = []
    tick = 0
    while any(p < len(s) for p, s in zip(positions, streams)):
        tick += 1
        assert tick < 1000000, "Device streams failed to make bounded progress"
        progressed = False
        for stream_index in rng.permutation(len(streams)):
            pos = positions[stream_index]
            if pos == len(streams[stream_index]):
                continue
            call = streams[stream_index][pos]
            name = call.op.name
            args = [int(x) for x in call.args]
            if name == "tl.tt.dfb_reserve":
                resource = args[0]
                address = addresses[resource]
                if address in owners:
                    continue
                owners[address] = resource
                high_water[groups[resource]] = max(high_water[groups[resource]], sum(g == groups[resource] for g, _ in owners))
            elif name == "tl.tt.dfb_wait":
                if args[0] not in published:
                    continue
                assert owners[addresses[args[0]]] == args[0]
            elif name in ("tl.tt.tensor_to_dfb_nd", "tl.tt.dfb_to_tensor_nd"):
                is_input = name == "tl.tt.tensor_to_dfb_nd"
                tensor, resource = args[:2] if is_input else args[1::-1]
                assert owners[addresses[resource]] == resource
                assert resource not in pending
                if not is_input:
                    assert resource in published
                region = tuple(slice(lo, lo + size) for lo, size in zip(args[2::2], args[3::2]))
                # Transfer data is read on completion, not eagerly captured at
                # issue; releasing or overwriting in flight is observable.
                pending[resource] = (tick + int(rng.integers(1, 8)), is_input, tensor, region)
                copy_high_water = max(copy_high_water, len(pending))
            elif name == "tl.tt.dfb_copy_wait":
                resource = args[0]
                ready, is_input, tensor, region = pending[resource]
                if tick < ready:
                    continue
                assert owners[addresses[resource]] == resource
                if is_input:
                    storage[addresses[resource]] = tensors[tensor][region].copy()
                    published.add(resource)
                else:
                    tensors[tensor][region] = storage[addresses[resource]]
                    exports.append((tensor, epochs[resource], tensors[tensor][region].copy()))
                del pending[resource]
            elif name == "tl.tt.dfb_release":
                resource = args[0]
                assert resource in published and resource not in pending
                assert owners.pop(addresses[resource]) == resource
                del storage[addresses[resource]]
                published.remove(resource)
                assert resource not in released
                released.add(resource)
            elif name == "tl.tt.dfb_compute":
                output, *operands = args
                assert owners[addresses[output]] == output
                assert all(i in published and owners[addresses[i]] == i for i in operands)
                values = {i: storage[addresses[i]] for i in operands}
                attrs = call.annotations
                kind = str(attrs["tt.compute_kind"].value)
                domain = tuple(int(x) for x in attrs["tt.logical_domain"])
                maps = {i: [int(x) for x in axes] for i, axes in zip(operands, attrs["tt.access_maps"])}
                if kind in ("elementwise", "typecast", "fill"):
                    result = np.broadcast_to(expression(attrs["tt.expression"], values, maps, domain), domain)
                elif kind == "copy":
                    result = values[operands[0]]
                elif kind == "transpose":
                    result = np.transpose(values[operands[0]], [int(x) for x in attrs["tt.axes"]])
                elif kind == "gemm":
                    a, b = (values[i] for i in operands[:2])
                    a = a.T if int(attrs["tt.transpose_a"]) else a
                    b = b.T if int(attrs["tt.transpose_b"]) else b
                    result = a.astype(np.float32) @ b.astype(np.float32)
                    if not int(attrs["tt.clear"]):
                        result = result + values[operands[2]]
                elif kind == "reduce":
                    axis, reduction = int(attrs["tt.reduce_axis"]), str(attrs["tt.reduce_kind"].value)
                    source = values[operands[0]].astype(np.float32)
                    if reduction == "sum":
                        result, combine = np.sum(source, axis=axis, dtype=np.float32), np.add
                    else:
                        propagate = bool(int(attrs["tt.nan_propagate"]))
                        combine = (np.maximum if propagate else np.fmax) if reduction == "max" else (np.minimum if propagate else np.fmin)
                        result = combine.reduce(source, axis=axis)
                    if not int(attrs["tt.clear"]):
                        result = combine(values[operands[1]], result)
                else:
                    raise AssertionError(f"Unsupported reference compute: {kind}")
                storage[addresses[output]] = cast(result, attrs["tt.compute_dtype"].value).copy()
                published.add(output)
            else:
                raise AssertionError(f"Unsupported reference op: {name}")
            positions[stream_index] += 1
            progressed = True
        assert progressed or pending, "Device streams deadlocked without any in-flight copy"
    assert not pending and not owners and not published and not storage
    assert released == set(dfbs)
    return tensors, {"exports": exports, "pool_high_water": dict(high_water), "copy_high_water": copy_high_water}


@pytest.mark.parametrize("stages,extent,start", [(1, 1, 0), (2, 1, 3), (2, 5, 0), (3, 7, 2), (4, 2, 5), (32, 1, 0), (32, 33, 0)])
@pytest.mark.parametrize("seed", [0, 1, 19])
@pytest.mark.parametrize("wait_policy", ["conservative", "delayed"])
def test_pipeline_device_numerics_and_ring_reuse(stages, extent, start, seed, wait_policy):
    from testing.python.target.test_tilelang_tenstorrent_phase5_pipeline import make_pipeline, lower_pipeline

    mod = lower_pipeline(make_pipeline(stages=stages, extent=extent, start=start, wait_policy=wait_policy))
    mod = ir.load_json(ir.save_json(mod))
    a = np.arange(1024, dtype=np.float32).reshape(32, 32) / 8
    b = np.flip(a, axis=1).copy()
    actual, trace = run_pipeline_device(mod, [a, b, np.zeros_like(a)], seed=seed)
    assert len(trace["exports"]) == extent
    for (_, epoch, value), iteration in zip(trace["exports"], range(start, start + extent)):
        assert epoch == iteration - start
        np.testing.assert_array_equal(value, a + b + np.float32(iteration))
    np.testing.assert_array_equal(actual[-1], a + b + np.float32(start + extent - 1))
    assert max(trace["pool_high_water"].values()) <= min(stages, extent)
    if wait_policy == "delayed":
        assert trace["copy_high_water"] >= 2 * min(stages, extent)
    else:
        assert trace["copy_high_water"] == 1


def test_pipeline_matches_serial_device_semantics_with_extra_capacity():
    from testing.python.target.test_tilelang_tenstorrent_phase4_semantics import run_device
    from testing.python.target.test_tilelang_tenstorrent_phase5_pipeline import make_pipeline, lower_pipeline

    a = np.arange(1024, dtype=np.float32).reshape(32, 32) / 7
    inputs = [a, a.T.copy(), np.zeros_like(a)]
    serial = lower_pipeline(make_pipeline(stages=0, extent=7, start=3))
    pipelined = lower_pipeline(make_pipeline(stages=3, extent=7, start=3, block_count=4))
    expected = run_device(serial, inputs)
    actual, _ = run_pipeline_device(pipelined, inputs, seed=27)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def make_mixed_pipeline(stages, dtype):
    @T.prim_func
    def program(
        A: T.Tensor((32, 32), dtype),
        B: T.Tensor((32, 32), dtype),
        C: T.Tensor((32, 32), "float32"),
        R: T.Tensor((32,), "float32"),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), dtype)
            b = T.alloc_shared((32, 32), dtype)
            acc = T.alloc_shared((32, 32), "float32")
            summed = T.alloc_shared((32,), "float32")
            for _ in T.Pipelined(4, num_stages=stages):
                T.copy(A, a)
                T.copy(B, b)
                T.gemm(a, b, acc, clear_accum=True)
                for i, j in T.Parallel(32, 32):
                    acc[i, j] = acc[i, j] * T.float32(2)
                T.reduce_sum(acc, summed, dim=1)
                T.copy(acc, C)
                T.copy(summed, R)

    return program


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
def test_pipeline_retains_general_compute_generations(dtype, arch):
    from testing.python.target.test_tilelang_tenstorrent_phase4_semantics import run_device
    from testing.python.target.test_tilelang_tenstorrent_phase5_pipeline import lower_pipeline

    rng = np.random.default_rng(71)
    a, b = [rng.integers(-2, 3, (32, 32)).astype(np.float32) for _ in range(2)]
    inputs = [a, b, np.zeros_like(a), np.zeros(32, dtype=np.float32)]
    serial = lower_pipeline(make_mixed_pipeline(0, dtype), arch)
    pipelined = lower_pipeline(make_mixed_pipeline(3, dtype), arch)
    actual, trace = run_pipeline_device(pipelined, inputs, seed=29)
    expected = run_device(serial, inputs)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    result = 2 * (inputs[0] @ inputs[1])
    np.testing.assert_array_equal(actual[2], result)
    np.testing.assert_array_equal(actual[3], result.sum(axis=1))
    assert len(trace["exports"]) == 8

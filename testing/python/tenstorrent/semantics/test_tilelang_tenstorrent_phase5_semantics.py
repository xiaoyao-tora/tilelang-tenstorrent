"""Independent asynchronous Device IR model; no TTL or hardware execution."""

from collections import defaultdict

import numpy as np
import pytest

from tilelang.tenstorrent import device_ir as D
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR
from tvm import ir

from testing.python.tenstorrent.semantics.test_tilelang_tenstorrent_phase4_semantics import cast, expression
from testing.python.tenstorrent._support.reference import _calls


def run_pipeline_device(mod, inputs, *, seed=0):
    """Execute slot streams with finite storage and delayed copy completion.

    Memory is keyed by (pool, epoch modulo capacity), so illegal overwrite
    cannot hide behind an unlimited dictionary of immutable generations.
    Randomized slot order and copy latency exercise legal interleavings.
    """
    VerifyTenstorrentDeviceIR()(mod)
    mod = D.expand_current_device_tables(mod)
    assert str(mod.attrs["tt.ir_stage"]) == "device"
    assert "tt.device_ir_version" not in mod.attrs
    rng = np.random.default_rng(seed)
    tensors = [cast(value, desc.dtype).copy() for value, desc in zip(inputs, mod.attrs["tt.tensor_table"])]
    dfbs = {int(d.dfb_id): d for d in mod.attrs["tt.dfb_table"]}
    groups = {identifier: identifier for identifier in dfbs}
    capacities = {identifier: int(descriptor.block_count) for identifier, descriptor in dfbs.items()}
    next_epoch = defaultdict(int)
    published = defaultdict(list)
    producer_active, consumer_active = {}, {}
    streams = []
    for slot in ("ncrisc", "trisc", "brisc"):
        func = next(f for f in mod.functions.values() if str(f.attrs["tt.kernel_slot"]) == slot)
        streams.append(list(_calls(func.body)))
    positions = [0] * len(streams)
    owners, storage, pending = {}, {}, {}
    released_epochs = defaultdict(int)
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
                if resource in producer_active:
                    continue
                epoch = next_epoch[resource]
                address = (resource, epoch % capacities[resource])
                if address in owners:
                    continue
                producer_active[resource] = epoch
                owners[address] = (resource, epoch)
                high_water[groups[resource]] = max(high_water[groups[resource]], sum(g == groups[resource] for g, _ in owners))
            elif name == "tl.tt.dfb_wait":
                resource = args[0]
                if resource not in consumer_active:
                    if not published[resource]:
                        continue
                    consumer_active[resource] = published[resource][0]
                epoch = consumer_active[resource]
                assert owners[(resource, epoch % capacities[resource])] == (resource, epoch)
            elif name in ("tl.tt.tensor_to_dfb_nd", "tl.tt.dfb_to_tensor_nd"):
                is_input = name == "tl.tt.tensor_to_dfb_nd"
                tensor, resource = args[:2] if is_input else args[1::-1]
                epoch = producer_active[resource] if is_input else consumer_active[resource]
                address = (resource, epoch % capacities[resource])
                assert owners[address] == (resource, epoch)
                key = (resource, "producer" if is_input else "consumer")
                assert key not in pending
                if not is_input:
                    assert published[resource] and published[resource][0] == epoch
                region = tuple(slice(lo, lo + size) for lo, size in zip(args[2::2], args[3::2]))
                # Transfer data is read on completion, not eagerly captured at
                # issue; releasing or overwriting in flight is observable.
                pending[key] = (tick + int(rng.integers(1, 8)), epoch, is_input, tensor, region)
                copy_high_water = max(copy_high_water, len(pending))
            elif name == "tl.tt.dfb_copy_wait":
                resource = args[0]
                key = (resource, "producer") if (resource, "producer") in pending else (resource, "consumer")
                ready, epoch, is_input, tensor, region = pending[key]
                if tick < ready:
                    continue
                address = (resource, epoch % capacities[resource])
                assert owners[address] == (resource, epoch)
                if is_input:
                    storage[address] = tensors[tensor][region].copy()
                    published[resource].append(epoch)
                    del producer_active[resource]
                    next_epoch[resource] += 1
                else:
                    tensors[tensor][region] = storage[address]
                    exports.append((tensor, epoch, tensors[tensor][region].copy()))
                del pending[key]
            elif name == "tl.tt.dfb_release":
                resource = args[0]
                epoch = consumer_active[resource]
                assert published[resource] and published[resource][0] == epoch
                assert (resource, "consumer") not in pending
                address = (resource, epoch % capacities[resource])
                assert owners.pop(address) == (resource, epoch)
                del storage[address]
                published[resource].pop(0)
                del consumer_active[resource]
                released_epochs[resource] += 1
            elif name == "tl.tt.dfb_compute":
                output, *operands = args
                output_epoch = producer_active[output]
                output_address = (output, output_epoch % capacities[output])
                assert owners[output_address] == (output, output_epoch)
                assert all(i in consumer_active for i in operands)
                values = {i: storage[(i, consumer_active[i] % capacities[i])] for i in operands}
                attrs = call.annotations
                kind = str(attrs["tt.compute_kind"].value)
                domain = tuple(int(x) for x in attrs["tt.logical_domain"])
                maps = {i: [int(x) for x in axes] for i, axes in zip(operands, attrs.get("tt.access_maps", []))}
                maps.update({("dfb_operand", position): resource for position, resource in enumerate(operands)})
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
                    if len(operands) == 3:
                        result = result + values[operands[2]]
                elif kind == "reduce":
                    axis, reduction = int(attrs["tt.reduce_axis"]), str(attrs["tt.reduce_kind"].value)
                    source = values[operands[0]].astype(np.float32)
                    if reduction == "sum":
                        result, combine = np.sum(source, axis=axis, dtype=np.float32), np.add
                    else:
                        propagate = bool(int(attrs.get("tt.nan_propagate", 0)))
                        combine = (np.maximum if propagate else np.fmax) if reduction == "max" else (np.minimum if propagate else np.fmin)
                        result = combine.reduce(source, axis=axis)
                    if len(operands) == 2:
                        result = combine(values[operands[1]], result)
                else:
                    raise AssertionError(f"Unsupported reference compute: {kind}")
                storage[output_address] = cast(result, dfbs[output].element_dtype).copy()
                published[output].append(output_epoch)
                del producer_active[output]
                next_epoch[output] += 1
            else:
                raise AssertionError(f"Unsupported reference op: {name}")
            positions[stream_index] += 1
            progressed = True
        assert progressed or pending, "Device streams deadlocked without any in-flight copy"
    assert not pending and not owners and not producer_active and not consumer_active and not storage
    assert all(not epochs for epochs in published.values())
    assert released_epochs == {identifier: int(descriptor.transaction_count_or_loop_relation) for identifier, descriptor in dfbs.items()}
    return tensors, {"exports": exports, "pool_high_water": dict(high_water), "copy_high_water": copy_high_water}


@pytest.mark.parametrize("stages,extent,start", [(1, 1, 0), (2, 1, 3), (2, 5, 0), (3, 7, 2), (4, 2, 5), (32, 1, 0), (32, 33, 0)])
@pytest.mark.parametrize("seed", [0, 1, 19])
@pytest.mark.parametrize("wait_policy", ["conservative", "delayed"])
def test_pipeline_device_numerics_and_ring_reuse(stages, extent, start, seed, wait_policy):
    from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_phase5_pipeline import (
        lower_pipeline,
        make_pipeline,
    )

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
    # Lower preserves the policy on the structured loop; this Device reference
    # executes lexical order. TTL/TT-Lang owns delayed window scheduling.
    assert str(mod.attrs["tt.pipeline_wait_policy"]) == wait_policy
    assert trace["copy_high_water"] == 1


def test_pipeline_matches_serial_device_semantics_with_extra_capacity():
    from testing.python.tenstorrent.semantics.test_tilelang_tenstorrent_phase4_semantics import run_device
    from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_phase5_pipeline import (
        lower_pipeline,
        make_pipeline,
    )

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
            acc = T.alloc_shared(
                (32, 32),
                "float32",
                annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": max(stages, 1)},
            )
            summed = T.alloc_shared((32,), "float32")
            for _ in T.Pipelined(4, num_stages=stages):
                T.copy(A, a)
                T.copy(B, b)
                T.gemm(a, b, acc, clear_accum=True)
                for i, j in T.Tiles(acc):
                    acc[i, j] = acc[i, j] * T.float32(2)
                T.reduce_sum(acc, summed, dim=1)
                T.copy(acc, C)
                T.copy(summed, R)

    return program


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
def test_pipeline_retains_general_compute_generations(dtype, arch):
    from testing.python.tenstorrent.semantics.test_tilelang_tenstorrent_phase4_semantics import run_device
    from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_phase5_pipeline import lower_pipeline

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

"""Logical Device IR v4 evaluation; no TTL, simulator or hardware execution."""

import numpy as np
import pytest

from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_phase4_semantics import cast, expression


def run_multicore_device(mod, inputs, *, seed=0):
    """Interpret explicit slot streams with delayed point delivery and storage.

    Data is read at completion, so early source release cannot be hidden by an
    eager snapshot in the model. This models only the logical Device contract.
    """
    VerifyTenstorrentDeviceIR()(mod)
    assert int(mod.attrs["tt.device_ir_version"]) in (4, 5, 6)
    rng = np.random.default_rng(seed)
    tensors = [cast(value, desc.dtype).copy() for value, desc in zip(inputs, mod.attrs["tt.tensor_table"])]
    dfbs = {int(d.dfb_id): d for d in mod.attrs["tt.dfb_table"]}
    transfers = {int(t.transfer_id): t for t in mod.attrs["tt.pipe_transfer_table"]}
    accumulator_descs = {int(a.accumulator_id): a for a in mod.attrs.get("tt.accumulator_table", [])}
    accumulators = {}
    streams = []
    for name in mod.attrs["tt.kernel_order"]:
        func = mod[str(name)]
        statements = list(func.body.seq) if isinstance(func.body, tirx.SeqStmt) else [func.body]
        assert all(isinstance(s, tirx.Evaluate) for s in statements)
        streams.append([s.value for s in statements if isinstance(s.value, tirx.Call)])
    positions = [0] * len(streams)
    reserved, published, released = set(), set(), set()
    values, copies, sends, receives, delivered = {}, {}, {}, set(), set()
    send_waited, recv_waited = set(), set()
    exports, deliveries = [], []
    tick = 0
    while any(p < len(s) for p, s in zip(positions, streams)):
        tick += 1
        assert tick < 100000, "Logical Device evaluation exceeded progress bound"
        progressed = False
        for si in rng.permutation(len(streams)):
            pos = positions[si]
            if pos == len(streams[si]):
                continue
            call = streams[si][pos]
            name, args = call.op.name, [int(a) for a in call.args]
            if name == "tl.tt.dfb_reserve":
                d = args[0]
                assert d not in reserved and d not in released
                assert int(dfbs[d].block_count) >= args[1] == 1
                reserved.add(d)
            elif name == "tl.tt.dfb_wait":
                if args[0] not in published:
                    continue
                # Producer-forwarded panels become visible to local compute
                # only after all outgoing deliveries have completed.
                if (
                    int(mod.attrs["tt.device_ir_version"]) == 6
                    and str(dfbs[args[0]].producer_slot) == "ncrisc"
                    and any(tid not in send_waited for tid, t in transfers.items() if int(t.source_dfb_id) == args[0])
                ):
                    continue
                assert args[0] in reserved
            elif name in ("tl.tt.tensor_to_dfb_nd", "tl.tt.dfb_to_tensor_nd"):
                is_input = name == "tl.tt.tensor_to_dfb_nd"
                t, d = args[:2] if is_input else args[1::-1]
                assert d in reserved and d not in copies
                if not is_input:
                    assert d in published
                region = tuple(slice(lo, lo + size) for lo, size in zip(args[2::2], args[3::2]))
                copies[d] = (tick + int(rng.integers(1, 8)), is_input, t, region)
            elif name == "tl.tt.dfb_copy_wait":
                d = args[0]
                ready, is_input, t, region = copies[d]
                if tick < ready:
                    continue
                assert d in reserved
                if is_input:
                    values[d] = tensors[t][region].copy()
                    published.add(d)
                else:
                    tensors[t][region] = values[d]
                    exports.append((t, d, values[d].copy()))
                del copies[d]
            elif name == "tl.tt.dfb_pipe_send":
                tid, d, count = args
                assert count == 1 and tid not in sends
                assert d == int(transfers[tid].source_dfb_id) and d in published and d in reserved
                sends[tid] = tick + int(rng.integers(1, 8))
            elif name == "tl.tt.dfb_pipe_recv":
                tid, d, count = args
                assert count == 1 and tid not in receives
                assert d == int(transfers[tid].destination_dfb_id) and d in reserved and d not in published
                receives.add(tid)
            elif name == "tl.tt.dfb_pipe_wait":
                tid, d, count = args
                assert count == 1
                if tid not in sends or tid not in receives or tick < sends[tid]:
                    continue
                t = transfers[tid]
                src, dst = int(t.source_dfb_id), int(t.destination_dfb_id)
                if d == dst and tid not in send_waited:
                    continue
                if tid not in delivered:
                    assert src in reserved and dst in reserved and src in published
                    values[dst] = values[src].copy()
                    delivered.add(tid)
                    deliveries.append(tid)
                if d == src:
                    assert tid not in send_waited
                    send_waited.add(tid)
                else:
                    assert d == dst and tid not in recv_waited
                    recv_waited.add(tid)
                    published.add(dst)
            elif name == "tl.tt.dfb_release":
                d = args[0]
                assert d in published and d in reserved and d not in copies
                assert all(tid in send_waited for tid, t in transfers.items() if int(t.source_dfb_id) == d)
                assert all(tid in recv_waited for tid, t in transfers.items() if int(t.destination_dfb_id) == d)
                reserved.remove(d)
                published.remove(d)
                released.add(d)
                del values[d]
            elif name == "tl.tt.accumulator_init":
                (aid,) = args
                assert aid not in accumulators
                shape = tuple(int(axis.extent) for axis in accumulator_descs[aid].accumulator_region.region)
                accumulators[aid] = np.zeros(shape, np.float32)
            elif name == "tl.tt.gemm_update":
                lhs, rhs, aid, transpose_a, transpose_b = args
                assert lhs in published and rhs in published and aid in accumulators
                a = values[lhs].T if transpose_a else values[lhs]
                b = values[rhs].T if transpose_b else values[rhs]
                accumulators[aid] = cast(
                    accumulators[aid] + a.astype(np.float32) @ b.astype(np.float32),
                    accumulator_descs[aid].accumulation_dtype,
                )
            elif name == "tl.tt.accumulator_materialize":
                aid, out = args
                assert out in reserved and out not in published
                values[out] = cast(accumulators.pop(aid), dfbs[out].element_dtype).copy()
                published.add(out)
            elif name == "tl.tt.dfb_compute":
                out, *operands = args
                assert out in reserved and out not in published
                assert all(d in reserved and d in published for d in operands)
                attrs = call.annotations
                kind = attrs["tt.compute_kind"].value
                domain = tuple(int(x) for x in attrs["tt.logical_domain"])
                maps = {d: [int(x) for x in axes] for d, axes in zip(operands, attrs["tt.access_maps"])}
                if kind in ("elementwise", "fill", "typecast"):
                    result = np.broadcast_to(expression(attrs["tt.expression"], values, maps, domain), domain)
                elif kind == "copy":
                    result = values[operands[0]]
                elif kind == "gemm":
                    a, b = (values[d] for d in operands[:2])
                    a = a.T if int(attrs["tt.transpose_a"]) else a
                    b = b.T if int(attrs["tt.transpose_b"]) else b
                    result = a.astype(np.float32) @ b.astype(np.float32)
                    if not int(attrs["tt.clear"]):
                        result += values[operands[2]]
                else:
                    raise AssertionError(f"Compute outside Phase 6 reference scope: {kind}")
                values[out] = cast(result, attrs["tt.compute_dtype"].value).copy()
                published.add(out)
            else:
                raise AssertionError(f"Unsupported v4 op: {name}")
            positions[si] += 1
            progressed = True
        future = any(tick < c[0] for c in copies.values()) or any(tick < ready for tid, ready in sends.items() if tid not in delivered)
        assert progressed or future, "Logical Device streams deadlocked"
    assert not reserved and not published and not values and not copies and not accumulators
    assert released == set(dfbs)
    assert delivered == send_waited == recv_waited == set(transfers)
    return tensors, {"exports": exports, "deliveries": deliveries, "ticks": tick}


@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
@pytest.mark.parametrize("seed", [0, 1, 19])
def test_row_column_gemm_device_numerics(arch, seed):
    from tilelang import tvm
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    target = tvm.target.Target({"kind": "tenstorrent", "arch": arch})
    from testing.python.target.test_tilelang_tenstorrent_phase6_lower import make_row_column_gemm

    source = tvm.IRModule({"row_column_gemm": make_row_column_gemm()})
    mod = TenstorrentPassPipelineBody(source, target)
    repeated = TenstorrentPassPipelineBody(source, target)
    assert mod.script() == repeated.script()
    assert ir.save_json(mod) == ir.save_json(repeated)
    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(restored, mod)
    assert len(mod.functions) == 27
    assert len(mod.attrs["tt.pipe_table"]) == 4
    assert len(mod.attrs["tt.pipe_transfer_table"]) == 8
    rng = np.random.default_rng(6)
    a = rng.integers(-3, 4, (64, 32)).astype(np.float32)
    b = rng.integers(-3, 4, (32, 64)).astype(np.float32)
    result, trace = run_multicore_device(restored, [a, b, np.zeros((64, 64), np.float32)], seed=seed)
    np.testing.assert_array_equal(result[-1], a @ b)
    assert len(trace["exports"]) == 4 and len(trace["deliveries"]) == 8


@pytest.mark.parametrize("seed", [0, 7, 19])
@pytest.mark.parametrize("mode", ["p2p", "collective", "scatter", "duplicate", "gather", "reverse", "discard", "shared_source_gemm"])
def test_communication_payload_semantics(mode, seed):
    from testing.python.target.test_tilelang_tenstorrent_phase6_lower import (
        lower_multicore,
        make_point_to_point,
        make_collective,
        make_scatter,
        make_gather,
        make_gemm_2d,
    )

    a = np.arange(1024, dtype=np.float32).reshape(32, 32) / 8
    if mode == "p2p":
        kernel, inputs, expected = make_point_to_point(), [a, np.zeros_like(a)], a
    elif mode == "collective":
        expected = np.zeros((96, 64), np.float32)
        for x in (1, 2):
            for y in (0, 1):
                expected[x * 32 : (x + 1) * 32, y * 32 : (y + 1) * 32] = a
        kernel, inputs = make_collective(), [a, np.zeros_like(expected)]
    elif mode in ("scatter", "duplicate"):
        expected = np.zeros((96, 32), np.float32)
        expected[32:64] = 21 if mode == "duplicate" else 1
        expected[64:96] = 12
        kernel, inputs = make_scatter(duplicate=mode == "duplicate"), [np.zeros_like(expected)]
    elif mode in ("gather", "reverse", "discard"):
        expected = np.zeros((64, 32), np.float32)
        reverse = mode in ("reverse", "discard")
        expected[:32] = 1 if mode == "discard" else 2 if reverse else 1
        if mode != "discard":
            expected[32:] = 1 if reverse else 2
        kernel = make_gather(reverse=reverse, last_only=mode == "discard")
        inputs = [np.zeros_like(expected)]
    else:
        a = np.arange(64 * 32, dtype=np.float32).reshape(64, 32) % 5
        b = np.arange(32 * 64, dtype=np.float32).reshape(32, 64) % 7
        expected = a @ b
        kernel, inputs = make_gemm_2d(), [a, b, np.zeros_like(expected)]
    mod = ir.load_json(ir.save_json(lower_multicore(kernel)))
    actual, trace = run_multicore_device(mod, inputs, seed=seed)
    np.testing.assert_array_equal(actual[-1], expected)
    assert len(trace["deliveries"]) == len(mod.attrs["tt.pipe_transfer_table"])


@pytest.mark.parametrize("seed", [0, 19])
def test_bfloat16_multitile_delivery(seed):
    from tilelang.tenstorrent import language as T
    from testing.python.target.test_tilelang_tenstorrent_phase6_lower import lower_multicore

    net = T.comm.PipeNet([T.comm.Pipe((0, 0), (1, 0))])

    @T.prim_func
    def program(A: T.Tensor((64, 32), "bfloat16"), C: T.Tensor((64, 32), "bfloat16")):
        with T.Kernel(2, 1, threads=1):
            value = T.alloc_shared((64, 32), "bfloat16")
            for pipe in T.comm.foreach_src(net):
                T.copy(A, value)
                T.copy(value, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, value)
                T.copy(value, C)

    a = np.arange(2048, dtype=np.float32).reshape(64, 32) / 7
    mod = lower_multicore(program)
    actual, _ = run_multicore_device(mod, [a, np.zeros_like(a)], seed=seed)
    np.testing.assert_array_equal(actual[-1], cast(a, "bfloat16"))
    assert all(tuple(int(x) for x in d.block_shape_in_tiles) == (2, 1) for d in mod.attrs["tt.dfb_table"])

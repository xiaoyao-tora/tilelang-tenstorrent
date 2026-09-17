"""Complete fixed-source SUMMA lowering, independent of TT-Lang or hardware."""

from collections import Counter

import numpy as np
import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T, lower_tenstorrent_ir
from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_phase4_semantics import cast
from testing.python.target.test_tilelang_tenstorrent_phase6_semantics import run_multicore_device


def make_summa(core_m=2, core_n=2, stages=2, block_m=32, block_n=32, block_k=32, input_dtype="bfloat16", accum_dtype="float32"):
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
            for stage in T.serial(stages):
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


def lower(func, arch="wormhole_b0"):
    return lower_tenstorrent_ir(tvm.IRModule({"summa": func}), tvm.target.Target({"kind": "tenstorrent", "arch": arch}))


@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
@pytest.mark.parametrize(
    "config",
    [
        {},
        dict(core_m=8, core_n=8, stages=16, block_m=64, block_n=64),
        dict(block_m=64, block_n=64, block_k=64, accum_dtype="bfloat16"),
        dict(core_m=1, core_n=3, stages=3),
        dict(core_m=3, core_n=1, stages=3),
    ],
)
def test_summa_complete_device_ir(config, arch):
    mod = lower(make_summa(**config), arch)
    cores = config.get("core_m", 2) * config.get("core_n", 2)
    stages = config.get("stages", 2)
    gx, gy = config.get("core_n", 2), config.get("core_m", 2)
    assert "tt.ir_stage" not in mod.attrs
    assert len(mod.functions) == cores * 3
    assert len(mod.attrs["tt.accumulator_table"]) == cores
    assert {int(a.full_k_tiles) for a in mod.attrs["tt.accumulator_table"]} == {stages * (config.get("block_k", 32) // 32)}
    transfers = mod.attrs["tt.pipe_transfer_table"]
    assert len(transfers) == stages * (gy * (gx - 1) + gx * (gy - 1))
    assert Counter(int(t.occurrence) for t in transfers) == {stage: gy * (gx - 1) + gx * (gy - 1) for stage in range(stages)}
    tensor_load_ids = set()
    for func in mod.functions.values():
        calls = []
        tirx.stmt_functor.post_order_visit(func.body, lambda n, calls=calls: calls.append(n) if isinstance(n, tirx.Call) else None)
        names = [c.op.name for c in calls]
        assert all(name.startswith("tl.tt.") for name in names)
        assert "tl.tt.dfb_compute" not in names  # Source panels need no snapshot copy.
        tensor_load_ids.update(int(c.args[1]) for c in calls if c.op.name == "tl.tt.tensor_to_dfb_nd")
        if str(func.attrs["tt.kernel_slot"]) == "trisc":
            assert names.count("tl.tt.accumulator_init") == 1
            assert names.count("tl.tt.gemm_update") == stages
            assert names.count("tl.tt.accumulator_materialize") == 1
    assert all(int(t.source_dfb_id) in tensor_load_ids for t in transfers)
    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(restored, mod)
    VerifyTenstorrentDeviceIR()(restored)


def test_summa_deterministic_lowering():
    source = make_summa(stages=3)
    assert ir.save_json(lower(source)) == ir.save_json(lower(source))


@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
@pytest.mark.parametrize("seed", [0, 19])
@pytest.mark.parametrize("dtype,accum", [("bfloat16", "float32"), ("bfloat16", "bfloat16"), ("float32", "float32")])
def test_summa_device_numerics(arch, seed, dtype, accum):
    rng = np.random.default_rng(77)
    a = rng.integers(-2, 3, (64, 96)).astype(np.float32)
    b = rng.integers(-2, 3, (96, 64)).astype(np.float32)
    mod = lower(make_summa(stages=3, input_dtype=dtype, accum_dtype=accum), arch)
    outputs, trace = run_multicore_device(mod, [a, b, np.zeros((64, 64), np.float32)], seed=seed)
    expected = np.zeros((64, 64), np.float32)
    for k in range(0, 96, 32):
        expected = cast(expected + a[:, k : k + 32] @ b[k : k + 32], accum)
    np.testing.assert_array_equal(outputs[-1], cast(expected, dtype))
    assert len(trace["exports"]) == 4
    assert len(trace["deliveries"]) == 12


def test_summa_fp32_accumulation_survives_bf16_cancellation():
    weights = [512.0, 1.0, -512.0]
    eye = np.eye(32, dtype=np.float32)
    a = np.tile(np.concatenate([eye] * 3, axis=1), (2, 1))
    b = np.tile(np.concatenate([w * eye for w in weights], axis=0), (1, 2))
    mod = lower(make_summa(stages=3))
    outputs, _ = run_multicore_device(mod, [a, b, np.zeros((64, 64), np.float32)], seed=7)
    np.testing.assert_array_equal(outputs[-1], np.tile(eye, (2, 2)))

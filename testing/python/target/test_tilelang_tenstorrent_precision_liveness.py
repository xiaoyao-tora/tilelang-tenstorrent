"""Precision boundaries preserve only fragment values that remain live."""

from collections import Counter

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import lower_tenstorrent_ir, transform
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_frontend02 import METADATA, TARGET


def _stores_by_buffer(mod):
    values = {int(value.value_id): str(value.buffer.name) for value in mod.attrs["tt.compute_value_table"]}
    stores = []
    for func in mod.functions.values():
        tirx.stmt_functor.post_order_visit(
            func.body,
            lambda node: (
                stores.append(values[int(node.args[0])])
                if isinstance(node, tirx.Call) and node.op.name == "tl.tt.compute_value_store"
                else None
            ),
        )
    return stores


@pytest.mark.parametrize("overwrite", [False, True])
def test_dead_fp32_fragment_does_not_cross_bf16_precision_region(overwrite):
    @T.prim_func
    def main(A: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            acc = T.alloc_fragment((32, 32), "bfloat16")
            dead = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            T.fill(dead, 1)
            for _step in T.serial(2, 4):
                T.gemm(a, a, acc, clear_accum=True)
                if overwrite:
                    T.fill(dead, 2)
                for i, j in T.Tiles(acc):
                    acc[i, j] = T.Cast("bfloat16", T.Cast("float32", acc[i, j]) * T.float32(1.25))
                T.copy(acc, C)

    mod = lower_tenstorrent_ir(tvm.IRModule({"main": main}), TARGET)
    assert "dead" not in _stores_by_buffer(mod)
    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(mod, restored)
    transform.VerifyTenstorrentDeviceIR()(restored)


def test_loop_carried_read_modify_write_fragment_keeps_precision_snapshots():
    @T.prim_func
    def main(A: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            acc = T.alloc_fragment((32, 32), "bfloat16")
            state = T.alloc_fragment((32, 32), "bfloat16")
            T.copy(A, a)
            T.fill(state, 1)
            for _step in T.serial(2, 5):
                T.gemm(a, a, acc, clear_accum=True)
                for i, j in T.Tiles(state):
                    state[i, j] = T.Cast("bfloat16", T.Cast("float32", state[i, j]) * T.float32(1.25)) + acc[i, j]
            T.copy(state, C)

    mod = lower_tenstorrent_ir(tvm.IRModule({"main": main}), TARGET)
    assert _stores_by_buffer(mod).count("state") >= 3
    transform.VerifyTenstorrentDeviceIR()(mod)


def test_constant_branch_full_overwrite_kills_old_fp32_value():
    @T.prim_func
    def main(A: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            acc = T.alloc_fragment((32, 32), "bfloat16")
            wide = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            for step in T.serial(2, 4):
                T.fill(wide, 1)
                T.gemm(a, a, acc, clear_accum=True)
                if step == 2:
                    T.fill(wide, 2)
                else:
                    T.fill(wide, 3)
                for i, j in T.Tiles(acc):
                    acc[i, j] = acc[i, j] + T.Cast("bfloat16", wide[i, j])
                T.copy(acc, C)

    mod = lower_tenstorrent_ir(tvm.IRModule({"main": main}), TARGET)
    assert "wide" not in _stores_by_buffer(mod)
    transform.VerifyTenstorrentDeviceIR()(mod)


def test_pipe_transactions_do_not_keep_dead_flash_attention_fragments_alive():
    from testing.python.target.fixtures.tenstorrent_flash_attn_03 import make_flash_attention

    counts = []
    for cores in (1, 2):
        frontend = make_flash_attention(batch=1, heads=1, seq_q=32 * cores, seq_kv=64, head_dim=32, core_q=cores, core_bh=1)
        mod = lower_tenstorrent_ir(tvm.IRModule({"main": frontend}), TARGET)
        counts.append(Counter(name.split("_core")[0] for name in _stores_by_buffer(mod)))
    assert counts[0]
    assert counts[1] == Counter({name: count * 2 for name, count in counts[0].items()})

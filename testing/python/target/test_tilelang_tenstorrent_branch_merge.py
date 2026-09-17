"""Runtime pure-compute branch merges preserve values and fixed transactions."""

import numpy as np
import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T, lower_tenstorrent_ir, transform
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_compute_values import interpret_values
from testing.python.target.test_tilelang_tenstorrent_phase4_semantics import run_device


METADATA = {"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2}


def branch_program(fragment=True, initialize=False, has_else=True, iterations=1):
    @T.prim_func
    def main(A: T.Tensor((64, 64), "float32"), P: T.Tensor((32, 32), "float32"), C: T.Tensor((64, 64), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((64, 64), "float32", annotations=METADATA)
            p = T.alloc_shared((32, 32), "float32", annotations=METADATA)
            if fragment:
                c = T.alloc_fragment((64, 64), "float32")
            else:
                c = T.alloc_shared((64, 64), "float32", annotations=METADATA)
            T.copy(A, a)
            T.copy(P, p)
            if initialize:
                for i, j in T.Tiles(c):
                    c[i, j] = a[i, j]
            for _ in T.serial(iterations):
                if p[0, 0] > 0:
                    for i, j in T.Tiles(c):
                        if initialize:
                            c[i, j] = c[i, j] * 2
                        else:
                            c[i, j] = a[i, j] * 2
                else:
                    if has_else:
                        for i, j in T.Tiles(c):
                            if initialize:
                                c[i, j] = c[i, j] - 3
                            else:
                                c[i, j] = a[i, j] - 3
            T.copy(c, C)

    return main


def lower(func, arch="wormhole_b0"):
    return lower_tenstorrent_ir(tvm.IRModule({"main": func}), tvm.target.Target({"kind": "tenstorrent", "arch": arch}))


@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
@pytest.mark.parametrize("fragment", [False, True])
@pytest.mark.parametrize("predicate", [-1, 0, 1, np.nan])
def test_dynamic_branch_selects_a_complete_value(arch, fragment, predicate):
    mod = lower(branch_program(fragment=fragment), arch)
    assert "tt.ir_stage" not in mod.attrs
    selects, branches = [], []
    for func in mod.functions.values():
        tirx.stmt_functor.post_order_visit(
            func.body,
            lambda node: (
                selects.append(node)
                if isinstance(node, tirx.Select)
                else branches.append(node)
                if isinstance(node, tirx.IfThenElse)
                else None
            ),
        )
    assert selects and not branches
    data = np.arange(64 * 64, dtype="float32").reshape(64, 64)
    flag = np.full((32, 32), predicate, dtype="float32")
    inputs = [data, flag, np.zeros_like(data)]
    outputs = interpret_values(mod, inputs)[0] if fragment else run_device(mod, inputs)
    np.testing.assert_array_equal(outputs[-1], data * 2 if predicate > 0 else data - 3)
    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(mod, restored)
    transform.VerifyTenstorrentDeviceIR()(restored)


@pytest.mark.parametrize("predicate", [-1, 1])
@pytest.mark.parametrize("has_else", [False, True])
def test_loop_carried_old_version_survives_conditional_update(predicate, has_else):
    func = branch_program(initialize=True, has_else=has_else, iterations=3)
    mod = lower(func)
    data = np.full((64, 64), 7, dtype="float32")
    inputs = [data, np.full((32, 32), predicate, dtype="float32"), np.zeros_like(data)]
    outputs, _ = interpret_values(mod, inputs)
    expected = data * 8 if predicate > 0 else data - 9 if has_else else data
    np.testing.assert_array_equal(outputs[-1], expected)
    assert ir.save_json(lower(func)) == ir.save_json(mod)


def test_missing_else_requires_an_initialized_old_value():
    with pytest.raises(ValueError, match="initialization|undefined|before"):
        lower(branch_program(has_else=False))


def test_predicate_must_be_initialized_even_when_both_arms_define_output():
    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations=METADATA)
            p = T.alloc_fragment((32, 32), "float32")
            c = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            if p[0, 0] > 0:
                for i, j in T.Tiles(c):
                    c[i, j] = a[i, j]
            else:
                for i, j in T.Tiles(c):
                    c[i, j] = a[i, j] * 2
            T.copy(c, C)

    with pytest.raises(ValueError, match="definite initialization"):
        lower(main)


def test_branch_with_transfer_effect_is_not_speculated():
    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), P: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations=METADATA)
            p = T.alloc_shared((32, 32), "float32", annotations=METADATA)
            T.copy(A, a)
            T.copy(P, p)
            if p[0, 0] > 0:
                T.copy(a, C)

    with pytest.raises(NotImplementedError, match="control flow|transaction"):
        lower(main)

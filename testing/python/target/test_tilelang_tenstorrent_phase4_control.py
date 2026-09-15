"""Structured iteration, conditional normalization and verifier boundaries."""

import numpy as np
import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_phase4_compute import shared, elementwise_program
from testing.python.target.test_tilelang_tenstorrent_phase4_device import replace_compute_annotation
from testing.python.target.test_tilelang_tenstorrent_phase4_semantics import lower, run_device
from testing.python.target.test_tilelang_tenstorrent_phase2_device_ir import _replace_dfb, _replace_slot_body, _slots


@T.prim_func
def repeated_inout(A: T.Tensor((64, 64), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = shared((64, 64))
        for _repeat in T.serial(2, 5):
            T.copy(A, a)
            for i, j in T.Parallel(64, 64):
                a[i, j] = a[i, j] + T.float32(2)
            T.copy(a, A)


@T.prim_func
def index_condition(A: T.Tensor((64, 64), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = shared((64, 64))
        for repeat in T.serial(3):
            T.copy(A, a)
            if repeat % 2 == 0:
                for i, j in T.Parallel(64, 64):
                    a[i, j] = a[i, j] + T.float32(2)
            else:
                for i, j in T.Parallel(64, 64):
                    a[i, j] = a[i, j] * T.float32(3)
            T.copy(a, A)


def test_independent_iteration_stays_structured_and_preserves_tensor_inout():
    mod = lower(repeated_inout)
    slots = _slots(mod)
    for slot in ("trisc", "ncrisc"):
        loop = slots[slot].body
        assert isinstance(loop, tirx.For)
        assert (int(loop.min), int(loop.extent)) == (2, 3)
    assert not slots["trisc"].body.loop_var.same_as(slots["ncrisc"].body.loop_var)
    assert all(int(d.transaction_count_or_loop_relation) == 3 for d in mod.attrs["tt.dfb_table"])
    a = np.arange(4096, dtype=np.float32).reshape(64, 64)
    np.testing.assert_array_equal(run_device(mod, [a])[0], a + 6)
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    target = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
    assert TenstorrentPassPipelineBody(mod, target).same_as(mod)


def test_index_dependent_conditions_normalize_without_semantic_loss():
    mod = lower(index_condition)
    assert all(not isinstance(func.body, tirx.For) for func in mod.functions.values())
    a = np.arange(4096, dtype=np.float32).reshape(64, 64)
    np.testing.assert_array_equal(run_device(mod, [a])[0], (a + 2) * 3 + 2)


@pytest.mark.parametrize("mutation", ["trip_count", "relation", "unwrapped", "step"])
def test_structured_iteration_malformed_schedule_is_rejected(mutation):
    mod = lower(repeated_inout)
    func = _slots(mod)["ncrisc"]
    loop = func.body
    if mutation == "relation":
        dfbs = list(mod.attrs["tt.dfb_table"])
        dfbs[0] = _replace_dfb(dfbs[0], transaction_count_or_loop_relation=2)
        mod = mod.with_attr("tt.dfb_table", dfbs)
    elif mutation == "unwrapped":
        _replace_slot_body(mod, "ncrisc", loop.body)
    else:
        changed = tirx.For(
            loop.loop_var,
            loop.min,
            4 if mutation == "trip_count" else loop.extent,
            loop.kind,
            loop.body,
            step=2 if mutation == "step" else None,
        )
        _replace_slot_body(mod, "ncrisc", changed)
    before = ir.save_json(mod)
    with pytest.raises(ValueError, match="loop|iteration|structured"):
        VerifyTenstorrentDeviceIR()(mod)
    assert ir.save_json(mod) == before


@pytest.mark.parametrize("kind", ["select", "float16_cast"])
def test_device_expression_subset_is_verified(kind):
    mod = lower(elementwise_program())
    call = next(
        stmt.value
        for stmt in _slots(mod)["trisc"].body.seq
        if isinstance(stmt.value, tirx.Call) and stmt.value.op.name == "tl.tt.dfb_compute"
    )
    expression = call.annotations["tt.expression"]
    if kind == "select":
        expression = tirx.Select(tirx.const(True, "bool"), expression, tirx.const(0, "float32"))
    else:
        expression = tirx.Cast("float32", tirx.Cast("float16", expression))
    replace_compute_annotation(mod, "tt.expression", expression)
    with pytest.raises(ValueError, match="unsupported.*(dtype|expression)|expression.*unsupported"):
        VerifyTenstorrentDeviceIR()(mod)

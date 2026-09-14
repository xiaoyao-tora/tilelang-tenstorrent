from __future__ import annotations

import pytest
from tvm import ir, tirx
from tvm.ir import Op

from tilelang import tvm
from tilelang.backend import create_backend_context
from tilelang.tenstorrent import execution_backend, transform
from tilelang.tenstorrent.device_ir import DFBDescriptor

from testing.python.target.test_tilelang_tenstorrent_phase0_contract import (
    FRONTEND_PROGRAMS,
)


TARGET_CONFIG = {"kind": "tenstorrent", "arch": "wormhole_b0"}
GOLDEN_MARKERS = {
    "trisc": (
        ("tl.tt.dfb_reserve", (2, 1)),
        ("tl.tt.dfb_wait", (0, 1)),
        ("tl.tt.dfb_wait", (1, 1)),
        ("tl.tt.dfb_add", (0, 1, 2, 1)),
    ),
    "ncrisc": (
        ("tl.tt.dfb_reserve", (0, 1)),
        ("tl.tt.tensor_to_dfb", (0, 0, 0, 0, 32, 32)),
        ("tl.tt.dfb_reserve", (1, 1)),
        ("tl.tt.tensor_to_dfb", (1, 1, 0, 0, 32, 32)),
        ("tl.tt.dfb_wait", (2, 1)),
        ("tl.tt.dfb_to_tensor", (2, 2, 0, 0, 32, 32)),
    ),
    "brisc": (),
}


def _context(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    return create_backend_context(
        TARGET_CONFIG,
        target_host="c",
        execution_backend="ttnn",
    )


def _lower_add(monkeypatch):
    context = _context(monkeypatch)
    frontend = tvm.IRModule({"add": FRONTEND_PROGRAMS["add"]})
    return context.lower(frontend)


def _slots(mod):
    return {str(func.attrs["tt.kernel_slot"]): func for func in mod.functions.values()}


def _marker_signature(func):
    if isinstance(func.body, tirx.Evaluate):
        assert int(func.body.value) == 0
        return ()
    assert isinstance(func.body, tirx.SeqStmt)
    result = []
    for statement in func.body.seq:
        assert isinstance(statement, tirx.Evaluate)
        assert isinstance(statement.value, tirx.Call)
        assert statement.value.dtype.bits == 0
        assert statement.value.dtype.lanes == 0
        result.append(
            (
                statement.value.op.name,
                tuple(int(argument) for argument in statement.value.args),
            )
        )
    return tuple(result)


def _replace_dfb(dfb, **changes):
    fields = {
        "dfb_id": int(dfb.dfb_id),
        "source_buffer_identity": str(dfb.source_buffer_identity),
        "element_dtype": dfb.element_dtype,
        "tile_shape": dfb.tile_shape,
        "block_shape_in_tiles": dfb.block_shape_in_tiles,
        "block_count": dfb.block_count,
        "tensor_backing": dfb.tensor_backing,
        "producer_slot": str(dfb.producer_slot),
        "producer_domain": dfb.producer_domain,
        "consumer_slot": str(dfb.consumer_slot),
        "consumer_domain": dfb.consumer_domain,
        "transaction_count_or_loop_relation": dfb.transaction_count_or_loop_relation,
        "source_span": dfb.source_span,
    }
    fields.update(changes)
    return DFBDescriptor(**fields)


def _replace_slot_body(mod, slot, body):
    for global_var, func in mod.functions.items():
        if str(func.attrs["tt.kernel_slot"]) == slot:
            mod[global_var] = func.with_body(body, func.span)
            return
    raise AssertionError(f"missing slot {slot}")


def test_phase2_add_device_ir_golden(monkeypatch):
    mod = _lower_add(monkeypatch)
    verified = transform.VerifyTenstorrentDeviceIR()(mod)
    assert verified.same_as(mod)

    tensors = list(mod.attrs["tt.tensor_table"])
    assert [int(tensor.global_arg_index) for tensor in tensors] == [0, 1, 2]
    assert [str(tensor.effect) for tensor in tensors] == ["input", "input", "output"]

    dfbs = list(mod.attrs["tt.dfb_table"])
    assert [int(dfb.dfb_id) for dfb in dfbs] == [0, 1, 2]
    assert [int(dfb.tensor_backing.global_arg_index) for dfb in dfbs] == [0, 1, 2]
    assert [(str(dfb.producer_slot), str(dfb.consumer_slot)) for dfb in dfbs] == [
        ("ncrisc", "trisc"),
        ("ncrisc", "trisc"),
        ("trisc", "ncrisc"),
    ]
    assert [int(dfb.block_count) for dfb in dfbs] == [2, 2, 2]
    assert [int(dfb.transaction_count_or_loop_relation) for dfb in dfbs] == [1, 1, 1]
    assert not mod.attrs["tt.pipe_table"]

    slots = _slots(mod)
    assert {slot: _marker_signature(func) for slot, func in slots.items()} == GOLDEN_MARKERS
    assert [int(index) for index in slots["ncrisc"].attrs["tt.tensor_arg_indices"]] == [
        0,
        1,
        2,
    ]
    assert not slots["trisc"].attrs["tt.tensor_arg_indices"]
    assert not slots["brisc"].attrs["tt.tensor_arg_indices"]
    assert str(slots["trisc"].attrs["tt.logical_kernel"].role) == "add"
    assert str(slots["ncrisc"].attrs["tt.logical_kernel"].role) == "tensor_io"
    assert str(slots["brisc"].attrs["tt.logical_kernel"].role) == "idle"


def test_phase2_add_json_round_trip_and_determinism(monkeypatch):
    first = _lower_add(monkeypatch)
    second = _lower_add(monkeypatch)
    ir.assert_structural_equal(first, second)
    assert ir.structural_hash(first) == ir.structural_hash(second)
    assert str(first) == str(second)

    restored = ir.load_json(ir.save_json(first))
    ir.assert_structural_equal(restored, first)
    assert str(restored) == str(first)
    assert transform.VerifyTenstorrentDeviceIR()(restored).same_as(restored)


@pytest.mark.parametrize(
    ("index", "changes", "message"),
    (
        (0, {"tensor_backing": None}, "requires Tensor backing"),
        (0, {"consumer_slot": "ncrisc"}, "distinct SPSC"),
        (0, {"producer_slot": "brisc"}, "producer_slot must be `ncrisc`"),
        (2, {"transaction_count_or_loop_relation": 2}, "transaction count must be exactly one"),
    ),
)
def test_phase2_add_rejects_malformed_dfb_contract(monkeypatch, index, changes, message):
    mod = _lower_add(monkeypatch)
    dfbs = list(mod.attrs["tt.dfb_table"])
    dfbs[index] = _replace_dfb(dfbs[index], **changes)
    mod = mod.with_attr("tt.dfb_table", dfbs)
    before = ir.save_json(mod)
    with pytest.raises(Exception, match=message):
        transform.VerifyTenstorrentDeviceIR()(mod)
    assert ir.save_json(mod) == before


def test_phase2_add_rejects_marker_order_and_idle_brisc_body(monkeypatch):
    mod = _lower_add(monkeypatch)
    trisc = _slots(mod)["trisc"]
    reordered = list(trisc.body.seq)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    _replace_slot_body(mod, "trisc", tirx.SeqStmt(reordered))
    with pytest.raises(Exception, match="marker 0 must be `tl.tt.dfb_reserve`"):
        transform.VerifyTenstorrentDeviceIR()(mod)

    mod = _lower_add(monkeypatch)
    trisc_first_marker = _slots(mod)["trisc"].body.seq[0]
    _replace_slot_body(mod, "brisc", trisc_first_marker)
    with pytest.raises(Exception, match="brisc body must be the canonical"):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_phase2_add_rejects_cross_slot_var_and_intermediate_tile_add(monkeypatch):
    mod = _lower_add(monkeypatch)
    slots = _slots(mod)
    trisc_body = slots["trisc"].body
    first = trisc_body.seq[0]
    foreign_param = slots["ncrisc"].params[0]
    cross_slot_call = tirx.Call(
        "void",
        first.value.op,
        [foreign_param, first.value.args[1]],
        span=first.value.span,
    )
    cross_slot_body = list(trisc_body.seq)
    cross_slot_body[0] = tirx.Evaluate(cross_slot_call)
    _replace_slot_body(mod, "trisc", tirx.SeqStmt(cross_slot_body))
    with pytest.raises(Exception, match="undefined or cross-function Var"):
        transform.VerifyTenstorrentDeviceIR()(mod)

    mod = _lower_add(monkeypatch)
    trisc_body = _slots(mod)["trisc"].body
    first = trisc_body.seq[0]
    intermediate_call = tirx.Call(
        "void",
        Op.get("tl.tt.tile_add"),
        [first.value.args[0], first.value.args[0], first.value.args[0]],
        span=first.value.span,
    )
    intermediate_body = list(trisc_body.seq)
    intermediate_body[0] = tirx.Evaluate(intermediate_call)
    _replace_slot_body(mod, "trisc", tirx.SeqStmt(intermediate_body))
    with pytest.raises(Exception, match="retains intermediate op `tl.tt.tile_add`"):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_phase2_add_rejects_non_add_topology(monkeypatch):
    context = _context(monkeypatch)
    with pytest.raises(NotImplementedError, match="TileOp|Topology|Pipe"):
        context.lower(tvm.IRModule({"p2p": FRONTEND_PROGRAMS["p2p"]}))

"""Reject malformed multicore accumulation and repeated communication IR."""

import pytest

from tilelang.tenstorrent.device_ir import ComputeRequirements
from tvm import tirx

from testing.python.target.test_tilelang_tenstorrent_phase5_resources import rejected
from testing.python.target.test_tilelang_tenstorrent_phase6_resources import replace_call, replace_transfer
from testing.python.target.test_tilelang_tenstorrent_summa_lower import lower, make_summa


def module():
    return lower(make_summa(core_n=3, stages=2))


def function(mod, slot, x=0, y=0):
    return next(
        (gv, func)
        for gv, func in mod.functions.items()
        if str(func.attrs["tt.kernel_slot"]) == slot
        and int(func.attrs["tt.core_domain"].begin.x) == x
        and int(func.attrs["tt.core_domain"].begin.y) == y
    )


def op(stmt):
    return stmt.value.op.name


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("cross_core", "cross Core or processor ownership"),
        ("missing_update", "full-K updates"),
        ("duplicate_init", "initialized more than once"),
        ("after_materialize", "live initialized accumulator"),
    ],
)
def test_summa_accumulator_lifetime_and_owner(mutation, match):
    mod = module()
    gv, func = function(mod, "trisc")
    body = list(func.body.seq)
    update = next(s for s in body if op(s) == "tl.tt.gemm_update")
    if mutation == "cross_core":
        _, other = function(mod, "trisc", x=1)
        other_init = next(s for s in other.body.seq if op(s) == "tl.tt.accumulator_init")
        args = list(update.value.args)
        args[2] = other_init.value.args[0]
        body[body.index(update)] = replace_call(update, args)
    elif mutation == "missing_update":
        body.remove(update)
    elif mutation == "duplicate_init":
        init = next(s for s in body if op(s) == "tl.tt.accumulator_init")
        body.insert(body.index(init), init)
    else:
        body.append(update)
    mod.update_func(gv, func.with_body(tirx.SeqStmt(body)))
    rejected(mod, match)


def test_summa_precision_requirement_is_not_repaired():
    mod = module()
    gv, func = function(mod, "trisc")
    mod.update_func(gv, func.with_attr("tt.compute_requirements", ComputeRequirements("bits16_required", "forbidden")))
    rejected(mod, "tt.compute_requirements")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "duplicate destination transaction"),
        ("gap", "contiguous from zero"),
        ("negative", "nonnegative epoch"),
        ("missing_receiver", "missing receiver"),
        ("inconsistent_source", "inconsistent source payloads"),
    ],
)
def test_summa_communication_epochs(mutation, match):
    mod = module()
    transfers = list(mod.attrs["tt.pipe_transfer_table"])
    second = next(i for i, t in enumerate(transfers) if int(t.occurrence) == 1)
    if mutation == "duplicate":
        transfers[second] = replace_transfer(transfers[second], occurrence=0)
    elif mutation in ("gap", "negative"):
        value = 2 if mutation == "gap" else -1
        transfers = [replace_transfer(t, occurrence=value) if int(t.occurrence) == 1 else t for t in transfers]
    elif mutation == "missing_receiver":
        # Keep stable IDs so the closed destination set is the failing invariant.
        transfers.pop(second)
        transfers = [replace_transfer(t, transfer_id=i) for i, t in enumerate(transfers)]
    else:
        first = transfers[0]
        peer = next(
            i
            for i, t in enumerate(transfers[1:], 1)
            if int(t.pipe_net_id) == int(first.pipe_net_id) and int(t.record_index) == int(first.record_index) and int(t.occurrence) == 1
        )
        sibling = next(
            i
            for i, t in enumerate(transfers[peer + 1 :], peer + 1)
            if int(t.pipe_net_id) == int(transfers[peer].pipe_net_id)
            and int(t.record_index) == int(transfers[peer].record_index)
            and int(t.occurrence) == 1
        )
        transfers[sibling] = replace_transfer(transfers[sibling], source_dfb_id=int(first.source_dfb_id))
    rejected(mod.with_attr("tt.pipe_transfer_table", transfers), match)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_completion", "missing completion synchronization"),
        ("forward_before_copy", "Tensor copy completion"),
    ],
)
def test_summa_forwarding_keeps_transport_lifetime(mutation, match):
    mod = module()
    gv, func = function(mod, "ncrisc")
    body = list(func.body.seq)
    if mutation == "missing_completion":
        body.pop(next(i for i, s in enumerate(body) if op(s) == "tl.tt.dfb_pipe_wait"))
    else:
        index = next(i for i, s in enumerate(body) if op(s) == "tl.tt.dfb_copy_wait")
        completion = body.pop(index)
        send = next(i for i, s in enumerate(body) if op(s) == "tl.tt.dfb_pipe_send")
        body.insert(send + 1, completion)
    mod.update_func(gv, func.with_body(tirx.SeqStmt(body)))
    rejected(mod, match)


def test_summa_explicit_l1_budget_remains_enforced():
    rejected(module().with_attr("tt.l1_capacity_bytes", 1), "exceeds explicit tt.l1_capacity_bytes")


def test_summa_forwarding_completion_cannot_move_to_brisc():
    mod = module()
    gv, func = function(mod, "ncrisc")
    body = list(func.body.seq)
    index = next(i for i, stmt in enumerate(body) if op(stmt) == "tl.tt.dfb_pipe_wait")
    completion = body.pop(index)
    mod.update_func(gv, func.with_body(tirx.SeqStmt(body)))
    brisc_gv, brisc = function(mod, "brisc")
    mod.update_func(brisc_gv, brisc.with_body(completion))
    rejected(mod, "source completion must follow send in its source slot")

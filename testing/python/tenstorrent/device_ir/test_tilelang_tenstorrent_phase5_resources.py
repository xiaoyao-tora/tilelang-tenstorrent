"""Logical DFB stream capacity and asynchronous lifetime verification."""

import inspect

import pytest

from tilelang import tvm
from tilelang.tenstorrent import device_ir as D, transform
from tvm import ir, tirx

from testing.python.tenstorrent._support.reference import _calls, _with_core_statements
from testing.python.tenstorrent.legacy.test_tilelang_tenstorrent_phase2_device_ir import (
    _lower_add,
    _slots,
)
from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_phase4_compute import (
    elementwise_program,
)
from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_phase5_pipeline import (
    lower_pipeline,
    make_pipeline,
)
from testing.python.tenstorrent.semantics.test_tilelang_tenstorrent_phase4_semantics import (
    lower,
)


def pipeline(stages=2, extent=3, capacity=None):
    mod = lower_pipeline(make_pipeline(stages, extent, block_count=capacity))
    assert str(mod.attrs["tt.ir_stage"]) == "device"
    assert "tt.device_ir_version" not in mod.attrs
    assert "tt.dfb_families" not in mod.attrs
    assert "tt.dfb_storage_group_relations" not in mod.attrs
    return mod


def statements(mod, slot):
    return [
        tirx.Evaluate(call)
        for call in _calls(
            _slots(mod)[slot].body,
            collapse_resource_refs=False,
        )
    ]


def stream_id(expression):
    assert isinstance(expression, tirx.Call)
    assert expression.op.name == "tl.tt.dfb_stream_ref"
    return int(expression.args[0])


def replace_statements(mod, slot, body):
    for global_var, func in mod.functions.items():
        if str(func.attrs["tt.kernel_slot"]) == slot:
            mod[global_var] = _with_core_statements(func, body)
            return
    raise AssertionError(f"missing slot {slot}")


def operation(statement):
    return statement.value.op.name


def rejected(mod, match):
    before = ir.save_json(mod)
    with pytest.raises(ValueError, match=match):
        transform.VerifyTenstorrentDeviceIR()(mod)
    assert ir.save_json(mod) == before, "verifier must not repair malformed input"


def replace_group(group, **changes):
    fields = {name: getattr(group, name) for name in inspect.signature(D.DFBStorageGroupDescriptor.__init__).parameters if name != "self"}
    fields.update(changes)
    return D.DFBStorageGroupDescriptor(**fields)


def pool_payload(mod):
    total = 0
    for group in mod.attrs["tt.dfb_storage_groups"]:
        size = int(group.block_count) * tvm.DataType(str(group.element_dtype)).bits // 8
        for extent in [*group.tile_shape, *group.block_shape_in_tiles]:
            size *= int(extent)
        total += size
    return total


@pytest.mark.parametrize("stages,extent", [(1, 1), (2, 1), (2, 3), (3, 5), (4, 2)])
def test_stream_capacity_and_closed_lifetimes(stages, extent):
    mod = pipeline(stages, extent)
    depth = min(stages, extent)
    groups = list(mod.attrs["tt.dfb_storage_groups"])
    dfbs = list(mod.attrs["tt.dfb_table"])
    assert len(groups) == len(dfbs) == 4
    assert [int(group.storage_group_id) for group in groups] == list(range(len(groups)))
    assert all(int(group.block_count) == depth for group in groups)
    releases = [
        stream_id(stmt.value.args[0])
        for slot in ("trisc", "ncrisc")
        for stmt in statements(mod, slot)
        if operation(stmt) == "tl.tt.dfb_release"
    ]
    assert sorted(releases) == sorted(list(range(len(groups))) * extent)
    transform.VerifyTenstorrentDeviceIR()(ir.load_json(ir.save_json(mod)))


def test_pipeline_keeps_one_lexical_copy_completion_per_site():
    mod = pipeline(3, 4)
    body = statements(mod, "ncrisc")
    first_completion = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_copy_wait")
    prefix = body[:first_completion]
    assert sum(operation(stmt) == "tl.tt.tensor_to_dfb_nd" for stmt in prefix) == 1
    assert sum(operation(stmt) == "tl.tt.dfb_reserve" for stmt in prefix) == 1
    assert all(operation(stmt) != "tl.tt.dfb_release" for stmt in prefix)


def test_explicit_stream_capacity_preserved():
    mod = pipeline(2, 3, capacity=4)
    assert all(int(group.block_count) == 4 for group in mod.attrs["tt.dfb_storage_groups"])
    assert pool_payload(mod) > pool_payload(pipeline(2, 3))


def test_insufficient_explicit_capacity_rejected_during_lower():
    with pytest.raises(ValueError, match="capacity|block_count|window"):
        pipeline(3, 4, capacity=2)


def test_invalid_stream_capacity_rejected_by_verifier():
    mod = pipeline(2, 3)
    groups = list(mod.attrs["tt.dfb_storage_groups"])
    groups[0] = replace_group(groups[0], block_count=0)
    rejected(
        mod.with_attr("tt.dfb_storage_groups", groups),
        "capacity must be in",
    )


def test_lifecycle_operations_have_one_stream_operand():
    mod = pipeline()
    lifecycle = {
        "tl.tt.dfb_reserve",
        "tl.tt.dfb_wait",
        "tl.tt.dfb_copy_wait",
        "tl.tt.dfb_release",
    }
    calls = [stmt.value for slot in ("trisc", "ncrisc") for stmt in statements(mod, slot) if operation(stmt) in lifecycle]
    assert calls and all(len(call.args) == 1 for call in calls)


def test_legacy_lifecycle_transaction_count_rejected():
    mod = pipeline()
    body = statements(mod, "trisc")
    index = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_wait")
    call = body[index].value
    body[index] = tirx.Evaluate(
        tirx.Call(
            call.dtype,
            call.op,
            [*call.args, tirx.IntImm("int32", 1)],
            call.annotations,
            call.span,
        )
    )
    replace_statements(mod, "trisc", body)
    rejected(mod, "lifecycle operation requires one stream operand")


@pytest.mark.parametrize(
    "name,match",
    [
        ("tl.tt.dfb_wait", "missing consumer wait|no active acquisition"),
        ("tl.tt.dfb_release", "missing release|no matching producer publication"),
    ],
)
def test_missing_consumer_synchronization(name, match):
    mod = pipeline()
    body = statements(mod, "trisc")
    body.pop(next(i for i, stmt in enumerate(body) if operation(stmt) == name))
    replace_statements(mod, "trisc", body)
    rejected(mod, match)


def test_missing_copy_completion():
    mod = pipeline()
    body = statements(mod, "ncrisc")
    body.pop(next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_copy_wait"))
    replace_statements(mod, "ncrisc", body)
    rejected(mod, "no matching producer|missing completion|nested producer reservations")


def test_output_release_before_completion():
    mod = pipeline()
    body = statements(mod, "ncrisc")
    index = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_release")
    assert operation(body[index - 1]) == "tl.tt.dfb_copy_wait"
    body[index - 1], body[index] = body[index], body[index - 1]
    replace_statements(mod, "ncrisc", body)
    rejected(
        mod,
        "no active acquisition|no active transaction|release before copy completion|dependency cycle",
    )


def test_use_after_release():
    mod = pipeline()
    body = statements(mod, "trisc")
    index = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_release")
    released = stream_id(body[index].value.args[0])
    prior = next(stmt for stmt in body[:index] if operation(stmt) == "tl.tt.dfb_wait" and stream_id(stmt.value.args[0]) == released)
    body.insert(index + 1, prior)
    replace_statements(mod, "trisc", body)
    rejected(mod, "after release|no matching producer|release precedes")


def test_duplicate_reserve_rejected():
    mod = pipeline()
    body = statements(mod, "ncrisc")
    index = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_reserve")
    body.insert(index, body[index])
    replace_statements(mod, "ncrisc", body)
    rejected(
        mod,
        "reserved more than once|nested producer reservations|transaction count disagrees",
    )


def test_l1_budget_counts_streams_once_and_has_exact_boundary():
    mod = pipeline(2, 5)
    payload = pool_payload(mod)
    transform.VerifyTenstorrentDeviceIR()(mod.with_attr("tt.l1_capacity_bytes", payload))
    rejected(
        mod.with_attr("tt.l1_capacity_bytes", payload - 1),
        "L1 logical payload lower bound.*exceeds",
    )
    rejected(mod.with_attr("tt.l1_capacity_bytes", 0), "must be positive")


def test_legacy_v2_explicit_l1_budget():
    mod = lower(elementwise_program())
    rejected(
        mod.with_attr("tt.l1_capacity_bytes", 1),
        "L1 logical payload lower bound.*exceeds",
    )
    transform.VerifyTenstorrentDeviceIR()(mod.with_attr("tt.l1_capacity_bytes", 1 << 30))


def test_legacy_v1_explicit_l1_budget(monkeypatch):
    mod = _lower_add(monkeypatch)
    rejected(
        mod.with_attr("tt.l1_capacity_bytes", 1),
        "L1 logical payload lower bound.*exceeds",
    )
    transform.VerifyTenstorrentDeviceIR()(mod.with_attr("tt.l1_capacity_bytes", 1 << 30))


def test_malformed_stream_table_rejected():
    mod = pipeline()
    groups = list(mod.attrs["tt.dfb_storage_groups"])
    groups[0] = replace_group(groups[0], storage_group_id=99)
    rejected(
        mod.with_attr("tt.dfb_storage_groups", groups),
        "IDs must be dense",
    )


def test_invalid_wait_policy_rejected():
    rejected(
        pipeline().with_attr("tt.pipeline_wait_policy", "unknown"),
        "wait policy",
    )


def test_pipeline_resource_metadata_determinism():
    program = make_pipeline(3, 5)
    first = lower_pipeline(program)
    second = lower_pipeline(program)
    ir.assert_structural_equal(first, second)
    assert ir.structural_hash(first) == ir.structural_hash(second)
    assert [int(group.storage_group_id) for group in first.attrs["tt.dfb_storage_groups"]] == list(
        range(len(first.attrs["tt.dfb_storage_groups"]))
    )

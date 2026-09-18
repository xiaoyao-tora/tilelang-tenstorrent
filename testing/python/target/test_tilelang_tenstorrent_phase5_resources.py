"""Pipeline pool capacity and asynchronous lifetime verification, without hardware."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import transform
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_phase2_device_ir import (
    _lower_add,
    _replace_dfb,
    _replace_slot_body,
    _slots,
)
from testing.python.target.test_tilelang_tenstorrent_phase4_compute import elementwise_program
from testing.python.target.test_tilelang_tenstorrent_phase4_semantics import lower


from testing.python.target.test_tilelang_tenstorrent_phase5_pipeline import make_pipeline, lower_pipeline


def pipeline(stages=2, extent=3, capacity=None):
    mod = lower_pipeline(make_pipeline(stages, extent, block_count=capacity))
    assert int(mod.attrs["tt.device_ir_version"]) == 3
    return mod


def statements(mod, slot):
    return list(_slots(mod)[slot].body.seq)


def operation(statement):
    return statement.value.op.name


def rejected(mod, match):
    before = ir.save_json(mod)
    with pytest.raises(ValueError, match=match):
        transform.VerifyTenstorrentDeviceIR()(mod)
    assert ir.save_json(mod) == before, "verifier must not repair malformed input"


def pool_payload(mod):
    groups = {int(key): int(value) for key, value in mod.attrs["tt.dfb_storage_groups"].items()}
    seen = set()
    total = 0
    for dfb in mod.attrs["tt.dfb_table"]:
        group = groups[int(dfb.dfb_id)]
        if group in seen:
            continue
        seen.add(group)
        size = int(dfb.block_count) * tvm.DataType(str(dfb.element_dtype)).bits // 8
        for extent in [*dfb.tile_shape, *dfb.block_shape_in_tiles]:
            size *= int(extent)
        total += size
    return total


@pytest.mark.parametrize("stages,extent", [(1, 1), (2, 1), (2, 3), (3, 5), (4, 2)])
def test_pool_capacity_and_closed_lifetimes(stages, extent):
    mod = pipeline(stages, extent)
    assert all(int(dfb.block_count) == min(stages, extent) for dfb in mod.attrs["tt.dfb_table"])
    releases = [
        int(stmt.value.args[0]) for slot in ("trisc", "ncrisc") for stmt in statements(mod, slot) if operation(stmt) == "tl.tt.dfb_release"
    ]
    assert sorted(releases) == sorted(int(dfb.dfb_id) for dfb in mod.attrs["tt.dfb_table"])
    transform.VerifyTenstorrentDeviceIR()(ir.load_json(ir.save_json(mod)))


def test_nested_reservations_and_multiple_inflight_copies():
    mod = pipeline(3, 4)
    body = statements(mod, "ncrisc")
    first_completion = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_copy_wait")
    prefix = body[:first_completion]
    assert sum(operation(stmt) == "tl.tt.tensor_to_dfb_nd" for stmt in prefix) == 6
    assert sum(operation(stmt) == "tl.tt.dfb_reserve" for stmt in prefix) == 6
    # Acquires are held across all six issues. Completion alone does not free them.
    assert all(operation(stmt) != "tl.tt.dfb_release" for stmt in prefix)


def test_explicit_pool_capacity_preserved():
    mod = pipeline(2, 3, capacity=4)
    assert all(int(dfb.block_count) == 4 for dfb in mod.attrs["tt.dfb_table"])
    assert pool_payload(mod) > pool_payload(pipeline(2, 3))


def test_insufficient_explicit_capacity_rejected_during_lower():
    with pytest.raises(ValueError, match="capacity|block_count|window"):
        pipeline(3, 4, capacity=2)


def test_insufficient_capacity_rejected_by_verifier():
    mod = pipeline(2, 3)
    changed = [_replace_dfb(dfb, block_count=1) for dfb in mod.attrs["tt.dfb_table"]]
    rejected(mod.with_attr("tt.dfb_table", changed), "capacity insufficient")


@pytest.mark.parametrize("field,values", [("stage", [0, 1]), ("iteration", [99, 0])])
def test_wrong_transaction_relation(field, values):
    mod = pipeline()
    relations = dict(mod.attrs["tt.pipeline_relations"].items())
    key = next(key for key, value in relations.items() if int(value[0]) == 0)
    relations[key] = tvm.runtime.convert(values)
    rejected(mod.with_attr("tt.pipeline_relations", relations), "transaction relation")


@pytest.mark.parametrize(
    "name,match",
    [
        ("tl.tt.dfb_wait", "preceded by dfb_wait|release must follow"),
        ("tl.tt.dfb_release", "missing release"),
    ],
)
def test_missing_consumer_synchronization(name, match):
    mod = pipeline()
    body = statements(mod, "trisc")
    body.pop(next(i for i, stmt in enumerate(body) if operation(stmt) == name))
    _replace_slot_body(mod, "trisc", tirx.SeqStmt(body))
    rejected(mod, match)


def test_missing_copy_completion():
    mod = pipeline()
    body = statements(mod, "ncrisc")
    body.pop(next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_copy_wait"))
    _replace_slot_body(mod, "ncrisc", tirx.SeqStmt(body))
    rejected(mod, "never published|missing completion")


def test_output_release_before_completion():
    mod = pipeline()
    body = statements(mod, "ncrisc")
    index = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_release")
    assert operation(body[index - 1]) == "tl.tt.dfb_copy_wait"
    body[index - 1], body[index] = body[index], body[index - 1]
    _replace_slot_body(mod, "ncrisc", tirx.SeqStmt(body))
    rejected(mod, "release before copy completion")


def test_use_after_release():
    mod = pipeline()
    body = statements(mod, "trisc")
    index = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_release")
    released = int(body[index].value.args[0])
    prior = next(stmt for stmt in body[:index] if operation(stmt) == "tl.tt.dfb_wait" and int(stmt.value.args[0]) == released)
    body.insert(index + 1, prior)
    _replace_slot_body(mod, "trisc", tirx.SeqStmt(body))
    rejected(mod, "after release")


def test_duplicate_nested_acquire_rejected():
    mod = pipeline()
    body = statements(mod, "ncrisc")
    index = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_reserve")
    body.insert(index, body[index])
    _replace_slot_body(mod, "ncrisc", tirx.SeqStmt(body))
    rejected(mod, "reserved more than once")


def test_ring_reuse_dependency_catches_missing_window_drain():
    mod = pipeline(2, 3)
    body = statements(mod, "ncrisc")
    # Delaying every input completion until all windows' issues creates a cycle:
    # the next ring acquire needs a release that itself needs a delayed completion.
    inputs = [stmt for stmt in body if operation(stmt) in ("tl.tt.dfb_reserve", "tl.tt.tensor_to_dfb_nd")]
    rest = [stmt for stmt in body if operation(stmt) not in ("tl.tt.dfb_reserve", "tl.tt.tensor_to_dfb_nd")]
    _replace_slot_body(mod, "ncrisc", tirx.SeqStmt(inputs + rest))
    rejected(mod, "dependency cycle")


def test_l1_budget_counts_pools_once_and_has_exact_boundary():
    mod = pipeline(2, 5)
    payload = pool_payload(mod)
    transform.VerifyTenstorrentDeviceIR()(mod.with_attr("tt.l1_capacity_bytes", payload))
    rejected(mod.with_attr("tt.l1_capacity_bytes", payload - 1), "L1 logical payload lower bound.*exceeds")
    rejected(mod.with_attr("tt.l1_capacity_bytes", 0), "must be positive")


def test_legacy_v2_explicit_l1_budget():
    mod = lower(elementwise_program())
    rejected(mod.with_attr("tt.l1_capacity_bytes", 1), "L1 logical payload lower bound.*exceeds")
    transform.VerifyTenstorrentDeviceIR()(mod.with_attr("tt.l1_capacity_bytes", 1 << 30))


def test_legacy_v1_explicit_l1_budget(monkeypatch):
    mod = _lower_add(monkeypatch)
    rejected(mod.with_attr("tt.l1_capacity_bytes", 1), "L1 logical payload lower bound.*exceeds")
    transform.VerifyTenstorrentDeviceIR()(mod.with_attr("tt.l1_capacity_bytes", 1 << 30))


def test_globally_permuted_iteration_relations_rejected():
    mod = pipeline(3, 3)
    relations = {
        key: tvm.runtime.convert([2 - int(value[0]), 2 - int(value[0])]) for key, value in mod.attrs["tt.pipeline_relations"].items()
    }
    rejected(mod.with_attr("tt.pipeline_relations", relations), "order disagrees with transaction relation")


@pytest.mark.parametrize("mutation", ["missing", "duplicate_epoch", "pool_mismatch"])
def test_malformed_resource_group_metadata(mutation):
    mod = pipeline()
    if mutation == "missing":
        groups = dict(mod.attrs["tt.dfb_storage_groups"].items())
        groups.pop(next(iter(groups)))
        mod = mod.with_attr("tt.dfb_storage_groups", groups)
        match = "cover exactly"
    elif mutation == "duplicate_epoch":
        relations = dict(mod.attrs["tt.pipeline_relations"].items())
        key = next(key for key, value in relations.items() if int(value[0]) == 1)
        relations[key] = tvm.runtime.convert([0, 0])
        mod = mod.with_attr("tt.pipeline_relations", relations)
        match = "duplicate iteration"
    else:
        dfbs = list(mod.attrs["tt.dfb_table"])
        dfbs[0] = _replace_dfb(dfbs[0], block_count=3)
        mod = mod.with_attr("tt.dfb_table", dfbs)
        match = "generations disagree"
    rejected(mod, match)


def test_invalid_wait_policy_rejected():
    rejected(pipeline().with_attr("tt.pipeline_wait_policy", "unknown"), "wait_policy")


def test_l1_payload_overflow_rejected():
    mod = pipeline()
    dfbs = [_replace_dfb(dfb, tile_shape=[tirx.IntImm("int64", 1 << 62), tirx.IntImm("int32", 32)]) for dfb in mod.attrs["tt.dfb_table"]]
    rejected(mod.with_attr("tt.dfb_table", dfbs), "overflows static byte/capacity arithmetic")


def test_pipeline_generation_publication_count_is_not_iteration_count():
    mod = pipeline()
    dfbs = list(mod.attrs["tt.dfb_table"])
    dfbs[0] = _replace_dfb(dfbs[0], transaction_count_or_loop_relation=3)
    rejected(mod.with_attr("tt.dfb_table", dfbs), "exactly one publication")


@pytest.mark.parametrize("attribute", ["tt.dfb_storage_groups", "tt.pipeline_relations"])
@pytest.mark.parametrize("key", ["00", "not-a-dfb", "-1", "999999999999999999999999999999"])
def test_pipeline_metadata_requires_canonical_decimal_keys(attribute, key):
    mod = pipeline()
    table = dict(mod.attrs[attribute].items())
    original = next(iter(table))
    table[key] = table.pop(original)
    rejected(mod.with_attr(attribute, table), "canonical decimal DFB key")


def test_pipeline_resource_metadata_determinism():
    program = make_pipeline(3, 5)
    first = lower_pipeline(program)
    second = lower_pipeline(program)
    ir.assert_structural_equal(first, second)
    assert ir.structural_hash(first) == ir.structural_hash(second)
    for attribute in ("tt.dfb_storage_groups", "tt.pipeline_relations"):
        assert all(str(int(key)) == str(key) for key in first.attrs[attribute])

"""Read-only resource and communication validation for static Device IR v4.

These tests mutate lowered Device IR; they do not execute a TT runtime or NoC.
"""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T, transform
from tilelang.tenstorrent.device_ir import CoreCoord, CoreDomain, PipeTransferDescriptor
from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_phase2_device_ir import _replace_dfb
from testing.python.target.test_tilelang_tenstorrent_phase5_resources import rejected


def resource_program(collective=False, duplicate=False):
    destination = T.comm.CoreRange((1, 0), (3, 1)) if collective else (1, 0)
    pipes = [T.comm.Pipe((0, 0), destination)]
    if duplicate:
        pipes.append(T.comm.Pipe((0, 0), destination))
    net = T.comm.PipeNet(pipes)
    grid = 3 if collective else 2

    @T.prim_func
    def kernel(A: T.Tensor((32, 32), "float32"), C: T.Tensor((96, 32), "float32")):
        with T.Kernel(grid, 1, threads=1) as (x, y):
            source = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32)})
            destination_buffer = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32)})
            if T.comm.is_src(net):
                T.copy(A, source)
            for pipe in T.comm.foreach_src(net):
                T.copy(source, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, destination_buffer)
                T.copy(destination_buffer, C[x * 32 : (x + 1) * 32, :])

    return kernel


def resource_module(collective=False, duplicate=False):
    target = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
    return TenstorrentPassPipelineBody(tvm.IRModule({"main": resource_program(collective, duplicate)}), target)


def slot_function(mod, x, slot):
    return next(
        (global_var, func)
        for global_var, func in mod.functions.items()
        if int(func.attrs["tt.core_domain"].begin.x) == x and str(func.attrs["tt.kernel_slot"]) == slot
    )


def body_statements(mod, x, slot):
    _, func = slot_function(mod, x, slot)
    return list(func.body.seq) if isinstance(func.body, tirx.SeqStmt) else [func.body]


def replace_body(mod, x, slot, body):
    global_var, func = slot_function(mod, x, slot)
    mod[global_var] = func.with_body(tirx.SeqStmt(body), func.span)


def operation(statement):
    return statement.value.op.name


def replace_call(statement, args):
    call = statement.value
    return tirx.Evaluate(tirx.Call(call.dtype, call.op, args, annotations=call.annotations, span=call.span))


def replace_transfer(transfer, **changes):
    fields = {
        "transfer_id": int(transfer.transfer_id),
        "pipe_net_id": int(transfer.pipe_net_id),
        "record_index": int(transfer.record_index),
        "occurrence": int(transfer.occurrence),
        "src_coord": transfer.src_coord,
        "dst_coord": transfer.dst_coord,
        "source_dfb_id": int(transfer.source_dfb_id),
        "destination_dfb_id": int(transfer.destination_dfb_id),
        "transaction_count": int(transfer.transaction_count),
        "source_span": transfer.source_span,
    }
    fields.update(changes)
    return PipeTransferDescriptor(**fields)


@pytest.mark.parametrize("collective,duplicate", [(False, False), (True, False), (False, True), (True, True)])
def test_resources_have_closed_lifetimes_and_preserve_record_multiplicity(collective, duplicate):
    mod = resource_module(collective, duplicate)
    transfers = mod.attrs["tt.pipe_transfer_table"]
    assert len(transfers) == (2 if collective else 1) * (2 if duplicate else 1)
    assert len({int(t.destination_dfb_id) for t in transfers}) == len(transfers)
    assert {int(t.record_index) for t in transfers} == set(range(2 if duplicate else 1))
    assert all(int(t.transaction_count) == 1 and int(t.occurrence) == 0 for t in transfers)
    ir.assert_structural_equal(mod, transform.VerifyTenstorrentDeviceIR()(mod))
    restored = ir.load_json(ir.save_json(mod))
    transform.VerifyTenstorrentDeviceIR()(restored)
    ir.assert_structural_equal(mod, restored)


@pytest.mark.parametrize(
    "slot,x,name,match",
    [
        ("brisc", 0, "tl.tt.dfb_wait", "preceded by dfb_wait|release must follow"),
        ("brisc", 0, "tl.tt.dfb_pipe_send", "completion must follow send|matching is not closed"),
        ("brisc", 0, "tl.tt.dfb_pipe_wait", "missing completion synchronization"),
        ("ncrisc", 1, "tl.tt.dfb_pipe_recv", "completion must follow receive|matching is not closed"),
        ("ncrisc", 1, "tl.tt.dfb_pipe_wait", "never published|missing completion synchronization"),
        ("brisc", 0, "tl.tt.dfb_release", "missing release"),
    ],
)
def test_missing_synchronization_and_endpoint(slot, x, name, match):
    mod = resource_module()
    body = body_statements(mod, x, slot)
    body.pop(next(i for i, stmt in enumerate(body) if operation(stmt) == name))
    replace_body(mod, x, slot, body)
    rejected(mod, match)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("transaction_count", 2, "exactly one transaction"),
        ("occurrence", 1, "one occurrence"),
        ("record_index", 9, "missing original record"),
        ("src_coord", CoreCoord(1, 0), "source Core disagrees"),
        ("dst_coord", CoreCoord(0, 0), "outside original record domain"),
        ("source_dfb_id", 9999, "missing source/destination DFB"),
    ],
)
def test_invalid_transfer_contract(field, value, match):
    mod = resource_module()
    transfers = list(mod.attrs["tt.pipe_transfer_table"])
    transfers[0] = replace_transfer(transfers[0], **{field: value})
    rejected(mod.with_attr("tt.pipe_transfer_table", transfers), match)


def test_missing_multicast_receiver():
    mod = resource_module(collective=True)
    rejected(mod.with_attr("tt.pipe_transfer_table", list(mod.attrs["tt.pipe_transfer_table"])[:-1]), "missing receiver")


def test_duplicate_multicast_destination():
    mod = resource_module(collective=True)
    transfers = list(mod.attrs["tt.pipe_transfer_table"])
    transfers[1] = replace_transfer(transfers[1], dst_coord=transfers[0].dst_coord)
    rejected(mod.with_attr("tt.pipe_transfer_table", transfers), "duplicate destination transaction")


@pytest.mark.parametrize("name", ["tl.tt.dfb_pipe_send", "tl.tt.dfb_pipe_wait", "tl.tt.dfb_reserve"])
def test_duplicate_transaction_operations(name):
    mod = resource_module()
    x, slot = (0, "ncrisc") if name.endswith("reserve") else (0, "brisc")
    body = body_statements(mod, x, slot)
    index = next(i for i, stmt in enumerate(body) if operation(stmt) == name)
    body.insert(index, body[index])
    replace_body(mod, x, slot, body)
    rejected(mod, "more than once|duplicated")


def test_wrong_pipe_operation_transaction_count():
    mod = resource_module()
    body = body_statements(mod, 0, "brisc")
    index = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_pipe_send")
    args = list(body[index].value.args)
    args[-1] = tirx.IntImm("int32", 2)
    body[index] = replace_call(body[index], args)
    replace_body(mod, 0, "brisc", body)
    rejected(mod, "exactly one transaction")


def test_source_release_before_transport_completion():
    mod = resource_module()
    body = body_statements(mod, 0, "brisc")
    release = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_release")
    wait = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_pipe_wait")
    statement = body.pop(release)
    body.insert(wait, statement)
    replace_body(mod, 0, "brisc", body)
    rejected(mod, "release before send completion")


def test_wait_after_release():
    mod = resource_module()
    body = body_statements(mod, 0, "brisc")
    wait = next(stmt for stmt in body if operation(stmt) == "tl.tt.dfb_wait")
    body.append(wait)
    replace_body(mod, 0, "brisc", body)
    rejected(mod, "wait after release")


def test_cross_slot_receive_cycle():
    mod = resource_module()
    transfer = mod.attrs["tt.pipe_transfer_table"][0]
    destination = int(transfer.destination_dfb_id)
    dfb = next(d for d in mod.attrs["tt.dfb_table"] if int(d.dfb_id) == destination)
    assert str(dfb.consumer_slot) == "trisc"
    body = body_statements(mod, 1, "ncrisc")
    # Export waits on a TRISC snapshot which first waits on this NCRISC receive.
    start = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_wait")
    body = body[start:] + body[:start]
    replace_body(mod, 1, "ncrisc", body)
    rejected(mod, "dependency cycle")


def test_dfbs_require_single_core_owner_and_transaction():
    mod = resource_module()
    dfbs = list(mod.attrs["tt.dfb_table"])
    dfbs[0] = _replace_dfb(dfbs[0], transaction_count_or_loop_relation=2)
    rejected(mod.with_attr("tt.dfb_table", dfbs), "exactly one transaction")
    dfbs[0] = _replace_dfb(dfbs[0], transaction_count_or_loop_relation=1, consumer_domain=CoreDomain(CoreCoord(0, 0), CoreCoord(2, 1)))
    rejected(mod.with_attr("tt.dfb_table", dfbs), "exactly one Core")


def test_capacity_boundary_per_core():
    mod = resource_module(collective=True)
    payload = {}
    for dfb in mod.attrs["tt.dfb_table"]:
        key = (int(dfb.producer_domain.begin.x), int(dfb.producer_domain.begin.y))
        size = int(dfb.block_count) * tvm.DataType(str(dfb.element_dtype)).bits // 8
        for extent in [*dfb.tile_shape, *dfb.block_shape_in_tiles]:
            size *= int(extent)
        payload[key] = payload.get(key, 0) + size
    peak = max(payload.values())
    assert sum(payload.values()) > peak
    transform.VerifyTenstorrentDeviceIR()(mod.with_attr("tt.l1_capacity_bytes", peak))
    rejected(mod.with_attr("tt.l1_capacity_bytes", peak - 1), "L1 logical payload lower bound.*exceeds")
    dfbs = list(mod.attrs["tt.dfb_table"])
    dfbs[0] = _replace_dfb(dfbs[0], block_count=0)
    rejected(mod.with_attr("tt.dfb_table", dfbs), "block_count must be positive|capacity")


def test_cross_core_output_overlap():
    mod = resource_module(collective=True)
    body = body_statements(mod, 2, "ncrisc")
    index = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_to_tensor_nd")
    args = list(body[index].value.args)
    args[2] = tirx.IntImm("int32", 32)
    body[index] = replace_call(body[index], args)
    replace_body(mod, 2, "ncrisc", body)
    rejected(mod, "output overlap.*overwrite")


def test_kernel_order_and_core_slot_identity():
    mod = resource_module()
    order = list(mod.attrs["tt.kernel_order"])
    order[0], order[1] = order[1], order[0]
    rejected(mod.with_attr("tt.kernel_order", order), "kernel_order")
    global_var, func = slot_function(mod, 1, "brisc")
    _, original = slot_function(mod, 0, "brisc")
    mod[global_var] = func.with_attr("tt.core_domain", original.attrs["tt.core_domain"])
    rejected(mod, "duplicate Core/slot")


def test_topology_pipeline_requires_complete_storage_metadata():
    mod = resource_module()
    rejected(mod.with_attr("tt.dfb_storage_groups", {}), "missing required Module attr.*pipeline_stages")


def test_original_record_order_is_enforced():
    mod = resource_module(duplicate=True)
    body = body_statements(mod, 0, "brisc")
    chunks, current = [], []
    for statement in body:
        current.append(statement)
        if operation(statement) == "tl.tt.dfb_release":
            chunks.append(current)
            current = []
    assert len(chunks) == 2 and not current
    replace_body(mod, 0, "brisc", chunks[1] + chunks[0])
    rejected(mod, "foreach record order")


def test_destination_generation_cannot_have_two_producers():
    mod = resource_module(duplicate=True)
    transfers = list(mod.attrs["tt.pipe_transfer_table"])
    transfers[1] = replace_transfer(transfers[1], destination_dfb_id=transfers[0].destination_dfb_id)
    rejected(mod.with_attr("tt.pipe_transfer_table", transfers), "multiple producers.*overwrite")


def test_transfer_metadata_is_not_silently_accepted_by_legacy_schema():
    mod = resource_module()
    rejected(mod.with_attr("tt.device_ir_version", 3), "requires Device IR schema v4")


def test_received_generation_may_be_explicitly_drained():
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), (1, 0))])

    @T.prim_func
    def discarded():
        with T.Kernel(2, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32)})
            for pipe in T.comm.foreach_src(net):
                T.fill(a, 3)
                T.copy(a, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, a)

    target = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": discarded}), target)
    transfer = mod.attrs["tt.pipe_transfer_table"][0]
    body = body_statements(mod, 1, "ncrisc")
    assert [operation(s) for s in body] == [
        "tl.tt.dfb_reserve",
        "tl.tt.dfb_pipe_recv",
        "tl.tt.dfb_pipe_wait",
        "tl.tt.dfb_wait",
        "tl.tt.dfb_release",
    ]
    assert int(body[-1].value.args[0]) == int(transfer.destination_dfb_id)
    transform.VerifyTenstorrentDeviceIR()(mod)
    body.pop(-2)
    replace_body(mod, 1, "ncrisc", body)
    rejected(mod, "release must follow wait")


def test_two_core_rendezvous_cycle_has_no_local_cycle():
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), (1, 0)), T.comm.Pipe((1, 0), (0, 0))])

    @T.prim_func
    def bidirectional(A: T.Tensor((64, 32), "float32"), C: T.Tensor((64, 32), "float32")):
        with T.Kernel(2, 1, threads=1) as (x, y):
            source = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32)})
            destination = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32)})
            T.copy(A[x * 32 : (x + 1) * 32, :], source)
            for pipe in T.comm.foreach_src(net):
                T.copy(source, pipe)
            for pipe in T.comm.foreach_dst(net):
                T.copy(pipe, destination)
                T.copy(destination, C[x * 32 : (x + 1) * 32, :])

    target = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": bidirectional}), target)
    for core in (0, 1):
        body = body_statements(mod, core, "ncrisc")
        receive = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_pipe_recv")
        assert operation(body[receive - 1]) == "tl.tt.dfb_reserve"
        assert operation(body[receive + 1]) == "tl.tt.dfb_pipe_wait"
        block = body[receive - 1 : receive + 2]
        body = block + body[: receive - 1] + body[receive + 2 :]
        assert operation(body[2]) == "tl.tt.dfb_pipe_wait"
        assert any(operation(stmt) == "tl.tt.tensor_to_dfb_nd" for stmt in body[3:])
        replace_body(mod, core, "ncrisc", body)
    # Each Core can reserve/receive without waiting for one of its own values.
    # Only the two remote send-completion edges close the dependency cycle.
    rejected(mod, "dependency cycle.*cross-Core/slot deadlock")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("consumer_slot", "trisc", "BRISC affinity"),
        ("element_dtype", "bfloat16", "backing dtype mismatch|payload dtype/shape mismatch"),
    ],
)
def test_source_resource_metadata_cannot_evade_transfer_contract(field, value, match):
    mod = resource_module()
    source = int(mod.attrs["tt.pipe_transfer_table"][0].source_dfb_id)
    dfbs = [_replace_dfb(dfb, **{field: value}) if int(dfb.dfb_id) == source else dfb for dfb in mod.attrs["tt.dfb_table"]]
    rejected(mod.with_attr("tt.dfb_table", dfbs), match)


def test_per_core_payload_arithmetic_does_not_wrap():
    mod = resource_module()
    dfbs = list(mod.attrs["tt.dfb_table"])
    dfbs[0] = _replace_dfb(dfbs[0], block_shape_in_tiles=[tirx.IntImm("int64", 1 << 62), 1])
    rejected(mod.with_attr("tt.dfb_table", dfbs), "overflows static byte/capacity arithmetic")


def test_received_logical_shape_is_checked_even_when_tile_grid_matches():
    mod = resource_module()
    body = body_statements(mod, 1, "trisc")
    index = next(i for i, stmt in enumerate(body) if operation(stmt) == "tl.tt.dfb_compute")
    call = body[index].value
    annotations = dict(call.annotations.items())
    annotations["tt.input_shapes"] = tvm.runtime.convert([[tirx.IntImm("int32", 1), tirx.IntImm("int32", 32)]])
    body[index] = tirx.Evaluate(tirx.Call(call.dtype, call.op, call.args, annotations=annotations, span=call.span))
    replace_body(mod, 1, "trisc", body)
    rejected(mod, "input logical shape disagrees with its producer")

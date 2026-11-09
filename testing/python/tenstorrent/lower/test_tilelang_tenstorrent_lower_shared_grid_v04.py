"""Lower regressions for the shared-grid v04 Device IR contract."""

import pytest
from tilelang import tvm
from tilelang.tenstorrent import codegen, device_ir as D
from tilelang.tenstorrent import language as T, lower_tenstorrent_ir, transform
from tilelang.tenstorrent.ttl_codegen.records import RecordResolutionError, resolve_transport_records
from tvm import arith, ir, tirx


TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})


def lower(program):
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": program}), TARGET)
    assert str(mod.attrs["tt.ir_stage"]) == "device"
    assert "tt.device_ir_version" not in mod.attrs
    assert len(mod.functions) == 3
    ir.assert_structural_equal(mod, ir.load_json(ir.save_json(mod)))
    assert transform.VerifyTenstorrentDeviceIR()(mod).same_as(mod)
    return mod


def calls(body):
    found = []

    def visit(node):
        if isinstance(node, tirx.Call) and hasattr(node.op, "name"):
            found.append(node)

    tirx.stmt_functor.post_order_visit(body, visit)
    return found


def expanded(mod):
    return D.expand_current_device_tables(mod)


@T.prim_func
def point_transfer(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
    with T.Kernel(2, 1, threads=1) as (x, _y):
        src = T.alloc_shared((32, 32), "float32")
        dst = T.alloc_shared((32, 32), "float32")
        if x == 0:
            T.copy(A, src)
        T.comm.transfer(src=src, dst=dst, src_core=(0, 0), dst_cores=(1, 0))
        if x == 1:
            T.copy(dst, C)


@T.prim_func
def local_collective_then_transfer(A: T.Tensor((64, 32), "float32"), C: T.Tensor((64, 32), "float32")):
    with T.Kernel(2, 1, threads=1) as (x, _y):
        local_src = T.alloc_shared((32, 32), "float32")
        local_dst = T.alloc_shared((32, 32), "float32")
        remote_src = T.alloc_shared((32, 32), "float32")
        remote_dst = T.alloc_shared((32, 32), "float32")
        if x == 0:
            T.copy(A[0:32, 0:32], local_src)
        T.comm.all_gather(
            src=local_src,
            dst=local_dst,
            group=T.comm.CoreRange((0, 0), (1, 1)),
            axis=0,
        )
        if x == 0:
            T.copy(local_dst, C[0:32, 0:32])
            T.copy(A[32:64, 0:32], remote_src)
        T.comm.transfer(src=remote_src, dst=remote_dst, src_core=(0, 0), dst_cores=(1, 0))
        if x == 1:
            T.copy(remote_dst, C[32:64, 0:32])


@T.prim_func
def all_gather(A: T.Tensor((64, 32), "float32"), C: T.Tensor((128, 32), "float32")):
    with T.Kernel(2, 1, threads=1) as (x, _y):
        src = T.alloc_shared((32, 32), "float32")
        dst = T.alloc_shared((64, 32), "float32")
        T.copy(A[x * 32 : (x + 1) * 32, 0:32], src)
        T.comm.all_gather(
            src=src,
            dst=dst,
            group=T.comm.CoreRange((0, 0), (2, 1)),
            axis=0,
        )
        T.copy(dst, C[x * 64 : (x + 1) * 64, 0:32])


@T.prim_func
def all_reduce(A: T.Tensor((64, 32), "float32"), C: T.Tensor((64, 32), "float32")):
    with T.Kernel(2, 1, threads=1) as (x, _y):
        src = T.alloc_shared((32, 32), "float32")
        dst = T.alloc_shared((32, 32), "float32")
        T.copy(A[x * 32 : (x + 1) * 32, 0:32], src)
        T.comm.all_reduce(
            src=src,
            dst=dst,
            group=T.comm.CoreRange((0, 0), (2, 1)),
            op="sum",
        )
        T.copy(dst, C[x * 32 : (x + 1) * 32, 0:32])


@pytest.mark.parametrize(
    "program,kind,algorithm",
    [
        (point_transfer, "transfer", "direct_p2p"),
        (all_gather, "all_gather", "direct_all_to_all"),
        (all_reduce, "all_reduce", "root0_left_fold_broadcast"),
    ],
)
def test_high_level_communication_forms_complete_device_program(program, kind, algorithm):
    mod = lower(program)
    assert [str(symbol) for symbol in mod.attrs["tt.kernel_order"]] == [
        "main_trisc",
        "main_ncrisc",
        "main_brisc",
    ]
    communication = list(mod.attrs["tt.communication_table"])
    assert len(communication) == 1
    assert str(communication[0].kind) == kind
    assert str(communication[0].algorithm) == algorithm
    assert "tt.communication_step_families" not in mod.attrs
    assert resolve_transport_records(mod) or kind in ("all_gather", "all_reduce")
    body_names = {call.op.name for function in mod.functions.values() for call in calls(function.body)}
    assert not {"tl.tt.transfer", "tl.tt.all_gather", "tl.tt.all_reduce"} & body_names


@T.prim_func
def mixed_access_identity(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = T.alloc_shared(
            (32, 32),
            "float32",
            annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 1},
        )
        out = T.alloc_shared(
            (32, 32),
            "float32",
            annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 1},
        )
        T.copy(A, a)
        for i, j in T.Tiles(out):
            out[i, j] = a[i, j] + a[0, j]
        T.copy(out, C)


def test_same_buffer_identity_and_broadcast_accesses_are_rejected():
    with pytest.raises(ValueError, match="different logical regions"):
        lower(mixed_access_identity)


def test_current_device_tirx_parses_and_reverifies():
    from tvm.script import from_source

    mod = lower(make_loop_carried_fragment_update(2))
    derived = {
        "tt.compute_dtype",
        "tt.compute_tile_shape",
        "tt.logical_domain",
    }
    value_calls = [call for function in mod.functions.values() for call in calls(function.body) if call.op.name == "tl.tt.compute_value"]
    assert value_calls
    assert all(derived.isdisjoint(call.annotations) for call in value_calls)
    optional_arrays = {"tt.input_shapes", "tt.value_inputs", "tt.value_input_shapes"}
    assert all(all(key not in call.annotations or len(call.annotations[key]) for key in optional_arrays) for call in value_calls)
    source = D.device_ir_to_tirx(mod)
    assert "tt__compute_kind" in source
    assert "tt.compute_kind=" not in source
    assert 'T.tt.dfb_load("float32",' in source
    restored = from_source(source)
    assert transform.VerifyTenstorrentDeviceIR()(restored).same_as(restored)


@T.prim_func
def single_slot_in_place(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
    with T.Kernel(1, 1, threads=1):
        value = T.alloc_shared(
            (32, 32),
            "float32",
            annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 1},
        )
        T.copy(A, value)
        for i, j in T.Tiles(value):
            value[i, j] = value[i, j] + T.float32(1)
        T.copy(value, C)


@T.prim_func
def straight_line_in_place_reuse(
    A: T.Tensor((32, 32), "float32"),
    C: T.Tensor((32, 32), "float32"),
):
    with T.Kernel(1, 1, threads=1):
        value = T.alloc_shared(
            (32, 32),
            "float32",
            annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 1},
        )
        T.copy(A, value)
        for i, j in T.Tiles(value):
            value[i, j] = value[i, j] + T.float32(1)
        for i, j in T.Tiles(value):
            value[i, j] = value[i, j] + T.float32(1)
        for i, j in T.Tiles(value):
            value[i, j] = value[i, j] + T.float32(1)
        for i, j in T.Tiles(value):
            value[i, j] = value[i, j] + T.float32(1)
        T.copy(value, C)


def make_stateful_serial_update(iterations):
    @T.prim_func
    def stateful_serial_update(
        A: T.Tensor((32, 32), "float32"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1, 1, threads=1):
            value = T.alloc_shared(
                (32, 32),
                "float32",
                annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 1},
            )
            T.copy(A, value)
            for _ in T.serial(iterations):
                for i, j in T.Tiles(value):
                    value[i, j] = value[i, j] + T.float32(1)
            T.copy(value, C)

    return stateful_serial_update


def make_loop_carried_fragment_update(iterations):
    @T.prim_func
    def loop_carried_fragment_update(
        A: T.Tensor((iterations * 32, 32), "float32"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1, 1, threads=1):
            panel = T.alloc_shared(
                (32, 32),
                "float32",
                annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 1},
            )
            state = T.alloc_fragment((32, 32), "float32")
            T.clear(state)
            for step in T.serial(iterations):
                T.copy(A[step * 32 : step * 32 + 32, 0:32], panel)
                for i, j in T.Tiles(state):
                    state[i, j] = state[i, j] + panel[i, j]
            T.copy(state, C)

    return loop_carried_fragment_update


def test_stateful_serial_loop_uses_extent_invariant_device_resources_and_body():
    modules = [lower(make_stateful_serial_update(iterations)) for iterations in (8, 16, 32)]
    assert [len(mod.attrs["tt.dfb_storage_groups"]) for mod in modules] == [4, 4, 4]
    assert all("tt.compute_value_families" not in mod.attrs for mod in modules)

    trisc_functions = [
        next(function for function in mod.functions.values() if str(function.attrs["tt.kernel_slot"]) == "trisc") for mod in modules
    ]
    assert all(_compact_loop_names(mod) for mod in modules)
    assert all(not _dynamic_dfb_operands(mod) for mod in modules)
    assert len({_body_counts(function)["evaluate"] for function in trisc_functions}) == 1
    script_sizes = [len(function.script()) for function in trisc_functions]
    assert max(script_sizes) - min(script_sizes) <= 1


def test_loop_carried_fragment_snapshots_are_extent_invariant():
    modules = [lower(make_loop_carried_fragment_update(iterations)) for iterations in (8, 16, 32)]
    group_counts = [len(mod.attrs["tt.dfb_storage_groups"]) for mod in modules]
    assert len(set(group_counts)) == 1
    assert all(not _dynamic_dfb_operands(mod) for mod in modules)

    trisc_functions = [
        next(function for function in mod.functions.values() if str(function.attrs["tt.kernel_slot"]) == "trisc") for mod in modules
    ]
    expected_state_ops = {
        "tl.tt.compute_value_state_init",
        "tl.tt.compute_value_state_input",
        "tl.tt.compute_value_state_yield",
        "tl.tt.compute_value_state_result",
    }
    assert all(expected_state_ops <= {call.op.name for call in calls(function.body)} for function in trisc_functions)
    assert len({_body_counts(function)["evaluate"] for function in trisc_functions}) == 1
    script_sizes = [len(function.script()) for function in trisc_functions]
    assert max(script_sizes) - min(script_sizes) <= 1


def test_loop_state_yield_requires_a_matching_initialized_state():
    mod = ir.load_json(ir.save_json(lower(make_loop_carried_fragment_update(2))))
    changed = False

    def corrupt(call):
        nonlocal changed
        if changed or call.op.name != "tl.tt.compute_value_state_yield":
            return None
        changed = True
        args = list(call.args)
        args[0] = tirx.IntImm("int32", 99)
        return tirx.Call(call.dtype, call.op, args, call.annotations, call.span)

    for global_var, function in list(mod.functions.items()):
        body = tirx.stmt_functor.ir_transform(function.body, None, corrupt, ["tirx.Call"])
        mod.update_func(global_var, function.with_body(body))
    assert changed
    with pytest.raises(ValueError, match="loop state used before initialization"):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_single_slot_in_place_update_uses_logical_streams():
    mod = lower(single_slot_in_place)
    dfbs = list(expanded(mod).attrs["tt.dfb_table"])
    assert len(dfbs) == 3
    assert all(int(dfb.block_count) == 1 for dfb in dfbs)
    assert [str(dfb.source_buffer_identity) for dfb in dfbs] == [
        "stream.0",
        "stream.1",
        "stream.2",
    ]
    assert len({int(dfb.dfb_id) for dfb in dfbs}) == 3


def test_nonpipeline_lifetime_plan_uses_two_groups_for_straight_line_in_place_updates():
    mod = lower(straight_line_in_place_reuse)
    streams = [
        stream
        for stream in mod.attrs["tt.dfb_storage_groups"]
        if str(stream.producer_slot) == "trisc" and str(stream.consumer_slot) == "trisc"
    ]
    assert len(streams) == 2
    assert all(int(stream.block_count) == 1 for stream in streams)


def test_explicit_l1_budget_checks_current_device_storage_lower_bound():
    mod = lower(point_transfer).with_attr("tt.l1_capacity_bytes", 1)
    with pytest.raises(ValueError, match="L1 logical payload lower bound"):
        transform.VerifyTenstorrentDeviceIR()(mod)


@T.prim_func
def producer_forwarded_broadcast(
    A: T.Tensor((32, 32), "bfloat16"),
    C: T.Tensor((64, 32), "bfloat16"),
):
    with T.Kernel(2, 1, threads=1) as (x, _y):
        panel = T.alloc_shared(
            (32, 32),
            "bfloat16",
            annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 1},
        )
        accumulator = T.alloc_fragment((32, 32), "bfloat16")
        if x == 0:
            T.copy(A, panel)
        T.comm.broadcast(
            src=panel,
            dst=panel,
            src_core=(0, 0),
            dst_cores=T.comm.CoreRange((1, 0), (2, 1)),
        )
        T.clear(accumulator)
        T.gemm(panel, panel, accumulator, clear_accum=False)
        T.copy(accumulator, C[x * 32 : (x + 1) * 32, 0:32])


def test_tensor_panel_producer_forwarding_is_a_closed_ncrisc_transaction():
    mod = lower(producer_forwarded_broadcast)
    sends = [
        (str(function.attrs["tt.kernel_slot"]), call.op.name)
        for function in mod.functions.values()
        for call in calls(function.body)
        if call.op.name == "tl.tt.dfb_pipe_send_record"
    ]
    assert sends == [("ncrisc", "tl.tt.dfb_pipe_send_record")]
    role_blocks = []
    for function in mod.functions.values():
        tirx.stmt_functor.post_order_visit(
            function.body,
            lambda node, role_blocks=role_blocks: (
                role_blocks.append(node) if isinstance(node, tirx.SBlock) and "tt.communication_role" in node.annotations else None
            ),
        )
    assert {(str(block.annotations["tt.communication_role"]), int(block.annotations["tt.work_group"])) for block in role_blocks} == {
        ("source", 0),
        ("destination", 0),
    }


def test_collective_transport_families_resolve_to_ordered_pipenet_records():
    from examples.tenstorrent.summa import make_summa

    mod = lower(make_summa(core_m=4, core_n=4, k_stages=2, block_m=32, block_n=32, block_k=32))
    records = resolve_transport_records(mod)
    assert set(records) == {0, 1}
    assert len(records[0]) == len(records[1]) == 4
    for row, record in enumerate(records[0]):
        assert record.pipe_net_id == 0
        assert record.communication_id == 0
        assert record.record_index == row
        assert record.source == (0, row)
        assert record.destination_begin == (1, row)
        assert record.destination_end == (4, row + 1)
        assert record.contract == "collective"
    for column, record in enumerate(records[1]):
        assert record.pipe_net_id == 1
        assert record.communication_id == 1
        assert record.record_index == column
        assert record.source == (column, 0)
        assert record.destination_begin == (column, 1)
        assert record.destination_end == (column + 1, 4)
        assert record.contract == "collective"


def test_pipe_net_namespace_is_independent_from_parent_communication():
    mod = lower(local_collective_then_transfer)
    communications = list(mod.attrs["tt.communication_table"])
    assert [int(entry.communication_id) for entry in communications] == [0, 1]
    records = resolve_transport_records(mod)
    assert set(records) == {0}
    (record,) = records[0]
    assert record.pipe_net_id == 0
    assert record.communication_id == 1
    assert record.record_index == 0
    assert "tt.communication_step_families" not in mod.attrs


def test_codegen_record_resolver_rejects_sparse_record_indices():
    mod = lower(point_transfer)
    families = list(mod.attrs["tt.transport_families"])
    family = families[0]
    fields = dict(family.fields)
    fields["record_index"] = D.DeviceIRIntColumn(1, 0, 1, [])
    families[0] = D.DeviceIRDescriptorFamily(family.prototype, family.positions, fields, family.source_parts)
    malformed = mod.with_attr("tt.transport_families", families)
    with pytest.raises(RecordResolutionError, match="record indices are not dense"):
        resolve_transport_records(malformed)


def test_device_verifier_rejects_transport_with_wrong_parent_communication():
    mod = lower(point_transfer)
    families = list(mod.attrs["tt.transport_families"])
    family = families[0]
    fields = dict(family.fields)
    fields["communication_id"] = D.DeviceIRIntColumn(1, 0, 1, [])
    families[0] = D.DeviceIRDescriptorFamily(family.prototype, family.positions, fields, family.source_parts)
    malformed = mod.with_attr("tt.transport_families", families)
    with pytest.raises(ValueError, match="missing parent communication"):
        transform.VerifyTenstorrentDeviceIR()(malformed)


def _body_counts(function):
    counts = {"if": 0, "for": 0, "block": 0, "evaluate": 0}

    def visit(node):
        if isinstance(node, tirx.IfThenElse):
            counts["if"] += 1
        elif isinstance(node, tirx.For):
            counts["for"] += 1
        elif isinstance(node, tirx.SBlock):
            counts["block"] += 1
        elif isinstance(node, tirx.Evaluate):
            counts["evaluate"] += 1

    tirx.stmt_functor.post_order_visit(function.body, visit)
    return counts


def _compact_loop_names(mod):
    names = []
    for function in mod.functions.values():
        tirx.stmt_functor.post_order_visit(
            function.body,
            lambda node: (
                names.append(str(node.loop_var.name))
                if isinstance(node, tirx.For) and str(node.loop_var.name).startswith("tt_compact_")
                else None
            ),
        )
    return names


def _dynamic_dfb_operands(mod):
    found = []
    analyzer = arith.Analyzer()
    positions = {
        "tl.tt.dfb_reserve": (0,),
        "tl.tt.dfb_wait": (0,),
        "tl.tt.dfb_copy_wait": (0,),
        "tl.tt.dfb_release": (0,),
        "tl.tt.dfb_pipe_send": (1,),
        "tl.tt.dfb_pipe_recv": (1,),
        "tl.tt.dfb_pipe_wait": (1,),
        "tl.tt.dfb_pipe_send_record": (2,),
        "tl.tt.dfb_pipe_recv_record": (2,),
        "tl.tt.dfb_pipe_wait_record": (2,),
        "tl.tt.tensor_to_dfb_nd": (1,),
        "tl.tt.dfb_to_tensor_nd": (0,),
        "tl.tt.gemm_update": (0, 1),
        "tl.tt.accumulator_materialize": (1,),
        "tl.tt.compute_value_store": (1,),
        "tl.tt.compute_value_gemm": (1, 2),
    }

    def visit(node):
        if not isinstance(node, tirx.Call) or not hasattr(node.op, "name"):
            return
        operands = positions.get(node.op.name, ())
        if node.op.name == "tl.tt.dfb_compute":
            operands = range(len(node.args))
        elif node.op.name == "tl.tt.compute_value":
            operands = range(1, len(node.args))
        found.extend(
            (node.op.name, node.args[index]) for index in operands if not isinstance(analyzer.simplify(node.args[index]), tirx.IntImm)
        )

    for function in mod.functions.values():
        tirx.stmt_functor.post_order_visit(function.body, visit)
    return found


def _column_values(column):
    values = list(map(int, column.values))
    period = int(column.period)
    return [
        values[row % period] + int(column.step) * (row // period)
        if period
        else values[row]
        if values
        else int(column.start) + int(column.step) * row
        for row in range(int(column.count))
    ]


def _assert_logical_streams(mod):
    streams = list(mod.attrs["tt.dfb_storage_groups"])
    assert [int(stream.storage_group_id) for stream in streams] == list(range(len(streams)))
    assert "tt.dfb_families" not in mod.attrs
    assert "tt.dfb_storage_group_relations" not in mod.attrs


def test_summa_uses_parameterized_core_occurrence_and_resource_families():
    from examples.tenstorrent.summa import make_summa

    configs = ((2, 2), (2, 4), (4, 4))
    modules = []
    for cores, stages in configs:
        modules.append(
            lower(
                make_summa(
                    core_m=cores,
                    core_n=cores,
                    k_stages=stages,
                    block_m=32,
                    block_n=32,
                    block_k=32,
                )
            )
        )

    family_keys = (
        "tt.accumulator_families",
        "tt.transport_families",
    )
    assert all(key in mod.attrs for mod in modules for key in family_keys)
    assert all(
        key not in mod.attrs
        for mod in modules
        for key in (
            "tt.dfb_table",
            "tt.accumulator_table",
            "tt.transport_table",
        )
    )
    assert [tuple(len(mod.attrs[key]) for key in family_keys) for mod in modules] == [
        (1, 1),
        (1, 1),
        (1, 1),
    ]

    for mod in modules:
        assert len(mod.attrs["tt.dfb_storage_groups"]) == 3
        functions = {str(func.attrs["tt.kernel_slot"]): func for func in mod.functions.values()}
        trisc = _body_counts(functions["trisc"])
        ncrisc = _body_counts(functions["ncrisc"])
        assert trisc["if"] == 0
        assert trisc["block"] == 1
        assert trisc["for"] >= 1
        assert "tt.storage_group_statements_specialized" not in mod.attrs
        assert _compact_loop_names(mod)
        assert not _dynamic_dfb_operands(mod)
        _assert_logical_streams(mod)
        assert ncrisc["if"] == 4
        assert ncrisc["block"] == 6
        stream_blocks = []
        role_conditions = []
        tirx.stmt_functor.post_order_visit(
            functions["ncrisc"].body,
            lambda node, stream_blocks=stream_blocks, role_conditions=role_conditions: (
                stream_blocks.append(node)
                if isinstance(node, tirx.SBlock) and node.name_hint == "tt_device_stream_work"
                else role_conditions.append(node.condition)
                if isinstance(node, tirx.IfThenElse)
                else None
            ),
        )
        role_blocks = [block for block in stream_blocks if "tt.communication_role" in block.annotations]
        assert len(role_blocks) == 4
        assert {
            (
                str(block.annotations["tt.communication_role"]),
                tuple(map(int, block.annotations["tt.communication_ids"])),
                tuple(map(int, block.annotations["tt.pipe_net_ids"])),
            )
            for block in role_blocks
        } == {
            ("source", (0,), (0,)),
            ("destination", (0,), (0,)),
            ("source", (1,), (1,)),
            ("destination", (1,), (1,)),
        }
        for condition in role_conditions:
            coordinates = set()
            tirx.stmt_functor.post_order_visit(
                condition,
                lambda node, coordinates=coordinates: (
                    coordinates.add(node.op.name)
                    if isinstance(node, tirx.Call) and node.op.name in ("tl.tt.core_x", "tl.tt.core_y")
                    else None
                ),
            )
            assert len(coordinates) == 1
        source_blocks = [block for block in role_blocks if str(block.annotations["tt.communication_role"]) == "source"]
        for block in source_blocks:
            source_calls = calls(block.body)
            assert sum(call.op.name == "tl.tt.dfb_pipe_send_record" for call in source_calls) == 1
            assert sum(call.op.name == "tl.tt.dfb_pipe_wait_record" for call in source_calls) == 1
            assert not any(call.op.name == "tl.tt.dfb_pipe_send" for call in source_calls)
            assert all(
                len(call.args) == 3 and not call.annotations
                for call in source_calls
                if call.op.name in ("tl.tt.dfb_pipe_send_record", "tl.tt.dfb_pipe_wait_record")
            )
            nested_loops = []
            tirx.stmt_functor.post_order_visit(
                block.body,
                lambda node, nested_loops=nested_loops: nested_loops.append(node) if isinstance(node, tirx.For) else None,
            )
            assert not nested_loops
        assert "tt.communication_step_families" not in mod.attrs
        blocks = []
        tirx.stmt_functor.post_order_visit(
            functions["trisc"].body,
            lambda node, blocks=blocks: blocks.append(node) if isinstance(node, tirx.SBlock) else None,
        )
        assert all("tt.core_domain" in block.annotations for block in blocks)
        assert all("tt.core_x" not in block.annotations for block in blocks)
        assert all("tt.core_y" not in block.annotations for block in blocks)

    assert len(tvm.ir.save_json(modules[1])) < 2 * len(tvm.ir.save_json(modules[0]))
    assert transform.VerifyTenstorrentDeviceIR()(modules[-1]).same_as(modules[-1])


def test_device_verifier_rejects_mismatched_stream_work_role():
    from examples.tenstorrent.summa import make_summa

    mod = lower(make_summa(core_m=2, core_n=2, k_stages=2, block_m=32, block_n=32, block_k=32))
    malformed = ir.load_json(ir.save_json(mod))
    changed = False

    def corrupt(node):
        nonlocal changed
        if changed or not isinstance(node, tirx.SBlock) or node.annotations.get("tt.communication_role") != "source":
            return None
        changed = True
        annotations = dict(node.annotations)
        annotations["tt.communication_role"] = "destination"
        return tirx.SBlock(
            node.iter_vars,
            node.reads,
            node.writes,
            node.name_hint,
            node.body,
            node.init,
            node.alloc_buffers,
            node.match_buffers,
            annotations,
        )

    for global_var, function in list(malformed.functions.items()):
        if str(function.attrs["tt.kernel_slot"]) != "ncrisc":
            continue
        body = tirx.stmt_functor.ir_transform(function.body, None, corrupt, ["tirx.SBlock"])
        malformed.update_func(global_var, function.with_body(body))
    assert changed
    with pytest.raises(ValueError, match="destination communication role contains a send|role annotation disagrees"):
        transform.VerifyTenstorrentDeviceIR()(malformed)


def test_device_verifier_rejects_mismatched_stream_work_pipe_net_summary():
    from examples.tenstorrent.summa import make_summa

    malformed = ir.load_json(ir.save_json(lower(make_summa(core_m=2, core_n=2, k_stages=2, block_m=32, block_n=32, block_k=32))))
    changed = False

    def corrupt(node):
        nonlocal changed
        if changed or not isinstance(node, tirx.SBlock) or node.annotations.get("tt.communication_role") != "source":
            return None
        changed = True
        annotations = dict(node.annotations)
        annotations["tt.pipe_net_ids"] = [99]
        return tirx.SBlock(
            node.iter_vars,
            node.reads,
            node.writes,
            node.name_hint,
            node.body,
            node.init,
            node.alloc_buffers,
            node.match_buffers,
            annotations,
        )

    for global_var, function in list(malformed.functions.items()):
        if str(function.attrs["tt.kernel_slot"]) != "ncrisc":
            continue
        body = tirx.stmt_functor.ir_transform(function.body, None, corrupt, ["tirx.SBlock"])
        malformed.update_func(global_var, function.with_body(body))
    assert changed
    with pytest.raises(ValueError, match="block summaries disagree"):
        transform.VerifyTenstorrentDeviceIR()(malformed)


def test_device_verifier_rejects_missing_pipe_record_identity():
    from examples.tenstorrent.summa import make_summa

    mod = ir.load_json(ir.save_json(lower(make_summa(core_m=2, core_n=2, k_stages=2, block_m=32, block_n=32, block_k=32))))
    changed = False

    def corrupt(call):
        nonlocal changed
        if changed or call.op.name != "tl.tt.dfb_pipe_send_record":
            return None
        changed = True
        args = list(call.args)
        args[1] = tirx.IntImm(args[1].dtype, 999)
        return tirx.Call(call.dtype, call.op, args, call.annotations, call.span)

    for global_var, function in list(mod.functions.items()):
        body = tirx.stmt_functor.ir_transform(function.body, None, corrupt, ["tirx.Call"])
        mod.update_func(global_var, function.with_body(body))
    assert changed
    with pytest.raises(ValueError, match="missing transport"):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_summa_accumulator_identity_is_spmd_local():
    from examples.tenstorrent.summa import make_summa

    modules = [
        lower(
            make_summa(
                core_m=cores,
                core_n=cores,
                k_stages=2,
                block_m=32,
                block_n=32,
                block_k=32,
            )
        )
        for cores in (1, 2, 4)
    ]
    for mod in modules:
        (family,) = mod.attrs["tt.accumulator_families"]
        assert set(_column_values(family.fields["accumulator_id"])) == {0}
        trisc = next(function for function in mod.functions.values() if str(function.attrs["tt.kernel_slot"]) == "trisc")
        accumulator_operands = []
        for call in calls(trisc.body):
            if call.op.name in (
                "tl.tt.accumulator_init",
                "tl.tt.accumulator_materialize",
            ):
                accumulator_operands.append(call.args[0])
            elif call.op.name == "tl.tt.gemm_update":
                accumulator_operands.append(call.args[2])
        assert accumulator_operands
        assert all(isinstance(operand, tirx.IntImm) and int(operand) == 0 for operand in accumulator_operands)
        diagnostic = expanded(mod)
        assert transform.VerifyTenstorrentDeviceIR()(diagnostic).same_as(diagnostic)


def test_duplicate_accumulator_local_id_on_one_core_is_rejected():
    from examples.tenstorrent.summa import make_summa

    mod = lower(
        make_summa(
            core_m=2,
            core_n=2,
            k_stages=2,
            block_m=32,
            block_n=32,
            block_k=32,
        )
    )
    families = list(mod.attrs["tt.accumulator_families"])
    family = families[0]
    fields = dict(family.fields)
    fields["core_begin_x"] = D.DeviceIRIntColumn(0, 0, 4, [0, 0, 1, 1])
    fields["core_begin_y"] = D.DeviceIRIntColumn(0, 0, 4, [0, 0, 0, 1])
    fields["core_end_x"] = D.DeviceIRIntColumn(0, 0, 4, [1, 1, 2, 2])
    fields["core_end_y"] = D.DeviceIRIntColumn(0, 0, 4, [1, 1, 1, 2])
    families[0] = D.DeviceIRDescriptorFamily(family.prototype, family.positions, fields, family.source_parts)
    malformed = mod.with_attr("tt.accumulator_families", families)
    with pytest.raises(ValueError, match="duplicate Core-local accumulator ID"):
        transform.VerifyTenstorrentDeviceIR()(malformed)


def test_current_family_column_length_mismatch_is_rejected():
    mod = lower(point_transfer)
    families = list(mod.attrs["tt.transport_families"])
    family = families[0]
    fields = dict(family.fields)
    name, column = next(iter(fields.items()))
    fields[name] = D.DeviceIRIntColumn(
        int(column.start),
        int(column.step),
        int(column.count) + 1,
        list(column.values),
        int(column.period),
    )
    families[0] = D.DeviceIRDescriptorFamily(family.prototype, family.positions, fields, family.source_parts)
    malformed = mod.with_attr("tt.transport_families", families)
    with pytest.raises(ValueError, match="length differs from family positions"):
        transform.VerifyTenstorrentDeviceIR()(malformed)


def test_device_body_requires_a_declared_logical_stream():
    mod = lower(point_transfer)
    groups = list(mod.attrs["tt.dfb_storage_groups"])
    malformed = mod.with_attr("tt.dfb_storage_groups", groups[:-1])
    with pytest.raises(ValueError, match="missing DFB|missing logical stream"):
        transform.VerifyTenstorrentDeviceIR()(malformed)


def test_tensor_storage_backing_requires_the_complete_cb_capacity():
    mod = lower(point_transfer)
    groups = list(mod.attrs["tt.dfb_storage_groups"])
    group = groups[0]
    groups[0] = D.DFBStorageGroupDescriptor(
        int(group.storage_group_id),
        group.element_dtype,
        group.tile_shape,
        group.block_shape_in_tiles,
        group.block_count,
        str(group.producer_slot),
        group.producer_domain,
        str(group.consumer_slot),
        group.consumer_domain,
        group.transaction_count_or_loop_relation,
        D.TensorStorageBacking(0, 0, 1),
        group.source_span,
    )
    malformed = mod.with_attr("tt.dfb_storage_groups", groups)
    with pytest.raises(ValueError, match="byte size must equal the complete DFB capacity"):
        transform.VerifyTenstorrentDeviceIR()(malformed)


def test_final_record_operation_rejects_step_refs():
    mod = ir.load_json(ir.save_json(lower(point_transfer)))
    changed = False

    def corrupt(call):
        nonlocal changed
        if changed or call.op.name != "tl.tt.dfb_pipe_send_record":
            return None
        changed = True
        annotations = dict(call.annotations)
        annotations["tt_communication_step_refs"] = [tirx.IntImm("int32", 0)]
        return tirx.Call(call.dtype, call.op, call.args, annotations, call.span)

    for global_var, function in list(mod.functions.items()):
        body = tirx.stmt_functor.ir_transform(function.body, None, corrupt, ["tirx.Call"])
        mod.update_func(global_var, function.with_body(body))
    assert changed
    with pytest.raises(ValueError, match="step refs|transient planning metadata"):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_current_codegen_preflight_rejects_before_loading_ttlang(monkeypatch):
    mod = lower(point_transfer)
    imported = False

    def fail_import(*_args, **_kwargs):
        nonlocal imported
        imported = True
        raise AssertionError("TT-Lang must not load before mapping preflight")

    monkeypatch.setattr("importlib.import_module", fail_import)
    with pytest.raises(NotImplementedError, match="TTL mapping preflight failed"):
        codegen.build_ttl_without_compile(mod, TARGET)
    assert not imported


def test_lower_preserves_parameterized_storage_group_selection_for_codegen():
    from examples.tenstorrent.summa import make_summa
    from tilelang.tenstorrent.ttl_codegen.resources import (
        resolve_current_function_groups,
    )

    mod = lower(
        make_summa(
            core_m=2,
            core_n=2,
            k_stages=4,
            block_m=32,
            block_n=32,
            block_k=32,
        )
    )
    assert "tt.storage_group_statements_specialized" not in mod.attrs
    assert _compact_loop_names(mod)
    assert not _dynamic_dfb_operands(mod)
    _assert_logical_streams(mod)
    assert len(mod.attrs["tt.dfb_storage_groups"]) == len(expanded(mod).attrs["tt.dfb_table"])
    resolved = resolve_current_function_groups(mod)
    assert set(resolved) == set(map(str, mod.attrs["tt.kernel_order"]))
    assert resolved["main_trisc"]
    assert resolved["main_ncrisc"]
    assert not resolved["main_brisc"]
    with pytest.raises(NotImplementedError) as error:
        codegen.prepare_ttl_codegen(mod, TARGET)
    assert "DFB stream binding resolution failed" not in str(error.value)


def test_core_predicate_must_match_its_typed_work_domain():
    mod = lower(point_transfer)
    for global_var, function in list(mod.functions.items()):
        if str(function.attrs["tt.kernel_slot"]) != "ncrisc":
            continue
        statements = list(function.body.seq)
        branch = next(stmt for stmt in statements if isinstance(stmt, tirx.IfThenElse))
        index = statements.index(branch)
        statements[index] = tirx.IfThenElse(tirx.IntImm("bool", 1), branch.then_case, branch.else_case)
        mod.update_func(global_var, function.with_body(tirx.SeqStmt(statements)))
        break
    with pytest.raises(ValueError, match="predicate disagrees with tt.core_domain"):
        transform.VerifyTenstorrentDeviceIR()(mod)


def test_flash_attention_core_and_kv_repetition_remain_parameterized():
    from examples.tenstorrent.flash_attn import make_flash_attention

    modules = []
    for seq_q, seq_kv, cores in ((64, 64, 2), (64, 96, 2), (128, 96, 4)):
        modules.append(
            lower(
                make_flash_attention(
                    batch=1,
                    heads=1,
                    seq_q=seq_q,
                    seq_kv=seq_kv,
                    head_dim=32,
                    core_q=cores,
                    core_bh=1,
                    input_dtype="float32",
                    accum_dtype="float32",
                    output_dtype="float32",
                )
            )
        )

    family_keys = (
        "tt.accumulator_families",
        "tt.compute_value_families",
        "tt.transport_families",
    )
    assert [tuple(len(mod.attrs[key]) for key in family_keys) for mod in modules] == [
        (2, 8, 1),
        (2, 8, 1),
        (2, 8, 1),
    ]
    assert [len(mod.attrs["tt.dfb_storage_groups"]) for mod in modules] == [14, 14, 14]
    for mod in modules:
        assert "tt.storage_group_statements_specialized" not in mod.attrs
        assert not _dynamic_dfb_operands(mod)
        _assert_logical_streams(mod)
        trisc = next(function for function in mod.functions.values() if str(function.attrs["tt.kernel_slot"]) == "trisc")
        counts = _body_counts(trisc)
        assert counts["if"] == 0
        assert counts["block"] == 1
        assert counts["for"] >= 1

    sizes = [len(tvm.ir.save_json(mod)) for mod in modules]
    assert sizes[1] < 3 * sizes[0] // 2
    assert sizes[2] < 3 * sizes[1] // 2


def test_stateful_loop_compaction_absorbs_the_terminal_steady_state_period():
    from examples.tenstorrent.flash_attn import make_flash_attention

    mod = lower(
        make_flash_attention(
            batch=1,
            heads=1,
            seq_q=32,
            seq_kv=256,
            head_dim=64,
            core_q=1,
            core_bh=1,
            input_dtype="float32",
            accum_dtype="float32",
            output_dtype="float32",
        )
    )
    trisc = next(function for function in mod.functions.values() if str(function.attrs["tt.kernel_slot"]) == "trisc")
    compact_loops = []
    tirx.stmt_functor.post_order_visit(
        trisc.body,
        lambda node: (
            compact_loops.append(node)
            if isinstance(node, tirx.For) and str(node.loop_var.name).startswith("tt_compact_iteration")
            else None
        ),
    )
    assert [int(loop.extent) for loop in compact_loops] == [7]
    assert sum(call.op.name == "tl.tt.compute_value_gemm" for call in calls(trisc.body)) == 4
    assert sum("state_" in call.op.name for call in calls(trisc.body)) == 6
    assert not _dynamic_dfb_operands(mod)


def test_mixed_precision_flash_uses_full_fp32_accumulation():
    from examples.tenstorrent.flash_attn import make_flash_attention

    frontend = make_flash_attention(
        batch=1,
        heads=1,
        seq_q=32,
        seq_kv=64,
        head_dim=32,
        core_q=1,
        core_bh=1,
    )
    mod = lower(frontend)
    concrete = expanded(mod)
    assert {
        (str(entry.input_dtype), str(entry.accumulation_dtype), str(entry.output_dtype)) for entry in concrete.attrs["tt.accumulator_table"]
    } == {("bfloat16", "float32", "bfloat16")}
    trisc = next(function for function in mod.functions.values() if str(function.attrs["tt.kernel_slot"]) == "trisc")
    requirements = trisc.attrs["tt.compute_requirements"]
    assert str(requirements.destination_width) == "bits32_required"
    assert str(requirements.matmul_full_fp32) == "required"
    assert len(mod.attrs["tt.dfb_storage_groups"]) == 16


def test_all_bf16_flash_uses_bf16_constants_and_destination():
    from examples.tenstorrent.flash_attn import make_flash_attention

    frontend = make_flash_attention(
        batch=1,
        heads=1,
        seq_q=32,
        seq_kv=64,
        head_dim=32,
        core_q=1,
        core_bh=1,
        accum_dtype="bfloat16",
    )
    source = frontend.script()
    assert "T.bfloat16(0.17677669529663687)" in source
    assert "T.bfloat16(1.4426950408889634)" in source
    assert 'T.Cast("float32", scores_fragment' not in source
    mod = lower(frontend)
    concrete = expanded(mod)
    assert {
        (str(entry.input_dtype), str(entry.accumulation_dtype), str(entry.output_dtype)) for entry in concrete.attrs["tt.accumulator_table"]
    } == {("bfloat16", "bfloat16", "bfloat16")}
    trisc = next(function for function in mod.functions.values() if str(function.attrs["tt.kernel_slot"]) == "trisc")
    requirements = trisc.attrs["tt.compute_requirements"]
    assert str(requirements.destination_width) == "bits16_required"
    assert str(requirements.matmul_full_fp32) == "forbidden"
    assert len(mod.attrs["tt.dfb_storage_groups"]) == 14

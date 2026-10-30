"""Lower regressions for the shared-grid v04 Device IR contract."""

import pytest
from tilelang import tvm
from tilelang.tenstorrent import codegen, device_ir as D
from tilelang.tenstorrent import language as T, lower_tenstorrent_ir, transform
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
    steps = list(expanded(mod).attrs["tt.communication_step_table"])
    assert steps
    assert [int(step.step_id) for step in steps] == list(range(len(steps)))
    assert all(
        all(int(dependency) < int(step.step_id) for dependency in step.dependencies)
        for step in steps
    )
    body_names = {
        call.op.name
        for function in mod.functions.values()
        for call in calls(function.body)
    }
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


def test_same_buffer_accesses_keep_distinct_identity_and_broadcast_maps():
    mod = lower(mixed_access_identity)
    compute = next(
        call
        for function in mod.functions.values()
        for call in calls(function.body)
        if call.op.name == "tl.tt.dfb_compute"
    )
    access_ids = [int(value) for value in compute.annotations["tt.access_ids"]]
    assert access_ids == [0, 1]
    assert [list(map(int, mapping)) for mapping in compute.annotations["tt.access_maps"]] == [
        [0, 1],
        [-1, 1],
    ]


def test_current_device_tirx_parses_and_reverifies():
    from tvm.script import from_source

    mod = lower(mixed_access_identity)
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


def test_single_slot_in_place_update_uses_independent_generations():
    mod = lower(single_slot_in_place)
    dfbs = list(expanded(mod).attrs["tt.dfb_table"])
    assert len(dfbs) == 3
    assert all(int(dfb.block_count) == 1 for dfb in dfbs)
    assert all(str(dfb.source_buffer_identity).startswith("buffer.0.generation") for dfb in dfbs)
    assert len({int(dfb.dfb_id) for dfb in dfbs}) == 3


def test_nonpipeline_lifetime_plan_uses_two_groups_for_straight_line_in_place_updates():
    mod = lower(straight_line_in_place_reuse)
    candidates = [
        (family, relation)
        for family, relation in zip(
            mod.attrs["tt.dfb_families"],
            mod.attrs["tt.dfb_storage_group_relations"],
        )
        if int(family.positions.count) >= 4
        and str(family.prototype.producer_slot) == "trisc"
        and str(family.prototype.consumer_slot) == "trisc"
    ]
    assert len(candidates) == 1
    family, relation = candidates[0]
    groups = _column_values(relation.storage_group_ids)
    assert int(family.prototype.block_count) == 1
    assert len(groups) >= 4
    assert len(set(groups)) == 2


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
    steps = list(expanded(mod).attrs["tt.communication_step_table"])
    source_steps = [
        step for step in steps if str(step.kind) in ("send", "completion")
    ]
    assert source_steps
    assert all(str(step.owner_slot) == "ncrisc" for step in source_steps)
    sends = [
        (str(function.attrs["tt.kernel_slot"]), call.op.name)
        for function in mod.functions.values()
        for call in calls(function.body)
        if call.op.name == "tl.tt.dfb_pipe_send"
    ]
    assert sends == [("ncrisc", "tl.tt.dfb_pipe_send")]


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
            lambda node: names.append(str(node.loop_var.name))
            if isinstance(node, tirx.For) and str(node.loop_var.name).startswith("tt_compact_")
            else None,
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
            (node.op.name, node.args[index])
            for index in operands
            if not isinstance(analyzer.simplify(node.args[index]), tirx.IntImm)
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


def _assert_storage_group_relations(mod):
    declared = {
        int(group.storage_group_id) for group in mod.attrs["tt.dfb_storage_groups"]
    }
    ordinals = {}
    for relation in mod.attrs["tt.dfb_storage_group_relations"]:
        groups = _column_values(relation.storage_group_ids)
        transactions = _column_values(relation.transaction_ordinals)
        assert len(groups) == len(transactions)
        for group, ordinal in zip(groups, transactions):
            assert group in declared
            ordinals.setdefault(group, []).append(ordinal)
    assert set(ordinals) == declared
    assert all(sorted(values) == list(range(len(values))) for values in ordinals.values())


def test_summa_uses_parameterized_core_occurrence_and_resource_families():
    from examples.tenstorrent.summa import make_summa

    modules = []
    for cores, stages in ((2, 2), (2, 4), (4, 4)):
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
        "tt.dfb_families",
        "tt.accumulator_families",
        "tt.communication_step_families",
        "tt.transport_families",
    )
    assert all(key in mod.attrs for mod in modules for key in family_keys)
    assert all(
        key not in mod.attrs
        for mod in modules
        for key in (
            "tt.dfb_table",
            "tt.accumulator_table",
            "tt.communication_step_table",
            "tt.transport_table",
        )
    )
    assert [tuple(len(mod.attrs[key]) for key in family_keys) for mod in modules] == [
        (3, 1, 10, 1),
        (3, 1, 10, 1),
        (3, 1, 10, 1),
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
        assert _dynamic_dfb_operands(mod)
        _assert_storage_group_relations(mod)
        assert ncrisc["if"] <= 4
        assert ncrisc["block"] <= 4
        blocks = []
        tirx.stmt_functor.post_order_visit(
            functions["trisc"].body,
            lambda node: blocks.append(node) if isinstance(node, tirx.SBlock) else None,
        )
        assert all("tt.core_domain" in block.annotations for block in blocks)
        assert all("tt.core_x" not in block.annotations for block in blocks)
        assert all("tt.core_y" not in block.annotations for block in blocks)

    assert len(tvm.ir.save_json(modules[1])) < 2 * len(tvm.ir.save_json(modules[0]))
    assert transform.VerifyTenstorrentDeviceIR()(modules[-1]).same_as(modules[-1])


def test_current_family_column_length_mismatch_is_rejected():
    mod = lower(point_transfer)
    families = list(mod.attrs["tt.communication_step_families"])
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
    families[0] = D.DeviceIRDescriptorFamily(
        family.prototype, family.positions, fields, family.source_parts
    )
    malformed = mod.with_attr("tt.communication_step_families", families)
    with pytest.raises(ValueError, match="length differs from family positions"):
        transform.VerifyTenstorrentDeviceIR()(malformed)


def test_storage_group_relation_requires_a_declared_group():
    mod = lower(point_transfer)
    relations = list(mod.attrs["tt.dfb_storage_group_relations"])
    relation = relations[0]
    relations[0] = D.DFBStorageGroupRelation(
        int(relation.dfb_family_id),
        D.DeviceIRIntColumn(999, 0, int(relation.storage_group_ids.count)),
        relation.transaction_ordinals,
    )
    malformed = mod.with_attr("tt.dfb_storage_group_relations", relations)
    with pytest.raises(ValueError, match="invalid ID or ordinal"):
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
        D.TensorStorageBacking(0, 0, 1),
        group.source_span,
    )
    malformed = mod.with_attr("tt.dfb_storage_groups", groups)
    with pytest.raises(ValueError, match="byte size must equal the complete DFB capacity"):
        transform.VerifyTenstorrentDeviceIR()(malformed)


def test_communication_step_kind_must_match_the_referencing_body_op():
    mod = lower(point_transfer)
    families = list(mod.attrs["tt.communication_step_families"])
    family = families[0]
    step = family.prototype
    prototype = D.CommunicationStepDescriptor(
        int(step.step_id),
        int(step.communication_id),
        int(step.occurrence),
        "unknown_step",
        step.source_core,
        step.destination_core,
        int(step.source_dfb_id),
        int(step.destination_dfb_id),
        str(step.owner_slot),
        list(step.dependencies),
        step.parameter,
        step.source_span,
    )
    families[0] = D.DeviceIRDescriptorFamily(
        prototype, family.positions, family.fields, family.source_parts
    )
    malformed = mod.with_attr("tt.communication_step_families", families)
    with pytest.raises(ValueError, match="disagrees with its communication step|unknown kind"):
        transform.VerifyTenstorrentDeviceIR()(malformed)


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
    assert _dynamic_dfb_operands(mod)
    _assert_storage_group_relations(mod)
    assert len(mod.attrs["tt.dfb_storage_groups"]) < len(expanded(mod).attrs["tt.dfb_table"])
    resolved = resolve_current_function_groups(mod)
    assert set(resolved) == set(map(str, mod.attrs["tt.kernel_order"]))
    assert resolved["main_trisc"]
    assert resolved["main_ncrisc"]
    assert not resolved["main_brisc"]
    with pytest.raises(NotImplementedError) as error:
        codegen.prepare_ttl_codegen(mod, TARGET)
    assert "DFB-family binding resolution failed" not in str(error.value)


def test_core_predicate_must_match_its_typed_work_domain():
    mod = lower(point_transfer)
    for global_var, function in list(mod.functions.items()):
        if str(function.attrs["tt.kernel_slot"]) != "ncrisc":
            continue
        statements = list(function.body.seq)
        branch = next(stmt for stmt in statements if isinstance(stmt, tirx.IfThenElse))
        index = statements.index(branch)
        statements[index] = tirx.IfThenElse(
            tirx.IntImm("bool", 1), branch.then_case, branch.else_case
        )
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
        "tt.dfb_families",
        "tt.accumulator_families",
        "tt.compute_value_families",
        "tt.communication_step_families",
        "tt.transport_families",
    )
    assert [tuple(len(mod.attrs[key]) for key in family_keys) for mod in modules] == [
        (14, 2, 8, 10, 1),
        (14, 2, 8, 10, 1),
        (14, 2, 8, 10, 1),
    ]
    for mod in modules:
        assert "tt.storage_group_statements_specialized" not in mod.attrs
        assert _compact_loop_names(mod)
        assert _dynamic_dfb_operands(mod)
        _assert_storage_group_relations(mod)
        assert any(
            int(family.positions.count) > int(family.prototype.block_count)
            and len(set(_column_values(relation.storage_group_ids)))
            == int(family.positions.count)
            for family, relation in zip(
                mod.attrs["tt.dfb_families"],
                mod.attrs["tt.dfb_storage_group_relations"],
            )
        )
        trisc = next(
            function
            for function in mod.functions.values()
            if str(function.attrs["tt.kernel_slot"]) == "trisc"
        )
        counts = _body_counts(trisc)
        assert counts["if"] == 0
        assert counts["block"] == 1
        assert counts["for"] >= 1

    sizes = [len(tvm.ir.save_json(mod)) for mod in modules]
    assert sizes[1] < 3 * sizes[0] // 2
    assert sizes[2] < 3 * sizes[1] // 2


def test_bf16_flash_precision_conflict_returns_atomic_structured_module():
    from testing.python.tenstorrent.legacy.fixtures.tenstorrent_flash_attn_03 import (
        make_flash_attention,
    )

    frontend = make_flash_attention(
        batch=1,
        heads=1,
        seq_q=32,
        seq_kv=64,
        head_dim=32,
        core_q=1,
        core_bh=1,
    )
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": frontend}), TARGET)
    assert str(mod.attrs["tt.ir_stage"]) == "structured"
    assert len(mod.functions) == 1
    assert "tt.device_ir_version" not in mod.attrs
    assert "tt.dfb_table" not in mod.attrs
    assert any(
        "conflicting hard destination width requirements" in str(reason)
        for reason in mod.attrs["tt.deferred_reasons"]
    )

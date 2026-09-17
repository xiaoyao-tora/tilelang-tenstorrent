"""Independent verifier checks for composed v7 lowering contracts."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import lower_tenstorrent_ir, transform
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_compute_value_verifier import check_rejected, rewrite_calls
from testing.python.target.test_tilelang_tenstorrent_frontend02 import TARGET


def multicore_values(mixed_precision=False):
    @T.prim_func
    def main(C: T.Tensor((64, 32), "float32")):
        with T.Kernel(2, 1, threads=1) as (x, y):
            f = T.alloc_fragment((32, 32), "float32")
            h = T.alloc_fragment((32, 32), "bfloat16")
            if mixed_precision and x == 0:
                T.fill(h, 2)
            else:
                T.fill(f, 3)
                for i, j in T.Tiles(32, 32):
                    f[i, j] = f[i, j] + 1
                T.copy(f, C[x * 32 : (x + 1) * 32, :])

    return lower_tenstorrent_ir(tvm.IRModule({"main": main}), TARGET)


def test_value_precision_is_scoped_to_owning_core():
    mod = multicore_values(mixed_precision=True)
    requirements = {
        int(func.attrs["tt.core_domain"].begin.x): str(func.attrs["tt.compute_requirements"].destination_width)
        for func in mod.functions.values()
        if func.attrs["tt.kernel_slot"] == "trisc"
    }
    assert requirements == {0: "unconstrained", 1: "bits32_required"}
    transform.VerifyTenstorrentDeviceIR()(ir.load_json(ir.save_json(mod)))


def test_cross_core_value_operand_is_rejected_without_mutation():
    mod = multicore_values()
    owners = {}
    for func in mod.functions.values():

        def collect(node, func=func):
            if isinstance(node, tirx.Call) and node.op.name == "tl.tt.compute_value":
                owners[int(node.args[0])] = int(func.attrs["tt.core_domain"].begin.x)

        tirx.stmt_functor.post_order_visit(func.body, collect)
    changed = False

    def edit(call):
        nonlocal changed
        if changed or call.op.name != "tl.tt.compute_value" or not len(call.annotations["tt.value_inputs"]):
            return None
        changed = True
        owner = owners[int(call.args[0])]
        remote = next(identifier for identifier, core in owners.items() if core != owner)
        attrs = dict(call.annotations)
        attrs["tt.value_inputs"] = [tirx.IntImm("int64", remote)]
        expression = attrs["tt.expression"]

        def replace_load(node):
            if node.op.name == "tl.tt.compute_value_load":
                return tirx.Call(node.dtype, node.op, [tirx.IntImm("int64", remote)], node.annotations, node.span)
            return None

        attrs["tt.expression"] = tirx.stmt_functor.ir_transform(tirx.Evaluate(expression), None, replace_load, ["tirx.Call"]).value
        return tirx.Call(call.dtype, call.op, call.args, attrs, call.span)

    malformed = rewrite_calls(mod, edit)
    assert changed
    check_rejected(malformed, "Core.*ownership|not dominated")


def test_cross_core_value_definition_is_rejected():
    mod = multicore_values()
    trisc = [func for func in mod.functions.values() if func.attrs["tt.kernel_slot"] == "trisc"]
    definition = next(stmt for stmt in trisc[0].body.seq if stmt.value.op.name == "tl.tt.compute_value")
    other = trisc[1]
    mod.update_func(mod.get_global_var(str(other.attrs["global_symbol"])), other.with_body(tirx.SeqStmt([definition, *other.body.seq])))
    check_rejected(mod, "duplicated")


def test_partial_pipeline_metadata_cannot_bypass_validation():
    from testing.python.target.test_tilelang_tenstorrent_compute_value_verifier import device

    mod = device().with_attr("tt.pipeline_stages", 2)
    check_rejected(mod, "missing required Module attr.*pipeline_extent")


def test_boolean_cast_cannot_be_used_as_numeric_compute():
    from testing.python.target.test_tilelang_tenstorrent_compute_value_verifier import device

    changed = False

    def edit(call):
        nonlocal changed
        if changed or call.op.name != "tl.tt.compute_value":
            return None
        changed = True
        attrs = dict(call.annotations)
        expression = attrs["tt.expression"]
        attrs["tt.expression"] = tirx.Cast(expression.dtype, expression > tirx.FloatImm(expression.dtype, 0))
        return tirx.Call(call.dtype, call.op, call.args, attrs, call.span)

    malformed = rewrite_calls(device(), edit)
    assert changed
    check_rejected(malformed, "boolean values are predicates")


def test_operation_diagnostic_identifies_symbol_resource_and_source():
    from testing.python.target.test_tilelang_tenstorrent_compute_value_verifier import device

    mod = device()
    owner = next(func for func in mod.functions.values() if func.attrs["tt.kernel_slot"] == "trisc")
    changed = False

    def edit(call):
        nonlocal changed
        if not changed and call.op.name == "tl.tt.compute_value":
            changed = True
            attrs = dict(call.annotations)
            attrs["tt.compute_dtype"] = tirx.StringImm("bfloat16")
            return tirx.Call(call.dtype, call.op, call.args, attrs, call.span)
        return None

    malformed = rewrite_calls(mod, edit)
    before = ir.save_json(malformed)
    with pytest.raises(ValueError) as failure:
        transform.VerifyTenstorrentDeviceIR()(malformed)
    message = str(failure.value)
    assert "[VerifyTenstorrentDeviceIR]" in message
    assert str(owner.attrs["global_symbol"]) in message
    assert "resource_ids=" in message
    assert "source_span=" in message
    assert "frontend02.py" in message
    assert ir.save_json(malformed) == before


def test_value_cannot_borrow_foreign_core_dfb():
    @T.prim_func
    def main(A: T.Tensor((64, 32), "float32"), C: T.Tensor((64, 32), "float32")):
        with T.Kernel(2, 1, threads=1) as (x, y):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 1})
            f = T.alloc_fragment((32, 32), "float32")
            T.copy(A[x * 32 : (x + 1) * 32, :], a)
            for i, j in T.Tiles(32, 32):
                f[i, j] = a[i, j] * 2
            T.copy(f, C[x * 32 : (x + 1) * 32, :])

    mod = lower_tenstorrent_ir(tvm.IRModule({"main": main}), TARGET)
    dfbs = {int(dfb.dfb_id): dfb for dfb in mod.attrs["tt.dfb_table"]}
    changed = False

    def edit(call):
        nonlocal changed
        if changed or call.op.name != "tl.tt.compute_value" or len(call.args) != 2:
            return None
        changed = True
        source = dfbs[int(call.args[1])]
        remote = next(
            identifier
            for identifier, dfb in dfbs.items()
            if int(dfb.consumer_domain.begin.x) != int(source.consumer_domain.begin.x) and str(dfb.producer_slot) == "ncrisc"
        )
        attrs = dict(call.annotations)

        def replace_load(node):
            if node.op.name == "tl.tt.dfb_load":
                return tirx.Call(node.dtype, node.op, [tirx.IntImm("int64", remote)], node.annotations, node.span)
            return None

        attrs["tt.expression"] = tirx.stmt_functor.ir_transform(
            tirx.Evaluate(attrs["tt.expression"]), None, replace_load, ["tirx.Call"]
        ).value
        return tirx.Call(call.dtype, call.op, [call.args[0], tirx.IntImm("int64", remote)], attrs, call.span)

    malformed = rewrite_calls(mod, edit)
    assert changed
    check_rejected(malformed, "borrows a DFB owned by another Core")


def test_pipeline_value_cannot_resurrect_released_previous_epoch():
    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2})
            f = T.alloc_fragment((32, 32), "float32")
            for _k in T.Pipelined(3, num_stages=2):
                T.copy(A, a)
                for i, j in T.Tiles(32, 32):
                    f[i, j] = a[i, j] * 2
                for i, j in T.Tiles(32, 32):
                    f[i, j] = f[i, j] + 1
                T.copy(f, C)

    mod = lower_tenstorrent_ir(tvm.IRModule({"main": main}), TARGET)
    first = None
    changed = False

    def edit(call):
        nonlocal first, changed
        if call.op.name != "tl.tt.compute_value" or not len(call.annotations["tt.value_inputs"]):
            return None
        if first is None:
            first = int(call.args[0])
            return None
        if changed:
            return None
        changed = True
        attrs = dict(call.annotations)
        attrs["tt.value_inputs"] = [tirx.IntImm("int64", first)]

        def replace_load(node):
            if node.op.name == "tl.tt.compute_value_load":
                return tirx.Call(node.dtype, node.op, [tirx.IntImm("int64", first)], node.annotations, node.span)
            return None

        attrs["tt.expression"] = tirx.stmt_functor.ir_transform(
            tirx.Evaluate(attrs["tt.expression"]), None, replace_load, ["tirx.Call"]
        ).value
        return tirx.Call(call.dtype, call.op, call.args, attrs, call.span)

    malformed = rewrite_calls(mod, edit)
    assert changed
    check_rejected(malformed, "after release")


@pytest.mark.parametrize("mutation", ["stage", "shared_pool"])
def test_pipeline_singleton_contract_cannot_be_forged(mutation):
    from testing.python.target.test_tilelang_tenstorrent_lower_composition import full_k_pipeline, lower

    mod = lower(full_k_pipeline(), version=5)
    relations = {str(key): list(value) for key, value in mod.attrs["tt.pipeline_relations"].items()}
    singleton = next(key for key, relation in relations.items() if int(relation[0]) == -1)
    if mutation == "stage":
        relations[singleton] = [tirx.IntImm("int64", -1), tirx.IntImm("int64", 1)]
        malformed = mod.with_attr("tt.pipeline_relations", relations)
        check_rejected(malformed, "invalid iteration/stage")
    else:
        groups = {str(key): value for key, value in mod.attrs["tt.dfb_storage_groups"].items()}
        pooled = next(key for key, relation in relations.items() if int(relation[0]) >= 0)
        groups[singleton] = groups[pooled]
        malformed = mod.with_attr("tt.dfb_storage_groups", groups)
        check_rejected(malformed, "singleton")


def test_pipe_payload_cannot_cross_pipeline_epochs():
    from testing.python.target.test_tilelang_tenstorrent_lower_composition import lower, pipeline_pipe_values
    from testing.python.target.test_tilelang_tenstorrent_phase6_resources import replace_transfer

    mod = lower(pipeline_pipe_values())
    transfers = list(mod.attrs["tt.pipe_transfer_table"])
    transfers[1] = replace_transfer(transfers[1], destination_dfb_id=transfers[2].destination_dfb_id)
    check_rejected(mod.with_attr("tt.pipe_transfer_table", transfers), "crosses pipeline iterations")


def test_pipeline_pool_cannot_reuse_another_cores_storage():
    from testing.python.target.test_tilelang_tenstorrent_lower_composition import lower, pipeline_pipe_values

    mod = lower(pipeline_pipe_values())
    relations = {int(key): list(value) for key, value in mod.attrs["tt.pipeline_relations"].items()}
    groups = {str(key): value for key, value in mod.attrs["tt.dfb_storage_groups"].items()}
    by_core = {}
    for dfb in mod.attrs["tt.dfb_table"]:
        identifier = int(dfb.dfb_id)
        if int(relations[identifier][0]) == 1:
            by_core.setdefault(int(dfb.producer_domain.begin.x), identifier)
    left, right = str(by_core[0]), str(by_core[1])
    groups[left], groups[right] = groups[right], groups[left]
    check_rejected(mod.with_attr("tt.dfb_storage_groups", groups), "generations disagree")


@pytest.mark.parametrize("hidden_state", ["init", "allocation"])
def test_branch_merge_does_not_erase_malformed_structured_block_state(hidden_state):
    from testing.python.target.test_tilelang_tenstorrent_frontend02 import fragment_program

    source = tvm.IRModule({"main": fragment_program().with_attr("target", TARGET)})
    canonical = transform.CanonicalizeTTElementwise()(source)
    changed = False

    def inject(node):
        nonlocal changed
        if changed or "tl.tt.compute_kind" not in node.block.annotations:
            return None
        changed = True
        block = node.block
        scratch = tirx.decl_buffer((32, 32), "float32", name="hidden_scratch", scope="local.fragment")
        malformed = tirx.SBlock(
            block.iter_vars,
            block.reads,
            block.writes,
            block.name_hint,
            block.body,
            init=tirx.Evaluate(tirx.call_extern("void", "tt_test_hidden_effect")) if hidden_state == "init" else None,
            alloc_buffers=[scratch] if hidden_state == "allocation" else [],
            match_buffers=block.match_buffers,
            annotations=block.annotations,
        )
        predicate_buffer = block.reads[0].buffer
        condition = tirx.BufferLoad(predicate_buffer, [0, 0]) > tirx.FloatImm("float32", 0)
        return tirx.IfThenElse(condition, tirx.SBlockRealize([], True, malformed), node)

    func = canonical["main"]
    malformed = tvm.IRModule({"main": func.with_body(tirx.stmt_functor.ir_transform(func.body, None, inject, ["tirx.SBlockRealize"]))})
    assert changed
    preserved = transform.CanonicalizeTTElementwise()(malformed)
    hidden = []
    tirx.stmt_functor.post_order_visit(
        preserved["main"].body,
        lambda node: hidden.append(node) if isinstance(node, tirx.SBlock) and (node.init is not None or len(node.alloc_buffers)) else None,
    )
    assert hidden, "if-conversion must not silently erase malformed block state"
    with pytest.raises(ValueError, match="opaque with no local allocations"):
        transform.VerifyTTComputeBlocks()(preserved)

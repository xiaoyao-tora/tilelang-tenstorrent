"""Explicit DST regions preserve scalar arithmetic and accumulator storage."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import lower_tenstorrent_ir, transform
from tilelang.tenstorrent.device_ir import ComputeRequirements
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_compute_value_verifier import check_rejected, rewrite_calls
from testing.python.target.test_tilelang_tenstorrent_frontend02 import METADATA, TARGET


def mixed_precision_program(live_fp32=False):
    @T.prim_func
    def main(A: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            acc = T.alloc_fragment((32, 32), "bfloat16")
            wide = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            if live_fp32:
                T.fill(wide, 1)
            T.gemm(a, a, acc, clear_accum=True)
            for i, j in T.Tiles(acc):
                acc[i, j] = T.Cast("bfloat16", T.Cast("float32", acc[i, j]) * T.float32(1.25))
            if live_fp32:
                for i, j in T.Tiles(acc):
                    acc[i, j] = acc[i, j] + T.Cast("bfloat16", wide[i, j])
            T.copy(acc, C)

    return lower_tenstorrent_ir(tvm.IRModule({"main": main}), TARGET)


def test_precision_regions_roundtrip_and_keep_fp32_arithmetic():
    mod = mixed_precision_program()
    assert int(mod.attrs["tt.device_ir_version"]) == 8
    compute = next(f for f in mod.functions.values() if f.attrs["tt.kernel_slot"] == "trisc")
    assert compute.attrs["tt.compute_requirements"].destination_width == "region_scoped"
    requirements = compute.attrs["tt.compute_region_requirements"]
    assert {str(req.destination_width) for req in requirements} == {"bits16_required", "bits32_required"}
    found = []

    def visit(node):
        if isinstance(node, tirx.Call) and "tt.expression" in node.annotations:
            tirx.stmt_functor.post_order_visit(
                node.annotations["tt.expression"], lambda expr: found.append(expr) if isinstance(expr, tirx.Mul) else None
            )

    tirx.stmt_functor.post_order_visit(compute.body, visit)
    assert found and all(str(expr.dtype) == "float32" for expr in found)
    restored = ir.load_json(ir.save_json(mod))
    transform.VerifyTenstorrentDeviceIR()(restored)
    ir.assert_structural_equal(mod, restored)


def test_precision_mode_is_checked_against_actual_arithmetic():
    changed = False

    def edit(call):
        nonlocal changed
        if changed or call.op.name != "tl.tt.compute_precision":
            return None
        changed = True
        return tirx.Call(call.dtype, call.op, [tirx.IntImm("int32", 1 - int(call.args[0]))], call.annotations, call.span)

    check_rejected(rewrite_calls(mixed_precision_program(), edit), "precision region mode conflicts")
    assert changed


def test_precision_region_requirements_cannot_be_forged():
    mod = mixed_precision_program()
    for symbol, func in list(mod.functions.items()):
        if func.attrs["tt.kernel_slot"] == "trisc":
            regions = list(func.attrs["tt.compute_region_requirements"])
            regions[0] = ComputeRequirements("unconstrained", "allowed")
            mod.update_func(symbol, func.with_attr("tt.compute_region_requirements", regions))
    check_rejected(mod, "region requirements disagree")


def test_old_value_cannot_cross_precision_boundary_without_snapshot_reentry():
    mod = mixed_precision_program()
    definitions = {int(v.value_id): v for v in mod.attrs["tt.compute_value_table"]}
    changed = False

    def edit(call):
        nonlocal changed
        if changed or call.op.name != "tl.tt.compute_value":
            return None
        value = definitions[int(call.args[0])]
        if int(value.previous_value_id) < 0 or len(call.args) != 2 or len(call.annotations["tt.value_inputs"]):
            return None
        changed = True
        previous = int(value.previous_value_id)
        attrs = dict(call.annotations)
        attrs["tt.expression"] = tirx.call_intrin(value.buffer.dtype, ir.Op.get("tl.tt.compute_value_load"), previous)
        attrs["tt.access_maps"] = []
        attrs["tt.input_shapes"] = []
        attrs["tt.value_inputs"] = [tirx.IntImm("int32", previous)]
        attrs["tt.value_access_maps"] = [[tirx.IntImm("int32", 0), tirx.IntImm("int32", 1)]]
        attrs["tt.value_input_shapes"] = [value.buffer.shape]
        return tirx.Call(call.dtype, call.op, [call.args[0]], attrs, call.span)

    check_rejected(rewrite_calls(mod, edit), "crosses precision region")
    assert changed


def test_v7_cannot_claim_precision_regions():
    check_rejected(mixed_precision_program().with_attr("tt.device_ir_version", 7), "compute_precision requires v8")


def test_actual_fp32_fragment_is_not_reloaded_in_bf16_dst():
    with pytest.raises((ValueError, NotImplementedError), match="precision|destination width"):
        mixed_precision_program(live_fp32=True)


def test_missing_initial_precision_marker_is_rejected():
    mod = mixed_precision_program()
    for symbol, func in list(mod.functions.items()):
        if func.attrs["tt.kernel_slot"] == "trisc":
            statements = list(func.body.seq)
            first = next(i for i, stmt in enumerate(statements) if stmt.value.op.name == "tl.tt.compute_precision")
            del statements[first]
            mod.update_func(symbol, func.with_body(tirx.SeqStmt(statements)))
    check_rejected(mod, "explicit precision region")


def test_region_snapshot_cannot_materialize_an_incomplete_old_k_version():
    mod = mixed_precision_program()
    values = list(mod.attrs["tt.compute_value_table"])
    initial = next(int(value.value_id) for value in values if int(value.accumulator_id) >= 0)
    changed = False

    def edit(call):
        nonlocal changed
        if changed or call.op.name != "tl.tt.compute_value_store":
            return None
        changed = True
        return tirx.Call(call.dtype, call.op, [tirx.IntImm("int32", initial), call.args[1]], call.annotations, call.span)

    check_rejected(rewrite_calls(mod, edit), "materialization precedes complete K")
    assert changed


def test_pure_dfb_fp32_reduction_cannot_execute_in_bf16_region():
    from testing.python.target.test_tilelang_tenstorrent_attention_verifier import attention_row_program

    mod = attention_row_program().with_attr("tt.device_ir_version", 8)
    marker = ir.Op.get("tl.tt.compute_precision")
    for symbol, func in list(mod.functions.items()):
        if func.attrs["tt.kernel_slot"] != "trisc":
            continue
        statements = [tirx.Evaluate(tirx.call_intrin("void", marker, 0))]
        switched = False
        for stmt in func.body.seq:
            if not switched and stmt.value.op.name == "tl.tt.compute_value":
                statements.append(tirx.Evaluate(tirx.call_intrin("void", marker, 1)))
                switched = True
            statements.append(stmt)
        mod.update_func(symbol, func.with_body(tirx.SeqStmt(statements)).without_attr("tt.compute_requirements"))
    with pytest.raises(ValueError, match="FP32 DFB arithmetic"):
        transform.InferTenstorrentComputeRequirements()(mod)

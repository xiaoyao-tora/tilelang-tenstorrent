"""Adversarial v7 IR checks, independent of the frontend's guarantees."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import lower_tenstorrent_ir, transform
from tilelang.tenstorrent.device_ir import ComputeValueDescriptor
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_frontend02 import TARGET, fragment_program, gemm_epilogue


def device(gemm=False):
    func = gemm_epilogue("store") if gemm else fragment_program()
    return lower_tenstorrent_ir(tvm.IRModule({"main": func}), TARGET)


def rewrite_calls(mod, edit):
    result = ir.load_json(ir.save_json(mod))
    for global_var, func in list(result.functions.items()):
        result.update_func(global_var, func.with_body(tirx.stmt_functor.ir_transform(func.body, None, edit, ["tirx.Call"])))
    return result


def check_rejected(mod, message):
    before = ir.save_json(mod)
    with pytest.raises(ValueError, match=message):
        transform.VerifyTenstorrentDeviceIR()(mod)
    assert ir.save_json(mod) == before


@pytest.mark.parametrize("field,value", [("version", 99), ("previous_value_id", 999), ("accumulator_id", 999)])
def test_corrupted_value_identity_chain(field, value):
    mod = device()
    values = list(mod.attrs["tt.compute_value_table"])
    old = values[-1]
    fields = dict(
        value_id=old.value_id,
        buffer=old.buffer,
        version=old.version,
        previous_value_id=old.previous_value_id,
        accumulator_id=old.accumulator_id,
        source_span=old.source_span,
    )
    fields[field] = value
    values[-1] = ComputeValueDescriptor(**fields)
    check_rejected(mod.with_attr("tt.compute_value_table", values), "predecessor|accumulator descriptor")


def test_duplicate_value_id_and_undefined_descriptor():
    mod = device()
    values = list(mod.attrs["tt.compute_value_table"])
    check_rejected(mod.with_attr("tt.compute_value_table", [*values, values[-1]]), "duplicate compute value")
    old = values[-1]
    extra = ComputeValueDescriptor(99, old.buffer, old.version + 1, old.value_id, -1, old.source_span)
    check_rejected(mod.with_attr("tt.compute_value_table", [*values, extra]), "undefined compute value")


def test_future_value_reference_is_rejected():
    mod = device()
    future = int(mod.attrs["tt.compute_value_table"][-1].value_id)
    changed = False

    def edit(call):
        nonlocal changed
        if not changed and call.op.name == "tl.tt.compute_value" and len(call.annotations["tt.value_inputs"]):
            changed = True
            attrs = dict(call.annotations)
            attrs["tt.value_inputs"] = [tirx.IntImm("int64", future)]
            return tirx.Call(call.dtype, call.op, call.args, attrs, call.span)
        return None

    corrupted = rewrite_calls(mod, edit)
    assert changed
    check_rejected(corrupted, "not dominated")


def test_duplicate_value_definition():
    mod = device()
    trisc = next(f for f in mod.functions.values() if f.attrs["tt.kernel_slot"] == "trisc")
    statements = list(trisc.body.seq)
    definition = next(s for s in statements if isinstance(s, tirx.Evaluate) and s.value.op.name == "tl.tt.compute_value")
    mod.update_func(mod.get_global_var(str(trisc.attrs["global_symbol"])), trisc.with_body(tirx.SeqStmt([*statements, definition])))
    check_rejected(mod, "duplicated|after release")


def test_missing_release_and_release_before_read():
    mod = device()
    func = next(f for f in mod.functions.values() if f.attrs["tt.kernel_slot"] == "trisc")
    statements = list(func.body.seq)
    release = next(i for i, stmt in enumerate(statements) if stmt.value.op.name == "tl.tt.dfb_release")
    missing = ir.load_json(ir.save_json(mod))
    missing.update_func(
        missing.get_global_var(str(func.attrs["global_symbol"])),
        func.with_body(tirx.SeqStmt(statements[:release] + statements[release + 1 :])),
    )
    check_rejected(missing, "missing release")
    premature = ir.load_json(ir.save_json(mod))
    premature.update_func(
        premature.get_global_var(str(func.attrs["global_symbol"])),
        func.with_body(tirx.SeqStmt([statements[release], *statements[:release], *statements[release + 1 :]])),
    )
    check_rejected(premature, "must follow wait")


def test_accumulator_zero_initialization_cannot_be_forged():
    mod = device(gemm=True)

    def edit(call):
        if call.op.name == "tl.tt.compute_value" and call.annotations["tt.compute_kind"].value == "fill":
            attrs = dict(call.annotations)
            attrs["tt.expression"] = tirx.FloatImm("float32", 1)
            return tirx.Call(call.dtype, call.op, call.args, attrs, call.span)
        return None

    check_rejected(rewrite_calls(mod, edit), "unique zero")


def test_compute_precision_requirement_is_rederived():
    mod = device()
    from tilelang.tenstorrent.device_ir import ComputeRequirements

    func = next(f for f in mod.functions.values() if f.attrs["tt.kernel_slot"] == "trisc").with_attr(
        "tt.compute_requirements", ComputeRequirements()
    )
    mod.update_func(mod.get_global_var(str(func.attrs["global_symbol"])), func)
    check_rejected(mod, "disagrees with actual")

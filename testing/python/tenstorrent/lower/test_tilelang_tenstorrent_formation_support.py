"""Formation checks actual scalar DAGs and always uses normalized input."""

import pytest
from tilelang import tvm
from tvm import ir, tirx
from tvm.ir import Op
from tilelang.tenstorrent import transform
from tilelang.tenstorrent.pipeline import TENSTORRENT_LOWER_PASS_ORDER
from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_phase2_form import add_fp32
from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_phase1_pipeline import phase1_noop

TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})


def before_formation(frontend=add_fp32):
    mod = tirx.transform.BindTarget(TARGET)(tvm.IRModule({"main": frontend}))
    for name in TENSTORRENT_LOWER_PASS_ORDER:
        if name == "FormTenstorrentDeviceProgram":
            break
        if name != "BindTarget":
            mod = getattr(transform, name)()(mod)
    return mod


def replace_expression(mod, expression):
    replaced = False

    def rewrite(node):
        nonlocal replaced
        if (
            isinstance(node, tirx.Call)
            and isinstance(node.op, Op)
            and node.op.name == "tl.tt.tile_compute"
            and node.annotations["tt.compute_kind"].value == "elementwise"
        ):
            replaced = True
            _, a, b = [arg.buffer for arg in node.args]
            attrs = dict(node.annotations)
            attrs["tt.expression"] = expression(
                tirx.BufferLoad(a, [0, 0]),
                tirx.BufferLoad(b, [0, 0]),
            )
            return tirx.Call(node.dtype, node.op, node.args, attrs, node.span)
        return None

    func = mod["main"]
    mod["main"] = func.with_body(tirx.stmt_functor.ir_transform(func.body, None, rewrite), func.span)
    assert replaced
    return mod


@pytest.mark.parametrize("expression", [None, lambda a, b: a + b, lambda a, b: a * b, lambda a, b: a / b, lambda a, b: (a + b) * a / b])
def test_supported_operations_share_planner(expression):
    mod = before_formation()
    if expression is not None:
        mod = replace_expression(mod, expression)
    formed = transform.FormTenstorrentDeviceProgram()(mod)
    assert str(formed.attrs["tt.ir_stage"]) == "device"
    assert formed.attrs["tt.dfb_storage_groups"]
    assert list(formed.attrs["tt.kernel_order"]) == ["main_trisc", "main_ncrisc", "main_brisc"]
    formed = transform.InferTenstorrentComputeRequirements()(formed)
    transform.VerifyTenstorrentDeviceIR()(formed)


@pytest.mark.parametrize(
    "expression,reason",
    [
        (lambda a, b: a + tirx.call_intrin("float32", "tirx.erf", b), "tirx.erf"),
        (lambda a, b: a * tirx.FloorMod(a, b), "FloorMod"),
    ],
)
def test_nested_unsupported_is_reported_before_formation(expression, reason):
    mod = replace_expression(before_formation(), expression)
    before = ir.save_json(mod)
    with pytest.raises(NotImplementedError, match=reason):
        transform.FormTenstorrentDeviceProgram()(mod)
    assert ir.save_json(mod) == before


@pytest.mark.parametrize("frontend", [add_fp32, phase1_noop])
@pytest.mark.parametrize("value", [None, 0, 2])
def test_all_inputs_require_normalized_topology(frontend, value):
    mod = before_formation(frontend)
    mod["main"] = (
        mod["main"].without_attr("tt.topology_normalized") if value is None else mod["main"].with_attr("tt.topology_normalized", value)
    )
    before = ir.save_json(mod)
    with pytest.raises(ValueError, match="NormalizeTenstorrentTopology"):
        transform.FormTenstorrentDeviceProgram()(mod)
    assert ir.save_json(mod) == before


def test_empty_body_uses_common_planner():
    mod = transform.FormTenstorrentDeviceProgram()(before_formation(phase1_noop))
    assert str(mod.attrs["tt.ir_stage"]) == "device"
    assert not mod.attrs.get("tt.dfb_families", [])
    assert not mod.attrs.get("tt.dfb_storage_groups", [])
    transform.VerifyTenstorrentDeviceIR()(transform.InferTenstorrentComputeRequirements()(mod))


def test_unsupported_actual_operation_has_no_noop_fallback():
    mod = before_formation(phase1_noop)

    def rewrite(node):
        if isinstance(node, tirx.Evaluate):
            return tirx.Evaluate(tirx.call_extern("int32", "unsupported_formation_operation"))
        return None

    func = mod["main"]
    mod["main"] = func.with_body(tirx.stmt_functor.ir_transform(func.body, None, rewrite), func.span)
    before = ir.save_json(mod)
    with pytest.raises(NotImplementedError, match="unsupported program: operation"):
        transform.FormTenstorrentDeviceProgram()(mod)
    assert ir.save_json(mod) == before


def test_unsupported_nested_dtype_reports_unsupported():
    mod = replace_expression(before_formation(), lambda a, b: a + tirx.Cast("float32", tirx.Cast("float16", b)))
    before = ir.save_json(mod)
    with pytest.raises(NotImplementedError, match="scalar BF16/FP32"):
        transform.FormTenstorrentDeviceProgram()(mod)
    assert ir.save_json(mod) == before

"""Pure Device-IR-to-TTL mapping preflight with no TT-Lang dependency."""

from __future__ import annotations

from tvm import IRModule, tirx

from .resources import ResourceResolutionError, resolve_current_function_groups
from .records import RecordResolutionError, resolve_transport_records


class TTLMappingPreflightError(NotImplementedError):
    """The verified Device program has no complete current TTL mapping."""


def preflight_ttl_mapping(mod: IRModule) -> None:
    attrs = mod.attrs
    issues: set[str] = set()
    required = (
        "tt.dfb_storage_groups",
        "tt.transport_families",
        "tt.kernel_order",
    )
    for name in required:
        if name not in attrs:
            issues.add(f"missing required current Device attribute {name}")

    has_control_flow = False
    operation_names: set[str] = set()
    for function in mod.functions.values():

        def visit(node):
            nonlocal has_control_flow
            has_control_flow |= isinstance(node, (tirx.For, tirx.IfThenElse))
            if isinstance(node, tirx.Call) and hasattr(node.op, "name"):
                name = str(node.op.name)
                operation_names.add(name)

        tirx.stmt_functor.post_order_visit(function.body, visit)

    if has_control_flow:
        issues.add("shared Core/SCF control-flow emission is not implemented")
    try:
        resolve_current_function_groups(mod)
    except (KeyError, ResourceResolutionError) as error:
        issues.add(f"DFB stream binding resolution failed: {error}")
    try:
        resolve_transport_records(mod)
    except (KeyError, RecordResolutionError) as error:
        issues.add(f"PipeNet record resolution failed: {error}")
    current_operations = {
        "tl.tt.core_x",
        "tl.tt.core_y",
        "tl.tt.dfb_compute",
        "tl.tt.dfb_load",
        "tl.tt.compute_broadcast",
        "tl.tt.compute_precision",
        "tl.tt.compute_value",
        "tl.tt.compute_value_load",
        "tl.tt.compute_value_store",
        "tl.tt.compute_value_gemm",
        "tl.tt.tensor_to_dfb_nd",
        "tl.tt.dfb_to_tensor_nd",
        "tl.tt.dfb_reserve",
        "tl.tt.dfb_wait",
        "tl.tt.dfb_copy_wait",
        "tl.tt.dfb_release",
        "tl.tt.dfb_pipe_send_record",
        "tl.tt.dfb_pipe_recv_record",
        "tl.tt.dfb_pipe_wait_record",
        "tl.tt.dfb_local_copy",
        "tl.tt.dfb_gather_copy",
        "tl.tt.dfb_collective_reduce",
        "tl.tt.accumulator_init",
        "tl.tt.gemm_update",
        "tl.tt.accumulator_materialize",
    }
    unsupported = sorted(name for name in operation_names if name not in current_operations)
    if unsupported:
        issues.add("current Device operation mappings are incomplete: " + ", ".join(unsupported))

    if not issues:
        issues.add("stage-marked shared-grid TTL module emission is not implemented")
    raise TTLMappingPreflightError("TTL mapping preflight failed: " + "; ".join(sorted(issues)))


__all__ = ["TTLMappingPreflightError", "preflight_ttl_mapping"]

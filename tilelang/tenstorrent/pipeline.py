"""Tenstorrent structured lowering and the strict Device compilation boundary."""

from __future__ import annotations

from tvm import IRModule, tirx
from tvm.target import Target

from tilelang.backend.pass_pipeline import PassPipeline

from . import transform


TENSTORRENT_LOWER_PASS_ORDER = (
    "BindTarget",
    "ValidateTenstorrentFrontendIR",
    "CanonicalizeTTElementwise",
    "NormalizeTenstorrentLaunch",
    "NormalizeTenstorrentBufferMetadata",
    "NormalizeTenstorrentTopology",
    "NormalizeTenstorrentRegions",
    "VerifyTTStructuredDataflow",
    "LegalizeTenstorrentTileOps",
    "InferTenstorrentTensorLayout",
    "FormTenstorrentDeviceProgram",
    "InferTenstorrentComputeRequirements",
    "VerifyTenstorrentDeviceIR",
)


def _reject_prebound_frontend_targets(mod: IRModule) -> None:
    for symbol, func in mod.functions.items():
        if not isinstance(func, tirx.PrimFunc):
            continue
        if func.attrs is not None and "target" in func.attrs:
            raise ValueError(f"Tenstorrent lowering requires target-free frontend IR; function {symbol.name_hint!r} has a pre-bound target")


def _has_compute_blocks(func: tirx.PrimFunc) -> bool:
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, tirx.SBlock) and "tl.tt.compute_kind" in node.annotations:
            found = True

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return found


def _has_fragment_compute(func: tirx.PrimFunc) -> bool:
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, tirx.SBlock) and "tl.tt.compute_kind" in node.annotations:
            found |= any(region.buffer.scope() == "local.fragment" for region in (*node.reads, *node.writes))

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return found


def lower_tenstorrent_ir(mod: IRModule, target: Target) -> IRModule:
    """Return verified Device IR or a complete, validated structured module.

    This hardware-independent entry point does not resolve an execution backend.
    A structured result has ``tt.ir_stage="structured"`` and is not executable.
    Malformed IR and failed semantic proofs propagate as errors, never fallback.
    """

    if target.kind.name != "tenstorrent":
        raise ValueError("Tenstorrent lowering requires a tenstorrent target")

    stage = "" if mod.attrs is None else str(mod.attrs.get("tt.ir_stage", ""))
    if stage:
        raise ValueError("lower_tenstorrent_ir accepts only the original frontend module")
    if mod.attrs is not None and "tt.device_ir_version" in mod.attrs:
        raise ValueError("lower_tenstorrent_ir does not accept versioned Device IR")
    _reject_prebound_frontend_targets(mod)
    if mod.attrs is not None and "tt.deferred_reasons" in mod.attrs:
        mod = mod.without_attr("tt.deferred_reasons")

    passes = (
        tirx.transform.BindTarget(target),
        transform.ValidateTenstorrentFrontendIR(),
        # Capture Tiles while its logical loop binders and original
        # allocation metadata are intact, before any execution lowering.
        transform.CanonicalizeTTElementwise(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
        transform.NormalizeTenstorrentTopology(),
        transform.NormalizeTenstorrentRegions(),
        transform.VerifyTTStructuredDataflow(),
    )
    for compiler_pass in passes:
        mod = compiler_pass(mod)

    normalized = mod
    mod = transform.LegalizeTenstorrentTileOps()(normalized)
    reasons = []
    for symbol, func in mod.functions.items():
        if not isinstance(func, tirx.PrimFunc):
            raise ValueError("Tenstorrent lowering accepts only PrimFunc globals")
        if _has_fragment_compute(func):
            reasons.append(
                f"{symbol.name_hint}: fragment compute block has no Device consumer; value-version and materialization lowering is required"
            )
        elif _has_compute_blocks(func):
            reasons.append(f"{symbol.name_hint}: compute block has no Device consumer")
        # Pipe-only programs have a communication ABI and are verified by the
        # multicore formation path even when no Tensor parameter is present.
        has_topology = func.attrs is not None and "tt.normalized_topology" in func.attrs and len(func.attrs["tt.normalized_topology"]) > 0
        if not func.buffer_map and not has_topology and _has_computation(func):
            reasons.append(f"{symbol.name_hint}: standalone computation has no Tensor ABI")
    if reasons:
        # Return the whole original module, including supported siblings. Never
        # leak a mixture of selected Device instructions and expression templates.
        return normalized.with_attrs({"tt.ir_stage": "structured", "tt.deferred_reasons": sorted(reasons)})

    try:
        mod = transform.InferTenstorrentTensorLayout()(mod)
        mod = transform.FormTenstorrentDeviceProgram()(mod)
    except NotImplementedError as exc:
        return normalized.with_attrs(
            {
                "tt.ir_stage": "structured",
                "tt.deferred_reasons": [str(exc)],
            }
        )
    try:
        mod = transform.InferTenstorrentComputeRequirements()(mod)
    except NotImplementedError as exc:
        return normalized.with_attrs(
            {
                "tt.ir_stage": "structured",
                "tt.deferred_reasons": [str(exc)],
            }
        )
    mod = transform.VerifyTenstorrentDeviceIR()(mod)
    return mod.with_attr("tt.ir_stage", "device")


def _has_computation(func: tirx.PrimFunc) -> bool:
    found = _has_compute_blocks(func)

    def visit(node):
        nonlocal found
        if isinstance(node, tirx.Call) and hasattr(node.op, "name"):
            found |= str(node.op.name) in {
                "tl.tt.tile_add",
                "tl.tt.tile_compute",
                "tl.tileop.gemm",
                "tl.tileop.fill",
                "tl.tileop.reduce",
                "tl.tileop.transpose",
            }

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return found


def TenstorrentPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    """Require Device IR before shared host/device filtering and compilation."""
    result = lower_tenstorrent_ir(mod, target)
    if result.attrs is not None and str(result.attrs.get("tt.ir_stage", "")) == "structured":
        reasons = "; ".join(str(reason) for reason in result.attrs["tt.deferred_reasons"])
        raise NotImplementedError(
            f"Tenstorrent lowering produced structured IR, not executable Device IR: {reasons}. "
            "Use tilelang.tenstorrent.lower_tenstorrent_ir to inspect the structured result."
        )
    return result


TENSTORRENT_PIPELINE = PassPipeline("tenstorrent", TenstorrentPassPipelineBody)

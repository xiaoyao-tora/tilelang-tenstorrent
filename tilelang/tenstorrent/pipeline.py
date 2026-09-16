"""Tenstorrent structured lowering and the strict Device compilation boundary."""

from __future__ import annotations

from tvm import IRModule, tirx
from tvm.target import Target

from tilelang.backend.pass_pipeline import PassPipeline

from . import transform


TENSTORRENT_LOWER_PASS_ORDER = (
    "BindTarget",
    "CanonicalizeTTElementwise",
    "VerifyTTComputeBlocks",
    "ValidateTenstorrentFrontendIR",
    "NormalizeTenstorrentLaunch",
    "NormalizeTenstorrentBufferMetadata",
    "VerifyTTGemmAccumulators",
    "NormalizeTenstorrentTopology",
    "NormalizeTenstorrentRegions",
    "LegalizeTenstorrentTileOps",
    "InferTenstorrentTensorLayout",
    "FormTenstorrentDeviceProgram",
    "InferTenstorrentComputeRequirements",
    "VerifyTenstorrentDeviceIR",
)


def _has_compute_blocks(func: tirx.PrimFunc) -> bool:
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, tirx.SBlock) and "tl.tt.compute_kind" in node.annotations:
            found = True

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

    if mod.attrs is not None and "tt.device_ir_version" in mod.attrs:
        if "tt.ir_stage" in mod.attrs:
            raise ValueError("Device IR must not carry a structured stage marker")
        verified = transform.VerifyTenstorrentDeviceIR()(mod)
        if str(verified.attrs["tt.target_arch"]) != str(target.attrs["arch"]):
            raise ValueError("Tenstorrent Device IR architecture disagrees with lowering target")
        return verified

    # The normalized launch no longer has the frontend's thread-binding nest.
    # Do not skip frontend validation based on an untrusted stage annotation.
    if mod.attrs is not None and "tt.ir_stage" in mod.attrs:
        raise ValueError("Cannot re-lower normalized structured IR; pass the original frontend module instead")
    if mod.attrs is not None and "tt.deferred_reasons" in mod.attrs:
        mod = mod.without_attr("tt.deferred_reasons")

    passes = (
        tirx.transform.BindTarget(target),
        # Capture both frontends while their logical loop binders and original
        # allocation metadata are intact, before any execution lowering.
        transform.CanonicalizeTTElementwise(),
        transform.VerifyTTComputeBlocks(),
        transform.ValidateTenstorrentFrontendIR(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
        transform.VerifyTTGemmAccumulators(),
        transform.NormalizeTenstorrentTopology(),
        transform.NormalizeTenstorrentRegions(),
    )
    for compiler_pass in passes:
        mod = compiler_pass(mod)

    normalized = mod
    mod = transform.LegalizeTenstorrentTileOps()(normalized)
    reasons = []
    for symbol, func in mod.functions.items():
        if not isinstance(func, tirx.PrimFunc):
            raise ValueError("Tenstorrent lowering accepts only PrimFunc globals")
        if _has_compute_blocks(func):
            reasons.append(f"{symbol.name_hint}: compute block has no Device consumer")
        # Pipe-only programs have a communication ABI and are verified by the
        # multicore formation path even when no Tensor parameter is present.
        has_topology = func.attrs is not None and "tt.topology_normalized" in func.attrs
        if not func.buffer_map and not has_topology and _has_computation(func):
            reasons.append(f"{symbol.name_hint}: standalone computation has no Tensor ABI")
    if reasons:
        # Return the whole original module, including supported siblings. Never
        # leak a mixture of selected Device instructions and expression templates.
        normalized = transform.VerifyTTComputeBlocks()(normalized)
        normalized = transform.VerifyTTStructuredDataflow()(normalized)
        return normalized.with_attrs({"tt.ir_stage": "structured", "tt.deferred_reasons": reasons})

    for compiler_pass in (
        transform.InferTenstorrentTensorLayout(),
        transform.FormTenstorrentDeviceProgram(),
        transform.InferTenstorrentComputeRequirements(),
        transform.VerifyTenstorrentDeviceIR(),
    ):
        mod = compiler_pass(mod)
    return mod


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
    if result.attrs is not None and "tt.ir_stage" in result.attrs:
        reasons = "; ".join(str(reason) for reason in result.attrs["tt.deferred_reasons"])
        raise NotImplementedError(
            f"Tenstorrent lowering produced structured IR, not executable Device IR: {reasons}. "
            "Use tilelang.tenstorrent.lower_tenstorrent_ir to inspect the structured result."
        )
    return result


TENSTORRENT_PIPELINE = PassPipeline("tenstorrent", TenstorrentPassPipelineBody)

"""Tenstorrent computation capture and complete Device TIR lowering."""

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
    "NormalizeTenstorrentRegions",
    "NormalizeTenstorrentTopology",
    "LegalizeTenstorrentTileOps",
    "InferTenstorrentTensorLayout",
    "FormTenstorrentDeviceProgram",
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


def TenstorrentPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    """Lower a complete supported program, or diagnose the unsupported input.

    Capture-only clients can invoke CanonicalizeTTElementwise followed by
    VerifyTTComputeBlocks directly. This pipeline always returns verified
    Device IR, including when called on an existing Device module.

    FormTenstorrentDeviceProgram consumes supported static Pipelined loops and
    forms v3 window schedules, asynchronous copy completion and storage release.
    The final verifier checks cross-slot dependencies and capacity reuse. No
    separate synchronization pass or generic GPU pipeline pass is required.
    """

    if mod.attrs is not None and "tt.device_ir_version" in mod.attrs:
        verified = transform.VerifyTenstorrentDeviceIR()(mod)
        if str(verified.attrs["tt.target_arch"]) != str(target.attrs["arch"]):
            raise ValueError("Tenstorrent Device IR architecture disagrees with lowering target")
        return verified

    passes = (
        tirx.transform.BindTarget(target),
        # Capture both frontends while their logical loop binders and original
        # allocation metadata are intact, before any execution lowering.
        transform.CanonicalizeTTElementwise(),
        transform.VerifyTTComputeBlocks(),
        transform.ValidateTenstorrentFrontendIR(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
        transform.NormalizeTenstorrentRegions(),
        transform.NormalizeTenstorrentTopology(),
    )
    for compiler_pass in passes:
        mod = compiler_pass(mod)

    mod = transform.LegalizeTenstorrentTileOps()(mod)
    deferred = any(_has_compute_blocks(func) for func in mod.functions.values() if isinstance(func, tirx.PrimFunc))
    if deferred:
        raise NotImplementedError("LegalizeTenstorrentTileOps left an unconsumed compute block; complete Device Lower is unsupported")

    for compiler_pass in (
        transform.InferTenstorrentTensorLayout(),
        transform.FormTenstorrentDeviceProgram(),
        transform.VerifyTenstorrentDeviceIR(),
    ):
        mod = compiler_pass(mod)
    return mod


TENSTORRENT_PIPELINE = PassPipeline("tenstorrent", TenstorrentPassPipelineBody)

"""Tenstorrent structured elementwise and Device TIR v1 lowering pipeline."""

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
    """Capture elementwise semantics, then consume supported device operations.

    Modules with deferred compute return normalized structured IR marked with
    ``tt.ir_stage=structured``. Their expression templates and full effects
    remain intact for a future device consumer. No generic block erasure,
    scalar simplification, or SIMT layout inference runs on those templates.
    """

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

    # Keep the whole module at one semantic stage. In particular, a module
    # containing a deferred block must not retain partially lowered tile_adds.
    structured = mod
    mod = transform.LegalizeTenstorrentTileOps()(mod)
    deferred = any(_has_compute_blocks(func) for func in mod.functions.values() if isinstance(func, tirx.PrimFunc))
    # Standalone shared-buffer computations have no Tensor ABI/data movement
    # from which DeviceProgram formation could build an executable operation.
    standalone = any(
        not func.params and _has_compute_blocks(func) for func in structured.functions.values() if isinstance(func, tirx.PrimFunc)
    )
    if deferred or standalone:
        return transform.VerifyTTComputeBlocks()(structured).with_attr("tt.ir_stage", "structured")

    for compiler_pass in (
        transform.InferTenstorrentTensorLayout(),
        transform.FormTenstorrentDeviceProgram(),
        transform.VerifyTenstorrentDeviceIR(),
    ):
        mod = compiler_pass(mod)
    return mod


TENSTORRENT_PIPELINE = PassPipeline("tenstorrent", TenstorrentPassPipelineBody)

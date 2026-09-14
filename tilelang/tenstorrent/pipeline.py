"""Tenstorrent Device TIR v1 lowering pipeline."""

from __future__ import annotations

from tvm import IRModule, tirx
from tvm.target import Target

from tilelang.backend.pass_pipeline import PassPipeline

from . import transform


TENSTORRENT_LOWER_PASS_ORDER = (
    "BindTarget",
    "ValidateTenstorrentFrontendIR",
    "NormalizeTenstorrentLaunch",
    "NormalizeTenstorrentBufferMetadata",
    "NormalizeTenstorrentRegions",
    "InferTenstorrentTensorLayout",
    "LegalizeTenstorrentTileOps",
    "NormalizeTenstorrentTopology",
    "FormTenstorrentDeviceProgram",
    "VerifyTenstorrentDeviceIR",
)


def TenstorrentPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    """Lower the supported Phase 1 frontend subset to verified Device TIR."""

    passes = (
        tirx.transform.BindTarget(target),
        transform.ValidateTenstorrentFrontendIR(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
        transform.NormalizeTenstorrentRegions(),
        transform.InferTenstorrentTensorLayout(),
        transform.LegalizeTenstorrentTileOps(),
        transform.NormalizeTenstorrentTopology(),
        transform.FormTenstorrentDeviceProgram(),
        transform.VerifyTenstorrentDeviceIR(),
    )
    for compiler_pass in passes:
        mod = compiler_pass(mod)
    return mod


TENSTORRENT_PIPELINE = PassPipeline("tenstorrent", TenstorrentPassPipelineBody)

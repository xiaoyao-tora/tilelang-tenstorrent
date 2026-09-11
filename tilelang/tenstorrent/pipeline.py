"""Registration-stage Tenstorrent lowering pipeline."""

from __future__ import annotations

from tvm import IRModule, tirx
from tvm.target import Target

from tilelang.backend.pass_pipeline import PassPipeline


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
    """Bind the target while preserving canonical TIR for future TT lowering."""

    return tirx.transform.BindTarget(target)(mod)


TENSTORRENT_PIPELINE = PassPipeline("tenstorrent", TenstorrentPassPipelineBody)

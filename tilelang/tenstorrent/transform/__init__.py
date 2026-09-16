"""Tenstorrent-specific lowering pass factories."""

import tvm_ffi

from . import _ffi_api


tvm_ffi.register_error("NotImplementedError", NotImplementedError)


def CanonicalizeTTElementwise():
    """Capture Tiles and supported Parallel loops as structured compute blocks.

    Run after BindTarget and before frontend normalization or loop/block
    lowering so the shared analysis can preserve logical access relations.
    """
    return _ffi_api.CanonicalizeTTElementwise()  # type: ignore[attr-defined]


def VerifyTTComputeBlocks():
    """Verify structured expression templates, effects and access descriptors."""
    return _ffi_api.VerifyTTComputeBlocks()  # type: ignore[attr-defined]


def VerifyTTGemmAccumulators():
    """Verify GEMM fragment lifetimes and record logical precision requirements.

    This frontend check does not imply that Device Lower or TTL codegen can
    implement the requested accumulator lifetime on the selected target.
    """
    return _ffi_api.VerifyTTGemmAccumulators()  # type: ignore[attr-defined]


def ValidateTenstorrentFrontendIR():
    return _ffi_api.ValidateTenstorrentFrontendIR()  # type: ignore[attr-defined]


def NormalizeTenstorrentLaunch():
    return _ffi_api.NormalizeTenstorrentLaunch()  # type: ignore[attr-defined]


def NormalizeTenstorrentBufferMetadata():
    return _ffi_api.NormalizeTenstorrentBufferMetadata()  # type: ignore[attr-defined]


def NormalizeTenstorrentRegions():
    return _ffi_api.NormalizeTenstorrentRegions()  # type: ignore[attr-defined]


def InferTenstorrentTensorLayout():
    return _ffi_api.InferTenstorrentTensorLayout()  # type: ignore[attr-defined]


def LegalizeTenstorrentTileOps():
    return _ffi_api.LegalizeTenstorrentTileOps()  # type: ignore[attr-defined]


def NormalizeTenstorrentTopology():
    return _ffi_api.NormalizeTenstorrentTopology()  # type: ignore[attr-defined]


def FormTenstorrentDeviceProgram():
    return _ffi_api.FormTenstorrentDeviceProgram()  # type: ignore[attr-defined]


def VerifyTenstorrentDeviceIR():
    return _ffi_api.VerifyTenstorrentDeviceIR()  # type: ignore[attr-defined]


__all__ = (
    "CanonicalizeTTElementwise",
    "VerifyTTComputeBlocks",
    "VerifyTTGemmAccumulators",
    "FormTenstorrentDeviceProgram",
    "InferTenstorrentTensorLayout",
    "LegalizeTenstorrentTileOps",
    "NormalizeTenstorrentBufferMetadata",
    "NormalizeTenstorrentLaunch",
    "NormalizeTenstorrentRegions",
    "NormalizeTenstorrentTopology",
    "ValidateTenstorrentFrontendIR",
    "VerifyTenstorrentDeviceIR",
)

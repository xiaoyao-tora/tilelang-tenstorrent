"""Tenstorrent-specific lowering pass factories."""

import tvm_ffi

from . import _ffi_api


tvm_ffi.register_error("NotImplementedError", NotImplementedError)


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

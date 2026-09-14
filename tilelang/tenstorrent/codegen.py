"""Registration-stage Tenstorrent device codegen boundary."""

from __future__ import annotations

from tvm import IRModule
from tvm.target import Target

from .contracts import SUPPORTED_ARCHITECTURES, TARGET_KIND
from .transform import VerifyTenstorrentDeviceIR


TTL_CODEGEN_NOT_IMPLEMENTED = "Tenstorrent Target 已注册，但 TTL codegen 尚未实现"


def prepare_ttl_codegen(mod: IRModule, target: Target) -> IRModule:
    """Validate the frozen TTL boundary without importing optional TT-Lang packages."""

    if target.kind.name != TARGET_KIND:
        raise ValueError(
            f"Tenstorrent TTL preparation requires target kind {TARGET_KIND!r}, "
            f"got {target.kind.name!r}."
        )
    arch = str(target.attrs.get("arch", ""))
    if arch not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)
        raise ValueError(
            f"Tenstorrent TTL preparation requires a supported architecture "
            f"({supported}), got {arch!r}."
        )
    return VerifyTenstorrentDeviceIR()(mod)


def build_ttl_without_compile(mod: IRModule, target: Target) -> IRModule:
    """Reject compilation until TileLang-to-TTL source generation exists."""

    del mod, target
    raise NotImplementedError(TTL_CODEGEN_NOT_IMPLEMENTED)

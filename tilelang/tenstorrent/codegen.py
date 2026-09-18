"""Registration-stage Tenstorrent device codegen boundary."""

from __future__ import annotations

from tvm import IRModule
from tvm.target import Target


TTL_CODEGEN_NOT_IMPLEMENTED = "Tenstorrent Target 已注册，但 TTL codegen 尚未实现"


def build_ttl_without_compile(mod: IRModule, target: Target) -> IRModule:
    """Reject compilation until TileLang-to-TTL source generation exists."""

    del mod, target
    raise NotImplementedError(TTL_CODEGEN_NOT_IMPLEMENTED)

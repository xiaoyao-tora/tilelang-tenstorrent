"""Verified Device IR to source-only TTL codegen boundary."""

from __future__ import annotations

from tvm import IRModule
from tvm.target import Target

from .contracts import SUPPORTED_ARCHITECTURES, TARGET_KIND
from .transform import VerifyTenstorrentDeviceIR


def prepare_ttl_codegen(mod: IRModule, target: Target) -> IRModule:
    """Validate the frozen TTL boundary without importing optional TT-Lang packages."""

    if target.kind.name != TARGET_KIND:
        raise ValueError(f"Tenstorrent TTL preparation requires target kind {TARGET_KIND!r}, got {target.kind.name!r}.")
    arch = str(target.attrs.get("arch", ""))
    if arch not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)
        raise ValueError(f"Tenstorrent TTL preparation requires a supported architecture ({supported}), got {arch!r}.")
    if mod.attrs is not None and str(mod.attrs.get("tt.ir_stage", "")) == "structured":
        raise ValueError("TTL codegen requires Device IR; structured IR has operations without a Device consumer")
    module_arch = "" if mod.attrs is None else str(mod.attrs.get("tt.target_arch", ""))
    if module_arch and module_arch != arch:
        raise ValueError(f"TTL target architecture mismatch: Device IR is {module_arch!r}, target is {arch!r}")
    return VerifyTenstorrentDeviceIR()(mod)


def build_ttl_without_compile(mod: IRModule, target: Target):
    """Return a TVM source module containing verified initial TTL MLIR.

    This does not compile TTKernel kernels or launch TTNN. The optional compiler
    is loaded only after Device verification and capability checks succeed.
    """
    from tilelang import tvm
    from .ttl_codegen.module import emit_ttl

    mod = prepare_ttl_codegen(mod, target)
    generated = emit_ttl(mod)
    return tvm.ffi.get_global_func("runtime.SourceModuleCreate")(str(generated), "ttl")

"""Lower-produced Device IR to in-memory initial TTL codegen boundary."""

from __future__ import annotations

from tvm import IRModule
from tvm.target import Target

from .contracts import SUPPORTED_ARCHITECTURES, TARGET_KIND


def prepare_ttl_codegen(mod: IRModule, target: Target) -> IRModule:
    """Check only the target/stage handoff owned by DeviceCodegen."""

    if target.kind.name != TARGET_KIND:
        raise ValueError(f"Tenstorrent TTL preparation requires target kind {TARGET_KIND!r}, got {target.kind.name!r}.")
    arch = str(target.attrs.get("arch", ""))
    if arch not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)
        raise ValueError(f"Tenstorrent TTL preparation requires a supported architecture ({supported}), got {arch!r}.")
    if mod.attrs is not None and str(mod.attrs.get("tt.ir_stage", "")) == "structured":
        raise ValueError("TTL codegen requires Device IR; structured IR has operations without a Device consumer")
    if mod.attrs is None or str(mod.attrs.get("tt.ir_stage", "")) != "device":
        raise ValueError("TTL codegen accepts only tt.ir_stage='device' current Device IR")
    if "tt.device_ir_version" in mod.attrs:
        raise ValueError("TTL codegen rejects legacy tt.device_ir_version modules")
    if "tt.current_device_table_expansion" in mod.attrs:
        raise ValueError("TTL codegen rejects diagnostic concrete-table expansion")
    module_arch = "" if mod.attrs is None else str(mod.attrs.get("tt.target_arch", ""))
    if module_arch and module_arch != arch:
        raise ValueError(f"TTL target architecture mismatch: Device IR is {module_arch!r}, target is {arch!r}")
    return mod


def build_ttl_without_compile(mod: IRModule, target: Target):
    """Return an in-memory initial TTL module without running TT-Lang passes."""
    mod = prepare_ttl_codegen(mod, target)
    from .ttl_codegen.module import emit_ttl

    return emit_ttl(mod)

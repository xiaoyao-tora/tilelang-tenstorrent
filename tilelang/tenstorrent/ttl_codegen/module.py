"""Transactional current Device IR to in-memory initial TTL emission."""

from __future__ import annotations

from .artifact import TenstorrentTTLModuleNode, build_module_metadata
from .bindings import load_bindings
from .function import TTLFunctionEmitter


def _target_arch_attribute(bindings, context, architecture: str):
    architectures = {
        "wormhole_b0": bindings.ttcore.Arch.WormholeB0,
        "blackhole": bindings.ttcore.Arch.Blackhole,
    }
    try:
        value = architectures[architecture]
    except KeyError as error:
        raise NotImplementedError(f"TTL codegen has no target-architecture mapping for {architecture!r}") from error
    return bindings.ttcore.ir.ArchAttr.get(context, int(value))


def emit_ttl(device_mod) -> TenstorrentTTLModuleNode:
    """Emit and verify one initial TTL module from Lower-produced Device IR."""

    bindings = load_bindings()
    ir = bindings.ir
    context = ir.Context()
    bindings.ttl.ensure_dialects_registered(context)
    with context, ir.Location.unknown(context):
        module = ir.Module.create()
        grid = device_mod.attrs["tt.launch_grid"]
        module.operation.attributes["ttl.launch_grid"] = ir.ArrayAttr.get(
            [ir.IntegerAttr.get(ir.IntegerType.get_signless(64, context), int(dimension)) for dimension in (grid.x, grid.y)],
            context,
        )
        architecture = str(device_mod.attrs["tt.target_arch"])
        module.operation.attributes["ttl.target_arch"] = _target_arch_attribute(bindings, context, architecture)
        emitter = TTLFunctionEmitter(bindings, context, device_mod)
        functions = {str(function.attrs["global_symbol"]): function for function in device_mod.functions.values()}
        with ir.InsertionPoint(module.body):
            for symbol in device_mod.attrs["tt.kernel_order"]:
                name = str(symbol)
                try:
                    function = functions[name]
                except KeyError as error:
                    raise RuntimeError(f"Device kernel order references missing function {name!r}") from error
                emitter.emit(function)
        module.operation.verify()
    return TenstorrentTTLModuleNode(
        context,
        module,
        build_module_metadata(device_mod),
        bindings,
    )


def check_capability(device_mod) -> None:
    """Compatibility helper that exercises the same emitters as production."""

    emit_ttl(device_mod)


__all__ = ["check_capability", "emit_ttl"]

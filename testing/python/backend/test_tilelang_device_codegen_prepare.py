from __future__ import annotations

from dataclasses import replace
import importlib

import pytest

from tilelang import tvm
from tilelang.backend import BackendContext, DeviceCodegen, get_backend
from tilelang.tenstorrent.codegen import prepare_ttl_codegen


lower_engine = importlib.import_module("tilelang.engine.lower")


def _context_with_device_codegen(device_codegen: DeviceCodegen) -> BackendContext:
    target = tvm.target.Target("llvm")
    backend = get_backend("cpu")
    module = replace(
        backend,
        device_codegens={
            "c": backend.device_codegens["c"],
            "llvm": device_codegen,
        },
    )
    execution_backend = next(spec for spec in backend.execution_backends if spec.matches(target))
    return BackendContext(
        module=module,
        target=target,
        target_host=tvm.target.Target("c"),
        execution_backend=execution_backend,
    )


def test_device_codegen_prepare_hook_replaces_shared_prepare():
    original = tvm.IRModule()
    prepared = tvm.IRModule()
    calls = []

    def prepare(mod, target):
        calls.append((mod, target.kind.name))
        return prepared

    context = _context_with_device_codegen(DeviceCodegen("unit", prepare=prepare))

    assert lower_engine._prepare_device_codegen_mod(original, context) is prepared
    assert calls == [(original, "llvm")]


def test_device_codegen_without_hook_keeps_shared_prepare(monkeypatch):
    calls = []

    def factory(name):
        def make_pass():
            def run(mod):
                calls.append(name)
                return mod

            return run

        return make_pass

    monkeypatch.setattr(lower_engine.tilelang.transform, "LowerIntrin", factory("LowerIntrin"))
    monkeypatch.setattr(lower_engine.tirx.transform, "Simplify", factory("Simplify"))
    monkeypatch.setattr(lower_engine.tilelang.transform, "HoistBroadcastValues", factory("HoistBroadcastValues"))

    context = _context_with_device_codegen(DeviceCodegen("unit"))
    mod = tvm.IRModule()

    assert lower_engine._prepare_device_codegen_mod(mod, context) is mod
    assert calls == ["LowerIntrin", "Simplify", "HoistBroadcastValues"]


def test_tenstorrent_prepare_is_identity_and_validates_target():
    from tilelang.tenstorrent import language as T
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    @T.prim_func
    def noop():
        with T.Kernel(1, 1, threads=1):
            T.evaluate(0)

    target = tvm.target.Target({"kind": "tenstorrent", "arch": "blackhole"})
    mod = TenstorrentPassPipelineBody(tvm.IRModule({"noop": noop}), target)

    assert prepare_ttl_codegen(mod, target).same_as(mod)
    assert get_backend("tenstorrent").get_device_codegen(target).prepare is prepare_ttl_codegen

    with pytest.raises(ValueError, match="requires target kind 'tenstorrent'"):
        prepare_ttl_codegen(mod, tvm.target.Target("llvm"))

    with pytest.raises(ValueError, match="tt.device_ir_version"):
        prepare_ttl_codegen(tvm.IRModule(), target)

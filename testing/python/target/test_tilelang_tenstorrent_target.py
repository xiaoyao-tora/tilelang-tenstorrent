from __future__ import annotations

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend import create_backend_context, get_backend
from tilelang.tenstorrent import execution_backend
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.codegen import TTL_CODEGEN_NOT_IMPLEMENTED


def _target(arch: str = "wormhole_b0", *, keys: list[str] | None = None):
    config = {"kind": "tenstorrent", "arch": arch}
    if keys is not None:
        config["keys"] = keys
    return tvm.target.Target(config)


@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
def test_tenstorrent_target_kind_parses_supported_architectures(arch):
    target = _target(arch)

    assert target.kind.name == "tenstorrent"
    assert str(target.attrs["arch"]) == arch
    assert list(target.keys) == ["tenstorrent"]
    assert target.kind.default_device_type == tvm.device("ext_dev", 0).dlpack_device_type()


def test_tenstorrent_target_requires_explicit_supported_architecture():
    with pytest.raises(ValueError, match="requires an explicit 'arch'"):
        tvm.target.Target({"kind": "tenstorrent"})

    with pytest.raises(ValueError, match="Unsupported Tenstorrent architecture 'grayskull'"):
        _target("grayskull")


def test_tenstorrent_target_rejects_noncanonical_keys():
    assert list(_target(keys=["tenstorrent"]).keys) == ["tenstorrent"]

    with pytest.raises(ValueError, match="keys must be exactly.*tenstorrent"):
        _target(keys=["tenstorrent", "gpu"])


def test_tenstorrent_backend_declares_only_ttnn():
    target = _target()
    backend = get_backend("tenstorrent")

    assert backend.target_kinds == ("tenstorrent",)
    assert backend.get_pipeline(target).name == "tenstorrent"
    assert backend.get_device_codegen(target).name == "tenstorrent"
    assert backend.allowed_execution_backends(target) == ("ttnn",)


def test_ttnn_availability_is_checked_lazily(monkeypatch):
    checked = []

    def find_spec(module_name):
        checked.append(module_name)
        return object() if module_name == "ttl" else None

    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", find_spec)

    assert not execution_backend.is_ttnn_available()
    assert checked == ["ttl", "ttnn"]


def test_tenstorrent_context_resolves_ttnn_when_dependencies_are_available(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())

    context = create_backend_context(
        {"kind": "tenstorrent", "arch": "blackhole"},
        target_host="c",
        execution_backend="auto",
    )

    assert context.module is get_backend("tenstorrent")
    assert context.target.kind.name == "tenstorrent"
    assert str(context.target.attrs["arch"]) == "blackhole"
    assert context.execution_backend.name == "ttnn"
    assert not context.execution_backend.enable_host_codegen
    assert not context.execution_backend.enable_device_compile


def test_tenstorrent_context_rejects_unavailable_ttnn(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: None)

    with pytest.raises(ValueError, match="No available execution backend.*Allowed: ttnn"):
        create_backend_context(
            {"kind": "tenstorrent", "arch": "wormhole_b0"},
            target_host="c",
            execution_backend="auto",
        )

    with pytest.raises(ValueError, match="requires extra dependencies"):
        create_backend_context(
            {"kind": "tenstorrent", "arch": "wormhole_b0"},
            target_host="c",
            execution_backend="ttnn",
        )


def test_tenstorrent_pipeline_binds_target_without_simt_lowering(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    context = create_backend_context(
        {"kind": "tenstorrent", "arch": "wormhole_b0"},
        target_host="c",
        execution_backend="ttnn",
    )

    @T.prim_func
    def main(A: T.Tensor((1,), "float32")):
        with T.Kernel(1, 1, threads=1):
            A[0] = 0.0

    lowered = context.lower(tvm.IRModule({"main": main}))
    lowered_target = lowered["main"].attrs["target"]

    assert lowered_target.kind.name == "tenstorrent"
    assert str(lowered_target.attrs["arch"]) == "wormhole_b0"


def test_tenstorrent_compile_fails_at_unimplemented_ttl_codegen(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())

    @T.prim_func
    def main(A: T.Tensor((1,), "float32")):
        with T.Kernel(1, 1, threads=1):
            A[0] = 0.0

    with pytest.raises(NotImplementedError, match=TTL_CODEGEN_NOT_IMPLEMENTED):
        tilelang.compile(
            main,
            target={"kind": "tenstorrent", "arch": "wormhole_b0"},
            target_host="c",
            execution_backend="ttnn",
        )

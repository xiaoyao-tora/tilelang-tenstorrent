from __future__ import annotations

from types import SimpleNamespace
import importlib

import pytest

from tilelang.jit.adapter.ttnn import TTNNKernelAdapter
from tilelang.tenstorrent.ttl_codegen.artifact import (
    TTLModuleMetadata,
    TensorABI,
    TenstorrentTTLModuleNode,
)


class _MemoryConfig:
    def __init__(self, buffer_type, memory_layout="INTERLEAVED"):
        self.buffer_type = buffer_type
        self.memory_layout = memory_layout


class _Device:
    def __init__(self, identifier=0):
        self._identifier = identifier

    def id(self):
        return self._identifier


class _Tensor:
    def __init__(self, shape, dtype, layout, device, memory_config):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.layout = layout
        self._device = device
        self._memory_config = memory_config

    def device(self):
        return self._device

    def memory_config(self):
        return self._memory_config


class _CompiledKernel:
    kernel_paths = []

    def __init__(self):
        self.calls = []

    def __call__(self, *tensors):
        self.calls.append(tensors)


class _Module:
    operation = object()
    body = object()

    def __str__(self):
        return "module { }"


def _artifact():
    tensor_abi = (
        TensorABI(0, (32, 32), "float32", (32, 32), "dram", "interleaved", "input", 0),
        TensorABI(1, (32, 32), "float32", (32, 32), "dram", "interleaved", "input", 1),
        TensorABI(2, (32, 32), "float32", (32, 32), "dram", "interleaved", "output", 2),
    )
    metadata = TTLModuleMetadata(
        target_arch="wormhole_b0",
        launch_grid=(1, 1),
        tensor_abi=tensor_abi,
        precision_policy=(),
        contract_digest="1" * 64,
        dependency_revisions=(("tt-lang", "test"),),
        source_map_summary=(),
        build_digest="2" * 64,
    )
    return TenstorrentTTLModuleNode(object(), _Module(), metadata, object())


def _fake_ttnn():
    module = SimpleNamespace()
    module.Tensor = _Tensor
    module.float32 = object()
    module.bfloat16 = object()
    module.TILE_LAYOUT = object()
    module.DRAM_MEMORY_CONFIG = _MemoryConfig("DRAM")
    module.L1_MEMORY_CONFIG = _MemoryConfig("L1")

    def empty(shape, *, dtype, layout, device, memory_config):
        return _Tensor(shape, dtype, layout, device, memory_config)

    module.empty = empty
    return module


def _adapter(monkeypatch):
    implementation = importlib.import_module("tilelang.jit.adapter.ttnn.adapter")

    fake_ttnn = _fake_ttnn()
    compiled = _CompiledKernel()
    original_import = implementation.importlib.import_module

    def import_module(name):
        return fake_ttnn if name == "ttnn" else original_import(name)

    monkeypatch.setattr(implementation.importlib, "import_module", import_module)
    monkeypatch.setattr(implementation, "compile_ttl_module", lambda *_args, **_kwargs: compiled)
    adapter = TTNNKernelAdapter(
        params=[object(), object(), object()],
        result_idx=None,
        func_or_mod=object(),
        device_artifact=_artifact(),
    )
    return adapter, compiled, fake_ttnn


def test_ttnn_adapter_allocates_output_and_returns_it(monkeypatch):
    adapter, compiled, ttnn = _adapter(monkeypatch)
    device = _Device()
    left = _Tensor((32, 32), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    right = _Tensor((32, 32), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    output = adapter(left, right)
    assert isinstance(output, _Tensor)
    assert output.device() is device
    assert compiled.calls == [(left, right, output)]


def test_ttnn_adapter_accepts_preallocated_output(monkeypatch):
    adapter, compiled, ttnn = _adapter(monkeypatch)
    device = _Device()
    tensors = tuple(_Tensor((32, 32), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG) for _ in range(3))
    assert adapter(*tensors) is tensors[-1]
    assert compiled.calls == [tensors]


def test_ttnn_adapter_rejects_non_ttnn_and_wrong_layout(monkeypatch):
    adapter, _compiled, ttnn = _adapter(monkeypatch)
    with pytest.raises(TypeError, match="ttnn.Tensor"):
        adapter(object(), object())
    device = _Device()
    wrong = _Tensor((32, 32), ttnn.float32, object(), device, ttnn.DRAM_MEMORY_CONFIG)
    good = _Tensor((32, 32), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    with pytest.raises(ValueError, match="TILE_LAYOUT"):
        adapter(wrong, good)


def test_jit_dispatch_selects_ttnn_adapter(monkeypatch):
    from tilelang.jit.kernel import JITKernel

    package = importlib.import_module("tilelang.jit.adapter.ttnn")
    calls = []

    class FakeAdapter:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(package, "TTNNKernelAdapter", FakeAdapter)
    kernel = JITKernel.__new__(JITKernel)
    kernel.execution_backend = "ttnn"
    kernel.target = object()
    kernel.compile_flags = None
    kernel.verbose = False
    artifact = SimpleNamespace(
        params=[object()],
        device_artifact=object(),
        host_mod=None,
        device_mod=None,
        rt_mod=None,
        kernel_source="",
    )
    adapter = kernel._create_adapter_from_artifact(
        object(),
        [0],
        artifact,
        {},
        {
            "kernel": "main",
            "target": "tenstorrent",
            "target_host": None,
            "backend": "ttnn",
        },
    )
    assert isinstance(adapter, FakeAdapter)
    assert calls[0]["device_artifact"] is artifact.device_artifact

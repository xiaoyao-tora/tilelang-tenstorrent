from __future__ import annotations

from dataclasses import asdict, replace
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


def _artifact(storage_formats=None):
    tensor_abi = (
        TensorABI(0, (32, 32), "float32", (32, 32), "dram", "interleaved", "input", 0),
        TensorABI(1, (32, 32), "float32", (32, 32), "dram", "interleaved", "input", 1),
        TensorABI(2, (32, 32), "float32", (32, 32), "dram", "interleaved", "output", 2),
    )
    if storage_formats is not None:
        tensor_abi = tuple(replace(abi, dtype="bfloat16", storage_format=fmt) for abi, fmt in zip(tensor_abi, storage_formats))
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
    module.bfloat8_b = object()
    module.bfloat4_b = object()
    module.TILE_LAYOUT = object()
    module.DRAM_MEMORY_CONFIG = _MemoryConfig("DRAM")
    module.L1_MEMORY_CONFIG = _MemoryConfig("L1")

    def empty(shape, *, dtype, layout, device, memory_config):
        return _Tensor(shape, dtype, layout, device, memory_config)

    module.empty = empty
    return module


def _adapter(monkeypatch, storage_formats=None):
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
        device_artifact=_artifact(storage_formats),
    )
    return adapter, compiled, fake_ttnn


@pytest.mark.parametrize("storage_format", [None, "", "bfp_bf8", "bfp_bf4"])
def test_tensor_abi_metadata_preserves_storage_format(monkeypatch, storage_format):
    from tilelang.tenstorrent.ttl_codegen import artifact

    abi = _artifact().metadata.tensor_abi[0]
    assert abi.storage_format == ""
    fields = asdict(replace(abi, dtype="bfloat16"))
    if storage_format is None:
        del fields["storage_format"]
    else:
        fields["storage_format"] = storage_format
    module = SimpleNamespace(
        attrs={
            "tt.launch_grid": SimpleNamespace(x=1, y=1),
            "tt.tensor_table": [SimpleNamespace(**fields)],
            "tt.target_arch": "wormhole_b0",
        },
        functions={},
    )
    monkeypatch.setattr(artifact.tvm.ir, "save_json", lambda _module: "{}")
    metadata = artifact.build_module_metadata(module)
    assert metadata.tensor_abi == (replace(abi, dtype="bfloat16", storage_format=storage_format or ""),)


@pytest.mark.parametrize(
    "formats, dtype_names",
    [
        (("", "", ""), ("bfloat16", "bfloat16", "bfloat16")),
        (("bfp_bf8", "bfp_bf8", "bfp_bf8"), ("bfloat8_b", "bfloat8_b", "bfloat8_b")),
        (("bfp_bf4", "bfp_bf4", "bfp_bf4"), ("bfloat4_b", "bfloat4_b", "bfloat4_b")),
        (("bfp_bf8", "", "bfp_bf4"), ("bfloat8_b", "bfloat16", "bfloat4_b")),
    ],
)
@pytest.mark.parametrize("preallocated", [False, True])
def test_ttnn_adapter_uses_storage_dtype(monkeypatch, formats, dtype_names, preallocated):
    adapter, compiled, ttnn = _adapter(monkeypatch, formats)
    device = _Device()
    tensors = tuple(
        _Tensor((32, 32), getattr(ttnn, name), ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG) for name in dtype_names
    )
    output = adapter(*(tensors if preallocated else tensors[:2]))
    assert output.dtype is getattr(ttnn, dtype_names[-1])
    assert output.device() is device
    if preallocated:
        assert output is tensors[-1]
    assert compiled.calls == [(*tensors[:2], output)]
    assert all(abi.dtype == "bfloat16" for abi in adapter.tensor_abi)


@pytest.mark.parametrize("storage_format", ["bfp_bf8", "bfp_bf4"])
@pytest.mark.parametrize("wrong_dtype", ["bfloat16", "other_bfp"])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_ttnn_adapter_rejects_wrong_storage_dtype(monkeypatch, storage_format, wrong_dtype, index):
    adapter, compiled, ttnn = _adapter(monkeypatch, (storage_format,) * 3)
    device = _Device()
    dtype = ttnn.bfloat8_b if storage_format == "bfp_bf8" else ttnn.bfloat4_b
    other = ttnn.bfloat4_b if storage_format == "bfp_bf8" else ttnn.bfloat8_b
    tensors = [_Tensor((32, 32), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
    tensors[index].dtype = ttnn.bfloat16 if wrong_dtype == "bfloat16" else other
    with pytest.raises(ValueError, match=f"argument {index} dtype mismatch:.*storage_format '{storage_format}'"):
        adapter(*tensors)
    assert compiled.calls == []


@pytest.mark.parametrize("storage_format", ["bfp_bf8", "bfp_bf4"])
def test_ttnn_adapter_rejects_bfp_for_plain_bf16(monkeypatch, storage_format):
    adapter, compiled, ttnn = _adapter(monkeypatch, ("",) * 3)
    device = _Device()
    dtype = ttnn.bfloat8_b if storage_format == "bfp_bf8" else ttnn.bfloat4_b
    tensor = _Tensor((32, 32), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    with pytest.raises(ValueError, match="dtype mismatch"):
        adapter(tensor, tensor)
    assert compiled.calls == []


@pytest.mark.parametrize("storage_format", ["", "bfp_bf8", "bfp_bf4"])
def test_ttnn_adapter_rejects_raw_addresses(monkeypatch, storage_format):
    adapter, compiled, _ttnn = _adapter(monkeypatch, (storage_format,) * 3)
    with pytest.raises(TypeError, match="device-resident ttnn.Tensor"):
        adapter(0x1000, 0x2000)
    assert compiled.calls == []


@pytest.mark.parametrize(
    "dtype, storage_format, error",
    [
        ("bfloat16", "bfp_bad", NotImplementedError),
        ("float32", "bfp_bf8", ValueError),
        ("float32", "bfp_bf4", ValueError),
        ("float16", "bfp_bf8", ValueError),
        ("int32", "bfp_bf4", ValueError),
    ],
)
@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("preallocated", [False, True])
def test_ttnn_adapter_rejects_invalid_storage_contract(monkeypatch, dtype, storage_format, error, index, preallocated):
    adapter, compiled, ttnn = _adapter(monkeypatch)
    adapter.tensor_abi = tuple(
        replace(abi, dtype=dtype, storage_format=storage_format) if candidate == index else abi
        for candidate, abi in enumerate(adapter.tensor_abi)
    )
    device = _Device()
    tensors = tuple(_Tensor((32, 32), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG) for _ in range(3))
    with pytest.raises(error, match="storage_format"):
        adapter(*(tensors if preallocated else tensors[:2]))
    assert compiled.calls == []


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("preallocated", [False, True])
def test_ttnn_adapter_plain_dtype_does_not_require_bfp_support(monkeypatch, dtype, preallocated):
    adapter, compiled, ttnn = _adapter(monkeypatch, None if dtype == "float32" else ("",) * 3)
    del ttnn.bfloat8_b
    del ttnn.bfloat4_b
    device = _Device()
    tensors = tuple(_Tensor((32, 32), getattr(ttnn, dtype), ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG) for _ in range(3))
    output = adapter(*(tensors if preallocated else tensors[:2]))
    assert output.dtype is getattr(ttnn, dtype)
    assert compiled.calls == [(*tensors[:2], output)]
    if preallocated:
        assert output is tensors[-1]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_ttnn_adapter_plain_dtype_preserves_mismatch_message(monkeypatch, dtype):
    adapter, compiled, ttnn = _adapter(monkeypatch, None if dtype == "float32" else ("",) * 3)
    tensor = _Tensor((32, 32), "wrong", ttnn.TILE_LAYOUT, _Device(), ttnn.DRAM_MEMORY_CONFIG)
    with pytest.raises(ValueError, match=f"^Tenstorrent argument 0 dtype mismatch: expected {dtype}, got wrong$"):
        adapter(tensor, tensor)
    assert compiled.calls == []


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

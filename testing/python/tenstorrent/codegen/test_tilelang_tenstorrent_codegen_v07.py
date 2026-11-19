from __future__ import annotations

import importlib.util

import pytest

from tilelang import tvm
from tvm.target import Target

from tilelang.tenstorrent import codegen, lower_tenstorrent_ir
from tilelang.tenstorrent.ttl_codegen.artifact import (
    TenstorrentTTLModuleNode,
    build_module_metadata,
)
from tilelang.tenstorrent.ttl_codegen.bindings import TTLANG_REVISION, load_bindings
from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_phase2_form import (
    add_fp32,
)


TARGET = Target({"kind": "tenstorrent", "arch": "wormhole_b0"})


def _device_add():
    return lower_tenstorrent_ir(tvm.IRModule({"main": add_fp32}), TARGET)


def test_codegen_prepare_trusts_lower_verified_device_ir(monkeypatch):
    device = _device_add()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Codegen must not repeat Device verification or mapping preflight")

    import tilelang.tenstorrent.transform as transform

    monkeypatch.setattr(transform, "VerifyTenstorrentDeviceIR", unexpected)
    assert codegen.prepare_ttl_codegen(device, TARGET).same_as(device)


def test_codegen_mapping_error_occurs_at_emitter(monkeypatch):
    from tilelang.tenstorrent.ttl_codegen import module

    device = _device_add()
    attempts = []

    def missing_bindings():
        attempts.append(True)
        raise ImportError("pinned TT-Lang unavailable")

    monkeypatch.setattr(module, "load_bindings", missing_bindings)
    with pytest.raises(ImportError, match="pinned TT-Lang unavailable"):
        codegen.build_ttl_without_compile(device, TARGET)
    assert attempts == [True]


def test_codegen_prepare_keeps_target_stage_checks():
    structured = tvm.IRModule().with_attr("tt.ir_stage", "structured")
    with pytest.raises(ValueError, match="requires Device IR"):
        codegen.prepare_ttl_codegen(structured, TARGET)
    device = _device_add()
    with pytest.raises(ValueError, match="architecture mismatch"):
        codegen.prepare_ttl_codegen(
            device,
            Target({"kind": "tenstorrent", "arch": "blackhole"}),
        )


def test_ttl_module_metadata_is_deterministic_and_complete():
    device = _device_add()
    left = build_module_metadata(device)
    right = build_module_metadata(device)
    assert left == right
    assert left.target_arch == "wormhole_b0"
    assert left.launch_grid == (1, 1)
    assert [entry.effect for entry in left.tensor_abi] == ["input", "input", "output"]
    assert [entry.shape for entry in left.tensor_abi] == [(32, 32)] * 3
    assert len(left.contract_digest) == 64
    assert len(left.build_digest) == 64
    assert [entry[0] for entry in left.source_map_summary] == [
        "main_brisc",
        "main_ncrisc",
        "main_trisc",
    ]


class _Module:
    def __init__(self):
        self.operation = object()
        self.body = object()

    def __str__(self):
        return "module { }"


def test_ttl_module_node_owns_module_and_allows_one_pipeline():
    metadata = build_module_metadata(_device_add())
    context = object()
    module = _Module()
    artifact = TenstorrentTTLModuleNode(context, module, metadata, object())
    assert artifact.context is context
    assert artifact.mlir_module is module
    assert artifact.inspect_source() == "module { }"
    with artifact.exclusive_mutation() as owned:
        assert owned is module
        with pytest.raises(RuntimeError, match="concurrent"), artifact.exclusive_mutation():
            pass
    with pytest.raises(RuntimeError, match="already been consumed"), artifact.exclusive_mutation():
        pass


def test_program_hash_includes_compiler_policy(monkeypatch):
    from tilelang.tenstorrent import ttlang_adapter

    metadata = build_module_metadata(_device_add())
    artifact = TenstorrentTTLModuleNode(object(), _Module(), metadata, object())
    monkeypatch.setattr(ttlang_adapter, "_package_version", lambda _name: "test")
    left = ttlang_adapter.stable_program_hash(
        artifact,
        compiler_options=("maximize_dst", True),
    )
    right = ttlang_adapter.stable_program_hash(
        artifact,
        compiler_options=("maximize_dst", False),
    )
    assert left != right
    assert left == ttlang_adapter.stable_program_hash(
        artifact,
        compiler_options=("maximize_dst", True),
    )


def _real_bindings():
    if importlib.util.find_spec("ttl") is None:
        pytest.skip(f"requires pinned TT-Lang compiler {TTLANG_REVISION}")
    return load_bindings()


def test_real_current_add_emits_in_memory_ttl_and_compiles():
    _real_bindings()
    from ttl.passmanager import PassManager

    artifact = codegen.build_ttl_without_compile(_device_add(), TARGET)
    assert isinstance(artifact, TenstorrentTTLModuleNode)
    source = artifact.inspect_source()
    assert source.count("func.func") == 3
    assert "ttl.add" in source
    assert "ttl.cb_push" in source
    assert "ttl.cb_pop" in source
    with artifact.context:
        manager = PassManager.parse(
            "builtin.module(ttl-to-ttkernel-pipeline{lower-to-emitc=false})",
            context=artifact.context,
        )
        manager.enable_verifier(True)
        manager.run(artifact.operation)
        artifact.operation.verify()
    assert "ttl.add" not in str(artifact)


def test_real_current_loop_state_emits_scf_iter_args():
    _real_bindings()
    from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_lower_shared_grid_v04 import (
        make_loop_carried_fragment_update,
    )

    frontend = make_loop_carried_fragment_update(8)
    device = lower_tenstorrent_ir(tvm.IRModule({"main": frontend}), TARGET)
    artifact = codegen.build_ttl_without_compile(device, TARGET)
    source = artifact.inspect_source()
    assert "scf.for" in source
    assert "iter_args" in source
    artifact.operation.verify()


@pytest.mark.parametrize("reduction", ["sum", "max"])
def test_real_current_reduction_compile_only(reduction):
    _real_bindings()
    from ttl.passmanager import PassManager
    from testing.python.tenstorrent.backend.test_tilelang_tenstorrent_capabilities import (
        keepdims_reduction,
    )

    device = lower_tenstorrent_ir(
        tvm.IRModule({"main": keepdims_reduction("float32", reduction)}),
        TARGET,
    )
    artifact = codegen.build_ttl_without_compile(device, TARGET)
    assert "ttl.reduce" in artifact.inspect_source()
    with artifact.context:
        manager = PassManager.parse(
            "builtin.module(ttl-to-ttkernel-pipeline{lower-to-emitc=false})",
            context=artifact.context,
        )
        manager.enable_verifier(True)
        manager.run(artifact.operation)
        artifact.operation.verify()


def test_real_current_summa_pipenet_compile_only():
    _real_bindings()
    from ttl.passmanager import PassManager
    from examples.tenstorrent.summa import make_summa

    frontend = make_summa(
        core_m=2,
        core_n=2,
        k_stages=2,
        block_m=32,
        block_n=32,
        block_k=32,
        accum_dtype="bfloat16",
    )
    device = lower_tenstorrent_ir(tvm.IRModule({"main": frontend}), TARGET)
    artifact = codegen.build_ttl_without_compile(device, TARGET)
    source = artifact.inspect_source()
    assert "ttl.pipenet_foreach_src" in source
    assert "ttl.pipenet_foreach_dst" in source
    assert source.count("func.func") == 3
    with artifact.context:
        manager = PassManager.parse(
            "builtin.module(ttl-to-ttkernel-pipeline{lower-to-emitc=false})",
            context=artifact.context,
        )
        manager.enable_verifier(True)
        manager.run(artifact.operation)
        artifact.operation.verify()


def test_real_dynamic_dfb_family_selection_is_explicitly_unsupported():
    _real_bindings()
    from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_lower_shared_grid_v04 import (
        all_gather,
    )

    device = lower_tenstorrent_ir(tvm.IRModule({"main": all_gather}), TARGET)
    with pytest.raises(NotImplementedError, match="resource-selection support"):
        codegen.build_ttl_without_compile(device, TARGET)

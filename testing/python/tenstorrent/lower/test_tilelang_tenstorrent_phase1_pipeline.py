from __future__ import annotations

import pytest
from tvm import ir, tirx

from tilelang import tvm
from tilelang.backend import create_backend_context
from tilelang.tenstorrent import execution_backend, transform
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.codegen import prepare_ttl_codegen

from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_phase0_contract import (
    FRONTEND_PROGRAMS,
)


TARGET_CONFIG = {"kind": "tenstorrent", "arch": "wormhole_b0"}
MODULE_ATTRS = {
    "tt.ir_stage",
    "tt.target_arch",
    "tt.launch_grid",
    "tt.operation_identity",
    "tt.tensor_table",
    "tt.dfb_families",
    "tt.dfb_storage_groups",
    "tt.dfb_storage_group_relations",
    "tt.communication_table",
    "tt.communication_step_families",
    "tt.transport_families",
    "tt.kernel_order",
}


@T.prim_func
def phase1_noop(A: T.Tensor((32, 32), T.bfloat16)):
    with T.Kernel(1, 1, threads=1):
        T.evaluate(0)


@T.prim_func
def phase1_store(A: T.Tensor((32, 32), T.bfloat16)):
    with T.Kernel(1, 1, threads=1):
        A[0, 0] = T.cast(0, T.bfloat16)


@T.prim_func
def phase1_dfb_only():
    with T.Kernel(1, 1, threads=1):
        T.alloc_shared((32, 32), T.bfloat16)
        T.evaluate(0)


@T.prim_func
def phase1_multicore_noop():
    with T.Kernel(2, 1, threads=1):
        T.evaluate(0)


def _context(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    return create_backend_context(
        TARGET_CONFIG,
        target_host="c",
        execution_backend="ttnn",
    )


def test_backend_context_lowers_minimal_frontend_to_device_ir(monkeypatch):
    context = _context(monkeypatch)
    frontend = tvm.IRModule({"phase1_noop": phase1_noop})

    lowered = context.lower(frontend)
    verified = transform.VerifyTenstorrentDeviceIR()(lowered)

    assert verified.same_as(lowered)
    assert set(str(key) for key in lowered.attrs.keys()) == MODULE_ATTRS
    assert str(lowered.attrs["tt.ir_stage"]) == "device"
    assert "tt.device_ir_version" not in lowered.attrs
    assert str(lowered.attrs["tt.target_arch"]) == "wormhole_b0"
    assert (int(lowered.attrs["tt.launch_grid"].x), int(lowered.attrs["tt.launch_grid"].y)) == (
        1,
        1,
    )
    assert str(lowered.attrs["tt.operation_identity"].operation_id) == "phase1_noop"
    assert len(lowered.attrs["tt.tensor_table"]) == 1
    assert not lowered.attrs["tt.dfb_families"]
    assert not lowered.attrs["tt.communication_table"]
    assert not lowered.attrs["tt.communication_step_families"]
    assert not lowered.attrs["tt.transport_families"]

    functions = {str(func.attrs["tt.kernel_slot"]): func for func in lowered.functions.values()}
    assert set(functions) == {"trisc", "ncrisc", "brisc"}
    assert str(functions["trisc"].attrs["tt.logical_kernel"].role) == "general"
    for slot, role in (("ncrisc", "tensor_io"), ("brisc", "transport")):
        assert str(functions[slot].attrs["tt.logical_kernel"].role) == role
        assert isinstance(functions[slot].body, tirx.Evaluate)
        assert int(functions[slot].body.value) == 0
    for func in functions.values():
        assert "tt.launch_grid" not in func.attrs
        assert "tt.buffer_metadata_table" not in func.attrs
        assert func.attrs["target"].kind.name == "tenstorrent"
        assert str(func.attrs["target"].attrs["arch"]) == "wormhole_b0"
        assert func.span is not None

    restored = ir.load_json(ir.save_json(lowered))
    ir.assert_structural_equal(restored, lowered)
    assert str(restored) == str(lowered)


def test_phase1_pipeline_is_deterministic(monkeypatch):
    context = _context(monkeypatch)
    first = context.lower(tvm.IRModule({"phase1_noop": phase1_noop}))
    second = context.lower(tvm.IRModule({"phase1_noop": phase1_noop}))
    ir.assert_structural_equal(first, second)
    assert ir.structural_hash(first) == ir.structural_hash(second)
    assert str(first) == str(second)


def test_phase1_read_only_gates_are_identity_for_supported_subset(monkeypatch):
    context = _context(monkeypatch)
    mod = tirx.transform.BindTarget(context.target)(tvm.IRModule({"phase1_noop": phase1_noop}))
    for compiler_pass in (
        transform.ValidateTenstorrentFrontendIR(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
        transform.NormalizeTenstorrentRegions(),
        transform.InferTenstorrentTensorLayout(),
    ):
        mod = compiler_pass(mod)
    gated = transform.LegalizeTenstorrentTileOps()(mod)
    ir.assert_structural_equal(gated, mod)
    gated = transform.NormalizeTenstorrentTopology()(gated)
    assert int(gated["phase1_noop"].attrs["tt.topology_normalized"]) == 1
    assert not gated["phase1_noop"].attrs["tt.normalized_topology"]


def test_phase1_pipeline_rejects_uncaptured_store(monkeypatch):
    context = _context(monkeypatch)
    with pytest.raises(ValueError, match="uncaptured BufferStore"):
        context.lower(tvm.IRModule({"store": phase1_store}))


def test_unused_shared_allocation_forms_empty_device_program(monkeypatch):
    context = _context(monkeypatch)
    result = context.lower(tvm.IRModule({"dfb_only": phase1_dfb_only}))
    assert str(result.attrs["tt.ir_stage"]) == "device"
    assert not result.attrs["tt.dfb_families"]


def test_topology_rejects_legacy_pipe_when_called_directly(monkeypatch):
    context = _context(monkeypatch)
    mod = tirx.transform.BindTarget(context.target)(tvm.IRModule({"p2p": FRONTEND_PROGRAMS["p2p"]}))
    with pytest.raises(NotImplementedError, match="legacy PipeNet"):
        transform.ValidateTenstorrentFrontendIR()(mod)


def test_multicore_noop_forms_verified_device_ir(monkeypatch):
    context = _context(monkeypatch)
    lowered = context.lower(tvm.IRModule({"multicore": phase1_multicore_noop}))
    assert str(lowered.attrs["tt.ir_stage"]) == "device"
    assert "tt.device_ir_version" not in lowered.attrs
    transform.VerifyTenstorrentDeviceIR()(lowered)


def test_pipe_with_uninitialized_payload_is_rejected(monkeypatch):
    context = _context(monkeypatch)
    with pytest.raises(NotImplementedError, match="legacy PipeNet"):
        context.lower(tvm.IRModule({"p2p": FRONTEND_PROGRAMS["p2p"]}))

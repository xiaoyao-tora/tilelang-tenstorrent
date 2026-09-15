from __future__ import annotations

import pytest
from tvm import ir, tirx

from tilelang import tvm
from tilelang.backend import create_backend_context
from tilelang.tenstorrent import execution_backend, transform
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.codegen import prepare_ttl_codegen

from testing.python.target.test_tilelang_tenstorrent_phase0_contract import (
    FRONTEND_PROGRAMS,
)


TARGET_CONFIG = {"kind": "tenstorrent", "arch": "wormhole_b0"}
MODULE_ATTRS = {
    "tt.device_ir_version",
    "tt.target_arch",
    "tt.launch_grid",
    "tt.operation_identity",
    "tt.tensor_table",
    "tt.dfb_table",
    "tt.pipe_table",
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
    assert int(lowered.attrs["tt.device_ir_version"]) == 1
    assert str(lowered.attrs["tt.target_arch"]) == "wormhole_b0"
    assert (int(lowered.attrs["tt.launch_grid"].x), int(lowered.attrs["tt.launch_grid"].y)) == (
        1,
        1,
    )
    assert str(lowered.attrs["tt.operation_identity"].operation_id) == "phase1_noop"
    assert len(lowered.attrs["tt.tensor_table"]) == 1
    assert not lowered.attrs["tt.dfb_table"]
    assert not lowered.attrs["tt.pipe_table"]

    functions = {str(func.attrs["tt.kernel_slot"]): func for func in lowered.functions.values()}
    assert set(functions) == {"trisc", "ncrisc", "brisc"}
    assert str(functions["trisc"].attrs["tt.logical_kernel"].role) == "phase1_skeleton"
    for slot in ("ncrisc", "brisc"):
        assert str(functions[slot].attrs["tt.logical_kernel"].role) == "idle"
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
    assert prepare_ttl_codegen(restored, context.target).same_as(restored)


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
    before = ir.save_json(mod)
    gated = transform.LegalizeTenstorrentTileOps()(mod)
    ir.assert_structural_equal(gated, mod)
    gated = transform.NormalizeTenstorrentTopology()(gated)
    ir.assert_structural_equal(gated, mod)
    assert ir.save_json(gated) == before


@pytest.mark.parametrize(
    ("name", "func", "message"),
    (
        ("p2p", FRONTEND_PROGRAMS["p2p"], "NormalizeTenstorrentTopology"),
        ("store", phase1_store, "structured T.Parallel or T.Tiles operation"),
        ("dfb_only", phase1_dfb_only, "requires Phase 2 transaction planning"),
        ("multicore", phase1_multicore_noop, "multi-Core program formation"),
    ),
)
def test_phase1_pipeline_rejects_deferred_capabilities(monkeypatch, name, func, message):
    context = _context(monkeypatch)
    with pytest.raises(NotImplementedError, match=message):
        context.lower(tvm.IRModule({name: func}))


def test_phase1_topology_gate_rejects_pipe_when_called_directly(monkeypatch):
    context = _context(monkeypatch)
    mod = tirx.transform.BindTarget(context.target)(tvm.IRModule({"p2p": FRONTEND_PROGRAMS["p2p"]}))
    for compiler_pass in (
        transform.ValidateTenstorrentFrontendIR(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
        transform.NormalizeTenstorrentRegions(),
        transform.InferTenstorrentTensorLayout(),
    ):
        mod = compiler_pass(mod)
    with pytest.raises(NotImplementedError, match="NormalizeTenstorrentTopology"):
        transform.NormalizeTenstorrentTopology()(mod)

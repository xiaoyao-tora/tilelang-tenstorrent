from __future__ import annotations

import pytest
from tvm import ir, tirx
from tvm.ir import Op

from tilelang import tvm
from tilelang.backend import create_backend_context
from tilelang.tenstorrent import execution_backend, transform
from tilelang.tenstorrent import language as T

from testing.python.target.test_tilelang_tenstorrent_phase0_contract import (
    FRONTEND_PROGRAMS,
)


TARGET_CONFIG = {"kind": "tenstorrent", "arch": "wormhole_b0"}


@T.prim_func
def add_fp32(
    A: T.Tensor((32, 32), T.float32),
    B: T.Tensor((32, 32), T.float32),
    C: T.Tensor((32, 32), T.float32),
):
    with T.Kernel(1, 1, threads=1):
        a = T.alloc_shared(
            (32, 32),
            T.float32,
            annotations={"tt.dfb_block_count": 2, "tt.tile_shape": (32, 32)},
        )
        b = T.alloc_shared(
            (32, 32),
            T.float32,
            annotations={"tt.dfb_block_count": 2, "tt.tile_shape": (32, 32)},
        )
        c = T.alloc_shared(
            (32, 32),
            T.float32,
            annotations={"tt.dfb_block_count": 2, "tt.tile_shape": (32, 32)},
        )
        T.copy(A, a)
        T.copy(B, b)
        for i, j in T.Parallel(32, 32):
            c[i, j] = a[i, j] + b[i, j]
        T.copy(c, C)


def _context(monkeypatch):
    monkeypatch.setattr(execution_backend.importlib.util, "find_spec", lambda _: object())
    return create_backend_context(
        TARGET_CONFIG,
        target_host="c",
        execution_backend="ttnn",
    )


def _functions_by_slot(mod):
    return {str(func.attrs["tt.kernel_slot"]): func for func in mod.functions.values()}


def _body_calls(func):
    statements = func.body.seq if isinstance(func.body, tirx.SeqStmt) else [func.body]
    calls = []
    for statement in statements:
        assert isinstance(statement, tirx.Evaluate)
        assert isinstance(statement.value, tirx.Call)
        assert isinstance(statement.value.op, Op)
        assert statement.value.dtype.bits == 0
        assert statement.value.dtype.lanes == 0
        calls.append(statement.value)
    return calls


@pytest.mark.parametrize(
    ("name", "func", "dtype"),
    (
        ("add_bf16", FRONTEND_PROGRAMS["add"], "bfloat16"),
        ("add_fp32", add_fp32, "float32"),
    ),
)
def test_form_add_builds_three_dfbs_and_canonical_slots(monkeypatch, name, func, dtype):
    context = _context(monkeypatch)
    lowered = context.lower(tvm.IRModule({name: func}))

    tensors = lowered.attrs["tt.tensor_table"]
    assert [str(tensor.effect) for tensor in tensors] == ["input", "input", "output"]
    assert [str(tensor.dtype) for tensor in tensors] == [dtype, dtype, dtype]

    dfbs = lowered.attrs["tt.dfb_table"]
    assert [int(dfb.dfb_id) for dfb in dfbs] == [0, 1, 2]
    assert [str(dfb.source_buffer_identity) for dfb in dfbs] == [
        "buffer.0",
        "buffer.1",
        "buffer.2",
    ]
    assert [str(dfb.producer_slot) for dfb in dfbs] == [
        "ncrisc",
        "ncrisc",
        "trisc",
    ]
    assert [str(dfb.consumer_slot) for dfb in dfbs] == [
        "trisc",
        "trisc",
        "ncrisc",
    ]
    for index, dfb in enumerate(dfbs):
        assert [int(value) for value in dfb.tile_shape] == [32, 32]
        assert [int(value) for value in dfb.block_shape_in_tiles] == [1, 1]
        assert int(dfb.block_count) == 2
        assert int(dfb.transaction_count_or_loop_relation) == 1
        assert int(dfb.tensor_backing.global_arg_index) == index
        assert int(dfb.tensor_backing.byte_offset) == 0
        assert dfb.source_span is not None

    functions = _functions_by_slot(lowered)
    assert len(functions["trisc"].params) == 0
    assert len(functions["ncrisc"].params) == 3
    assert len(functions["brisc"].params) == 0
    assert [int(index) for index in functions["ncrisc"].attrs["tt.tensor_arg_indices"]] == [
        0,
        1,
        2,
    ]
    assert str(functions["trisc"].attrs["tt.logical_kernel"].kind) == "compute"
    assert str(functions["trisc"].attrs["tt.logical_kernel"].role) == "add"
    assert str(functions["ncrisc"].attrs["tt.logical_kernel"].kind) == "datamovement"
    assert str(functions["ncrisc"].attrs["tt.logical_kernel"].role) == "tensor_io"
    assert str(functions["brisc"].attrs["tt.logical_kernel"].role) == "idle"

    trisc_calls = _body_calls(functions["trisc"])
    assert [call.op.name for call in trisc_calls] == [
        "tl.tt.dfb_reserve",
        "tl.tt.dfb_wait",
        "tl.tt.dfb_wait",
        "tl.tt.dfb_add",
    ]
    assert [[int(arg) for arg in call.args] for call in trisc_calls] == [
        [2, 1],
        [0, 1],
        [1, 1],
        [0, 1, 2, 1],
    ]

    ncrisc_calls = _body_calls(functions["ncrisc"])
    assert [call.op.name for call in ncrisc_calls] == [
        "tl.tt.dfb_reserve",
        "tl.tt.tensor_to_dfb",
        "tl.tt.dfb_reserve",
        "tl.tt.tensor_to_dfb",
        "tl.tt.dfb_wait",
        "tl.tt.dfb_to_tensor",
    ]
    assert [[int(arg) for arg in call.args] for call in ncrisc_calls] == [
        [0, 1],
        [0, 0, 0, 0, 32, 32],
        [1, 1],
        [1, 1, 0, 0, 32, 32],
        [2, 1],
        [2, 2, 0, 0, 32, 32],
    ]
    assert isinstance(functions["brisc"].body, tirx.Evaluate)
    assert int(functions["brisc"].body.value) == 0
    assert not lowered.attrs["tt.pipe_table"]

    restored = ir.load_json(ir.save_json(lowered))
    ir.assert_structural_equal(restored, lowered)


def test_form_add_is_deterministic(monkeypatch):
    context = _context(monkeypatch)
    first = context.lower(tvm.IRModule({"add": FRONTEND_PROGRAMS["add"]}))
    second = context.lower(tvm.IRModule({"add": FRONTEND_PROGRAMS["add"]}))
    ir.assert_structural_equal(first, second)
    assert ir.structural_hash(first) == ir.structural_hash(second)
    assert str(first) == str(second)


def test_form_add_rejects_incomplete_dataflow_without_mutating_input(monkeypatch):
    context = _context(monkeypatch)
    malformed = tirx.transform.BindTarget(context.target)(tvm.IRModule({"add": FRONTEND_PROGRAMS["add"]}))
    for compiler_pass in (
        transform.CanonicalizeTTElementwise(),
        transform.VerifyTTComputeBlocks(),
        transform.ValidateTenstorrentFrontendIR(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
        transform.NormalizeTenstorrentRegions(),
        transform.InferTenstorrentTensorLayout(),
        transform.LegalizeTenstorrentTileOps(),
        transform.NormalizeTenstorrentTopology(),
    ):
        malformed = compiler_pass(malformed)

    changed = False

    def drop_output_copy(node):
        nonlocal changed
        if isinstance(node, tirx.SeqStmt) and len(node.seq) == 4:
            candidate = node.seq[2]
            if (
                isinstance(candidate, tirx.Evaluate)
                and isinstance(candidate.value, tirx.Call)
                and isinstance(candidate.value.op, Op)
                and candidate.value.op.name == "tl.tt.tile_add"
            ):
                changed = True
                return tirx.SeqStmt(list(node.seq[:3]))
        return None

    func = malformed["add"]
    body = tirx.stmt_functor.ir_transform(func.body, None, drop_output_copy)
    assert changed
    malformed["add"] = func.with_body(body, func.span)
    before = ir.save_json(malformed)

    with pytest.raises(NotImplementedError, match="two input copies, one Add"):
        transform.FormTenstorrentDeviceProgram()(malformed)

    assert ir.save_json(malformed) == before

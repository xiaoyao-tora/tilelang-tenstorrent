"""Structured success is atomic and cannot hide invalid frontend dataflow."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import lower_tenstorrent_ir, transform
from tilelang.tenstorrent.pipeline import TENSTORRENT_PIPELINE
from tvm import ir, tirx

TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})


def program(dtype="float16", *, initialize=True, partial=False):
    @T.prim_func
    def main(A: T.Tensor((32, 32), dtype), C: T.Tensor((32, 32), dtype)):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), dtype)
            c = T.alloc_shared((32, 32), dtype)
            if initialize:
                if partial:
                    T.copy(A[0:16, 0:32], a[0:16, 0:32])
                else:
                    T.copy(A, a)
            for i, j in T.Parallel(32, 32):
                c[i, j] = a[i, j] + T.cast(1, dtype)
            T.copy(c, C)

    return main


@T.prim_func
def standalone_fill():
    with T.Kernel(1, 1, threads=1):
        a = T.alloc_shared((32, 32), "float32")
        T.fill(a, 1)


def compute_blocks(mod):
    found = []
    for func in mod.functions.values():
        tirx.stmt_functor.post_order_visit(
            func.body,
            lambda node: found.append(node) if isinstance(node, tirx.SBlock) and "tl.tt.compute_kind" in node.annotations else None,
        )
    return found


def test_unsupported_dtype_returns_whole_structured_module_without_mutation():
    frontend = tvm.IRModule({"main": program()})
    before = ir.save_json(frontend)
    result = lower_tenstorrent_ir(frontend, TARGET)
    assert result.attrs["tt.ir_stage"] == "structured"
    assert "tt.device_ir_version" not in result.attrs
    assert len(compute_blocks(result)) == 1
    assert "tl.tt.tile_compute" not in result.script()
    assert "tl.tt.tile_add" not in result.script()
    assert ir.save_json(frontend) == before
    transform.VerifyTTComputeBlocks()(result)
    ir.assert_structural_equal(ir.load_json(ir.save_json(result)), result)
    with pytest.raises(ValueError, match="original frontend module"):
        lower_tenstorrent_ir(result, TARGET)


def test_supported_sibling_is_not_partially_rewritten_on_structured_return():
    frontend = tvm.IRModule({"deferred": program(), "supported": program("float32")})
    result = lower_tenstorrent_ir(frontend, TARGET)
    assert len(result.functions) == 2
    assert len(compute_blocks(result)) == 2
    assert "tl.tt.tile_compute" not in result.script()
    assert "tl.tt.tile_add" not in result.script()


@pytest.mark.parametrize("initialize,partial", [(False, False), (True, True)])
def test_deferred_capability_does_not_hide_read_before_initialization(initialize, partial):
    frontend = tvm.IRModule({"main": program(initialize=initialize, partial=partial)})
    with pytest.raises(ValueError, match="definite initialization"):
        lower_tenstorrent_ir(frontend, TARGET)


def test_invalid_supported_sibling_cannot_hide_behind_deferred_capability():
    frontend = tvm.IRModule({"deferred": program(), "invalid": program("float32", initialize=False)})
    with pytest.raises(ValueError, match="definite initialization"):
        lower_tenstorrent_ir(frontend, TARGET)


def test_standalone_computation_is_structured_and_compile_pipeline_is_strict():
    frontend = tvm.IRModule({"main": standalone_fill})
    result = lower_tenstorrent_ir(frontend, TARGET)
    assert result.attrs["tt.ir_stage"] == "structured"
    assert "Tensor ABI" in str(result.attrs["tt.deferred_reasons"])
    assert "T.fill(" in result.script()
    with pytest.raises(NotImplementedError, match="structured IR, not executable Device IR"):
        TENSTORRENT_PIPELINE.lower(frontend, TARGET)


def test_supported_computation_still_returns_device_ir():
    result = lower_tenstorrent_ir(tvm.IRModule({"main": program("float32")}), TARGET)
    assert "tt.ir_stage" not in result.attrs
    assert "tt.device_ir_version" in result.attrs
    assert lower_tenstorrent_ir(result, TARGET).same_as(result)


def test_unconsumed_compute_is_rejected_before_host_device_filtering():
    with pytest.raises(NotImplementedError, match="compute block has no Device consumer"):
        TENSTORRENT_PIPELINE.lower(tvm.IRModule({"main": program()}), TARGET)


def test_stage_marker_cannot_bypass_frontend_verification():
    malformed = tvm.IRModule({"main": program(initialize=False)}).with_attr("tt.ir_stage", "structured")
    with pytest.raises(ValueError, match="Cannot re-lower normalized structured IR"):
        lower_tenstorrent_ir(malformed, TARGET)


def _with_allocation_body(func, rewrite):
    def mutate(node):
        if isinstance(node, tirx.SBlock) and node.alloc_buffers:
            return tirx.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                rewrite(node),
                node.init,
                node.alloc_buffers,
                node.match_buffers,
                node.annotations,
            )
        return None

    body = tirx.stmt_functor.ir_transform(func.body, None, mutate, ["tirx.SBlock"])
    return func.with_body(body)


@pytest.mark.parametrize("location", ["bind", "index", "predicate", "loop_min"])
def test_opaque_expression_cannot_hide_behind_structured_fallback(location):
    func = program()
    source = next(iter(func.buffer_map.values()))

    def rewrite(block):
        opaque = tirx.call_extern("int32", "opaque_access", block.alloc_buffers[0].data)
        if location == "loop_min":
            return tirx.For(tirx.Var("k", "int32"), opaque, 1, tirx.ForKind.SERIAL, block.body)
        if location == "index":
            index = tirx.Max(tirx.Min(opaque, 31), 0)
            value = tirx.BufferLoad(source, [index, 0])
        elif location == "predicate":
            value = tirx.BufferLoad(source, [0, 0], opaque > 0)
        else:
            value = opaque
        bind = tirx.Bind(tirx.Var("result", value.dtype), value)
        return tirx.SeqStmt([bind, block.body])

    malformed = _with_allocation_body(func, rewrite)
    with pytest.raises(ValueError, match="unverified effectful expression"):
        lower_tenstorrent_ir(tvm.IRModule({"main": malformed}), TARGET)


@pytest.mark.parametrize("location", ["index", "predicate", "loop_min"])
def test_nested_reads_are_checked_before_initialization(location):
    func = program()
    source = next(iter(func.buffer_map.values()))

    def rewrite(block):
        uninitialized = tirx.BufferLoad(block.alloc_buffers[0], [0, 0])
        if location == "loop_min":
            return tirx.For(
                tirx.Var("k", "int32"),
                tirx.Cast("int32", uninitialized),
                1,
                tirx.ForKind.SERIAL,
                block.body,
            )
        if location == "index":
            index = tirx.Max(tirx.Min(tirx.Cast("int32", uninitialized), 31), 0)
            value = tirx.BufferLoad(source, [index, 0])
        else:
            value = tirx.BufferLoad(source, [0, 0], uninitialized > 0)
        bind = tirx.Bind(tirx.Var("result", value.dtype), value)
        return tirx.SeqStmt([bind, block.body])

    malformed = _with_allocation_body(func, rewrite)
    with pytest.raises(ValueError, match="definite initialization"):
        lower_tenstorrent_ir(tvm.IRModule({"main": malformed}), TARGET)


def test_allocation_shape_cannot_hide_an_opaque_effect():
    extent = tirx.call_extern("int32", "opaque_extent")
    buffer = tirx.decl_buffer([extent], "float32", scope="shared")
    body = tirx.SBlock([], [], [], "root", tirx.Evaluate(0), alloc_buffers=[buffer])
    mod = tvm.IRModule({"main": tirx.PrimFunc([], body)})
    with pytest.raises(ValueError, match="unverified effectful expression"):
        transform.VerifyTTStructuredDataflow()(mod)

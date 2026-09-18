"""Frontend 02 value lifetimes and versioned Device fragment execution."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import lower_tenstorrent_ir, transform
from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_gemm_accumulators import normalize

TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
METADATA = {"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2}


def fragment_program(mode="serial"):
    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations=METADATA)
            c = T.alloc_fragment((32, 32), "float32")
            old = T.alloc_fragment((32, 32), "float32")
            out = T.alloc_shared((32, 32), "float32", annotations=METADATA)
            T.copy(A, a)
            if mode == "branch" or mode == "missing_branch":
                if A[0, 0] > 0:
                    for i, j in T.Tiles(32, 32):
                        c[i, j] = a[i, j]
                else:
                    if mode == "branch":
                        for i, j in T.Tiles(c):
                            c[i, j] = a[i, j] * 2
            elif mode == "fill":
                T.fill(c, 1)
            elif mode != "uninitialized":
                for i, j in T.Tiles((32, 32)):
                    c[i, j] = a[i, j]
            for i, j in T.Tiles(old):
                old[i, j] = c[i, j]
            for _k in T.serial(2):
                for i, j in T.Tiles([32, 32]):
                    c[i, j] = c[i, j] * 2
            for i, j in T.Tiles(out):
                out[i, j] = c[i, j] + old[i, j]
            T.copy(out, C)

    return main


def blocks(mod):
    result = []
    for func in mod.functions.values():
        tirx.stmt_functor.post_order_visit(
            func.body,
            lambda node: result.append(node) if isinstance(node, tirx.SBlock) and "tl.tt.compute_kind" in node.annotations else None,
        )
    return result


@pytest.mark.parametrize("mode", ["serial", "fill"])
def test_fragment_lifetime_lowers_to_immutable_device_values(mode):
    frontend = tvm.IRModule({"main": fragment_program(mode)})
    before = ir.save_json(frontend)
    result = lower_tenstorrent_ir(frontend, TARGET)
    assert int(result.attrs["tt.device_ir_version"]) == 7
    assert len(result.attrs["tt.compute_value_table"]) >= 4
    calls = []
    for func in result.functions.values():
        tirx.stmt_functor.post_order_visit(func.body, lambda node: calls.append(node.op.name) if isinstance(node, tirx.Call) else None)
    assert "tl.tt.compute_value" in calls
    assert ir.save_json(frontend) == before
    ir.assert_structural_equal(ir.load_json(ir.save_json(result)), result)
    transform.VerifyTenstorrentDeviceIR()(result)


def test_dynamic_branch_has_valid_frontend_dominance_but_no_device_control_abi():
    frontend = normalize(fragment_program("branch"))
    verified = transform.VerifyTTStructuredDataflow()(frontend)
    assert ir.structural_equal(verified, frontend)
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match="condition|branch|static|control flow"):
        TenstorrentPassPipelineBody(tvm.IRModule({"main": fragment_program("branch")}), TARGET)


@pytest.mark.parametrize("mode", ["uninitialized", "missing_branch"])
def test_fragment_undefined_old_value_is_rejected(mode):
    with pytest.raises(ValueError, match="definite initialization"):
        lower_tenstorrent_ir(tvm.IRModule({"main": fragment_program(mode)}), TARGET)


def test_fragment_cannot_cross_processor_slots():
    result = normalize(fragment_program())
    result.update_func(result.get_global_var("main"), result["main"].with_attr("tt.kernel_slot", "ncrisc"))
    with pytest.raises(ValueError, match="cannot cross processor slots"):
        transform.VerifyTTStructuredDataflow()(result)


@pytest.mark.parametrize("initialized_before_loop", [False, True])
def test_standalone_dataflow_verifier_rejects_while_initialization(initialized_before_loop):
    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            fragment = T.alloc_fragment((32, 32), "float32")
            if initialized_before_loop:
                T.fill(fragment, 0)
            while A[0, 0] > 0:
                T.fill(fragment, 1)
            T.copy(fragment, C)

    # Exercise the public verifier directly: normal pipeline validation already
    # rejects while loops. Remove only the frontend launch-binding wrapper.
    def remove_launch(node):
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.THREAD_BINDING:
            return node.body
        return None

    body = tirx.stmt_functor.ir_transform(main.body, None, remove_launch, ["tirx.For"])
    mod = tvm.IRModule({"main": main.with_body(body)})
    before = ir.save_json(mod)
    with pytest.raises(ValueError, match="while loops cannot be verified for structured dataflow"):
        transform.VerifyTTStructuredDataflow()(mod)
    assert ir.save_json(mod) == before


def gemm_epilogue(mode="inplace"):
    @T.prim_func
    def main(A: T.Tensor((32, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            b = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            c = T.alloc_fragment((32, 32), "float32")
            out = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            T.copy(A, a)
            T.copy(B, b)
            T.clear(c)
            T.gemm(a, b, c)
            if mode == "overwrite":
                for i, j in T.Tiles(c):
                    c[i, j] = T.float32(1)
            else:
                for i, j in T.Tiles(c):
                    c[i, j] = c[i, j] * T.float32(0.5)
            if mode == "store":
                for i, j in T.Tiles(out):
                    out[i, j] = T.cast(c[i, j], "bfloat16")
                T.copy(out, C)
            else:
                T.copy(c, C)

    return main


@pytest.mark.parametrize("mode", ["inplace", "store"])
def test_gemm_tiles_epilogue_preserves_precision_and_update_relation(mode):
    result = transform.VerifyTTGemmAccumulators()(normalize(gemm_epilogue(mode)))
    (requirement,) = result["main"].attrs["tt.gemm_accumulator_requirements"]
    assert requirement["accum_dtype"].value == "float32"
    assert requirement["output_dtype"].value == "bfloat16"
    assert requirement["dest_precision_requirement"].value == "bits32_required"
    (update,) = requirement["elementwise_updates"]
    assert update["accumulator"].same_as(requirement["accumulator"])
    assert update["relation"].value == "pointwise_old_to_new"
    assert str(update["expression"].dtype) == "float32"
    assert ir.structural_equal(result, transform.VerifyTTGemmAccumulators()(result))
    if mode == "store":
        assert isinstance(blocks(result)[-1].body.value, tirx.Cast)
        assert str(blocks(result)[-1].body.value.dtype) == "bfloat16"
    device = lower_tenstorrent_ir(tvm.IRModule({"main": gemm_epilogue(mode)}), TARGET)
    assert int(device.attrs["tt.device_ir_version"]) == 7
    assert any(int(value.accumulator_id) >= 0 for value in device.attrs["tt.compute_value_table"])


def test_gemm_unrelated_tiles_overwrite_is_rejected():
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match="must not overwrite"):
        lower_tenstorrent_ir(tvm.IRModule({"main": gemm_epilogue("overwrite")}), TARGET)


def test_fragment_program_still_validates_shared_tileops():
    @T.prim_func
    def malformed_transpose(A: T.Tensor((32, 64), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 64), "float32")
            b = T.alloc_shared((32, 64), "float32")
            T.copy(A, a)
            T.transpose(a, b)

    mod = tvm.IRModule({"fragment": fragment_program(), "invalid": malformed_transpose})
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match="Transpose output shape"):
        lower_tenstorrent_ir(mod, TARGET)


def test_fragment_program_rejects_non_primfunc_sibling():
    from tvm import relax

    tensor = relax.Var("tensor", relax.TensorStructInfo((32, 32), "float32"))
    mod = tvm.IRModule({"fragment": fragment_program(), "relax": relax.Function([tensor], tensor)})
    with pytest.raises(ValueError, match="only PrimFunc globals"):
        lower_tenstorrent_ir(mod, TARGET)

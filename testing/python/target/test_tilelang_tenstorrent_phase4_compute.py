"""Hardware-free Phase 4 compute selection and semantic-contract tests."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import transform
from tvm import ir, tirx

TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})


def shared(shape, dtype="float32"):
    return T.alloc_shared(shape, dtype, annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2})


def elementwise_program(*, tiles=False, shape=(64, 64), broadcast=False, inplace=False):
    """Build equivalent frontend spellings, also reusable by Device IR tests."""
    rhs_shape = (32, shape[1]) if broadcast else shape
    iterator = (lambda *extents: T.Tiles(extents)) if tiles else T.Parallel

    @T.prim_func
    def program(A: T.Tensor(shape, "float32"), B: T.Tensor(rhs_shape, "float32"), C: T.Tensor(shape, "float32")):
        with T.Kernel(1, 1, threads=1):
            a = shared(shape)
            b = shared(rhs_shape)
            c = shared(shape)
            T.copy(A, a)
            T.copy(B, b)
            if inplace:
                for i, j in iterator(*shape):
                    a[i, j] = T.sqrt(a[i, j] * a[i, j] + b[0 if broadcast else i, j]) / T.float32(2)
                T.copy(a, C)
            else:
                for i, j in iterator(*shape):
                    c[i, j] = T.sqrt(a[i, j] * a[i, j] + b[0 if broadcast else i, j]) / T.float32(2)
                T.copy(c, C)

    return program


@T.prim_func
def batch_elementwise(A: T.Tensor((2, 64, 64), "float32"), B: T.Tensor((1, 32, 64), "float32"), C: T.Tensor((2, 64, 64), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = shared((2, 64, 64))
        b = shared((1, 32, 64))
        c = shared((2, 64, 64))
        T.copy(A, a)
        T.copy(B, b)
        for k, i, j in T.Parallel(2, 64, 64):
            c[k, i, j] = a[k, i, j] - b[0, 0, j]
        T.copy(c, C)


@T.prim_func
def fill_program(C: T.Tensor((64, 64), "float32")):
    with T.Kernel(1, 1, threads=1):
        c = shared((64, 64))
        T.fill(c, 3.25)
        T.copy(c, C)


@T.prim_func
def cast_program(A: T.Tensor((64, 64), "bfloat16"), C: T.Tensor((64, 64), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = shared((64, 64), "bfloat16")
        c = shared((64, 64))
        T.copy(A, a)
        for i, j in T.Parallel(64, 64):
            c[i, j] = T.Cast("float32", a[i, j])
        T.copy(c, C)


@T.prim_func
def transpose_program(A: T.Tensor((64, 96), "float32"), C: T.Tensor((96, 64), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = shared((64, 96))
        c = shared((96, 64))
        T.copy(A, a)
        T.transpose(a, c)
        T.copy(c, C)


def gemm_program(*, clear=True, transpose_a=False, transpose_b=False, input_dtype="bfloat16", output_dtype="float32"):
    a_shape = (64, 32) if transpose_a else (32, 64)
    b_shape = (96, 64) if transpose_b else (64, 96)

    @T.prim_func
    def program(A: T.Tensor(a_shape, input_dtype), B: T.Tensor(b_shape, input_dtype), C: T.Tensor((32, 96), output_dtype)):
        with T.Kernel(1, 1, threads=1):
            a = shared(a_shape, input_dtype)
            b = shared(b_shape, input_dtype)
            c = shared((32, 96), output_dtype)
            T.copy(A, a)
            T.copy(B, b)
            if not clear:
                T.fill(c, 1)
            T.gemm(a, b, c, transpose_A=transpose_a, transpose_B=transpose_b, clear_accum=clear)
            T.copy(c, C)

    return program


def reduction_program(*, kind="sum", axis=1, clear=True, dtype="float32", output_dtype="float32"):
    output_shape = (64,) if axis == 1 else (96,)

    @T.prim_func
    def program(A: T.Tensor((64, 96), dtype), C: T.Tensor(output_shape, output_dtype)):
        with T.Kernel(1, 1, threads=1):
            a = shared((64, 96), dtype)
            c = shared(output_shape, output_dtype)
            T.copy(A, a)
            if not clear:
                T.fill(c, 1)
            T.reduce(a, c, kind, axis, clear)
            T.copy(c, C)

    return program


def legalize(func):
    mod = tirx.transform.BindTarget(TARGET)(tvm.IRModule({"main": func}))
    for compiler_pass in (
        transform.CanonicalizeTTElementwise(),
        transform.VerifyTTComputeBlocks(),
        transform.ValidateTenstorrentFrontendIR(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
        transform.NormalizeTenstorrentRegions(),
        transform.LegalizeTenstorrentTileOps(),
    ):
        mod = compiler_pass(mod)
    return mod


def calls(mod, name="tl.tt.tile_compute"):
    found = []
    for func in mod.functions.values():
        tirx.stmt_functor.post_order_visit(
            func.body,
            lambda node: (
                found.append(node) if isinstance(node, tirx.Call) and isinstance(node.op, ir.Op) and node.op.name == name else None
            ),
        )
    return found


def test_tiles_parallel_same_consumed_expression_and_maps():
    outputs = [legalize(elementwise_program(tiles=tiles, broadcast=True)) for tiles in (False, True)]
    left, right = (calls(mod)[0] for mod in outputs)
    assert str(left.annotations["tt.compute_kind"].value) == "elementwise"
    assert [[int(axis) for axis in axes] for axes in left.annotations["tt.access_maps"]] == [[0, 1], [-1, 1]]
    # Structural comparison uses free-var mapping because separately parsed
    # programs own distinct Buffer/data identities.
    assert ir.structural_equal(left, right, map_free_vars=True)
    for mod in outputs:
        blocks = []
        tirx.stmt_functor.post_order_visit(
            mod["main"].body,
            lambda node, blocks=blocks: (
                blocks.append(node) if isinstance(node, tirx.SBlock) and "tl.tt.compute_kind" in node.annotations else None
            ),
        )
        assert not blocks


@pytest.mark.parametrize(
    "func,kind", [(batch_elementwise, "elementwise"), (fill_program, "fill"), (cast_program, "typecast"), (transpose_program, "transpose")]
)
def test_builtin_compute_is_consumed(func, kind):
    selected = calls(legalize(func))
    assert len(selected) == 1
    assert str(selected[0].annotations["tt.compute_kind"].value) == kind
    assert str(selected[0].annotations["tt.compute_dtype"].value) == "float32"


@pytest.mark.parametrize(
    "transpose_a,transpose_b,clear", [(False, False, True), (True, False, True), (False, True, False), (True, True, False)]
)
def test_gemm_flags_accumulation_and_read_dependency(transpose_a, transpose_b, clear):
    selected = calls(legalize(gemm_program(clear=clear, transpose_a=transpose_a, transpose_b=transpose_b)))[-1]
    assert str(selected.annotations["tt.compute_kind"].value) == "gemm"
    assert str(selected.annotations["tt.accum_dtype"].value) == "float32"
    assert int(selected.annotations["tt.transpose_a"]) == transpose_a
    assert int(selected.annotations["tt.transpose_b"]) == transpose_b
    assert int(selected.annotations["tt.clear"]) == clear
    assert len(selected.args) == (3 if clear else 4)


@pytest.mark.parametrize("kind", ["sum", "max", "min"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("clear", [True, False])
def test_reduction_axis_clear_and_accumulation(kind, axis, clear):
    selected = calls(legalize(reduction_program(kind=kind, axis=axis, clear=clear)))[-1]
    assert str(selected.annotations["tt.reduce_kind"].value) == kind
    assert int(selected.annotations["tt.reduce_axis"]) == axis
    assert int(selected.annotations["tt.clear"]) == clear
    assert len(selected.args) == (2 if clear else 3)
    assert str(selected.annotations["tt.accum_dtype"].value) == "float32"


def test_inplace_preserves_read_and_write_buffer_identity():
    selected = calls(legalize(elementwise_program(inplace=True)))[0]
    assert selected.args[0].buffer.same_as(selected.args[1].buffer)


def test_gemm_bfloat16_bringup_preserves_precision_requirement():
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": gemm_program(output_dtype="bfloat16")}), TARGET)
    selected = [call for call in calls(mod, "tl.tt.dfb_compute") if call.annotations["tt.compute_kind"].value == "gemm"][0]
    assert selected.annotations["tt.input_dtype"].value == "bfloat16"
    assert selected.annotations["tt.accum_dtype"].value == "bfloat16"
    assert selected.annotations["tt.output_dtype"].value == "bfloat16"
    assert selected.annotations["tt.dest_precision_requirement"].value == "bits16_required"
    assert selected.annotations["tt.matmul_full_fp32"].value == "forbidden"
    transform.VerifyTenstorrentDeviceIR()(mod)


def test_reduction_rejects_unsupported_kind():
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match="sum, max and min"):
        legalize(reduction_program(kind="abssum"))


@T.prim_func
def batch_tiles(A: T.Tensor((2, 64, 64), "float32"), B: T.Tensor((1, 32, 64), "float32"), C: T.Tensor((2, 64, 64), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = shared((2, 64, 64))
        b = shared((1, 32, 64))
        c = shared((2, 64, 64))
        T.copy(A, a)
        T.copy(B, b)
        for k, i, j in T.Tiles((2, 64, 64)):
            c[k, i, j] = a[k, i, j] - b[0, 0, j]
        T.copy(c, C)


def test_batch_tiles_parallel_same_device_selection():
    left, right = (calls(legalize(func))[0] for func in (batch_elementwise, batch_tiles))
    assert ir.structural_equal(left, right, map_free_vars=True)
    assert [[int(axis) for axis in axes] for axes in left.annotations["tt.access_maps"]] == [[0, 1, 2], [-1, -1, 2]]


@pytest.mark.parametrize("kind,accum_dtype", [("sum", "float32"), ("min", "bfloat16"), ("max", "bfloat16")])
def test_bfloat16_reduction_has_validated_device_accumulation(kind, accum_dtype):
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    func = reduction_program(kind=kind, dtype="bfloat16", output_dtype="bfloat16")
    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": func}), TARGET)
    assert int(mod.attrs["tt.device_ir_version"]) == 2
    selected = [call for call in calls(mod, "tl.tt.dfb_compute") if call.annotations["tt.compute_kind"].value == "reduce"][0]
    assert selected.annotations["tt.accum_dtype"].value == accum_dtype
    transform.VerifyTenstorrentDeviceIR()(mod)


@T.prim_func
def copy_cast_program(A: T.Tensor((64, 64), "bfloat16"), C: T.Tensor((64, 64), "float32")):
    with T.Kernel(1, 1, threads=1):
        a = shared((64, 64), "bfloat16")
        c = shared((64, 64))
        T.copy(A, a)
        T.copy(a, c)
        T.copy(c, C)


@T.prim_func
def tiles_fill_program(C: T.Tensor((64, 64), "float32")):
    with T.Kernel(1, 1, threads=1):
        c = shared((64, 64))
        for i, j in T.Tiles((64, 64)):
            c[i, j] = T.float32(3.25)
        T.copy(c, C)


@pytest.mark.parametrize("func,kind", [(copy_cast_program, "typecast"), (tiles_fill_program, "fill")])
def test_alternative_frontend_compute_consumes_to_verified_device_ir(func, kind):
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": func}), TARGET)
    selected = [call for call in calls(mod, "tl.tt.dfb_compute") if call.annotations["tt.compute_kind"].value == kind]
    assert len(selected) == 1
    assert int(mod.attrs["tt.device_ir_version"]) == 2
    assert "tt.ir_stage" not in mod.attrs
    transform.VerifyTenstorrentDeviceIR()(mod)

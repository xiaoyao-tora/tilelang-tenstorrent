"""GEMM fragment precision and complete reduction lifetime, without hardware."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import device_ir as D
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import transform
from tvm import ir, tirx
from testing.python.tenstorrent._support.reference import _calls

TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})


def program(input_dtype="bfloat16", accum_dtype="float32", output_dtype="bfloat16", mode="valid"):
    @T.prim_func
    def main(A: T.Tensor((32, 32), input_dtype), B: T.Tensor((32, 32), input_dtype), C: T.Tensor((32, 32), output_dtype)):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), input_dtype)
            b = T.alloc_shared((32, 32), input_dtype)
            c = T.alloc_fragment((32, 32), accum_dtype)
            T.copy(A, a)
            T.copy(B, b)
            if mode != "uninitialized" and mode != "clear_true":
                T.clear(c)
            if mode == "duplicate_clear":
                T.clear(c)
            if mode == "clear_true":
                T.gemm(a, b, c, clear_accum=True)
            else:
                for _k in T.serial(2):
                    T.gemm(a, b, c, clear_accum=False)
                    if mode == "intermediate_pack":
                        T.copy(c, C)
            if mode == "overwrite":
                T.fill(c, 1)
            if mode != "missing_output":
                T.copy(c, C)
            if mode == "update_after_output":
                T.gemm(a, b, c, clear_accum=False)

    return main


def normalize(func, verify_dataflow=True):
    mod = tirx.transform.BindTarget(TARGET)(tvm.IRModule({"main": func}))
    passes = [
        transform.ValidateTenstorrentFrontendIR(),
        transform.CanonicalizeTTElementwise(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
        transform.NormalizeTenstorrentTopology(),
        transform.NormalizeTenstorrentRegions(),
    ]
    if verify_dataflow:
        passes.append(transform.VerifyTTStructuredDataflow())
    for compiler_pass in passes:
        mod = compiler_pass(mod)
    return mod


def legalize(func):
    return transform.LegalizeTenstorrentTileOps()(normalize(func))


def structured_accumulator_calls(mod):
    result = []

    def collect(node):
        if not isinstance(node, tirx.Call) or node.op.name != "tl.tt.tile_compute":
            return
        kind = node.annotations.get("tt.compute_kind")
        if kind is not None and str(kind.value) in {
            "accumulator_init",
            "gemm_update",
            "accumulator_materialize",
        }:
            result.append(node)

    tirx.stmt_functor.post_order_visit(mod["main"].body, collect)
    return result


@pytest.mark.parametrize(
    "input_dtype,accum_dtype,output_dtype",
    [("bfloat16", "float32", "bfloat16"), ("float32", "float32", "float32"), ("bfloat16", "bfloat16", "bfloat16")],
)
@pytest.mark.parametrize("mode", ["valid", "clear_true"])
def test_fragment_dtype_contract_is_independent_from_storage(input_dtype, accum_dtype, output_dtype, mode):
    mod = legalize(program(input_dtype, accum_dtype, output_dtype, mode))
    init = next(call for call in structured_accumulator_calls(mod) if call.annotations["tt.compute_kind"].value == "accumulator_init")
    accumulator = init.args[0].buffer
    assert accumulator.scope() == "local.fragment"
    assert str(accumulator.dtype) == accum_dtype
    assert init.annotations["tt.input_dtype"].value == input_dtype
    assert init.annotations["tt.accum_dtype"].value == accum_dtype
    assert init.annotations["tt.output_dtype"].value == output_dtype
    assert init.annotations["tt.dest_precision_requirement"].value == ("bits32_required" if accum_dtype == "float32" else "bits16_required")
    assert init.annotations["tt.materialization"].value == "after_complete_k_reduction"
    assert "tt.gemm_accumulator_requirements" not in mod["main"].attrs


def test_reused_fragment_keeps_each_accumulator_lifetime_dtype_contract():
    @T.prim_func
    def reused(
        A: T.Tensor((32, 32), "bfloat16"),
        B: T.Tensor((32, 32), "bfloat16"),
        X: T.Tensor((32, 32), "float32"),
        Y: T.Tensor((32, 32), "float32"),
        C: T.Tensor((32, 32), "bfloat16"),
        D: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16")
            b = T.alloc_shared((32, 32), "bfloat16")
            x = T.alloc_shared((32, 32), "float32")
            y = T.alloc_shared((32, 32), "float32")
            acc = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(X, x)
            T.copy(Y, y)
            T.gemm(a, b, acc, clear_accum=True)
            T.copy(acc, C)
            T.gemm(x, y, acc, clear_accum=True)
            T.copy(acc, D)

    structured = legalize(reused)
    contracts = [
        (
            call.annotations["tt.compute_kind"].value,
            call.annotations["tt.input_dtype"].value,
            call.annotations["tt.accum_dtype"].value,
            call.annotations["tt.output_dtype"].value,
        )
        for call in structured_accumulator_calls(structured)
    ]
    assert contracts == [
        ("accumulator_init", "bfloat16", "float32", "bfloat16"),
        ("gemm_update", "bfloat16", "float32", "bfloat16"),
        ("accumulator_materialize", "bfloat16", "float32", "bfloat16"),
        ("accumulator_init", "float32", "float32", "float32"),
        ("gemm_update", "float32", "float32", "float32"),
        ("accumulator_materialize", "float32", "float32", "float32"),
    ]

    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    device = D.expand_current_device_tables(TenstorrentPassPipelineBody(tvm.IRModule({"main": reused}), TARGET))
    assert [
        (str(entry.input_dtype), str(entry.accumulation_dtype), str(entry.output_dtype)) for entry in device.attrs["tt.accumulator_table"]
    ] == [
        ("bfloat16", "float32", "bfloat16"),
        ("float32", "float32", "float32"),
    ]
    transform.VerifyTenstorrentDeviceIR()(device)


@pytest.mark.parametrize(
    "mode,message",
    [
        ("uninitialized", "definite initialization"),
        ("duplicate_clear", "unique initialization"),
        ("intermediate_pack", "cannot materialize inside"),
        ("overwrite", "initialization requires T.clear"),
        ("missing_output", "final materialization"),
        ("update_after_output", "update after final materialization"),
    ],
)
def test_accumulator_lifetime_is_checked(mode, message):
    with pytest.raises(ValueError, match=message):
        legalize(program(mode=mode))


@pytest.mark.parametrize("dtypes", [("float32", "bfloat16", "float32"), ("bfloat16", "float32", "float32")])
def test_unregistered_dtype_triples_are_rejected(dtypes):
    with pytest.raises(NotImplementedError, match="unsupported program:.*capability registry"):
        legalize(program(*dtypes))


def test_fragment_legalization_preserves_accumulator_operations():
    mod = legalize(program())
    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(restored, mod)
    calls = structured_accumulator_calls(mod)
    kinds = [call.annotations["tt.compute_kind"].value for call in calls]
    assert kinds == ["accumulator_init", "gemm_update", "accumulator_materialize"]
    for call in calls:
        assert call.annotations["tt.input_dtype"].value == "bfloat16"
        assert call.annotations["tt.accum_dtype"].value == "float32"
        assert call.annotations["tt.output_dtype"].value == "bfloat16"


def test_formation_rejects_accumulator_operation_without_dtype_contract():
    mod = legalize(program())

    def remove_output_dtype(node):
        if not isinstance(node, tirx.Call) or node.op.name != "tl.tt.tile_compute":
            return None
        kind = node.annotations.get("tt.compute_kind")
        if kind is None or kind.value != "accumulator_init":
            return None
        annotations = dict(node.annotations.items())
        annotations.pop("tt.output_dtype")
        return tirx.Call(node.dtype, node.op, node.args, annotations, node.span)

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, remove_output_dtype, ["tirx.Call"])
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(body))
    mod = transform.InferTenstorrentTensorLayout()(mod)
    with pytest.raises(ValueError, match="requires 'tt.output_dtype'"):
        transform.FormTenstorrentDeviceProgram()(mod)


def test_legacy_frontend_requirements_do_not_bypass_analysis():
    mod = normalize(program(mode="uninitialized"), verify_dataflow=False)
    mod.update_func(mod.get_global_var("main"), mod["main"].with_attr("tt.gemm_accumulator_requirements", []))
    with pytest.raises(ValueError, match="dominating T.clear"):
        transform.LegalizeTenstorrentTileOps()(mod)


def test_conditional_initializer_does_not_dominate():
    mod = normalize(program())

    def wrap_clear(node):
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call) and node.value.op.name == "tl.tileop.fill":
            return tirx.IfThenElse(tirx.Var("condition", "bool"), node, None)
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, wrap_clear, ["tirx.Evaluate"])
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(body))
    with pytest.raises(NotImplementedError, match="unsupported program:.*conditional access"):
        transform.LegalizeTenstorrentTileOps()(mod)


def test_fragment_pointer_cannot_escape_to_opaque_call():
    mod = normalize(program())
    fragment = next(item.buffer for item in mod["main"].attrs["tt.buffer_metadata_table"] if item.kind == "compute_fragment")
    escape = tirx.Evaluate(tirx.call_extern("int32", "opaque_writer", fragment.data))
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(tirx.SeqStmt([mod["main"].body, escape])))
    with pytest.raises(NotImplementedError, match="unsupported program:.*opaque pointer access"):
        transform.LegalizeTenstorrentTileOps()(mod)


def test_fragment_cannot_be_assigned_to_data_movement_slot():
    mod = normalize(program())
    mod.update_func(mod.get_global_var("main"), mod["main"].with_attr("tt.kernel_slot", "ncrisc"))
    with pytest.raises(NotImplementedError, match="unsupported program:.*cannot cross processor slots"):
        transform.LegalizeTenstorrentTileOps()(mod)


def test_while_loop_cannot_bypass_reduction_lifetime_proof():
    mod = normalize(program())

    def replace_loop(node):
        if isinstance(node, tirx.For) and "tt.logical_core_axis" not in node.annotations:
            return tirx.While(tirx.Var("condition", "bool"), node.body)
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, replace_loop, ["tirx.For"])
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(body))
    with pytest.raises(NotImplementedError, match="unsupported program:.*static positive serial K loops"):
        transform.LegalizeTenstorrentTileOps()(mod)


def test_ordinary_fp32_fragment_does_not_imply_gemm_precision():
    @T.prim_func
    def temporary():
        with T.Kernel(1, 1, threads=1):
            c = T.alloc_fragment((32, 32), "float32")
            T.clear(c)

    mod = normalize(temporary)
    checked = transform.LegalizeTenstorrentTileOps()(mod)
    assert "tt.gemm_accumulator_requirements" not in checked["main"].attrs
    assert not structured_accumulator_calls(checked)
    mod.update_func(mod.get_global_var("main"), mod["main"].with_attr("tt.gemm_accumulator_requirements", [tirx.StringImm("stale")]))
    checked = transform.LegalizeTenstorrentTileOps()(mod)
    assert "tt.gemm_accumulator_requirements" not in checked["main"].attrs


@pytest.mark.parametrize(
    "dtypes",
    [("bfloat16", "bfloat16", "bfloat16"), ("bfloat16", "float32", "bfloat16"), ("float32", "float32", "float32")],
)
@pytest.mark.parametrize("mode,updates", [("valid", 2), ("clear_true", 1)])
def test_full_pipeline_preserves_complete_accumulator_lifetime(dtypes, mode, updates):
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    mod = D.expand_current_device_tables(TenstorrentPassPipelineBody(tvm.IRModule({"main": program(*dtypes, mode=mode)}), TARGET))
    assert str(mod.attrs["tt.ir_stage"]) == "device"
    assert "tt.device_ir_version" not in mod.attrs
    (accumulator,) = mod.attrs["tt.accumulator_table"]
    assert str(accumulator.input_dtype) == dtypes[0]
    assert str(accumulator.accumulation_dtype) == dtypes[1]
    assert str(accumulator.output_dtype) == dtypes[2]
    assert accumulator.full_k_tiles == updates
    calls = device_calls(mod, "trisc")
    lifetime = [call for call in calls if call.op.name in {"tl.tt.accumulator_init", "tl.tt.gemm_update", "tl.tt.accumulator_materialize"}]
    assert [call.op.name for call in lifetime] == [
        "tl.tt.accumulator_init",
        *["tl.tt.gemm_update"] * updates,
        "tl.tt.accumulator_materialize",
    ]
    assert all(int(call.args[2]) == accumulator.accumulator_id for call in lifetime[1:-1])
    assert int(lifetime[0].args[0]) == int(lifetime[-1].args[0]) == accumulator.accumulator_id
    assert len(mod.attrs["tt.dfb_table"]) == 3
    output = next(dfb for dfb in mod.attrs["tt.dfb_table"] if dfb.dfb_id == int(lifetime[-1].args[1]))
    assert str(output.source_buffer_identity) == f"stream.{int(output.dfb_id)}"
    assert str(output.element_dtype) == dtypes[2]
    assert accumulator.accumulator_region.buffer.scope() == "local.fragment"
    assert str(accumulator.accumulator_region.buffer.dtype) == dtypes[1]
    assert not any(call.op.name == "tl.tt.dfb_compute" for call in calls)
    assert ir.structural_equal(mod, transform.VerifyTenstorrentDeviceIR()(mod))


def device_calls(mod, slot):
    calls = []
    for func in mod.functions.values():
        if func.attrs["tt.kernel_slot"] == slot:
            calls.extend(_calls(func.body))
    return calls


def test_serial_k_slices_reuse_two_input_streams():
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    @T.prim_func
    def sliced(A: T.Tensor((32, 96), "bfloat16"), B: T.Tensor((96, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16")
            b = T.alloc_shared((32, 32), "bfloat16")
            c = T.alloc_fragment((32, 32), "float32")
            T.clear(c)
            for k in T.serial(3):
                T.copy(A[:, k * 32 : k * 32 + 32], a)
                T.copy(B[k * 32 : k * 32 + 32, :], b)
                T.gemm(a, b, c)
            T.copy(c, C)

    mod = D.expand_current_device_tables(TenstorrentPassPipelineBody(tvm.IRModule({"main": sliced}), TARGET))
    (accumulator,) = mod.attrs["tt.accumulator_table"]
    assert accumulator.full_k_tiles == 3
    updates = [call for call in device_calls(mod, "trisc") if call.op.name == "tl.tt.gemm_update"]
    assert len(updates) == 3
    assert {int(call.args[index]) for call in updates for index in (0, 1)} == {0, 1}
    assert len(mod.attrs["tt.dfb_table"]) == 3
    copies = [call for call in device_calls(mod, "ncrisc") if call.op.name == "tl.tt.tensor_to_dfb_nd"]
    assert len(copies) == 6
    assert [int(call.args[4]) for call in copies[::2]] == [0, 32, 64]
    assert [int(call.args[2]) for call in copies[1::2]] == [0, 32, 64]


def test_accumulator_alias_cannot_change_identity():
    mod = normalize(program())
    fragment = next(item.buffer for item in mod["main"].attrs["tt.buffer_metadata_table"] if item.kind == "compute_fragment")
    alias = tirx.decl_buffer(fragment.shape, fragment.dtype, "alias", data=fragment.data, scope="local.fragment")

    def replace_accumulator(node):
        if isinstance(node, tirx.Call) and node.op.name == "tl.tileop.gemm":
            region = node.args[2]
            load = region.args[0]
            aliased_region = tirx.Call(
                region.dtype, region.op, [tirx.BufferLoad(alias, load.indices), *region.args[1:]], region.annotations
            )
            return tirx.Call(node.dtype, node.op, [*node.args[:2], aliased_region, *node.args[3:]], node.annotations)
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, replace_accumulator, ["tirx.Call"])
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(body))
    with pytest.raises(ValueError, match="unique compute_fragment"):
        transform.LegalizeTenstorrentTileOps()(mod)


def test_partial_final_materialization_cannot_change_region():
    mod = normalize(program())

    def partial_copy(node):
        if isinstance(node, tirx.Call) and node.op.name == "tl.tileop.copy":
            region = node.args[0]
            if region.buffer.scope() == "local.fragment":
                row = region.indices[0]
                partial = tirx.BufferLoad(
                    region.buffer,
                    [tirx.Ramp(row.base, row.stride, 16), *region.indices[1:]],
                )
                return tirx.Call(node.dtype, node.op, [partial, *node.args[1:]], node.annotations)
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, partial_copy, ["tirx.Call"])
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(body))
    with pytest.raises(ValueError, match="same full BufferRegion"):
        transform.LegalizeTenstorrentTileOps()(mod)


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("tt.dest_precision_requirement", "bits32_required", "hard precision requirement"),
        ("tt.matmul_full_fp32", "required", "forbids matmul_full_fp32"),
        ("tt.input_dtype", "float32", "input/output dtype requirements"),
    ],
)
def test_device_ir_rechecks_bfloat16_precision_contract(key, value, message):
    from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_phase4_compute import gemm_program

    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    mod = D.expand_current_device_tables(TenstorrentPassPipelineBody(tvm.IRModule({"main": gemm_program(output_dtype="bfloat16")}), TARGET))

    def corrupt_requirement(node):
        if isinstance(node, tirx.Call) and node.op.name == "tl.tt.dfb_compute" and node.annotations["tt.compute_kind"].value == "gemm":
            annotations = dict(node.annotations.items())
            annotations[key] = tirx.StringImm(value)
            return tirx.Call(node.dtype, node.op, node.args, annotations)
        return None

    for global_var, func in list(mod.functions.items()):
        if isinstance(func, tirx.PrimFunc):
            body = tirx.stmt_functor.ir_transform(func.body, None, corrupt_requirement, ["tirx.Call"])
            mod.update_func(global_var, func.with_body(body))
    with pytest.raises((ValueError, tvm.error.TVMError), match=message):
        transform.VerifyTenstorrentDeviceIR()(mod)


@pytest.mark.parametrize("keep_fp32_live", [False, True])
def test_compute_kernel_scopes_accumulator_precision_to_live_values(keep_fp32_live):
    @T.prim_func
    def mixed(
        A: T.Tensor((32, 32), "bfloat16"),
        B: T.Tensor((32, 32), "bfloat16"),
        C: T.Tensor((32, 32), "bfloat16"),
        D: T.Tensor((32, 32), "bfloat16"),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16")
            b = T.alloc_shared((32, 32), "bfloat16")
            c = T.alloc_fragment((32, 32), "float32")
            d = T.alloc_fragment((32, 32), "bfloat16")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, clear_accum=True)
            if not keep_fp32_live:
                T.copy(c, C)
            T.gemm(a, b, d, clear_accum=True)
            if keep_fp32_live:
                T.copy(c, C)
            T.copy(d, D)

    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    # Frontend proves individual lifetimes; the actual formed kernel owns the merge.
    assert (
        sum(call.annotations["tt.compute_kind"].value == "accumulator_init" for call in structured_accumulator_calls(legalize(mixed))) == 2
    )
    with pytest.raises((ValueError, NotImplementedError, tvm.error.TVMError), match="precision.*conflicts|conflicting.*destination"):
        TenstorrentPassPipelineBody(tvm.IRModule({"main": mixed}), TARGET)


@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (True, False), (False, True), (True, True)])
def test_accumulator_update_preserves_transpose_and_logical_k(transpose_a, transpose_b):
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    a_shape = (64, 32) if transpose_a else (32, 64)
    b_shape = (96, 64) if transpose_b else (64, 96)

    @T.prim_func
    def transposed(A: T.Tensor(a_shape, "bfloat16"), B: T.Tensor(b_shape, "bfloat16"), C: T.Tensor((32, 96), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared(a_shape, "bfloat16")
            b = T.alloc_shared(b_shape, "bfloat16")
            c = T.alloc_fragment((32, 96), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, transpose_A=transpose_a, transpose_B=transpose_b, clear_accum=True)
            T.copy(c, C)

    mod = D.expand_current_device_tables(TenstorrentPassPipelineBody(tvm.IRModule({"main": transposed}), TARGET))
    (accumulator,) = mod.attrs["tt.accumulator_table"]
    assert accumulator.full_k_tiles == 2
    (update,) = [call for call in device_calls(mod, "trisc") if call.op.name == "tl.tt.gemm_update"]
    assert [int(arg) for arg in update.args[3:]] == [int(transpose_a), int(transpose_b)]


def test_accumulator_materializes_to_shared_output_with_storage_dtype():
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    @T.prim_func
    def shared_output(A: T.Tensor((32, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16")
            b = T.alloc_shared((32, 32), "bfloat16")
            c = T.alloc_fragment((32, 32), "float32")
            output = T.alloc_shared((32, 32), "bfloat16")
            T.copy(A, a)
            T.copy(B, b)
            T.clear(c)
            for _k in T.serial(2):
                T.gemm(a, b, c)
            T.copy(c, output)
            T.copy(output, C)

    mod = D.expand_current_device_tables(TenstorrentPassPipelineBody(tvm.IRModule({"main": shared_output}), TARGET))
    (materialize,) = [call for call in device_calls(mod, "trisc") if call.op.name == "tl.tt.accumulator_materialize"]
    descriptor = next(dfb for dfb in mod.attrs["tt.dfb_table"] if dfb.dfb_id == int(materialize.args[1]))
    assert str(descriptor.element_dtype) == "bfloat16"
    assert mod.attrs["tt.accumulator_table"][0].full_k_tiles == 2


def epilogue_program(mode="fresh", output_dtype="bfloat16"):
    @T.prim_func
    def main(
        A: T.Tensor((32, 32), "bfloat16"),
        B: T.Tensor((32, 32), "bfloat16"),
        C: T.Tensor((32, 32), output_dtype),
        D: T.Tensor((32, 32), "bfloat16"),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations={"tt.dfb_block_count": 2})
            b = T.alloc_shared((32, 32), "bfloat16", annotations={"tt.dfb_block_count": 2})
            acc = T.alloc_fragment((32, 32), "float32")
            old = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, acc, clear_accum=True)
            if mode == "copies":
                T.copy(acc, C)
                T.copy(acc, D)
            else:
                for i, j in T.Tiles(old):
                    old[i, j] = acc[i, j] * T.float32(0.5)
                if mode == "old_new":
                    for i, j in T.Tiles(acc):
                        acc[i, j] = acc[i, j] * T.float32(2)
                    T.copy(acc, D)
                T.copy(old, C)

    return main


@pytest.mark.parametrize("mode", ["fresh", "old_new", "copies"])
def test_accumulator_epilogue_supports_fresh_values_and_multiple_consumers(mode):
    mod = legalize(epilogue_program(mode))
    init = next(call for call in structured_accumulator_calls(mod) if call.annotations["tt.compute_kind"].value == "accumulator_init")
    assert init.annotations["tt.input_dtype"].value == "bfloat16"
    assert init.annotations["tt.accum_dtype"].value == "float32"
    assert init.annotations["tt.output_dtype"].value == "bfloat16"
    assert "tt.gemm_accumulator_requirements" not in mod["main"].attrs
    assert "tl.tt.elementwise" not in mod.script()
    assert "tl.tileop.copy(old" not in mod.script()


def test_derived_accumulator_output_checks_dtype_capability():
    with pytest.raises(NotImplementedError, match="unsupported program:.*capability registry"):
        legalize(epilogue_program(output_dtype="float32"))


def test_both_branch_initializers_dominate_gemm_update():
    mod = normalize(program())

    def branch_clear(node):
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call) and node.value.op.name == "tl.tileop.fill":
            return tirx.IfThenElse(tirx.Var("condition", "bool"), node, node)
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, branch_clear, ["tirx.Evaluate"])
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(body))
    legalized = transform.LegalizeTenstorrentTileOps()(mod)
    assert sum(call.annotations["tt.compute_kind"].value == "accumulator_init" for call in structured_accumulator_calls(legalized)) == 2


def test_branch_without_update_does_not_complete_the_reduction():
    mod = normalize(program())

    def branch_update(node):
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call) and node.value.op.name == "tl.tileop.gemm":
            return tirx.IfThenElse(tirx.Var("condition", "bool"), node, None)
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, branch_update, ["tirx.Evaluate"])
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(body))
    with pytest.raises(ValueError, match="after all updates"):
        transform.LegalizeTenstorrentTileOps()(mod)

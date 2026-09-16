"""GEMM fragment precision and complete reduction lifetime, without hardware."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import transform
from tvm import ir, tirx

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


def normalize(func):
    mod = tirx.transform.BindTarget(TARGET)(tvm.IRModule({"main": func}))
    for compiler_pass in (
        transform.CanonicalizeTTElementwise(),
        transform.ValidateTenstorrentFrontendIR(),
        transform.NormalizeTenstorrentLaunch(),
        transform.NormalizeTenstorrentBufferMetadata(),
    ):
        mod = compiler_pass(mod)
    return mod


def verify(func):
    return transform.VerifyTTGemmAccumulators()(normalize(func))


@pytest.mark.parametrize(
    "input_dtype,accum_dtype,output_dtype",
    [("bfloat16", "float32", "bfloat16"), ("float32", "float32", "float32"), ("bfloat16", "bfloat16", "bfloat16")],
)
@pytest.mark.parametrize("mode", ["valid", "clear_true"])
def test_fragment_dtype_contract_is_independent_from_storage(input_dtype, accum_dtype, output_dtype, mode):
    mod = verify(program(input_dtype, accum_dtype, output_dtype, mode))
    requirements = mod["main"].attrs["tt.gemm_accumulator_requirements"]
    assert len(requirements) == 1
    contract = requirements[0]
    assert contract["accumulator"].scope() == "local.fragment"
    assert str(contract["accumulator"].dtype) == accum_dtype
    assert contract["input_dtype"].value == input_dtype
    assert contract["accum_dtype"].value == accum_dtype
    assert contract["output_dtype"].value == output_dtype
    assert contract["dest_precision_requirement"].value == ("bits32_required" if accum_dtype == "float32" else "bits16_required")
    assert contract["materialization"].value == "after_complete_k_reduction"
    assert ir.structural_equal(mod, transform.VerifyTTGemmAccumulators()(mod))


@pytest.mark.parametrize(
    "mode,message",
    [
        ("uninitialized", "dominating T.clear"),
        ("duplicate_clear", "unique initialization"),
        ("intermediate_pack", "cannot materialize inside"),
        ("overwrite", "initialization requires T.clear"),
        ("missing_output", "final materialization"),
        ("update_after_output", "update after final materialization"),
    ],
)
def test_accumulator_lifetime_is_checked(mode, message):
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match=message):
        verify(program(mode=mode))


@pytest.mark.parametrize("dtypes", [("float32", "bfloat16", "float32"), ("bfloat16", "float32", "float32")])
def test_unregistered_dtype_triples_are_rejected(dtypes):
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match="capability registry"):
        verify(program(*dtypes))


def test_fragment_legalization_preserves_accumulator_operations():
    mod = transform.LegalizeTenstorrentTileOps()(verify(program()))
    kinds = []
    tirx.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: (
            kinds.append(node.annotations["tt.compute_kind"].value)
            if isinstance(node, tirx.Call) and node.op.name == "tl.tt.tile_compute"
            else None
        ),
    )
    assert kinds == ["accumulator_init", "gemm_update", "accumulator_materialize"]


def test_frontend_requirements_do_not_bypass_reverification():
    mod = normalize(program(mode="uninitialized"))
    mod.update_func(mod.get_global_var("main"), mod["main"].with_attr("tt.gemm_accumulator_requirements", []))
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match="dominating T.clear"):
        transform.VerifyTTGemmAccumulators()(mod)


def test_conditional_initializer_does_not_dominate():
    mod = normalize(program())

    def wrap_clear(node):
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call) and node.value.op.name == "tl.tileop.fill":
            return tirx.IfThenElse(tirx.Var("condition", "bool"), node, None)
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, wrap_clear, ["tirx.Evaluate"])
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(body))
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match="conditional access"):
        transform.VerifyTTGemmAccumulators()(mod)


def test_fragment_pointer_cannot_escape_to_opaque_call():
    mod = normalize(program())
    fragment = next(item.buffer for item in mod["main"].attrs["tt.buffer_metadata_table"] if item.kind == "compute_fragment")
    escape = tirx.Evaluate(tirx.call_extern("int32", "opaque_writer", fragment.data))
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(tirx.SeqStmt([mod["main"].body, escape])))
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match="opaque pointer access"):
        transform.VerifyTTGemmAccumulators()(mod)


def test_fragment_cannot_be_assigned_to_data_movement_slot():
    mod = normalize(program())
    mod.update_func(mod.get_global_var("main"), mod["main"].with_attr("tt.kernel_slot", "ncrisc"))
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match="cannot cross processor slots"):
        transform.VerifyTTGemmAccumulators()(mod)


def test_while_loop_cannot_bypass_reduction_lifetime_proof():
    mod = normalize(program())

    def replace_loop(node):
        if isinstance(node, tirx.For) and "tt.logical_core_axis" not in node.annotations:
            return tirx.While(tirx.Var("condition", "bool"), node.body)
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, replace_loop, ["tirx.For"])
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(body))
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match="static positive serial K loops"):
        transform.VerifyTTGemmAccumulators()(mod)


def test_ordinary_fp32_fragment_does_not_imply_gemm_precision():
    @T.prim_func
    def temporary():
        with T.Kernel(1, 1, threads=1):
            c = T.alloc_fragment((32, 32), "float32")
            T.clear(c)

    mod = normalize(temporary)
    checked = transform.VerifyTTGemmAccumulators()(mod)
    assert checked["main"].same_as(mod["main"])
    assert "tt.gemm_accumulator_requirements" not in checked["main"].attrs
    mod.update_func(mod.get_global_var("main"), mod["main"].with_attr("tt.gemm_accumulator_requirements", [tirx.StringImm("stale")]))
    checked = transform.VerifyTTGemmAccumulators()(mod)
    assert "tt.gemm_accumulator_requirements" not in checked["main"].attrs


@pytest.mark.parametrize(
    "dtypes",
    [("bfloat16", "bfloat16", "bfloat16"), ("bfloat16", "float32", "bfloat16"), ("float32", "float32", "float32")],
)
@pytest.mark.parametrize("mode,updates", [("valid", 2), ("clear_true", 1)])
def test_full_pipeline_preserves_complete_accumulator_lifetime(dtypes, mode, updates):
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": program(*dtypes, mode=mode)}), TARGET)
    assert int(mod.attrs["tt.device_ir_version"]) == 5
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
    assert ".materialized" in str(output.source_buffer_identity)
    assert str(output.element_dtype) == dtypes[2]
    assert accumulator.accumulator_region.buffer.scope() == "local.fragment"
    assert str(accumulator.accumulator_region.buffer.dtype) == dtypes[1]
    assert not any(call.op.name == "tl.tt.dfb_compute" for call in calls)
    assert ir.structural_equal(mod, transform.VerifyTenstorrentDeviceIR()(mod))


def device_calls(mod, slot):
    calls = []
    for func in mod.functions.values():
        if func.attrs["tt.kernel_slot"] == slot:
            tirx.stmt_functor.post_order_visit(func.body, lambda node: calls.append(node) if isinstance(node, tirx.Call) else None)
    return calls


def test_serial_k_slices_keep_all_input_generations_live():
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

    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": sliced}), TARGET)
    (accumulator,) = mod.attrs["tt.accumulator_table"]
    assert accumulator.full_k_tiles == 3
    updates = [call for call in device_calls(mod, "trisc") if call.op.name == "tl.tt.gemm_update"]
    assert len(updates) == 3
    assert len({int(call.args[index]) for call in updates for index in (0, 1)}) == 6
    assert len(mod.attrs["tt.dfb_table"]) == 7
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
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match="unique compute_fragment"):
        transform.VerifyTTGemmAccumulators()(mod)


def test_partial_final_materialization_cannot_change_region():
    mod = normalize(program())

    def partial_copy(node):
        if isinstance(node, tirx.Call) and node.op.name == "tl.tileop.copy":
            region = node.args[0]
            if region.args[0].buffer.scope() == "local.fragment":
                partial = tirx.Call(
                    region.dtype, region.op, [*region.args[:2], tirx.IntImm("int32", 16), *region.args[3:]], region.annotations
                )
                return tirx.Call(node.dtype, node.op, [partial, *node.args[1:]], node.annotations)
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, partial_copy, ["tirx.Call"])
    mod.update_func(mod.get_global_var("main"), mod["main"].with_body(body))
    with pytest.raises((NotImplementedError, tvm.error.TVMError), match="same full BufferRegion"):
        transform.VerifyTTGemmAccumulators()(mod)


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("tt.dest_precision_requirement", "bits32_required", "hard precision requirement"),
        ("tt.matmul_full_fp32", "required", "forbids matmul_full_fp32"),
        ("tt.input_dtype", "float32", "input/output dtype requirements"),
    ],
)
def test_device_ir_rechecks_bfloat16_precision_contract(key, value, message):
    from test_tilelang_tenstorrent_phase4_compute import gemm_program

    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": gemm_program(output_dtype="bfloat16")}), TARGET)

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


def test_one_compute_kernel_rejects_conflicting_accumulator_precision():
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
            T.copy(c, C)
            T.gemm(a, b, d, clear_accum=True)
            T.copy(d, D)

    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    # Frontend proves individual lifetimes; the actual formed kernel owns the merge.
    assert len(verify(mixed)["main"].attrs["tt.gemm_accumulator_requirements"]) == 2
    with pytest.raises((ValueError, NotImplementedError, tvm.error.TVMError), match="conflicting.*destination"):
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

    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": transposed}), TARGET)
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

    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": shared_output}), TARGET)
    (materialize,) = [call for call in device_calls(mod, "trisc") if call.op.name == "tl.tt.accumulator_materialize"]
    descriptor = next(dfb for dfb in mod.attrs["tt.dfb_table"] if dfb.dfb_id == int(materialize.args[1]))
    assert str(descriptor.element_dtype) == "bfloat16"
    assert mod.attrs["tt.accumulator_table"][0].full_k_tiles == 2

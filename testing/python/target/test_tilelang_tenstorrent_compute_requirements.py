"""Final-kernel precision contracts are derived from executable Device operations."""

from __future__ import annotations

import pytest
from tvm import IRModule, error, ir, tirx

from tilelang.tenstorrent import transform
from tilelang.tenstorrent.device_ir import AccumulatorDescriptor, ComputeRequirements
from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
from test_tilelang_tenstorrent_gemm_accumulators import TARGET, program


def lower(accum_dtype="float32"):
    return TenstorrentPassPipelineBody(IRModule({"main": program(accum_dtype=accum_dtype)}), TARGET)


def compute(mod):
    return next((gv, func) for gv, func in mod.functions.items() if func.attrs["tt.kernel_slot"] == "trisc")


@pytest.mark.parametrize("dtype,width,full", [("float32", "bits32_required", "required"), ("bfloat16", "bits16_required", "forbidden")])
def test_accumulator_requirements_json_roundtrip(dtype, width, full):
    mod = lower(dtype)
    _, func = compute(mod)
    requirement = func.attrs["tt.compute_requirements"]
    assert isinstance(requirement, ComputeRequirements)
    assert requirement.destination_width == width
    assert requirement.matmul_full_fp32 == full
    assert len(requirement.accumulators) == 1
    assert isinstance(requirement.accumulators[0], AccumulatorDescriptor)
    restored = ir.load_json(ir.save_json(mod))
    assert ir.structural_equal(mod, restored)
    assert ir.structural_hash(mod) == ir.structural_hash(restored)
    assert transform.VerifyTenstorrentDeviceIR()(restored).same_as(restored)
    assert ir.structural_equal(restored, transform.InferTenstorrentComputeRequirements()(restored))


@pytest.mark.parametrize(
    "replacement", [None, ComputeRequirements("bits16_required", "forbidden"), ComputeRequirements("bits32_required", "allowed")]
)
def test_missing_or_forged_requirements_rejected_without_repair(replacement):
    mod = lower()
    gv, func = compute(mod)
    changed = (
        func.without_attr("tt.compute_requirements") if replacement is None else func.with_attr("tt.compute_requirements", replacement)
    )
    mod.update_func(gv, changed)
    before = ir.save_json(mod)
    with pytest.raises((ValueError, error.TVMError), match="tt.compute_requirements"):
        transform.VerifyTenstorrentDeviceIR()(mod)
    assert ir.save_json(mod) == before
    if replacement is not None:
        with pytest.raises((ValueError, error.TVMError), match="tt.compute_requirements"):
            transform.InferTenstorrentComputeRequirements()(mod)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("missing_init", "initialized accumulator"),
        ("duplicate_init", "initialized more than once"),
        ("missing_update", "full-K updates"),
        ("after_materialize", "live initialized accumulator"),
        ("missing_materialize", "final materialization"),
        ("bad_accumulator", "missing accumulator"),
        ("bad_transpose", "transpose flags"),
    ],
)
def test_actual_lifetime_is_reverified(mutation, message):
    mod = lower()
    gv, func = compute(mod)
    statements = list(func.body.seq)
    init = next(stmt for stmt in statements if stmt.value.op.name == "tl.tt.accumulator_init")
    update = next(stmt for stmt in statements if stmt.value.op.name == "tl.tt.gemm_update")
    materialize = next(stmt for stmt in statements if stmt.value.op.name == "tl.tt.accumulator_materialize")
    if mutation == "missing_init":
        statements.remove(init)
    elif mutation == "duplicate_init":
        statements.insert(statements.index(init), init)
    elif mutation == "missing_update":
        statements.remove(update)
    elif mutation == "after_materialize":
        statements.append(update)
    elif mutation == "missing_materialize":
        statements.remove(materialize)
    else:
        args = list(update.value.args)
        args[2 if mutation == "bad_accumulator" else 3] = tirx.IntImm("int32", 99)
        statements[statements.index(update)] = tirx.Evaluate(tirx.Call("void", update.value.op, args))
    mod.update_func(gv, func.with_body(tirx.SeqStmt(statements)).without_attr("tt.compute_requirements"))
    with pytest.raises((ValueError, error.TVMError), match=message):
        transform.InferTenstorrentComputeRequirements()(mod)


def test_descriptor_full_k_cannot_be_forged():
    mod = lower()
    (entry,) = mod.attrs["tt.accumulator_table"]
    fake = AccumulatorDescriptor(
        entry.accumulator_id,
        entry.accumulator_region,
        entry.input_dtype,
        entry.accumulation_dtype,
        entry.output_dtype,
        entry.full_k_tiles + 1,
        entry.source_span,
    )
    mod = mod.with_attr("tt.accumulator_table", [fake])
    gv, func = compute(mod)
    mod.update_func(gv, func.without_attr("tt.compute_requirements"))
    with pytest.raises((ValueError, error.TVMError), match="full-K updates"):
        transform.InferTenstorrentComputeRequirements()(mod)


def test_conflicting_kernels_requirements_are_not_silently_widened():
    mod = lower()
    gv, func = compute(mod)
    (entry,) = mod.attrs["tt.accumulator_table"]
    fragment = tirx.decl_buffer((32, 32), "bfloat16", scope="local.fragment", name="other")
    region = tirx.BufferRegion(fragment, [ir.Range(0, 32), ir.Range(0, 32)])
    second = AccumulatorDescriptor(1, region, "bfloat16", "bfloat16", "bfloat16", 1, entry.source_span)
    mod = mod.with_attr("tt.accumulator_table", [entry, second])
    init = tirx.Evaluate(tirx.call_intrin("void", ir.Op.get("tl.tt.accumulator_init"), 1))
    body = tirx.SeqStmt([*func.body.seq, init])
    mod.update_func(gv, func.with_body(body).without_attr("tt.compute_requirements"))
    with pytest.raises((ValueError, error.TVMError), match="conflicting hard destination width"):
        transform.InferTenstorrentComputeRequirements()(mod)


@pytest.mark.parametrize(
    "dtypes", [("bfloat16", "float32", "float32"), ("float32", "float32", "bfloat16"), ("float32", "bfloat16", "float32")]
)
def test_device_accumulator_dtype_registry_is_enforced(dtypes):
    from tilelang.tenstorrent.device_ir import is_supported_accumulator_dtype_triple

    assert not is_supported_accumulator_dtype_triple(*dtypes)
    mod = lower()
    (entry,) = mod.attrs["tt.accumulator_table"]
    fragment = tirx.decl_buffer((32, 32), dtypes[1], scope="local.fragment")
    region = tirx.BufferRegion(fragment, [ir.Range(0, 32), ir.Range(0, 32)])
    fake = AccumulatorDescriptor(entry.accumulator_id, region, *dtypes, entry.full_k_tiles, entry.source_span)
    mod = mod.with_attr("tt.accumulator_table", [fake])
    with pytest.raises((ValueError, error.TVMError), match="unsupported accumulator dtype triple"):
        transform.InferTenstorrentComputeRequirements()(mod)


@pytest.mark.parametrize("control", ["if", "for"])
def test_inference_does_not_treat_control_flow_as_dominance(control):
    mod = lower()
    gv, func = compute(mod)
    if control == "if":
        body = tirx.IfThenElse(tirx.Var("condition", "bool"), func.body, None)
    else:
        body = tirx.For(tirx.Var("k", "int32"), 0, 2, tirx.ForKind.SERIAL, func.body)
    mod.update_func(gv, func.with_body(body).without_attr("tt.compute_requirements"))
    with pytest.raises((ValueError, error.TVMError), match="schema v5"):
        transform.InferTenstorrentComputeRequirements()(mod)


def test_legacy_dfbs_merge_kernel_width_requirements():
    from tilelang.tenstorrent import language as T

    @T.prim_func
    def mixed(
        A: T.Tensor((32, 32), "bfloat16"),
        B: T.Tensor((32, 32), "bfloat16"),
        C: T.Tensor((32, 32), "bfloat16"),
        D: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16")
            b = T.alloc_shared((32, 32), "bfloat16")
            c = T.alloc_shared((32, 32), "bfloat16")
            d = T.alloc_shared((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, clear_accum=True)
            T.copy(c, C)
            T.gemm(a, b, d, clear_accum=True)
            T.copy(d, D)

    with pytest.raises((ValueError, error.TVMError), match="conflicting hard destination width"):
        TenstorrentPassPipelineBody(IRModule({"main": mixed}), TARGET)

"""Boundary tests run without TT-Lang; integration tests use real pinned MLIR."""

import importlib.util

import pytest

from tilelang import tvm
from tvm.target import Target
from tilelang.tenstorrent import codegen
from tilelang.tenstorrent.ttl_codegen.attributes import fp32_destination
from tilelang.tenstorrent.ttl_codegen.bindings import TTLANG_REVISION, load_bindings
from tilelang.tenstorrent.ttl_codegen.module import check_capability, emit_ttl
from tilelang.tenstorrent.device_ir import ComputeRequirements
from testing.python.target.test_tilelang_tenstorrent_phase2_device_ir import _lower_add


TARGET = Target({"kind": "tenstorrent", "arch": "wormhole_b0"})


def test_structured_gate_precedes_optional_import(monkeypatch):
    mod = tvm.IRModule().with_attr("tt.ir_stage", "structured")
    with pytest.raises(ValueError, match="requires Device IR"):
        codegen.build_ttl_without_compile(mod, TARGET)


def test_target_arch_must_match_device_arch(monkeypatch):
    mod = _lower_add(monkeypatch)
    before = tvm.ir.save_json(mod)
    with pytest.raises(ValueError, match="architecture mismatch"):
        codegen.prepare_ttl_codegen(mod, Target({"kind": "tenstorrent", "arch": "blackhole"}))
    assert tvm.ir.save_json(mod) == before


@pytest.mark.parametrize(
    "width, expected",
    [
        ("unconstrained", None),
        ("bits16_required", False),
        ("bits32_required", True),
    ],
)
def test_destination_mapping_uses_verified_requirement(width, expected):
    assert fp32_destination(ComputeRequirements(width)) is expected


@pytest.mark.parametrize("operation", ["exp", "exp2", "log", "log2", "sqrt", "rsqrt", "tanh", "sin", "cos", "fabs", "floor", "ceil"])
def test_ttl_expression_capability_covers_frontend_unary_operations(operation):
    from tvm import tirx
    from tilelang.tenstorrent.ttl_codegen.expression import check_expression

    load = tirx.Call("float32", tvm.ir.Op.get("tl.tt.compute_value_load"), [tirx.IntImm("int32", 0)])
    expression = tirx.Call("float32", tvm.ir.Op.get("tirx." + operation), [load])
    check_expression(expression)


def test_ttl_expression_capability_rejects_unmapped_semantics():
    from tvm import tirx
    from tilelang.tenstorrent.ttl_codegen.expression import check_expression

    load = tirx.Call("float32", tvm.ir.Op.get("tl.tt.compute_value_load"), [tirx.IntImm("int32", 0)])
    with pytest.raises(NotImplementedError, match="dtype int32"):
        check_expression(tirx.Cast("int32", load))
    with pytest.raises(NotImplementedError, match="no mapping for Select"):
        check_expression(tirx.Select(load > 0, load, -load))


def test_ttl_shared_composite_expression_preserves_v2_capability():
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
    from testing.python.target.test_tilelang_tenstorrent_phase4_compute import elementwise_program

    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": elementwise_program(tiles=True, broadcast=True)}), TARGET)
    assert int(mod.attrs["tt.device_ir_version"]) == 2
    check_capability(mod)


def test_unsupported_pipeline_schema_is_explicit(monkeypatch):
    # Capability probing is independent of the optional compiler. Verification
    # remains mandatory at the public source-generation boundary.
    mod = _lower_add(monkeypatch).with_attr("tt.device_ir_version", 3)
    with pytest.raises(NotImplementedError, match="pipelines and multicore"):
        check_capability(mod)


def test_optional_dependency_diagnostic(monkeypatch):
    from tilelang.tenstorrent.ttl_codegen import bindings

    original = bindings.importlib.import_module

    def missing(name):
        if name == "ttl":
            raise ModuleNotFoundError("No module named ttl")
        return original(name)

    monkeypatch.setattr(bindings.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="simulator-only package is insufficient"):
        load_bindings()


def _real_bindings():
    # Do not replace MLIR builders with mocks: a present but broken/unpinned
    # compiler is a test failure, not a skip.
    if importlib.util.find_spec("ttl") is None:
        pytest.skip(f"Requires TT-Lang compiler {TTLANG_REVISION}")
    return load_bindings()


def test_real_ttl_add_source_roundtrip(monkeypatch):
    bindings = _real_bindings()
    mod = _lower_add(monkeypatch)
    result = codegen.build_ttl_without_compile(mod, TARGET)
    source = result.inspect_source()
    assert "ttl.add" in source
    assert "ttl.copy" in source
    assert "ttl.target_arch" in source
    assert "ttl.logical_kernel" in source
    ctx = bindings.ir.Context()
    bindings.ttl.ensure_dialects_registered(ctx)
    with ctx:
        bindings.ir.Module.parse(source, ctx).operation.verify()


def test_real_ttl_add_compile_only(monkeypatch):
    _real_bindings()
    from ttl.passmanager import PassManager

    module = emit_ttl(_lower_add(monkeypatch))
    pipeline = "builtin.module(ttl-to-ttkernel-pipeline{lower-to-emitc=false})"
    with module.context:
        manager = PassManager.parse(pipeline, context=module.context)
        manager.enable_verifier(True)
        manager.run(module.operation)
        module.operation.verify()
    assert "ttl.add" not in str(module)


@pytest.mark.parametrize("full_fp32_option", ["", " matmul-full-fp32=false"])
def test_real_ttl_bf16_gemm_compile_only(monkeypatch, full_fp32_option):
    _real_bindings()
    from ttl.passmanager import PassManager
    from testing.python.target.test_tilelang_tenstorrent_phase2_device_ir import _context
    from testing.python.target.test_tilelang_tenstorrent_phase4_compute import gemm_program

    mod = _context(monkeypatch).lower(tvm.IRModule({"gemm": gemm_program(output_dtype="bfloat16")}))
    module = emit_ttl(mod)
    assert "ttl.matmul" in str(module)
    assert "fp32_dest_acc_en = false" in str(module)
    with module.context:
        manager = PassManager.parse(
            "builtin.module(ttl-to-ttkernel-pipeline{lower-to-emitc=false" + full_fp32_option + "})",
            context=module.context,
        )
        manager.enable_verifier(True)
        manager.run(module.operation)
        module.operation.verify()
    assert "ttl.matmul" not in str(module)
    assert "fp32_dest_acc_en = false" in str(module)


def test_capability_distinguishes_lower_from_validated_codegen():
    from tilelang.tenstorrent.capabilities import gemm_capability

    supported = gemm_capability("bfloat16", "bfloat16", "bfloat16", arch="wormhole_b0")
    assert supported.device_lower and supported.ttl_mapping
    assert not supported.compile_only_validated and not supported.hardware_validated
    fp32 = gemm_capability("bfloat16", "float32", "bfloat16", arch="blackhole")
    assert fp32.device_lower and not fp32.ttl_mapping
    assert "pack/reload" in fp32.reason
    multiple = gemm_capability("bfloat16", "bfloat16", "bfloat16", arch="wormhole_b0", update_count=2)
    assert multiple.device_lower and not multiple.ttl_mapping
    assert "Multiple K updates" in multiple.reason
    bad = gemm_capability("float32", "bfloat16", "bfloat16", arch="wormhole_b0")
    assert not bad.device_lower


def test_capability_matches_transpose_and_shape_gates():
    from tilelang.tenstorrent.capabilities import gemm_capability

    args = ("bfloat16", "bfloat16", "bfloat16")
    left = gemm_capability(*args, arch="wormhole_b0", transpose_a=True)
    assert left.device_lower and not left.ttl_mapping
    assert "transpose_a" in left.reason
    right = gemm_capability(*args, arch="wormhole_b0", transpose_b=True)
    assert right.ttl_mapping
    bad_shape = gemm_capability(*args, arch="wormhole_b0", tile_shape=(16, 32))
    assert not bad_shape.device_lower and not bad_shape.ttl_mapping


def _lower_fragment(*, accumulation_dtype="bfloat16", multiple_updates=False, transpose_a=False):
    from tilelang.tenstorrent import language as T
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
    from testing.python.target.test_tilelang_tenstorrent_gemm_accumulators import program

    if transpose_a:

        @T.prim_func
        def transposed(A: T.Tensor((64, 32), "bfloat16"), B: T.Tensor((64, 96), "bfloat16"), C: T.Tensor((32, 96), "bfloat16")):
            with T.Kernel(1, 1, threads=1):
                a = T.alloc_shared((64, 32), "bfloat16")
                b = T.alloc_shared((64, 96), "bfloat16")
                c = T.alloc_fragment((32, 96), "bfloat16")
                T.copy(A, a)
                T.copy(B, b)
                T.gemm(a, b, c, transpose_A=True, clear_accum=True)
                T.copy(c, C)

        function = transposed
    else:
        function = program(accum_dtype=accumulation_dtype, mode="valid" if multiple_updates else "clear_true")
    mod = TenstorrentPassPipelineBody(tvm.IRModule({"fragment": function}), TARGET)
    assert int(mod.attrs["tt.device_ir_version"]) == 5
    return mod


@pytest.mark.parametrize(
    "variant, message",
    [
        ({"accumulation_dtype": "float32"}, "FP32/full-K"),
        ({"multiple_updates": True}, "multiple updates"),
        ({"transpose_a": True}, "transpose_a"),
    ],
)
def test_public_codegen_rejects_unsupported_v5_before_compiler_import(monkeypatch, variant, message):
    from tilelang.tenstorrent.ttl_codegen import module

    mod = _lower_fragment(**variant)
    imports = []

    def unexpected_compiler_import():
        imports.append(True)
        raise AssertionError("Unsupported v5 Device IR must be rejected before loading TT-Lang")

    monkeypatch.setattr(module, "load_bindings", unexpected_compiler_import)
    before = tvm.ir.save_json(mod)
    with pytest.raises(NotImplementedError, match=message):
        codegen.build_ttl_without_compile(mod, TARGET)
    assert not imports
    assert tvm.ir.save_json(mod) == before


def test_real_ttl_v5_single_update_source_and_compile_only():
    bindings = _real_bindings()
    from ttl.passmanager import PassManager

    source_module = codegen.build_ttl_without_compile(_lower_fragment(), TARGET)
    source = source_module.inspect_source()
    assert source.count("ttl.matmul") == 1
    assert "fp32_dest_acc_en = false" in source
    assert "ttl.store" in source
    ctx = bindings.ir.Context()
    bindings.ttl.ensure_dialects_registered(ctx)
    with ctx:
        module = bindings.ir.Module.parse(source, ctx)
        module.operation.verify()
        # Exercise the upstream default matmul-full-fp32=true: the explicit
        # bits16 function constraint must still prohibit a Bits32 schedule.
        manager = PassManager.parse("builtin.module(ttl-to-ttkernel-pipeline{lower-to-emitc=false})", context=ctx)
        manager.enable_verifier(True)
        manager.run(module.operation)
        module.operation.verify()
    assert "ttl.matmul" not in str(module)
    assert "fp32_dest_acc_en = false" in str(module)


def _tiles_values(*, input_fragment=False, output_fragment=False, broadcast="identity", cast_output=False):
    from tilelang.tenstorrent import language as T
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    shape = (32, 64)
    rhs_shape = (32, 32) if broadcast in ("column", "scalar") else shape
    output_dtype = "bfloat16" if cast_output else "float32"

    @T.prim_func
    def main(A: T.Tensor(shape, "float32"), B: T.Tensor(rhs_shape, "float32"), C: T.Tensor(shape, output_dtype)):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared(shape, "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2})
            b = T.alloc_shared(rhs_shape, "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2})
            f = T.alloc_fragment(shape, "float32")
            g = T.alloc_fragment(shape, "float32")
            c = T.alloc_shared(shape, output_dtype, annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2})
            T.copy(A, a)
            T.copy(B, b)
            if input_fragment:
                T.copy(a, f)
            if output_fragment:
                for i, j in T.Tiles(shape):
                    g[i, j] = T.sqrt(
                        (f[i, j] if input_fragment else a[i, j]) * T.float32(2)
                        + b[0 if broadcast in ("row", "scalar") else i, 0 if broadcast in ("column", "scalar") else j]
                    )
                T.copy(g, c)
            else:
                for i, j in T.Tiles(shape):
                    c[i, j] = T.Cast(
                        output_dtype,
                        T.sqrt(
                            (f[i, j] if input_fragment else a[i, j]) * T.float32(2)
                            + b[0 if broadcast in ("row", "scalar") else i, 0 if broadcast in ("column", "scalar") else j]
                        ),
                    )
            T.copy(c, C)

    return TenstorrentPassPipelineBody(tvm.IRModule({"main": main}), TARGET)


@pytest.mark.parametrize("input_fragment,output_fragment", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("broadcast", ["identity", "row", "column", "scalar"])
def test_ttl_tiles_scope_and_broadcast_capability(input_fragment, output_fragment, broadcast):
    mod = _tiles_values(input_fragment=input_fragment, output_fragment=output_fragment, broadcast=broadcast)
    before = tvm.ir.save_json(mod)
    check_capability(mod)
    assert tvm.ir.save_json(mod) == before
    if input_fragment or output_fragment:
        compute = next(function for function in mod.functions.values() if str(function.attrs["tt.kernel_slot"]) == "trisc")
        assert fp32_destination(compute.attrs["tt.compute_requirements"]) is True


def test_ttl_fragment_cast_reaches_optional_compiler(monkeypatch):
    from tilelang.tenstorrent.ttl_codegen import module

    mod = _tiles_values(input_fragment=True, output_fragment=True, cast_output=True)
    before = tvm.ir.save_json(mod)
    attempts = []

    def missing():
        attempts.append(True)
        raise ImportError("compiler unavailable")

    monkeypatch.setattr(module, "load_bindings", missing)
    with pytest.raises(ImportError, match="compiler unavailable"):
        codegen.build_ttl_without_compile(mod, TARGET)
    assert attempts == [True]
    assert tvm.ir.save_json(mod) == before


@pytest.mark.parametrize("input_fragment,output_fragment", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("broadcast", ["identity", "row", "column", "scalar"])
def test_real_ttl_tiles_values_compile_only(input_fragment, output_fragment, broadcast):
    _real_bindings()
    from ttl.passmanager import PassManager

    mod = _tiles_values(input_fragment=input_fragment, output_fragment=output_fragment, broadcast=broadcast)
    module = emit_ttl(mod)
    source = str(module)
    assert "ttl.sqrt" in source
    assert "ttl.mul" in source
    assert ("ttl.block.broadcast" in source) == (broadcast != "identity")
    assert "compute_value" not in source
    # Every binding comes from an explicit Device DFB descriptor; fragments
    # add SSA expressions, never provisional CB allocations.
    expected_bindings = sum(len({str(dfb.producer_slot), str(dfb.consumer_slot)}) for dfb in mod.attrs["tt.dfb_table"])
    assert source.count("ttl.bind_cb") == expected_bindings
    with module.context:
        manager = PassManager.parse("builtin.module(ttl-to-ttkernel-pipeline{lower-to-emitc=false})", context=module.context)
        manager.enable_verifier(True)
        manager.run(module.operation)
        module.operation.verify()
    assert "ttl.sqrt" not in str(module)


def test_real_ttl_fragment_explicit_cast_roundtrip():
    _real_bindings()
    module = emit_ttl(_tiles_values(input_fragment=True, output_fragment=True, cast_output=True))
    assert "ttl.typecast" in str(module)
    assert "ttl.store" in str(module)
    module.operation.verify()


def test_real_ttl_passthrough_fragment_retains_input_until_last_consumer():
    _real_bindings()
    from ttl.passmanager import PassManager

    module = emit_ttl(_tiles_values(input_fragment=True, output_fragment=True))
    # Device release ends the direct DFB handle, but a passthrough fragment
    # still borrows its acquired SSA value. Physical release belongs to the
    # compiler's transitive-use analysis, after the final consumer.
    assert "ttl.cb_pop" not in str(module)
    with module.context:
        manager = PassManager.parse("builtin.module(func.func(ttl-insert-cb-sync))", context=module.context)
        manager.enable_verifier(True)
        manager.run(module.operation)
        module.operation.verify()
    compute = next(operation for operation in module.body.operations if "compute" in str(operation.attributes["ttl.kernel_thread"]))
    source = str(compute)
    assert source.index("ttl.sqrt") < source.index("ttl.cb_pop")


def _versioned_values():
    from tilelang.tenstorrent import language as T
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2})
            value = T.alloc_fragment((32, 32), "float32")
            old = T.alloc_fragment((32, 32), "float32")
            result = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            T.copy(a, value)
            T.copy(value, old)
            for i, j in T.Tiles((32, 32)):
                value[i, j] = value[i, j] * T.float32(2)
            for i, j in T.Tiles((32, 32)):
                result[i, j] = old[i, j] + value[i, j]
            T.copy(result, C)

    return TenstorrentPassPipelineBody(tvm.IRModule({"main": main}), TARGET)


def test_ttl_old_and_new_fragment_versions_are_supported():
    check_capability(_versioned_values())


def test_real_ttl_old_fragment_version_is_not_clobbered():
    _real_bindings()
    module = emit_ttl(_versioned_values())
    compute = next(operation for operation in module.body.operations if "compute" in str(operation.attributes["ttl.kernel_thread"]))
    addition = next(operation for operation in compute.body.blocks[0].operations if operation.operation.name == "ttl.add")
    assert addition.operands[0] != addition.operands[1]
    assert addition.operands[0].owner.name == "ttl.attach_cb"
    assert addition.operands[1].owner.name == "ttl.mul"
    module.operation.verify()


def _value_gemm(accum_dtype="bfloat16", updates=1):
    from tilelang.tenstorrent import language as T
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody

    @T.prim_func
    def main(A: T.Tensor((32, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16")
            b = T.alloc_shared((32, 32), "bfloat16")
            acc = T.alloc_fragment((32, 32), accum_dtype)
            result = T.alloc_fragment((32, 32), accum_dtype)
            T.copy(A, a)
            T.copy(B, b)
            T.clear(acc)
            for _ in T.serial(updates):
                T.gemm(a, b, acc)
            for i, j in T.Tiles((32, 32)):
                result[i, j] = acc[i, j] * T.Cast(accum_dtype, 2)
            T.copy(result, C)

    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": main}), TARGET)
    assert int(mod.attrs["tt.device_ir_version"]) == 7
    return mod


@pytest.mark.parametrize("accum_dtype,updates,message", [("float32", 1, "FP32/full-K"), ("bfloat16", 2, "multiple updates")])
def test_v7_gemm_precision_gate_precedes_optional_compiler(monkeypatch, accum_dtype, updates, message):
    from tilelang.tenstorrent.ttl_codegen import module

    mod = _value_gemm(accum_dtype, updates)

    def unexpected():
        raise AssertionError("The precision capability check must run before compiler import")

    monkeypatch.setattr(module, "load_bindings", unexpected)
    before = tvm.ir.save_json(mod)
    with pytest.raises(NotImplementedError, match=message):
        codegen.build_ttl_without_compile(mod, TARGET)
    assert tvm.ir.save_json(mod) == before


def test_v7_bf16_single_update_epilogue_capability():
    check_capability(_value_gemm())


def test_real_ttl_v7_bf16_single_update_epilogue_compile_only():
    _real_bindings()
    from ttl.passmanager import PassManager

    module = emit_ttl(_value_gemm())
    assert "ttl.matmul" in str(module)
    assert "ttl.mul" in str(module)
    assert "fp32_dest_acc_en = false" in str(module)
    with module.context:
        manager = PassManager.parse("builtin.module(ttl-to-ttkernel-pipeline{lower-to-emitc=false})", context=module.context)
        manager.enable_verifier(True)
        manager.run(module.operation)
        module.operation.verify()
    assert "ttl.matmul" not in str(module)
    assert "fp32_dest_acc_en = false" in str(module)


@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
@pytest.mark.parametrize("fragment_input", [False, True])
def test_bf16_broadcast_cannot_share_fp32_destination(monkeypatch, arch, fragment_input):
    from tilelang.tenstorrent import language as T
    from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
    from tilelang.tenstorrent.ttl_codegen import module

    @T.prim_func
    def main(A: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2})
            bf = T.alloc_fragment((32, 32), "bfloat16")
            f = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            if fragment_input:
                T.copy(a, bf)
            for i, j in T.Tiles((32, 32)):
                f[i, j] = T.Cast("float32", bf[0, j] if fragment_input else a[0, j])
            T.copy(f, C)

    target = Target({"kind": "tenstorrent", "arch": arch})
    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": main}), target)

    def unexpected():
        raise AssertionError("The known compiler limitation must be rejected before importing TT-Lang")

    monkeypatch.setattr(module, "load_bindings", unexpected)
    before = tvm.ir.save_json(mod)
    with pytest.raises(NotImplementedError, match="BF16 row/column/scalar broadcast"):
        codegen.build_ttl_without_compile(mod, target)
    assert tvm.ir.save_json(mod) == before

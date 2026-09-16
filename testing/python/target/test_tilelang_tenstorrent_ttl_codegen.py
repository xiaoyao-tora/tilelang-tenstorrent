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

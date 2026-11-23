from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tilelang.tenstorrent import ttlang_adapter


class _Operation:
    def __init__(self):
        self.verify_count = 0

    def verify(self):
        self.verify_count += 1


class _Context:
    def attach_diagnostic_handler(self, _handler):
        return nullcontext()


class _Module:
    def __init__(self):
        self.context = _Context()
        self.operation = _Operation()


class _PassManager:
    def __init__(self):
        self.verifier = None
        self.ran = None

    def enable_verifier(self, enabled):
        self.verifier = enabled

    def run(self, operation):
        self.ran = operation


def _compiler_options(**overrides):
    values = {
        "maximize_dst": True,
        "enable_fpu_binary_ops": True,
        "use_block_matmul": True,
        "subblock_sync": False,
        "combine_pack_tiles": True,
        "reduce_full_fp32": True,
        "matmul_full_fp32": True,
        "strict_f32_acc": False,
        "compiler_dfbs": True,
        "pipe_computed_addresses": True,
        "pipe_capacity_sync": True,
        "pipe_global_semaphores_only": False,
        "pipe_batch_tiles": 0,
        "l1_budget": 0,
        "reuse_user_dfbs": True,
        "unsafe_assume_dfb_allocation_groups": False,
        "dfb_exact_coloring_search_limit": 1_000_000,
        "specialize_cores": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_ttl_api():
    manager = _PassManager()
    pipelines = []
    calls = []

    def parse(pipeline, *, context):
        pipelines.append((pipeline, context))
        return manager

    def compile_kernel(*args, **kwargs):
        calls.append((args, kwargs))
        return "compiled"

    ttl_api = SimpleNamespace(
        PassManager=SimpleNamespace(parse=parse),
        _forward_mlir_warning=object(),
        _resolve_dfb_configs=lambda _module: ["cb"],
        _extract_pipe_sync_semaphore_count=lambda _module: 0,
        _extract_dfb_reset_count=lambda _module: 2,
        _extract_pipe_sram_scratch_bytes=lambda _module: 16,
        _extract_pipe_global_semaphore_count=lambda _module: 3,
        get_ttkernel_names=lambda _module: [
            ("compute", "compute"),
            ("reader", "noc"),
            ("writer", "noc"),
        ],
        _get_kernel_crta_indices=lambda _module, _name: [0],
        _get_kernel_bool_attr=lambda _module, _name, _attribute: False,
        _compile_ttnn_kernel=compile_kernel,
    )
    return ttl_api, manager, pipelines, calls


def test_prebuilt_module_compile_runs_verified_pipeline_and_packages():
    ttl_api, manager, pipelines, calls = _fake_ttl_api()
    module = _Module()
    abi = (SimpleNamespace(effect="input"), SimpleNamespace(effect="output"))
    result = ttlang_adapter._compile_prebuilt_module(
        ttl_api,
        module,
        tensor_abi=abi,
        grid=(2, 3),
        target_arch="wormhole_b0",
        compiler_options=_compiler_options(),
        precision_policy=(("compute", "bits16_required", "forbidden"),),
        program_hash=7,
    )

    assert result == "compiled"
    assert module.operation.verify_count == 2
    assert manager.verifier is True
    assert manager.ran is module.operation
    assert "ttl-to-ttkernel-pipeline" in pipelines[0][0]
    assert "lower-to-emitc=false" in pipelines[0][0]
    assert pipelines[0][0].endswith("},lower-affine,convert-ttkernel-to-emitc,canonicalize)")
    assert "form-expressions" not in pipelines[0][0]
    assert "matmul-full-fp32=false" in pipelines[0][0]
    assert "unsafe-assume-allocation-groups=false" in pipelines[0][0]
    args, kwargs = calls[0]
    assert args[1] == [None, None]
    assert args[2] == (2, 3)
    assert args[3] == 1
    assert kwargs["program_hash"] == 7
    assert kwargs["num_dfb_resets"] == 2


def test_prebuilt_module_compile_rejects_unprovable_required_full_fp32():
    ttl_api, *_ = _fake_ttl_api()
    with pytest.raises(NotImplementedError, match="hard-require API"):
        ttlang_adapter._compile_prebuilt_module(
            ttl_api,
            _Module(),
            tensor_abi=(SimpleNamespace(effect="output"),),
            grid=(1, 1),
            target_arch="blackhole",
            compiler_options=_compiler_options(),
            precision_policy=(("compute", "bits32_required", "required"),),
        )


def test_prebuilt_module_compile_checks_resolved_destination_width():
    ttl_api, *_ = _fake_ttl_api()
    with pytest.raises(RuntimeError, match="violating bits32_required"):
        ttlang_adapter._compile_prebuilt_module(
            ttl_api,
            _Module(),
            tensor_abi=(SimpleNamespace(effect="output"),),
            grid=(1, 1),
            target_arch="blackhole",
            compiler_options=_compiler_options(),
            precision_policy=(("compute", "bits32_required", "allowed"),),
        )

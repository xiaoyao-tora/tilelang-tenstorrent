"""Pinned TT-Lang compiler adapter for in-memory initial TTL modules."""

from __future__ import annotations

import importlib
import hashlib
import json
from importlib import metadata as package_metadata

from .ttl_codegen.bindings import TTLANG_REVISION, load_bindings
from .ttl_codegen.artifact import TenstorrentTTLModuleNode


_REQUIRED_TTL_API = (
    "PassManager",
    "_compile_ttnn_kernel",
    "_extract_dfb_reset_count",
    "_extract_pipe_global_semaphore_count",
    "_extract_pipe_sram_scratch_bytes",
    "_extract_pipe_sync_semaphore_count",
    "_forward_mlir_warning",
    "_get_kernel_bool_attr",
    "_get_kernel_crta_indices",
    "_resolve_dfb_configs",
    "get_ttkernel_names",
)


def stable_program_hash(
    artifact: TenstorrentTTLModuleNode,
    *,
    compiler_options=None,
) -> int:
    payload = {
        "contract": artifact.metadata.contract_digest,
        "dependencies": artifact.metadata.dependency_revisions,
        "compiler_options": repr(compiler_options),
        "packages": {name: _package_version(name) for name in ("tt-lang", "ttnn")},
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _package_version(name: str) -> str:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return "source"


def _compile_prebuilt_module(
    ttl_api,
    module,
    *,
    tensor_abi,
    grid,
    target_arch,
    compiler_options,
    precision_policy=(),
    program_hash=None,
    verbose=False,
):
    tensor_abi = tuple(tensor_abi)
    module.operation.verify()

    full_fp32_policies = {str(entry[2]) for entry in precision_policy}
    unknown_policies = full_fp32_policies - {"required", "forbidden", "allowed"}
    if unknown_policies:
        raise ValueError("invalid matmul_full_fp32 policy: " + ", ".join(sorted(unknown_policies)))
    if "required" in full_fp32_policies:
        raise NotImplementedError(
            "matmul_full_fp32=required needs a TT-Lang hard-require API and "
            "resolved-schedule query; the current boolean is only a preference"
        )
    matmul_full_fp32 = False if "forbidden" in full_fp32_policies else compiler_options.matmul_full_fp32

    options = {
        "lower-to-emitc": False,
        "maximize-dst": compiler_options.maximize_dst,
        "enable-fpu-binary-ops": compiler_options.enable_fpu_binary_ops,
        "use-block-matmul": compiler_options.use_block_matmul,
        "subblock-sync": compiler_options.subblock_sync,
        "combine-pack-tiles": compiler_options.combine_pack_tiles,
        "reduce-full-fp32": compiler_options.reduce_full_fp32,
        "matmul-full-fp32": matmul_full_fp32,
        "strict-f32-acc": compiler_options.strict_f32_acc,
        "compiler-dfbs": compiler_options.compiler_dfbs,
        "pipe-computed-addresses": compiler_options.pipe_computed_addresses,
        "pipe-capacity-sync": compiler_options.pipe_capacity_sync,
        "pipe-global-semaphores-only": compiler_options.pipe_global_semaphores_only,
        "pipe-batch-tiles": compiler_options.pipe_batch_tiles,
        "l1-budget-override": compiler_options.l1_budget,
        "reuse-user-dfbs": compiler_options.reuse_user_dfbs,
        "unsafe-assume-allocation-groups": compiler_options.unsafe_assume_dfb_allocation_groups,
        "exact-coloring-search-limit": compiler_options.dfb_exact_coloring_search_limit,
        "specialize-cores": compiler_options.specialize_cores,
    }
    option_string = " ".join(f"{name}={str(value).lower() if isinstance(value, bool) else value}" for name, value in options.items())
    manager = ttl_api.PassManager.parse(
        # The pinned LLVM form-expressions rewrite indexes duplicate operands
        # incorrectly. EmitC translation also accepts the unformed operations.
        f"builtin.module(ttl-to-ttkernel-pipeline{{{option_string}}},"
        "lower-affine,convert-ttkernel-to-emitc,canonicalize)",
        context=module.context,
    )
    manager.enable_verifier(True)
    with module.context.attach_diagnostic_handler(ttl_api._forward_mlir_warning):
        manager.run(module.operation)
    module.operation.verify()

    compute_kernels = [name for name, thread_type in ttl_api.get_ttkernel_names(module) if thread_type == "compute"]
    for _symbol, destination_width, full_fp32_policy in precision_policy:
        expected_width = {
            "bits16_required": False,
            "bits32_required": True,
            "unconstrained": None,
        }.get(str(destination_width))
        if expected_width is None and str(destination_width) != "unconstrained":
            raise ValueError(f"invalid destination-width policy {destination_width!r}")
        for kernel_name in compute_kernels:
            resolved_width = ttl_api._get_kernel_bool_attr(module, kernel_name, "fp32_dest_acc_en")
            if expected_width is not None and resolved_width != expected_width:
                raise RuntimeError(f"TT-Lang resolved fp32_dest_acc_en={resolved_width} for {kernel_name!r}, violating {destination_width}")
            if str(full_fp32_policy) == "forbidden" and resolved_width:
                raise RuntimeError(f"TT-Lang enabled a 32-bit matmul destination for {kernel_name!r} despite matmul_full_fp32=forbidden")

    pipe_sync_semaphore_count = ttl_api._extract_pipe_sync_semaphore_count(module)
    if pipe_sync_semaphore_count is None:
        raise RuntimeError("compiled module is missing pipe synchronization semaphore metadata")
    kernel_info = ttl_api.get_ttkernel_names(module)
    thread_tensor_indices = [ttl_api._get_kernel_crta_indices(module, name) for name, _ in kernel_info]
    return ttl_api._compile_ttnn_kernel(
        module,
        [None] * len(tensor_abi),
        tuple(map(int, grid)),
        sum(str(entry.effect) in ("output", "inout") for entry in tensor_abi),
        thread_tensor_indices,
        ttl_api._resolve_dfb_configs(module),
        program_hash=program_hash,
        num_pipe_sync_semaphores=pipe_sync_semaphore_count,
        num_dfb_resets=ttl_api._extract_dfb_reset_count(module),
        pipe_sram_scratch_bytes=ttl_api._extract_pipe_sram_scratch_bytes(module),
        num_pipe_global_semaphores=ttl_api._extract_pipe_global_semaphore_count(module),
        target_arch=target_arch,
        operation_name="tilelang",
        verbose=verbose,
    )


def compile_ttl_module(
    artifact: TenstorrentTTLModuleNode,
    *,
    compiler_options=None,
    program_hash: int | None = None,
    verbose: bool = False,
):
    if not isinstance(artifact, TenstorrentTTLModuleNode):
        raise TypeError(f"TT-Lang adapter requires TenstorrentTTLModuleNode, got {type(artifact).__name__}")
    load_bindings()
    ttl_api = importlib.import_module("ttl.ttl_api")
    missing = [name for name in _REQUIRED_TTL_API if not hasattr(ttl_api, name)]
    if missing:
        raise ImportError(
            f"pinned TT-Lang {TTLANG_REVISION} is missing the private compiler ABI required by TileLang: " + ", ".join(missing)
        )
    if compiler_options is None:
        compiler_options = importlib.import_module("ttl.compiler_options").CompilerOptions()
    if program_hash is None:
        program_hash = stable_program_hash(
            artifact,
            compiler_options=compiler_options,
        )
    metadata = artifact.metadata
    with artifact.exclusive_mutation() as module:
        return _compile_prebuilt_module(
            ttl_api,
            module,
            tensor_abi=metadata.tensor_abi,
            grid=metadata.launch_grid,
            target_arch=metadata.target_arch,
            compiler_options=compiler_options,
            precision_policy=metadata.precision_policy,
            program_hash=program_hash,
            verbose=verbose,
        )


__all__ = ["compile_ttl_module", "stable_program_hash"]

"""Read-only capability status for the Lower / optional TTL compiler boundary.

Dtype legality is owned by the native backend. Compiler and hardware validation
are deliberately separate from having an emitter implementation.
"""

from dataclasses import dataclass

from .contracts import SUPPORTED_ARCHITECTURES
from .device_ir import is_supported_accumulator_dtype_triple
from .ttl_codegen.bindings import TTLANG_REVISION


@dataclass(frozen=True)
class GemmCapability:
    device_lower: bool
    ttl_mapping: bool
    compile_only_validated: bool
    hardware_validated: bool
    reason: str
    compiler_revision: str = TTLANG_REVISION


def gemm_capability(
    input_dtype,
    accumulation_dtype,
    output_dtype,
    *,
    arch,
    ir_version=5,
    tile_shape=(32, 32),
    rank=2,
    update_count=1,
    pipelined=False,
    multicore=False,
    transpose_a=False,
    transpose_b=False,
):
    """Describe the implemented static rank-2 accumulator route, without probing hardware."""
    if arch not in SUPPORTED_ARCHITECTURES:
        return GemmCapability(False, False, False, False, f"Unsupported architecture {arch!r}")
    if not is_supported_accumulator_dtype_triple(input_dtype, accumulation_dtype, output_dtype):
        return GemmCapability(False, False, False, False, "Dtype triple is not supported by the native accumulator contract")
    if ir_version != 5 or pipelined or multicore or rank != 2 or tuple(tile_shape) != (32, 32):
        return GemmCapability(
            False, False, False, False, "Accumulator Lower requires v5, single Core, static rank-2 32x32 tiles without pipelines"
        )
    if update_count < 1:
        return GemmCapability(False, False, False, False, "Accumulator requires at least one update")
    if transpose_a:
        return GemmCapability(True, False, False, False, "TTL accumulator GEMM transpose_a mapping is not implemented")
    # The production TTL matmul builder directly supports transpose_rhs.
    if not isinstance(transpose_b, bool):
        raise TypeError("transpose_b must be a bool")
    if str(accumulation_dtype) == "float32":
        return GemmCapability(
            True,
            False,
            False,
            False,
            "FP32/full-K TTL mapping requires a verified persistent-DST schedule with no intermediate BF16 pack/reload",
        )
    if update_count != 1:
        return GemmCapability(True, False, False, False, "Multiple K updates require a verified persistent-DST TTL schedule")
    return GemmCapability(
        True, True, False, False, "Typed TTL mapping implemented; real pinned compiler and hardware validation are still required"
    )

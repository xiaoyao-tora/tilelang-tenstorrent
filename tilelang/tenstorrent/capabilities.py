"""Read-only capability status for the Lower / optional TTL compiler boundary.

Dtype legality is owned by the native backend. Compiler and hardware validation
are deliberately separate from having an emitter implementation.
"""

from dataclasses import dataclass
from enum import Enum

from .contracts import SUPPORTED_ARCHITECTURES
from .device_ir import is_supported_accumulator_dtype_triple
from .ttl_codegen.bindings import TTLANG_REVISION


class CapabilityStatus(str, Enum):
    """Status at the Device Lower boundary, independent of compiler readiness."""

    SUPPORTED = "supported"
    LEGALIZABLE = "legalizable"
    DEFERRED = "deferred"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class LowerCapability:
    status: CapabilityStatus
    reason: str
    device_ir_version: int | None = None

    @property
    def device_lower(self):
        return self.status in (CapabilityStatus.SUPPORTED, CapabilityStatus.LEGALIZABLE)


def lower_capability(
    operation,
    *,
    arch,
    input_dtype="float32",
    output_dtype=None,
    accumulation_dtype=None,
    rank=2,
    shape_class="static_tile_aligned",
    tile_shape=(32, 32),
    memory_layout="interleaved",
    sharded=False,
    compact=True,
    aliased=False,
    multicore=False,
    control_flow="straight_line",
    input_value_kind="shared",
    output_value_kind="shared",
    materialization="exact_dtype",
    pipelined=False,
    loop_carried=False,
    ir_version=None,
    update_count=1,
    reduction_kind="sum",
    keepdims=False,
    transpose_axes="last_two",
):
    """Query the implemented Device Lower envelope without compiling or probing.

    ``rank`` describes the input (output for Fill); Reduction removes one axis.
    ``static_tile_aligned`` assumes positive static extents, 32-divisible final
    dimensions, and matching operation-specific shapes. Elementwise means the
    canonical arithmetic/unary expression subset accepted by legalization.
    Value kinds are ``shared``, ``fragment``, ``mixed`` (shared and fragment
    inputs), or ``accumulator``. GEMM accumulator queries use the last as the
    output kind and describe its final materialization with ``output_dtype``.

    Supported queries still require frontend/dataflow/resource verification.
    ``legalizable`` denotes implemented control-flow normalization;
    ``deferred`` denotes a known form that cannot yet become Device IR. Neither
    status implies a TTL mapping, compiler validation, or hardware validation.
    This function does not replace any native verifier or mutate the registry.

    ``pure_value_branch`` is the narrow runtime branch form with one pure
    full-buffer elementwise assignment per arm, an initialized local scalar
    predicate, and no transfer effects or global predicate reads. An omitted
    else arm requires an initialized old output value.

    Pipelines assume the bounded automatic schedule and independent iterations,
    except the v5/v6 persistent GEMM accumulator across K updates. Set
    ``loop_carried`` for values read from previous iterations. Physical capacity,
    stage counts and per-Core tensor ownership remain verifier checks.
    """
    unsupported = CapabilityStatus.UNSUPPORTED
    deferred = CapabilityStatus.DEFERRED

    def reject(reason, status=unsupported):
        return LowerCapability(status, reason)

    if arch not in SUPPORTED_ARCHITECTURES:
        return reject(f"Unsupported architecture {arch!r}")
    if operation not in ("copy", "elementwise", "fill", "typecast", "transpose", "gemm", "reduce", "pipe"):
        return reject(f"Unknown Lower operation {operation!r}")
    input_dtype = str(input_dtype)
    output_dtype = input_dtype if output_dtype is None else str(output_dtype)
    if input_dtype not in ("bfloat16", "float32") or output_dtype not in ("bfloat16", "float32"):
        return reject("Device Lower storage supports scalar BF16/FP32 only")
    if rank < 1 or tuple(tile_shape) != (32, 32):
        return reject("Device Lower requires positive rank and 32x32 tiles")
    if shape_class != "static_tile_aligned":
        return reject("Device Lower requires positive static tile-aligned shapes; padding/masks are not implemented")
    if memory_layout != "interleaved" or sharded or not compact or aliased:
        return reject("Device Lower requires compact interleaved storage without sharding or cross-Buffer aliases")
    if materialization != "exact_dtype":
        return reject("Only explicit exact-dtype materialization is implemented")
    if input_value_kind not in ("shared", "fragment", "mixed", "accumulator"):
        return reject(f"Unknown input value kind {input_value_kind!r}")
    if output_value_kind not in ("shared", "fragment", "accumulator"):
        return reject(f"Unknown output value kind {output_value_kind!r}")
    accumulator = output_value_kind == "accumulator" or (operation == "gemm" and output_value_kind == "fragment")
    fragment = input_value_kind != "shared" or (output_value_kind == "fragment" and not accumulator)
    input_fragment = input_value_kind != "shared"
    output_rank = rank - 1 if operation == "reduce" else rank
    if (input_fragment and rank != 2) or ((output_value_kind == "fragment" or accumulator) and output_rank != 2):
        return reject("Compute fragments and GEMM accumulators require rank-2 geometry")
    if accumulator and operation != "gemm":
        return reject("Persistent accumulator output is reserved for GEMM")
    if operation == "gemm":
        if rank != 2:
            return reject("GEMM requires rank-2 matrices; batched GEMM is not implemented")
        accumulation_dtype = output_dtype if accumulation_dtype is None else str(accumulation_dtype)
        supported_dtypes = (
            is_supported_accumulator_dtype_triple(input_dtype, accumulation_dtype, output_dtype)
            if accumulator
            else output_dtype == "float32" or input_dtype == output_dtype == "bfloat16"
        )
        if not supported_dtypes:
            return reject("Dtype triple is not supported by the native accumulator contract")
        if output_value_kind == "shared" and accumulation_dtype != output_dtype:
            return reject("Shared-output GEMM accumulation dtype must equal its output dtype")
        if update_count < 1:
            return reject("GEMM requires at least one update")
    elif accumulation_dtype is not None and operation != "reduce":
        return reject("Accumulation dtype applies only to GEMM and Reduction")
    if operation == "transpose" and (rank < 2 or transpose_axes != "last_two" or input_dtype != output_dtype):
        return reject("Transpose requires equal dtypes, rank >= 2, and exchange of the final two axes")
    if operation == "reduce":
        if rank < 2 or keepdims or reduction_kind not in ("sum", "max", "min"):
            return reject("Reduction requires rank >= 2, sum/max/min, and removal of exactly one axis")
        if input_dtype != output_dtype and output_dtype != "float32":
            return reject("Reduction output must use input dtype or float32")
        expected_accumulation = "float32" if reduction_kind == "sum" else output_dtype
        if accumulation_dtype is not None and str(accumulation_dtype) != expected_accumulation:
            return reject(f"{reduction_kind} Reduction uses {expected_accumulation} accumulation")
    if operation == "pipe" and (not multicore or fragment or accumulator):
        return reject("PipeNet transport requires multicore shared DFBs; materialize fragment values before communication")
    if operation == "pipe" and input_dtype != output_dtype:
        return reject("PipeNet transport preserves the shared DFB dtype")
    if control_flow not in ("straight_line", "static_loop", "static_branch", "pure_value_branch", "dynamic_loop", "dynamic_branch"):
        return reject(f"Unknown control-flow placement {control_flow!r}")
    if control_flow in ("dynamic_loop", "dynamic_branch"):
        return reject("Dynamic control flow needs value merges and balanced cross-slot transactions", deferred)
    if control_flow == "pure_value_branch" and operation != "elementwise":
        return reject("Pure value branch legalization requires full-buffer elementwise assignments", deferred)
    if pipelined and accumulator and (fragment or ir_version == 7):
        return reject("Pipelined persistent accumulators require the v5/v6 route without ordinary compute values", deferred)
    if pipelined and loop_carried and not accumulator:
        return reject("Pipelines require independently initialized values; general loop-carried state is deferred", deferred)
    if pipelined and control_flow != "straight_line":
        return reject("Bounded pipelines require independent iterations without nested/conditional transaction sites", deferred)

    repeated_pipe = operation == "pipe" and (pipelined or control_flow == "static_loop")
    if fragment:
        version = 7
    elif multicore and (accumulator or repeated_pipe):
        version = 6
    elif accumulator:
        version = 5
    elif multicore:
        version = 4
    else:
        version = 3 if pipelined else 2
    if ir_version is not None:
        # The accumulator query also describes the accumulator subset of v7.
        allowed = (version, 7) if accumulator else (version,)
        if ir_version not in allowed:
            return reject(f"This Lower route requires Device IR v{version}")
        version = ir_version
    status = CapabilityStatus.LEGALIZABLE if control_flow != "straight_line" else CapabilityStatus.SUPPORTED
    reason = "Implemented Device Lower route; native shape, dataflow, and resource checks still apply"
    if control_flow == "pure_value_branch":
        reason = "Pure local-predicate value branches legalize to Select with fixed transactions and definite initialization checks"
    return LowerCapability(status, reason, version)


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
    lower = lower_capability(
        "gemm",
        arch=arch,
        input_dtype=input_dtype,
        accumulation_dtype=accumulation_dtype,
        output_dtype=output_dtype,
        ir_version=ir_version,
        tile_shape=tile_shape,
        rank=rank,
        update_count=update_count,
        pipelined=pipelined,
        multicore=multicore,
        output_value_kind="accumulator",
    )
    if not lower.device_lower:
        return GemmCapability(False, False, False, False, lower.reason)
    if pipelined:
        return GemmCapability(
            True, False, False, False, "Pipelined accumulator Device Lower is implemented; TTL mapping is not implemented"
        )
    if multicore:
        return GemmCapability(
            True, False, False, False, "Multicore accumulator Device Lower is implemented; TTL mapping is not implemented"
        )
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

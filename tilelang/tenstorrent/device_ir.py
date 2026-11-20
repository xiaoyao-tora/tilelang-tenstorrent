"""Typed Tenstorrent Device IR construction and diagnostic helpers.

The active contract is selected only by ``tt.ir_stage="device"`` and carries
typed descriptor families without a numeric schema version. Versioned helpers
below remain solely for explicit legacy artifact verification and migration.

``TTBufferMetadata`` is a normalization-stage object.  It is attached as an
array under :data:`BUFFER_METADATA_TABLE_ATTR` before program formation and
must be consumed before :func:`attach_device_module_metadata` produces final
Device TIR.

The reflected descriptor constructors retain their v1 defaults. Pipeline v3
adds module attributes for storage groups and iteration relations; these must
be attached alongside the common metadata before invoking the verifier.
Multicore v4 adds ``tt.pipe_transfer_table`` containing typed point deliveries
and orders all Core/slot functions by global name in ``tt.kernel_order``.
Schema v5 adds ``tt.accumulator_table`` and mandatory typed
``tt.compute_requirements`` on TRISC. Persistent fragments have one init, the
complete static K update sequence, and one final materialization. Bounded
pipeline metadata composes with this lifetime without packing partial K results.
Schema v6 combines specialized
Core/slot functions with persistent accumulators and static PipeNet occurrence
epochs. Each accumulator belongs to the unique TRISC function containing its
lifetime; fragment Buffer identities are local to that Core. Forwarded Tensor
panels become ready for their local TRISC consumer only after all outgoing Pipe
transfers complete. Earlier metadata constructors and defaults remain unchanged.
Schema v7 adds immutable ``tt.compute_value_table`` definitions, ordinary fragment
expressions, GEMM epilogues and explicit compute materialization. Values and
fragment Buffer identities belong to exactly one TRISC function; topology v7
also carries ``tt.pipe_transfer_table`` (possibly empty). Pipeline storage groups
carry one generation per iteration, or one prologue/epilogue resource with the
relation ``[-1, 0]``; singleton and iterative resources never share a group.

Schema v8 adds explicit ``compute_precision(0|1)`` regions for 16/32-bit DST.
TRISC ``tt.compute_region_requirements`` carries one typed requirement per
region. Aggregate ``tt.compute_requirements.destination_width`` is
``region_scoped``; it must not be interpreted as one physical kernel setting.
Values cross boundaries only through completed exact-dtype DFB snapshots and
new identity definitions. Earlier codegen consumers must reject version 8.

Schema v9 represents repeated v7/v8 schedules using serial loops and typed
descriptor families. ``expand_device_ir`` losslessly restores the original
schema; the existing Device verifier checks that expanded schedule and returns
the compact input. Older consumers must explicitly expand or reject version 9.
"""

from __future__ import annotations

import ast
import io
import tokenize
from typing import Any

import tvm_ffi
from tvm import DataType, IRModule, tirx
from tvm.ir import CallingConv, Node, Span
from tvm.target import Target

from . import _ffi_api
from .contracts import DEVICE_IR_VERSION, KERNEL_SLOT_ORDER

DEVICE_IR_VERSION_ATTR = "tt.device_ir_version"
TARGET_ARCH_ATTR = "tt.target_arch"
LAUNCH_GRID_ATTR = "tt.launch_grid"
OPERATION_IDENTITY_ATTR = "tt.operation_identity"
TENSOR_TABLE_ATTR = "tt.tensor_table"
ACCUMULATOR_TABLE_ATTR = "tt.accumulator_table"
COMPUTE_VALUE_TABLE_ATTR = "tt.compute_value_table"
COMPUTE_REQUIREMENTS_ATTR = "tt.compute_requirements"
COMPUTE_REGION_REQUIREMENTS_ATTR = "tt.compute_region_requirements"
DFB_TABLE_ATTR = "tt.dfb_table"
DFB_FAMILIES_ATTR = "tt.dfb_families"
DFB_STORAGE_GROUPS_ATTR = "tt.dfb_storage_groups"
DFB_STORAGE_GROUP_RELATIONS_ATTR = "tt.dfb_storage_group_relations"
ACCUMULATOR_FAMILIES_ATTR = "tt.accumulator_families"
COMPUTE_VALUE_FAMILIES_ATTR = "tt.compute_value_families"
TRANSPORT_FAMILIES_ATTR = "tt.transport_families"
PIPE_TABLE_ATTR = "tt.pipe_table"
PIPE_TRANSFER_TABLE_ATTR = "tt.pipe_transfer_table"
COMMUNICATION_TABLE_ATTR = "tt.communication_table"
TRANSPORT_TABLE_ATTR = "tt.transport_table"
KERNEL_ORDER_ATTR = "tt.kernel_order"
KERNEL_SLOT_ATTR = "tt.kernel_slot"
KERNEL_THREAD_ATTR = "tt.kernel_thread"
NOC_INDEX_ATTR = "tt.noc_index"
LOGICAL_KERNEL_ATTR = "tt.logical_kernel"
TENSOR_ARG_INDICES_ATTR = "tt.tensor_arg_indices"
CORE_DOMAIN_ATTR = "tt.core_domain"
BUFFER_METADATA_TABLE_ATTR = "tt.buffer_metadata_table"
PIPELINE_RELATIONS_ATTR = "tt.pipeline_relations"
PIPELINE_STAGES_ATTR = "tt.pipeline_stages"
PIPELINE_EXTENT_ATTR = "tt.pipeline_extent"
L1_CAPACITY_BYTES_ATTR = "tt.l1_capacity_bytes"
COMPACT_ORIGINAL_VERSION_ATTR = "tt.compact_original_version"
COMPACT_DFB_FAMILIES_ATTR = "tt.compact_dfb_families"
COMPACT_VALUE_FAMILIES_ATTR = "tt.compact_value_families"


@tvm_ffi.register_object("tl.tenstorrent.DeviceIRIntColumn")
class DeviceIRIntColumn(Node):
    """Exact affine, periodic, or explicit integer column for a resource family."""

    def __init__(self, start: int, step: int, count: int, values=(), period: int = 0):
        self.__init_handle_by_constructor__(_ffi_api.DeviceIRIntColumn, start, step, count, list(values), period)


@tvm_ffi.register_object("tl.tenstorrent.DeviceIRDescriptorFamily")
class DeviceIRDescriptorFamily(Node):
    """Distinct resource rows sharing one immutable descriptor prototype."""

    def __init__(self, prototype, positions: DeviceIRIntColumn, fields, source_parts=()):
        self.__init_handle_by_constructor__(_ffi_api.DeviceIRDescriptorFamily, prototype, positions, fields, list(source_parts))


@tvm_ffi.register_object("tl.tenstorrent.DFBStorageGroupRelation")
class DFBStorageGroupRelation(Node):
    """Family-aligned storage-group ID and transaction-ordinal columns."""

    def __init__(
        self,
        dfb_family_id: int,
        storage_group_ids: DeviceIRIntColumn,
        transaction_ordinals: DeviceIRIntColumn,
        pipeline_iterations: DeviceIRIntColumn | None = None,
    ):
        if pipeline_iterations is None:
            pipeline_iterations = DeviceIRIntColumn(-1, 0, int(storage_group_ids.count))
        self.__init_handle_by_constructor__(
            _ffi_api.DFBStorageGroupRelation,
            int(dfb_family_id),
            storage_group_ids,
            transaction_ordinals,
            pipeline_iterations,
        )


def compact_device_ir(mod: IRModule, *, force: bool = False) -> IRModule:
    """Use the existing Lower representation helper without creating a pass.

    The default path only compacts large v7/v8 static schedules. ``force`` is
    useful for inspecting and testing small schedules. Validate the returned
    module with ``VerifyTenstorrentDeviceIR`` before consuming untrusted IR.
    """
    return _ffi_api.CompactDeviceIR(mod, force)


def expand_device_ir(mod: IRModule) -> IRModule:
    """Losslessly expand v9 into v7/v8, with bounded allocation and arithmetic.

    This representation helper does not replace semantic Device verification.
    Noncompact modules pass through unchanged; compact metadata in an older
    schema is rejected.
    """
    return _ffi_api.ExpandDeviceIR(mod)


def expand_current_device_tables(mod: IRModule) -> IRModule:
    """Materialize current-stage family instances for bounded inspection.

    Lower, Codegen and serialization keep the parameterized representation.
    This helper exists for tests, diagnostics and verifier-equivalent audits.
    """
    return _ffi_api.ExpandCurrentDeviceTables(mod)


def device_ir_to_tirx(mod: IRModule) -> str:
    """Return parseable TVMScript for current Device IR.

    TVM's generic printer emits ``tirx.Call.annotations`` as keyword arguments.
    Tenstorrent annotation names use dotted ``tt.*`` keys, which are valid IR
    keys but invalid Python keyword identifiers. Encode only those keyword
    tokens at the text boundary; ``T.tt`` constructors decode them while parsing.
    """

    if mod.attrs is None or str(mod.attrs.get("tt.ir_stage", "")) != "device":
        raise ValueError("device_ir_to_tirx requires tt.ir_stage='device'")

    source = mod.script(show_meta=True)
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    encoded = []
    index = 0
    while index < len(tokens):
        if (
            index + 3 < len(tokens)
            and tokens[index].type == tokenize.NAME
            and tokens[index].string == "tt"
            and tokens[index + 1].type == tokenize.OP
            and tokens[index + 1].string == "."
            and tokens[index + 2].type == tokenize.NAME
            and tokens[index + 3].type == tokenize.OP
            and tokens[index + 3].string == "="
        ):
            first = tokens[index]
            name = tokens[index + 2]
            encoded.append(
                tokenize.TokenInfo(
                    tokenize.NAME,
                    f"tt__{name.string}",
                    first.start,
                    name.end,
                    first.line,
                )
            )
            index += 3
            continue
        encoded.append(tokens[index])
        index += 1

    result = tokenize.untokenize(encoded)
    legend = (
        "# Device ABI legend: calling_conv=2 is DEVICE_KERNEL_LAUNCH; "
        "tt.noc_index=0/1 selects NOC0/NOC1; tt.tensor_arg_indices indexes "
        "tt.tensor_table.\n"
    )
    result = legend + result
    ast.parse(result)
    return result


def is_supported_accumulator_dtype_triple(input_dtype, accumulation_dtype, output_dtype) -> bool:
    """Query the shared native Lower registry; this does not imply TTL schedule support."""
    return bool(_ffi_api.IsSupportedAccumulatorDTypeTriple(DataType(input_dtype), DataType(accumulation_dtype), DataType(output_dtype)))


def _expr(value: Any):
    return value if isinstance(value, tirx.PrimExpr) else tirx.IntImm("int64", int(value))


def _shape(values) -> list:
    return [_expr(value) for value in values]


@tvm_ffi.register_object("tl.tenstorrent.CoreCoord")
class CoreCoord(Node):
    def __init__(self, x: int, y: int):
        self.__init_handle_by_constructor__(_ffi_api.CoreCoord, int(x), int(y))


@tvm_ffi.register_object("tl.tenstorrent.CoreDomain")
class CoreDomain(Node):
    def __init__(self, begin: CoreCoord | tuple[int, int], end: CoreCoord | tuple[int, int]):
        begin = begin if isinstance(begin, CoreCoord) else CoreCoord(*begin)
        end = end if isinstance(end, CoreCoord) else CoreCoord(*end)
        self.__init_handle_by_constructor__(_ffi_api.CoreDomain, begin, end)


@tvm_ffi.register_object("tl.tenstorrent.ShardSpec")
class ShardSpec(Node):
    def __init__(self, core_domain: CoreDomain, shard_shape, orientation: str):
        self.__init_handle_by_constructor__(_ffi_api.ShardSpec, core_domain, _shape(shard_shape), orientation)


@tvm_ffi.register_object("tl.tenstorrent.TensorBacking")
class TensorBacking(Node):
    def __init__(self, global_arg_index: int, byte_offset=0):
        self.__init_handle_by_constructor__(_ffi_api.TensorBacking, int(global_arg_index), _expr(byte_offset))


@tvm_ffi.register_object("tl.tenstorrent.TensorStorageBacking")
class TensorStorageBacking(Node):
    """Proven Tensor byte range used as storage for a complete DFB capacity."""

    def __init__(self, global_arg_index: int, byte_offset, byte_size):
        self.__init_handle_by_constructor__(
            _ffi_api.TensorStorageBacking,
            int(global_arg_index),
            _expr(byte_offset),
            _expr(byte_size),
        )


@tvm_ffi.register_object("tl.tenstorrent.OperationIdentity")
class OperationIdentity(Node):
    def __init__(self, operation_id: str, source_span: Span):
        self.__init_handle_by_constructor__(_ffi_api.OperationIdentity, operation_id, source_span)


@tvm_ffi.register_object("tl.tenstorrent.TensorDescriptor")
class TensorDescriptor(Node):
    def __init__(
        self,
        global_arg_index: int,
        shape,
        dtype: str | DataType,
        strides,
        tile_shape,
        tile_grid_shape,
        memory_space: str,
        memory_layout: str,
        shard_spec: ShardSpec | None,
        effect: str,
        alias_group: int,
        source_span: Span,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.TensorDescriptor,
            int(global_arg_index),
            _shape(shape),
            DataType(dtype),
            _shape(strides),
            _shape(tile_shape),
            _shape(tile_grid_shape),
            memory_space,
            memory_layout,
            shard_spec,
            effect,
            int(alias_group),
            source_span,
        )


@tvm_ffi.register_object("tl.tenstorrent.NormalizedTopologyDescriptor")
class NormalizedTopologyDescriptor(Node):
    """Structured-stage facts for one high-level communication site."""

    def __init__(
        self,
        site_id,
        kind,
        source_region,
        destination_region,
        source_coord,
        group_begin,
        group_end,
        parameter,
        contract,
        guard,
        occurrence_extents=(),
        source_span=None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.NormalizedTopologyDescriptor,
            int(site_id),
            kind,
            source_region,
            destination_region,
            list(source_coord),
            list(group_begin),
            list(group_end),
            parameter,
            contract,
            guard,
            list(occurrence_extents),
            source_span,
        )


@tvm_ffi.register_object("tl.tenstorrent.CommunicationDescriptor")
class CommunicationDescriptor(Node):
    def __init__(
        self,
        communication_id,
        kind,
        algorithm,
        source_region,
        destination_region,
        source_coord,
        group_begin,
        group_end,
        parameter,
        contract,
        occurrence_extents=(),
        source_span=None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.CommunicationDescriptor,
            int(communication_id),
            kind,
            algorithm,
            source_region,
            destination_region,
            list(source_coord),
            list(group_begin),
            list(group_end),
            parameter,
            contract,
            list(occurrence_extents),
            source_span,
        )


@tvm_ffi.register_object("tl.tenstorrent.DFBDescriptor")
class DFBDescriptor(Node):
    def __init__(
        self,
        dfb_id: int,
        source_buffer_identity: str,
        element_dtype: str | DataType,
        tile_shape,
        block_shape_in_tiles,
        block_count,
        tensor_backing: TensorBacking | None,
        producer_slot: str,
        producer_domain: CoreDomain,
        consumer_slot: str,
        consumer_domain: CoreDomain,
        transaction_count_or_loop_relation,
        source_span: Span,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.DFBDescriptor,
            int(dfb_id),
            source_buffer_identity,
            DataType(element_dtype),
            _shape(tile_shape),
            _shape(block_shape_in_tiles),
            _expr(block_count),
            tensor_backing,
            producer_slot,
            producer_domain,
            consumer_slot,
            consumer_domain,
            _expr(transaction_count_or_loop_relation),
            source_span,
        )


@tvm_ffi.register_object("tl.tenstorrent.DFBStorageGroupDescriptor")
class DFBStorageGroupDescriptor(Node):
    """One finite logical CB stream shared by compatible DFB generations."""

    def __init__(
        self,
        storage_group_id: int,
        element_dtype: str | DataType,
        tile_shape,
        block_shape_in_tiles,
        block_count,
        producer_slot: str,
        producer_domain: CoreDomain,
        consumer_slot: str,
        consumer_domain: CoreDomain,
        transaction_count_or_loop_relation,
        storage_backing: TensorStorageBacking | None,
        source_span: Span,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.DFBStorageGroupDescriptor,
            int(storage_group_id),
            DataType(element_dtype),
            _shape(tile_shape),
            _shape(block_shape_in_tiles),
            _expr(block_count),
            producer_slot,
            producer_domain,
            consumer_slot,
            consumer_domain,
            _expr(transaction_count_or_loop_relation),
            storage_backing,
            source_span,
        )


@tvm_ffi.register_object("tl.tenstorrent.AccumulatorDescriptor")
class AccumulatorDescriptor(Node):
    """One persistent fragment contract.

    In current shared-grid Device IR, ``accumulator_id`` is local to the active
    Core; the family Core domain and occurrence relation identify the instance.
    """

    def __init__(self, accumulator_id, accumulator_region, input_dtype, accumulation_dtype, output_dtype, full_k_tiles, source_span):
        self.__init_handle_by_constructor__(
            _ffi_api.AccumulatorDescriptor,
            int(accumulator_id),
            accumulator_region,
            DataType(input_dtype),
            DataType(accumulation_dtype),
            DataType(output_dtype),
            int(full_k_tiles),
            source_span,
        )


@tvm_ffi.register_object("tl.tenstorrent.ComputeValueDescriptor")
class ComputeValueDescriptor(Node):
    """Immutable fragment version with Core-local current-stage IDs.

    ``previous_value_id`` and ``accumulator_id`` use -1 when absent. Versioned
    legacy artifacts retain their historical globally unique ID contract.
    """

    def __init__(self, value_id, buffer, version, previous_value_id, accumulator_id, source_span):
        self.__init_handle_by_constructor__(
            _ffi_api.ComputeValueDescriptor,
            int(value_id),
            buffer,
            int(version),
            int(previous_value_id),
            int(accumulator_id),
            source_span,
        )


@tvm_ffi.register_object("tl.tenstorrent.ComputeRequirements")
class ComputeRequirements(Node):
    """Hard destination and full-K precision requirements of one compute kernel."""

    def __init__(self, destination_width="unconstrained", matmul_full_fp32="allowed", accumulators=()):
        self.__init_handle_by_constructor__(
            _ffi_api.ComputeRequirements,
            destination_width,
            matmul_full_fp32,
            list(accumulators),
        )


@tvm_ffi.register_object("tl.tenstorrent.PipeDescriptor")
class PipeDescriptor(Node):
    def __init__(
        self,
        pipe_net_id: int,
        communication_id: int,
        record_index: int,
        src_coord: CoreCoord,
        dst_begin: CoreCoord,
        dst_end: CoreCoord,
        contract: str,
        payload_dfb_id: int,
        source_span: Span,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.PipeDescriptor,
            int(pipe_net_id),
            int(communication_id),
            int(record_index),
            src_coord,
            dst_begin,
            dst_end,
            contract,
            int(payload_dfb_id),
            source_span,
        )


@tvm_ffi.register_object("tl.tenstorrent.PipeTransferDescriptor")
class PipeTransferDescriptor(Node):
    """One v4 record delivery to a destination Core, with explicit DFB IDs."""

    def __init__(
        self,
        transfer_id: int,
        pipe_net_id: int,
        communication_id: int,
        record_index: int,
        occurrence: int,
        src_coord: CoreCoord,
        dst_coord: CoreCoord,
        source_dfb_id: int,
        destination_dfb_id: int,
        transaction_count: int,
        source_span: Span,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.PipeTransferDescriptor,
            int(transfer_id),
            int(pipe_net_id),
            int(communication_id),
            int(record_index),
            int(occurrence),
            src_coord,
            dst_coord,
            int(source_dfb_id),
            int(destination_dfb_id),
            int(transaction_count),
            source_span,
        )


@tvm_ffi.register_object("tl.tenstorrent.LogicalKernel")
class LogicalKernel(Node):
    def __init__(self, kernel_id: str, kind: str, role: str, source_span: Span):
        self.__init_handle_by_constructor__(_ffi_api.LogicalKernel, kernel_id, kind, role, source_span)


@tvm_ffi.register_object("tl.tenstorrent.TTBufferMetadata")
class TTBufferMetadata(Node):
    """Typed Form-input metadata associated with a Buffer handle."""

    def __init__(
        self,
        buffer_id: str,
        buffer: tirx.Buffer,
        kind: str,
        global_arg_index: int | None = None,
        tile_shape=(),
        tile_grid_shape=(),
        memory_layout: str = "",
        shard_spec: ShardSpec | None = None,
        dfb_block_count=None,
        tensor_backing: TensorBacking | None = None,
        alias_of: str | None = None,
        tile_shape_origin: str = "unset",
        block_count_origin: str = "unset",
        tensor_backing_origin: str = "unset",
        layout_origin: str = "unset",
        source_span: Span | None = None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.TTBufferMetadata,
            buffer_id,
            buffer,
            kind,
            global_arg_index,
            _shape(tile_shape),
            _shape(tile_grid_shape),
            memory_layout,
            shard_spec,
            None if dfb_block_count is None else _expr(dfb_block_count),
            tensor_backing,
            alias_of,
            tile_shape_origin,
            block_count_origin,
            tensor_backing_origin,
            layout_origin,
            source_span,
        )


@tvm_ffi.register_object("tl.tenstorrent.DeviceModuleMetadata")
class DeviceModuleMetadata(Node):
    def __init__(
        self,
        target_arch: str,
        launch_grid: CoreCoord,
        operation_identity: OperationIdentity,
        tensor_table=(),
        dfb_table=(),
        pipe_table=(),
        *,
        device_ir_version: int = DEVICE_IR_VERSION,
        kernel_order=KERNEL_SLOT_ORDER,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.DeviceModuleMetadata,
            int(device_ir_version),
            target_arch,
            launch_grid,
            operation_identity,
            list(tensor_table),
            list(dfb_table),
            list(pipe_table),
            list(kernel_order),
        )


@tvm_ffi.register_object("tl.tenstorrent.DeviceFunctionMetadata")
class DeviceFunctionMetadata(Node):
    def __init__(
        self,
        kernel_slot: str,
        kernel_thread: str,
        noc_index: int | None,
        logical_kernel: LogicalKernel,
        tensor_arg_indices,
        core_domain: CoreDomain,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.DeviceFunctionMetadata,
            kernel_slot,
            kernel_thread,
            noc_index,
            logical_kernel,
            list(tensor_arg_indices),
            core_domain,
        )


def attach_device_module_metadata(mod: IRModule, metadata: DeviceModuleMetadata) -> IRModule:
    """Attach only the eight frozen final Device IR Module attributes."""
    return mod.with_attrs(
        {
            DEVICE_IR_VERSION_ATTR: metadata.device_ir_version,
            TARGET_ARCH_ATTR: metadata.target_arch,
            LAUNCH_GRID_ATTR: metadata.launch_grid,
            OPERATION_IDENTITY_ATTR: metadata.operation_identity,
            TENSOR_TABLE_ATTR: metadata.tensor_table,
            DFB_TABLE_ATTR: metadata.dfb_table,
            PIPE_TABLE_ATTR: metadata.pipe_table,
            KERNEL_ORDER_ATTR: metadata.kernel_order,
        }
    )


def attach_device_function_metadata(
    func: tirx.PrimFunc,
    metadata: DeviceFunctionMetadata,
    *,
    global_symbol: str,
    target: Target,
) -> tirx.PrimFunc:
    attrs = {
        "global_symbol": global_symbol,
        "calling_conv": int(CallingConv.DEVICE_KERNEL_LAUNCH),
        "target": target,
        KERNEL_SLOT_ATTR: metadata.kernel_slot,
        KERNEL_THREAD_ATTR: metadata.kernel_thread,
        LOGICAL_KERNEL_ATTR: metadata.logical_kernel,
        TENSOR_ARG_INDICES_ATTR: metadata.tensor_arg_indices,
        CORE_DOMAIN_ATTR: metadata.core_domain,
    }
    if metadata.noc_index is not None:
        attrs[NOC_INDEX_ATTR] = metadata.noc_index
    return func.with_attr(attrs)


def VerifyTenstorrentDeviceIR():
    """Compatibility alias for the backend transform factory."""
    from .transform import VerifyTenstorrentDeviceIR as _verify

    return _verify()


__all__ = (
    "is_supported_accumulator_dtype_triple",
    "AccumulatorDescriptor",
    "ComputeValueDescriptor",
    "COMPUTE_VALUE_TABLE_ATTR",
    "ComputeRequirements",
    "CommunicationDescriptor",
    "COMMUNICATION_TABLE_ATTR",
    "ACCUMULATOR_TABLE_ATTR",
    "COMPUTE_REQUIREMENTS_ATTR",
    "COMPUTE_REGION_REQUIREMENTS_ATTR",
    "BUFFER_METADATA_TABLE_ATTR",
    "CoreCoord",
    "CoreDomain",
    "DFBDescriptor",
    "DFBStorageGroupDescriptor",
    "DFBStorageGroupRelation",
    "DFB_STORAGE_GROUPS_ATTR",
    "DFB_STORAGE_GROUP_RELATIONS_ATTR",
    "DeviceIRDescriptorFamily",
    "DeviceIRIntColumn",
    "DeviceFunctionMetadata",
    "DeviceModuleMetadata",
    "LogicalKernel",
    "NormalizedTopologyDescriptor",
    "OperationIdentity",
    "PipeDescriptor",
    "PipeTransferDescriptor",
    "PIPE_TRANSFER_TABLE_ATTR",
    "ShardSpec",
    "TTBufferMetadata",
    "TRANSPORT_TABLE_ATTR",
    "TensorBacking",
    "TensorStorageBacking",
    "TensorDescriptor",
    "VerifyTenstorrentDeviceIR",
    "attach_device_function_metadata",
    "attach_device_module_metadata",
    "device_ir_to_tirx",
)

"""Typed construction helpers shared by Tenstorrent Device IR v1/v2/v3/v4.

``TTBufferMetadata`` is a normalization-stage object.  It is attached as an
array under :data:`BUFFER_METADATA_TABLE_ATTR` before program formation and
must be consumed before :func:`attach_device_module_metadata` produces final
Device TIR.

The reflected descriptor constructors retain their v1 defaults. Pipeline v3
adds module attributes for storage groups and iteration relations; these must
be attached alongside the common metadata before invoking the verifier.
Multicore v4 adds ``tt.pipe_transfer_table`` containing typed point deliveries
and orders all Core/slot functions by global name in ``tt.kernel_order``.
The earlier metadata constructors and their default version remain unchanged.
"""

from __future__ import annotations

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
DFB_TABLE_ATTR = "tt.dfb_table"
PIPE_TABLE_ATTR = "tt.pipe_table"
PIPE_TRANSFER_TABLE_ATTR = "tt.pipe_transfer_table"
KERNEL_ORDER_ATTR = "tt.kernel_order"
KERNEL_SLOT_ATTR = "tt.kernel_slot"
KERNEL_THREAD_ATTR = "tt.kernel_thread"
NOC_INDEX_ATTR = "tt.noc_index"
LOGICAL_KERNEL_ATTR = "tt.logical_kernel"
TENSOR_ARG_INDICES_ATTR = "tt.tensor_arg_indices"
CORE_DOMAIN_ATTR = "tt.core_domain"
BUFFER_METADATA_TABLE_ATTR = "tt.buffer_metadata_table"
DFB_STORAGE_GROUPS_ATTR = "tt.dfb_storage_groups"
PIPELINE_RELATIONS_ATTR = "tt.pipeline_relations"
PIPELINE_STAGES_ATTR = "tt.pipeline_stages"
PIPELINE_EXTENT_ATTR = "tt.pipeline_extent"
L1_CAPACITY_BYTES_ATTR = "tt.l1_capacity_bytes"


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


@tvm_ffi.register_object("tl.tenstorrent.PipeDescriptor")
class PipeDescriptor(Node):
    def __init__(
        self,
        pipe_net_id: int,
        event_index: int,
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
            int(event_index),
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
    "BUFFER_METADATA_TABLE_ATTR",
    "CoreCoord",
    "CoreDomain",
    "DFBDescriptor",
    "DeviceFunctionMetadata",
    "DeviceModuleMetadata",
    "LogicalKernel",
    "OperationIdentity",
    "PipeDescriptor",
    "PipeTransferDescriptor",
    "PIPE_TRANSFER_TABLE_ATTR",
    "ShardSpec",
    "TTBufferMetadata",
    "TensorBacking",
    "TensorDescriptor",
    "VerifyTenstorrentDeviceIR",
    "attach_device_function_metadata",
    "attach_device_module_metadata",
)

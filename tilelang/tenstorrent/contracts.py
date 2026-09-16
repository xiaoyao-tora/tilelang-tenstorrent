"""Frozen Phase 0 contracts for the Tenstorrent compiler path."""

from __future__ import annotations


TARGET_KIND = "tenstorrent"
SUPPORTED_ARCHITECTURES = ("wormhole_b0", "blackhole")
TARGET_KEYS = ("tenstorrent",)
REQUIRED_TARGET_ATTRIBUTES = ("arch",)
TARGET_ALIASES = ()
TARGET_AUTO_DETECTION = False
TARGET_DEVICE_TYPE = "ext_dev"

DEVICE_OUTPUT_FORMAT = "ttl"
EXECUTION_BACKEND_ORDER = ("ttnn",)
TTNN_ENABLE_HOST_CODEGEN = False
TTNN_ENABLE_DEVICE_COMPILE = False

DEVICE_IR_VERSION = 1
# Version 1 remains the construction default for existing Add/idle modules.
# General computation uses version 2; consumers must dispatch on the version.
GENERAL_DEVICE_IR_VERSION = 2
PIPELINE_DEVICE_IR_VERSION = 3
SUPPORTED_DEVICE_IR_VERSIONS = (DEVICE_IR_VERSION, GENERAL_DEVICE_IR_VERSION, PIPELINE_DEVICE_IR_VERSION)

# v3 adds module metadata; the v1/v2 reflected descriptor constructors retain
# their fields and defaults. Each immutable DFB generation has one transaction.
# Storage pools relate lexical write sites across statically expanded epochs.
PIPELINE_MODULE_FIELDS = (
    "tt.dfb_storage_groups",  # Map<decimal DFB ID string, pool ID>
    "tt.pipeline_relations",  # Map<decimal DFB ID string, [ordinal, stage]>
    "tt.pipeline_stages",  # requested depth; effective depth=min(depth, extent)
    "tt.pipeline_extent",
)

MODULE_FIELDS = (
    "tt.device_ir_version",
    "tt.target_arch",
    "tt.launch_grid",
    "tt.operation_identity",
    "tt.tensor_table",
    "tt.dfb_table",
    "tt.pipe_table",
    "tt.kernel_order",
)

TENSOR_DESCRIPTOR_FIELDS = (
    "global_arg_index",
    "shape",
    "dtype",
    "strides",
    "tile_shape",
    "tile_grid_shape",
    "memory_space",
    "memory_layout",
    "shard_spec",
    "effect",
    "alias_group",
    "source_span",
)

DFB_DESCRIPTOR_FIELDS = (
    "dfb_id",
    "source_buffer_identity",
    "element_dtype",
    "tile_shape",
    "block_shape_in_tiles",
    "block_count",
    "tensor_backing",
    "producer_slot",
    "producer_domain",
    "consumer_slot",
    "consumer_domain",
    "transaction_count_or_loop_relation",
    "source_span",
)

PIPE_DESCRIPTOR_FIELDS = (
    "pipe_net_id",
    "event_index",
    "src_coord",
    "dst_begin",
    "dst_end",
    "contract",
    "payload_dfb_id",
    "source_span",
)

PRIM_FUNC_ATTRIBUTE_FIELDS = (
    "global_symbol",
    "calling_conv",
    "target",
    "tt.kernel_slot",
    "tt.kernel_thread",
    "tt.noc_index",
    "tt.logical_kernel",
    "tt.tensor_arg_indices",
    "tt.core_domain",
)

KERNEL_SLOT_ORDER = ("trisc", "ncrisc", "brisc")
KERNEL_SLOT_THREAD_NOC = (
    ("trisc", "compute", None),
    ("ncrisc", "datamovement", 0),
    ("brisc", "datamovement", 1),
)

INITIAL_TTL_REQUIRED = (
    "tensor_types_and_layouts",
    "ttl.launch_grid",
    "ttl.target_arch",
    "three_kernel_slot_functions_and_attributes",
    "logical_dfb_id",
    "ttl.bind_cb",
    "ttl.cb_reserve",
    "ttl.cb_wait",
    "ttl.attach_cb",
    "ttl.tensor_slice",
    "ttl.copy",
    "transfer_handle",
    "tensor_level_compute_and_store",
    "pipe_and_pipenet_structure",
    "source_location",
)

INITIAL_TTL_DEFERRED = (
    "all_ttl.wait",
    "all_ttl.cb_push_and_ttl.cb_pop",
    "physical_cb_index",
    "dst_index",
    "ttl.compute_tile_ops_and_subblock_schedule",
    "ttkernel",
    "emitc",
    "cxx",
)

DEFERRED_FRONTEND_CAPABILITIES = ("tt.tensor_backed",)

TTLANG_REFERENCE_COMMIT = "6e3051bb9b2f7bfbd5fd045aa38deb68b7a6e6bd"
TTLANG_REFERENCE_VERSION = "1.1.9.dev51"
TTLANG_LEGACY_REFERENCE_COMMIT = "e49ce12c2bcd2350565632d9b51239e9eafb2546"
TTLANG_PUBLIC_BINDING_MODULES = (
    "ttl.ir",
    "ttl.dialects.ttl",
    "ttl.dialects.ttcore",
    "ttl.dialects.ttkernel",
    "ttl.passmanager",
    "ttl.passes",
)
TTLANG_LOWERING_PIPELINE = "ttl-to-ttkernel-pipeline"


__all__ = (
    "DEFERRED_FRONTEND_CAPABILITIES",
    "DEVICE_IR_VERSION",
    "GENERAL_DEVICE_IR_VERSION",
    "PIPELINE_DEVICE_IR_VERSION",
    "PIPELINE_MODULE_FIELDS",
    "SUPPORTED_DEVICE_IR_VERSIONS",
    "DEVICE_OUTPUT_FORMAT",
    "DFB_DESCRIPTOR_FIELDS",
    "EXECUTION_BACKEND_ORDER",
    "INITIAL_TTL_DEFERRED",
    "INITIAL_TTL_REQUIRED",
    "KERNEL_SLOT_ORDER",
    "KERNEL_SLOT_THREAD_NOC",
    "MODULE_FIELDS",
    "PIPE_DESCRIPTOR_FIELDS",
    "PRIM_FUNC_ATTRIBUTE_FIELDS",
    "SUPPORTED_ARCHITECTURES",
    "TARGET_ALIASES",
    "TARGET_AUTO_DETECTION",
    "TARGET_DEVICE_TYPE",
    "TARGET_KIND",
    "TARGET_KEYS",
    "TENSOR_DESCRIPTOR_FIELDS",
    "TTNN_ENABLE_DEVICE_COMPILE",
    "TTNN_ENABLE_HOST_CODEGEN",
    "TTLANG_LEGACY_REFERENCE_COMMIT",
    "TTLANG_LOWERING_PIPELINE",
    "TTLANG_PUBLIC_BINDING_MODULES",
    "TTLANG_REFERENCE_COMMIT",
    "TTLANG_REFERENCE_VERSION",
    "REQUIRED_TARGET_ATTRIBUTES",
)

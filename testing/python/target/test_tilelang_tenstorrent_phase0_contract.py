from __future__ import annotations

import json

import pytest
from tvm import ir, tirx
from tvm.ir import Op

from tilelang import tvm
from tilelang.backend import get_backend, list_target_detectors
from tilelang.tenstorrent import contracts
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.pipeline import TENSTORRENT_LOWER_PASS_ORDER


def _make_frontend_programs():
    @T.prim_func
    def add(
        A: T.Tensor((32, 32), T.bfloat16),
        B: T.Tensor((32, 32), T.bfloat16),
        C: T.Tensor((32, 32), T.bfloat16),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared(
                (32, 32),
                T.bfloat16,
                annotations={"tt.dfb_block_count": 2, "tt.tile_shape": (32, 32)},
            )
            b = T.alloc_shared(
                (32, 32),
                T.bfloat16,
                annotations={"tt.dfb_block_count": 2, "tt.tile_shape": (32, 32)},
            )
            c = T.alloc_shared(
                (32, 32),
                T.bfloat16,
                annotations={"tt.dfb_block_count": 2, "tt.tile_shape": (32, 32)},
            )
            T.copy(A, a)
            T.copy(B, b)
            for i, j in T.Parallel(32, 32):
                c[i, j] = a[i, j] + b[i, j]
            T.copy(c, C)

    point_to_point = T.comm.PipeNet([T.comm.Pipe(src=(0, 0), dst=(1, 0))])

    @T.prim_func
    def p2p():
        with T.Kernel(2, 1, threads=1):
            send = T.alloc_shared((32, 32), T.bfloat16)
            recv = T.alloc_shared((32, 32), T.bfloat16)
            for pipe in T.comm.foreach_src(point_to_point):
                T.copy(send, pipe)
            for pipe in T.comm.foreach_dst(point_to_point):
                T.copy(pipe, recv)

    gather_net = T.comm.PipeNet(
        [
            T.comm.Pipe(src=(1, 0), dst=(0, 0)),
            T.comm.Pipe(src=(2, 0), dst=(0, 0)),
            T.comm.Pipe(src=(2, 0), dst=(0, 0)),
        ]
    )

    @T.prim_func
    def gather():
        with T.Kernel(3, 1, threads=1):
            payload = T.alloc_shared((32, 32), T.bfloat16)
            if T.comm.is_src(gather_net):
                T.evaluate(0)
            if T.comm.is_dst(gather_net):
                T.evaluate(0)
            for pipe in T.comm.foreach_src(gather_net):
                T.copy(payload, pipe)
            for pipe in T.comm.foreach_dst(gather_net):
                T.copy(pipe, payload)

    collective_net = T.comm.PipeNet(
        [
            T.comm.Pipe(
                src=(0, 0),
                dst=T.comm.CoreRange(begin=(1, 0), end=(4, 1)),
            )
        ]
    )

    @T.prim_func
    def collective():
        with T.Kernel(4, 1, threads=1):
            payload = T.alloc_shared((32, 32), T.bfloat16)
            for pipe in T.comm.foreach_src(collective_net):
                begin, end = T.comm.pipe_dst_range(pipe)
                T.evaluate(begin[0] + begin[1] + end[0] + end[1])
                T.copy(payload, pipe)
            for pipe in T.comm.foreach_dst(collective_net):
                T.copy(pipe, payload)

    return {
        "add": add,
        "p2p": p2p,
        "gather": gather,
        "collective": collective,
    }


FRONTEND_PROGRAMS = _make_frontend_programs()


def _collect_tt_calls(func):
    calls = []

    def visit(node):
        if isinstance(node, tirx.Call) and isinstance(node.op, Op) and node.op.name.startswith("tl.tt."):
            calls.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls


def _collect_foreach_descriptors(func):
    descriptors = []

    def visit(node):
        if not isinstance(node, tirx.For):
            return
        for key in ("tl.tt.foreach_src", "tl.tt.foreach_dst"):
            if key in node.annotations:
                descriptors.append(json.loads(str(node.annotations[key])))

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return descriptors


def test_frozen_device_ir_schema_and_slot_contract():
    assert contracts.TARGET_KIND == "tenstorrent"
    assert contracts.SUPPORTED_ARCHITECTURES == ("wormhole_b0", "blackhole")
    assert contracts.DEVICE_IR_VERSION == 1
    assert contracts.MODULE_FIELDS == (
        "tt.device_ir_version",
        "tt.target_arch",
        "tt.launch_grid",
        "tt.operation_identity",
        "tt.tensor_table",
        "tt.dfb_table",
        "tt.pipe_table",
        "tt.kernel_order",
    )
    assert contracts.TENSOR_DESCRIPTOR_FIELDS == (
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
    assert contracts.DFB_DESCRIPTOR_FIELDS == (
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
    assert contracts.PIPE_DESCRIPTOR_FIELDS == (
        "pipe_net_id",
        "event_index",
        "src_coord",
        "dst_begin",
        "dst_end",
        "contract",
        "payload_dfb_id",
        "source_span",
    )
    assert contracts.PRIM_FUNC_ATTRIBUTE_FIELDS == (
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
    assert contracts.KERNEL_SLOT_ORDER == ("trisc", "ncrisc", "brisc")
    assert contracts.KERNEL_SLOT_THREAD_NOC == (
        ("trisc", "compute", None),
        ("ncrisc", "datamovement", 0),
        ("brisc", "datamovement", 1),
    )
    assert "tt.tensor_backed" in contracts.DEFERRED_FRONTEND_CAPABILITIES


def test_frozen_target_and_execution_preflight_matches_registered_backend():
    assert contracts.TARGET_KEYS == ("tenstorrent",)
    assert contracts.REQUIRED_TARGET_ATTRIBUTES == ("arch",)
    assert contracts.TARGET_ALIASES == ()
    assert contracts.TARGET_AUTO_DETECTION is False
    assert contracts.TARGET_DEVICE_TYPE == "ext_dev"
    assert contracts.DEVICE_OUTPUT_FORMAT == "ttl"
    assert contracts.EXECUTION_BACKEND_ORDER == ("ttnn",)
    assert contracts.TTNN_ENABLE_HOST_CODEGEN is False
    assert contracts.TTNN_ENABLE_DEVICE_COMPILE is False

    target = tvm.target.Target({"kind": contracts.TARGET_KIND, "arch": "wormhole_b0"})
    assert tuple(str(key) for key in target.keys) == contracts.TARGET_KEYS
    assert all(attribute in target.attrs for attribute in contracts.REQUIRED_TARGET_ATTRIBUTES)
    assert target.kind.default_device_type == tvm.device(contracts.TARGET_DEVICE_TYPE, 0).dlpack_device_type()
    assert contracts.TARGET_KIND not in list_target_detectors()

    backend = get_backend("tenstorrent")
    device_codegen = backend.get_device_codegen(target)
    assert tuple(spec.name for spec in backend.execution_backends) == contracts.EXECUTION_BACKEND_ORDER
    assert backend.host_codegens == {}
    assert device_codegen.build is None
    assert device_codegen.build_without_compile is not None
    assert backend.execution_backends[0].enable_host_codegen is contracts.TTNN_ENABLE_HOST_CODEGEN
    assert backend.execution_backends[0].enable_device_compile is contracts.TTNN_ENABLE_DEVICE_COMPILE


def test_structured_compute_precedes_device_lowering_in_pass_order():
    assert TENSTORRENT_LOWER_PASS_ORDER == (
        "BindTarget",
        "CanonicalizeTTElementwise",
        "VerifyTTComputeBlocks",
        "ValidateTenstorrentFrontendIR",
        "NormalizeTenstorrentLaunch",
        "NormalizeTenstorrentBufferMetadata",
        "NormalizeTenstorrentTopology",
        "NormalizeTenstorrentRegions",
        "LegalizeTenstorrentTileOps",
        "InferTenstorrentTensorLayout",
        "FormTenstorrentDeviceProgram",
        "VerifyTenstorrentDeviceIR",
    )


def test_frozen_ttl_reference_and_initial_module_contract():
    assert contracts.TTLANG_REFERENCE_COMMIT == "6e3051bb9b2f7bfbd5fd045aa38deb68b7a6e6bd"
    assert contracts.TTLANG_REFERENCE_VERSION == "1.1.9.dev51"
    assert contracts.TTLANG_PUBLIC_BINDING_MODULES == (
        "ttl.ir",
        "ttl.dialects.ttl",
        "ttl.dialects.ttcore",
        "ttl.dialects.ttkernel",
        "ttl.passmanager",
        "ttl.passes",
    )
    assert contracts.TTLANG_LOWERING_PIPELINE == "ttl-to-ttkernel-pipeline"
    assert contracts.INITIAL_TTL_REQUIRED == (
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
    assert contracts.INITIAL_TTL_DEFERRED == (
        "all_ttl.wait",
        "all_ttl.cb_push_and_ttl.cb_pop",
        "physical_cb_index",
        "dst_index",
        "ttl.compute_tile_ops_and_subblock_schedule",
        "ttkernel",
        "emitc",
        "cxx",
    )


def test_frontend_language_contract_uses_public_comm_and_internal_tt_ops():
    assert "comm" in T.__all__
    assert "tt" not in T.__all__
    for name in ("p2p", "gather", "collective"):
        calls = _collect_tt_calls(FRONTEND_PROGRAMS[name])
        assert calls
        assert all(call.op.name.startswith("tl.tt.") for call in calls)


def test_add_freezes_alloc_shared_metadata_contract():
    script = str(FRONTEND_PROGRAMS["add"])
    assert "tl.alloc_buffer_annotations" in script
    assert '"tt.dfb_block_count"' in script
    assert '"tt.tile_shape"' in script


def test_gather_preserves_pipe_order_and_duplicate_events():
    descriptors = _collect_foreach_descriptors(FRONTEND_PROGRAMS["gather"])
    assert len(descriptors) == 2
    expected_sources = [[1, 0], [2, 0], [2, 0]]
    for descriptor in descriptors:
        assert descriptor["kind"] == "point_to_point"
        assert [pipe["src"] for pipe in descriptor["pipes"]] == expected_sources
        assert descriptor["pipes"][1] == descriptor["pipes"][2]


@pytest.mark.parametrize("name", tuple(FRONTEND_PROGRAMS))
def test_frontend_tirx_is_deterministic_and_json_round_trips(name):
    mod = tvm.IRModule({name: FRONTEND_PROGRAMS[name]})
    first = str(mod)
    assert first == str(mod)

    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(mod, restored)
    assert str(restored) == first

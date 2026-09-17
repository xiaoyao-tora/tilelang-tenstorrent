"""Capability declarations agree with executable Lower routes, without codegen."""

from dataclasses import FrozenInstanceError

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T, lower_tenstorrent_ir
from tilelang.tenstorrent.capabilities import CapabilityStatus, gemm_capability, lower_capability
from testing.python.target.test_tilelang_tenstorrent_branch_merge import branch_program
from testing.python.target.test_tilelang_tenstorrent_compute_values import gemm_fragment_operand, value_chain
from testing.python.target.test_tilelang_tenstorrent_phase4_compute import (
    cast_program,
    batch_elementwise,
    elementwise_program,
    fill_program,
    gemm_program,
    reduction_program,
    transpose_program,
)
from testing.python.target.test_tilelang_tenstorrent_phase4_control import index_condition
from testing.python.target.test_tilelang_tenstorrent_lower_composition import (
    full_k_pipeline,
    multicore_gemm_epilogue,
    multicore_values,
    pipe_values,
    pipeline_values,
    pipeline_pipe_values,
    pipeline_summa,
    transpose_fragment,
)
from testing.python.target.test_tilelang_tenstorrent_phase5_pipeline import make_pipeline
from testing.python.target.test_tilelang_tenstorrent_phase6_lower import make_point_to_point
from testing.python.target.test_tilelang_tenstorrent_summa_lower import make_summa


def fragment_input_operation(operation):
    output_shape = (32,) if operation == "reduce" else (32, 32)

    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), C: T.Tensor(output_shape, "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32")
            f = T.alloc_fragment((32, 32), "float32")
            c = T.alloc_shared(output_shape, "float32")
            T.copy(A, a)
            T.copy(a, f)
            if operation == "reduce":
                T.reduce_sum(f, c, dim=1)
            else:
                T.transpose(f, c)
            T.copy(c, C)

    return main


def reduction_fragment_output():
    @T.prim_func
    def main(A: T.Tensor((2, 32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((2, 32, 32), "float32")
            f = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            T.reduce_sum(a, f, dim=0)
            T.copy(f, C)

    return main


def shared_pipe_pipeline():
    net = T.comm.PipeNet([T.comm.Pipe((0, 0), (1, 0))])

    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(2, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.dfb_block_count": 2})
            for _ in T.Pipelined(3, num_stages=2):
                if T.comm.is_src(net):
                    T.copy(A, a)
                for pipe in T.comm.foreach_src(net):
                    T.copy(a, pipe)
                for pipe in T.comm.foreach_dst(net):
                    T.copy(pipe, a)
                    T.copy(a, C)

    return main


def multicore_shared_pipeline():
    @T.prim_func
    def main(A: T.Tensor((64, 32), "float32"), C: T.Tensor((64, 32), "float32")):
        with T.Kernel(2, 1, threads=1) as (x, y):
            a = T.alloc_shared((32, 32), "float32", annotations={"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2})
            for _ in T.Pipelined(3, num_stages=2):
                T.copy(A[x * 32 : (x + 1) * 32, :], a)
                for i, j in T.Tiles(a):
                    a[i, j] = a[i, j] * 2
                T.copy(a, C[x * 32 : (x + 1) * 32, :])

    return main


@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
@pytest.mark.parametrize(
    "operation,factory,query",
    [
        ("elementwise", elementwise_program, {}),
        ("elementwise", lambda: batch_elementwise, {"rank": 3}),
        ("fill", lambda: fill_program, {}),
        ("typecast", lambda: cast_program, {"input_dtype": "bfloat16", "output_dtype": "float32"}),
        ("transpose", lambda: transpose_program, {}),
        ("gemm", gemm_program, {"input_dtype": "bfloat16", "output_dtype": "float32"}),
        ("reduce", reduction_program, {}),
        ("transpose", lambda: fragment_input_operation("transpose"), {"input_value_kind": "fragment"}),
        ("reduce", lambda: fragment_input_operation("reduce"), {"input_value_kind": "fragment"}),
        ("reduce", reduction_fragment_output, {"rank": 3, "output_value_kind": "fragment"}),
        ("transpose", transpose_fragment, {"output_value_kind": "fragment"}),
        ("transpose", lambda: transpose_fragment(source_fragment=True), {"input_value_kind": "fragment", "output_value_kind": "fragment"}),
        ("pipe", make_point_to_point, {"multicore": True}),
        ("elementwise", make_pipeline, {"pipelined": True}),
        ("elementwise", lambda: index_condition, {"control_flow": "static_branch"}),
        ("elementwise", value_chain, {"input_value_kind": "mixed", "output_value_kind": "fragment", "control_flow": "static_loop"}),
        ("elementwise", branch_program, {"output_value_kind": "fragment", "control_flow": "pure_value_branch"}),
        ("elementwise", lambda: branch_program(fragment=False), {"control_flow": "pure_value_branch"}),
        ("elementwise", multicore_values, {"multicore": True, "output_value_kind": "fragment"}),
        ("elementwise", pipe_values, {"multicore": True, "output_value_kind": "fragment"}),
        ("elementwise", pipeline_values, {"pipelined": True, "output_value_kind": "fragment"}),
        ("pipe", shared_pipe_pipeline, {"pipelined": True, "multicore": True}),
        ("elementwise", multicore_shared_pipeline, {"pipelined": True, "multicore": True}),
        ("elementwise", pipeline_pipe_values, {"pipelined": True, "multicore": True, "output_value_kind": "fragment"}),
        (
            "gemm",
            multicore_gemm_epilogue,
            {
                "multicore": True,
                "ir_version": 7,
                "output_value_kind": "accumulator",
                "input_dtype": "bfloat16",
                "output_dtype": "bfloat16",
                "accumulation_dtype": "float32",
            },
        ),
        (
            "gemm",
            full_k_pipeline,
            {
                "pipelined": True,
                "loop_carried": True,
                "output_value_kind": "accumulator",
                "input_dtype": "bfloat16",
                "output_dtype": "bfloat16",
                "accumulation_dtype": "float32",
                "update_count": 5,
            },
        ),
        (
            "gemm",
            pipeline_summa,
            {
                "pipelined": True,
                "multicore": True,
                "loop_carried": True,
                "output_value_kind": "accumulator",
                "input_dtype": "bfloat16",
                "output_dtype": "bfloat16",
                "accumulation_dtype": "float32",
                "update_count": 5,
            },
        ),
        (
            "gemm",
            gemm_fragment_operand,
            {
                "input_value_kind": "mixed",
                "output_value_kind": "accumulator",
                "input_dtype": "bfloat16",
                "output_dtype": "bfloat16",
                "accumulation_dtype": "float32",
            },
        ),
    ],
)
def test_supported_queries_match_real_lower(operation, factory, query, arch):
    capability = lower_capability(operation, arch=arch, **query)
    assert capability.device_lower, capability.reason
    target = tvm.target.Target({"kind": "tenstorrent", "arch": arch})
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": factory()}), target)
    assert "tt.ir_stage" not in mod.attrs
    assert int(mod.attrs["tt.device_ir_version"]) == capability.device_ir_version
    expected = CapabilityStatus.LEGALIZABLE if query.get("control_flow") else CapabilityStatus.SUPPORTED
    assert capability.status == expected


@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
@pytest.mark.parametrize("accumulation_dtype", ["bfloat16", "float32"])
def test_multicore_accumulator_is_lower_supported_without_claiming_codegen(arch, accumulation_dtype):
    capability = gemm_capability("bfloat16", accumulation_dtype, "bfloat16", arch=arch, ir_version=6, multicore=True, update_count=2)
    assert capability.device_lower
    assert not capability.ttl_mapping
    assert not capability.compile_only_validated
    assert not capability.hardware_validated
    target = tvm.target.Target({"kind": "tenstorrent", "arch": arch})
    mod = lower_tenstorrent_ir(tvm.IRModule({"main": make_summa(accum_dtype=accumulation_dtype)}), target)
    assert int(mod.attrs["tt.device_ir_version"]) == 6
    assert len(mod.attrs["tt.accumulator_table"]) == 4


@pytest.mark.parametrize(
    "operation,query,reason",
    [
        ("elementwise", {"control_flow": "dynamic_branch"}, "Dynamic control flow"),
        ("elementwise", {"control_flow": "dynamic_loop"}, "Dynamic control flow"),
        ("elementwise", {"pipelined": True, "input_value_kind": "fragment", "loop_carried": True}, "loop-carried"),
        ("elementwise", {"pipelined": True, "control_flow": "static_branch"}, "independent iterations"),
        ("gemm", {"pipelined": True, "output_value_kind": "accumulator", "ir_version": 7}, "v5/v6 route"),
        ("gemm", {"pipelined": True, "output_value_kind": "accumulator", "input_value_kind": "mixed"}, "v5/v6 route"),
        ("gemm", {"control_flow": "pure_value_branch"}, "elementwise assignments"),
    ],
)
def test_deferred_combinations_are_not_reported_as_device_lower(operation, query, reason):
    capability = lower_capability(operation, arch="wormhole_b0", **query)
    assert capability.status == CapabilityStatus.DEFERRED
    assert not capability.device_lower
    assert capability.device_ir_version is None
    assert reason in capability.reason


@pytest.mark.parametrize(
    "operation,query,reason",
    [
        ("unknown", {}, "Unknown Lower operation"),
        ("elementwise", {"input_dtype": "float16"}, "BF16/FP32"),
        ("elementwise", {"output_dtype": "float16"}, "BF16/FP32"),
        ("elementwise", {"rank": 0}, "positive rank"),
        ("elementwise", {"tile_shape": (16, 32)}, "32x32"),
        ("elementwise", {"shape_class": "dynamic"}, "static tile-aligned"),
        ("elementwise", {"shape_class": "padded"}, "padding/masks"),
        ("elementwise", {"memory_layout": "row_major"}, "interleaved"),
        ("elementwise", {"sharded": True}, "sharding"),
        ("elementwise", {"compact": False}, "compact"),
        ("elementwise", {"aliased": True}, "aliases"),
        ("elementwise", {"materialization": "implicit_bf16"}, "exact-dtype"),
        ("elementwise", {"rank": 3, "input_value_kind": "fragment"}, "rank-2"),
        ("elementwise", {"output_value_kind": "accumulator"}, "reserved for GEMM"),
        ("gemm", {"rank": 3}, "batched GEMM"),
        ("gemm", {"input_dtype": "float32", "output_dtype": "bfloat16", "accumulation_dtype": "bfloat16"}, "Dtype triple"),
        ("gemm", {"input_dtype": "bfloat16", "output_dtype": "bfloat16", "accumulation_dtype": "float32"}, "must equal"),
        ("gemm", {"update_count": 0}, "at least one update"),
        ("transpose", {"transpose_axes": "arbitrary"}, "final two axes"),
        ("reduce", {"keepdims": True}, "exactly one axis"),
        ("reduce", {"reduction_kind": "prod"}, "sum/max/min"),
        ("reduce", {"accumulation_dtype": "bfloat16"}, "float32 accumulation"),
        ("pipe", {"input_value_kind": "fragment", "multicore": True}, "shared DFBs"),
        ("pipe", {"input_dtype": "bfloat16", "output_dtype": "float32", "multicore": True}, "preserves"),
        ("gemm", {"output_value_kind": "accumulator", "ir_version": 4}, "requires Device IR v5"),
    ],
)
def test_unsupported_contracts_have_actionable_reasons(operation, query, reason):
    capability = lower_capability(operation, arch="wormhole_b0", **query)
    assert capability.status == CapabilityStatus.UNSUPPORTED
    assert not capability.device_lower
    assert reason in capability.reason


def test_capability_results_are_read_only_and_arch_specific():
    capability = lower_capability("elementwise", arch="blackhole")
    with pytest.raises(FrozenInstanceError):
        capability.status = CapabilityStatus.UNSUPPORTED
    assert not lower_capability("elementwise", arch="unknown").device_lower
    assert lower_capability("elementwise", arch="blackhole") == capability


@pytest.mark.parametrize(
    "version,multicore,pipelined",
    [(5, False, False), (7, False, False), (6, True, False), (7, True, False), (5, False, True), (6, True, True)],
)
def test_legacy_gemm_query_keeps_lower_separate_from_validation(version, multicore, pipelined):
    capability = gemm_capability(
        "bfloat16", "float32", "bfloat16", arch="blackhole", ir_version=version, multicore=multicore, pipelined=pipelined
    )
    assert capability.device_lower
    assert not capability.ttl_mapping
    assert not capability.compile_only_validated
    assert not capability.hardware_validated

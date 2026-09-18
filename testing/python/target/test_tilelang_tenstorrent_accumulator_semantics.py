"""Numerically execute Device IR accumulator lifetimes on the CPU.

This checks the Lower operation contract, including full-K persistence and final
rounding. It does not certify TT-Lang scheduling or hardware arithmetic.
"""

import numpy as np
import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
from tilelang.tenstorrent.transform import VerifyTenstorrentDeviceIR
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_phase4_semantics import cast


def execute_accumulators(mod, inputs, *, inject_intermediate_bf16_pack=False):
    """Execute final slot streams with waits, using only typed Device contracts."""
    VerifyTenstorrentDeviceIR()(mod)
    tensors = [cast(value, desc.dtype).copy() for value, desc in zip(inputs, mod.attrs["tt.tensor_table"])]
    descriptors = {int(item.accumulator_id): item for item in mod.attrs["tt.accumulator_table"]}
    dfbs = {int(item.dfb_id): item for item in mod.attrs["tt.dfb_table"]}
    accumulators, values, materializations = {}, {}, []
    streams = []
    for func in mod.functions.values():
        statements = list(func.body.seq) if isinstance(func.body, tirx.SeqStmt) else [func.body]
        streams.append([stmt.value for stmt in statements if isinstance(stmt.value, tirx.Call)])
    positions = [0] * len(streams)
    while any(position < len(stream) for position, stream in zip(positions, streams)):
        progressed = False
        for slot, stream in enumerate(streams):
            if positions[slot] == len(stream):
                continue
            call = stream[positions[slot]]
            name, args = call.op.name, [int(arg) for arg in call.args]
            if name == "tl.tt.dfb_wait" and args[0] not in values:
                continue
            if name in {"tl.tt.dfb_wait", "tl.tt.dfb_reserve"}:
                pass
            elif name in {"tl.tt.tensor_to_dfb_nd", "tl.tt.dfb_to_tensor_nd"}:
                to_dfb = name == "tl.tt.tensor_to_dfb_nd"
                tensor, resource = args[:2] if to_dfb else args[1::-1]
                region = tuple(slice(start, start + extent) for start, extent in zip(args[2::2], args[3::2]))
                if to_dfb:
                    values[resource] = tensors[tensor][region].copy()
                else:
                    tensors[tensor][region] = values[resource]
            elif name == "tl.tt.accumulator_init":
                (accumulator,) = args
                shape = tuple(int(axis.extent) for axis in descriptors[accumulator].accumulator_region.region)
                assert accumulator not in accumulators
                accumulators[accumulator] = np.zeros(shape, np.float32)
            elif name == "tl.tt.gemm_update":
                lhs, rhs, accumulator, transpose_a, transpose_b = args
                a = values[lhs].T if transpose_a else values[lhs]
                b = values[rhs].T if transpose_b else values[rhs]
                result = accumulators[accumulator] + a.astype(np.float32) @ b.astype(np.float32)
                accumulators[accumulator] = cast(result, descriptors[accumulator].accumulation_dtype)
                if inject_intermediate_bf16_pack:
                    accumulators[accumulator] = cast(accumulators[accumulator], "bfloat16")
            elif name == "tl.tt.accumulator_materialize":
                accumulator, resource = args
                result = accumulators.pop(accumulator)
                materializations.append(result.copy())
                values[resource] = cast(result, dfbs[resource].element_dtype)
            elif name == "tl.tt.dfb_compute":
                assert call.annotations["tt.compute_kind"].value == "copy"
                output, source = args
                values[output] = values[source].copy()
            else:
                raise AssertionError(f"Unsupported accumulator reference op: {name}")
            positions[slot] += 1
            progressed = True
        assert progressed, "Device slot streams deadlocked"
    assert not accumulators, "live accumulators survived their final materialization"
    return tensors, materializations


def reduction_program(blocks, accumulation_dtype, input_dtype="bfloat16", output_dtype="bfloat16"):
    @T.prim_func
    def main(
        A: T.Tensor((32, blocks * 32), input_dtype),
        B: T.Tensor((blocks * 32, 32), input_dtype),
        C: T.Tensor((32, 32), output_dtype),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), input_dtype)
            b = T.alloc_shared((32, 32), input_dtype)
            c = T.alloc_fragment((32, 32), accumulation_dtype)
            T.clear(c)
            for k in T.serial(blocks):
                T.copy(A[:, k * 32 : k * 32 + 32], a)
                T.copy(B[k * 32 : k * 32 + 32, :], b)
                T.gemm(a, b, c)
            T.copy(c, C)

    return main


def lower(func, arch):
    target = tvm.target.Target({"kind": "tenstorrent", "arch": arch})
    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": func}), target)
    assert int(mod.attrs["tt.device_ir_version"]) == 5
    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(restored, mod)
    return restored


@pytest.mark.parametrize("arch", ["wormhole_b0", "blackhole"])
def test_long_k_cancellation_retains_fp32_until_final_bf16_rounding(arch):
    # Every operand is exactly BF16-representable. At 512, a BF16 accumulator
    # loses +1 updates; FP32 retains all 65 before cancellation and final +3/8.
    contributions = [512.0, *([1.0] * 65), -512.0, 0.375]
    identity = np.eye(32, dtype=np.float32)
    a = np.concatenate([identity] * len(contributions), axis=1)
    b = np.concatenate([value * identity for value in contributions], axis=0)
    exact = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(exact, identity * np.float32(65.375))
    inputs = [a, b, np.zeros((32, 32), np.float32)]

    fp32 = lower(reduction_program(len(contributions), "float32"), arch)
    outputs, final_values = execute_accumulators(fp32, inputs)
    assert len(final_values) == 1
    np.testing.assert_array_equal(final_values[0], exact)
    np.testing.assert_array_equal(outputs[-1], identity * np.float32(65.5))
    np.testing.assert_array_equal(outputs[-1], cast(exact, "bfloat16"))

    # A realistic regression (BF16 pack/reload between K updates) must produce
    # a different answer even though the input and final output dtypes agree.
    defective, _ = execute_accumulators(fp32, inputs, inject_intermediate_bf16_pack=True)
    assert not np.array_equal(defective[-1], outputs[-1])
    np.testing.assert_array_equal(defective[-1], identity * np.float32(0.375))

    bf16 = lower(reduction_program(len(contributions), "bfloat16"), arch)
    bf16_outputs, _ = execute_accumulators(bf16, inputs)
    np.testing.assert_array_equal(bf16_outputs[-1], defective[-1])


def test_fp32_inputs_accumulator_and_output_execute_all_k_slices():
    rng = np.random.default_rng(57)
    a = rng.integers(-3, 4, (32, 128)).astype(np.float32)
    b = rng.integers(-3, 4, (128, 32)).astype(np.float32)
    mod = lower(reduction_program(4, "float32", "float32", "float32"), "wormhole_b0")
    outputs, final_values = execute_accumulators(mod, [a, b, np.zeros((32, 32), np.float32)])
    assert len(final_values) == 1
    np.testing.assert_array_equal(outputs[-1], a @ b)

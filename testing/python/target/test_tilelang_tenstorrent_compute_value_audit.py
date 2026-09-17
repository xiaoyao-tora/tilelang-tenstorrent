"""Independent precision and descriptor integrity checks for Device values."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import lower_tenstorrent_ir, transform
from tilelang.tenstorrent.device_ir import AccumulatorDescriptor
from tvm import tirx

from testing.python.target.test_tilelang_tenstorrent_compute_value_verifier import device
from testing.python.target.test_tilelang_tenstorrent_frontend02 import TARGET, METADATA


def slot_function(mod, slot):
    return next((symbol, func) for symbol, func in mod.functions.items() if func.attrs["tt.kernel_slot"] == slot)


def clear_requirements(mod):
    for symbol, func in list(mod.functions.items()):
        mod.update_func(symbol, func.without_attr("tt.compute_requirements"))
    return mod


def test_bf16_value_with_fp32_intermediate_requires_fp32_destination():
    @T.prim_func
    def main(A: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            value = T.alloc_fragment((32, 32), "bfloat16")
            result = T.alloc_fragment((32, 32), "bfloat16")
            T.copy(A, a)
            T.copy(a, value)
            for i, j in T.Tiles(result):
                result[i, j] = T.cast(T.cast(value[i, j], "float32") * T.float32(0.5), "bfloat16")
            T.copy(result, C)

    mod = lower_tenstorrent_ir(tvm.IRModule({"main": main}), TARGET)
    assert all(str(entry.buffer.dtype) == "bfloat16" for entry in mod.attrs["tt.compute_value_table"])
    requirement = slot_function(mod, "trisc")[1].attrs["tt.compute_requirements"]
    assert requirement.destination_width == "bits32_required"


@pytest.mark.parametrize("start,extent", [(-32, 32), (0, 0), (0, 64), (1, 31)])
def test_raw_accumulator_region_cannot_disagree_with_its_fragment(start, extent):
    mod = device(gemm=True)
    (old,) = mod.attrs["tt.accumulator_table"]
    region = tirx.BufferRegion(old.accumulator_region.buffer, [tvm.ir.Range(start, start + extent), tvm.ir.Range(0, 32)])
    descriptor = AccumulatorDescriptor(
        old.accumulator_id,
        region,
        old.input_dtype,
        old.accumulation_dtype,
        old.output_dtype,
        old.full_k_tiles,
        old.source_span,
    )
    corrupt = mod.with_attr("tt.accumulator_table", [descriptor])
    with pytest.raises(ValueError, match="region|Region|bounds|aligned"):
        # Recompute the derived cache, so stale compute_requirements cannot
        # accidentally mask the independently forged region contract.
        corrupt = transform.InferTenstorrentComputeRequirements()(clear_requirements(corrupt))
        transform.VerifyTenstorrentDeviceIR()(corrupt)


def forge_fp32_output(mod):
    import inspect

    from tilelang.tenstorrent.device_ir import DFBDescriptor, TensorDescriptor
    from testing.python.target.test_tilelang_tenstorrent_compute_value_verifier import rewrite_calls

    def replace(node, cls, **changes):
        fields = {name: getattr(node, name) for name in inspect.signature(cls.__init__).parameters if name != "self"}
        fields.update(changes)
        return cls(**fields)

    tensors = list(mod.attrs["tt.tensor_table"])
    target = next(i for i, tensor in enumerate(tensors) if tensor.effect == "output")
    tensor_id = int(tensors[target].global_arg_index)
    tensors[target] = replace(tensors[target], TensorDescriptor, dtype="float32")
    dfbs = list(mod.attrs["tt.dfb_table"])
    output = next(
        i for i, dfb in enumerate(dfbs) if dfb.tensor_backing is not None and int(dfb.tensor_backing.global_arg_index) == tensor_id
    )
    dfb_id = int(dfbs[output].dfb_id)
    dfbs[output] = replace(dfbs[output], DFBDescriptor, element_dtype="float32")
    mod = mod.with_attr("tt.tensor_table", tensors).with_attr("tt.dfb_table", dfbs).without_attr("tt.l1_payload_bytes")
    symbol, ncrisc = slot_function(mod, "ncrisc")
    buffers = dict(ncrisc.buffer_map)
    for param, buffer in list(buffers.items()):
        position = list(ncrisc.params).index(param)
        if int(ncrisc.attrs["tt.tensor_arg_indices"][position]) == tensor_id:
            buffers[param] = tirx.decl_buffer(buffer.shape, "float32", name=buffer.name, data=buffer.data)
    mod.update_func(symbol, tirx.PrimFunc(ncrisc.params, ncrisc.body, ncrisc.ret_type, buffers, ncrisc.attrs, ncrisc.span))

    def remove_final_cast(call):
        if call.op.name == "tl.tt.dfb_compute" and int(call.args[0]) == dfb_id:
            attrs = dict(call.annotations)
            expression = attrs["tt.expression"]
            attrs["tt.expression"] = (
                expression.value
                if isinstance(expression, tirx.Cast) and str(expression.value.dtype) == "float32"
                else tirx.Cast("float32", expression)
            )
            attrs["tt.compute_dtype"] = tirx.StringImm("float32")
            attrs["tt.compute_kind"] = tirx.StringImm("elementwise")
            return tirx.Call(call.dtype, call.op, call.args, attrs, call.span)
        return None

    return rewrite_calls(mod, remove_final_cast)


def test_raw_final_tensor_dtype_cannot_bypass_accumulator_capability():
    from testing.python.target.test_tilelang_tenstorrent_frontend02 import gemm_epilogue

    mod = lower_tenstorrent_ir(tvm.IRModule({"main": gemm_epilogue("inplace")}), TARGET)
    corrupt = forge_fp32_output(mod)
    with pytest.raises(ValueError, match="GEMM final output dtype"):
        corrupt = transform.InferTenstorrentComputeRequirements()(clear_requirements(corrupt))
        transform.VerifyTenstorrentDeviceIR()(corrupt)


def test_copy_kind_cannot_hide_an_unrelated_expression():
    from testing.python.target.test_tilelang_tenstorrent_compute_value_verifier import check_rejected, rewrite_calls

    mod = device()
    changed = False

    def substitute_expression(call):
        nonlocal changed
        if not changed and call.op.name == "tl.tt.compute_value" and len(call.args) == 2:
            changed = True
            attrs = dict(call.annotations)
            attrs["tt.compute_kind"] = tirx.StringImm("copy")
            attrs["tt.expression"] = tirx.FloatImm("float32", 7)
            return tirx.Call(call.dtype, call.op, call.args, attrs, call.span)
        return None

    corrupt = rewrite_calls(mod, substitute_expression)
    assert changed
    check_rejected(corrupt, "expression|unused|copy")


def test_raw_dfb_to_fragment_reentry_retains_accumulator_output_contract():
    @T.prim_func
    def main(A: T.Tensor((32, 32), "bfloat16"), B: T.Tensor((32, 32), "bfloat16"), C: T.Tensor((32, 32), "bfloat16")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            b = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            acc = T.alloc_fragment((32, 32), "float32")
            storage = T.alloc_shared((32, 32), "bfloat16", annotations=METADATA)
            reloaded = T.alloc_fragment((32, 32), "bfloat16")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, acc, clear_accum=True)
            for i, j in T.Tiles(storage):
                storage[i, j] = T.cast(acc[i, j], "bfloat16")
            T.copy(storage, reloaded)
            T.copy(reloaded, C)

    mod = lower_tenstorrent_ir(tvm.IRModule({"main": main}), TARGET)
    # The body must prove provenance even when the reloaded descriptor has
    # no explicit accumulator ID. DFB materialization cannot erase that proof.
    assert any(int(value.accumulator_id) == -1 for value in mod.attrs["tt.compute_value_table"])
    corrupt = forge_fp32_output(mod)
    with pytest.raises(ValueError, match="GEMM final output dtype"):
        corrupt = transform.InferTenstorrentComputeRequirements()(clear_requirements(corrupt))
        transform.VerifyTenstorrentDeviceIR()(corrupt)

"""Device contracts needed by attention, including adversarial metadata edits."""

import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import lower_tenstorrent_ir, transform
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_compute_value_verifier import check_rejected, device, rewrite_calls
from testing.python.target.test_tilelang_tenstorrent_frontend02 import METADATA, TARGET


def attention_row_program(dtype="float32", cores=1):
    @T.prim_func
    def main(A: T.Tensor((2, 1, cores * 32, 32), dtype), C: T.Tensor((2, 1, cores * 32, 32), dtype)):
        with T.Kernel(cores, 1, threads=1) as (x, y):
            a = T.alloc_shared((32, 32), dtype, annotations=METADATA)
            row = T.alloc_fragment((32, 1), dtype)
            column = T.alloc_shared((32, 32), dtype, annotations=METADATA)
            output = T.alloc_shared((32, 32), dtype, annotations=METADATA)
            T.copy(A[1, 0, x * 32 : (x + 1) * 32, :], a)
            T.reduce_sum(a, row, dim=1, clear=True)
            T.copy(row, column[:, 0:1])
            for i, j in T.Tiles(output):
                output[i, j] = column[i, 0]
            T.copy(output, C[1, 0, x * 32 : (x + 1) * 32, :])

    return lower_tenstorrent_ir(tvm.IRModule({"main": main}), TARGET)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("cores", [1, 2])
def test_attention_row_metadata_roundtrip(dtype, cores):
    mod = attention_row_program(dtype, cores)
    assert int(mod.attrs["tt.device_ir_version"]) == 7
    assert len(mod.functions) == cores * 3
    assert all(len(tensor.shape) == 4 for tensor in mod.attrs["tt.tensor_table"])
    assert any(tuple(int(n) for n in value.buffer.shape) == (32, 1) for value in mod.attrs["tt.compute_value_table"])
    restored = ir.load_json(ir.save_json(mod))
    transform.VerifyTenstorrentDeviceIR()(restored)
    assert ir.structural_equal(mod, restored)


@pytest.mark.parametrize("cores", [1, 2])
def test_tensor_rank_mapping_rejects_non_singleton_axis(cores):
    changed = False

    def edit(call):
        nonlocal changed
        if changed or call.op.name != "tl.tt.tensor_to_dfb_nd":
            return None
        changed = True
        args = list(call.args)
        args[2] = tirx.IntImm(args[2].dtype, 0)
        args[3] = tirx.IntImm(args[3].dtype, 2)
        return tirx.Call(call.dtype, call.op, args, call.annotations, call.span)

    corrupted = rewrite_calls(attention_row_program(cores=cores), edit)
    assert changed
    check_rejected(corrupted, "leading singleton")


def test_shared_only_tensor_rank_mapping_keeps_slice_checks():
    from testing.python.target.test_tilelang_tenstorrent_attention_regions import tensor_slice_program

    mod = lower_tenstorrent_ir(tvm.IRModule({"main": tensor_slice_program()}), TARGET)
    assert int(mod.attrs["tt.device_ir_version"]) == 2

    def edit(call):
        if call.op.name != "tl.tt.tensor_to_dfb_nd":
            return None
        args = list(call.args)
        args[2] = tirx.IntImm(args[2].dtype, 0)
        args[3] = tirx.IntImm(args[3].dtype, 2)
        return tirx.Call(call.dtype, call.op, args, call.annotations, call.span)

    check_rejected(rewrite_calls(mod, edit), "leading singleton")


@pytest.mark.parametrize("corruption", ["shape", "accumulation"])
def test_keepdim_reduction_contract_is_independently_verified(corruption):
    changed = False

    def edit(call):
        nonlocal changed
        if changed or call.op.name != "tl.tt.dfb_compute" or call.annotations["tt.compute_kind"].value != "reduce":
            return None
        changed = True
        attrs = dict(call.annotations)
        if corruption == "shape":
            attrs["tt.logical_domain"] = [tirx.IntImm("int32", 32), tirx.IntImm("int32", 32)]
        else:
            attrs["tt.accum_dtype"] = tirx.StringImm("bfloat16")
        return tirx.Call(call.dtype, call.op, call.args, attrs, call.span)

    corrupted = rewrite_calls(attention_row_program("bfloat16"), edit)
    assert changed
    check_rejected(corrupted, "reduction output logical shape|unsupported accumulation dtype")


def test_value_geometry_proof_ids_cannot_be_used_as_real_dfbs():
    changed = False

    def edit(call):
        nonlocal changed
        if changed or call.op.name != "tl.tt.compute_value" or not len(call.annotations["tt.value_inputs"]):
            return None
        changed = True
        attrs = dict(call.annotations)

        def forge_load(node):
            if node.op.name == "tl.tt.compute_value_load":
                return tirx.call_intrin(node.dtype, ir.Op.get("tl.tt.dfb_load"), 0)
            return None

        attrs["tt.expression"] = tirx.stmt_functor.ir_transform(
            tirx.Evaluate(attrs["tt.expression"]), None, forge_load, ["tirx.Call"]
        ).value
        return tirx.Call(call.dtype, call.op, call.args, attrs, call.span)

    corrupted = rewrite_calls(device(), edit)
    assert changed
    check_rejected(corrupted, "undeclared input DFB")

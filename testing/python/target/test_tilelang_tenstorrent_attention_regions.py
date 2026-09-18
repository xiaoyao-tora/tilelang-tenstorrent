"""Attention transfer views and provably unobserved row-statistic padding."""

import numpy as np
import pytest

from tilelang import tvm
from tilelang.tenstorrent import language as T
from tilelang.tenstorrent import lower_tenstorrent_ir, transform
from tvm import ir, tirx

from testing.python.target.test_tilelang_tenstorrent_lower_composition import interpret

TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
METADATA = {"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2}


def tensor_slice_program(mode="valid"):
    @T.prim_func
    def main(A: T.Tensor((2, 3, 64, 64), "float32"), C: T.Tensor((2, 3, 64, 64), "float32")):
        with T.Kernel(1, 1, threads=1) as (x, y):
            tile = T.alloc_shared((32, 32), "float32", annotations=METADATA)
            if mode == "non_singleton":
                T.copy(A[0:2, 2:3, 32:64, 0:32], tile)
            elif mode == "suffix_mismatch":
                T.copy(A[1:2, 2:3, 32:64, 0:16], tile)
            elif mode == "out_of_bounds":
                T.copy(A[1:2, 3:4, 32:64, 0:32], tile)
            elif mode == "unit_coordinates":
                T.copy(A[x + 1 : x + 2, y + 2 : y + 3, 32:64, 0:32], tile)
            else:
                T.copy(A[1:2, 2:3, 32:64, 0:32], tile)
            for i, j in T.Tiles(32, 32):
                tile[i, j] = tile[i, j] + T.float32(1)
            T.copy(tile, C[0:1, 1:2, 0:32, 32:64])

    return main


def bound(func):
    return tirx.transform.BindTarget(TARGET)(tvm.IRModule({"main": func}))


@pytest.mark.parametrize("mode", ["valid", "unit_coordinates"])
def test_singleton_tensor_axes_preserve_original_abi_and_slice_coordinates(mode):
    frontend = bound(tensor_slice_program(mode))
    normalized = transform.NormalizeTenstorrentRegions()(transform.NormalizeTenstorrentLaunch()(frontend))
    ir.assert_structural_equal(transform.NormalizeTenstorrentRegions()(normalized), normalized)
    for parameter, buffer in frontend["main"].buffer_map.items():
        assert normalized["main"].buffer_map[parameter].same_as(buffer)
    result = lower_tenstorrent_ir(tvm.IRModule({"main": tensor_slice_program(mode)}), TARGET)
    assert "tt.device_ir_version" in result.attrs
    specs = result.attrs["tt.tensor_table"]
    assert all([int(axis) for axis in spec.shape] == [2, 3, 64, 64] for spec in specs)
    transform.VerifyTenstorrentDeviceIR()(result)
    ir.assert_structural_equal(ir.load_json(ir.save_json(result)), result)
    source = np.arange(2 * 3 * 64 * 64, dtype="float32").reshape(2, 3, 64, 64)
    output = np.full(source.shape, -17, dtype="float32")
    expected = output.copy()
    expected[0, 1, :32, 32:64] = source[1, 2, 32:64, :32] + 1
    tensors, _ = interpret(result, [source, output])
    np.testing.assert_array_equal(tensors[1], expected)


@pytest.mark.parametrize(
    "mode,message",
    [
        ("non_singleton", "leading Tensor slice axes"),
        ("suffix_mismatch", "extents differ"),
        ("out_of_bounds", "out of bounds"),
    ],
)
def test_tensor_axis_removal_does_not_flatten_or_hide_invalid_regions(mode, message):
    with pytest.raises(ValueError, match=message):
        transform.NormalizeTenstorrentRegions()(bound(tensor_slice_program(mode)))


@pytest.mark.parametrize("local_source", [False, True])
def test_rank_mapping_is_limited_to_leading_tensor_axes(local_source):
    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            wide = T.alloc_shared((1, 32, 32), "float32", annotations=METADATA)
            tile = T.alloc_shared((32, 32), "float32", annotations=METADATA)
            if local_source:
                T.copy(wide[:, :, :], tile[:, :])
            else:
                T.copy(A[:, :], wide[:, :, :])

    with pytest.raises(ValueError, match="region ranks differ"):
        transform.NormalizeTenstorrentRegions()(bound(main))


def row_padding_program(mode="column_zero"):
    @T.prim_func
    def main(A: T.Tensor((32, 32), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((32, 32), "float32", annotations=METADATA)
            row = T.alloc_fragment((32, 1), "float32")
            padded = T.alloc_shared((32, 32), "float32", annotations=METADATA)
            output = T.alloc_shared((32, 32), "float32", annotations=METADATA)
            T.copy(A, a)
            T.reduce_sum(a, row, dim=1)
            T.copy(row, padded[:, 0:1])
            if mode == "full_copy":
                T.copy(padded, C)
            elif mode == "other_write":
                for i, j in T.Parallel(32, 32):
                    padded[i, j] = a[i, j]
            else:
                for i, j in T.Parallel(32, 32):
                    if mode == "nonzero_column":
                        output[i, j] = padded[i, j]
                    else:
                        output[i, j] = padded[i, 0]
                T.copy(output, C)

    return main


@pytest.mark.parametrize("mode", ["column_zero", "full_copy", "nonzero_column", "other_write"])
def test_padding_extension_requires_all_uses_to_observe_only_column_zero(mode):
    result = transform.CanonicalizeTTElementwise()(bound(row_padding_program(mode)))
    ir.assert_structural_equal(transform.CanonicalizeTTElementwise()(result), result)
    calls = []
    blocks = []
    for func in result.functions.values():

        def visit(node):
            if isinstance(node, tirx.Call) and node.op.name == "tl.tileop.copy":
                calls.append(node)
            if isinstance(node, tirx.SBlock) and "tl.tt.compute_kind" in node.annotations:
                blocks.append(node)

        tirx.stmt_functor.post_order_visit(func.body, visit)
    padding_blocks = [block for block in blocks if str(block.writes[0].buffer.name) == "padded"]
    if mode == "column_zero":
        assert len(calls) == 2
        assert len(padding_blocks) == 1
        assert [int(axis.extent) for axis in padding_blocks[0].reads[0].region] == [32, 1]
        assert [int(axis.extent) for axis in padding_blocks[0].writes[0].region] == [32, 32]
    else:
        assert len(calls) == (2 if mode == "other_write" else 3)
    transform.VerifyTTComputeBlocks()(result)


@pytest.mark.parametrize("through_match_buffer", [False, True])
def test_padding_extension_rejects_opaque_pointer_escape(through_match_buffer):
    mod = bound(row_padding_program("full_copy"))

    def replace_output_copy(node):
        if not isinstance(node, tirx.Evaluate) or not isinstance(node.value, tirx.Call):
            return None
        call = node.value
        if call.op.name != "tl.tileop.copy":
            return None
        source = call.args[0].args[0]
        if source.buffer.name == "padded":
            if through_match_buffer:
                alias = tirx.decl_buffer(source.buffer.shape, source.buffer.dtype, name="alias", scope=source.buffer.scope())
                region = tirx.BufferRegion(source.buffer, [ir.Range.from_min_extent(0, extent) for extent in source.buffer.shape])
                block = tirx.SBlock(
                    [],
                    [],
                    [],
                    "alias_escape",
                    tirx.Evaluate(tirx.call_extern("int32", "inspect_padding", alias.data)),
                    match_buffers=[tirx.MatchBufferRegion(alias, region)],
                )
                return tirx.SBlockRealize([], True, block)
            return tirx.Evaluate(tirx.call_extern("int32", "inspect_padding", source.buffer.data))
        return None

    func = mod["main"]
    func = func.with_body(tirx.stmt_functor.ir_transform(func.body, None, replace_output_copy, ["tirx.Evaluate"]))
    result = transform.CanonicalizeTTElementwise()(tvm.IRModule({"main": func}))
    copies = []
    tirx.stmt_functor.post_order_visit(
        result["main"].body,
        lambda node: copies.append(node) if isinstance(node, tirx.Call) and node.op.name == "tl.tileop.copy" else None,
    )
    assert len(copies) == 2  # Input load and the original partial row store.

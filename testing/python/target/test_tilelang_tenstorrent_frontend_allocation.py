"""Buffer identity and allocation metadata across frontend IR representations."""

import pytest
from tilelang import transform, tvm
from tilelang.tenstorrent.transform import NormalizeTenstorrentBufferMetadata
from tvm import ir, tirx


def _module(annotations=None):
    first = tirx.decl_buffer((32, 64), "bfloat16", name="same_name", scope="shared")
    second = tirx.decl_buffer((32, 64), "bfloat16", name="same_name", scope="shared")
    block = tirx.SBlock(
        [],
        [],
        [],
        "root",
        tirx.Evaluate(0),
        alloc_buffers=[first, second],
        annotations={
            "tl.alloc_buffer_annotations": {
                first.data: annotations if annotations is not None else {"tt.dfb_block_count": tirx.IntImm("int32", 3)},
                second.data: {"tt.dfb_block_count": tirx.IntImm("int32", 2)},
            }
        },
    )
    return tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, block))})


def test_normalize_preserves_allocation_contract_after_lower_opaque_block():
    original = _module()
    normalize = NormalizeTenstorrentBufferMetadata()
    before_table = normalize(original)["main"].attrs["tt.buffer_metadata_table"]
    lowered = transform.LowerOpaqueBlock()(original)
    result = normalize(lowered)
    after_table = result["main"].attrs["tt.buffer_metadata_table"]
    assert len(after_table) == len(before_table) == 2
    for before, after in zip(before_table, after_table):
        assert before.buffer.data.same_as(after.buffer.data)
        assert before.buffer_id == after.buffer_id
        assert int(before.dfb_block_count) == int(after.dfb_block_count)
        assert before.block_count_origin == after.block_count_origin == "explicit"
    ir.assert_structural_equal(normalize(result), result)
    normalized_then_lowered = transform.LowerOpaqueBlock()(normalize(original))
    ir.assert_structural_equal(normalize(normalized_then_lowered), normalized_then_lowered)


@pytest.mark.parametrize("lowered", [False, True])
def test_normalize_rejects_table_missing_allocation_identity(lowered):
    mod = _module()
    if lowered:
        mod = transform.LowerOpaqueBlock()(mod)
    normalize = NormalizeTenstorrentBufferMetadata()
    normalized = normalize(mod)
    table = normalized["main"].attrs["tt.buffer_metadata_table"]
    malformed = tvm.IRModule({"main": normalized["main"].with_attr("tt.buffer_metadata_table", table[:1])})
    before = ir.save_json(malformed)
    with pytest.raises(ValueError, match="omits allocated Buffer"):
        normalize(malformed)
    assert ir.save_json(malformed) == before


@pytest.mark.parametrize("lowered", [False, True])
@pytest.mark.parametrize(
    "annotations",
    [
        {"tt.dfb_block_count": tirx.IntImm("bool", True)},
        {"tt.tile_shape": [tirx.IntImm("bool", True), tirx.IntImm("int32", 32)]},
    ],
)
def test_normalize_rejects_boolean_integer_metadata(lowered, annotations):
    mod = _module(annotations)
    if lowered:
        mod = transform.LowerOpaqueBlock()(mod)
    with pytest.raises(ValueError, match="tt.dfb_block_count|tt.tile_shape"):
        NormalizeTenstorrentBufferMetadata()(mod)

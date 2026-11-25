"""Physical storage survives frontend normalization and descriptor families."""

import pytest

from tilelang import transform as common_transform, tvm
from tilelang.tenstorrent import device_ir as D, language as T, transform
from tilelang.tenstorrent.language.annotations import annotate_storage_format
from tilelang.language.allocate import _normalize_alloc_shared_annotations
from tilelang.tenstorrent.pipeline import TenstorrentPassPipelineBody
from tvm import ir, tirx

from testing.python.tenstorrent.device_ir.test_tilelang_tenstorrent_compact_tables import _dfb, _normalized_tables


def program(input_format, output_format):
    @T.prim_func
    def main(A: T.Tensor((64, 64), "bfloat16"), C: T.Tensor((64, 64), "bfloat16")):
        T.annotate_storage_format({A: input_format, C: output_format})
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((64, 64), "bfloat16", annotations={"tt.storage_format": input_format})
            c = T.alloc_shared((64, 64), "bfloat16", annotations={"tt.storage_format": output_format})
            T.copy(A, a)
            for i, j in T.Tiles(c):
                c[i, j] = a[i, j] + a[i, j]
            T.copy(c, C)

    return main


@pytest.mark.parametrize("formats", [("", ""), ("bfp_bf8", "bfp_bf8"), ("bfp_bf4", "bfp_bf4"), ("bfp_bf8", "bfp_bf4")])
@pytest.mark.parametrize("opaque", [False, True])
def test_frontend_storage_metadata(formats, opaque):
    mod = tvm.IRModule({"main": program(*formats)})
    if opaque:
        mod = common_transform.LowerOpaqueBlock()(mod)
    mod = transform.NormalizeTenstorrentBufferMetadata()(mod)
    metadata = mod["main"].attrs["tt.buffer_metadata_table"]
    assert [str(item.storage_format) for item in metadata] == list(formats) * 2
    assert all(str(item.buffer.dtype) == "bfloat16" for item in metadata)
    ir.assert_structural_equal(mod, ir.load_json(ir.save_json(mod)))


@pytest.mark.parametrize("formats", [("", ""), ("bfp_bf8", "bfp_bf8"), ("bfp_bf4", "bfp_bf4"), ("bfp_bf8", "bfp_bf4")])
def test_current_family_storage_roundtrip(formats):
    target = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
    mod = TenstorrentPassPipelineBody(tvm.IRModule({"main": program(*formats)}), target)
    assert [str(t.storage_format) for t in mod.attrs["tt.tensor_table"]] == list(formats)
    assert {str(g.storage_format) for g in mod.attrs["tt.dfb_storage_groups"]} == set(formats)
    ir.assert_structural_equal(mod, ir.load_json(ir.save_json(mod)))
    for restored in (mod, ir.load_json(ir.save_json(mod)), tvm.script.from_source(D.device_ir_to_tirx(mod))):
        for table in ("tt.tensor_table", "tt.dfb_storage_groups"):
            ir.assert_structural_equal(mod.attrs[table], restored.attrs[table], map_free_vars=True)
        expanded = D.expand_current_device_tables(restored)
        assert {str(d.storage_format) for d in expanded.attrs["tt.dfb_table"]} == set(formats)
        assert all(str(d.element_dtype) == "bfloat16" for d in expanded.attrs["tt.dfb_table"])
        transform.VerifyTenstorrentDeviceIR()(expanded)


@pytest.mark.parametrize("format", ["", "bfp_bf8", "bfp_bf4"])
def test_legacy_dfb_family_preserves_storage(format):
    import inspect

    prototype = _dfb(
        0, "buffer.0", span=ir.Span(ir.SourceName("storage.py"), 1, 1, 1, 10), domain=D.CoreDomain((0, 0), (1, 1))
    )
    fields = {name: getattr(prototype, name) for name in inspect.signature(D.DFBDescriptor.__init__).parameters if name != "self"}
    descriptors = [
        D.DFBDescriptor(**(fields | {"dfb_id": i, "source_buffer_identity": f"buffer.{i}", "storage_format": f}))
        for i, f in enumerate((format, format, "bfp_bf4" if format != "bfp_bf4" else "bfp_bf8"))
    ]
    original = _normalized_tables(dfbs=descriptors)
    compact = D.compact_device_ir(original, force=True)
    assert len(compact.attrs["tt.compact_dfb_families"]) == 2
    for encoded in (compact, ir.load_json(ir.save_json(compact))):
        ir.assert_structural_equal(original, D.expand_device_ir(encoded))


@pytest.mark.parametrize("format,error", [(1, TypeError), ("bfp_bad", ValueError)])
def test_frontend_rejects_invalid_format(format, error):
    buffer = tirx.decl_buffer((32, 32), "bfloat16")
    with pytest.raises(error, match="storage_format"):
        annotate_storage_format({buffer: format})
    shared = tirx.decl_buffer((32, 32), "bfloat16", scope="shared")
    with pytest.raises(error, match="storage_format"):
        _normalize_alloc_shared_annotations(shared, {"tt.storage_format": format})


def test_frontend_rejects_non_bfloat16():
    with pytest.raises(ValueError, match="logical bfloat16"):
        annotate_storage_format({tirx.decl_buffer((32, 32), "float32"): "bfp_bf8"})
    with pytest.raises(ValueError, match="logical bfloat16"):
        _normalize_alloc_shared_annotations(tirx.decl_buffer((32, 32), "float32", scope="shared"), {"tt.storage_format": "bfp_bf4"})


def test_normalize_rejects_dangling_tensor_format():
    func = program("", "").with_attr("tt.tensor_storage_formats", {tirx.Var("other", "handle"): "bfp_bf8"})
    with pytest.raises(ValueError, match="non-parameter"):
        transform.NormalizeTenstorrentBufferMetadata()(tvm.IRModule({"main": func}))
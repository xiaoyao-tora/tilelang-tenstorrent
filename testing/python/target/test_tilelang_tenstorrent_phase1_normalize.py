from __future__ import annotations

import pytest
from tvm import ir, tirx

from tilelang import tvm
from tilelang.tenstorrent import language as T

from testing.python.target.test_tilelang_tenstorrent_phase0_contract import (
    FRONTEND_PROGRAMS,
)


TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})
PREFIX = "tl.tenstorrent.transform."


def _bound(name: str):
    mod = tvm.IRModule({name: FRONTEND_PROGRAMS[name]})
    return tirx.transform.BindTarget(TARGET)(mod)


def _apply(name: str, mod):
    return tvm.get_global_func(PREFIX + name)()(mod)


def _normalized_through_layout(name: str):
    mod = _bound(name)
    for pass_name in (
        "ValidateTenstorrentFrontendIR",
        "NormalizeTenstorrentLaunch",
        "NormalizeTenstorrentBufferMetadata",
        "NormalizeTenstorrentRegions",
        "InferTenstorrentTensorLayout",
    ):
        mod = _apply(pass_name, mod)
    return mod


def test_validate_is_read_only_and_deterministic():
    mod = _bound("add")
    before = str(mod)
    validated = _apply("ValidateTenstorrentFrontendIR", mod)
    ir.assert_structural_equal(validated, mod)
    assert str(validated) == before
    assert str(_apply("ValidateTenstorrentFrontendIR", validated)) == before


def test_validate_rejects_missing_launch_as_malformed():
    @T.prim_func
    def no_launch():
        T.evaluate(0)

    mod = tirx.transform.BindTarget(TARGET)(tvm.IRModule({"no_launch": no_launch}))
    with pytest.raises(ValueError, match="Malformed Tenstorrent frontend IR"):
        _apply("ValidateTenstorrentFrontendIR", mod)


def test_launch_normalizes_static_2d_and_is_idempotent():
    mod = _apply("NormalizeTenstorrentLaunch", _bound("p2p"))
    func = mod["p2p"]
    assert [int(value) for value in func.attrs["tt.launch_grid"]] == [2, 1]
    outer = func.body
    assert isinstance(outer, tirx.For)
    assert outer.kind == tirx.ForKind.SERIAL
    assert str(outer.annotations["tt.logical_core_axis"]) == "x"
    inner = outer.body
    assert isinstance(inner, tirx.For)
    assert inner.kind == tirx.ForKind.SERIAL
    assert str(inner.annotations["tt.logical_core_axis"]) == "y"

    twice = _apply("NormalizeTenstorrentLaunch", mod)
    ir.assert_structural_equal(twice, mod)
    assert str(twice) == str(mod)


def test_launch_rejects_non_unit_threads_as_unsupported():
    @T.prim_func
    def threaded():
        with T.Kernel(1, 1, threads=2):
            T.evaluate(0)

    mod = tirx.transform.BindTarget(TARGET)(tvm.IRModule({"threaded": threaded}))
    with pytest.raises(NotImplementedError, match="requires threads=1"):
        _apply("NormalizeTenstorrentLaunch", mod)


def test_buffer_metadata_is_typed_stable_and_preserves_buffer_identity():
    mod = _apply("NormalizeTenstorrentLaunch", _bound("add"))
    mod = _apply("NormalizeTenstorrentBufferMetadata", mod)
    func = mod["add"]
    table = func.attrs["tt.buffer_metadata_table"]

    assert [str(item.buffer_id) for item in table] == [
        "tensor.0",
        "tensor.1",
        "tensor.2",
        "buffer.0",
        "buffer.1",
        "buffer.2",
    ]
    assert [str(item.kind) for item in table[:3]] == ["tensor"] * 3
    assert [str(item.kind) for item in table[3:]] == ["logical_dfb_candidate"] * 3
    for index, parameter in enumerate(func.params):
        assert table[index].buffer.same_as(func.buffer_map[parameter])
        assert int(table[index].global_arg_index) == index
        if table[index].buffer.span is None:
            assert table[index].source_span is None
        else:
            assert table[index].source_span.same_as(table[index].buffer.span)
    assert all(str(item.tile_shape_origin) == "explicit" for item in table[3:])
    assert all(int(item.dfb_block_count) == 2 for item in table[3:])

    twice = _apply("NormalizeTenstorrentBufferMetadata", mod)
    ir.assert_structural_equal(twice, mod)


@pytest.mark.parametrize("name", ("add", "p2p"))
def test_regions_accept_basic_tensor_dfb_and_pipe_paths(name):
    mod = _apply("NormalizeTenstorrentLaunch", _bound(name))
    once = _apply("NormalizeTenstorrentRegions", mod)
    twice = _apply("NormalizeTenstorrentRegions", once)
    ir.assert_structural_equal(twice, once)
    assert str(twice) == str(once)
    if name == "add":
        script = str(once)
        assert "tt.transfer_kind" in script
        assert "tensor_to_dfb" in script
        assert "dfb_to_tensor" in script


def test_basic_layout_infers_32x32_bfloat16_metadata_and_round_trips():
    mod = _normalized_through_layout("add")
    table = mod["add"].attrs["tt.buffer_metadata_table"]
    for item in table:
        assert [int(value) for value in item.tile_shape] == [32, 32]
        assert [int(value) for value in item.tile_grid_shape] == [1, 1]
        assert str(item.memory_layout) == "interleaved"
        assert item.buffer.same_as(item.buffer)
        if item.buffer.span is None:
            assert item.source_span is None
        else:
            assert item.source_span.same_as(item.buffer.span)

    restored = ir.load_json(ir.save_json(mod))
    ir.assert_structural_equal(restored, mod)
    twice = _apply("InferTenstorrentTensorLayout", mod)
    ir.assert_structural_equal(twice, mod)


def test_layout_rejects_unsupported_dtype():
    @T.prim_func
    def int_tensor(A: T.Tensor((32, 32), T.int32)):
        with T.Kernel(1, 1, threads=1):
            tmp = T.alloc_shared((32, 32), T.int32)
            T.copy(A, tmp)

    mod = tirx.transform.BindTarget(TARGET)(tvm.IRModule({"int_tensor": int_tensor}))
    mod = _apply("NormalizeTenstorrentLaunch", mod)
    mod = _apply("NormalizeTenstorrentBufferMetadata", mod)
    with pytest.raises(NotImplementedError, match="supports bfloat16 and float32"):
        _apply("InferTenstorrentTensorLayout", mod)

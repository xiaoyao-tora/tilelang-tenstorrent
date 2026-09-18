import pytest

import tilelang
from tilelang import tvm
from tilelang.tenstorrent import language as T


def _collect_loops(func):
    loops = []

    def collect(node):
        if isinstance(node, tvm.tirx.For):
            loops.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, collect)
    return loops


def _tiles_loops(func):
    loops = _collect_loops(func)
    outer = next(loop for loop in loops if "tl.tt.tiles_scope" in loop.annotations)
    inner = next(loop for loop in loops if loop is not outer)
    return outer, inner


def _assert_tiles_annotations(outer, inner, domain, parallel):
    assert int(outer.annotations["tl.tt.tiles_scope"].value) == 1
    assert [int(extent) for extent in outer.annotations["tl.tt.tiles_domain"]] == domain
    assert "tl.tt.tiles_scope" not in inner.annotations
    assert "tl.tt.tiles_domain" not in inner.annotations

    for loop in (outer, inner):
        assert loop.kind == tvm.tirx.ForKind.SERIAL
        assert int(loop.annotations["tl.tt.tiles_parallel"].value) == parallel
        assert int(loop.annotations["tl.tt.tiles_stage"].value) == 0


def test_tiles_buffer_domain_constructs_annotated_serial_loops():
    @T.prim_func
    def main(A: T.Tensor((4, 8), T.float32), B: T.Tensor((4, 8), T.float32)):
        for i, j in T.Tiles(B):
            B[i, j] = A[i, j]

    outer, inner = _tiles_loops(main)
    assert [int(outer.extent), int(inner.extent)] == [4, 8]
    _assert_tiles_annotations(outer, inner, [4, 8], 1)

    store = inner.body
    assert isinstance(store, tvm.tirx.BufferStore)
    assert store.indices[0].same_as(outer.loop_var)
    assert store.indices[1].same_as(inner.loop_var)
    assert isinstance(store.value, tvm.tirx.BufferLoad)
    assert store.value.indices[0].same_as(outer.loop_var)
    assert store.value.indices[1].same_as(inner.loop_var)


def test_tiles_explicit_domain_and_parallel_false():
    @T.prim_func
    def main(A: T.Tensor((4, 8), T.float32), B: T.Tensor((4, 8), T.float32)):
        for i, j in T.Tiles([4, 8], parallel=False):
            B[i, j] = A[i, j]

    outer, inner = _tiles_loops(main)
    assert [int(outer.extent), int(inner.extent)] == [4, 8]
    _assert_tiles_annotations(outer, inner, [4, 8], 0)


@pytest.mark.parametrize("domain", [[], [4], [4, 8, 16]])
def test_tiles_rejects_unsupported_domain_rank(domain):
    expected = "non-empty" if not domain else "rank 2"
    with pytest.raises(ValueError, match=expected):

        @T.prim_func
        def main():
            for _, _ in T.Tiles(domain):
                T.evaluate(0)


def test_tiles_rejects_non_iterable_domain():
    with pytest.raises(TypeError, match="Buffer or an iterable"):

        @T.prim_func
        def main():
            for _, _ in T.Tiles(4):
                T.evaluate(0)


@pytest.mark.parametrize("parallel", [0, 1, None, "true"])
def test_tiles_rejects_non_bool_parallel(parallel):
    with pytest.raises(TypeError, match="parallel must be a bool"):

        @T.prim_func
        def main():
            for _, _ in T.Tiles([4, 8], parallel=parallel):
                T.evaluate(0)


def test_tiles_is_tenstorrent_dialect_only():
    assert "Tiles" in T.__all__
    assert T.Tiles is tilelang.language.loop.Tiles
    assert "Tiles" not in tilelang.language.__all__
    assert not hasattr(tilelang.language, "Tiles")


def test_parallel_construction_is_unchanged():
    @T.prim_func
    def main():
        for i, j in T.Parallel(4, 8):
            T.evaluate(i + j)

    loops = _collect_loops(main)
    assert len(loops) == 2
    assert all(loop.kind == tvm.tirx.ForKind.PARALLEL for loop in loops)
    assert sum(bool(loop.annotations) for loop in loops) == 0

import inspect

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


@pytest.mark.parametrize("args", [((32, 32),), ([32, 32],), (32, 32), (tvm.tirx.decl_buffer((32, 32), "float32"),)])
def test_tiles_parallel_false_is_rejected_before_ir_construction(monkeypatch, args):
    def unexpected_builder_call(*args, **kwargs):
        pytest.fail("Unsupported parallel=False reached the IR builder")

    monkeypatch.setattr(tilelang.language.loop._ffi_api, "Tiles", unexpected_builder_call)
    with pytest.raises(NotImplementedError, match="requires parallel=True"):
        T.Tiles(*args, parallel=False)


def test_tiles_domain_forms_construct_equivalent_ir():
    @T.prim_func
    def from_buffer(A: T.Tensor((64, 128), T.float32), B: T.Tensor((64, 128), T.float32)):
        for i, j in T.Tiles(B):
            B[i, j] = A[i, j]

    @T.prim_func
    def from_list(A: T.Tensor((64, 128), T.float32), B: T.Tensor((64, 128), T.float32)):
        for i, j in T.Tiles([64, 128]):
            B[i, j] = A[i, j]

    @T.prim_func
    def from_tuple(A: T.Tensor((64, 128), T.float32), B: T.Tensor((64, 128), T.float32)):
        for i, j in T.Tiles((64, 128)):
            B[i, j] = A[i, j]

    @T.prim_func
    def from_extents(A: T.Tensor((64, 128), T.float32), B: T.Tensor((64, 128), T.float32)):
        for i, j in T.Tiles(64, 128):
            B[i, j] = A[i, j]

    for func in (from_list, from_tuple, from_extents):
        tvm.ir.assert_structural_equal(from_buffer.without_attr("global_symbol"), func.without_attr("global_symbol"))


@pytest.mark.parametrize("scope", ["shared", "shared.dyn", "local.fragment"])
def test_tiles_buffer_domain_needs_no_allocation_metadata(scope):
    domain = tvm.tirx.decl_buffer((64, 128), "float32", scope=scope)
    frame = T.Tiles(domain)
    assert [int(extent.extent) for extent in frame.doms] == [64, 128]


@pytest.mark.parametrize("as_buffer", [False, True])
def test_tiles_rejects_dynamic_domain(as_buffer):
    shape = (tvm.tirx.Var("n", "int32"), 32)
    domain = tvm.tirx.decl_buffer(shape, "float32") if as_buffer else shape
    with pytest.raises(ValueError, match="compile-time constant"):
        T.Tiles(domain)


def test_tiles_parallel_is_keyword_only():
    assert inspect.signature(T.Tiles).parameters["parallel"].kind == inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="one Buffer, one tuple/list"):
        T.Tiles((32, 32), False)
    with pytest.raises(TypeError, match="scalar integer"):
        T.Tiles(32, False)


@pytest.mark.parametrize("domain", [[], [4], [2, 32, 32]])
def test_tiles_rejects_unsupported_domain_rank(domain):
    expected = "non-empty" if not domain else "rank 2"
    with pytest.raises(ValueError, match=expected):

        @T.prim_func
        def main():
            for _, _ in T.Tiles(domain):
                T.evaluate(0)


def test_tiles_rejects_single_extent_domain():
    with pytest.raises(ValueError, match="rank 2"):

        @T.prim_func
        def main():
            for _, _ in T.Tiles(4):
                T.evaluate(0)


@pytest.mark.parametrize("domain", ["32", b"32", [32, 1.5], [True, 32], [32, None]])
def test_tiles_rejects_non_integer_domain(domain):
    with pytest.raises(TypeError, match="scalar integer"):
        T.Tiles(domain)


@pytest.mark.parametrize(
    "args",
    [
        (),
        (tvm.tirx.decl_buffer((32, 32), "float32"), 32),
        (32, tvm.tirx.decl_buffer((32, 32), "float32")),
        ((32, 32), 32),
        (32, 32, 32),
        (32, tvm.tirx.Var("n", "int32")),
        (32, 0),
        (True, 32),
    ],
)
def test_tiles_invalid_arguments_do_not_create_ir(monkeypatch, args):
    def unexpected_builder_call(*args, **kwargs):
        pytest.fail("Invalid domain reached the IR builder")

    monkeypatch.setattr(tilelang.language.loop._ffi_api, "Tiles", unexpected_builder_call)
    with pytest.raises((TypeError, ValueError)):
        T.Tiles(*args)


@pytest.mark.parametrize("extent", [tvm.tirx.const(True), tvm.tirx.const(32, "float32"), tvm.tirx.Broadcast(32, 2)])
def test_tiles_rejects_non_integer_scalar_expression(extent):
    with pytest.raises(TypeError, match="scalar integer"):
        T.Tiles([extent, 32])


@pytest.mark.parametrize("extent", [0, -32, tvm.tirx.const(0, "int64")])
def test_tiles_rejects_non_positive_domain(extent):
    with pytest.raises(ValueError, match="must be positive"):
        T.Tiles([extent, 32])


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

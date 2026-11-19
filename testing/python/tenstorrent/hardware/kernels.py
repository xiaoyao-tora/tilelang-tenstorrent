from __future__ import annotations

from tilelang.tenstorrent import language as T


_TILE = 32
_DFB_ANNOTATIONS = {
    "tt.tile_shape": (_TILE, _TILE),
    "tt.dfb_block_count": 2,
}


def make_add(dtype: str):
    @T.prim_func
    def add(
        A: T.Tensor((_TILE, _TILE), dtype),
        B: T.Tensor((_TILE, _TILE), dtype),
        C: T.Tensor((_TILE, _TILE), dtype),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((_TILE, _TILE), dtype, annotations=_DFB_ANNOTATIONS)
            b = T.alloc_shared((_TILE, _TILE), dtype, annotations=_DFB_ANNOTATIONS)
            c = T.alloc_shared((_TILE, _TILE), dtype, annotations=_DFB_ANNOTATIONS)
            T.copy(A, a)
            T.copy(B, b)
            for i, j in T.Tiles(c):
                c[i, j] = a[i, j] + b[i, j]
            T.copy(c, C)

    return add


def make_keepdims_reduction(dtype: str, reduction: str):
    @T.prim_func
    def reduce(
        A: T.Tensor((_TILE, _TILE), dtype),
        C: T.Tensor((_TILE, _TILE), dtype),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((_TILE, _TILE), dtype, annotations=_DFB_ANNOTATIONS)
            value = T.alloc_fragment((_TILE, _TILE), dtype)
            row = T.alloc_fragment((_TILE, 1), dtype)
            output = T.alloc_shared((_TILE, _TILE), dtype, annotations=_DFB_ANNOTATIONS)
            T.copy(A, a)
            T.copy(a, value)
            if reduction == "sum":
                T.reduce_sum(value, row, dim=1)
            elif reduction == "max":
                T.reduce_max(value, row, dim=1)
            else:
                raise ValueError(f"unsupported reduction {reduction!r}")
            for i, j in T.Tiles(output):
                output[i, j] = row[i, 0]
            T.copy(output, C)

    return reduce


def make_elementwise_chain():
    @T.prim_func
    def chain(
        A: T.Tensor((_TILE, _TILE), "float32"),
        B: T.Tensor((_TILE, _TILE), "float32"),
        C: T.Tensor((_TILE, _TILE), "float32"),
    ):
        with T.Kernel(1, 1, threads=1):
            a = T.alloc_shared((_TILE, _TILE), "float32", annotations=_DFB_ANNOTATIONS)
            b = T.alloc_shared((_TILE, _TILE), "float32", annotations=_DFB_ANNOTATIONS)
            c = T.alloc_shared((_TILE, _TILE), "float32", annotations=_DFB_ANNOTATIONS)
            T.copy(A, a)
            T.copy(B, b)
            for i, j in T.Tiles(c):
                c[i, j] = T.sqrt(a[i, j] * a[i, j] + b[i, j]) / T.float32(2)
            T.copy(c, C)

    return chain

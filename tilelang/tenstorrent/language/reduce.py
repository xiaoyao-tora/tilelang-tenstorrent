"""Preserve shared-buffer reductions for Tenstorrent TileOp legalization."""

from __future__ import annotations

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.utils import _normalize_annotations
from tilelang.utils.language import to_buffer_region, to_tile_region


def reduce(
    buffer: BufferLikeType,
    out: BufferLikeType,
    reduce_type: str,
    dim: int,
    clear: bool,
    batch: int = 1,
    nan_propagate: bool = False,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """Reduce a logical axis directly, without introducing SIMT fragments.

    Lower validates the operation, shape, dtype and accumulation contract.
    ``batch`` is the common API's AllReduce scheduling option, distinct from
    leading logical batch dimensions; only its default is supported here.
    """
    source = to_buffer_region(buffer).buffer
    rank = len(source.shape)
    if not isinstance(dim, int) or not -rank <= dim < rank:
        raise ValueError(f"Tenstorrent reduction axis must be in [-{rank}, {rank}), got {dim}")
    if not isinstance(clear, bool) or not isinstance(nan_propagate, bool):
        raise TypeError("Tenstorrent reduction clear and nan_propagate must be compile-time bools")
    if batch != 1:
        raise NotImplementedError("Tenstorrent reduction does not support AllReduce batch scheduling")
    entries = _normalize_annotations(annotations)
    if nan_propagate:
        entries["nan_propagate"] = tirx.IntImm("bool", True)
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.reduce"),
        to_tile_region(buffer, access_type="r"),
        to_tile_region(out, access_type="w" if clear else "rw"),
        reduce_type,
        dim % rank,
        clear,
        annotations=entries,
    )


def reduce_sum(buffer, out, dim=-1, clear=True, batch=1, annotations=None):
    """Sum with FP32 accumulation and a final conversion to the output dtype."""
    return reduce(buffer, out, "sum", dim, clear, batch, annotations=annotations)


def reduce_max(buffer, out, dim=-1, clear=True, batch=1, nan_propagate=False, annotations=None):
    """Maximum along an axis, with an explicit NaN propagation policy."""
    return reduce(buffer, out, "max", dim, clear, batch, nan_propagate, annotations)


def reduce_min(buffer, out, dim=-1, clear=True, batch=1, nan_propagate=False, annotations=None):
    """Minimum along an axis, with an explicit NaN propagation policy."""
    return reduce(buffer, out, "min", dim, clear, batch, nan_propagate, annotations)

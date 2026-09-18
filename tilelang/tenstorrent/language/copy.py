"""Tenstorrent copy overloads for PipeRef transfers."""

from __future__ import annotations

from typing import Any, Literal

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import copy as _common_copy
from tilelang.utils.language import to_buffer_region

from . import comm


def _reject_pipe_copy_options(
    *,
    coalesced_width: int | None,
    disable_tma: bool,
    eviction_policy: str | None,
    prefer_instruction: str | None,
    annotations: dict | None,
    loop_layout: Any | None,
) -> None:
    if (
        coalesced_width is not None
        or disable_tma
        or eviction_policy is not None
        or prefer_instruction is not None
        or annotations is not None
        or loop_layout is not None
    ):
        raise ValueError("T.copy with PipeRef does not support non-default copy options")


def _validate_payload(value: object) -> tirx.Buffer:
    if not isinstance(value, tirx.Buffer):
        raise TypeError("T.copy with PipeRef requires a complete tirx.Buffer payload")
    scope = value.scope()
    if scope not in ("shared", "shared.dyn"):
        raise ValueError(f"T.copy with PipeRef requires a shared or shared.dyn payload buffer, got scope {scope!r}")
    return value


def copy(
    src: BufferLikeType | tirx.Var,
    dst: BufferLikeType | tirx.Var,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    prefer_instruction: str | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tirx.PrimExpr | tirx.Stmt:
    """Copy buffers normally, or transfer a complete shared buffer via PipeRef."""

    src_pipe = comm._try_selected_pipe(src)
    dst_pipe = comm._try_selected_pipe(dst)
    if src_pipe is None and dst_pipe is None:
        return _common_copy(
            src,
            dst,
            coalesced_width=coalesced_width,
            disable_tma=disable_tma,
            eviction_policy=eviction_policy,
            prefer_instruction=prefer_instruction,
            annotations=annotations,
            loop_layout=loop_layout,
        )

    _reject_pipe_copy_options(
        coalesced_width=coalesced_width,
        disable_tma=disable_tma,
        eviction_policy=eviction_policy,
        prefer_instruction=prefer_instruction,
        annotations=annotations,
        loop_layout=loop_layout,
    )
    if src_pipe is not None and dst_pipe is not None:
        raise TypeError("T.copy does not support PipeRef-to-PipeRef transfers")

    if dst_pipe is not None:
        selected_pipe, net, side = dst_pipe
        if side != "src":
            raise ValueError("T.copy send requires a PipeRef from T.comm.foreach_src")
        payload = _validate_payload(src)
        comm._validate_pipe_payload(net, payload)
        region = to_buffer_region(payload, access_type="r", extents=list(payload.shape))
        return tirx.call_intrin("handle", tirx.op.Op.get("tl.tt.pipe_send"), region, selected_pipe)

    selected_pipe, net, side = src_pipe
    if side != "dst":
        raise ValueError("T.copy receive requires a PipeRef from T.comm.foreach_dst")
    payload = _validate_payload(dst)
    comm._validate_pipe_payload(net, payload)
    region = to_buffer_region(payload, access_type="w", extents=list(payload.shape))
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tt.pipe_recv"), selected_pipe, region)


__all__ = ("copy",)

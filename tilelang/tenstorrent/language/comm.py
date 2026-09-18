"""Compile-time multi-Core communication topology primitives for Tenstorrent.

Here ``comm`` means communication. It is unrelated to ``T.comm_reducer``,
where ``comm`` abbreviates commutative.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Literal

from tvm import ir, tirx

from tilelang.language.eager.builder import Builder
from tilelang.language.kernel import KernelLaunchFrame
from tilelang.language.loop import serial


_FOREACH_SRC_ANNOTATION = "tl.tt.foreach_src"
_FOREACH_DST_ANNOTATION = "tl.tt.foreach_dst"

CoreCoord = tuple[int, int]
CoreExpr = tuple[tirx.PrimExpr, tirx.PrimExpr]
PipeKind = Literal["point_to_point", "collective"]
PipeSide = Literal["src", "dst"]


def _normalize_coord(value: object, name: str) -> CoreCoord:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a two-element (x, y) tuple")
    if any(isinstance(coord, bool) or not isinstance(coord, int) for coord in value):
        raise TypeError(f"{name} coordinates must be compile-time integers")
    x, y = value
    if x < 0 or y < 0:
        raise ValueError(f"{name} coordinates must be non-negative, got {value}")
    return x, y


@dataclass(frozen=True, init=False)
class CoreRange:
    """A half-open, two-dimensional rectangle of Core coordinates."""

    begin: CoreCoord
    end: CoreCoord

    def __init__(self, begin: CoreCoord, end: CoreCoord):
        normalized_begin = _normalize_coord(begin, "CoreRange.begin")
        normalized_end = _normalize_coord(end, "CoreRange.end")
        if normalized_begin[0] >= normalized_end[0] or normalized_begin[1] >= normalized_end[1]:
            raise ValueError(f"CoreRange must be non-empty in both dimensions: begin={normalized_begin}, end={normalized_end}")
        object.__setattr__(self, "begin", normalized_begin)
        object.__setattr__(self, "end", normalized_end)


@dataclass(frozen=True, init=False)
class Pipe:
    """One static point-to-point or collective Core communication edge."""

    src: CoreCoord
    dst: CoreCoord | CoreRange

    def __init__(self, src: CoreCoord, dst: CoreCoord | CoreRange):
        normalized_src = _normalize_coord(src, "Pipe.src")
        if isinstance(dst, CoreRange):
            normalized_dst: CoreCoord | CoreRange = dst
        else:
            normalized_dst = _normalize_coord(dst, "Pipe.dst")
        object.__setattr__(self, "src", normalized_src)
        object.__setattr__(self, "dst", normalized_dst)

    @property
    def kind(self) -> PipeKind:
        return "collective" if isinstance(self.dst, CoreRange) else "point_to_point"


@dataclass(frozen=True, init=False)
class PipeNet:
    """An ordered, non-empty collection of Pipes with one transfer contract."""

    pipes: tuple[Pipe, ...]
    kind: PipeKind

    def __init__(self, pipes: Sequence[Pipe]):
        if isinstance(pipes, (str, bytes)) or not isinstance(pipes, Sequence):
            raise TypeError("PipeNet.pipes must be a sequence of Pipe objects")
        normalized_pipes = tuple(pipes)
        if not normalized_pipes:
            raise ValueError("PipeNet requires at least one Pipe")
        if any(not isinstance(pipe, Pipe) for pipe in normalized_pipes):
            raise TypeError("PipeNet.pipes must contain only Pipe objects")
        kind = normalized_pipes[0].kind
        if any(pipe.kind != kind for pipe in normalized_pipes[1:]):
            raise ValueError("PipeNet cannot mix point-to-point and collective Pipes")
        object.__setattr__(self, "pipes", normalized_pipes)
        object.__setattr__(self, "kind", kind)


def _builder_state() -> tuple[Builder, dict]:
    builder = Builder.current()
    if builder is None:
        raise RuntimeError("T.comm topology primitives can only be used while constructing a TileLang PrimFunc")
    state = getattr(builder, "_tt_topology_state", None)
    if state is None:
        state = {
            "pipenets": [],
            "foreach_frames": {},
            "foreach_vars": [],
            "payload_contracts": [],
        }
        builder._tt_topology_state = state
    return builder, state


def _pipenet_id(net: PipeNet) -> int:
    _, state = _builder_state()
    for known_net, net_id in state["pipenets"]:
        if known_net is net:
            return net_id
    net_id = len(state["pipenets"])
    state["pipenets"].append((net, net_id))
    return net_id


def _pipe_descriptor(pipe: Pipe) -> dict:
    if isinstance(pipe.dst, CoreRange):
        dst = {"begin": list(pipe.dst.begin), "end": list(pipe.dst.end)}
    else:
        dst = list(pipe.dst)
    return {"src": list(pipe.src), "dst": dst}


def _encode_pipenet(net: PipeNet) -> str:
    descriptor = {
        "id": _pipenet_id(net),
        "kind": net.kind,
        "pipes": [_pipe_descriptor(pipe) for pipe in net.pipes],
    }
    return json.dumps(descriptor, sort_keys=True, separators=(",", ":"))


def _validate_in_grid(net: PipeNet) -> None:
    launch = KernelLaunchFrame.Current()
    if launch is None:
        raise RuntimeError("T.comm topology primitives must be used inside T.Kernel")
    block_frames = launch.frames[0:-4]
    if len(block_frames) != 2:
        raise ValueError(f"T.comm topology primitives require a 2-D T.Kernel grid, got {len(block_frames)} dimensions")
    grid: list[int] = []
    for block_frame in block_frames:
        extent = block_frame.doms[0].extent
        if not isinstance(extent, tirx.IntImm):
            raise ValueError("T.comm topology primitives require compile-time constant T.Kernel grid extents")
        grid.append(int(extent.value))

    def check_coord(coord: CoreCoord, name: str) -> None:
        if coord[0] >= grid[0] or coord[1] >= grid[1]:
            raise ValueError(f"{name} coordinate {coord} is outside T.Kernel grid ({grid[0]}, {grid[1]})")

    for index, pipe in enumerate(net.pipes):
        check_coord(pipe.src, f"PipeNet.pipes[{index}].src")
        if isinstance(pipe.dst, CoreRange):
            if pipe.dst.end[0] > grid[0] or pipe.dst.end[1] > grid[1]:
                raise ValueError(
                    f"PipeNet.pipes[{index}].dst range [{pipe.dst.begin}, {pipe.dst.end}) is outside T.Kernel grid ({grid[0]}, {grid[1]})"
                )
        else:
            check_coord(pipe.dst, f"PipeNet.pipes[{index}].dst")


def _require_pipenet(net: object) -> PipeNet:
    if not isinstance(net, PipeNet):
        raise TypeError(f"expected a T.comm.PipeNet, got {type(net).__name__}")
    _validate_in_grid(net)
    return net


def _predicate(op_name: str, net: PipeNet) -> tirx.PrimExpr:
    net = _require_pipenet(net)
    return tirx.call_intrin("bool", tirx.op.Op.get(op_name), tirx.StringImm(_encode_pipenet(net)))


def is_src(net: PipeNet) -> tirx.PrimExpr:
    """Return whether the current Core is a source in ``net``."""

    return _predicate("tl.tt.is_src", net)


def is_dst(net: PipeNet) -> tirx.PrimExpr:
    """Return whether the current Core is a destination in ``net``."""

    return _predicate("tl.tt.is_dst", net)


def is_active(net: PipeNet) -> tirx.PrimExpr:
    """Return whether the current Core belongs to ``net``'s active set."""

    return _predicate("tl.tt.is_active", net)


def _foreach(net: PipeNet, side: PipeSide):
    net = _require_pipenet(net)
    annotation = _FOREACH_SRC_ANNOTATION if side == "src" else _FOREACH_DST_ANNOTATION
    frame = serial(0, len(net.pipes), annotations={annotation: _encode_pipenet(net)})
    _, state = _builder_state()
    state["foreach_frames"][id(frame)] = (frame, net, side)
    state["foreach_vars"].extend((var, net, side) for var in frame.vars)
    return frame


def foreach_src(net: PipeNet):
    """Iterate, in PipeNet order, over Pipes sourced by the current Core."""

    return _foreach(net, "src")


def foreach_dst(net: PipeNet):
    """Iterate, in PipeNet order, over Pipes targeting the current Core."""

    return _foreach(net, "dst")


def _try_selected_pipe(pipe: object) -> tuple[tirx.Var, PipeNet, PipeSide] | None:
    """Return active PipeRef metadata without treating ordinary values as pipes."""

    if not isinstance(pipe, tirx.Var):
        return None
    builder, state = _builder_state()
    for active_frame in reversed(builder.frames):
        entry = state["foreach_frames"].get(id(active_frame))
        if entry is None:
            continue
        frame, net, side = entry
        if any(var.same_as(pipe) for var in frame.vars):
            return pipe, net, side
    if any(var.same_as(pipe) for var, _, _ in state["foreach_vars"]):
        raise ValueError("selected PipeRef can only be used inside its T.comm.foreach_src/dst region")
    return None


def _selected_pipe(pipe: object) -> tuple[tirx.Var, PipeNet, PipeSide]:
    selected = _try_selected_pipe(pipe)
    if selected is not None:
        return selected
    if not isinstance(pipe, tirx.Var):
        raise TypeError("selected PipeRef must be the loop variable from T.comm.foreach_src/dst")
    raise ValueError("selected PipeRef must be the loop variable from T.comm.foreach_src/dst")


def _validate_pipe_payload(net: PipeNet, buffer: tirx.Buffer) -> None:
    """Record and enforce the shape/dtype contract shared by one PipeNet."""

    _, state = _builder_state()
    for known_net, shape, dtype in state["payload_contracts"]:
        if known_net is not net:
            continue
        if dtype != buffer.dtype or not ir.structural_equal(shape, buffer.shape):
            raise ValueError(
                "T.comm PipeNet payload shape/dtype mismatch: "
                f"expected shape {shape} and dtype {dtype}, "
                f"got shape {buffer.shape} and dtype {buffer.dtype}"
            )
        return
    state["payload_contracts"].append((net, buffer.shape, buffer.dtype))


def _pipe_coord(op_name: str, pipe: object) -> CoreExpr:
    selected_pipe, _, _ = _selected_pipe(pipe)
    op = tirx.op.Op.get(op_name)
    return (
        tirx.call_intrin("int32", op, selected_pipe, 0),
        tirx.call_intrin("int32", op, selected_pipe, 1),
    )


def pipe_src(pipe: object) -> CoreExpr:
    """Return the source coordinate of a selected PipeRef."""

    return _pipe_coord("tl.tt.pipe_src", pipe)


def pipe_dst(pipe: object) -> CoreExpr:
    """Return the destination coordinate of a selected point-to-point PipeRef."""

    _, net, _ = _selected_pipe(pipe)
    if net.kind != "point_to_point":
        raise ValueError("T.comm.pipe_dst requires a point-to-point PipeNet")
    return _pipe_coord("tl.tt.pipe_dst", pipe)


def pipe_dst_range(pipe: object) -> tuple[CoreExpr, CoreExpr]:
    """Return the half-open destination range of a selected collective PipeRef."""

    selected_pipe, net, _ = _selected_pipe(pipe)
    if net.kind != "collective":
        raise ValueError("T.comm.pipe_dst_range requires a collective PipeNet")
    op = tirx.op.Op.get("tl.tt.pipe_dst_range")

    def endpoint(which: int) -> CoreExpr:
        return (
            tirx.call_intrin("int32", op, selected_pipe, which, 0),
            tirx.call_intrin("int32", op, selected_pipe, which, 1),
        )

    return endpoint(0), endpoint(1)


__all__ = (
    "CoreRange",
    "Pipe",
    "PipeNet",
    "foreach_dst",
    "foreach_src",
    "is_active",
    "is_dst",
    "is_src",
    "pipe_dst",
    "pipe_dst_range",
    "pipe_src",
)

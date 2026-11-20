"""Compile-time multi-Core communication topology primitives for Tenstorrent.

Here ``comm`` means communication. It is unrelated to ``T.comm_reducer``,
where ``comm`` abbreviates commutative.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Literal

from tvm import DataType, arith, ir, tirx

from tilelang.language.eager.builder import Builder
from tilelang.language.kernel import KernelLaunchFrame
from tilelang.language.loop import serial
from tilelang.utils.language import to_buffer_region


_FOREACH_SRC_ANNOTATION = "tl.tt.foreach_src"
_FOREACH_DST_ANNOTATION = "tl.tt.foreach_dst"

CoreCoord = tuple[int, int]
CoreExpr = tuple[tirx.PrimExpr, tirx.PrimExpr]
CoreEndpoint = tuple[int | tirx.PrimExpr, int | tirx.PrimExpr]
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


def _normalize_endpoint(value: object, name: str) -> CoreEndpoint:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a two-element (x, y) tuple")
    coords = []
    for coord in value:
        if isinstance(coord, bool) or not isinstance(coord, (int, tirx.PrimExpr)):
            raise TypeError(f"{name} coordinates must be scalar integer expressions")
        if isinstance(coord, tirx.PrimExpr):
            if not coord.dtype.startswith(("int", "uint")) or DataType(coord.dtype).lanes != 1:
                raise TypeError(f"{name} coordinates must be scalar integer expressions")
            coord = arith.Analyzer().simplify(coord)
            if isinstance(coord, tirx.IntImm):
                coord = int(coord.value)
        if isinstance(coord, int) and coord < 0:
            raise ValueError(f"{name} coordinates must be non-negative, got {value}")
        coords.append(coord)
    return tuple(coords)


@dataclass(frozen=True, init=False)
class CoreRange:
    """A half-open Core rectangle; transfer endpoints may use Core expressions."""

    begin: CoreEndpoint
    end: CoreEndpoint

    def __init__(self, begin: CoreEndpoint, end: CoreEndpoint):
        normalized_begin = _normalize_endpoint(begin, "CoreRange.begin")
        normalized_end = _normalize_endpoint(end, "CoreRange.end")
        analyzer = arith.Analyzer()
        if any(analyzer.can_prove(lo >= hi) for lo, hi in zip(normalized_begin, normalized_end)):
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
            _normalize_coord(dst.begin, "Pipe.dst.begin")
            _normalize_coord(dst.end, "Pipe.dst.end")
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
            "communication_count": 0,
        }
        builder._tt_topology_state = state
    return builder, state


def _next_communication_site_id() -> tirx.IntImm:
    _, state = _builder_state()
    site_id = state["communication_count"]
    state["communication_count"] += 1
    return tirx.IntImm("int32", site_id)


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


def _kernel_grid():
    builder = Builder.current()
    launch = next((frame for frame in reversed(builder.frames) if isinstance(frame, KernelLaunchFrame)), None) if builder else None
    if launch is None:
        raise RuntimeError("T.comm topology primitives must be used inside T.Kernel")
    block_frames = launch.frames[0:-4]
    if len(block_frames) != 2:
        raise ValueError(f"T.comm topology primitives require a 2-D T.Kernel grid, got {len(block_frames)} dimensions")
    grid: list[int] = []
    for block_frame in block_frames:
        extent = block_frame.doms[0].extent
        if not isinstance(extent, tirx.IntImm) or extent.dtype == "bool" or extent.value <= 0:
            raise ValueError("T.comm topology primitives require positive compile-time integer T.Kernel grid extents")
        grid.append(int(extent.value))
    return block_frames, grid


def _validate_in_grid(net: PipeNet) -> None:
    _, grid = _kernel_grid()

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


def _validate_payload(value: object, operation: str) -> tirx.Buffer:
    if not isinstance(value, tirx.Buffer):
        raise TypeError(f"{operation} requires a complete tirx.Buffer payload")
    scope = value.scope()
    if scope not in ("shared", "shared.dyn"):
        raise ValueError(f"{operation} requires a shared or shared.dyn payload buffer, got scope {scope!r}")
    if len(value.shape) != 2:
        raise ValueError(f"{operation} requires a rank-2 payload buffer")
    for extent in value.shape:
        if not isinstance(extent, tirx.IntImm) or extent.value <= 0 or extent.value % 32:
            raise ValueError(f"{operation} requires positive static payload extents divisible by the 32x32 tile shape")
    return value


def transfer(*, src: tirx.Buffer, dst: tirx.Buffer, src_core: CoreEndpoint, dst_cores: CoreEndpoint | CoreRange) -> tirx.PrimExpr:
    """Describe a logically synchronous inter-Core transfer.

    Payloads are complete rank-2 shared buffers with equal shape and dtype.
    A coordinate tuple selects point-to-point communication; a CoreRange selects
    collective communication even when it contains only one destination. A
    collective range may include the source, in which case the source is also a
    destination and ``dst`` is defined on that Core by loopback delivery.
    Coordinates may contain integer expressions such as the current Kernel's
    Core coordinates. Lowering must validate symbolic endpoints, participant
    correspondence, and buffer lifetimes before generating the two endpoints.

    The frontend emits one opaque call, not executable communication or a
    user-managed completion handle. Its operation-local site ID distinguishes
    repeated statements; the enclosing TIR loops retain dynamic occurrences.
    """
    block_frames, grid = _kernel_grid()
    source = _validate_payload(src, "T.comm.transfer")
    destination = _validate_payload(dst, "T.comm.transfer")
    if source.dtype != destination.dtype or not ir.structural_equal(source.shape, destination.shape):
        raise ValueError("T.comm.transfer payload shape/dtype mismatch between src and dst")
    source_coord = _normalize_endpoint(src_core, "transfer.src_core")
    if isinstance(dst_cores, CoreRange):
        kind = "collective"
        begin, end = dst_cores.begin, dst_cores.end
    else:
        kind = "point_to_point"
        begin = _normalize_endpoint(dst_cores, "transfer.dst_cores")
        end = tuple(coord + 1 for coord in begin)

    analyzer = arith.Analyzer()
    for frame in block_frames:
        analyzer.bind(frame.vars[0], frame.doms[0])
    for axis, extent in enumerate(grid):
        for coord in (source_coord[axis], begin[axis]):
            if analyzer.can_prove(tirx.any(coord < 0, coord >= extent)):
                raise ValueError("T.comm.transfer endpoint is outside T.Kernel grid")
        if analyzer.can_prove(end[axis] > extent):
            raise ValueError("T.comm.transfer destination range is outside T.Kernel grid")
        if analyzer.can_prove(begin[axis] >= end[axis]):
            raise ValueError("T.comm.transfer destination range must be non-empty")
    source_in_destination = tirx.all(*(condition for s, lo, hi in zip(source_coord, begin, end) for condition in (s >= lo, s < hi)))
    if kind == "point_to_point" and analyzer.can_prove(source_in_destination):
        raise ValueError("T.comm.transfer point destination must differ from src_core")

    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tt.transfer"),
        to_buffer_region(source, access_type="r", extents=list(source.shape)),
        to_buffer_region(destination, access_type="w", extents=list(destination.shape)),
        *source_coord,
        *begin,
        *end,
        tirx.StringImm(kind),
        _next_communication_site_id(),
    )


def put(*, src: tirx.Buffer, dst: tirx.Buffer, src_core: CoreEndpoint, dst_core: CoreEndpoint) -> tirx.PrimExpr:
    """Describe one point-to-point transfer with explicit source and destination.

    This is declaration syntax for a point-to-point transfer, including its
    logically synchronous contract and compiler-generated receive. Place it in
    the common participant scope, just like transfer; src_core is explicit and
    does not default to the calling Core. dst_core must be one coordinate tuple.
    """
    destination = _normalize_endpoint(dst_core, "T.comm.put.dst_core")
    return transfer(src=src, dst=dst, src_core=src_core, dst_cores=destination)


def broadcast(*, src: tirx.Buffer, dst: tirx.Buffer, src_core: CoreEndpoint, dst_cores: CoreRange) -> tirx.PrimExpr:
    """Describe a collective transfer to a CoreRange.

    All destinations receive the same source payload. A singleton range retains
    the collective contract. The range may include the source to request
    loopback delivery into the source Core's local ``dst`` instance.
    Participant scope, logical synchronization, and payload restrictions are
    inherited from transfer.
    """
    if not isinstance(dst_cores, CoreRange):
        raise TypeError("T.comm.broadcast dst_cores must be a T.comm.CoreRange")
    return transfer(src=src, dst=dst, src_core=src_core, dst_cores=dst_cores)


def _validate_collective_group(group: object, operation: str) -> int:
    block_frames, grid = _kernel_grid()
    if not isinstance(group, CoreRange):
        raise TypeError(f"{operation} group must be a T.comm.CoreRange")
    analyzer = arith.Analyzer()
    for frame in block_frames:
        analyzer.bind(frame.vars[0], frame.doms[0])
    size = 1
    for begin, end, extent in zip(group.begin, group.end, grid):
        if analyzer.can_prove(tirx.any(begin < 0, end > extent)):
            raise ValueError(f"{operation} group is outside T.Kernel grid")
        width = analyzer.simplify(end - begin)
        if not isinstance(width, tirx.IntImm) or width.value <= 0:
            raise ValueError(f"{operation} group dimensions must be positive compile-time integers")
        size *= int(width.value)
    return size


def _validate_collective_buffers(src: object, dst: object, operation: str) -> tuple[tirx.Buffer, tirx.Buffer]:
    source = _validate_payload(src, operation)
    destination = _validate_payload(dst, operation)
    if source.dtype != destination.dtype:
        raise ValueError(f"{operation} src and dst must have the same dtype")
    if source.dtype not in ("bfloat16", "float32"):
        raise ValueError(f"{operation} supports only bfloat16 and float32 payloads")
    if source.data.same_as(destination.data):
        raise ValueError(f"{operation} requires independent src and dst storage; in-place and aliased buffers are unsupported")

    analyzer = arith.Analyzer()
    for buffer in (source, destination):
        if not analyzer.can_prove_equal(buffer.elem_offset, 0) or buffer.axis_separators:
            raise ValueError(f"{operation} requires complete compact buffers without storage offsets or axis separators")
        if buffer.strides and (
            len(buffer.strides) != 2
            or not analyzer.can_prove_equal(buffer.strides[0], buffer.shape[1])
            or not analyzer.can_prove_equal(buffer.strides[1], 1)
        ):
            raise ValueError(f"{operation} requires compact row-major buffers")

    # MatchBuffer views can have distinct data Vars despite sharing storage.
    frames = list(Builder.current().frames)
    while frames:
        frame = frames.pop()
        if isinstance(frame, KernelLaunchFrame):
            frames.extend(frame.frames)
        for match in getattr(frame, "match_buffers", ()):
            if match.buffer.data.same_as(source.data) or match.buffer.data.same_as(destination.data):
                raise ValueError(f"{operation} does not support match_buffer views")
    return source, destination


def _emit_collective(op_name: str, src: tirx.Buffer, dst: tirx.Buffer, group: CoreRange, parameter: tirx.PrimExpr) -> tirx.PrimExpr:
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get(op_name),
        to_buffer_region(src, access_type="r", extents=list(src.shape)),
        to_buffer_region(dst, access_type="w", extents=list(dst.shape)),
        *group.begin,
        *group.end,
        parameter,
        _next_communication_site_id(),
    )


def all_gather(*, src: tirx.Buffer, dst: tirx.Buffer, group: CoreRange, axis: int) -> tirx.PrimExpr:
    """Concatenate all group members' inputs along axis into every member's dst.

    Group rank increases along x, then y. Each member contributes its own input;
    singleton groups still copy src to dst. Inputs and outputs are independent,
    complete, tile-aligned rank-2 shared buffers with matching BF16/FP32 dtype.
    Axis is a compile-time integer in [-2, 2); the output extent on that axis is
    group_size times the input extent, and the other extent is unchanged.

    This captures one opaque high-level operation. Lowering must validate group
    participation and generate communication, completion, and storage reuse.
    """
    operation = "T.comm.all_gather"
    size = _validate_collective_group(group, operation)
    source, destination = _validate_collective_buffers(src, dst, operation)
    if isinstance(axis, tirx.IntImm) and axis.dtype != "bool":
        axis = int(axis.value)
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError(f"{operation} axis must be a compile-time integer")
    if axis not in (-2, -1, 0, 1):
        raise ValueError(f"{operation} axis must be in [-2, 2) for rank-2 payloads")
    axis %= 2
    expected = [int(extent.value) for extent in source.shape]
    expected[axis] *= size
    actual = [int(extent.value) for extent in destination.shape]
    if actual != expected:
        raise ValueError(f"{operation} dst shape must be {expected}, got {actual}")
    return _emit_collective("tl.tt.all_gather", source, destination, group, tirx.IntImm("int32", axis))


def all_reduce(*, src: tirx.Buffer, dst: tirx.Buffer, group: CoreRange, op: Literal["sum", "max", "min"] = "sum") -> tirx.PrimExpr:
    """Reduce corresponding elements across group members into every member's dst.

    The tensor shape is unchanged; no local tensor dimension is reduced. Src and
    dst must be independent, complete, tile-aligned rank-2 shared buffers with
    matching BF16/FP32 dtype and shape. A singleton group still copies src to dst.
    This captures the reduction kind and dependencies in initial IR; numerical
    execution, accumulator precision, and synchronization require backend support.
    """
    operation = "T.comm.all_reduce"
    _validate_collective_group(group, operation)
    source, destination = _validate_collective_buffers(src, dst, operation)
    if not isinstance(op, str):
        raise TypeError(f"{operation} op must be a compile-time string")
    if op not in ("sum", "max", "min"):
        raise ValueError(f"{operation} op must be one of sum, max, min")
    if [int(extent.value) for extent in source.shape] != [int(extent.value) for extent in destination.shape]:
        raise ValueError(f"{operation} src and dst must have the same shape")
    return _emit_collective("tl.tt.all_reduce", source, destination, group, tirx.StringImm(op))


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
    "all_gather",
    "all_reduce",
    "broadcast",
    "foreach_dst",
    "foreach_src",
    "is_active",
    "is_dst",
    "is_src",
    "pipe_dst",
    "pipe_dst_range",
    "pipe_src",
    "put",
    "transfer",
)

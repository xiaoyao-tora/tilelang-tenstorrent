# Tenstorrent frontend dialect

Import the dialect explicitly:

```python
from tilelang.tenstorrent import language as T
```

The dialect constructs frontend TIR without requiring TT-Lang, TTNN, or device
hardware. Target registration, semantic lowering, Device IR, and code generation
are separate compiler components; constructing a function does not compile or
execute it.

## Logical compute regions

`T.Tiles` accepts a Buffer, a tuple/list of extents, or two scalar extents. It
constructs a static rank-2 logical domain using annotated serial loops. The
Buffer form contributes only its shape; the body supplies the inputs and output.
The body must contain exactly one elementwise assignment. Write additional
assignments as separate `T.Tiles` scopes. Only `parallel=True` is accepted. The
annotations preserve the logical domain for later compute capture; they do not
specify processor placement or scheduling.

```python
@T.prim_func
def add(A: T.Tensor((32, 32), "float32"), B: T.Tensor((32, 32), "float32")):
    for i, j in T.Tiles(B):
        B[i, j] = A[i, j] + 1
```

The dialect's reduction helpers construct `tl.tileop.reduce` directly, preserving
the source and destination storage scopes. They normalize negative axes and
preserve the reduction kind, clear flag, and NaN policy. Shape, dtype, and
accumulation compatibility are checked during lowering.

## Storage metadata

`T.alloc_shared(..., annotations=...)` attaches metadata to Buffer data identity.
The supported Tenstorrent keys are `tt.dfb_block_count` (an integer in `[1, 32]`)
and `tt.tile_shape` (`(32, 32)`). Explicit Tenstorrent metadata requires positive
static shared storage with tile-aligned final axes. Leading batch axes are
untiled. `tt.tensor_backed` is not supported.

Allocation metadata is recorded under `tl.alloc_buffer_annotations` and retained
by the common `LowerOpaqueBlock` pass. Buffers with identical names retain
distinct metadata. Calls without annotations preserve existing behavior.

## Communication

`T.comm.transfer` captures one complete source-exclusive communication statement:

```python
T.comm.transfer(
    src=panel,
    dst=panel,
    src_core=(0, core_y),
    dst_cores=T.comm.CoreRange(begin=(1, core_y), end=(core_n, core_y + 1)),
)
```

Two restricted forms share this transfer contract and IR:

| API | Destination | Contract |
| --- | --- | --- |
| `T.comm.put(src=..., dst=..., src_core=..., dst_core=...)` | One coordinate tuple | `point_to_point` |
| `T.comm.broadcast(src=..., dst=..., src_core=..., dst_cores=...)` | `CoreRange` | `collective`, including singleton ranges |

Both forms call `transfer` after validating the destination type. Each emits one
`tl.tt.transfer` with one call-site ID. `put` declares the complete communication
relationship with an explicit source Core and a compiler-generated receive; it
belongs in the common participant scope, just like `transfer`. It does not infer
the source from the calling Core. `broadcast` distributes one source payload to
the specified remote range, which must exclude the source. Source-only loading
may be guarded separately, with the communication statement outside that guard.

The source and destination are complete rank-2 shared buffers with matching
shape and dtype and positive tile-aligned extents. The same Buffer may appear on
both sides: it denotes distinct local storage on different Cores. A destination
coordinate tuple selects `point_to_point`; a half-open `CoreRange` selects
`collective`, including a range with only one destination. Transfer coordinates
and range bounds accept scalar integer TIR expressions. Construction rejects
provably invalid bounds and source inclusion. Full symbolic endpoint legality,
participant correspondence, and synchronization require subsequent lowering.

The initial TIR contains one opaque `tl.tt.transfer` call with ten operands:
source read region, destination write region, source x/y, destination begin x/y,
destination end x/y, contract string, and operation-local call-site ID. Point
destinations are encoded as singleton rectangles with their original contract.
Distinct statements retain distinct IDs even when their endpoints and payloads
match; enclosing serial loops retain repeated occurrences. The logical contract
requires data arrival before destination consumption and completion before source
reuse. This frontend does not generate sends, receives, waits, or executable TTL.

`T.comm.CoreRange`, `Pipe`, and `PipeNet` describe static communication topology.
PipeNet records are ordered; repeated records represent separate events.
`T.comm.foreach_src` and `T.comm.foreach_dst` select a PipeRef in its endpoint
scope. `T.copy(buffer, pipe)` constructs a send, and `T.copy(pipe, buffer)`
constructs a receive. Ordinary copies delegate to the common language helper.

Pipe payloads must be complete rank-2 shared buffers with positive static extents
divisible by 32. A PipeNet's send and receive payloads must agree in shape and
dtype. PipeRefs cannot escape their foreach scope or be used from the wrong
endpoint role. These checks validate construction; communication scheduling and
completion are responsibilities of lowering.

## All-gather and all-reduce

The frontend captures these collectives as separate high-level operations:

```python
row_group = T.comm.CoreRange((0, core_y), (core_n, core_y + 1))
T.comm.all_gather(src=local_part, dst=full_data, group=row_group, axis=1)
T.comm.all_reduce(src=partial, dst=result, group=row_group, op="sum")
```

All arguments are keyword-only. Both operations include every member's own
contribution. Group rank increases along x, then y, so rank is
`(y - begin_y) * width + (x - begin_x)`. Group dimensions must simplify to positive
compile-time integers. Symbolic group positions are captured, with provably
invalid grid bounds rejected. Full endpoint legality and matching participation
remain lowering responsibilities. Nonmembers perform no collective reads or
writes; a singleton group still copies its input into the distinct output.

Operands must be complete, compact rank-2 shared/shared.dyn buffers with positive
32-aligned static extents and the same BF16/FP32 dtype. In-place storage, shared
data identity, match_buffer views, partial regions, noncompact strides, storage
offsets, and axis separators are rejected. No dtype or layout conversion is
implicit.

`all_gather` requires an explicit compile-time axis in `[-2, 2)`, normalized to
`0` or `1`. Its destination shape equals the source shape except that the selected
extent is multiplied by the group size. Each source rank occupies the matching
slice in every destination. `all_reduce` preserves the entire shape and reduces
corresponding elements across Cores with `sum` (default), `max`, or `min`; it does
not reduce a local tensor dimension. Accumulator precision and numerical
execution rules require the later backend implementation.

The eight operands of `tl.tt.all_gather` and `tl.tt.all_reduce` are source read
region, destination write region, group begin x/y, group end x/y, normalized axis
or reduction string, and call-site ID. Both are opaque operations and share the
same per-PrimFunc ID sequence with transfer/put/broadcast. Serial loops and source
spans are retained. The frontend does not expand a collective into transfers or
generate synchronization, TTL code, or device execution.

## Validation

Frontend tests cover dialect exports, logical loop construction, allocation
metadata preservation, reduction construction, ordered topology, and PipeRef
scope and payload errors. Transfer tests cover symbolic endpoints, singleton
collective identity, payload compatibility, statement identity, serial loops,
serialization, and opaque effects. Sugar tests check IR equivalence, destination
type restrictions, shared validation, and call-site identity when the three APIs
are mixed. Collective tests cover group geometry, shape and dtype contracts,
axis normalization, reduction kinds, independent storage, mixed call-site IDs,
loops, serialization, and source spans. They inspect TIR without compiling or
running kernels.

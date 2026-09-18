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
Only `parallel=True` is accepted. The annotations preserve the logical domain
for later compute capture; they do not specify processor placement or scheduling.

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

## Validation

Frontend tests cover dialect exports, logical loop construction, allocation
metadata preservation, reduction construction, ordered topology, and PipeRef
scope and payload errors. They inspect TIR without compiling or running kernels.

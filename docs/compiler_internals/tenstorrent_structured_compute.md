# Tenstorrent structured elementwise lowering

`T.Tiles` and rank-2-or-higher `T.Parallel` elementwise maps share one structured TIRX
contract. This extends the original Tiles design to the existing Tenstorrent
Device TIR pipeline without retaining the frontend Add pattern matcher.
All stages described here run without Tenstorrent hardware, TT-Lang, or TTNN.
TTL generation and device execution remain unimplemented.

## Frontend and canonical IR

Both forms below describe the same computation:

```python
from tilelang.tenstorrent import language as T


@T.prim_func
def tiles_example():
    with T.Kernel(1, 1, threads=1):
        metadata = {"tt.tile_shape": (32, 32), "tt.dfb_block_count": 2}
        a = T.alloc_shared((64, 64), T.float32, annotations=metadata)
        b = T.alloc_shared((64, 64), T.float32, annotations=metadata)
        c = T.alloc_shared((64, 64), T.float32, annotations=metadata)
        for i, j in T.Tiles(c):
            c[i, j] = a[i, j] * b[i, j] + a[i, j]
        # The equivalent loop header is: for i, j in T.Parallel(64, 64):
```

This example demonstrates semantic capture; its shared inputs are not
initialized for execution. A complete operation also needs data movement and
a supported device consumer.

`CanonicalizeTTElementwise` consumes both the annotated serial loop nests
produced by `T.Tiles` and supported parallel loop nests in one traversal.
It recognizes each frontend at the outermost loop and uses the same analysis
and block construction, independently of whether
the expression is Add, subtraction, multiplication, division, or a compound
expression. The ordinary `T.Parallel` frontend and other targets are unchanged.
Sibling scopes can mix both frontends in either order within one `PrimFunc`;
one invocation captures all of them. An individual loop nest must use one
frontend consistently. The unified Python and FFI entry point replaces the
two former frontend-specific factories.

Analysis precedes mutation. Each accepted scope becomes an opaque
`SBlockRealize` containing an `SBlock` named `tl.tt.elementwise`, with:

- Full logical read/write `BufferRegion` effects, deduplicated by buffer.
- Logical scalar domain, physical tile shape `(32, 32)`, and the two-dimensional
  tile grid. For example, `(64, 64)` becomes `(2, 2)` and `(128, 32)` becomes
  `(4, 1)`; their geometry is not reduced to a tile count.
- Parallel iterator types and `tl.tt.tiles_stage=1`.
- Binder-independent access maps keyed by `buffer.data` identity.
- Explicit row, column, or scalar broadcast recipes where required.
- The original expression tree with zero-coordinate loads and store, used as
  an expression template rather than scalar executable code.

Allocation metadata stays on its owning allocation block. It is separate from
the computation's effects, access maps, and tile geometry.
The shared `LowerOpaqueBlock` pass also transfers per-buffer annotations by
data identity when invoked separately; alignment and initialization annotations
remain intact. That allocation contract is tested independently, without adding
generic block erasure to the active Tenstorrent pipeline.

`VerifyTTComputeBlocks` checks the complete structured contract, including
allocation identities, effect regions, expression purity, tile geometry,
access maps, and broadcast recipes. Canonicalization is idempotent, and failed
analysis does not partially mutate its input.

## Supported semantic subset

Scopes must have static, positive domains with at least two axes, last two axes divisible by `(32, 32)`, one
store, and shared or `shared.dyn` buffers with compatible tile metadata.
Supported expressions include arithmetic, casts, compile-time scalar values,
and supported pure unary calls. An output can also be read for an in-place
elementwise update. Independent scopes and enclosing serial loops retain their
structure.

Identity accesses use `[i, j]`. Broadcast accesses and their required physical
allocation shapes for output domain `(M, N)` are:

| Access | Physical shape | Logical read region | Axis map |
| --- | --- | --- | --- |
| `[i, j]` | `(M, N)` | `(M, N)` | `[0, 1]` |
| `[0, j]` | `(32, N)` | `(1, N)` | `[-1, 1]` |
| `[i, 0]` | `(M, 32)` | `(M, 1)` | `[0, -1]` |
| `[0, 0]` | `(32, 32)` | `(1, 1)` | `[-1, -1]` |

No expanded broadcast allocation is created. Physical broadcast emission is
deferred to the future TTL consumer.

Nested Tiles scopes, multiple stores, masks, predicates, dynamic shapes,
transpose, offsets, gather/scatter, coordinate values in the RHS, unsupported
buffer views or aliases, global stores, and side effects inside the elementwise
scope fail explicitly. Copy, GEMM, reduction, and communication are separate
operations; they cannot be hidden inside an elementwise expression.

## Pass order and output stages

The shared prefix is explicit in `tilelang/tenstorrent/pipeline.py`:

```text
BindTarget
CanonicalizeTTElementwise
VerifyTTComputeBlocks
ValidateTenstorrentFrontendIR
NormalizeTenstorrentLaunch
NormalizeTenstorrentBufferMetadata
NormalizeTenstorrentRegions
NormalizeTenstorrentTopology
LegalizeTenstorrentTileOps
```

Capture runs immediately after `BindTarget`, which establishes whether ordinary
Parallel loops belong to Tenstorrent. At this point the original loop binders,
Tiles annotations, and allocation metadata are intact. Running it before
frontend validation lets both surface forms share the structured contract;
running it before launch, region, and buffer normalization avoids recovering
logical accesses from already lowered IR. `VerifyTTComputeBlocks` then checks
that contract before normalization and instruction selection consume it.

The same pass preserves the metadata distinction between the frontends:
Tiles must provide explicit tile metadata, while ordinary Parallel may use
backend defaults. Placing metadata normalization ahead of capture could erase
that distinction by supplying defaults too early.

`LegalizeTenstorrentTileOps` consumes verified structured operations into explicit
computation calls with scalar expression DAGs, access maps, shapes and dtypes.
It preserves the classic Add operation for compatible v1 programs and uses the
[Phase 4 Device contract](tenstorrent_phase4_lower.md) for general computation.

Every complete pipeline invocation continues through:

```text
InferTenstorrentTensorLayout
FormTenstorrentDeviceProgram
VerifyTenstorrentDeviceIR
```

The result has `tt.device_ir_version=1` (legacy Add/no-op) or `2` (general
computation). Unsupported input raises a diagnostic; the pipeline no longer
returns a structured fallback. Standalone uninitialized shared computations
fail read-before-write checks. Clients that only need capture use
`CanonicalizeTTElementwise` and `VerifyTTComputeBlocks` directly.

SIMT layout inference, generic simplification, `LowerOpaqueBlock`, buffer
flattening, and scalar loop lowering do not run on the structured templates.
Only the explicit compute consumer may replace them. The TTL codegen entry
continues to report that source generation is unimplemented.

## Validation boundary

The target tests cover both frontend forms, canonical effects and geometry,
broadcasts, compound expressions, idempotency, malformed contracts, and
pipeline stages. Device IR tests retain descriptor and instruction-protocol
checks for the existing Add consumer.

Numerical evaluation of structured templates in tests checks logical indexing
and expression semantics on the CPU. It does not validate BF16 device rounding,
physical broadcast instructions, CB synchronization, or TTNN execution.

For a development checkout, initialize TileLang before running the selected
tests so the bundled TVM Python and native library paths are configured:

```bash
TILELANG_CACHE_DIR=/tmp/tilelang-tests .venv/bin/python - <<'PY'
from pathlib import Path
import tilelang
import pytest

tests = sorted(str(p) for p in Path("testing/python/target").glob("test_tilelang_tenstorrent*.py"))
tests += [
    "testing/python/backend",
    "testing/python/language/test_tilelang_language_tiles.py",
    "testing/python/language/test_tilelang_language_alloc_shared_metadata.py",
    "testing/python/tenstorrent/test_tilelang_tenstorrent_topology.py",
    "testing/python/transform/test_tilelang_transform_lexical_alloc_scope.py",
]
raise SystemExit(pytest.main(["-q", *tests]))
PY
```

Rebuild native sources before running these tests. The lexical allocation suite
includes additional tests that require CUDA hardware and may skip on CPU hosts.

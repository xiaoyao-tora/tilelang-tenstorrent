# Tenstorrent Phase 2 Add lowering

Phase 2 lowers one canonical, single-tile elementwise Add from normalized
Frontend TIRX into verified Device TIR schema v1. It remains a
hardware-independent compiler stage: it does not import TT-Lang, emit TTL MLIR,
compile kernels, or execute through TTNN.

## Supported operation

The positive path is intentionally exact:

- one frontend `PrimFunc` on a static `1x1` Core grid;
- three rank-2 `32x32` BF16 or FP32 Tensor ABI arguments in `A`, `B`, `C`
  order;
- three shared logical DFB buffers in lexical `a`, `b`, `c` order;
- full-region `A -> a`, `B -> b`, `c -> C` copies;
- one elementwise `c = a + b` with no broadcast or type conversion;
- 32x32 tiled, interleaved, unsharded Tensor layout;
- a static DFB block count of two.

`LegalizeTenstorrentTileOps` recognizes the canonical scalar BufferStore loop
and replaces it with the intermediate `tl.tt.tile_add` operation. Program
formation consumes that intermediate operation atomically; final Device TIR
must not retain it.

Shape mismatch, broadcast, partial regions, additional compute, non-unit tile
grids, multi-Core launch, Pipe topology, and other tile operations fail at the
owning validation, legalization, or formation boundary.

## Logical dataflow

Program formation publishes exactly three `DFBDescriptor` records. Their IDs
and Tensor backings follow stable ABI/allocation order:

| DFB | Tensor | Producer | Consumer | Meaning |
| --- | --- | --- | --- | --- |
| `0` | `A` / Tensor `0` | `ncrisc` | `trisc` | First Add input |
| `1` | `B` / Tensor `1` | `ncrisc` | `trisc` | Second Add input |
| `2` | `C` / Tensor `2` | `trisc` | `ncrisc` | Add output |

Each record is single-producer/single-consumer, has zero Tensor byte offset,
uses a `[1, 1]` block shape, retains the frontend block capacity, and declares
one logical transaction. Tensor effects are `input`, `input`, and `output`.
Phase 2 Add does not create Pipe descriptors.

## Canonical slot bodies

Device bodies use registered internal TileLang intrinsics. DFB and Tensor
operands are stable integer table indices; transfer regions are encoded as
`row_start, col_start, rows, cols`.

```text
TRISC (compute, role=add):
  dfb_reserve(2, 1)
  dfb_wait(0, 1)
  dfb_wait(1, 1)
  dfb_add(0, 1, 2, 1)

NCRISC (datamovement, noc_index=0, role=tensor_io):
  dfb_reserve(0, 1)
  tensor_to_dfb(0, 0, 0, 0, 32, 32)
  dfb_reserve(1, 1)
  tensor_to_dfb(1, 1, 0, 0, 32, 32)
  dfb_wait(2, 1)
  dfb_to_tensor(2, 2, 0, 0, 32, 32)

BRISC (datamovement, noc_index=1, role=idle):
  Evaluate(0)
```

TRISC owns no Tensor ABI parameters. NCRISC owns Tensor indices `[0, 1, 2]`.
BRISC owns no parameters and remains the canonical idle slot.

## Synchronization boundary

At this phase, a transaction is closed at the Device TIR abstraction level by
one producer-side reserve and use plus one consumer-side wait and use. The
verifier ties those markers to the descriptor producer/consumer slots and
`transaction_count_or_loop_relation`.

Physical circular-buffer push/pop operations are deliberately absent from
Phase 2 Device TIR. The later TT-Lang `ttl-insert-cb-sync` pipeline is expected
to insert complete CB synchronization. Phase 2 therefore does not claim that
its marker sequence is directly executable hardware synchronization.

## Verification

`VerifyTenstorrentDeviceIR` retains all Phase 1 schema and ABI checks and adds:

- exactly three Tensor-backed DFBs for non-empty Phase 2 Device IR;
- unique Tensor backing and the frozen input/output SPSC slot directions;
- Tensor/DFB dtype, tile shape, domain, byte offset, block shape, and
  transaction consistency;
- exact TRISC and NCRISC marker sequence, arity, static arguments, table
  references, transfer regions, and logical-kernel roles;
- canonical idle BRISC;
- no intermediate `tl.tt.tile_add` in final Device IR;
- no Var identity shared across slot PrimFuncs, in addition to per-function
  undefined-Var and SSA checks.

The verifier is read-only and accepts both freshly formed and JSON-restored
Device TIR. Tests freeze the descriptor table and slot marker sequences as a
golden, assert deterministic printing/structural hashing/JSON round-trip, and
exercise malformed SPSC, Tensor backing, transaction, marker order, and idle
slot cases.

## Deferred work

TTL MLIR emission, TT-Lang compilation, physical CB and DST allocation,
push/pop insertion, TTKernel/EmitC lowering, TTNN runtime execution, Pipe
topology, multi-Core placement, larger Tensor tile grids, broadcasting, and
additional compute operations remain later phases.

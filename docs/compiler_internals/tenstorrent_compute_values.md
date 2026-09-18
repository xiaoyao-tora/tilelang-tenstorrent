# Tenstorrent compute values (Device IR v7)

The v7 protocol extends structured `T.Tiles` lowering to ordinary
fragments and GEMM epilogues. It preserves immutable values independently of
DFB storage and keeps the existing v1–v6 protocols for their original programs.
The complete pass sequence remains in `tilelang/tenstorrent/pipeline.py`.
Device IR v8 extends this contract with explicit precision regions and exact
DFB snapshots between them; see [Flash Attention Lower](tenstorrent_flash_attention_lower.md).

## Supported frontend contract

- Static rank-2 BF16/FP32 local buffers and 32×32 compute tiles, including
  keepdims row fragments with a singleton final axis.
- Shared or fragment inputs and outputs, including mixed inputs, explicit
  casts and padded row/column/scalar broadcast.
- Original Buffer identity, old/new versions, multiple consumers and bounded
  static serial updates. Known conditions specialize before formation.
- GEMM initialization, complete K-stage updates, in-place scale, epilogues
  producing new fragments, multiple final consumers and independent lifetimes
  in an outer serial loop.

Shared allocation metadata continues to describe DFB storage. Fragment geometry
is inferred without inventing DFB capacity or Tensor backing. Data-dependent
branches with one pure assignment per arm can become `Select` expressions when
their local scalar predicate and old output satisfy definite initialization.
This does not introduce a Device control-flow ABI. Dynamic tails/masks and
direct cross-slot fragments remain unsupported. Per-Core values, materialized
PipeNet communication and independent pipelined value chains are described in
[the Lower composition contract](tenstorrent_lower_composition.md).

## Authoritative IR

`tt.compute_value_table` contains typed `ComputeValueDescriptor` entries:

| Field | Meaning |
| --- | --- |
| `value_id` | Operation-local immutable definition ID |
| `buffer` | Logical fragment identity, shape and dtype |
| `version` | Zero-based write version of that logical Buffer |
| `previous_value_id` | Previous write to that Buffer, or -1 |
| `accumulator_id` | Owning or single contributing accumulator, or -1 |
| `source_span` | Original source location |

Multiple contributing accumulator roots are recovered from the body def-use
graph, not compressed into one scalar descriptor field. Their precision and
complete-K requirements are checked separately.

The TRISC body owns the definitions and actual dependencies:

| Intrinsic | Contract |
| --- | --- |
| `compute_value(output, dfb_inputs...)` | Define a typed pure elementwise/fill/typecast value |
| `compute_value_load(value)` | Pure expression reference to a dominating definition |
| `compute_value_gemm(output, lhs_dfb, rhs_dfb, old, transpose_a, transpose_b)` | Produce a new accumulator version |
| `compute_value_store(value, output_dfb)` | Exact-dtype compute store to a reserved DFB |

Expression annotations reuse the existing logical domain, compute dtype and
tile geometry. `tt.access_maps` / `tt.input_shapes` describe only DFB operands;
`tt.value_inputs`, `tt.value_access_maps` and `tt.value_input_shapes` describe
compute-local operands. Tensor exports use an explicit compute cast when their
storage dtype differs; they never silently retag a value.

The verifier checks definitions, predecessor chains, exact expression effects,
geometry, ownership, accumulator provenance, complete K, final output dtype
and precision requirements independently of the frontend. JSON roundtrips
retain the complete contract. Runtime objects and frontend side tables are not
needed for codegen.

## Materialization and lifetime

Pure fragment chains do not allocate one DFB per value. A consumer requiring
DFB-attached operands triggers exact-dtype materialization; repeated uses of
the same version share that materialization. Explicit shared/Tensor outputs
produce their own DFB transactions.

All values may borrow their source DFBs. Logical release conservatively follows
the last transitive value consumer, including identity expressions and saved
old versions. The independent verifier checks this rule. The compiler does not
promise physical DST residency or elide spills without a validated schedule.

## TTL and validation boundaries

Source emission maps v7 values to typed TTL SSA arithmetic, unary operations,
cast, broadcast and stores. It uses the pinned compiler builders; the TT-Lang
pipeline owns physical CB allocation, sync insertion and tile scheduling.
BF16 broadcasts cannot share a Kernel requiring 32-bit DST on the supported
architectures and fail before the optional compiler is imported.

BF16 single-update GEMM has a restricted mapping. FP32/full-K and multiple
update GEMM remain fail-closed at the TTL boundary: the pinned compiler's
same-tile-type matmul and stateful accumulation lowering do not provide a
validated lossless schedule for those programs. Logical Device numerical tests
are separate from real TTL compile-only tests; absent compiler bindings cause
the latter to skip, not to pass using substitute builders. Hardware execution
and a production TTNN adapter are outside this implementation.

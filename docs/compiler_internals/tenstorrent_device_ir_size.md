# Tenstorrent Device IR size

## Why FlashAttention expanded

`hyf-test/kernel/flash_attn_03.py` has a small frontend loop nest, but the
v7/v8 Device contract assigns distinct IDs to static transactions.
`GeneralDataflowPlanner::PlanImpl` specializes serial-loop iterations, and
`FormMulticoreProgram` repeats the schedule for each Core. The default kernel
has 4 batch/head waves, 4 query waves and 32 KV iterations on each of 32 Cores.

Three costs compounded this expansion:

- Precision transitions saved and restored every entry of `current_values_`,
  including fragments overwritten before their next read. This created
  unnecessary DFBs, SSA values, stores, loads and releases.
- Repeated transactions and descriptor rows were fully materialized in the
  exported IR. Identical literals, shape/access-map containers, domains and
  requirements also occupied separate serialized objects.
- TVM's generic `IRDocsifierNode::AddMetadata` linearly scanned previous
  descriptors. With hundreds of thousands of descriptors, complete TIR printing
  exhibited quadratic lookup cost.

## Changes within existing passes

No compiler pass was added, and the Lower pass list/order is unchanged.

`FormTenstorrentDeviceProgram` collects the statically specialized operation
sequence before planning precision regions. Per-fragment access information
checks whether the current value is read before its next full overwrite. Only
live values cross a precision boundary. Read-modify-write operations, partial
writes and unknown effects remain conservative. Static branches, nonzero loop
minima and loop-carried values use their actual execution order. Pipe operations
use their actual buffer effects. A live FP32 value still cannot cross a BF16
region through implicit narrowing.

`InferTenstorrentComputeRequirements` shares immutable metadata using keys that
include node type, canonical child identities and source-span identity.
Buffer/Var identities, resource descriptors and effectful Calls are not merged.
Floating-point literal nodes are not interned; raw floating-point fields use
bitwise keys, preserving signed zero and NaN payloads. Single-operation
precision regions use `SeqStmt::Flatten` instead of an invalid one-element
sequence.

The same existing pass finishes large static v7/v8 modules in Device IR v9:

- Repeated flat instruction blocks become ordinary serial `tirx.For` loops.
  Only integer leaves with an exactly verified affine progression vary. Every
  occurrence must match the same structure, dtype, Span and identity-bearing
  operands. Expansion restores the original expressions and instruction order.
- DFB and compute-value tables become typed `DeviceIRDescriptorFamily` arrays.
  A family retains one prototype and integer columns for original table
  positions and varying fields. Columns are affine, periodic-with-offset, or
  explicit. DFB source identities retain every literal part and decimal field.
  Expansion recreates each distinct logical resource and its original IDs;
  families do not introduce shared lifetimes or storage reuse.
- `tt.compact_original_version` records the original v7/v8 version. Accumulator,
  requirement and Pipe transfer tables retain their original representation.

Automatic compaction requires at least 4,096 DFB/value rows and a substantial
reduction in descriptor count. Small modules retain their existing schema to
avoid the fixed cost of family metadata. The ordinary Python helpers
`compact_device_ir(mod, force=True)` and `expand_device_ir(mod)` expose this
representation for inspection; they are not transform passes.

`VerifyTenstorrentDeviceIR` expands v9, runs the existing complete semantic
verifier, and returns the compact input. Expansion checks column shapes,
positions, integer arithmetic and loop forms before allocation. Expansion is
limited to 4 million resource/instruction instances and 64 nesting levels;
decoded integer columns and reconstructed source strings each have a 256 MiB
budget. Automatic compaction falls back to the original schema when its output
would exceed these limits, preserving support for larger existing schedules.
Forced compaction reports a budget error instead. The verifier never regenerates
requirements to conceal invalid input metadata. Consumers of Device IR must
explicitly expand v9 or reject it; v9 does not imply additional TTL codegen or
hardware execution support.

## Complete metadata printing

Backend-owned printers for large descriptor tables cache metadata references
in the docsifier's existing identity map, scoped to the root frame. This avoids
the generic linear scan while retaining first-use order, indices and full
metadata. The small-kernel complete TIR was byte-for-byte identical before and
after the printer change. No TVM submodule changes were needed.

## Default-kernel measurements

Measured with the unchanged default kernel, SHA-256
`a0a84cf4b6357bae1ecefb7d008a96d71f5d4504fc800e8541dce309a7e50a29`,
using `lower_tenstorrent_ir` for `wormhole_b0`. TIR exports include
`show_meta=True`; JSON exports use `tvm.ir.save_json` without truncation.

| Metric | Original v8 | Updated v9 |
| --- | ---: | ---: |
| Complete device TIR, bytes | 833,513,234 | 67,022,045 |
| Complete device JSON, bytes | 2,947,491,325 | 204,713,763 |
| Logical DFBs | 334,048 | 265,728 |
| Logical compute values | 282,848 | 198,144 |
| Accumulators | 32,768 | 32,768 |
| Core/slot functions | 96 | 96 |

The final representation contains 608 DFB families, 384 value families and
160 serial loops. Complete TIR is 92.0% smaller and JSON is 93.1% smaller.
Default Lower took approximately 142 seconds on the local CPU build; this is
an artifact-size improvement, not a claim of faster kernel execution.

Complete JSON roundtrip and independent Device verification passed. Expanding
the complete v9 module is structurally equal to the full v8 module produced
with the liveness fix and metadata sharing (453,909,560-byte TIR and
1,065,352,115-byte JSON). This distinguishes the semantic removal of dead
snapshots from the lossless representation change.

Regression tests cover precision-boundary liveness, BF16 interpreter results,
identity and Span preservation, full metadata serialization, exact expansion,
malformed columns and loops, arithmetic overflow, budgets, and independent
verification of forged requirements. The final local regression run passed
1,216 tests, with 25 optional TT-Lang integration tests skipped. The complete
default kernel was validated separately: automatic compaction, exact expansion,
independent verification and comparison against the saved complete v9 artifact
all passed with the final native build. Native build and pre-commit checks also
passed.

Static formation and semantic verification
still temporarily materialize the expanded schedule; this change does not make
compiler peak memory constant in kernel work. No Tenstorrent hardware run was
performed.

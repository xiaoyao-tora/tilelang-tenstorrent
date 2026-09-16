# Tenstorrent accumulator and precision contract

Device IR v5 adds a persistent GEMM accumulator independently of DFB storage.
The initial supported lowering subset is one Core, rank-2 full fragment regions,
32×32 physical tiles and bounded static serial K loops. Pipeline and multicore
accumulator combinations remain unsupported. Existing v1–v4 operations retain
their original storage and scheduling contracts.

## Lowering and ownership

The frontend allocates the accumulator with `T.alloc_fragment`. A dominating
`T.clear` or a first GEMM with `clear_accum=True` initializes it. All later
updates use the same region and accumulation dtype. One final `T.copy` after
the complete K reduction materializes it to a shared output or Tensor.
Ordinary stores, intermediate copies, ambiguous aliases and cross-slot uses
are rejected.

`VerifyTTGemmAccumulators` establishes frontend lifetime facts, without claiming
that a target can compile or execute the program. Regions normalization makes
bounded static K slices concrete. Legalization preserves accumulator operations;
layout validation does not assign them a CB or a physical DST index. Formation
creates one logical accumulator, its updates, and a final output DFB. DCE keeps
the accumulator's input dependencies live until materialization.

```text
BF16 Tensor inputs → BF16 input DFB generations
→ persistent FP32 accumulator, full K updates
→ final BF16 output DFB → Tensor writer
```

The module attribute `tt.accumulator_table` contains typed
`AccumulatorDescriptor` objects with:

- `accumulator_id`, independently numbered from DFB IDs;
- `accumulator_region`, preserving fragment Buffer/data identity;
- `input_dtype`, `accumulation_dtype`, `output_dtype`;
- `full_k_tiles`, the sum of K tiles across every update;
- `source_span`.

Final operations are registered, effectful TileLang intrinsics:

| Operation | Operands |
| --- | --- |
| `tl.tt.accumulator_init` | accumulator ID |
| `tl.tt.gemm_update` | lhs DFB, rhs DFB, accumulator ID, transpose A, transpose B |
| `tl.tt.accumulator_materialize` | accumulator ID, output DFB |

Only TRISC owns these operations. Input DFB waits and output reserve/writer
ordering remain part of the ordinary verified DFB protocol. A final Tensor copy
gets an output DFB even if the frontend did not explicitly allocate one.

## Final precision requirements

`InferTenstorrentComputeRequirements` runs after formation and before the final
Device verifier. It derives typed `tt.compute_requirements` per compute function:

- `destination_width`: `unconstrained`, `bits16_required`, `bits32_required`;
- `matmul_full_fp32`: `allowed`, `required`, `forbidden`;
- `accumulators`: the function's typed descriptors.

| Input | Accumulation | Output | Destination width | Full FP32 |
| --- | --- | --- | --- | --- |
| BF16 | BF16 | BF16 | bits16_required | forbidden |
| BF16 | FP32 | BF16 | bits32_required | required |
| FP32 | FP32 | FP32 | bits32_required | required |

The final verifier independently checks lexical lifetime, full K tile count,
operation operands, dtype/shape, slot ownership and the stored requirements.
Forged or stale requirements are rejected, not repaired. A 16/32-bit conflict
in one compute kernel is an error; no implicit widening or narrowing occurs.
Merely allocating an FP32 fragment does not imply a GEMM precision requirement.

Version 5 requires this attribute. Older Device modules remain readable;
GEMM source generation requires explicit verified requirements and asks callers
to run the current Lower pipeline when they are missing.

## Source-only TTL generation

`build_ttl_without_compile` first verifies Device IR, architecture and consumer
capability, then lazily imports the pinned TT-Lang production bindings. It
constructs typed MLIR operations, verifies both the module and its serialized
form, and returns a TVM source module with `inspect_source()` and format `ttl`.
It neither loads TTNN nor launches kernels.

The current consumer is deliberately narrower than Device Lower: direct Add,
copy and restricted BF16 GEMM on one Core. A v5 GEMM must have one complete-K
update and a supported transpose configuration. Multiple-update persistent-DST
scheduling and FP32 GEMM emission remain gated until the pinned compiler can
prove their precision contract. No fallback silently materializes an FP32
accumulator into BF16 between updates.

Verified destination width maps to compute function `fp32_dest_acc_en`:
32-bit sets `true`, 16-bit sets `false`, unconstrained omits the attribute.
The function constraint must survive TT-Lang compute configuration. Setting
an attribute alone is not evidence of full-K numerical correctness.

Device `tensor_backing` currently records copy provenance. It is not blindly
translated to a TTL zero-copy L1 alias. Physical CB allocation, synchronization
completion, DST allocation and TTKernel lowering remain owned by TT-Lang.

## Validation layers

Hardware-independent tests cover schema serialization, deterministic lowering,
full-K lifetime, conflicting requirements, structured return and a CPU numerical
reference executing the generated Device stream. The numerical reference models
rounding explicitly and distinguishes final BF16 materialization from erroneous
intermediate BF16 pack/reload; it does not certify hardware arithmetic.

Optional integration tests require actual TT-Lang bindings for typed emission,
parser/verifier checks and compiler tools for compile-only checks. Missing
dependencies are reported or skipped explicitly. Simulator and Tenstorrent
hardware validation are separate gates; source generation is not execution.

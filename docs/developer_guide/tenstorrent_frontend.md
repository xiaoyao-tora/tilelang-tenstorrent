# Tenstorrent frontend contracts

Use `from tilelang.tenstorrent import language as T` for `T.Tiles`,
`T.comm`, and the PipeRef overload of `T.copy`. Communication records are
ordered events: repeated records remain distinct, and a selected PipeRef is
valid only inside its source or destination iteration.

`T.Tiles` retains its logical extents until `CanonicalizeTTElementwise`
captures the complete expression, access maps, broadcast recipes, and effects.
The annotations are a structural contract; missing scope/kind markers and
storage descriptors containing deleted loop variables are rejected.
The rank-2, 32-by-32 tiled contract is the baseline. Existing supported batch
dimensions remain available. Scalar `T.Parallel` and the default language
dialect retain their existing behavior.

Allocation annotations describe storage, independently of compute metadata.
`tt.dfb_block_count` must be an integer in `[1, 32]`, and `tt.tile_shape`
currently requires `(32, 32)`. Explicit metadata requires positive static,
tile-aligned shared storage. Metadata normalization follows Buffer identity
across SBlock allocations and statement-level allocations produced by
`LowerOpaqueBlock`; buffer names are not identities.

## GEMM numerical requirements

An accumulator fragment's dtype describes its required numerical precision.
It does not change the input or output tensor's storage dtype. The standalone
`tilelang.tenstorrent.transform.VerifyTTGemmAccumulators` pass runs after
launch and allocation metadata normalization, before copy region lowering. It recomputes the
GEMM def-use contract and records `tt.gemm_accumulator_requirements`.

The fragment contract accepts these input/accumulator/output combinations:

| Input storage | Accumulator | Output storage |
| --- | --- | --- |
| BF16 | BF16 | BF16 |
| BF16 | FP32 | BF16 |
| FP32 | FP32 | FP32 |

The supported lifetime uses one dominating `T.clear` or `clear_accum=True`,
followed by updates to the same complete fragment, and one final `T.copy`
after all K updates. Static serial K loops are supported by this analysis.
Ambiguous aliases, conditional lifetimes, intermediate materialization,
ordinary overwrites, and unsupported dtype combinations are diagnosed.
An unrelated FP32 temporary does not create a GEMM precision requirement.

Frontend acceptance is distinct from Device Lower support. Device TIR does
not yet represent a fragment's complete DST lifetime, so lowering a verified
fragment GEMM reports that limitation explicitly. It does not silently turn
the fragment into a lower-precision DFB. The existing shared-output GEMM path
remains available, including the BF16/BF16/BF16 bring-up path, with precision
requirements checked against the actual operand and output storage types.

## Validation boundary

Hardware-independent tests cover frontend construction, semantic capture,
metadata preservation, fragment contract validation, Device IR verification,
and logical interpretation of communication combined with Tiles broadcast.
The logical interpreter is not a TT-Lang or hardware simulator.

TTL code generation and TTNN execution are still unimplemented. These tests
therefore do not establish generated TTL validity, physical DST scheduling,
or device numerical accuracy. Tensor-backed DFBs and cross-device topology
remain deferred. No user-visible transfer token, DFB lifecycle method,
processor role, or physical DST configuration is introduced.

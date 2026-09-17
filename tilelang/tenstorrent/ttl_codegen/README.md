# TTL source code generation

The optional compiler ABI is pinned to TT-Lang commit
`6e3051bb9b2f7bfbd5fd045aa38deb68b7a6e6bd`. `load_bindings()` checks the
compiler's `build_info()["ttlang"]`; source/simulator-only installs and unknown
revisions are not sufficient. Importing TileLang does not import TT-Lang.

`build_ttl_without_compile` accepts verified Device IR and returns a TVM source
module (`inspect_source()`, format `ttl`). It constructs typed MLIR operations,
verifies the module, and parses/verifies its serialized form. It does not construct
TTNN objects or launch kernels. The upstream `ttl` package initializer may
import TTNN transitively; a compiler-only installation must provide the
upstream compiler import environment. The only textual type construction is the fixed
`!ttl.transfer_handle<read/write>` token, whose pinned binding has no constructor.

Implemented consumers are single-Core, straight-line rank-2 DRAM interleaved
elementwise/copy/fill/typecast (v1/v2), clear BF16 GEMM (v2), and BF16 accumulator
GEMM with one complete K update (v5/v7). Device v7 compute values map immutable
value IDs to TTL SSA. Shared/fragment expressions use the same typed arithmetic,
unary, scalar-fill, explicit-cast, and padded zero-axis broadcast builders.
DFB and fragment access maps are resolved separately, including intra-tile
broadcast when the source and destination have identical block shapes.
Compute materialization uses `ttl.store`; fragments never create CB bindings
or transfer handles. A passthrough fragment retains its acquired DFB SSA value.
Device lifetime covers its transitive consumers; Device release ends the
emitter's direct DFB reference, while TT-Lang's acquire/release analysis emits
the physical pop after the last transitive SSA use.
BF16 GEMM sets the compute function's `fp32_dest_acc_en=false`.
At the pinned revision, `ComputeKernelConfigAnalysis.cpp`'s
`resolveKernelConfig` removes all Bits32 candidates for that explicit constraint;
the default `matmul-full-fp32` preference cannot override it. FP32 requirements
and multiple fragment updates are rejected until a persistent-DST schedule can
be verified. Pipeline/multicore Device Lower remains available, but its TTL
mapping is not implemented.

Device `TensorBacking` describes the tensor origin of a copied DFB. It is not
mapped to TTL `tensor_backing`, which asserts an actual zero-copy L1 storage
alias. DFB IDs are seeded exactly as in the production TT-Lang frontend;
TT-Lang's DFB finalizer owns physical slot allocation. Acquire/release analysis
owns automatic DFB pops; explicit reserves/waits/publication preserve Lower's
operation order.

The real integration tests in
`testing/python/target/test_tilelang_tenstorrent_ttl_codegen.py` exercise MLIR
round-trip verification and TTL-to-TTKernel compilation, without hardware. They
skip only when the compiler package is absent; an installed but incompatible
compiler fails. No real pinned compiler or hardware validation has yet been
performed in this checkout. `tenstorrent.capabilities.gemm_capability` reports
that distinction explicitly; its dtype legality comes from the native registry.

## Pinned API and lowering audit

The typed builders match the pinned `TTLOps.td` definitions and production
`python/ttl/operators.py` calls. The generated `_ttl_ops_gen.py` is a build
artifact and is absent from the source-only checkout.

| Expression | Builder contract | Lowering evidence |
| --- | --- | --- |
| Add/Sub/Mul/Div/Min/Max | Two operands; result inferred by `SameOperandsAndResultType` | `TTLElementwiseOps.def` maps every operation through compute to TTKernel |
| Unary operations | One operand; same result type | `TTLElementwiseOps.def`; Exp carries ODS default flags |
| Log2 | `ttl.log` followed by `ttl.mul_unary_const(input, F32Attr)` | No pinned Log2 operation; both emitted operations have compute/tile lowering |
| Fill/scalar | `ttl.fill(result_type, F32Attr)` | Production `operators.py` fill implementation; `LowerFillToCompute` |
| Cast | `ttl.typecast(result_type, input)` | `LowerTypecastToCompute`; tile lowering derives input/output dtype explicitly |
| Broadcast | `ttl.block_broadcast(result_type, input, DenseI64ArrayAttr, DenseI64ArrayAttr)` | Production broadcast implementation; `LowerBlockBroadcastToCompute` |

Only BF16 and FP32 block expression dtypes are mapped. For both supported
architectures, a BF16 row/column/scalar broadcast conflicts with required
32-bit DST, including a following FP32 cast or FP32 computation in the same
kernel. `check_capability` rejects that combination before optional compiler
import. This restriction is explicit in
`ComputeKernelConfigAnalysis.cpp::supportsDestinationElementWidth` and the
pinned `broadcast_dst_mode_invalid.mlir` tests. FP32 broadcast does not have
that restriction. Ordinary FP32 compute values use the verified kernel
requirement to emit `fp32_dest_acc_en=true`.

Direct tile builders do not remove the multi-update GEMM limitation.
`TileMatmulBlockOp::verify` preserves matmul's dtype constraints, and its
TTKernel lowering reloads an optional accumulator from a DFB. The pinned
`analyzeTensorAccumulationForDst` requires an acquired, same-type contribution
and accepts an additive recurrence without a Matmul in the loop body.
Stateful tensor accumulation scopes explicitly reject required DST lowering.
A persistent GEMM schedule therefore needs further compiler integration and
real compilation evidence; neither a DST attribute nor manually chosen tile
indices establishes the required no-intermediate-BF16-pack guarantee.

This audit establishes API/lowering source correspondence, not a successful
compiler execution. All parser, sync-placement, and full compile-only tests
remain real-compiler tests, never mock-builder tests.

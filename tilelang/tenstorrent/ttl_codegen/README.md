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
Add/copy (v1/v2), clear BF16 GEMM (v2), and BF16 accumulator GEMM with one complete
K update (v5). BF16 GEMM sets the compute function's `fp32_dest_acc_en=false`.
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

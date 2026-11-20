# Tenstorrent initial TTL code generation

The optional compiler ABI is pinned to TT-Lang commit
`4ca4b89de95242ccb9bca8ade71200ac81a59e38`. Importing TileLang does not load
TT-Lang or TTNN. Selecting Tenstorrent device code generation lazily loads the
pinned compiler bindings and checks `build_info()["ttlang"]`.

The production input is current `tt.ir_stage="device"` IR returned by the
fixed Tenstorrent Lower pipeline. Codegen trusts that Pass 13 already verified
the Device contract. It does not repeat `VerifyTenstorrentDeviceIR` and does
not perform a whole-module mapping preflight. Type, resource, statement and
expression emitters report unsupported mappings while constructing a temporary
TTL module.

The emitter consumes the three shared TRISC/NCRISC/BRISC functions in
`tt.kernel_order`. It creates typed Tensor layouts, logical DFB bindings,
explicit DFB lifecycle operations, SCF control flow, compute SSA values and
PipeNet record regions. Resource aliases and inline kinded references are
resolved from the current Device body; descriptor families are typed lookup
data and do not create a second operation stream. Dynamic family-row DFB
selection requires the pinned TT-Lang resource-selection API and fails at the
operand emitter when that API cannot preserve direct `ttl.bind_cb` provenance.

The result is `TenstorrentTTLModuleNode`, which owns both the MLIR context and
the initial TTL module. It also carries the target, launch grid, static Tensor
ABI, hard precision policy, dependency revision and deterministic contract
digest. Normal compilation passes this object directly to the TT-Lang adapter.
TTL text is available through `inspect_source()` for debugging only; the
production handoff does not serialize and parse the module.

The TileLang TT-Lang adapter verifies the received module and drives the pinned
`ttl-to-ttkernel-pipeline` with the verifier enabled. It then uses the pinned
TT-Lang Python compiler ABI to translate final TTKernel functions to C++,
resolve physical DFB and PipeNet runtime metadata, and return the existing
`CompiledTTNNKernel`. The adapter checks the complete helper surface before
mutating the module, so a TT-Lang ABI mismatch fails before compilation. A
required full-FP32 matmul remains unsupported until TT-Lang exposes a
hard-require API and a resolved-schedule query; the existing Boolean is a
preference.

`TTNNKernelAdapter` connects the compiled object to TileLang JIT. Its initial
runtime ABI accepts only device-resident, tiled `ttnn.Tensor` objects. It
checks shape, dtype, memory space, device and output positions, allocates
missing outputs, serializes calls that share mutable runtime-resource lifetime,
and delegates Program construction and execution to
`CompiledTTNNKernel -> ttnn.generic_op`.

Frontend construction, Device Lower, initial TTL verification, TT-Lang
compile-only, TTNN runtime execution, hardware numerics and performance remain
separate validation levels.

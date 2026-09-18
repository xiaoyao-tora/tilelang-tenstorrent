# Tenstorrent Phase 1 Device TIR skeleton

Phase 1 implements the hardware-independent boundary from a deliberately small
Frontend TIRX subset to verified Device TIR schema v1. It does not implement
Add dataflow lowering, TTL MLIR emission, TT-Lang compilation, or TTNN runtime
execution.

This document records the Phase 1 contract. The current backend also implements
the Phase 2 Add device path described in
`docs/compiler_internals/tenstorrent_phase2_add_lower.md` and the shared
[Tiles/Parallel structured prefix](tenstorrent_structured_compute.md).
TTL codegen and execution remain deferred.

## Implemented pipeline

The current pipeline extends the original ten-stage design with structured
compute capture. The no-op and supported Add device paths execute:

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
InferTenstorrentTensorLayout
FormTenstorrentDeviceProgram
VerifyTenstorrentDeviceIR
```

The prefix validates and normalizes the frontend contract. For
the Phase 1 no-op subset, `LegalizeTenstorrentTileOps` and
`NormalizeTenstorrentTopology` are read-only capability gates. Phase 2 extends
the former to legalize the canonical Add dataflow while Pipe constructs remain
unsupported. Valid compute without an implemented device consumer returns
explicitly marked structured IR before tensor layout and device formation;
see the structured compute document for that output contract.

`FormTenstorrentDeviceProgram` accepts one single-Core operation at a time. It
consumes the function-level `tt.launch_grid` and typed
`tt.buffer_metadata_table`, validates the complete supported plan, and then
atomically creates a new module. The output contains exactly the eight frozen
module attributes and exactly the `trisc`, `ncrisc`, and `brisc` PrimFuncs.
`trisc` owns the Phase 1 skeleton ABI; the two data-movement slots are
canonical idle functions.

The codegen prepare hook runs `VerifyTenstorrentDeviceIR` again after device
module filtering. It remains independent of the optional `ttl` package.

## Supported subset

The current positive path is intentionally narrow:

- one frontend `PrimFunc`;
- a static `1x1` Core grid with unit thread extents;
- optional two-dimensional BF16 or FP32 Tensor ABI parameters;
- 32x32 tiled, interleaved, unsharded tensor metadata;
- a semantic no-op kernel body;
- deterministic typed metadata, source spans, printing, structural hashing,
  and JSON round-trip.

Constructs unsupported by the Phase 1 subset fail before Device TIR is
published:

- TileOps, including `T.copy`;
- Buffer loads/stores and other compute bodies;
- logical DFB allocations or transactions;
- Pipe/PipeNet topology operations;
- multi-Core program formation;
- multiple frontend operations, runtime scalar arguments, dynamic layout,
  padding, sharding, and `tt.tensor_backed` frontend metadata.

These exclusions are capability boundaries, not silently dropped behavior.
Phase 2 is responsible for Add, Tensor-to-DFB and DFB-to-Tensor transfers,
transaction planning, and non-idle data-movement slot bodies.

## Typed schema and verification

Native TVM ObjectRef metadata defines Core coordinates/domains, Tensor backing,
Tensor/DFB/Pipe descriptors, logical kernels, normalization-stage Buffer
metadata, and module/function metadata. A DFB Tensor backing is a structured
`TensorBacking(global_arg_index, byte_offset)`, even though the frontend
`tt.tensor_backed` capability is deferred.

The verifier checks the eight module fields, Tensor/DFB/Pipe references,
ordered Pipe events, exact three-slot ABI, target architecture, calling
convention, thread and NoC assignment, Tensor parameters, SSA, source spans,
idle bodies, and absence of temporary frontend metadata or forbidden backend
operations. It is read-only and is also used for JSON-restored Device TIR.

## Remaining boundary

`build_ttl_without_compile` still reports `NotImplementedError`. No generated
TTL source or TTNN executable can be produced in Phase 1, and no Tenstorrent
hardware or TT-Lang installation is required by the Phase 1 tests.

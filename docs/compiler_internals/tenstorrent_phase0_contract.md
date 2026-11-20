# Tenstorrent Phase 0 compiler contract

Phase 0 freezes the interfaces between the TileLang frontend, Tenstorrent
Device TIR, direct TTL MLIR code generation, and TT-Lang. It does not implement
Device TIR lowering or TTL emission.

The machine-readable source of truth is
`tilelang/tenstorrent/contracts.py`. Changes to those constants are schema or
toolchain compatibility changes and require contract-test updates.

## Target and Device TIR schema

The target kind is `tenstorrent`. An explicit `arch` is required and must be
`wormhole_b0` or `blackhole`. Device TIR schema version 1 is a TVM TIRX
`IRModule`, not a separate program graph or a Python side channel.

The canonical target keys are exactly `("tenstorrent",)`, the TVM device type
is `ext_dev`, and `arch` is the only required target attribute. Phase 0 defines
no target aliases and no Tenstorrent auto detection.

The module fields are `tt.device_ir_version`, `tt.target_arch`,
`tt.launch_grid`, `tt.operation_identity`, `tt.tensor_table`, `tt.dfb_table`,
`tt.pipe_table`, and `tt.kernel_order`. The frozen Tensor, DFB, Pipe, and
PrimFunc descriptor field sequences are listed in `contracts.py` and are kept
as tuples so ordering cannot depend on maps, sets, or traversal order.

The descriptor fields are:

```text
Tensor:
  global_arg_index, shape, dtype, strides, tile_shape, tile_grid_shape,
  memory_space, memory_layout, shard_spec, effect, alias_group, source_span

DFB:
  dfb_id, source_buffer_identity, element_dtype, tile_shape,
  block_shape_in_tiles, block_count, tensor_backing, producer_slot,
  producer_domain, consumer_slot, consumer_domain,
  transaction_count_or_loop_relation, source_span

Pipe:
  pipe_net_id, communication_id, record_index, src_coord, dst_begin, dst_end, contract,
  payload_dfb_id, source_span

PrimFunc attributes:
  global_symbol, calling_conv, target, tt.kernel_slot, tt.kernel_thread,
  tt.noc_index, tt.logical_kernel, tt.tensor_arg_indices, tt.core_domain
```

Every operation will eventually contain exactly three slot functions in this
order:

| Slot | Thread | NoC index |
| --- | --- | --- |
| `trisc` | `compute` | not present |
| `ncrisc` | `datamovement` | `0` |
| `brisc` | `datamovement` | `1` |

Idle slots remain present as legal no-op functions. `tt.tensor_backed` is
explicitly deferred; Phase 0 does not promise frontend or Device TIR support
for it.

## Frontend and serialization

The public topology API is `tilelang.tenstorrent.language.comm`; emitted target
operations and loop annotations use the internal `tl.tt.*` namespace. PipeNet
records are ordered and retain duplicate events. Tenstorrent DFB candidates use
`T.alloc_shared` metadata under `tl.alloc_buffer_annotations`, including
`tt.dfb_block_count` and `tt.tile_shape`.

Phase 0 tests freeze real Add, point-to-point, gather, and collective frontend
TIRX. The persistence contract uses `tvm.ir.save_json` and
`tvm.ir.load_json`, followed by structural equality. Printed TIRX is required
to be deterministic, but its human-readable text is not a second persistence
format or an input to codegen.

## Lowering and codegen boundary

`TENSTORRENT_LOWER_PASS_ORDER` records the current sequence from `BindTarget`
through `VerifyTenstorrentDeviceIR`. It extends the original ten-pass design
with shared Tiles/Parallel semantic capture and verification; see
[structured compute lowering](tenstorrent_structured_compute.md) for the
structured output boundary and
`docs/compiler_internals/tenstorrent_phase1_device_ir.md` for the no-op Device
TIR contract.

`DeviceCodegen.prepare` lets a backend replace the common
`LowerIntrin`/`Simplify`/`HoistBroadcastValues` preparation without adding a
target-kind branch to the shared engine. Tenstorrent's hook validates the target
kind, architecture and Device stage. It trusts the Device verifier at the end
of the fixed Lower pipeline and does not repeat whole-program verification or
mapping preflight in Codegen.

The in-memory device output format is `ttl`; no host codegen is declared.
The only compatible execution backend is `ttnn`, and its current declaration
requests neither host codegen nor eager device compilation.

Direct typed TTL MLIR is the only product codegen route. Generated TT-Lang
Python and Python-AST compilation are reference material only and will not be
inputs to TileLang codegen. Initial TTL must contain the types/layouts, module
and slot attributes, logical DFB bindings and acquire operations, TensorSlice
and copy operations, tensor compute/store, Pipe structure, and source
locations. User DFB waits, publications and releases are explicit. TT-Lang
supplies lifecycle operations only for compiler-created DFBs, along with
physical CB and DST indices, structured compute scheduling, TTKernel, EmitC,
and C++.

## TT-Lang reference and blockers

The active ABI reference is TT-Lang commit
`4ca4b89de95242ccb9bca8ade71200ac81a59e38`, packaged locally as
`1.1.9.dev20260824+light`. The public binding surface is `ttl.ir`, `ttl.dialects.ttl`,
`ttl.dialects.ttcore`, `ttl.dialects.ttkernel`, `ttl.passmanager`, and
`ttl.passes`; the expected registered pipeline is
`ttl-to-ttkernel-pipeline`.

The repository vendors that exact source as the `3rdparty/tt-lang` gitlink and
uses the official compiler wheel for the same commit. TileLang's local adapter
drives the pinned pipeline and packaging helpers without modifying TT-Lang.
TTL parser and compile-only validation run against those bindings. TTNN
hardware execution, numerical correctness and performance remain separate
validation stages.

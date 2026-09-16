"""Emit initial TTL using the pinned compiler's typed MLIR builders.

This consumer deliberately supports a narrower set than Device Lower. Physical
CB allocation, synchronization completion and instruction scheduling remain
owned by TT-Lang. No TTNN objects or runtime launch APIs are used here.
"""

from collections import Counter

from tvm import tirx

from .attributes import attach_function_attributes
from .bindings import load_bindings


def _calls(body):
    if isinstance(body, tirx.SeqStmt):
        for stmt in body.seq:
            yield from _calls(stmt)
    elif isinstance(body, tirx.Evaluate) and isinstance(body.value, tirx.Call):
        yield body.value
    elif isinstance(body, tirx.Evaluate) and isinstance(body.value, tirx.IntImm) and int(body.value) == 0:
        return
    else:
        raise NotImplementedError("TTL emitter currently requires straight-line, statically specialized Device functions")


def _kind(call):
    return str(call.annotations["tt.compute_kind"].value)


def check_capability(mod):
    """Reject unsupported semantics before importing the optional compiler."""
    version = int(mod.attrs["tt.device_ir_version"])
    if version not in (1, 2, 5):
        raise NotImplementedError("TTL mapping for Device pipelines and multicore programs (v3/v4) is not implemented")
    grid = mod.attrs["tt.launch_grid"]
    if (int(grid.x), int(grid.y)) != (1, 1):
        raise NotImplementedError("TTL emitter currently supports a single Core")
    for tensor in mod.attrs["tt.tensor_table"]:
        if len(tensor.shape) != 2 or str(tensor.memory_space) != "dram" or str(tensor.memory_layout) != "interleaved":
            raise NotImplementedError("TTL emitter currently requires rank-2 DRAM interleaved tensors")
    if len(mod.attrs["tt.dfb_table"]) > 32:
        raise NotImplementedError("TTL emitter cannot seed more than 32 logical DFB bindings; allocation virtualization is required")
    for dfb in mod.attrs["tt.dfb_table"]:
        if str(dfb.element_dtype) not in ("bfloat16", "float32") or len(dfb.block_shape_in_tiles) != 2:
            raise NotImplementedError("TTL emitter requires rank-2 BF16/FP32 DFB blocks")
    supported = {
        "tl.tt.dfb_reserve",
        "tl.tt.dfb_wait",
        "tl.tt.dfb_add",
        "tl.tt.tensor_to_dfb",
        "tl.tt.dfb_to_tensor",
        "tl.tt.tensor_to_dfb_nd",
        "tl.tt.dfb_to_tensor_nd",
        "tl.tt.dfb_compute",
        "tl.tt.accumulator_init",
        "tl.tt.gemm_update",
        "tl.tt.accumulator_materialize",
    }
    if version == 5:
        from ..capabilities import gemm_capability

        for descriptor in mod.attrs["tt.accumulator_table"]:
            capability = gemm_capability(
                str(descriptor.input_dtype),
                str(descriptor.accumulation_dtype),
                str(descriptor.output_dtype),
                arch=str(mod.attrs["tt.target_arch"]),
            )
            if not capability.ttl_mapping:
                raise NotImplementedError(capability.reason)
    for function in mod.functions.values():
        calls = list(_calls(function.body))
        updates = Counter()
        requirements = function.attrs.get("tt.compute_requirements")
        for call in calls:
            name = call.op.name
            if name not in supported:
                raise NotImplementedError(f"TTL emitter has no mapping for Device operation {name}")
            if name == "tl.tt.dfb_compute":
                kind = _kind(call)
                if kind == "elementwise":
                    expr = call.annotations["tt.expression"]
                    if not isinstance(expr, tirx.Add) or not all(
                        isinstance(value, tirx.Call) and value.op.name == "tl.tt.dfb_load" for value in (expr.a, expr.b)
                    ):
                        raise NotImplementedError("TTL elementwise consumer currently supports direct Add only")
                    if any([int(axis) for axis in axes] != [0, 1] for axes in call.annotations["tt.access_maps"]):
                        raise NotImplementedError("TTL Add broadcast mapping is not implemented")
                elif kind == "gemm":
                    if not int(call.annotations["tt.clear"]) or int(call.annotations["tt.transpose_a"]):
                        raise NotImplementedError("TTL legacy GEMM requires clear=True and transpose_a=False")
                    _check_gemm_requirements(requirements)
                elif kind != "copy":
                    raise NotImplementedError(f"TTL consumer for {kind!r} is not implemented")
            if name == "tl.tt.gemm_update":
                _check_gemm_requirements(requirements)
                if int(call.args[3]):
                    raise NotImplementedError("TTL accumulator GEMM transpose_a mapping is not implemented")
                updates[int(call.args[2])] += 1
        if any(count != 1 for count in updates.values()):
            raise NotImplementedError(
                "TTL accumulator mapping currently requires one complete-K GEMM update; "
                "multiple updates need a verified persistent-DST schedule"
            )


def _check_gemm_requirements(requirements):
    if requirements is None:
        raise ValueError("TTL GEMM requires tt.compute_requirements; re-run the Tenstorrent Lower pipeline")
    if str(requirements.destination_width) != "bits16_required" or str(requirements.matmul_full_fp32) != "forbidden":
        raise NotImplementedError(
            "TTL GEMM currently supports BF16/BF16/BF16 only. FP32/full-K GEMM requires "
            "compile-only schedule validation proving no intermediate BF16 pack/reload."
        )


def emit_ttl(mod):
    """Return a verified MLIR module; serialization occurs only at the boundary."""
    check_capability(mod)
    bindings = load_bindings()
    ir, ttl = bindings.ir, bindings.ttl
    ctx = ir.Context()
    ttl.ensure_dialects_registered(ctx)
    with ctx, ir.Location.unknown(ctx):
        output = ir.Module.create()
        grid = mod.attrs["tt.launch_grid"]
        output.operation.attributes["ttl.launch_grid"] = ir.ArrayAttr.get(
            [ir.IntegerAttr.get(ir.IntegerType.get_signless(64, ctx), int(dim)) for dim in (grid.x, grid.y)], ctx
        )
        arch = {"wormhole_b0": bindings.ttcore.Arch.WormholeB0, "blackhole": bindings.ttcore.Arch.Blackhole}
        output.operation.attributes["ttl.target_arch"] = bindings.ttcore.ir.ArchAttr.get(ctx, int(arch[str(mod.attrs["tt.target_arch"])]))
        emitter = _Emitter(bindings, ctx, mod)
        functions = {str(f.attrs["tt.kernel_slot"]): f for f in mod.functions.values()}
        with ir.InsertionPoint(output.body):
            for slot in mod.attrs["tt.kernel_order"]:
                emitter.function(functions[str(slot)])
        output.operation.verify()
        # Verify serialization as well: the consumer receives this exact source.
        ir.Module.parse(str(output), ctx).operation.verify()
        return output


class _Emitter:
    def __init__(self, bindings, ctx, mod):
        self.b = bindings
        self.ctx = ctx
        self.mod = mod
        self.dfbs = {int(d.dfb_id): d for d in mod.attrs["tt.dfb_table"]}
        self.tensors = {int(t.global_arg_index): t for t in mod.attrs["tt.tensor_table"]}
        self.types = {index: self.block_type(d) for index, d in self.dfbs.items()}

    def tile_type(self, dtype, tile_shape):
        dtype = {"bfloat16": self.b.ttcore.DataType.BFloat16, "float32": self.b.ttcore.DataType.Float32}[str(dtype)]
        return self.b.ttcore.ir.TileType.get(self.ctx, *map(int, tile_shape), int(dtype))

    def block_type(self, dfb):
        return self.b.ir.RankedTensorType.get(list(map(int, dfb.block_shape_in_tiles)), self.tile_type(dfb.element_dtype, dfb.tile_shape))

    def tensor_type(self, tensor):
        tile = self.tile_type(tensor.dtype, tensor.tile_shape)
        layout = self.b.ttl.ir.LayoutAttr.get(
            self.ctx,
            list(map(int, tensor.shape)),
            tile,
            int(self.b.ttl.BufferType.DRAM),
            [1, 1],
            int(self.b.ttl.TensorMemoryLayout.Interleaved),
        )
        return self.b.ir.RankedTensorType.get(list(map(int, tensor.tile_grid_shape)), tile, layout)

    def location(self, span):
        if span is None or span.source_name is None:
            return self.b.ir.Location.unknown(self.ctx)
        return self.b.ir.Location.file(str(span.source_name.name), max(1, int(span.line)), max(1, int(span.column)), self.ctx)

    def function(self, function):
        ir, ttl = self.b.ir, self.b.ttl
        indices = list(map(int, function.attrs["tt.tensor_arg_indices"]))
        operation = self.b.func.FuncOp(str(function.attrs["global_symbol"]), ([self.tensor_type(self.tensors[i]) for i in indices], []))
        attach_function_attributes(self.b, self.ctx, operation, function, self.mod, len(self.dfbs))
        block = operation.add_entry_block()
        self.args = dict(zip(indices, block.arguments))
        self.cbs, self.values, self.reserved, self.accumulators = {}, {}, {}, {}
        calls = list(_calls(function.body))
        slot = str(function.attrs["tt.kernel_slot"])
        with ir.InsertionPoint(block):
            # These are logical seeds, matching TT-Lang's own frontend. The
            # finalizer owns physical indices. Device tensor_backing records
            # copy provenance, not TTL's zero-copy L1 storage alias contract.
            for index, dfb in self.dfbs.items():
                if slot not in (str(dfb.producer_slot), str(dfb.consumer_slot)):
                    continue
                cb_type = ttl.CircularBufferType.get(
                    self.ctx, list(map(int, dfb.block_shape_in_tiles)), self.types[index].element_type, int(dfb.block_count)
                )
                self.cbs[index] = ttl.bind_cb(cb_type, cb_index=index, block_count=int(dfb.block_count), dfb_id=index)
            for call in calls:
                with self.location(call.span):
                    self.call(call)
            self.b.func.ReturnOp([])

    def call(self, call):
        ttl, ir = self.b.ttl, self.b.ir
        name = call.op.name
        args = list(map(int, call.args))
        if name == "tl.tt.dfb_reserve":
            self.reserved[args[0]] = ttl.cb_reserve(self.types[args[0]], self.cbs[args[0]])
        elif name == "tl.tt.dfb_wait":
            value = ttl.cb_wait(self.types[args[0]], self.cbs[args[0]])
            self.values[args[0]] = ttl.attach_cb(self.types[args[0]], value, self.cbs[args[0]])
        elif name in ("tl.tt.tensor_to_dfb", "tl.tt.tensor_to_dfb_nd", "tl.tt.dfb_to_tensor", "tl.tt.dfb_to_tensor_nd"):
            read = name in ("tl.tt.tensor_to_dfb", "tl.tt.tensor_to_dfb_nd")
            tensor, dfb = args[:2] if read else args[1::-1]
            # v1 carries [row, col, rows, cols]; v2+ carries min/extent pairs.
            offsets = args[2::2] if name.endswith("_nd") else args[2:4]
            tile_shape = list(map(int, self.tensors[tensor].tile_shape))
            if any(offset % tile for offset, tile in zip(offsets, tile_shape)):
                raise NotImplementedError("TTL copies require tile-aligned region offsets")
            indices = [
                self.b.arith.ConstantOp(ir.IndexType.get(self.ctx), offset // tile).result for offset, tile in zip(offsets, tile_shape)
            ]
            tensor_type = self.tensor_type(self.tensors[tensor])
            view_type = ir.RankedTensorType.get(
                list(map(int, self.dfbs[dfb].block_shape_in_tiles)), tensor_type.element_type, tensor_type.encoding
            )
            view = ttl.tensor_slice(view_type, self.args[tensor], indices)
            # TransferHandleType has no Python constructor at the pinned revision.
            # Only this fixed type token uses the parser; operations are typed.
            handle_type = ir.Type.parse("!ttl.transfer_handle<read>" if read else "!ttl.transfer_handle<write>", self.ctx)
            handle = ttl.copy(handle_type, view, self.cbs[dfb]) if read else ttl.copy(handle_type, self.cbs[dfb], view)
            ttl.wait(handle)
            if read:
                ttl.cb_push(self.cbs[dfb])
            # TT-Lang's acquire/release analysis inserts the matching pops.
        elif name == "tl.tt.dfb_add":
            self.store(args[2], ttl.add(self.values[args[0]], self.values[args[1]]))
        elif name == "tl.tt.dfb_compute":
            kind = _kind(call)
            if kind == "elementwise":
                expr = call.annotations["tt.expression"]
                value = ttl.add(self.values[int(expr.a.args[0])], self.values[int(expr.b.args[0])])
            elif kind == "gemm":
                value = ttl.matmul(
                    self.types[args[0]],
                    self.values[args[1]],
                    self.values[args[2]],
                    transpose_rhs=bool(int(call.annotations["tt.transpose_b"])),
                )
            else:
                value = self.values[args[1]]
            self.store(args[0], value)
        elif name == "tl.tt.accumulator_init":
            self.accumulators[args[0]] = None
        elif name == "tl.tt.gemm_update":
            descriptor = next(a for a in self.mod.attrs["tt.accumulator_table"] if int(a.accumulator_id) == args[2])
            shape = [int(r.extent) // int(t) for r, t in zip(descriptor.accumulator_region.region, self.dfbs[args[0]].tile_shape)]
            result_type = ir.RankedTensorType.get(shape, self.types[args[0]].element_type)
            self.accumulators[args[2]] = ttl.matmul(result_type, self.values[args[0]], self.values[args[1]], transpose_rhs=bool(args[4]))
        elif name == "tl.tt.accumulator_materialize":
            self.store(args[1], self.accumulators[args[0]])

    def store(self, dfb, value):
        self.b.ttl.store(value, self.reserved[dfb])
        self.b.ttl.cb_push(self.cbs[dfb])

"""Recursive current Device PrimFunc to TTL function emission."""

from __future__ import annotations

from tvm import tirx

from .attributes import attach_function_attributes
from .expression import ExpressionEmitter
from .records import RecordResolutionError, build_pipenet_attributes
from .resources import (
    CurrentDFBResolver,
    ResourceResolutionError,
    resource_alias_prologue,
    resolve_resource_operand,
)
from .scalar import ScalarEmitter
from .stmt import TTLStatementEmitterMixin
from .types import TTLTypeConverter


def _column_values(column) -> tuple[int, ...]:
    count = int(column.count)
    values = tuple(map(int, column.values))
    period = int(column.period)
    if period:
        return tuple(values[index % period] + int(column.step) * (index // period) for index in range(count))
    if values:
        return values
    return tuple(int(column.start) + int(column.step) * index for index in range(count))


def _family_descriptor_map(mod, attribute: str, field: str):
    result = {}
    for family in mod.attrs.get(attribute, []):
        if field not in family.fields:
            raise ValueError(f"{attribute} family is missing {field}")
        for identifier in _column_values(family.fields[field]):
            previous = result.get(identifier)
            if previous is not None and _descriptor_type_signature(previous) != _descriptor_type_signature(family.prototype):
                raise RuntimeError(f"{attribute} has incompatible type descriptors for local ID {identifier}")
            result[identifier] = family.prototype
    return result


def _descriptor_type_signature(descriptor):
    if hasattr(descriptor, "logical_shape"):
        return (
            tuple(map(int, descriptor.logical_shape)),
            tuple(map(int, descriptor.compute_tile_shape)),
            str(descriptor.dtype),
            str(descriptor.value_kind),
            str(descriptor.owner_slot),
        )
    return (
        tuple(int(region.extent) for region in descriptor.accumulator_region.region),
        str(descriptor.input_dtype),
        str(descriptor.accumulation_dtype),
        str(descriptor.output_dtype),
    )


class TTLFunctionEmitter(TTLStatementEmitterMixin):
    def __init__(self, bindings, context, device_mod) -> None:
        self.bindings = bindings
        self.b = bindings
        self.context = context
        self.ctx = context
        self.device_mod = device_mod
        self.type_converter = TTLTypeConverter(bindings, context)
        self.groups = {int(group.storage_group_id): group for group in device_mod.attrs["tt.dfb_storage_groups"]}
        self.tensors = {int(tensor.global_arg_index): tensor for tensor in device_mod.attrs["tt.tensor_table"]}
        self.accumulator_descriptors = _family_descriptor_map(device_mod, "tt.accumulator_families", "accumulator_id")
        self.compute_value_descriptors = _family_descriptor_map(device_mod, "tt.compute_value_families", "value_id")
        try:
            self.pipenet_attributes = build_pipenet_attributes(bindings, context, device_mod)
        except RecordResolutionError as error:
            raise RuntimeError(f"verified Device transport metadata is inconsistent: {error}") from error
        self.dfb_resolver = CurrentDFBResolver(device_mod)
        self.scalar = ScalarEmitter(self)

    def emit(self, function):
        indices = tuple(map(int, function.attrs["tt.tensor_arg_indices"]))
        operation = self.bindings.func.FuncOp(
            str(function.attrs["global_symbol"]),
            ([self.type_converter.tensor_type(self.tensors[index]) for index in indices], []),
            loc=self.location(function.span),
        )
        attach_function_attributes(
            self.bindings,
            self.context,
            operation,
            function,
            self.device_mod,
            len(self.groups),
        )
        block = operation.add_entry_block()
        self.function = function
        self.aliases = resource_alias_prologue(function)
        self.tensor_arguments = dict(zip(indices, block.arguments))
        self.scalar_values = {}
        self.cbs = {}
        self.reserved = {}
        self.values = {}
        self.pending_copies = {}
        self.pending_pipe_copies = {}
        self.completed_pipe_transfers = set()
        self.assembled = {}
        self.compute_values = {}
        self.loop_states = {}
        self.accumulators = {}
        self.selected_pipe = None
        self.selected_pipe_net = None

        slot = str(function.attrs["tt.kernel_slot"])
        with self.bindings.ir.InsertionPoint(block):
            for group_id in self.dfb_resolver.groups_for_slot(slot):
                self._bind_group(group_id)
            self.emit_statement(function.body)
            for group_id in tuple(self.assembled):
                self.publish_assembled(group_id)
            self.bindings.func.ReturnOp([])
        return operation

    def _bind_group(self, group_id: int) -> None:
        group = self.groups[group_id]
        cb_type = self.type_converter.circular_buffer_type(group)
        attributes = {
            "block_count": int(group.block_count),
            "dfb_id": group_id,
        }
        backing = group.storage_backing
        if backing is not None:
            attributes["tensor_backing"] = self.bindings.ttl.TensorBackingAttr.get(
                self.context,
                int(backing.global_arg_index),
                int(backing.byte_offset),
                int(backing.byte_size),
            )
        with self.location(group.source_span):
            self.cbs[group_id] = self.bindings.ttl.bind_cb(
                cb_type,
                group_id,
                **attributes,
            )

    def location(self, span):
        if span is None or span.source_name is None:
            return self.bindings.ir.Location.unknown(self.context)
        return self.bindings.ir.Location.file(
            str(span.source_name.name),
            max(1, int(span.line)),
            max(1, int(span.column)),
            self.context,
        )

    def resource_id(self, operand, kind: str) -> int:
        try:
            return resolve_resource_operand(operand, self.aliases, kind).identifier
        except ResourceResolutionError as error:
            if "dynamic" in str(error):
                raise
            raise RuntimeError(f"verified Device resource operand is inconsistent: {error}") from error

    def dfb_id(self, operand) -> int:
        identifier = self.resource_id(operand, "tl.tt.dfb_stream_ref")
        if identifier not in self.cbs:
            raise RuntimeError(f"function {self.function.attrs['global_symbol']} references unbound DFB stream {identifier}")
        return identifier

    def compute_value_id(self, operand) -> int:
        return self.resource_id(operand, "tl.tt.compute_value_ref")

    def value_for_dfb_operand(self, operand):
        identifier = self.dfb_id(operand)
        try:
            return self.values[identifier]
        except KeyError as error:
            raise RuntimeError(f"DFB stream {identifier} is used without an active wait") from error

    def value_for_compute_operand(self, operand):
        identifier = self.compute_value_id(operand)
        try:
            return self.compute_values[identifier]
        except KeyError as error:
            raise RuntimeError(f"compute value {identifier} is used before definition") from error

    @staticmethod
    def _annotation_string(call, name: str, default: str = "") -> str:
        value = call.annotations.get(name)
        if value is None:
            return default
        return str(getattr(value, "value", value))

    def _transfer_handle_type(self, direction: str | None = None):
        suffix = "" if direction is None else f"<{direction}>"
        return self.bindings.ir.Type.parse(f"!ttl.transfer_handle{suffix}", self.context)

    def _tensor_slice(self, tensor_id: int, dfb_id: int, starts):
        descriptor = self.tensors[tensor_id]
        tile_shape = tuple(map(int, descriptor.tile_shape))
        rank = len(descriptor.shape)
        if len(starts) != rank:
            raise ValueError("Tensor slice rank does not match Tensor rank")
        tile_shape = (1,) * (rank - len(tile_shape)) + tile_shape
        indices = []
        for start, tile in zip(starts, tile_shape):
            if isinstance(start, tirx.IntImm):
                if int(start) % tile:
                    raise NotImplementedError("TTL Tensor copies require tile-aligned starts")
                indices.append(self.scalar.index_constant(int(start) // tile))
            else:
                indices.append(self.bindings.arith.floordivsi(self.scalar.emit(start), self.scalar.index_constant(tile)))
        tensor_type = self.type_converter.tensor_type(descriptor)
        view_type = self.bindings.ir.RankedTensorType.get(
            self.type_converter.static_shape(self.groups[dfb_id].block_shape_in_tiles, "DFB block shape"),
            tensor_type.element_type,
            tensor_type.encoding,
        )
        return self.bindings.ttl.tensor_slice(
            view_type,
            self.tensor_arguments[tensor_id],
            indices,
        )

    def store_and_publish(self, dfb_id: int, value) -> None:
        if dfb_id not in self.reserved:
            raise RuntimeError(f"DFB stream {dfb_id} is produced without a reservation")
        self.bindings.ttl.store(value, self.reserved.pop(dfb_id))
        self.bindings.ttl.cb_push(self.cbs[dfb_id])
        self.assembled.pop(dfb_id, None)

    def publish_assembled(self, dfb_id: int) -> None:
        if dfb_id in self.assembled:
            self.store_and_publish(dfb_id, self.assembled[dfb_id])

    def emit_call(self, call) -> None:
        name = str(call.op.name)
        ttl = self.bindings.ttl
        if name == "tl.tt.dfb_reserve":
            identifier = self.dfb_id(call.args[0])
            self.reserved[identifier] = ttl.cb_reserve(
                self.type_converter.block_type(self.groups[identifier]),
                self.cbs[identifier],
            )
            return
        if name == "tl.tt.dfb_wait":
            identifier = self.dfb_id(call.args[0])
            self.publish_assembled(identifier)
            value = ttl.cb_wait(
                self.type_converter.block_type(self.groups[identifier]),
                self.cbs[identifier],
            )
            self.values[identifier] = ttl.attach_cb(
                self.type_converter.block_type(self.groups[identifier]),
                value,
                self.cbs[identifier],
            )
            return
        if name == "tl.tt.dfb_release":
            identifier = self.dfb_id(call.args[0])
            ttl.cb_pop(self.cbs[identifier])
            self.values.pop(identifier, None)
            return
        if name in ("tl.tt.tensor_to_dfb_nd", "tl.tt.dfb_to_tensor_nd"):
            read = name == "tl.tt.tensor_to_dfb_nd"
            tensor_operand, dfb_operand = (call.args[0], call.args[1]) if read else (call.args[1], call.args[0])
            tensor_id = self.resource_id(tensor_operand, "tl.tt.tensor_ref")
            dfb_id = self.dfb_id(dfb_operand)
            starts = list(call.annotations["tt.starts"])
            extents = list(call.annotations["tt.extents"])
            expected_extents = [
                block * tile
                for block, tile in zip(
                    map(int, self.groups[dfb_id].block_shape_in_tiles),
                    map(int, self.groups[dfb_id].tile_shape),
                )
            ]
            try:
                actual_extents = list(map(int, extents))
            except (TypeError, ValueError) as error:
                raise NotImplementedError("TTL Tensor copies require static transfer extents") from error
            if len(actual_extents) > len(expected_extents):
                expected_extents = [1] * (len(actual_extents) - len(expected_extents)) + expected_extents
            if actual_extents != expected_extents:
                raise RuntimeError(f"verified Tensor copy extent {actual_extents} does not match DFB block extent {expected_extents}")
            view = self._tensor_slice(tensor_id, dfb_id, starts)
            handle_type = self._transfer_handle_type("read" if read else "write")
            self.pending_copies[dfb_id] = (
                ttl.copy(handle_type, view, self.cbs[dfb_id]) if read else ttl.copy(handle_type, self.cbs[dfb_id], view)
            )
            return
        if name == "tl.tt.dfb_copy_wait":
            identifier = self.dfb_id(call.args[0])
            try:
                handle = self.pending_copies.pop(identifier)
            except KeyError as error:
                raise RuntimeError(f"DFB stream {identifier} has no pending Tensor copy") from error
            ttl.wait(handle)
            if identifier in self.reserved:
                ttl.cb_push(self.cbs[identifier])
                self.reserved.pop(identifier)
            return
        if name == "tl.tt.dfb_compute":
            self.emit_dfb_compute(call)
            return
        if name == "tl.tt.compute_value":
            self.emit_compute_value(call)
            return
        if name == "tl.tt.compute_value_store":
            value_id = self.compute_value_id(call.args[0])
            self.store_and_publish(self.dfb_id(call.args[1]), self.compute_values[value_id])
            return
        if name == "tl.tt.compute_value_gemm":
            self.emit_compute_value_gemm(call)
            return
        if name == "tl.tt.compute_value_state_init":
            state = self.resource_id(call.args[0], "tl.tt.compute_state_ref")
            self.loop_states[state] = self.value_for_compute_operand(call.args[1])
            return
        if name == "tl.tt.compute_value_state_input":
            value = self.compute_value_id(call.args[0])
            state = self.resource_id(call.args[1], "tl.tt.compute_state_ref")
            self.compute_values[value] = self.loop_states[state]
            return
        if name == "tl.tt.compute_value_state_yield":
            state = self.resource_id(call.args[0], "tl.tt.compute_state_ref")
            self.loop_states[state] = self.value_for_compute_operand(call.args[1])
            return
        if name == "tl.tt.compute_value_state_result":
            value = self.compute_value_id(call.args[0])
            state = self.resource_id(call.args[1], "tl.tt.compute_state_ref")
            self.compute_values[value] = self.loop_states[state]
            return
        if name == "tl.tt.accumulator_init":
            self.emit_accumulator_init(call)
            return
        if name == "tl.tt.gemm_update":
            self.emit_gemm_update(call)
            return
        if name == "tl.tt.accumulator_materialize":
            accumulator = self.resource_id(call.args[0], "tl.tt.accumulator_ref")
            self.store_and_publish(self.dfb_id(call.args[1]), self.accumulators[accumulator])
            return
        if name == "tl.tt.dfb_local_copy":
            source = self.dfb_id(call.args[0])
            destination = self.dfb_id(call.args[1])
            self.assembled[destination] = self.values[source]
            return
        if name == "tl.tt.dfb_gather_copy":
            self.emit_gather_copy(call)
            return
        if name == "tl.tt.dfb_collective_reduce":
            self.emit_collective_reduce(call)
            return
        if name in (
            "tl.tt.dfb_pipe_send_record",
            "tl.tt.dfb_pipe_recv_record",
            "tl.tt.dfb_pipe_wait_record",
        ):
            self.emit_pipe_call(call)
            return
        raise NotImplementedError(f"TTL statement emitter has no mapping for Device operation {name}")

    def emit_dfb_compute(self, call) -> None:
        output = self.dfb_id(call.args[0])
        result_type = self.type_converter.block_type(self.groups[output])
        kind = self._annotation_string(call, "tt.compute_kind")
        if "tt.expression" in call.annotations:
            value = ExpressionEmitter(
                self,
                call,
                result_type,
                self.groups[output].element_dtype,
            ).emit(call.annotations["tt.expression"])
        elif kind == "copy" and len(call.args) == 2:
            value = self.value_for_dfb_operand(call.args[1])
        elif kind == "gemm" and len(call.args) >= 3:
            if bool(int(call.annotations.get("tt.transpose_a", 0))):
                raise NotImplementedError("TTL matmul does not support transpose_a")
            value = self.bindings.ttl.matmul(
                result_type,
                self.value_for_dfb_operand(call.args[1]),
                self.value_for_dfb_operand(call.args[2]),
                transpose_rhs=bool(int(call.annotations.get("tt.transpose_b", 0))),
            )
        elif kind == "reduce" and len(call.args) == 2:
            reduction = self._annotation_string(call, "tt.reduce_kind")
            reduce_types = {"sum": 0, "max": 1}
            if reduction not in reduce_types:
                raise NotImplementedError(f"pinned TT-Lang has no TTL reduce mapping for {reduction!r}")
            input_value = self.value_for_dfb_operand(call.args[1])
            i32 = self.bindings.ir.IntegerType.get_signless(32, self.context)
            reduce_type = self.bindings.ir.IntegerAttr.get(i32, reduce_types[reduction])
            axis = int(call.annotations["tt.reduce_axis"])
            dims = self.bindings.ir.DenseI64ArrayAttr.get([axis], self.context)
            scaler_type = self.bindings.ir.RankedTensorType.get(
                [1, 1],
                input_value.type.element_type,
                input_value.type.encoding,
            )
            one = self.bindings.ir.FloatAttr.get(self.bindings.ir.F32Type.get(self.context), 1.0)
            scaler = self.bindings.ttl.fill(scaler_type, one)
            value = self.bindings.ttl.reduce(
                result_type,
                input_value,
                scaler,
                reduce_type,
                dims,
            )
        elif kind == "transpose" and len(call.args) == 2:
            value = self.bindings.ttl.transpose(
                result_type,
                self.value_for_dfb_operand(call.args[1]),
            )
        else:
            raise NotImplementedError(f"TTL compute kind {kind!r} has no complete mapping")
        self.store_and_publish(output, value)

    def emit_compute_value(self, call) -> None:
        identifier = self.compute_value_id(call.args[0])
        try:
            descriptor = self.compute_value_descriptors[identifier]
        except KeyError as error:
            raise RuntimeError(f"compute value {identifier} has no descriptor") from error
        result_type = self.type_converter.value_type(descriptor)
        self.compute_values[identifier] = ExpressionEmitter(
            self,
            call,
            result_type,
            descriptor.dtype,
        ).emit(call.annotations["tt.expression"])

    def emit_compute_value_gemm(self, call) -> None:
        if bool(int(call.annotations["tt.transpose_a"])):
            raise NotImplementedError("TTL matmul does not support transpose_a")
        output = self.compute_value_id(call.args[0])
        descriptor = self.compute_value_descriptors[output]
        result = self.bindings.ttl.matmul(
            self.type_converter.value_type(descriptor),
            self.value_for_dfb_operand(call.args[1]),
            self.value_for_dfb_operand(call.args[2]),
            transpose_rhs=bool(int(call.annotations["tt.transpose_b"])),
        )
        old = self.compute_value_id(call.args[3])
        if old >= 0:
            result = self.bindings.ttl.add(self.compute_values[old], result)
        self.compute_values[output] = result

    def emit_accumulator_init(self, call) -> None:
        identifier = self.resource_id(call.args[0], "tl.tt.accumulator_ref")
        descriptor = self.accumulator_descriptors[identifier]
        if str(descriptor.accumulation_dtype) != "bfloat16":
            raise NotImplementedError("matmul_full_fp32=required remains unsupported until TT-Lang exposes a hard full-K schedule API")
        shape = [int(region.extent) // 32 for region in descriptor.accumulator_region.region]
        result_type = self.bindings.ir.RankedTensorType.get(
            shape,
            self.type_converter.tile_type(descriptor.input_dtype, (32, 32)),
        )
        zero = self.bindings.ir.FloatAttr.get(self.bindings.ir.F32Type.get(self.context), 0.0)
        self.accumulators[identifier] = self.bindings.ttl.fill(result_type, zero)

    def emit_gemm_update(self, call) -> None:
        if bool(int(call.args[3])):
            raise NotImplementedError("TTL matmul does not support transpose_a")
        accumulator = self.resource_id(call.args[2], "tl.tt.accumulator_ref")
        current = self.accumulators[accumulator]
        product = self.bindings.ttl.matmul(
            current.type,
            self.value_for_dfb_operand(call.args[0]),
            self.value_for_dfb_operand(call.args[1]),
            transpose_rhs=bool(int(call.args[4])),
        )
        self.accumulators[accumulator] = self.bindings.ttl.add(current, product)

    def emit_gather_copy(self, call) -> None:
        source = self.dfb_id(call.args[0])
        destination = self.dfb_id(call.args[1])
        axis = int(call.args[2])
        source_rank = int(call.args[3])
        group_size = int(call.args[4])
        destination_type = self.type_converter.block_type(self.groups[destination])
        current = self.assembled.get(destination)
        if current is None:
            zero = self.bindings.ir.FloatAttr.get(self.bindings.ir.F32Type.get(self.context), 0.0)
            current = self.bindings.ttl.fill(destination_type, zero)
        source_value = self.values[source]
        source_shape = list(map(int, source_value.type.shape))
        offsets = [0] * len(source_shape)
        offsets[axis] = source_rank * source_shape[axis]
        self.assembled[destination] = self.bindings.tensor.insert_slice(
            source_value,
            current,
            [],
            [],
            [],
            offsets,
            source_shape,
            [1] * len(source_shape),
            results=[destination_type],
        )
        if source_rank == group_size - 1:
            self.publish_assembled(destination)

    def emit_collective_reduce(self, call) -> None:
        left = self.dfb_id(call.args[0])
        right = self.dfb_id(call.args[1])
        output = self.dfb_id(call.args[2])
        lhs = self.assembled.get(left, self.values.get(left))
        rhs = self.values[right]
        reduction = str(getattr(call.args[3], "value", call.args[3]))
        operations = {
            "sum": self.bindings.ttl.add,
            "max": self.bindings.ttl.max,
            "min": self.bindings.ttl.min,
        }
        if reduction not in operations:
            raise NotImplementedError(f"TTL collective reduction {reduction!r} is not supported")
        self.assembled[output] = operations[reduction](lhs, rhs)

    def emit_pipe_call(self, call) -> None:
        record = call.args[0]
        if not isinstance(record, tirx.Call) or str(record.op.name) != "tl.tt.pipe_record_ref":
            raise RuntimeError(f"{call.op.name} requires pipe_record_ref")
        pipe_net = self.resource_id(record.args[0], "tl.tt.pipe_net_ref")
        if self.selected_pipe is None:
            name = str(call.op.name)
            if name == "tl.tt.dfb_pipe_wait_record":
                dfb = self.dfb_id(call.args[1])
                key = (pipe_net, dfb)
                if key not in self.completed_pipe_transfers:
                    raise RuntimeError(f"PipeNet transfer {key} has no preceding send/receive")
                self.completed_pipe_transfers.remove(key)
                return
            records = self.pipenet_attributes[pipe_net]
            if name == "tl.tt.dfb_pipe_send_record":
                operation = self.bindings.ttl.pipenet_foreach_src(records)
                pipe_type = self.bindings.ttl.SelectedPipeSrcType.get(self.context)
            else:
                operation = self.bindings.ttl.pipenet_foreach_dst(records)
                pipe_type = self.bindings.ttl.SelectedPipeDstType.get(self.context)
            region_block = self.bindings.ir.Block.create_at_start(operation.body, [pipe_type])
            previous_pipe, previous_net = self.selected_pipe, self.selected_pipe_net
            with self.bindings.ir.InsertionPoint(region_block):
                self.selected_pipe = region_block.arguments[0]
                self.selected_pipe_net = pipe_net
                self._emit_selected_pipe_transfer(call, complete=True)
                self.bindings.ttl.yield_([])
            self.selected_pipe, self.selected_pipe_net = previous_pipe, previous_net
            self.completed_pipe_transfers.add((pipe_net, self.dfb_id(call.args[1])))
            return
        self._emit_selected_pipe_transfer(call, complete=False)

    def _emit_selected_pipe_transfer(self, call, *, complete: bool) -> None:
        record = call.args[0]
        pipe_net = self.resource_id(record.args[0], "tl.tt.pipe_net_ref")
        if pipe_net != self.selected_pipe_net:
            raise RuntimeError(f"PipeNet stream region selected {self.selected_pipe_net}, operation references {pipe_net}")
        dfb = self.dfb_id(call.args[1])
        key = (pipe_net, dfb)
        name = str(call.op.name)
        if name == "tl.tt.dfb_pipe_send_record":
            self.pending_pipe_copies[key] = self.bindings.ttl.copy(
                self._transfer_handle_type("write"),
                self.cbs[dfb],
                self.selected_pipe,
            )
        elif name == "tl.tt.dfb_pipe_recv_record":
            if dfb not in self.reserved:
                raise RuntimeError(f"PipeNet receive into DFB stream {dfb} has no reservation")
            self.pending_pipe_copies[key] = self.bindings.ttl.copy(
                self._transfer_handle_type(),
                self.selected_pipe,
                self.reserved[dfb],
            )
        else:
            try:
                handle = self.pending_pipe_copies.pop(key)
            except KeyError as error:
                raise RuntimeError(f"PipeNet transfer {key} has no pending send/receive") from error
            self.bindings.ttl.wait(handle)
            if dfb in self.reserved:
                self.bindings.ttl.cb_push(self.cbs[dfb])
                self.reserved.pop(dfb)
        if complete and name != "tl.tt.dfb_pipe_wait_record":
            handle = self.pending_pipe_copies.pop(key)
            self.bindings.ttl.wait(handle)
            if dfb in self.reserved:
                self.bindings.ttl.cb_push(self.cbs[dfb])
                self.reserved.pop(dfb)


__all__ = ["TTLFunctionEmitter"]
